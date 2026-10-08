"""
Real-time 3D scanner viewer.

Install:
    pip install pyqt5 pyqtgraph pyopengl pyserial numpy scipy trimesh

Design (why it doesn't hang):
  * Serial is read in a background thread (never blocks the GUI).
  * Points go into pre-allocated numpy buffers (no list appends / re-allocation).
  * The GUI timer drains the queue with a ~12 ms time budget per tick and
    updates ONE OpenGL scatter item (GPU-rendered, no matplotlib).
  * The final mesh is built in a worker thread with vectorized numpy
    (fixed angular bins per layer) and shown as a GLMeshItem.
"""
import sys
import time
import queue
import threading

import numpy as np
import serial
import serial.tools.list_ports
import trimesh
from scipy.signal import medfilt
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# ---------------- CONFIG ----------------
CENTER_OFFSET = 83.0          # mm, sensor to rotation center
MIN_DIST = 20.0               # mm, ignore readings closer than this
Z_STEP_MM = 0.00625
Z_LAYER_STEPS = 400
LAYER_HEIGHT = Z_STEP_MM * Z_LAYER_STEPS   # 2.5 mm
MEDIAN_FILTER = 5
ANGLE_BINS = 360              # resampled points per layer for the final mesh
BAUD = 9600                   # must match Arduino (115200 is much faster if you can)
MAX_POINTS = 300_000
EXPECTED_HEIGHT = 100.0       # mm, only used for live colour ramp
UI_INTERVAL_MS = 33           # ~30 FPS
TICK_BUDGET_S = 0.012         # max time spent draining data per GUI tick


# ---------------- SERIAL THREAD ----------------
class SerialReader(threading.Thread):
    def __init__(self, port, baud, out_q, stop_evt):
        super().__init__(daemon=True)
        self.port, self.baud, self.q, self.stop_evt = port, baud, out_q, stop_evt
        self.bytes = 0
        self.parsed = 0
        self.bad = 0
        self.last_line = ""

    def run(self):
        try:
            with serial.Serial(self.port, self.baud, timeout=0.2) as ser, \
                    open("scan_raw.log", "w") as log:
                buf = b""
                while not self.stop_evt.is_set():
                    chunk = ser.read(max(1, ser.in_waiting))
                    if not chunk:
                        continue
                    self.bytes += len(chunk)
                    buf += chunk
                    *lines, buf = buf.replace(b"\r", b"\n").split(b"\n")
                    batch = []
                    done = False
                    for ln in lines:
                        ln = ln.strip()
                        if not ln:
                            continue
                        log.write(ln.decode(errors="ignore") + "\n")
                        self.last_line = ln.decode(errors="ignore")
                        if ln == b"SCAN_DONE":
                            done = True
                            break
                        try:
                            a, d, l = ln.split(b",")
                            batch.append((float(a), float(d), float(l)))
                            self.parsed += 1
                        except ValueError:
                            self.bad += 1
                            if self.bad <= 5:
                                print("Unparsed line:", ln)
                    if batch:
                        self.q.put(batch)
                    if done:
                        self.q.put("DONE")
                        return
        except serial.SerialException as e:
            self.q.put(("ERR", str(e)))


class DemoReader(threading.Thread):
    """Fake scanner (a wavy vase) to test the GUI without hardware."""
    def __init__(self, out_q, stop_evt):
        super().__init__(daemon=True)
        self.q, self.stop_evt = out_q, stop_evt
        self.bytes = self.parsed = self.bad = 0
        self.last_line = "demo"

    def run(self):
        for layer in range(30):
            batch = []
            for ang in np.arange(0, 360, 2.0):
                r = 30 + 8 * np.sin(layer / 4.0) + 4 * np.sin(np.radians(ang) * 3)
                batch.append((float(ang), float(CENTER_OFFSET - r), float(layer)))
            self.parsed += len(batch)
            self.q.put(batch)
            time.sleep(0.15)
            if self.stop_evt.is_set():
                return
        self.q.put("DONE")


# ---------------- MESH BUILDING (vectorized) ----------------
def build_mesh(ang, rad, lay):
    grid_deg = np.linspace(0, 360, ANGLE_BINS, endpoint=False)
    rings, zs = [], []

    for k in np.unique(lay):
        m = lay == k
        if m.sum() < max(MEDIAN_FILTER, 3):
            continue
        a = ang[m] % 360.0
        r = rad[m]
        o = np.argsort(a)
        a, r = a[o], r[o]
        r = medfilt(r, MEDIAN_FILTER)                 # filter radius, not x/y
        rings.append(np.interp(grid_deg, a, r, period=360.0))
        zs.append(k * LAYER_HEIGHT)

    if len(rings) < 2:
        raise RuntimeError("Not enough layers to build a mesh")

    R = np.array(rings)
    Z = np.array(zs)
    L, N = R.shape
    th = np.radians(grid_deg)
    X = R * np.cos(th)
    Y = R * np.sin(th)
    ZZ = np.repeat(Z[:, None], N, axis=1)
    verts = np.stack([X, Y, ZZ], -1).reshape(-1, 3)

    i = np.arange(L - 1)[:, None]
    j = np.arange(N)[None, :]
    a0 = i * N + j
    a1 = i * N + (j + 1) % N
    b0 = (i + 1) * N + j
    b1 = (i + 1) * N + (j + 1) % N
    side = np.concatenate([
        np.stack([a0, a1, b0], -1).reshape(-1, 3),
        np.stack([a1, b1, b0], -1).reshape(-1, 3),
    ])

    # caps: fan to a centre vertex on the rotation axis
    bc, tc = len(verts), len(verts) + 1
    verts = np.vstack([verts, [0, 0, Z[0]], [0, 0, Z[-1]]])
    jj = np.arange(N)
    jn = (jj + 1) % N
    bottom = np.stack([jn, jj, np.full(N, bc)], -1)
    tb = (L - 1) * N
    top = np.stack([tb + jj, tb + jn, np.full(N, tc)], -1)

    faces = np.concatenate([side, bottom, top])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


# ---------------- MAIN WINDOW ----------------
def dims_text(mn, mx):
    d = np.asarray(mx) - np.asarray(mn)
    return (f"Length (X): {d[0]:.1f} mm    Width (Y): {d[1]:.1f} mm    "
            f"Height (Z): {d[2]:.1f} mm")



class Scanner(QtWidgets.QMainWindow):
    mesh_ready = QtCore.pyqtSignal(object)   # mesh or Exception

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live 3D Scanner")
        self.resize(1100, 750)

        # pre-allocated buffers
        self.pos = np.zeros((MAX_POINTS, 3), dtype=np.float32)
        self.col = np.zeros((MAX_POINTS, 4), dtype=np.float32)
        self.raw_ang = np.zeros(MAX_POINTS, dtype=np.float32)
        self.raw_rad = np.zeros(MAX_POINTS, dtype=np.float32)
        self.raw_lay = np.zeros(MAX_POINTS, dtype=np.int32)
        self.n = 0
        self.rejected = 0
        self.bmin = np.full(3, np.inf)
        self.bmax = np.full(3, -np.inf)
        self.t_start = 0.0

        self.q = queue.Queue()
        self.stop_evt = threading.Event()
        self.reader = None
        self.mesh = None
        self.mesh_item = None
        self.scanning = False

        # --- UI ---
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        lay = QtWidgets.QVBoxLayout(central)

        bar = QtWidgets.QHBoxLayout()
        self.port_box = QtWidgets.QComboBox()
        self.refresh_ports()
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_start = QtWidgets.QPushButton("Start scan")
        self.btn_demo = QtWidgets.QPushButton("Demo (no Arduino)")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_save = QtWidgets.QPushButton("Save STL")
        self.chk_points = QtWidgets.QCheckBox("Points")
        self.chk_points.setChecked(True)
        self.chk_mesh = QtWidgets.QCheckBox("Mesh")
        self.chk_mesh.setChecked(True)
        for w in (self.port_box, self.btn_refresh, self.btn_start, self.btn_demo, self.btn_stop,
                  self.btn_clear, self.btn_save, self.chk_points, self.chk_mesh):
            bar.addWidget(w)
        bar.addStretch()
        lay.addLayout(bar)

        self.info = QtWidgets.QLabel("Surface area: -    Volume: -")
        self.info.setStyleSheet("font-size: 30px; font-weight: bold; padding: 6px;")
        lay.addWidget(self.info)

        self.dims = QtWidgets.QLabel(dims_text([0]*3, [0]*3))
        self.dims.setStyleSheet("font-size: 15px; padding: 0 6px 6px 6px;")
        lay.addWidget(self.dims)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(30, 30, 30)
        self.view.opts["center"] = pg.Vector(0, 0, 40)
        self.view.setCameraPosition(distance=250, elevation=25, azimuth=45)
        grid = gl.GLGridItem()
        grid.setSize(200, 200)
        grid.setSpacing(10, 10)
        self.view.addItem(grid)
        self.view.addItem(gl.GLAxisItem(size=pg.Vector(40, 40, 40)))
        self.scatter = gl.GLScatterPlotItem(size=4, pxMode=True)
        self.view.addItem(self.scatter)
        lay.addWidget(self.view, 1)

        self.status = QtWidgets.QLabel("Idle")
        lay.addWidget(self.status)

        # --- signals ---
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_start.clicked.connect(self.start_scan)
        self.btn_demo.clicked.connect(self.start_demo)
        self.btn_stop.clicked.connect(self.stop_scan)
        self.btn_clear.clicked.connect(self.clear)
        self.btn_save.clicked.connect(self.save_stl)
        self.chk_points.toggled.connect(lambda v: self.scatter.setVisible(v))
        self.chk_mesh.toggled.connect(
            lambda v: self.mesh_item.setVisible(v) if self.mesh_item else None)
        self.mesh_ready.connect(self.on_mesh_ready)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(UI_INTERVAL_MS)

    # ----- helpers -----
    def refresh_ports(self):
        self.port_box.clear()
        for p in serial.tools.list_ports.comports():
            self.port_box.addItem(f"{p.device} - {p.description}", p.device)
        for i in range(self.port_box.count()):       # prefer Arduino/USB
            t = self.port_box.itemText(i)
            if "Arduino" in t or "USB" in t:
                self.port_box.setCurrentIndex(i)
                break

    def clear(self):
        self.stop_scan()
        self.n = 0
        self.rejected = 0
        self.bmin = np.full(3, np.inf)
        self.bmax = np.full(3, -np.inf)
        self.info.setText("Surface area: -    Volume: -\n" + dims_text([0]*3, [0]*3))
        self.scatter.setData(pos=np.zeros((1, 3), dtype=np.float32))
        if self.mesh_item:
            self.view.removeItem(self.mesh_item)
            self.mesh_item = None
        self.mesh = None
        self.status.setText("Cleared")

    # ----- scanning -----
    def start_scan(self):
        port = self.port_box.currentData()
        if not port:
            self.status.setText("No serial port selected")
            return
        self.clear()
        self.q = queue.Queue()
        self.stop_evt = threading.Event()
        self.reader = SerialReader(port, BAUD, self.q, self.stop_evt)
        self.reader.start()
        self.scanning = True
        self.t_start = time.time()
        self.status.setText(f"Scanning on {port} ...")

    def start_demo(self):
        self.clear()
        self.q = queue.Queue()
        self.stop_evt = threading.Event()
        self.reader = DemoReader(self.q, self.stop_evt)
        self.reader.start()
        self.scanning = True
        self.t_start = time.time()
        self.status.setText("Demo scan running ...")

    def stop_scan(self):
        self.stop_evt.set()
        self.scanning = False

    def tick(self):
        t0 = time.perf_counter()
        dirty = False
        finished = False

        while time.perf_counter() - t0 < TICK_BUDGET_S:
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                break

            if isinstance(item, list):
                arr = np.asarray(item, dtype=np.float32)
                a, d, l = arr[:, 0], arr[:, 1], arr[:, 2]
                ok = (d > MIN_DIST) & (d < CENTER_OFFSET)
                self.rejected += int((~ok).sum())
                a, d, l = a[ok], d[ok], l[ok]
                k = min(len(a), MAX_POINTS - self.n)
                if k <= 0:
                    continue
                a, d, l = a[:k], d[:k], l[:k]
                r = CENTER_OFFSET - d
                th = np.radians(a)
                z = l * LAYER_HEIGHT
                s = slice(self.n, self.n + k)
                self.pos[s, 0] = r * np.cos(th)
                self.pos[s, 1] = r * np.sin(th)
                self.pos[s, 2] = z
                t = np.clip(z / EXPECTED_HEIGHT, 0, 1)       # colour by height
                self.col[s, 0] = t
                self.col[s, 1] = 0.3 + 0.5 * (1 - np.abs(2 * t - 1))
                self.col[s, 2] = 1 - t
                self.col[s, 3] = 1.0
                self.raw_ang[s], self.raw_rad[s], self.raw_lay[s] = a, r, l
                self.bmin = np.minimum(self.bmin, self.pos[s].min(axis=0))
                self.bmax = np.maximum(self.bmax, self.pos[s].max(axis=0))
                self.n += k
                dirty = True
            elif item == "DONE":
                finished = True
                break
            elif isinstance(item, tuple) and item[0] == "ERR":
                self.scanning = False
                self.status.setText(f"Serial error: {item[1]}")

        if dirty:
            self.scatter.setData(pos=self.pos[:self.n], color=self.col[:self.n])
            self.status.setText(f"Scanning... {self.n} points")
            self.info.setText("Surface area: (after scan)    Volume: (after scan)\n"
                              + dims_text(self.bmin, self.bmax))

        if self.scanning and self.reader is not None and not dirty:
            rd = self.reader
            msg = (f"Waiting... bytes={rd.bytes} parsed={rd.parsed} bad={rd.bad} "
                   f"plotted={self.n} out-of-range={self.rejected} last='{rd.last_line}'")
            if rd.bytes == 0 and time.time() - self.t_start > 5:
                msg += "  -> NO DATA: check port, baud, Arduino reset/scan start"
            self.status.setText(msg)

        if finished:
            self.scanning = False
            self.status.setText(f"Scan done ({self.n} points) - building mesh...")
            n = self.n
            data = (self.raw_ang[:n].copy(), self.raw_rad[:n].copy(),
                    self.raw_lay[:n].copy())
            threading.Thread(target=self._mesh_worker, args=data, daemon=True).start()

    # ----- mesh -----
    def _mesh_worker(self, a, r, l):
        try:
            self.mesh_ready.emit(build_mesh(a, r, l))
        except Exception as e:
            self.mesh_ready.emit(e)

    def on_mesh_ready(self, result):
        if isinstance(result, Exception):
            self.status.setText(f"Mesh failed: {result}")
            return
        self.mesh = result
        if self.mesh_item:
            self.view.removeItem(self.mesh_item)
        md = gl.MeshData(vertexes=np.asarray(result.vertices, dtype=np.float32),
                         faces=np.asarray(result.faces, dtype=np.int32))
        self.mesh_item = gl.GLMeshItem(meshdata=md, smooth=True, shader="shaded",
                                       color=(0.6, 0.75, 1.0, 1.0))
        self.mesh_item.setVisible(self.chk_mesh.isChecked())
        self.view.addItem(self.mesh_item)

        zmax = float(result.bounds[1][2])
        self.view.opts["center"] = pg.Vector(0, 0, zmax / 2)
        result.export("scan_result.stl")
        self.info.setText(
            f"Surface area: {result.area:.1f} mm\u00b2    "
            f"Volume: {abs(result.volume):.1f} mm\u00b3 ({abs(result.volume)/1000:.2f} cm\u00b3)\n"
            + dims_text(result.bounds[0], result.bounds[1]))
        self.status.setText(
            f"Mesh done. Watertight: {result.is_watertight}   (saved scan_result.stl)")

    def save_stl(self):
        if self.mesh is None:
            self.status.setText("No mesh yet")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save STL", "scan_result.stl", "STL (*.stl)")
        if path:
            self.mesh.export(path)

    def closeEvent(self, e):
        self.stop_evt.set()
        super().closeEvent(e)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Scanner()
    w.show()
    sys.exit(app.exec_())