/////////////////////////////////////////////////////////////////////////////
// 3D Scanner – Rotating Object + Z Layer Scanning
// Sensor: Sharp GP2Y0A51SK0F
/////////////////////////////////////////////////////////////////////////////

// ---------------- PIN DEFINITIONS ----------------
// Rotation (X)
#define X_STEP 2
#define X_DIR  5
#define X_EN   8

// Z Axis
#define Z_STEP 4
#define Z_DIR  7
#define Z_EN   8
#define Z_END  11   // DOWN endstop

// Sensor
#define SHARP_PIN A0

// ---------------- CONFIG ----------------
#define STEPS_PER_REV   200     // 1.8° stepper
#define Z_LAYER_STEPS  400
#define TOTAL_LAYERS   30      // adjust height

#define STEP_DELAY_US  800     // motor speed

// Sharp sensor constants
const float A = 48.375;
const float B = 0.0675;

// ---------------- GLOBALS ----------------
float z_position = 0;

// ---------------- SETUP ----------------
void setup() {
  Serial.begin(9600);
  delay(2000);

  pinMode(X_STEP, OUTPUT);
  pinMode(X_DIR, OUTPUT);
  pinMode(X_EN, OUTPUT);

  pinMode(Z_STEP, OUTPUT);
  pinMode(Z_DIR, OUTPUT);
  pinMode(Z_EN, OUTPUT);
  pinMode(Z_END, INPUT_PULLUP);

  digitalWrite(X_DIR, HIGH);   // rotation direction
  digitalWrite(X_EN, LOW);     // enable motors

  Serial.println("START");

  homeZ();                     // go down to endstop
  moveZUp(Z_LAYER_STEPS);      // move up 400 steps
  z_position = 0;              // define Z = 0

  Serial.println("Z_HOME_DONE");
}

// ---------------- LOOP ----------------
void loop() {

  for (int layer = 0; layer < TOTAL_LAYERS; layer++) {

    // -------- ROTATE FULL 360° --------
    for (int step = 0; step < STEPS_PER_REV; step++) {

      stepMotor(X_STEP);

      float angle = step * (360.0 / STEPS_PER_REV);
      float distance = readSharpMM();

      // Send data: angle, distance, z
      Serial.print(angle, 2);
      Serial.print(",");
      Serial.print(distance, 2);
      Serial.print(",");
      Serial.println(z_position, 2);

      delay(20); // stabilize sensor
    }

    // -------- NEXT Z LAYER --------
    moveZUp(Z_LAYER_STEPS);
    z_position += 1;   // layer index (convert to mm in Python if needed)
  }

  digitalWrite(X_EN, HIGH); // disable motors
  Serial.println("SCAN_DONE");

  while (1); // stop
}

// ---------------- FUNCTIONS ----------------

void stepMotor(int pin) {
  digitalWrite(pin, HIGH);
  delayMicroseconds(STEP_DELAY_US);
  digitalWrite(pin, LOW);
  delayMicroseconds(STEP_DELAY_US);
}

// ---- Z HOMING ----
void homeZ() {
  digitalWrite(Z_DIR, LOW); // DOWN direction

  while (digitalRead(Z_END) == HIGH) {
    stepMotor(Z_STEP);
  }

  delay(300);
}

// ---- MOVE Z UP ----
void moveZUp(int steps) {
  digitalWrite(Z_DIR, HIGH); // UP
  for (int i = 0; i < steps; i++) {
    stepMotor(Z_STEP);
  }
}

// ---- READ SHARP SENSOR ----
float readSharpMM() {
  int raw = analogRead(SHARP_PIN);
  float voltage = (5.0 * raw) / 1024.0;

  if (voltage <= B) return 0;

  float dist = A / (voltage - B);
  return dist;
}
