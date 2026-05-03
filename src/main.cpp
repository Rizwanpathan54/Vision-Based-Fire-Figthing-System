#include <Arduino.h>
#include <ESP32Servo.h>

/* ---------- SERVOS ---------- */
Servo panServo;
Servo tiltServo;

/* ---------- PIN DEFINITIONS ---------- */
#define PAN_PIN   18
#define TILT_PIN  19
#define RELAY_PIN 25   // Pump relay (ACTIVE LOW)

/* ---------- MG90S SAFE LIMITS ---------- */
#define PAN_MIN   0
#define PAN_MAX   120
#define TILT_MIN  0
#define TILT_MAX  150

/* ---------- SERVO CENTERS ---------- */
#define PAN_CENTER   65
#define TILT_CENTER  90

/* ---------- SCALE ---------- */
#define PAN_SCALE   2.1f
#define TILT_SCALE  2.1f

/* ---------- EMA SMOOTHING
   Alpha closer to 1.0 = faster response (less smoothing)
   Alpha closer to 0.0 = slower / smoother motion
   0.25 is a good starting point for 80 ms update rate.
   Increase to 0.4 if tracking feels too slow on a fast fire. ---------- */
#define SMOOTH_ALPHA 0.25f

/* ---------- DEADBAND
   Ignore angle changes smaller than this (degrees) to
   prevent micro-jitter from driving the servos constantly. ---------- */
#define DEADBAND_DEG 0.8f

/* ---------- STATE ---------- */
String  rxBuffer   = "";
float   smoothPan  = PAN_CENTER;    // tracks actual written servo position
float   smoothTilt = TILT_CENTER;

/* ---------- FUNCTION DECLARATIONS ---------- */
void  parseSerial(String data);
float extractValue(String data, const char *key);
void  moveToAngle(float panOffset, float tiltOffset);

/* ---------- SETUP ---------- */
void setup() {
    Serial.begin(115200);

    panServo.attach(PAN_PIN,  500, 2400);
    tiltServo.attach(TILT_PIN, 500, 2400);

    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, HIGH);   // Pump OFF on boot (Active Low)

    // Move to center and seed the smoother at the same position
    panServo.write((int)smoothPan);
    tiltServo.write((int)smoothTilt);
}

/* ---------- LOOP ---------- */
void loop() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n') {
            rxBuffer.trim();
            if (rxBuffer.length() > 0) {
                parseSerial(rxBuffer);
            }
            rxBuffer = "";
        } else {
            rxBuffer += c;
        }
    }
}

/* ---------- PARSE COMMANDS ---------- */
void parseSerial(String data) {

    // --- Pump control (works in both merged and legacy FIRE:0 packet) ---
    if (data.indexOf("FIRE:1") != -1) {
        digitalWrite(RELAY_PIN, LOW);   // Pump ON
    } else if (data.indexOf("FIRE:0") != -1) {
        digitalWrite(RELAY_PIN, HIGH);  // Pump OFF
    }

    // --- Pan / Tilt movement (only when PAN key present) ---
    if (data.indexOf("PAN:") != -1) {
        float panOffset  = extractValue(data, "PAN");
        float tiltOffset = extractValue(data, "TILT");
        moveToAngle(panOffset, tiltOffset);
    }
}

/* ---------- SERVO MOVEMENT WITH EMA SMOOTHING ---------- */
void moveToAngle(float panOffset, float tiltOffset) {

    // Raw target angles (same formula as original)
    float targetPan  = 180.0f - (PAN_CENTER  + panOffset  * PAN_SCALE);
    float targetTilt =           TILT_CENTER + tiltOffset * TILT_SCALE;

    // Clamp targets to safe mechanical limits BEFORE smoothing
    // so the smoother never chases an out-of-range value
    targetPan  = constrain(targetPan,  (float)PAN_MIN,  (float)PAN_MAX);
    targetTilt = constrain(targetTilt, (float)TILT_MIN, (float)TILT_MAX);

    // Exponential Moving Average — blends new target into current position
    // smoothPos = alpha * target + (1 - alpha) * smoothPos
    smoothPan  = SMOOTH_ALPHA * targetPan  + (1.0f - SMOOTH_ALPHA) * smoothPan;
    smoothTilt = SMOOTH_ALPHA * targetTilt + (1.0f - SMOOTH_ALPHA) * smoothTilt;

    // Deadband — skip the write entirely if the change is too small.
    // Prevents constant micro-stepping that wears out MG90S gears.
    int iPan  = (int)smoothPan;
    int iTilt = (int)smoothTilt;

    bool panChanged  = abs(iPan  - panServo.read())  >= (int)DEADBAND_DEG;
    bool tiltChanged = abs(iTilt - tiltServo.read()) >= (int)DEADBAND_DEG;

    if (panChanged)  panServo.write(iPan);
    if (tiltChanged) tiltServo.write(iTilt);
}

/* ---------- HELPER: extract float value after "KEY:" ---------- */
float extractValue(String data, const char *key) {
    int idx = data.indexOf(String(key) + ":");
    if (idx == -1) return 0.0f;
    int start = idx + strlen(key) + 1;
    int end   = data.indexOf(',', start);
    if (end == -1) end = data.length();
    return data.substring(start, end).toFloat();
}