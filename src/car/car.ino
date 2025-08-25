#include <Arduino.h>
#include <stdbool.h>

#include "CarESPNow.h"
#include "GestureClassifier.h"
#include "IMUBuffer.h"
#include "MPUSensorData.h"
#include "MotionController.h"

#define MPU_SDA_PIN 18
#define MPU_SCL_PIN 19


TwoWire wire = TwoWire(1);
CarESPNow espNow;
MotorDriver motor;
IMUBuffer imuBuffer;
GestureClassifier gestureClassifier;
MPU6050_WE mpu(&wire);
MotionController motion(motor, mpu);


typedef enum {
  driving = 0,
  buttonPressed = 1,
  bufferFilling = 2,
  classifying = 3,
} State;



const int buzzerPin = 2;

static float imu_window[WINDOW_SIZE * NUM_AXES];
float pitch = 0, roll = 0;
State state = driving;



void onDataReceived(const SensorData& data) {
  if (isClassifying(state) || isButtonPressed(state))
    return;

  if (isBufferFilling(state)) {
    imuBuffer.addSample(data);
    return;
  }

  pitch = data.pitch;
  roll = data.roll;
  if (data.isButtonPressed) state = buttonPressed;
}

void setup() {
  Serial.begin(115200);

  wire.begin(MPU_SDA_PIN, MPU_SCL_PIN);

  motion.init();

  if (!gestureClassifier.init())
    while (1)
      ;

  if (!espNow.begin(onDataReceived)) {
    Serial.println("ESP-NOW init failed!");
    while (1)
      ;
  }

  Serial.println("🚀 System ready. Waiting for IMU data...");
}

void loop() {
  if (isButtonPressed(state)) {
    Serial.println("Starting gesture recognition");
    motion.stop();
    delay(1000);
    buzz();

    state = bufferFilling;

    return;
  }

  if (isDriving(state)) {
    motion.update(pitch, roll);
  } else if (isBufferFilling(state) && imuBuffer.isFull()) {
    buzz();
    Serial.print("Running gesture: ");

    state = classifying;
    imuBuffer.getWindow(imu_window);
    Gesture gesture = gestureClassifier.classify(imu_window);

    switch (gesture) {
      case circle_left:
        Serial.println("circle_left");
        motion.circleLeft();
        break;
      case circle_right:
        Serial.println("circle_right");
        motion.circleRight();
        break;
      case spin_left:
        Serial.println("spin_left");
        motion.spinLeft();
        break;
      case spin_right:
        Serial.println("spin_right");
        motion.spinRight();
        break;
      case zigzag_left:
        Serial.println("zigzag_left");
        motion.zigzagLeft();
        break;
      case zigzag_right:
        Serial.println("zigzag_right");
        motion.zigzagRight();
        break;
      default:
        Serial.println("none");
        break;
    }

    Serial.println("Finished gesture recognition");
    state = driving;
    imuBuffer.reset();
    buzz();
  }
}


void buzz() {
  tone(buzzerPin, 2000);
  delay(400);
  noTone(buzzerPin);
}


bool isDriving(State state) {
  return state == driving;
}

bool isButtonPressed(State state) {
  return state == buttonPressed;
}

bool isBufferFilling(State state) {
  return state == bufferFilling;
}

bool isClassifying(State state) {
  return state == classifying;
}
