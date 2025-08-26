#include <Arduino.h>
#include <stdbool.h>

#include "CarESPNow.h"
#include "GestureClassifier.h"
#include "IMUBuffer.h"
#include "MPUSensorData.h"

#define MPU_SDA_PIN 18
#define MPU_SCL_PIN 19


TwoWire wire = TwoWire(1);
CarESPNow espNow;
IMUBuffer imuBuffer;
GestureClassifier gestureClassifier;

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
    delay(1000);
    buzz();
    state = bufferFilling;
    return;
  }

 if (isBufferFilling(state) && imuBuffer.isFull()) {
    buzz();
    Serial.print("Running gesture: ");

    state = classifying;
    imuBuffer.getWindow(imu_window);
    Gesture gesture = gestureClassifier.classify(imu_window);

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
