#include <Arduino.h>
#include <stdbool.h>

#include "CarESPNow.h"

CarESPNow espNow;

void onDataReceived(const SensorData& data) {
  Serial.printf("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                data.accel[0], data.accel[1], data.accel[2],
                data.gyro[0], data.gyro[1], data.gyro[2]);
}

void setup() {
  Serial.begin(115200);

  if (!espNow.begin(onDataReceived)) {
    Serial.println("ESP-NOW init failed!");
    while (1)
      ;
  }

  Serial.println("🚀 System ready. Waiting for IMU data...");
}

void loop() {

}
