#include <MPU6050_WE.h>
#include <Wire.h>
#include <esp_now.h>
#include <WiFi.h>
#include "MPUSensorData.h"
#include "HandESPNow.h"

const int csPin = 27;
const int buttonPin = 16;
bool useSPI = true;

MPU6500_WE mpu = MPU6500_WE(&SPI, csPin, useSPI);
SensorData sensorData;
uint8_t receiverMAC[] = { 0xE8, 0x6B, 0xEA, 0xD8, 0xF5, 0xBC };
HandESPNow espNow(receiverMAC);

void setup() {
  Serial.begin(115200);

  pinMode(buttonPin, INPUT_PULLUP);

  if (!mpu.init()) {
    Serial.println("MPU6050 does not respond");
  } else {
    Serial.println("MPU6050 is connected");
  }

  Serial.println("Position you MPU6050 flat and don't move it - calibrating...");
  delay(1000);
  mpu.autoOffsets();
  Serial.println("Done!");

  mpu.setSampleRateDivider(5);
  mpu.setAccRange(MPU6500_ACC_RANGE_2G);
  mpu.enableAccDLPF(true);
  mpu.setAccDLPF(MPU6500_DLPF_3);

  delay(200);

  if (!espNow.begin()) {
    Serial.println("ESP-NOW init failed!");
    return;
  }
}





void loop() {
  xyzFloat gValue = mpu.getGValues();
  xyzFloat gyr = mpu.getGyrValues();

  float pitch = mpu.getPitch();
  float roll = mpu.getRoll();


  sensorData.accel[0] = gValue.x;
  sensorData.accel[1] = gValue.y;
  sensorData.accel[2] = gValue.z;

  sensorData.gyro[0] = gyr.x;
  sensorData.gyro[1] = gyr.y;
  sensorData.gyro[2] = gyr.z;

  sensorData.pitch = pitch;
  sensorData.roll = roll;

  sensorData.isButtonPressed = digitalRead(buttonPin) == LOW;


  Serial.printf("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                sensorData.accel[0], sensorData.accel[1], sensorData.accel[2],
                sensorData.gyro[0], sensorData.gyro[1], sensorData.gyro[2]);

  esp_err_t result = espNow.sendData(sensorData);

  if (result == ESP_OK) {
    Serial.print("Send Error: ");
    Serial.println(result);
  }

  delay(5);
}
