#include <Arduino.h>
#include "GestureClassifier.h"
#include "IMUBuffer.h"
#include "model_data.h"   // imu_mean, imu_std, WINDOW_SIZE, NUM_AXES

GestureClassifier gestureClassifier;
IMUBuffer imuBuffer;

static float imu_window[WINDOW_SIZE * NUM_AXES];

void setup() {
  Serial.begin(115200);

  if (!gestureClassifier.init()) {
    Serial.println("Model init failed!");
    while (1);
  }

  Serial.println("✅ ESP32 ready for serial test (send START ... END).");
}

void loop() {
  static bool receiving = false;

  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.equalsIgnoreCase("START")) {
      imuBuffer.reset();
      receiving = true;
      return;
    }

    if (line.equalsIgnoreCase("END")) {
      if (imuBuffer.isFull()) {
        // Run classification
        imuBuffer.getWindow(imu_window);
        Gesture gesture = gestureClassifier.classify(imu_window);

        // Print result as index
        Serial.println((int)gesture);
      } else {
        Serial.println("-1"); // not enough samples
      }
      receiving = false;
      return;
    }

    if (receiving) {
      float values[6];
      int count = sscanf(line.c_str(), "%f,%f,%f,%f,%f,%f",
                         &values[0], &values[1], &values[2],
                         &values[3], &values[4], &values[5]);

      if (count == 6) {
        SensorData data;
        for (int i = 0; i < 3; i++) {
          data.accel[i] = values[i];
          data.gyro[i]  = values[i + 3];
        }
        imuBuffer.addSample(data);
      }
    }
  }
}
