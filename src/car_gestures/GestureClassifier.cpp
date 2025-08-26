#include "GestureClassifier.h"

GestureClassifier::GestureClassifier() {}

bool GestureClassifier::init() {
  static tflite::AllOpsResolver resolver;
  for (int i = 0; i < NUM_MODELS; ++i) {
    models[i] = tflite::GetModel(reinterpret_cast<const void*>(tflite_models[i]));
    if (models[i]->version() != TFLITE_SCHEMA_VERSION) {
      Serial.println("❌ Model version mismatch!");
      return false;
    }
  }

  static tflite::MicroInterpreter static_interpreters[NUM_MODELS] = {
    tflite::MicroInterpreter(models[0], resolver, tensor_arenas[0], kTensorArenaSize),
    tflite::MicroInterpreter(models[1], resolver, tensor_arenas[1], kTensorArenaSize),
    tflite::MicroInterpreter(models[2], resolver, tensor_arenas[2], kTensorArenaSize),
    tflite::MicroInterpreter(models[3], resolver, tensor_arenas[3], kTensorArenaSize),
    tflite::MicroInterpreter(models[4], resolver, tensor_arenas[4], kTensorArenaSize)
  };

  for (int i = 0; i < NUM_MODELS; ++i) {
    interpreters[i] = &static_interpreters[i];

    if (interpreters[i]->AllocateTensors() != kTfLiteOk) {
      Serial.println("❌ AllocateTensors() failed");
      return false;
    }

    inputs[i] = interpreters[i]->input(0);
    outputs[i] = interpreters[i]->output(0);
  }

  return true;
}

void countdown() {
  Serial.println("3");
  delay(1000);
  Serial.println("2");
  delay(1000);
  Serial.println("1");
  delay(1000);
}

Gesture GestureClassifier::classify(const float* imu_window) {
  int votes[GESTURE_CLASSES] = {0};

  // Labels for debug printing
  const char* gestureLabels[GESTURE_CLASSES] = {
    "circle_left", "circle_right",
    "spin_left", "spin_right",
    "zigzag_left", "zigzag_right",
    "none"
  };

  for (int m = 0; m < NUM_MODELS; ++m) {
    // Fill input buffer
    int input_index = 0;
    for (int i = 0; i < WINDOW_SIZE * NUM_AXES; ++i) {
      int8_t quantized = (int8_t)(imu_window[i] / SCALE + ZERO_POINT);
      inputs[m]->data.int8[input_index++] = quantized;
    }

    // Run inference
    if (interpreters[m]->Invoke() != kTfLiteOk) {
      Serial.println("Inference failed on model " + String(m));
      continue;
    }

    // Print probabilities for this model
    Serial.println("📊 Model " + String(m) + " probabilities:");
    int8_t* output_data = outputs[m]->data.int8;
    float max_prob = -1.0f;
    int max_index = -1;
    for (int i = 0; i < GESTURE_CLASSES; ++i) {
      float prob = (output_data[i] - outputs[m]->params.zero_point) * outputs[m]->params.scale;
      Serial.print("   ");
      Serial.print(gestureLabels[i]);
      Serial.print(": ");
      Serial.println(prob, 4);

      if (prob > max_prob) {
        max_prob = prob;
        max_index = i;
      }
    }

    // Only count as a "vote" if it's above threshold
    if (max_prob > DETECTION_THRESHOLD) {
      votes[max_index]++;
      Serial.println("✅ Model " + String(m) + " votes for: " + String(gestureLabels[max_index]));
    } else {
      Serial.println("⚠️ Model " + String(m) + " no vote (max=" + String(max_prob, 4) + ")");
    }
  }

  // Print votes summary
  Serial.println("🗳 Votes summary:");
  for (int i = 0; i < GESTURE_CLASSES; ++i) {
    Serial.print("   ");
    Serial.print(gestureLabels[i]);
    Serial.print(": ");
    Serial.println(votes[i]);
  }

  // Find class with most votes
  int best_class = -1;
  int best_votes = 0;
  for (int i = 0; i < GESTURE_CLASSES; ++i) {
    if (votes[i] > best_votes) {
      best_votes = votes[i];
      best_class = i;
    }
  }

  if (best_class != -1) {
    Serial.println("🏆 Final prediction: " + String(gestureLabels[best_class]));
  } else {
    Serial.println("🚫 No gesture detected");
  }

  return (best_class != -1) ? intToGesture(best_class) : none;
}
