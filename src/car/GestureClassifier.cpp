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

  for (int m = 0; m < NUM_MODELS; ++m) {
    // Fill input buffer
    int input_index = 0;
    for (int i = 0; i < WINDOW_SIZE * NUM_AXES; ++i) {
      int8_t quantized = (int8_t)(imu_window[i] / inputs[m]->params.scale + inputs[m]->params.zero_point);
      inputs[m]->data.int8[input_index++] = quantized;
    }

    // Run inference
    if (interpreters[m]->Invoke() != kTfLiteOk) {
      Serial.println("Inference failed on model " + String(m));
      continue;
    }

    // Find argmax for this model
    int8_t* output_data = outputs[m]->data.int8;
    float max_prob = -1.0f;
    int max_index = -1;
    for (int i = 0; i < GESTURE_CLASSES; ++i) {
      float prob = (output_data[i] - outputs[m]->params.zero_point) * outputs[m]->params.scale;
      if (prob > max_prob) {
        max_prob = prob;
        max_index = i;
      }
    }

    // Only count as a "vote" if it's above threshold
    if (max_prob > DETECTION_THRESHOLD) {
      votes[max_index]++;
    } else {
      votes[none]++;
    }
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

  return (best_class != -1)? intToGesture(best_class) : none;
}
