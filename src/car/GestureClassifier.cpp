#include "GestureClassifier.h"

GestureClassifier::GestureClassifier()
  : model(nullptr), interpreter(nullptr), input(nullptr), output(nullptr) {}

bool GestureClassifier::init() {
  model = tflite::GetModel(reinterpret_cast<const void*>(gesture_model_tflite));
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("❌ Model version mismatch!");
    return false;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("❌ AllocateTensors() failed");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

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
  int input_index = 0;
  for (int i = 0; i < WINDOW_SIZE * NUM_AXES; ++i) {
    int8_t quantized = (int8_t)(imu_window[i] / INPUT_SCALE + INPUT_ZERO_POINT);
    input->data.int8[input_index++] = quantized;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    return none;
  }

  int8_t* output_data = output->data.int8;
  float max_prob = 0.0f;
  int max_index = -1;

  for (int i = 0; i < GESTURE_CLASSES; ++i) {
    float prob = (output_data[i] - output->params.zero_point) * output->params.scale;
    if (prob > max_prob) {
      max_prob = prob;
      max_index = i;
    }
  }
  if (max_prob > DETECTION_THRESHOLD) {
    return intToGesture(max_index);
  }
  return none;
}
