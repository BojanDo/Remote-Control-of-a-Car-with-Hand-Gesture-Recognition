#ifndef GESTURECLASSIFIER_H
#define GESTURECLASSIFIER_H

#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include "model_data.h"
#include "MPUSensorData.h"
#include "IMUBuffer.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define DETECTION_THRESHOLD 0.90f

class GestureClassifier {
public:
  GestureClassifier();
  bool init();
  Gesture* classify(const float* imu_window);

private:
  const tflite::Model* models[NUM_MODELS];
  tflite::MicroInterpreter* interpreters[NUM_MODELS];
  TfLiteTensor* inputs[NUM_MODELS];
  TfLiteTensor* outputs[NUM_MODELS];

  static constexpr int kTensorArenaSize = 9 * 1024;
  alignas(16) uint8_t tensor_arenas[NUM_MODELS][kTensorArenaSize];
  Gesture findBestGesture(int votes[]);
};



#endif  //GESTURECLASSIFIER_H
