GESTURES = [
    "circle_left",
    "circle_right",
    "spin_left",
    "spin_right",
    "zigzag_left",
    "zigzag_right",
    "none"
]
NUM_MODELS = 5
DATA_DIR = "../gesture_data"
SAMPLES_PER_SEQUENCE = 500
EPOCHS = 200
BATCH_SIZE = 16
TFLITE_MODEL_PATH = "gesture_model.tflite"
HEADER_OUTPUT_PATH = "../car/model_data.h"
