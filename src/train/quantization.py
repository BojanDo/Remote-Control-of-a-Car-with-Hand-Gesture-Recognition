import tensorflow as tf
from config import TFLITE_MODEL_PATH

def representative_dataset(X_train):
    for i in range(len(X_train)):
        yield [X_train[i:i+1]]

def convert_model_to_tflite(model, X_train):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)
    return tflite_model
