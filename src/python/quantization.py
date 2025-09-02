import tensorflow as tf
import numpy as np

def representative_dataset(X_train):
    for i in range(len(X_train)):
        yield [X_train[i:i+1]]


def representative_dataset1(X_train, y_train, none_class_index, num_samples=1000, bias_factor=3):
    none_idx = np.where(y_train == none_class_index)[0]
    other_idx = np.where(y_train != none_class_index)[0]

    choices = []
    for _ in range(bias_factor):
        choices.extend(none_idx)
    choices.extend(other_idx)

    np.random.shuffle(choices)
    selected = choices[:num_samples]

    for i in selected:
        yield [X_train[i:i+1]]


def convert_model_to_tflite_simple(model, X_train, y_train):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    return tflite_model


def convert_model_to_tflite_biased(model, X_train, y_train):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset1(
        X_train, y_train, 6
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    return tflite_model
