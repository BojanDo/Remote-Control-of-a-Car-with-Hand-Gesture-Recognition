import os
from config import GESTURES, EPOCHS, BATCH_SIZE, NUM_MODELS, HEADER_SEPARATE_OUTPUT_PATH, HEADER_ENSEMBLE_OUTPUT_PATH, \
    THRESHOLD
from data_loader import load_sequences, load_test_sequences
from data_augmentation import augment_training_data
from normalize import standardize_dataset
from model import build_model
from quantization import convert_model_to_tflite_simple, convert_model_to_tflite_biased
from header_utils import created_header_from_models
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import resample
from scipy.stats import mode
from confusion_matrix import plot_and_save_confusion_matrix

# Load data
X, y = load_sequences()
X_test, y_test = load_test_sequences()

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Add augmented data
warp_factors = [0.5, 0.75, 1.25, 1.5]
X_train, y_train = augment_training_data(X_train, y_train, warp_factors, target_len=X_train.shape[1])

# standardize data
X_train, X_val, X_test, mean, std = standardize_dataset(X_train, X_val, X_test)


def train_model(X_m, y_m):
    # Train model
    model = build_model()
    model.fit(X_m, y_m, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Quantize
    tflite_model = convert_model_to_tflite_simple(model, X_train, y_train)

    # Get quant params
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_scale, input_zero_point = input_details[0]['quantization']

    return model, tflite_model, input_scale, input_zero_point


def train_model_biased(X_m, y_m):
    # Train model
    model = build_model()
    model.fit(X_m, y_m, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Quantize
    tflite_model = convert_model_to_tflite_biased(model, X_train, y_train)

    # Get quant params
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_scale, input_zero_point = input_details[0]['quantization']

    return model, tflite_model, input_scale, input_zero_point


def model_accuracy(model, title, filename):
    # Evaluate
    y_pred_probs = model.predict(X_test)
    y_pred = []
    none_idx = GESTURES.index("none")

    for probs in y_pred_probs:
        max_prob = np.max(probs)
        if max_prob >= THRESHOLD:
            y_pred.append(np.argmax(probs))
        else:
            y_pred.append(none_idx)
    y_pred = np.array(y_pred)

    acc = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {acc:.2f}")

    plot_and_save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        labels=GESTURES,
        title=title,
        filename=filename
    )


def tflite_model_accuracy(tflite_model, title, filename):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]['quantization']

    preds = []
    none_idx = GESTURES.index("none")

    for i in range(len(X_test)):
        x = X_test[i:i + 1].astype(np.float32)

        # Quantize input
        x = x / input_scale + input_zero_point
        x = np.clip(x, -128, 127).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]  # (num_classes,)

        # Apply threshold
        max_prob = np.max(output)
        if max_prob >= THRESHOLD:
            preds.append(np.argmax(output))
        else:
            preds.append(none_idx)

    y_pred = np.array(preds)

    # Accuracy
    acc = np.mean(y_pred == y_test)
    print(f"TFLite Test Accuracy: {acc:.2f}")

    # Confusion matrix
    plot_and_save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        labels=GESTURES,
        title=title,
        filename=filename
    )

    # Accuracy
    acc = np.mean(y_pred == y_test)
    print(f"TFLite Test Accuracy: {acc:.2f}")

    # Confusion matrix
    plot_and_save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        labels=GESTURES,
        title=title,
        filename=filename
    )


def train_ensemble_models():
    models = []
    tflite_models = []
    for i in range(NUM_MODELS):
        X_boot, y_boot = resample(
            X_train, y_train,
            replace=True,
            n_samples=len(X_train),
            random_state=i
        )

        model, tflite_model, input_scale, input_zero_point = train_model(X_boot, y_boot)
        models.append(model)
        tflite_models.append(tflite_model)

        model_accuracy(model, f"Confusion Matrix - Model {i}", f"generated/ensemble/confusion_matrix_{i}.png")
        tflite_model_accuracy(tflite_model, f"Confusion Matrix - TFLite Model {i}",
                              f"generated/ensemble/confusion_matrix_tflite_{i}.png")

    # Majority vote
    for n in range(3, NUM_MODELS + 1, 2):
        preds = [np.argmax(models[i].predict(X_test), axis=1) for i in range(n)]
        preds = np.array(preds).T

        y_pred, _ = mode(preds, axis=1)
        y_pred = y_pred.ravel()

        plot_and_save_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            labels=GESTURES,
            title=f"Ensemble Confusion Matrix ({n} models)",
            filename=f"generated/ensemble/confusion_matrix_ensemble_{n}.png"
        )

    # Generate header
    created_header_from_models("ensemble", HEADER_ENSEMBLE_OUTPUT_PATH, NUM_MODELS, tflite_models, input_scale,
                               input_zero_point,
                               mean, std)


def train_separate_models():
    models = []
    tflite_models = []
    for i in range(NUM_MODELS):
        model, tflite_model, input_scale, input_zero_point = train_model(X_train, y_train)
        models.append(model)
        tflite_models.append(tflite_model)
        model_accuracy(model, f"Confusion Matrix - Model {i}", f"generated/separate/confusion_matrix_{i}.png")
        tflite_model_accuracy(tflite_model, f"Confusion Matrix - TFLite Model {i}",
                              f"generated/separate/confusion_matrix_tflite_{i}.png")

    # Generate header
    created_header_from_models("separate", HEADER_SEPARATE_OUTPUT_PATH, 1, tflite_models, input_scale,
                               input_zero_point,
                               mean, std)


def train_ensemble_models_biased():
    models = []
    tflite_models = []
    for i in range(NUM_MODELS):
        X_boot, y_boot = resample(
            X_train, y_train,
            replace=True,
            n_samples=len(X_train),
            random_state=i
        )

        model, tflite_model, input_scale, input_zero_point = train_model_biased(X_boot, y_boot)
        models.append(model)
        tflite_models.append(tflite_model)

        model_accuracy(model, f"Confusion Matrix - Model {i}", f"generated/ensemble_biased/confusion_matrix_{i}.png")
        tflite_model_accuracy(tflite_model, f"Confusion Matrix - TFLite Model {i}",
                              f"generated/ensemble_biased/confusion_matrix_tflite_{i}.png")

    # Majority vote
    for n in range(3, NUM_MODELS + 1, 2):
        preds = [np.argmax(models[i].predict(X_test), axis=1) for i in range(n)]
        preds = np.array(preds).T

        y_pred, _ = mode(preds, axis=1)
        y_pred = y_pred.ravel()

        plot_and_save_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            labels=GESTURES,
            title=f"Ensemble Confusion Matrix ({n} models)",
            filename=f"generated/ensemble_biased/confusion_matrix_ensemble_{n}.png"
        )

    # Generate header
    created_header_from_models("ensemble_biased", "../car/biased/model_data_ensemble.h", NUM_MODELS, tflite_models, input_scale,
                               input_zero_point,
                               mean, std)


def train_separate_models_biased():
    models = []
    tflite_models = []
    for i in range(NUM_MODELS):
        model, tflite_model, input_scale, input_zero_point = train_model_biased(X_train, y_train)
        models.append(model)
        tflite_models.append(tflite_model)
        model_accuracy(model, f"Confusion Matrix - Model {i}", f"generated/separate_biased/confusion_matrix_{i}.png")
        tflite_model_accuracy(tflite_model, f"Confusion Matrix - TFLite Model {i}",
                              f"generated/separate_biased/confusion_matrix_tflite_{i}.png")

    # Generate header
    created_header_from_models("separate_biased", "../car/biased/model_data_separate.h", 1, tflite_models, input_scale,
                               input_zero_point,
                               mean, std)


if __name__ == "__main__":
    os.makedirs("../car/biased", exist_ok=True)
    os.makedirs("generated/ensemble", exist_ok=True)
    os.makedirs("generated/separate", exist_ok=True)
    os.makedirs("generated/ensemble_biased", exist_ok=True)
    os.makedirs("generated/separate_biased", exist_ok=True)

    train_separate_models()
    train_ensemble_models()
    train_ensemble_models_biased()
    train_separate_models_biased()
