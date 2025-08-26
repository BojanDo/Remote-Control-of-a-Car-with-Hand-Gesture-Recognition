from config import GESTURES, EPOCHS, BATCH_SIZE, NUM_MODELS
from data_loader import load_sequences
from data_augmentation import augment_training_data
from normalize import standardize_dataset
from model import build_model
from quantization import convert_model_to_tflite
from header_utils import created_header_from_models
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import resample


from confusion_matrix import plot_and_save_confusion_matrix

# Load data
X, y = load_sequences()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Add augmented data
warp_factors = [0.5, 0.75, 1.25, 1.5]
X_train, y_train = augment_training_data(X_train, y_train, warp_factors, target_len=X_train.shape[1])

# standardize data
X_train, X_test, mean, std = standardize_dataset(X_train, X_test)

models = []
tflite_models = []
quant_params = []
for i in range(NUM_MODELS):
    # Bootstrap sampling
    X_boot, y_boot = resample(
        X_train, y_train,
        replace=True,
        n_samples=len(X_train),
        random_state=i
    )

    # Train model
    model = build_model()
    model.fit(X_boot, y_boot, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
    models.append(model)

    # Quantize
    tflite_model = convert_model_to_tflite(model, X_train)
    tflite_models.append(tflite_model)

    # Get quant params
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]['quantization']

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.2f}")


    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    plot_and_save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        labels=GESTURES,
        title=f"Confusion Matrix - Model {i}",
        filename=f"generated/confusion_matrix_{i}.png"
    )


pred_probs_list = [model.predict(X_test) for model in models]
avg_pred_probs = np.mean(pred_probs_list, axis=0)
y_pred = np.argmax(avg_pred_probs, axis=1)

plot_and_save_confusion_matrix(
    y_true=y_test,
    y_pred=y_pred,
    labels=GESTURES,
    title="Ensemble Confusion Matrix",
    filename="generated/confusion_matrix_ensemble.png"
)



# Generate header
created_header_from_models(tflite_models, input_scale, input_zero_point, mean, std)
