from config import GESTURES, EPOCHS, BATCH_SIZE
from data_loader import load_sequences
from data_augmentation import augment_training_data
from normalize import standardize_dataset
from model import build_model
from quantization import convert_model_to_tflite
from header_utils import create_header_from_model
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Load data
X, y = load_sequences()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

warp_factors = [0.5, 0.75, 1.25, 1.5]
X_train, y_train = augment_training_data(X_train, y_train, warp_factors, target_len=X_train.shape[1])

X_train, X_test, mean, std = standardize_dataset(X_train, X_test)

# Train model
model = build_model()
model.summary()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Quantize
tflite_model = convert_model_to_tflite(model, X_train)

# Get quant params
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

# Generate header
create_header_from_model(tflite_model, input_scale, input_zero_point, output_scale, output_zero_point, mean, std, GESTURES)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")



# Display confusion matrix
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GESTURES)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)


