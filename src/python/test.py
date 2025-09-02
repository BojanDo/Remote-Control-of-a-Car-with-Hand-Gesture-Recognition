import time
import serial
import numpy as np
from config import GESTURES
from data_loader import load_test_sequences
from confusion_matrix import plot_and_save_confusion_matrix


BASE_DIR = "ensemble"
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
TIMEOUT = 2

START_CMD = "START\n"
END_CMD = "END\n"

NUM_MODELS = 5
TOTAL_RESULTS = NUM_MODELS + 2


def send_sequence(ser, sequence):
    ser.write(START_CMD.encode())
    ser.flush()

    for row in sequence:
        line = ",".join([f"{v:.5f}" for v in row]) + "\n"
        ser.write(line.encode())
        ser.flush()
        time.sleep(0.001)

    ser.write(END_CMD.encode())
    ser.flush()

    pred_line = ser.readline().decode().strip()
    try:
        preds = [int(x) for x in pred_line.split(",")]
        if len(preds) != TOTAL_RESULTS:
            print(f"Warning: expected {TOTAL_RESULTS} predictions, got {len(preds)}")
            preds = [-1] * TOTAL_RESULTS
    except ValueError:
        preds = [-1] * TOTAL_RESULTS

    return preds

def generate_confusion_matrix(cm_array, title, filename):
    labels = ["circle_left", "circle_right", "spin_left", "spin_right",
              "zigzag_left", "zigzag_right", "none"]

    n = cm_array.shape[0]
    rows, cols = np.indices((n, n))
    y_true = np.repeat(rows.ravel(), cm_array.ravel())
    y_pred = np.repeat(cols.ravel(), cm_array.ravel())

    plot_and_save_confusion_matrix(
        y_true, y_pred, labels,
        title=title,
        filename=filename
    )


def main():
    base_dir = os.path.dirname(__file__) + "/tests/" + BASE_DIR
    os.makedirs(base_dir, exist_ok=True)

    X_test, y_test = load_test_sequences()
    print(f"Loaded {len(X_test)} test sequences")

    num_classes = len(GESTURES)

    confusion_matrices = [
        np.zeros((num_classes, num_classes), dtype=int) for _ in range(TOTAL_RESULTS)
    ]

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    time.sleep(2)

    for i, (seq, label) in enumerate(zip(X_test, y_test)):
        preds = send_sequence(ser, seq)

        for m_idx, pred_idx in enumerate(preds):
            if 0 <= pred_idx < num_classes:
                confusion_matrices[m_idx][label, pred_idx] += 1
            else:
                print(f"Sample {i}: Invalid prediction for model {m_idx} -> '{pred_idx}'")

        pred_str = ", ".join(
            [GESTURES[p] if 0 <= p < num_classes else "INVALID" for p in preds]
        )
        print(f"Sample {i}: true={GESTURES[label]}, preds=[{pred_str}]")

    ser.close()

    output_files = (
            [f"confusion_matrix_{i}" for i in range(NUM_MODELS)]
            + ["confusion_matrix_ensemble_3", "confusion_matrix_ensemble_5"]
    )
    output_names = (
            [f"Confusion Matrix - Model {i}" for i in range(NUM_MODELS)]
            + ["Ensemble Confusion Matrix (3 models)", "Ensemble Confusion Matrix (5 models)"]
    )

    for name, file, cm in zip(output_names, output_files, confusion_matrices):
        out_path = os.path.join(base_dir, file)
        generate_confusion_matrix(cm, name, out_path)




import os

if __name__ == "__main__":
    main()
