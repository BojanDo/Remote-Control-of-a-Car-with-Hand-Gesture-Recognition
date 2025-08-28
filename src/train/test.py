import time
import serial
import numpy as np
from config import GESTURES
from data_loader import load_test_sequences
from manual_test import generate_confusion_matrix


FILENAME = "confusion_matrix_1_model_3.png"
TITLE = "Confusion Matrix - Model 3"
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
TIMEOUT = 2

START_CMD = "START\n"
END_CMD = "END\n"

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

    pred = ser.readline().decode().strip()
    try:
        pred_idx = int(pred)
    except ValueError:
        pred_idx = -1
    return pred_idx


def main():
    base_dir = os.path.dirname(__file__)
    X_test, y_test = load_test_sequences()
    print(f"Loaded {len(X_test)} test sequences")

    num_classes = len(GESTURES)
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    time.sleep(2)

    for i, (seq, label) in enumerate(zip(X_test, y_test)):
        pred_idx = send_sequence(ser, seq)
        if 0 <= pred_idx < num_classes:
            confusion[label, pred_idx] += 1
        else:
            print(f"Sample {i}: Invalid prediction received -> '{pred_idx}'")

        print(f"Sample {i}: true={GESTURES[label]}, pred={GESTURES[pred_idx] if pred_idx >= 0 else 'INVALID'}")

    ser.close()
    out_path = os.path.join(base_dir, "tests", FILENAME)
    generate_confusion_matrix(confusion, TITLE, out_path)

import os



if __name__ == "__main__":
    main()

