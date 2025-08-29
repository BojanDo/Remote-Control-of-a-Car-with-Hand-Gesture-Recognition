import os
import pandas as pd
import numpy as np
from config import GESTURES, DATA_DIR, TEST_DIR, SAMPLES_PER_SEQUENCE


def load_sequences_from_directory(data_dir, gestures):
    X, y = [], []
    for label_idx, gesture in enumerate(gestures):
        gesture_dir = os.path.join(data_dir, gesture)
        if not os.path.exists(gesture_dir):
            continue
        for file in os.listdir(gesture_dir):
            if file.endswith(".csv"):
                path = os.path.join(gesture_dir, file)
                try:
                    df = pd.read_csv(path)
                    if len(df) == SAMPLES_PER_SEQUENCE:
                        X.append(df.values)
                        y.append(label_idx)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    return X, y

def load_sequences():
    X, y = load_sequences_from_directory(DATA_DIR, GESTURES)
    return np.array(X), np.array(y)

def load_test_sequences():
    X, y = load_sequences_from_directory(TEST_DIR, GESTURES)
    return np.array(X), np.array(y)

