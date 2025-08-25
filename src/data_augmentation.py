import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# -------------------------
# 1. Load and save helpers
# -------------------------
def load_csv(filepath):
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(filepath)

def save_csv(data, columns, filepath):
    """Save numpy array data (shape: n_samples x n_features) as CSV with headers."""
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filepath, index=False)

# -------------------------
# 2. Core augmentation step
# -------------------------
def warp_sequence(data, factor, target_len):
    """
    Apply window-warping to a sequence:
    - data: numpy array shape (n_samples, n_features)
    - factor: float (e.g. 0.5, 0.75, 1.25, 1.5)
    - target_len: final desired length (500 in your case)
    """
    n = len(data)

    # original time axis (normalized to 0..1)
    x_old = np.linspace(0, 1, n)

    # warped time axis (shorter or longer depending on factor)
    x_new = np.linspace(0, 1, int(n * factor))

    # interpolate to warped sequence
    f = interp1d(x_old, data, axis=0, kind='linear')
    warped = f(x_new)

    # resample warped sequence back to target length
    x_final = np.linspace(0, 1, target_len)
    f_final = interp1d(np.linspace(0, 1, warped.shape[0]), warped, axis=0, kind='linear')
    return f_final(x_final)

# -------------------------
# 3. Augmentation pipeline
# -------------------------
def augment_file(filepath, warp_factors, target_len=500):
    """
    Augment a single CSV file with multiple warp factors.
    Returns: dict of {augmented_filename: augmented_data}
    """
    df = load_csv(filepath)
    data = df.values
    augmented = {}

    for factor in warp_factors:
        warped = warp_sequence(data, factor, target_len)
        out_name = filepath.replace(".csv", f"_warp{factor}.csv")
        augmented[out_name] = warped

    return augmented, df.columns

def augment_dataset(root, warp_factors, target_len=500):
    """
    Traverse dataset folder structure and augment all CSV files.
    """
    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue

        for fname in os.listdir(cls_path):
            if not fname.endswith(".csv"):
                continue

            filepath = os.path.join(cls_path, fname)
            augmented, columns = augment_file(filepath, warp_factors, target_len)

            for out_path, data in augmented.items():
                save_csv(data, columns, out_path)

# -------------------------
# 4. Run augmentation
# -------------------------
if __name__ == "__main__":
    root = "gesture_data"
    warp_factors = [0.5, 0.75, 1.25, 1.5]
    augment_dataset(root, warp_factors)
