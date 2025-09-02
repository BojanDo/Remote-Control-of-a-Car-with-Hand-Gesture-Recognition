import numpy as np
def standardize_dataset(X_train, X_val, X_test):
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    mean = X_train_reshaped.mean(axis=0)
    std = X_train_reshaped.std(axis=0)

    std[std == 0] = 1e-6 # Avoid division by zero

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm.astype(np.float32), X_val_norm.astype(np.float32), X_test_norm.astype(np.float32), mean, std
