import numpy as np
from scipy.interpolate import interp1d

def warp_sequence(data, factor, target_len):
    n = len(data)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, int(n * factor))

    f = interp1d(x_old, data, axis=0, kind='linear')
    warped = f(x_new)

    x_final = np.linspace(0, 1, target_len)
    f_final = interp1d(np.linspace(0, 1, warped.shape[0]), warped, axis=0, kind='linear')
    return f_final(x_final)


def augment_training_data(X_train, y_train, warp_factors, target_len=500):
    X_aug, y_aug = [], []
    for i, seq in enumerate(X_train):
        for factor in warp_factors:
            warped = warp_sequence(seq, factor, target_len)
            X_aug.append(warped)
            y_aug.append(y_train[i])

    if len(X_aug) > 0:
        X_train = np.concatenate([X_train, np.array(X_aug, dtype=np.float32)], axis=0)
        y_train = np.concatenate([y_train, np.array(y_aug, dtype=np.int32)], axis=0)

    print(f"Training samples after augmentation: {len(X_train)}")

    return X_train, y_train
