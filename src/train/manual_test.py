import numpy as np
from confusion_matrix import plot_and_save_confusion_matrix

import numpy as np


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


_1_model_0 = np.array([
    [10, 0, 0, 0, 0, 0, 0],  # circle_left
    [0, 10, 0, 0, 0, 0, 0],  # circle_right
    [0, 0, 10, 0, 0, 0, 0],  # spin_left
    [0, 6, 0, 2, 0, 0, 2],  # spin_right
    [0, 0, 0, 0, 10, 0, 0],  # zigzag_left
    [0, 0, 0, 0, 4, 2, 4],  # zigzag_right
    [2, 0, 0, 0, 0, 0, 8],  # none
], dtype=int)

generate_confusion_matrix(_1_model_0, "Confusion Matrix - Model 0", "manual/confusion_matrix_1_model_0.png")

_1_model_1 = np.array([
    [7, 0, 0, 0, 0, 0, 3],  # circle_left
    [0, 10, 0, 0, 0, 0, 0],  # circle_right
    [0, 0, 10, 0, 0, 0, 0],  # spin_left
    [0, 0, 0, 10, 0, 0, 0],  # spin_right
    [0, 0, 0, 0, 9, 0, 1],  # zigzag_left
    [0, 0, 0, 0, 8, 0, 2],  # zigzag_right
    [0, 0, 0, 0, 1, 0, 9],  # none
], dtype=int)

generate_confusion_matrix(_1_model_1, "Confusion Matrix - Model 1", "manual/confusion_matrix_1_model_1.png")

_1_model_2 = np.array([
    [0, 0, 10, 0, 0, 0, 0],  # circle_left
    [0, 10, 0, 0, 0, 0, 0],  # circle_right
    [0, 0, 10, 0, 0, 0, 0],  # spin_left
    [0, 0, 2, 0, 0, 0, 8],  # spin_right
    [0, 0, 0, 0, 8, 0, 2],  # zigzag_left
    [0, 0, 0, 0, 0, 10, 0],  # zigzag_right
    [2, 0, 0, 0, 1, 0, 7],  # none
], dtype=int)

generate_confusion_matrix(_1_model_2, "Confusion Matrix - Model 2", "manual/confusion_matrix_1_model_2.png")

_1_model_3 = np.array([
    [10, 0, 0, 0, 0, 0, 0],  # circle_left
    [0, 9, 0, 0, 0, 0, 1],  # circle_right
    [0, 0, 10, 0, 0, 0, 0],  # spin_left
    [0, 0, 9, 0, 0, 0, 1],  # spin_right
    [0, 0, 0, 0, 3, 0, 7],  # zigzag_left
    [0, 0, 0, 0, 0, 10, 0],  # zigzag_right
    [0, 1, 0, 0, 1, 0, 8],  # none
], dtype=int)

generate_confusion_matrix(_1_model_3, "Confusion Matrix - Model 3", "manual/confusion_matrix_1_model_3.png")

_1_model_4 = np.array([
    [5, 1, 0, 0, 0, 0, 4],  # circle_left
    [0, 10, 0, 0, 0, 0, 0],  # circle_right
    [0, 0, 10, 0, 0, 0, 0],  # spin_left
    [0, 0, 0, 10, 0, 0, 0],  # spin_right
    [0, 0, 0, 0, 9, 0, 1],  # zigzag_left
    [0, 0, 0, 0, 4, 6, 0],  # zigzag_right
    [1, 0, 0, 0, 1, 0, 8],  # none
], dtype=int)

generate_confusion_matrix(_1_model_4, "Confusion Matrix - Model 4", "manual/confusion_matrix_1_model_4.png")


_3_model = np.array([
    [22, 0, 3, 0, 0, 0, 5],  # circle_left
    [0, 30, 0, 0, 0, 0, 0],  # circle_right
    [0, 0, 30, 0, 0, 0, 0],  # spin_left
    [0, 23, 3, 4, 0, 0, 0],  # spin_right
    [0, 0, 0, 3, 26, 0, 1],  # zigzag_left
    [0, 0, 0, 0, 18, 12, 0],  # zigzag_right
    [3, 0, 2, 0, 1, 1, 23],  # none
], dtype=int)

generate_confusion_matrix(_3_model, "Ensemble Confusion Matrix (3 models)", "manual/confusion_matrix_3_model.png")

_5_model = np.array([
    [29, 0, 0, 0, 0, 0, 1],  # circle_left
    [0, 30, 0, 0, 0, 0, 0],  # circle_right
    [0, 0, 30, 0, 0, 0, 0],  # spin_left
    [0, 0, 1, 29, 0, 0, 0],  # spin_right
    [0, 0, 0, 0, 30, 0, 0],  # zigzag_left
    [0, 0, 0, 0, 0, 30, 0],  # zigzag_right
    [4, 0, 1, 0, 3, 1, 21],  # none
], dtype=int)

generate_confusion_matrix(_5_model, "Ensemble Confusion Matrix (5 models)", "manual/confusion_matrix_5_model.png")


# (\d{2}:\d{2}:\d{2}\.\d{3} -> Finished gesture recognition\s*|\d{2}:\d{2}:\d{2}\.\d{3} -> Starting gesture recognition)