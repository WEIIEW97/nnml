"""Write a Python function that performs feature scaling on a dataset
using both standardization and min-max normalization.

The function should take a 2D NumPy array as input, where each row
represents a data sample and each column represents a feature. It should
return two 2D NumPy arrays: one scaled by standardization and one by
min-max normalization. Make sure all results are rounded to the nearest
4th decimal.
"""

import numpy as np


def feature_scaling(X: np.ndarray) -> (np.ndarray, np.ndarray):
    # Your code here
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_standardized = (X - mean) / std

    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    range_ = X_max - X_min
    range_[range_ == 0] = 1.0

    X_minmax = (X - X_min) / range_

    X_standardized = np.round(X_standardized, 4)
    X_minmax = np.round(X_minmax, 4)

    return X_standardized, X_minmax


if __name__ == "__main__":
    print(feature_scaling(np.array([[1, 2], [3, 4], [5, 6]])))
