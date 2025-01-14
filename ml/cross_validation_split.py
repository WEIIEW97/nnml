"""Write a Python function that performs k-fold cross-validation data
splitting from scratch.

The function should take a dataset (as a 2D NumPy array where each row
represents a data sample and each column represents a feature) and an
integer k representing the number of folds. The function should split
the dataset into k parts, systematically use one part as the test set
and the remaining as the training set, and return a list where each
element is a tuple containing the training set and test set for each
fold.
"""
import numpy as np


def cross_validation_split(data: np.ndarray, k: int, seed=42) -> list:
    np.random.seed(seed)
    # Your code here
    np.random.shuffle(data)

    fold_size = len(data) // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(data)
        fold = data[start:end]
        folds.append(fold)

    cv = []

    for i in range(k):
        test_split = folds[i]
        train_split = np.concatenate([folds[j] for j in range(k) if j != i], axis=0)
        cv.append((train_split, test_split))
    return cv


if __name__ == "__main__":
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    k = 5
    print(cross_validation_split(data, k))
