"""Write a Python function that performs linear regression using
gradient descent.

The function should take NumPy arrays X (features with a column of ones
for the intercept) and y (target) as input, along with learning rate
alpha and the number of iterations, and return the coefficients of the
linear regression model as a NumPy array. Round your answer to four
decimal places. -0.0 is a valid result for rounding a very small number.
"""

import numpy as np


def linear_regression_gradient_descent(
    X: np.ndarray, y: np.ndarray, alpha: float, iterations: int
) -> np.ndarray:
    # Your code here, make sure to round
    m, n = X.shape
    theta = np.zeros((n, 1))
    for _ in range(iterations):
        grad = X.T @ (X @ theta - y.reshape(-1, 1)) / m
        theta -= alpha * grad
    return np.round(theta, 4)


if __name__ == "__main__":
    print(
        linear_regression_gradient_descent(
            np.array([[1, 1], [1, 2], [1, 3]]), np.array([1, 2, 3]), 0.01, 1000
        )
    )
