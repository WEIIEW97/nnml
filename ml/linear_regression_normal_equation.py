"""Write a Python function that performs linear regression using the
normal equation.

The function should take a matrix X (features) and a vector y (target)
as input, and return the coefficients of the linear regression model.
Round your answer to four decimal places, -0.0 is a valid result for
rounding a very small number.
"""

import numpy as np


def linear_regression_normal_equation(
    X: list[list[float]], y: list[float]
) -> list[float]:
    # Your code here, make sure to round
    X = np.array(X)
    y = np.array(y)
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return np.round(theta, 4)


if __name__ == "__main__":
    print(linear_regression_normal_equation([[1, 1], [1, 2], [1, 3]], [1, 2, 3]))
    print(
        linear_regression_normal_equation([[1, 3, 4], [1, 2, 5], [1, 3, 2]], [1, 2, 1])
    )
