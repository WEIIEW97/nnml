"""Write a Python function that calculates the eigenvalues of a 2x2 matrix. 
The function should return a list containing the eigenvalues, sort values from highest to lowest.
"""

import math


def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    tr = matrix[0][0] + matrix[1][1]
    lmbda1 = (tr - math.sqrt(tr**2 - 4 * det)) / 2
    lmbda2 = (tr + math.sqrt(tr**2 - 4 * det)) / 2

    return sorted([lmbda1, lmbda2], reverse=True)


if __name__ == "__main__":
    print(calculate_eigenvalues([[2, 1], [1, 2]]))
