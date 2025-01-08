"""Write a Python function that uses the Jacobi method to solve a system
of linear equations given by Ax = b. The function should iterate n
times, rounding each intermediate solution to four decimal places, and
return the approximate solution x.

Example:
        input: A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]], b = [-1, 2, 3], n=2
        output: [0.146, 0.2032, -0.5175]
        reasoning: The Jacobi method iteratively solves each equation for x[i] using the formula x[i] = (1/a_ii) * (b[i] - sum(a_ij * x[j] for j != i)), where a_ii is the diagonal element of A and a_ij are the off-diagonal elements.
"""

import numpy as np


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    x = np.zeros_like(b)
    D = np.diagonal(A)  # Diagonal elements of A
    LU = A - np.diag(D)
    for _ in range(n):
        x = (b - LU @ x) / D
    return np.round(x, 4)


if __name__ == "__main__":
    A = np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]])
    b = np.array([-1, 2, 3])
    print(solve_jacobi(A, b, 2))
