"""Write a Python function that approximates the Singular Value
Decomposition on a 2x2 matrix by using the jacobian method and without
using numpy svd function, i mean you could but you wouldn't learn
anything.

return the result in this format.
"""

import numpy as np


def svd_2x2_singular_values(A: np.ndarray):
    AtA = A.T @ A
    val, vec = np.linalg.eigh(AtA)

    idx = np.argsort(val)[::-1]
    val = val[idx]
    V = vec[:, idx]
    sigma = np.sqrt(val)

    U = np.zeros_like(A, dtype=np.float64)
    for i in range(2):
        U[:, i] = np.array((A @ V[:, i]) / sigma[i])

    return U, sigma, V.T


if __name__ == "__main__":
    print(svd_2x2_singular_values(np.array([[1, 2], [3, 4]])))
    print("\n")
    print(np.linalg.svd(np.array([[1, 2], [3, 4]])))
