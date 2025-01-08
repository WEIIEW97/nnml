"""Write a Python function that approximates the Singular Value Decomposition on a 2x2 matrix by using 
the jacobian method and without using numpy svd function, i mean you could but you wouldn't learn anything. 
return the result in this format.
"""

import numpy as np


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    aa = A@A.T
    val, vec = np.linalg.eigh(aa)

    sorted_indices = np.argsort(val)[::-1]
    val = val[sorted_indices]
    vec = vec[:, sorted_indices]

    sigma = np.sqrt(np.maximum(val, 0))
    # v = np.zeros_like(A.T)

    # for i in range(len(sigma)):
    #     v[:, i] = A.T @ vec[:, i] / sigma[i]
    u = A.T@vec/sigma
    
    return (vec, sigma, u)


if __name__ == "__main__":
    print(svd_2x2_singular_values(np.array([[2, 1], [1, 2]])))
    print(np.linalg.svd(np.array([[2, 1], [1, 2]])))
