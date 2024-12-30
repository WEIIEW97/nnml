"""
Given basis vectors in two different bases B and C for R^3, write a Python function to compute the transformation matrix P from basis B to C.
"""

def invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    """
    Compute the inverse of an n x n matrix using Gaussian elimination.
    
    Parameters:
    matrix (list[list[float]]): An n x n matrix to invert.
    
    Returns:
    list[list[float]]: The inverse matrix if invertible.
    
    Raises:
    ValueError: If the matrix is not square or not invertible.
    """
    n = len(matrix)

    # Check if the matrix is square
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square!")

    # Create the augmented matrix [A | I]
    aug = [row + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(matrix)]

    # Perform Gaussian elimination
    for i in range(n):
        # Find the pivot element
        pivot = aug[i][i]
        if abs(pivot) < 1e-10:  # Handle numerical instability
            for j in range(i + 1, n):
                if abs(aug[j][i]) > 1e-10:
                    aug[i], aug[j] = aug[j], aug[i]  # Swap rows
                    pivot = aug[i][i]
                    break
            if abs(pivot) < 1e-10:
                raise ValueError("Matrix is not invertible!")

        # Normalize the pivot row
        for j in range(2 * n):
            aug[i][j] /= pivot

        # Eliminate all other entries in the current column
        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(2 * n):
                    aug[k][j] -= factor * aug[i][j]

    # Extract the inverse matrix from the augmented matrix
    return [row[n:] for row in aug]


def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    """
    C = BP
    P = B^{-1}C
    """
    i_b, j_b = len(B), len(B[0])
    i_c, j_c = len(C), len(C[0])

    if j_b != i_c: raise ValueError("Incompatible matrix dimensions for multiplication")

    P = [[0] * j_c for _ in range(i_b)]

    B_inv = invert_matrix(B)

    for i in range(i_b):
        for j in range(j_c):
            P[i][j] = sum(B_inv[i][k] * C[k][j] for k in range(i_c))
    return P


def transform_basis_np(B, C):
    from numpy.linalg import inv
    import numpy as np

    # Convert input to numpy arrays for easier manipulation
    B = np.array(B)
    C = np.array(C)

    # Compute the inverse of B
    B_inv = inv(B)

    # Compute P = B^{-1} * C
    P = C @ B_inv
    return P

if __name__ == "__main__":
    B = [[1, 0, 0], 
             [0, 1, 0], 
             [0, 0, 1]]
    C = [[1, 2.3, 3], 
            [4.4, 25, 6], 
            [7.4, 8, 9]]
    
    print(transform_basis(B, C))
    print(transform_basis_np(B, C))