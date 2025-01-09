"""Write a Python function that calculates the determinant of a 4x4
matrix using Laplace's Expansion method.

The function should take a single argument, a 4x4 matrix represented as
a list of lists, and return the determinant of the matrix. The elements
of the matrix can be integers or floating-point numbers. Implement the
function recursively to handle the computation of determinants for the
3x3 minor matrices.
"""


def determinant_4x4(M: list[list[int | float]]) -> float:
    # Your recursive implementation here
    if len(M) == 1:
        return M[0][0]

    det = 0
    for col, ele in enumerate(M[0]):
        K = [x[:col] + x[col + 1 :] for x in M[1:]]
        s = 1 if col % 2 == 0 else -1
        det += s * ele * determinant_4x4(K)
    return det


if __name__ == "__main__":
    print(determinant_4x4([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))
    print(determinant_4x4([[4, 3, 2, 1], [3, 2, 1, 4], [2, 1, 4, 3], [1, 4, 3, 2]]))