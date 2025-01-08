"""Write a Python function that calculates the covariance matrix from a
list of vectors.

Assume that the input list represents a dataset where each vector is a
feature, and vectors are of equal length.
"""


def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n = len(vectors)
    m = len(vectors[0])

    mus = [sum(v) / m for v in vectors]
    cov = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            c = sum(
                (vectors[i][k] - mus[i]) * (vectors[j][k] - mus[j]) for k in range(m)
            ) / (m - 1)
            cov[i][j] = cov[j][i] = c
    return cov


if __name__ == "__main__":
    vectors = [[1, 2, 3], [4, 5, 6]]
    print(calculate_covariance_matrix(vectors))
