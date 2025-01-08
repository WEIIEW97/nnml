"""Write a Python function that calculates the mean of a matrix either
by row or by column, based on a given mode.

The function should take a matrix (list of lists) and a mode ('row' or
'column') as input and return a list of means according to the specified
mode.
"""


def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    h = len(matrix)
    w = len(matrix[0])
    means = []
    if mode == "row":
        for i in range(h):
            mu = 0
            for j in range(w):
                mu += matrix[i][j]
            means.append(mu / w)
    else:
        for j in range(w):
            mu = 0
            for i in range(h):
                mu += matrix[i][j]
            means.append(mu / h)

    return means


def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == "column":
        return [sum(col) / len(matrix) for col in zip(*matrix)]
    elif mode == "row":
        return [sum(row) / len(row) for row in matrix]
    else:
        raise ValueError("Mode must be 'row' or 'column'")


if __name__ == "__main__":
    print(calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "column"))
