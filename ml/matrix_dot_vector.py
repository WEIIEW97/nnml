"""Write a Python function that takes the dot product of a matrix and a
vector.

return -1 if the matrix could not be dotted with the vector
"""


def matrix_dot_vector(
    a: list[list[int | float]], b: list[int | float]
) -> list[int | float]:
    ah = len(a)
    aw = len(a[0])
    bw = len(b)
    if bw != aw:
        return -1
    c = []
    for i in range(ah):
        s = 0
        for j in range(aw):
            s += a[i][j] * b[j]
        c.append(s)
    return c


if __name__ == "__main__":
    a = [[1,2],[2,4]]
    b = [1,2]
    print(matrix_dot_vector(a, b))