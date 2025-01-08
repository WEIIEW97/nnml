"""Write a Python function that computes the transpose of a given
matrix."""


def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    b = []
    for i in range(len(a[0])):
        ele = []
        for j in range(len(a)):
            ele.append(a[j][i])
        b.append(ele)
    return b


# for very pythonic way
def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    return [list(i) for i in zip(*a)]


if __name__ == "__main__":
    a = [[1, 2, 3], [4, 5, 6]]
    print(transpose_matrix(a))
