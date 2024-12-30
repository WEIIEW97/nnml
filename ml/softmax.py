"""
Write a Python function that computes the softmax activation for a given list of scores. The function should return the softmax values as a list, each rounded to four decimal places.
"""

import math


def softmax(scores: list[float]) -> list[float]:
    denom = sum(math.exp(j) for j in scores)
    prob = [round(math.exp(i) / denom, 4) for i in scores]
    return prob


if __name__ == "__main__":
    scores = [1, 2, 3]
    print(softmax(scores))
