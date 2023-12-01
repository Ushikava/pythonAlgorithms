import re
import matplotlib.pyplot as plt
import math
from typing import Dict, Iterable, Tuple, List, Callable
from collections import defaultdict, Counter
import random
from scratch.linear_algebra import Vector, dot

#Vector = List[float]
Matrix = List[List[float]]


def add(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), "Векторы должны быть одной длины!"
    return [(v + w) for v, w in zip(v, w)]


def subtract(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), "Векторы должны быть одной длины!"
    return [(v - w) for v, w in zip(v, w)]


def vec_summ(vectors: List[Vector]) -> Vector:
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "Размеры векторов разные!"
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


def vec_scalar(v: Vector, s: float) -> Vector:
    return [(v_i * s) for v_i in v]


def vector_mean(vectors: List[Vector]) -> Vector:
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "Размеры векторов разные!"
    n = len(vectors)
    return vec_scalar(vec_summ(vectors), 1 / n)


def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w), "Векторы должны быть одной длины!"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v: Vector) -> float:
    return dot(v, v)


def magnitude(v: Vector) -> float:
    return math.sqrt(sum_of_squares(v))


def squared_distance(v: Vector, w: Vector) -> float:
    return sum_of_squares(subtract(v, w))


def euklid_distance(v: Vector, w: Vector) -> float:
    return math.sqrt(squared_distance(v, w))


def shape(A: Matrix) -> Tuple[int, int]:
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols


def get_row(A: Matrix, i: int) -> Vector:
    return A[i]


def get_column(A: Matrix, j: int) -> Vector:
    return [A_i[j] for A_i in A]


def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]


def identity_matrix(n: int) -> Matrix:
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _median_odd(xs: List[float]) -> float:
    return sorted(xs)[len(xs) // 2]


def _median_even(xs: List[float]) -> float:
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2


def median(v: List[float]) -> float:
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)


def quantile(xs: List[float], p: float) -> float:
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]


def mode(x: List[float]) -> List[float]:
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]


def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)


def square(a):
    return a ** 2


def my_sum_of_squares(xs: List[float]):
    s = 0
    for i in range(len(xs)):
        s += square(i)
    return s


def de_mean(xs: List[float]) -> List[float]:
    x_bar = mean(xs)
    return [x - x_bar for x in xs]


def variance(xs: List[float]) -> float:
    assert len(xs) >= 2
    n = len(xs)
    deviations = de_mean(xs)
    return my_sum_of_squares(deviations) / (n - 1)


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


def bernoulli_trial(p: float) -> int:
    return 1 if random.random() < p else 0


def binomial(n: int, p: float) -> int:
    return sum(bernoulli_trial(p) for _ in range(n))


def binomial_histogram(p: float, n: int, num_points: int) -> None:
    data = [binomial(n, p) for _ in range(num_points)]
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')
    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) for i in xs]
    plt.plot(xs, ys)
    plt.title("Биномиальное распределение")
    plt.show()


def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma


def difference_quotient(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x)) / h


num_friends = [12, 15, 16, 20, 100, 15, 18, 18]
print(variance(num_friends))

#binomial_histogram(0.75, 100, 10000)

inputs = [(x, 20 * x + 5) for x in range(-50, 50)]
print(inputs)
