"""
Simple linear regression (y = wx + b) trained from scratch with gradient descent.

Every numerical step -- predictions, loss, gradients, parameter updates -- is
plain Python (loops, lists, arithmetic). matplotlib is used only to render
checkpoint plots, never for computation.
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt


def generate_synthetic_data(
    n_samples: int,
    true_w: float,
    true_b: float,
    noise_std: float,
    seed: int | None = None,
) -> Tuple[List[float], List[float]]:
    """Generate points scattered around y = true_w * x + true_b with Gaussian noise."""
    if seed is not None:
        random.seed(seed)

    xs = [random.uniform(-10.0, 10.0) for _ in range(n_samples)]
    ys = [true_w * x + true_b + random.gauss(0.0, noise_std) for x in xs]
    return xs, ys


def predict(x: float, w: float, b: float) -> float:
    """Linear model output for a single input."""
    return w * x + b


def compute_loss(xs: List[float], ys: List[float], w: float, b: float) -> float:
    """Mean squared error between predictions and targets."""
    squared_errors = [(predict(x, w, b) - y) ** 2 for x, y in zip(xs, ys)]
    return sum(squared_errors) / len(xs)


def compute_gradients(
    xs: List[float], ys: List[float], w: float, b: float
) -> Tuple[float, float]:
    """Partial derivatives of the MSE loss with respect to w and b."""
    n = len(xs)
    error_sum_wx = 0.0
    error_sum_b = 0.0
    for x, y in zip(xs, ys):
        error = predict(x, w, b) - y
        error_sum_wx += error * x
        error_sum_b += error

    grad_w = (2.0 / n) * error_sum_wx
    grad_b = (2.0 / n) * error_sum_b
    return grad_w, grad_b
