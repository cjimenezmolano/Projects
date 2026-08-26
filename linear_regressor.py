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
