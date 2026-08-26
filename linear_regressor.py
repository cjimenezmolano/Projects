"""
Linear regression (y = wx + b) fit by hand with gradient descent.

No numpy/sklearn -- predictions, loss, and gradients are plain Python.
matplotlib is only for plotting, not computation.
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


def save_checkpoint_plot(
    xs: List[float],
    ys: List[float],
    w: float,
    b: float,
    epoch: int,
    output_dir: str,
) -> None:
    """Plot the data against the current fitted line and save it as epoch_XXX.png."""
    os.makedirs(output_dir, exist_ok=True)

    # Two points fully determine the line; no need to sample along the whole range.
    x_min, x_max = min(xs), max(xs)
    line_xs = [x_min, x_max]
    line_ys = [predict(x, w, b) for x in line_xs]

    fig, ax = plt.subplots()
    ax.scatter(xs, ys, s=15, alpha=0.6, label="data")
    ax.plot(line_xs, line_ys, color="crimson", label=f"y = {w:.2f}x + {b:.2f}")
    ax.set_title(f"epoch {epoch}")
    ax.legend()

    fig.savefig(os.path.join(output_dir, f"epoch_{epoch:03d}.png"))
    plt.close(fig)  # avoid accumulating open figures across many checkpoints


def train(
    xs: List[float],
    ys: List[float],
    learning_rate: float,
    epochs: int,
    checkpoint_interval: int,
    output_dir: str,
) -> Tuple[float, float]:
    """Fit w and b by batch gradient descent, logging and plotting at checkpoints."""
    w, b = 0.0, 0.0  # flat line through the origin; a neutral, arbitrary start

    for epoch in range(1, epochs + 1):
        grad_w, grad_b = compute_gradients(xs, ys, w, b)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        # Always checkpoint the final epoch too, even if it doesn't land on the interval.
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            loss = compute_loss(xs, ys, w, b)
            print(f"epoch {epoch:4d} | loss {loss:.4f} | w {w:.4f} | b {b:.4f}")
            save_checkpoint_plot(xs, ys, w, b, epoch, output_dir)

    return w, b


if __name__ == "__main__":
    LEARNING_RATE = 0.01
    EPOCHS = 200
    CHECKPOINT_INTERVAL = 10
    N_SAMPLES = 60
    TRUE_W, TRUE_B = 2.5, -1.0
    NOISE_STD = 2.0
    SEED = 42
    OUTPUT_DIR = "plots"

    xs, ys = generate_synthetic_data(N_SAMPLES, TRUE_W, TRUE_B, NOISE_STD, seed=SEED)
    final_w, final_b = train(
        xs, ys, LEARNING_RATE, EPOCHS, CHECKPOINT_INTERVAL, OUTPUT_DIR
    )

    print(f"true params:    w={TRUE_W}, b={TRUE_B}")
    print(f"learned params: w={final_w:.4f}, b={final_b:.4f}")
