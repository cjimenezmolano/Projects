"""
Day 1 — Logistic regression from scratch (sigmoid, cross-entropy, gradient descent).

Fill in every block marked "START CODE HERE" / "END CODE HERE". Everything
else is scaffolding, provided so you can focus on the math.

No numpy/sklearn/scipy — pure Python only. matplotlib is used only for
plotting, never for computation.
"""

from __future__ import annotations

import math
import os
import random
from typing import List, Tuple


def generate_synthetic_data(
    n_samples: int,
    true_w: float,
    true_b: float,
    noise_std: float,
    seed: int | None = None,
) -> Tuple[List[float], List[int]]:
    """Generate (x, label) pairs: label = 1 if true_w*x + true_b + noise > 0 else 0."""
    if seed is not None:
        random.seed(seed)

    xs = [random.uniform(-10.0, 10.0) for _ in range(n_samples)]
    ys = [
        1 if (true_w * x + true_b + random.gauss(0.0, noise_std)) > 0 else 0
        for x in xs
    ]
    return xs, ys


def sigmoid(z: float) -> float:
    """Return 1 / (1 + e^-z)."""
    # ## START CODE HERE ##
    return (1 / (1 + (math.exp(-z))))
    # ## END CODE HERE ##


def predict(x: float, w: float, b: float) -> float:
    """Return sigmoid(w*x + b): the predicted probability that x is class 1."""
    # ## START CODE HERE ##
    return sigmoid(w * x + b)
    # ## END CODE HERE ##


def compute_loss(xs: List[float], ys: List[int], w: float, b: float) -> float:
    """Binary cross-entropy loss, averaged over the dataset."""
    # ## START CODE HERE ##
    n = len(xs)
    error = 0
    for i in range(n):
        error += (ys[i] * math.log(predict(xs[i], w, b))) + ((1 - ys[i]) * math.log(1 - predict(xs[i], w, b)))

    return -(1 / n) * error
    # ## END CODE HERE ##


def compute_gradients(
    xs: List[float], ys: List[int], w: float, b: float
) -> Tuple[float, float]:
    """Partial derivatives of the BCE loss with respect to w and b."""
    # ## START CODE HERE ##
    n = len(xs)
    error_w = 0
    error_b = 0
    for i in range(n):
        error = predict(xs[i], w, b) - ys[i]
        error_b += error
        error_w += error * xs[i]

    return ((1/n) * error_w, (1/n) * error_b)
    # ## END CODE HERE ##


def save_checkpoint_plot(
    xs: List[float],
    ys: List[int],
    w: float,
    b: float,
    epoch: int,
    output_dir: str,
) -> None:
    """Plot the data, colored by class, against the current predicted-probability curve."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    class0_xs = [x for x, y in zip(xs, ys) if y == 0]
    class1_xs = [x for x, y in zip(xs, ys) if y == 1]

    fig, ax = plt.subplots()
    ax.scatter(class0_xs, [0] * len(class0_xs), s=15, alpha=0.6, label="class 0")
    ax.scatter(class1_xs, [1] * len(class1_xs), s=15, alpha=0.6, label="class 1")

    x_min, x_max = min(xs), max(xs)
    curve_xs = [x_min + i * (x_max - x_min) / 100 for i in range(101)]
    curve_ys = [predict(x, w, b) for x in curve_xs]
    ax.plot(curve_xs, curve_ys, color="crimson", label="predicted P(y=1)")

    ax.set_title(f"epoch {epoch}")
    ax.legend()

    fig.savefig(os.path.join(output_dir, f"epoch_{epoch:03d}.png"))
    plt.close(fig)


def train(
    xs: List[float],
    ys: List[int],
    learning_rate: float,
    epochs: int,
    checkpoint_interval: int,
    output_dir: str,
) -> Tuple[float, float]:
    """Fit w and b by batch gradient descent, logging and plotting at checkpoints."""
    w, b = 0.0, 0.0

    for epoch in range(1, epochs + 1):
        grad_w, grad_b = compute_gradients(xs, ys, w, b)

        ## START CODE HERE ## (apply the gradient descent update to w and b)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        ## END CODE HERE ##

        if epoch % checkpoint_interval == 0 or epoch == epochs:
            loss = compute_loss(xs, ys, w, b)
            print(f"epoch {epoch:4d} | loss {loss:.4f} | w {w:.4f} | b {b:.4f}")
            save_checkpoint_plot(xs, ys, w, b, epoch, output_dir)

    return w, b


if __name__ == "__main__":
    LEARNING_RATE = 0.1
    EPOCHS = 200
    CHECKPOINT_INTERVAL = 10
    N_SAMPLES = 100
    TRUE_W, TRUE_B = 1.5, -2.0
    NOISE_STD = 3.0
    SEED = 42
    OUTPUT_DIR = "plots"

    xs, ys = generate_synthetic_data(N_SAMPLES, TRUE_W, TRUE_B, NOISE_STD, seed=SEED)
    final_w, final_b = train(
        xs, ys, LEARNING_RATE, EPOCHS, CHECKPOINT_INTERVAL, OUTPUT_DIR
    )

    print(f"true params:    w={TRUE_W}, b={TRUE_B}")
    print(f"learned params: w={final_w:.4f}, b={final_b:.4f}")
