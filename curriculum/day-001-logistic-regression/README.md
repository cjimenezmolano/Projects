# Day 1 — Logistic Regression From Scratch

## Objective

Implement binary logistic regression — a linear model passed through a
sigmoid, trained with gradient descent on the binary cross-entropy loss —
using nothing but pure Python, the same way you did linear regression, but
for classification instead of predicting a continuous value.

## Task

Open `starter.py` and fill in every block marked:

```python
# ## START CODE HERE ##
...
# ## END CODE HERE ##
```

You need to implement:

1. `sigmoid(z)` — squash a real number into (0, 1).
2. `predict(x, w, b)` — the model's predicted probability that a point
   belongs to class 1.
3. `compute_loss(xs, ys, w, b)` — binary cross-entropy loss.
4. `compute_gradients(xs, ys, w, b)` — the loss's partial derivatives with
   respect to `w` and `b`.
5. The gradient descent parameter update step inside `train()`.

Everything else (`generate_synthetic_data`, `save_checkpoint_plot`, the
training loop's control flow, `main`) is provided so you can focus on the
math above.

## Acceptance criteria

- Running `python starter.py` (after filling in the blanks) completes 200
  epochs without errors.
- The printed loss decreases over training and settles well below its
  starting value.
- `plots/epoch_010.png` ... `plots/epoch_200.png` show the fitted sigmoid
  curve tightening around the true decision boundary as training progresses.
- No numpy, sklearn, or scipy anywhere in the file. matplotlib is used only
  inside `save_checkpoint_plot`.

## Constraints

- Pure Python only for every numeric computation (loops, lists, the `math`
  module). No numpy/sklearn/scipy.
- matplotlib is allowed only for plotting.

## Why this matters

Logistic regression is the building block for almost everything that
follows — the same "linear combination → nonlinearity → loss → gradient"
pattern reappears, scaled up, in every neural network later in this roadmap.
Getting the loss and gradient right by hand here is what makes manual
backprop (Day 9) click instead of feeling like magic.

## Hints

- `sigmoid(z)` is `1 / (1 + e^-z)`. `math.exp(x)` gives you `e^x`.
- Binary cross-entropy blows up (`log(0)`) if a prediction ever lands exactly
  on 0 or 1. Clip predictions into a safe range like `[1e-12, 1 - 1e-12]`
  before taking the log.
- For the gradients: differentiate the BCE loss with respect to `w` and `b`
  using the chain rule through `predict()`. Work it out for a single data
  point first, then average over the dataset — the sigmoid's derivative
  cancels neatly with the log-loss derivative, so the final expressions end
  up simpler than they look mid-derivation.
- If loss goes to `nan` during training, it's almost always the log(0)
  clipping issue above, not your gradient math.
