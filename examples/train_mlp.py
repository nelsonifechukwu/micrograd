"""Train a small MLP on a toy binary dataset using micrograd.

Run from the repo root:

    python examples/train_mlp.py
"""

import os
import random
import sys

# Make the local micrograd package importable when this file is run directly as
# `python examples/train_mlp.py`, taking precedence over any pip-installed
# package of the same name.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micrograd.nn import MLP


def main():
    random.seed(1)

    # 4 examples, 3 features each; targets are +1 / -1
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    model = MLP(3, [4, 4, 1])
    print(model)
    print(f"{len(model.parameters())} parameters\n")

    learning_rate = 0.05
    epochs = 100

    for epoch in range(epochs):
        # forward: mean-squared error over the dataset
        preds = [model(x) for x in xs]
        loss = sum((pred - y) ** 2 for y, pred in zip(ys, preds))

        # backward: always zero grads first (backward accumulates with +=)
        model.zero_grad()
        loss.backward()

        # update: simple gradient descent
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  loss {loss.data:.6f}")

    print("\npredictions vs targets:")
    for x, y in zip(xs, ys):
        print(f"  pred {model(x).data:+.3f}   target {y:+.0f}")


if __name__ == "__main__":
    main()
