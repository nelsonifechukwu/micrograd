"""Tests for the neural-net library: gradient parity with PyTorch, and training.

``test_neuron_matches_torch`` builds a single neuron with fixed weights and
checks every parameter/input gradient against ``torch``. ``test_mlp_trains``
runs a short training loop and asserts the loss actually decreases.
"""

import random

import torch

from micrograd.engine import Value
from micrograd.nn import Neuron, MLP

TOL = 1e-6


def test_neuron_matches_torch():
    random.seed(0)
    neuron = Neuron(3)
    x = [Value(1.0), Value(-2.0), Value(3.0)]

    out = neuron(x)
    out.backward()

    # rebuild the exact same neuron in torch: tanh(w . x + b)
    w = torch.tensor([wi.data for wi in neuron.w], dtype=torch.double, requires_grad=True)
    b = torch.tensor(neuron.b.data, dtype=torch.double, requires_grad=True)
    xt = torch.tensor([xi.data for xi in x], dtype=torch.double, requires_grad=True)
    out_t = torch.tanh(w @ xt + b)
    out_t.backward()

    # forward value
    assert abs(out.data - out_t.item()) < TOL
    # weight gradients
    for wi, gi in zip(neuron.w, w.grad):
        assert abs(wi.grad - gi.item()) < TOL
    # bias gradient
    assert abs(neuron.b.grad - b.grad.item()) < TOL
    # input gradients
    for xi, gi in zip(x, xt.grad):
        assert abs(xi.grad - gi.item()) < TOL


def test_mlp_trains():
    random.seed(42)
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]
    model = MLP(3, [4, 4, 1])

    def total_loss():
        preds = [model(x) for x in xs]
        return sum((p - y) ** 2 for y, p in zip(ys, preds))

    first = total_loss().data
    for _ in range(100):
        model.zero_grad()
        loss = total_loss()
        loss.backward()
        for p in model.parameters():
            p.data -= 0.05 * p.grad
    last = total_loss().data

    assert last < first          # learning happened
    assert last < 0.1            # and it actually fit this tiny dataset
