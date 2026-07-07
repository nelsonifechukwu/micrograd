"""Verify the autograd engine's forward values and gradients against PyTorch.

For each expression we build the identical computation with ``micrograd``
``Value`` objects and with autograd-enabled ``torch`` tensors, then assert both
the output and the leaf gradients agree to within a tight tolerance.
"""

import torch

from micrograd.engine import Value

TOL = 1e-6


def test_sanity_check():
    # micrograd
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    # PyTorch
    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass agrees
    assert abs(ymg.data - ypt.data.item()) < TOL
    # backward pass agrees
    assert abs(xmg.grad - xpt.grad.item()) < TOL


def test_more_ops():
    # micrograd -- exercises +, -, *, /, **, unary neg, tanh and reflected ops
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).tanh()
    d = d + 3 * d + (b - a).tanh()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    # PyTorch
    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).tanh()
    d = d + 3 * d + (b - a).tanh()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    assert abs(gmg.data - gpt.data.item()) < TOL   # forward
    assert abs(amg.grad - apt.grad.item()) < TOL   # da
    assert abs(bmg.grad - bpt.grad.item()) < TOL   # db


def test_exp_and_div():
    # exercises exp(), truediv and the reflected rtruediv (3.0 / x). Kept
    # well-conditioned (no near-zero denominators) so float rounding differences
    # stay well under tolerance.
    x = Value(0.5)
    y = (x.exp() + 2.0) / (x.exp() + 1.0) + 3.0 / x
    y.backward()
    xmg, ymg = x, y

    x = torch.tensor([0.5], dtype=torch.double, requires_grad=True)
    y = (x.exp() + 2.0) / (x.exp() + 1.0) + 3.0 / x
    y.backward()
    xpt, ypt = x, y

    assert abs(ymg.data - ypt.data.item()) < TOL
    assert abs(xmg.grad - xpt.grad.item()) < TOL


def test_shared_node_gradient():
    # A node reused on multiple paths must have its contributions summed, not
    # double-counted -- this is the case a naive (no visited-set) topo sort gets
    # wrong. torch is the ground truth.
    x = Value(3.0)
    h = x * 2
    y = (h + 10) + (h + 20)   # h feeds two parents
    y.backward()
    xmg = x

    x = torch.Tensor([3.0]).double()
    x.requires_grad = True
    h = x * 2
    y = (h + 10) + (h + 20)
    y.backward()
    xpt = x

    assert abs(xmg.grad - xpt.grad.item()) < TOL
