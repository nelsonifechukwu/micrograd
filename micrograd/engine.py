"""Scalar-valued autograd engine.

A tiny reverse-mode automatic differentiation engine that operates over
individual scalars. Each ``Value`` is a node in a dynamically-built
computational graph; calling ``.backward()`` on the output node populates
``.grad`` on every node that fed into it.
"""

import math


class Value:
    """A single scalar value and its gradient in the computational graph."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        # --- autograd internals ---
        # backprop closure for this node; overwritten by each operation
        self._backward = lambda: None
        # the nodes that produced this one (its operands)
        self._prev = set(_children)
        # the op that produced this node, for debugging / visualization
        self._op = _op
        self.label = label

    # ------------------------------------------------------------------ #
    # core operations
    # ------------------------------------------------------------------ #
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # activations
    # ------------------------------------------------------------------ #
    def tanh(self):
        e = math.exp(2 * self.data)
        t = (e - 1) / (e + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # backpropagation
    # ------------------------------------------------------------------ #
    def backward(self):
        # Build a topological ordering of the graph via post-order DFS, so that
        # every node appears only *after* all of its children. Processing the
        # reversed list guarantees a node's gradient is fully accumulated by its
        # parents before it propagates further. The ``visited`` set ensures each
        # node is processed exactly once (shared nodes would otherwise be
        # double-counted).
        topo, visited = [], set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------ #
    # operator sugar (reflected + derived ops)
    # ------------------------------------------------------------------ #
    def __neg__(self):
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return -self + other

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
