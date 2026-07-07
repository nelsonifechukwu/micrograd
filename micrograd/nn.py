"""A minimal neural-network library built on the autograd engine.

Mirrors the PyTorch API in miniature: ``Module`` is the base class,
``Neuron`` / ``Layer`` / ``MLP`` compose into a multi-layer perceptron whose
parameters are plain ``Value`` scalars trainable by the engine's autograd.
"""

import random

from .engine import Value


class Module:
    """Base class: anything with trainable ``parameters``."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    """A single neuron: tanh(w · x + b)."""

    def __init__(self, num_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        assert len(x) == len(self.w), (
            f"input length {len(x)} must equal number of weights {len(self.w)}"
        )
        # start the sum at the bias so it becomes w·x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"TanhNeuron({len(self.w)})"


class Layer(Module):
    """A fully-connected layer of ``num_outputs`` neurons."""

    def __init__(self, num_inputs, num_outputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        # unwrap a single-neuron layer so the network's output is a scalar
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """A multi-layer perceptron.

    Args:
        num_inputs: size of the input vector.
        layer_sizes: number of neurons in each successive layer.
    """

    def __init__(self, num_inputs, layer_sizes):
        sizes = [num_inputs] + list(layer_sizes)
        self.layers = [
            Layer(sizes[i], sizes[i + 1]) for i in range(len(layer_sizes))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
