# micrograd

A tiny scalar-valued autograd engine and a small neural-network library built
on top of it — a from-scratch implementation of reverse-mode automatic
differentiation (backpropagation), in the spirit of Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd).

Every number is a `Value` node in a dynamically-built computational graph.
Calling `.backward()` on an output walks the graph in reverse topological order
and fills in `.grad` on every node that contributed to it. On top of the engine
sits a miniature PyTorch-like `nn` API (`Neuron`, `Layer`, `MLP`).

## Layout

```
micrograd/
├── micrograd/            # the package
│   ├── engine.py         # Value: the scalar autograd engine
│   ├── nn.py             # Module, Neuron, Layer, MLP
│   └── visualize.py      # draw_dot(): Graphviz view of a graph
├── tests/                # pytest suite, checked against PyTorch
│   ├── test_engine.py
│   └── test_nn.py
├── examples/
│   └── train_mlp.py      # trains an MLP on a toy dataset
├── notebooks/
│   └── micrograd.ipynb   # the original derivation notebook
├── requirements.txt
└── setup.py
```

## Install

```bash
pip install -e .
# for the tests / visualizer:
pip install -r requirements.txt
```

## Quickstart

```python
from micrograd import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b + b ** 2
c.backward()

print(a.grad, b.grad)   # d c / d a, d c / d b
```

Training a network:

```python
from micrograd import MLP

model = MLP(3, [4, 4, 1])          # 3 inputs -> 4 -> 4 -> 1
preds = [model(x) for x in xs]
loss = sum((p - y) ** 2 for y, p in zip(ys, preds))

model.zero_grad()                  # backward() accumulates, so zero first
loss.backward()
for p in model.parameters():
    p.data -= 0.05 * p.grad        # gradient-descent step
```

See [`examples/train_mlp.py`](examples/train_mlp.py) for a full loop.

## Testing

The tests build each expression twice — once with `micrograd`, once with
`torch` — and assert the forward values and gradients match to `1e-6`.

```bash
python -m pytest tests/ -v
```

## A note on the backward pass

`Value.backward()` uses a **post-order DFS with a `visited` set** to build the
topological order, then processes it in reverse. Both details matter:

- the `visited` set ensures a node reused on multiple paths (e.g. an
  activation feeding several downstream neurons) is processed **once** —
  otherwise its gradient is double-counted;
- post-order + reverse guarantees a node runs only **after every parent** has
  contributed to its `.grad`, so gradients are never propagated half-formed.

A naive pre-order traversal without a visited set passes on simple tree-shaped
graphs but silently corrupts gradients the moment a value is reused — which is
every real network. `tests/test_engine.py::test_shared_node_gradient` guards
against exactly this.
