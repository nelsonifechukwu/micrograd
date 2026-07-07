# micrograd

A from-scratch implementation of a tiny scalar-valued autograd engine with reverse-mode automatic
differentiation (backpropagation) in the spirit of Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd). On top of this engine sits a miniature PyTorch-like `nn` API (`Neuron`, `Layer`, `MLP`).

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
