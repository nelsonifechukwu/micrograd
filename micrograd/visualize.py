"""Graphviz visualization of a computational graph.

``draw_dot(root)`` returns a ``graphviz.Digraph`` you can render in a notebook
or save to disk. Requires the ``graphviz`` Python package and the Graphviz
system binaries.
"""

from graphviz import Digraph


def trace(root):
    """Return the sets of (nodes, edges) reachable from ``root``."""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, format="svg", rankdir="LR"):
    """Build a Graphviz diagram of the graph that produced ``root``.

    Args:
        root: the output ``Value`` to trace back from.
        format: output format, e.g. ``"svg"`` or ``"png"``.
        rankdir: ``"LR"`` (left-to-right) or ``"TB"`` (top-to-bottom).
    """
    assert rankdir in ("LR", "TB")
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

    for n in nodes:
        # each Value is a record showing its label, data and gradient
        dot.node(
            name=str(id(n)),
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n._op:
            # add an op node and connect it to its result
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
