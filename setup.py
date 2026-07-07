from setuptools import setup, find_packages

setup(
    name="micrograd",
    version="0.1.0",
    description="A tiny scalar-valued autograd engine and neural-net library",
    packages=find_packages(exclude=["tests", "examples", "notebooks"]),
    python_requires=">=3.7",
    extras_require={
        "viz": ["graphviz"],
        "dev": ["pytest", "torch"],
    },
)
