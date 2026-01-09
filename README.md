# entropy-invariant

A Python package implementing an improved nearest neighbor method for estimating differential entropy for continuous variables. This is a port of the Julia [EntropyInvariant](https://github.com/truongfelix/EntropyInvariant-private) package.

## Key Features

- **Invariant under change of variables**: Scale and translation invariant entropy estimation
- **Always positive**: Solves Edwin Thompson Jaynes' limiting density of discrete points problem
- **Multiple methods**: Supports invariant (default), k-NN, and histogram methods

## Installation

```bash
pip install entropy-invariant
```

Or install from source:

```bash
pip install -e .
```

## Usage

```python
import numpy as np
from entropy_invariant import entropy, mutual_information

# Generate random data
n = 1000
x = np.random.rand(n)
y = 2 * x + np.random.rand(n)

# Compute entropy (invariant method, default)
h = entropy(x)
print(f"Entropy: {h}")

# Entropy is invariant under scaling and translation
h_scaled = entropy(1e5 * x - 123.456)
print(f"Entropy (scaled): {h_scaled}")  # Same value!

# Mutual information
mi = mutual_information(x, y)
print(f"Mutual Information: {mi}")

# Different methods
h_knn = entropy(x, method="knn")
h_hist = entropy(x, method="histogram", nbins=20)
```

## Available Functions

### Core Entropy
- `entropy(X, method="inv", k=3, base=e, ...)` - Unified entropy interface
- `entropy_inv(X, ...)` - Invariant method (default)
- `entropy_knn(X, ...)` - k-NN method
- `entropy_hist(X, ...)` - Histogram method

### Information Theory
- `conditional_entropy(X, Y, ...)` - H(Y|X)
- `mutual_information(X, Y, ...)` - I(X;Y)
- `conditional_mutual_information(X, Y, Z, ...)` - I(X;Y|Z)
- `normalized_mutual_information(X, Y, ...)` - NMI
- `interaction_information(X, Y, Z, ...)` - Three-way interaction

### Partial Information Decomposition
- `redundancy(X, Y, Z, ...)` - Shared information
- `unique(X, Y, Z, ...)` - Unique information
- `synergy(X, Y, Z, ...)` - Synergistic information

### Optimized Matrix Functions
- `MI(X, ...)` - Pairwise mutual information matrix
- `CMI(X, Z, ...)` - Pairwise conditional MI matrix

## Authors

- Felix Truong
- Alexandre Giuliani

## License

MIT
