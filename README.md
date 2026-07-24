# entropy-invariant

A Python package implementing an improved nearest neighbor method for estimating differential entropy for continuous variables. This is a port of the Julia [EntropyInvariant](https://github.com/Entropy-Invariant/EntropyInvariant.jl) package -- both packages are kept in sync, including the KSG/Frenzel-Pompe estimator described below.

## Key Features

- **Invariant under change of variables**: Scale and translation invariant entropy, mutual information, and conditional mutual information
- **Always positive**: Solves Edwin Thompson Jaynes' limiting density of discrete points problem
- **Bias-cancelling MI/CMI by default**: `mutual_information` and `conditional_mutual_information` (and everything built on them) default to `method="inv_ksg"` -- invariant-measure normalization combined with a KSG/Frenzel-Pompe shared-radius estimator, which is both more accurate and more outlier-robust than the naive plug-in differencing (`method="inv"`, also available)
- **Multiple methods**: `entropy` supports invariant (default), k-NN, and histogram methods

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

# Mutual information (method="inv_ksg" by default)
mi = mutual_information(x, y)
print(f"Mutual Information: {mi}")

# Different entropy methods
h_knn = entropy(x, method="knn")
h_hist = entropy(x, method="histogram", nbins=20)

# The plug-in ("inv") estimator is still available directly for MI/CMI too,
# e.g. when you need the individual entropy terms rather than just their difference
mi_plugin = mutual_information(x, y, method="inv")
```

See `examples/tutorial_getting_started.ipynb` for a hands-on walkthrough, including why
`inv_ksg` is the default and where it has a measurable edge over `std`-based
normalization (a small number of extreme outliers). `examples/` also has deep-dive
comparison notebooks against `ennemi` and `sklearn`.

## Available Functions

### Core Entropy
- `entropy(X, method="inv", k=3, base=e, ...)` - Unified entropy interface
- `entropy_inv(X, ...)` - Invariant method (default)
- `entropy_knn(X, ...)` - k-NN method
- `entropy_hist(X, ...)` - Histogram method

### Information Theory
All default to `method="inv_ksg"` (invariant-measure normalization + KSG/Frenzel-Pompe
shared-radius estimator); `method="inv"` (plug-in, entropy differencing), `"knn"`, and
`"histogram"` remain available.
- `conditional_entropy(X, Y, ...)` - H(Y|X)
- `mutual_information(X, Y, ...)` - I(X;Y)
- `conditional_mutual_information(X, Y, Z, ...)` - I(X;Y|Z)
- `normalized_mutual_information(X, Y, ...)` - NMI
- `interaction_information(X, Y, Z, ...)` - Three-way interaction
- `information_quality_ratio(X, Y, ...)` - IQR
- `mutual_information_ksg(X, Y, ...)` - The KSG estimator directly
- `conditional_mutual_information_ksg(X, Y, Z, ...)` - The Frenzel-Pompe estimator directly

### Partial Information Decomposition
- `redundancy(X, Y, Z, ...)` - Shared information
- `unique(X, Y, Z, ...)` - Unique information
- `synergy(X, Y, Z, ...)` - Synergistic information

### Optimized Matrix Functions
- `MI(X, method="inv_ksg", n_jobs=1, ...)` - Pairwise mutual information matrix
- `CMI(X, Z, method="inv_ksg", n_jobs=1, ...)` - Pairwise conditional MI matrix

For `method="inv_ksg"` (the default), each dimension's own k-NN tree is built
once and reused across all pairs, and `n_jobs` parallelizes the remaining
O(m²) per-pair work across processes (`n_jobs=-1` for all CPUs, scikit-learn
convention) -- useful for large matrices (hundreds of dimensions), e.g. mass-
spec or sensor-array data.

## Authors

- Felix Truong
- Alexandre Giuliani

## License

MIT

## Citation
If you use this code or data, please cite:

[An Invariant Measure for Differential Entropy: From Kullback–Leibler Divergence to Scale-Invariant Information Theory](https://www.mdpi.com/1099-4300/28/3/301) DOI: 10.3390/e28030301
