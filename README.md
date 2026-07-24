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

## N-source Partial Information Decomposition

`redundancy` / `unique` / `synergy` decompose `I({X,Y}; Z)` into four terms and are fixed
at two sources. `pid_lattice` gives the general N-source decomposition of Williams & Beer
(2010): `I({X_1,...,X_N}; Z)` splits into a lattice of partial-information atoms — 4 nodes
for 2 sources, 18 for 3, 166 for 4.

```python
import numpy as np
from entropy_invariant import pid_lattice

# continuous data, three sources
atoms = pid_lattice([x1, x2, x3], z, names=["a", "b", "c"])
atoms["{a}{b}{c}"]   # redundancy shared by all three individually
atoms["{ab}{c}"]     # shared by the pair {a,b} and by c alone
atoms["{abc}"]       # the top synergy atom

# discrete data, from an explicit joint distribution (target in the last axis).
# Two-input AND, reproducing Williams & Beer's published values in bits:
pmf = np.zeros((2, 2, 2))
pmf[0, 0, 0] = pmf[0, 1, 0] = pmf[1, 0, 0] = 0.25
pmf[1, 1, 1] = 0.25
a = pid_lattice(pmf, names=["X", "Y"])
a["{X}{Y}"]   # 0.3113  (redundancy)
a["{XY}"]     # 0.5     (synergy)
```

Three redundancy measures are available:

- **`"mmi"`** (default, continuous data) — `min` over coalitions of `I(X_A; Z)`. Works
  with any of the package's estimators and reduces exactly to `redundancy` / `unique` at
  two sources.
- **`"imin"`** (discrete data) — Williams & Beer's original measure, using specific
  information per target outcome. Guarantees non-negative atoms, which `"mmi"` does not
  for three or more sources. Over-credits redundancy: on two-bit COPY it reports two
  *independent* bits as fully redundant.
- **`"iccs"`** (discrete data) — Ince's (2017) pointwise common change in surprisal. Fixes
  the COPY case (`R = 0`, `U_X = U_Y = 1`, `Syn = 0`) and, unlike `"mmi"`, lets both unique
  atoms be positive at once. Atoms may be negative; Ince argues these are meaningful.

```python
# the same distribution under two measures -- Z = (X, Y), independent bits
pmf = np.zeros((2, 2, 4))
pmf[0, 0, 0] = pmf[0, 1, 1] = pmf[1, 0, 2] = pmf[1, 1, 3] = 0.25
pid_lattice(pmf, measure="imin", names=["X", "Y"])["{X}{Y}"]   # 1.0  -- over-credited
pid_lattice(pmf, measure="iccs", names=["X", "Y"])["{X}{Y}"]   # 0.0  -- correct
```

### Two things worth knowing before interpreting the output

**Estimated coalition MIs need repairing first.** True mutual information satisfies
`I(X_A; Z) <= I(X_B; Z)` whenever `A` is a subset of `B`, but finite-sample kNN estimates
measurably do not — adding a source that is nearly redundant with the existing ones
*lowers* the estimate. Both the `min` in `"mmi"` and the Möbius inversion assume that
ordering, so unrepaired estimates produce negative unique and synergy atoms, which are
impossible for true information. `pid_lattice` therefore applies `isotonic_repair` by
default (`repair="isotonic"`, or `"majorant"` / `"none"`). The repair deliberately does
**not** clamp to zero: an estimate of `-0.02` where the truth is near zero is ordinary
symmetric noise, and truncating it would bias every low-signal region upward.

**`"mmi"`'s unique atoms are winner-take-all.** At two sources they are
`max(0, I_X - I_Y)` and `max(0, I_Y - I_X)`, so exactly one is nonzero by construction.
That is appropriate for asking *how information is divided*, but it cannot express
"partly one source, partly the other" and will not move gradually as an underlying
relationship shifts. For *how much each source contributes*, use
`conditional_mutual_information` or `"iccs"`.

Note also that `sum(atoms) == I({all sources}; Z)` holds identically for any redundancy
measure — it is a property of the Möbius inversion, not a check that the decomposition is
right. Correctness is established in the test suite against published atom values on
discrete toy distributions, the Williams & Beer non-negativity theorem, and exact
agreement with `redundancy` / `unique` at two sources.


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
