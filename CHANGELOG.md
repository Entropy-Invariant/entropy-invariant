# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [2.2.0] - 2026-07-25

### Added
- `pid_lattice`: N-source Partial Information Decomposition over the Williams & Beer
  (2010) redundancy lattice, generalizing the two-source `redundancy`/`unique`/`synergy`
  triple. 4 atoms for 2 sources, 18 for 3, 166 for 4. Three redundancy measures:
  `"mmi"` (min over coalition mutual informations, estimable from continuous data with any
  of the package's estimators, and exactly reducing to `redundancy`/`unique` at two
  sources), `"imin"` (Williams & Beer's original specific-information measure, with
  guaranteed non-negative atoms) and `"iccs"` (Ince 2017, pointwise common change in
  surprisal).
- `iccs_redundancy`: Ince's `I_ccs`. Keeps only the pointwise co-information whose sign
  every coalition agrees on. Corrects the two-bit COPY case that `"imin"` gets wrong
  (`R = 0`, `U_X = U_Y = 1`, `Syn = 0` rather than `R = 1`), and unlike `"mmi"` allows both
  unique atoms to be positive simultaneously instead of exactly one by construction.
  Defined for any number of sources, unlike BROJA and other optimisation-based measures.
  Not guaranteed monotone on the lattice, so atoms may be negative.
- `redundancy_lattice`, `lattice_labels`, `moebius_atoms`: the lattice structure,
  human-readable atom labels, and the Möbius inversion, exposed separately so a custom
  redundancy measure can be plugged in.
- `coalition_mutual_information`: `I(X_A; Z)` for every coalition of sources. Uses the
  dimension-agnostic shared-radius KSG estimator for `method="inv_ksg"`; rejects
  coalitions beyond 3 total dimensions for the plug-in methods with an explicit message
  rather than an opaque failure from deeper in the stack.
- `isotonic_repair`: projects estimated coalition mutual informations onto the monotone
  cone `I(X_A;Z) <= I(X_B;Z)` for `A` a subset of `B`. Finite-sample kNN estimates violate
  this — adding a nearly-redundant source lowers the estimate — which otherwise
  manufactures negative unique and synergy atoms that are impossible for true information.
  Deliberately does not clamp to zero, since non-negativity constrains the level of a
  single estimate rather than the consistency between two, and clamping would bias
  low-signal regions upward.
- `specific_information`: Williams & Beer's per-target-outcome specific information.
- 29 tests covering the above, with values kept identical to the Julia package's suite so
  the two ports stay in sync.

## [2.1.0] - 2026-07-24

### Added
- `MI()` / `CMI()` (`method="inv_ksg"`) now build each dimension's own k-NN
  tree once and reuse it across all pairs, instead of rebuilding it (and,
  for `CMI()`, the Z-only tree) redundantly for every pair -- a pure speed
  improvement, no change in output values.
- `n_jobs` parameter on `MI()` / `CMI()` (scikit-learn convention: `1`
  default = sequential, `-1` = all CPUs): distributes the remaining O(m²)
  per-pair work across `multiprocessing.Pool`. Verified to produce
  bit-identical results to `n_jobs=1`; ~8x speedup measured on a 100-bin /
  5050-pair real CMI matrix.

## [2.0.0] - 2026-07-24

This is the first tagged release of `entropy-invariant`. It bundles a new
estimator, a breaking default-behavior change, and three correctness fixes
found while hardening the test suite after that change — bumped as a major
version because the default method now returns different numeric values than
before, even though no function signatures changed.

### Added
- `mutual_information_ksg` / `conditional_mutual_information_ksg`: a KSG
  (Kraskov, Stögbauer & Grassberger 2004) / Frenzel-Pompe (2007) shared-radius
  estimator, applied after invariant-measure normalization. Cancels the
  leading-order k-NN bias that the plug-in formula (`H(X)+H(Y)-H(X,Y)`, etc.)
  does not, most visibly on outlier-contaminated or near-degenerate data.
- Example notebooks: `examples/tutorial_getting_started.ipynb` and an expanded
  `examples/plot_cmi_comparison.ipynb`.
- `method="inv_ksg"` is now the option throughout `mutual_information`,
  `conditional_mutual_information`, `conditional_entropy`,
  `normalized_mutual_information`, `interaction_information`,
  `information_quality_ratio`, `redundancy`, `unique`, `synergy`, and the
  matrix fast-paths `MI()` / `CMI()`.

### Changed
- **Breaking**: the default `method` for all MI/CMI-derived functions changed
  from `"inv"` (plug-in) to `"inv_ksg"`. `method="inv"` is still available
  and unchanged. Anything relying on the default now gets different (more
  bias-corrected) numeric output.

### Fixed
- `MI()` / `CMI()` (the matrix fast-path) computed the invariant measure
  inline without filtering zero values, unlike the documented
  `compute_invariant_measure()` helper. On sparse data (mostly zeros, e.g.
  real spectral/sensor data) this produced a zero median nearest-neighbor
  distance, causing division by zero and NaN/inf propagation instead of
  matching the scalar `mutual_information()` / `conditional_mutual_information()`
  functions.
- `compute_knn_entropy_nats` used the *post-filtering* sample count (after
  dropping points with a degenerate zero k-th-neighbor distance) instead of
  the true total number of points for the `digamma(n)` term. These agree for
  continuous data with no ties, but diverge sharply on data with duplicates —
  up to several nats of error on realistic sparse test data.
- The KSG/Frenzel-Pompe shared-radius estimator silently produced `NaN`
  (`digamma(0) = -inf`) when ≥k+1 points coincide exactly in the joint space
  it searches over. Now raises a clear `ValueError`. `MI()`'s diagonal and
  `mutual_information_ksg(x, x)` (i.e. `I(X;X) = H(X)`) no longer run the
  shared-radius trick at all, since pairing a variable with itself makes this
  case trivial to hit for any column with duplicate values.
- `compute_invariant_measure()` now raises a clear `ValueError` when the
  invariant measure is degenerate (too many duplicate non-zero values),
  instead of silently returning a value that produces a cryptic downstream
  crash.

[2.2.0]: https://github.com/Entropy-Invariant/entropy-invariant/releases/tag/v2.2.0
[2.1.0]: https://github.com/Entropy-Invariant/entropy-invariant/releases/tag/v2.1.0
[2.0.0]: https://github.com/Entropy-Invariant/entropy-invariant/releases/tag/v2.0.0
