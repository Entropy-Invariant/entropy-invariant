"""KSG / Frenzel-Pompe estimators, computed after invariant-measure normalization.

The plug-in estimators in `mutual_information.py` / `advanced.py` build MI and CMI by
differencing independent Kozachenko-Leonenko entropy estimates (H(X) + H(Y) - H(X,Y),
etc). Each term picks its own k-NN radius independently, so their finite-sample biases
don't cancel -- this shows up as systematic drift on outlier-contaminated or near-
degenerate (low-rank manifold) data.

KSG (Kraskov, Stogbauer & Grassberger, 2004) and its conditional extension, Frenzel &
Pompe (2007), fix this for the *unnormalized* case by finding the k-th neighbor radius
once in the full joint space (Chebyshev metric) and reusing that same radius to count
neighbors in each marginal/subspace -- the shared radius makes the leading-order bias
terms cancel algebraically.

This module combines both ideas: each variable is first normalized by its own invariant
measure (median nearest-neighbor distance, `compute_invariant_measure`) -- giving
affine scale-invariance and outlier-robustness the same way `entropy_inv` does -- and
then KSG/Frenzel-Pompe's shared-radius neighbor counting is applied on the normalized
data, giving the bias cancellation that the plug-in formulas lack.

The `_from_normalized` functions are the reusable core (used directly by the `MI`/`CMI`
matrix functions in `optimized.py`, which normalize each column once up front rather
than repeating it per pair).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.special import digamma

from entropy_invariant._constants import E
from entropy_invariant.helpers.computation import (
    compute_invariant_measure,
    compute_knn_distances,
    extract_nonzero_log_distances,
    compute_knn_entropy_nats,
    convert_to_base,
)
from entropy_invariant.helpers.data import (
    ensure_2d,
    ensure_columns_are_points,
    get_shape,
    validate_dimensions_equal_one,
    validate_same_num_points,
)

# cKDTree's query_ball_point uses a non-strict (<=) radius comparison, but the
# KSG/Frenzel-Pompe algorithm requires strict (<) neighbor counts. Subtracting a
# small epsilon from the radius corrects this without materially affecting real
# (roughly unit-magnitude, post-normalization) distances. Same fix used by `ennemi`
# (see https://github.com/polsys/ennemi/issues/76).
_STRICT_RADIUS_EPS = 1e-12


def _invariant_normalize_1d(x: NDArray[np.float64]) -> NDArray[np.float64]:
    measure = compute_invariant_measure(x)
    return x / measure


def _entropy_nats_from_normalized(col: NDArray[np.float64], k: int, n: int) -> float:
    """
    H(Xi) in nats from an already invariant-normalized column, shape (1, n).

    Used for I(Xi; Xi) = H(Xi): pairing a variable with itself is never run
    through the shared-radius KSG trick above, since any duplicate value in
    Xi (e.g. repeated zeros in sparse data) then collides with itself in the
    joint (Xi, Xi) space, making the shared radius degenerate far more
    easily than a genuine two-variable pair would. The plain k-NN entropy
    estimate here tolerates duplicates by dropping degenerate (zero-distance)
    points from the log-distance average -- the same behavior as
    method="inv" -- instead of hard-failing.
    """
    knn_result = compute_knn_distances(col, k)
    log_dists = extract_nonzero_log_distances(knn_result.kth_distances)
    return compute_knn_entropy_nats(log_dists, 1, k, n)


def _check_no_degenerate_counts(*counts_by_name: tuple[str, NDArray]) -> None:
    """
    Raise a clear error if any marginal/subspace neighbor count is 0.

    A count of 0 means the shared KSG radius was degenerate (exactly 0) at
    that point -- i.e. at least k+1 points are exact duplicates in the joint
    space, most often because several dimensions are simultaneously sparse
    (e.g. many rows are all zero). digamma(0) is -inf, so this would
    otherwise propagate silently into NaN.
    """
    for name, counts in counts_by_name:
        n_degenerate = int(np.sum(counts == 0))
        if n_degenerate > 0:
            raise ValueError(
                f"Shared KSG radius is degenerate for {n_degenerate} point(s) "
                f"in the '{name}' subspace: at least k+1 points coincide "
                f"exactly in the joint space (e.g. multiple all-zero/duplicate "
                f"rows). Cannot compute a finite entropy estimate for these "
                f"points -- consider deduplicating, adding jitter, or "
                f"excluding the offending dimension(s)."
            )


def _mi_ksg_pair(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x_tree: cKDTree,
    y_tree: cKDTree,
    k: int,
) -> float:
    """
    KSG MI in nats, given x, y already invariant-normalized and column-shaped
    (n, 1), plus their two PRE-BUILT marginal (1D) cKDTrees.

    Splitting out the marginal trees lets a caller computing many pairs over
    the same set of dimensions (the `MI` matrix function in optimized.py)
    build each dimension's 1D tree once and reuse it across every pair it
    appears in, instead of rebuilding it from scratch for every single pair.
    Only the joint (2D) tree below is genuinely pair-specific.
    """
    n = x.shape[0]
    xy = np.column_stack([x, y])

    joint_tree = cKDTree(xy)

    # Shared radius: k-th neighbor distance in the joint (normalized) space.
    eps = joint_tree.query(xy, k=[k + 1], p=np.inf)[0].flatten()

    nx = x_tree.query_ball_point(x, eps - _STRICT_RADIUS_EPS, p=np.inf, return_length=True)
    ny = y_tree.query_ball_point(y, eps - _STRICT_RADIUS_EPS, p=np.inf, return_length=True)
    _check_no_degenerate_counts(("x", nx), ("y", ny))

    return float(digamma(n) + digamma(k) - np.mean(digamma(nx) + digamma(ny)))


def _mi_ksg_from_normalized(
    x: NDArray[np.float64], y: NDArray[np.float64], k: int
) -> float:
    """KSG MI in nats, given x, y already invariant-normalized and column-shaped (n, 1)."""
    return _mi_ksg_pair(x, y, cKDTree(x), cKDTree(y), k)


def _cmi_fp_pair(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    xz_tree: cKDTree,
    yz_tree: cKDTree,
    z_tree: cKDTree,
    k: int,
) -> float:
    """
    Frenzel-Pompe CMI in nats, given x, y, z already invariant-normalized,
    shape (n, 1), plus PRE-BUILT (X,Z) and (Y,Z) subspace trees and the Z tree.

    Splitting these out lets a caller computing many pairs against the same
    conditioning variable Z (the `CMI` matrix function in optimized.py) build
    each dimension's (Xi, Z) tree once and the single Z tree once, and reuse
    them across every pair -- instead of rebuilding all three (plus Z, which
    never changes) from scratch for every single pair. Only the full joint
    (3D) tree below is genuinely pair-specific.
    """
    xyz = np.column_stack([x, y, z])
    xz = np.column_stack([x, z])
    yz = np.column_stack([y, z])

    full_tree = cKDTree(xyz)

    # Shared radius: k-th neighbor distance in the full joint (normalized) space.
    eps = full_tree.query(xyz, k=[k + 1], p=np.inf)[0].flatten()

    nxz = xz_tree.query_ball_point(xz, eps - _STRICT_RADIUS_EPS, p=np.inf, return_length=True)
    nyz = yz_tree.query_ball_point(yz, eps - _STRICT_RADIUS_EPS, p=np.inf, return_length=True)
    nz = z_tree.query_ball_point(z, eps - _STRICT_RADIUS_EPS, p=np.inf, return_length=True)
    _check_no_degenerate_counts(("x,z", nxz), ("y,z", nyz), ("z", nz))

    return float(digamma(k) - np.mean(digamma(nxz) + digamma(nyz) - digamma(nz)))


def _cmi_fp_from_normalized(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    k: int,
) -> float:
    """Frenzel-Pompe CMI in nats, given x, y, z already invariant-normalized, shape (n, 1)."""
    xz_tree = cKDTree(np.column_stack([x, z]))
    yz_tree = cKDTree(np.column_stack([y, z]))
    z_tree = cKDTree(z)
    return _cmi_fp_pair(x, y, z, xz_tree, yz_tree, z_tree, k)


def mutual_information_ksg(
    X: NDArray,
    Y: NDArray,
    *,
    k: int = 3,
    base: float = E,
    verbose: bool = False,
    dim: int = 1,
) -> float:
    """
    KSG mutual information estimator, applied after invariant-measure normalization.

    Args:
        X: First variable (1-dimensional)
        Y: Second variable (1-dimensional)
        k: Number of neighbors (default 3)
        base: Logarithmic base (default e)
        verbose: Print computation info
        dim: Data layout (1: points as rows, 2: points as columns)

    Returns:
        Mutual information I(X;Y)
    """
    mat_x = ensure_columns_are_points(ensure_2d(X), dim)
    mat_y = ensure_columns_are_points(ensure_2d(Y), dim)

    shapes = [get_shape(mat_x), get_shape(mat_y)]
    validate_same_num_points(shapes)
    validate_dimensions_equal_one(shapes)

    x = _invariant_normalize_1d(mat_x[0, :]).reshape(-1, 1)

    if verbose:
        print(f"Number of points: {x.shape[0]}")
        print(f"k: {k}, base: {base}")

    if np.array_equal(mat_x, mat_y):
        # I(X;X) = H(X) exactly. Skip the shared-radius trick: pairing X with
        # itself makes any duplicate value in X collide with itself in the
        # joint (X, X) space, so it hits the degenerate-radius case far more
        # easily than a genuine two-variable pair -- see
        # _entropy_nats_from_normalized in optimized.py for the matrix-path
        # equivalent of this special case.
        mi_nats = _entropy_nats_from_normalized(x.reshape(1, -1), k, x.shape[0])
        return convert_to_base(mi_nats, base)

    y = _invariant_normalize_1d(mat_y[0, :]).reshape(-1, 1)
    mi_nats = _mi_ksg_from_normalized(x, y, k)
    return convert_to_base(mi_nats, base)


def conditional_mutual_information_ksg(
    X: NDArray,
    Y: NDArray,
    Z: NDArray,
    *,
    k: int = 3,
    base: float = E,
    verbose: bool = False,
    dim: int = 1,
) -> float:
    """
    Frenzel-Pompe conditional mutual information estimator, applied after
    invariant-measure normalization.

    Args:
        X: First variable (1-dimensional)
        Y: Second variable (1-dimensional)
        Z: Conditioning variable (1-dimensional)
        k: Number of neighbors (default 3)
        base: Logarithmic base (default e)
        verbose: Print computation info
        dim: Data layout (1: points as rows, 2: points as columns)

    Returns:
        Conditional mutual information I(X;Y|Z)
    """
    mat_x = ensure_columns_are_points(ensure_2d(X), dim)
    mat_y = ensure_columns_are_points(ensure_2d(Y), dim)
    mat_z = ensure_columns_are_points(ensure_2d(Z), dim)

    shapes = [get_shape(mat_x), get_shape(mat_y), get_shape(mat_z)]
    validate_same_num_points(shapes)
    validate_dimensions_equal_one(shapes)

    x = _invariant_normalize_1d(mat_x[0, :]).reshape(-1, 1)
    y = _invariant_normalize_1d(mat_y[0, :]).reshape(-1, 1)
    z = _invariant_normalize_1d(mat_z[0, :]).reshape(-1, 1)

    if verbose:
        print(f"Number of points: {x.shape[0]}")
        print(f"k: {k}, base: {base}")

    cmi_nats = _cmi_fp_from_normalized(x, y, z, k)
    return convert_to_base(cmi_nats, base)
