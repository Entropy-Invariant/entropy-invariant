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


def _mi_ksg_from_normalized(
    x: NDArray[np.float64], y: NDArray[np.float64], k: int
) -> float:
    """KSG MI in nats, given x, y already invariant-normalized and column-shaped (n, 1)."""
    n = x.shape[0]
    xy = np.column_stack([x, y])

    joint_tree = cKDTree(xy)
    x_tree = cKDTree(x)
    y_tree = cKDTree(y)

    # Shared radius: k-th neighbor distance in the joint (normalized) space.
    eps = joint_tree.query(xy, k=[k + 1], p=np.inf)[0].flatten()

    nx = x_tree.query_ball_point(x, eps - _STRICT_RADIUS_EPS, p=np.inf, return_length=True)
    ny = y_tree.query_ball_point(y, eps - _STRICT_RADIUS_EPS, p=np.inf, return_length=True)

    return float(digamma(n) + digamma(k) - np.mean(digamma(nx) + digamma(ny)))


def _cmi_fp_from_normalized(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    k: int,
) -> float:
    """Frenzel-Pompe CMI in nats, given x, y, z already invariant-normalized, shape (n, 1)."""
    xyz = np.column_stack([x, y, z])
    xz = np.column_stack([x, z])
    yz = np.column_stack([y, z])

    full_tree = cKDTree(xyz)
    xz_tree = cKDTree(xz)
    yz_tree = cKDTree(yz)
    z_tree = cKDTree(z)

    # Shared radius: k-th neighbor distance in the full joint (normalized) space.
    eps = full_tree.query(xyz, k=[k + 1], p=np.inf)[0].flatten()

    nxz = xz_tree.query_ball_point(xz, eps - _STRICT_RADIUS_EPS, p=np.inf, return_length=True)
    nyz = yz_tree.query_ball_point(yz, eps - _STRICT_RADIUS_EPS, p=np.inf, return_length=True)
    nz = z_tree.query_ball_point(z, eps - _STRICT_RADIUS_EPS, p=np.inf, return_length=True)

    return float(digamma(k) - np.mean(digamma(nxz) + digamma(nyz) - digamma(nz)))


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
    y = _invariant_normalize_1d(mat_y[0, :]).reshape(-1, 1)

    if verbose:
        print(f"Number of points: {x.shape[0]}")
        print(f"k: {k}, base: {base}")

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
