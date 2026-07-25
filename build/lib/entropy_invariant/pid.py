"""Partial Information Decomposition (PID) functions."""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from entropy_invariant._constants import E
from entropy_invariant._types import DataShape
from entropy_invariant.entropy import entropy
from entropy_invariant.helpers.computation import compute_invariant_measure, convert_to_base
from entropy_invariant.helpers.data import (
    ensure_columns_are_points,
    ensure_2d,
    get_shape,
    validate_same_num_points,
    validate_dimensions_equal_one,
)
from entropy_invariant.helpers.utility import log_computation_info
from entropy_invariant.ksg import _mi_ksg_from_normalized
from entropy_invariant.mutual_information import mutual_information


def _invariant_normalize_columns(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalize each column (dimension) of `mat`, shape (n, d), by its own
    invariant measure. Generalizes ksg.py's `_invariant_normalize_1d`
    (which only handles a single column) to an arbitrary number of columns,
    so a genuinely multi-dimensional block (e.g. two source variables
    treated jointly) can be fed into the KSG/Frenzel-Pompe shared-radius
    machinery.
    """
    out = np.empty_like(mat, dtype=np.float64)
    for i in range(mat.shape[1]):
        out[:, i] = mat[:, i] / compute_invariant_measure(mat[:, i])
    return out


def _joint_mutual_information(
    mat_1: NDArray, mat_2: NDArray, mat_3: NDArray, *, method: str, nbins: int, k: int, base: float, degenerate: bool
) -> float:
    """
    I({X,Y}; Z), the joint mutual information of the pair {X,Y} treated as
    one combined (multi-dimensional) variable against Z. This is distinct
    from `mutual_information(X, Y)`, which computes I(X;Y) between X and Y
    themselves -- `synergy()` needs the former, not the latter.

    The public `mutual_information()` cannot be called for this directly:
    its first argument is restricted to exactly one dimension in both the
    "inv" and "inv_ksg" code paths. For "inv"/"knn"/"histogram", the joint
    entropy chain rule via `entropy()` (which has no such restriction) is
    used instead. For "inv_ksg", the shared-radius KSG estimator
    (`_mi_ksg_from_normalized`) is called directly on a multi-column
    normalized block -- it is dimension-agnostic internally, it's only the
    public `mutual_information_ksg` wrapper that artificially restricts it
    to 1D.

    `mat_1`, `mat_2`, `mat_3` are canonical (d, n) row-shaped arrays
    (dimensions as rows), matching this module's internal convention.
    """
    joint_xy = np.vstack([mat_1, mat_2])  # (d1+d2, n)
    if method == "inv_ksg":
        x = _invariant_normalize_columns(joint_xy.T)  # (n, d1+d2)
        z = _invariant_normalize_columns(mat_3.T)      # (n, d3)
        return convert_to_base(_mi_ksg_from_normalized(x, z, k), base)
    ent_xy = entropy(joint_xy, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_z = entropy(mat_3, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_xyz = entropy(np.vstack([joint_xy, mat_3]), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    return ent_xy + ent_z - ent_xyz


def redundancy(
    X: NDArray,
    Y: NDArray,
    Z: NDArray,
    *,
    method: str = "inv_ksg",
    nbins: int = 10,
    k: int = 3,
    base: float = E,
    verbose: bool = False,
    degenerate: bool = False,
    dim: int = 1,
) -> float:
    """
    Compute redundancy R(X,Y;Z) = min(I(X;Z), I(Y;Z)).

    The shared information that both X and Y have about Z.

    Args:
        X: First source variable
        Y: Second source variable
        Z: Target variable
        method: Entropy estimation method (default "inv_ksg" -- see
            mutual_information / conditional_mutual_information for what this
            changes; PID quantities are built from those two)
        nbins: Bins for histogram method
        k: Neighbors for k-NN methods
        base: Logarithmic base
        verbose: Print info
        degenerate: Handle degenerate cases
        dim: Data layout

    Returns:
        Redundancy R(X,Y;Z)
    """
    mat_x = ensure_2d(X)
    mat_y = ensure_2d(Y)
    mat_z = ensure_2d(Z)
    mat_x = ensure_columns_are_points(mat_x, dim)
    mat_y = ensure_columns_are_points(mat_y, dim)
    mat_z = ensure_columns_are_points(mat_z, dim)

    shape_x = get_shape(mat_x)
    shape_y = get_shape(mat_y)
    shape_z = get_shape(mat_z)

    validate_same_num_points([shape_x, shape_y, shape_z])
    validate_dimensions_equal_one([shape_x, shape_y, shape_z])

    if verbose:
        total_dims = (
            shape_x.num_dimensions + shape_y.num_dimensions + shape_z.num_dimensions
        )
        print(f"Number of points: {shape_x.num_points}")
        print(f"Dimensions: {total_dims}")
        print(f"Base: {base}")

    # R(X,Y;Z) = min(I(X;Z), I(Y;Z))
    mi_xz = mutual_information(
        mat_x, mat_z, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )
    mi_yz = mutual_information(
        mat_y, mat_z, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )

    return min(mi_xz, mi_yz)


def unique(
    X: NDArray,
    Y: NDArray,
    Z: NDArray,
    *,
    method: str = "inv_ksg",
    nbins: int = 10,
    k: int = 3,
    base: float = E,
    verbose: bool = False,
    degenerate: bool = False,
    dim: int = 1,
) -> Tuple[float, float]:
    """
    Compute unique information U(X;Z) and U(Y;Z).

    U(X;Z) = I(X;Z) - R(X,Y;Z): Information X uniquely provides about Z
    U(Y;Z) = I(Y;Z) - R(X,Y;Z): Information Y uniquely provides about Z

    Args:
        X: First source variable
        Y: Second source variable
        Z: Target variable
        method: Entropy estimation method (default "inv_ksg" -- see
            mutual_information / conditional_mutual_information for what this
            changes; PID quantities are built from those two)
        nbins: Bins for histogram method
        k: Neighbors for k-NN methods
        base: Logarithmic base
        verbose: Print info
        degenerate: Handle degenerate cases
        dim: Data layout

    Returns:
        Tuple of (unique_x, unique_y)
    """
    mat_x = ensure_2d(X)
    mat_y = ensure_2d(Y)
    mat_z = ensure_2d(Z)
    mat_x = ensure_columns_are_points(mat_x, dim)
    mat_y = ensure_columns_are_points(mat_y, dim)
    mat_z = ensure_columns_are_points(mat_z, dim)

    shape_x = get_shape(mat_x)
    shape_y = get_shape(mat_y)
    shape_z = get_shape(mat_z)

    validate_same_num_points([shape_x, shape_y, shape_z])
    validate_dimensions_equal_one([shape_x, shape_y, shape_z])

    if verbose:
        total_dims = (
            shape_x.num_dimensions + shape_y.num_dimensions + shape_z.num_dimensions
        )
        print(f"Number of points: {shape_x.num_points}")
        print(f"Dimensions: {total_dims}")
        print(f"Base: {base}")

    mi_xz = mutual_information(
        mat_x, mat_z, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )
    mi_yz = mutual_information(
        mat_y, mat_z, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )

    # Redundancy with corrected MI values
    redundancy_xy_z = min(mi_xz, mi_yz)

    # Unique information
    unique_x = mi_xz - redundancy_xy_z
    unique_y = mi_yz - redundancy_xy_z

    return (unique_x, unique_y)


def synergy(
    X: NDArray,
    Y: NDArray,
    Z: NDArray,
    *,
    method: str = "inv_ksg",
    nbins: int = 10,
    k: int = 3,
    base: float = E,
    verbose: bool = False,
    degenerate: bool = False,
    dim: int = 1,
) -> float:
    """
    Compute synergy S(X,Y;Z) = I({X,Y};Z) - U(X;Z) - U(Y;Z) - R(X,Y;Z).

    Information that X and Y jointly provide about Z beyond their
    individual contributions. I({X,Y};Z) is the joint mutual information of
    the pair {X,Y} treated as one combined variable against Z -- NOT
    conditional_mutual_information(X,Y,Z) = I(X;Y|Z), which is a different
    quantity entirely and was used here previously. That mixup made this
    function's output not satisfy the PID reconstruction identity
    R+U_X+U_Y+Synergy=I_joint -- verified by testing that identity
    directly, which is how this was caught.

    Args:
        X: First source variable
        Y: Second source variable
        Z: Target variable
        method: Entropy estimation method (default "inv_ksg" -- see
            mutual_information / conditional_mutual_information for what this
            changes; PID quantities are built from those two)
        nbins: Bins for histogram method
        k: Neighbors for k-NN methods
        base: Logarithmic base
        verbose: Print info
        degenerate: Handle degenerate cases
        dim: Data layout

    Returns:
        Synergy S(X,Y;Z)
    """
    mat_x = ensure_2d(X)
    mat_y = ensure_2d(Y)
    mat_z = ensure_2d(Z)
    mat_x = ensure_columns_are_points(mat_x, dim)
    mat_y = ensure_columns_are_points(mat_y, dim)
    mat_z = ensure_columns_are_points(mat_z, dim)

    shape_x = get_shape(mat_x)
    shape_y = get_shape(mat_y)
    shape_z = get_shape(mat_z)

    validate_same_num_points([shape_x, shape_y, shape_z])
    validate_dimensions_equal_one([shape_x, shape_y, shape_z])

    if verbose:
        total_dims = (
            shape_x.num_dimensions + shape_y.num_dimensions + shape_z.num_dimensions
        )
        print(f"Number of points: {shape_x.num_points}")
        print(f"Dimensions: {total_dims}")
        print(f"Base: {base}")

    i_joint = _joint_mutual_information(
        mat_x, mat_y, mat_z, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate
    )

    unique_x, unique_y = unique(
        mat_x, mat_y, mat_z, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )

    redundancy_xy_z = redundancy(
        mat_x, mat_y, mat_z, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )

    # S = I({X,Y};Z) - U(X;Z) - U(Y;Z) - R(X,Y;Z)
    return i_joint - unique_x - unique_y - redundancy_xy_z
