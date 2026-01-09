"""Advanced information theory functions."""

from typing import Optional
import numpy as np
from numpy.typing import NDArray

from entropy_invariant._constants import E
from entropy_invariant._types import DataShape
from entropy_invariant.helpers.data import (
    ensure_columns_are_points,
    ensure_2d,
    get_shape,
    validate_same_num_points,
    validate_dimensions_equal_one,
)
from entropy_invariant.helpers.utility import log_computation_info
from entropy_invariant.entropy import entropy


def conditional_mutual_information(
    X: NDArray,
    Y: NDArray,
    Z: NDArray,
    *,
    method: str = "inv",
    nbins: int = 10,
    k: int = 3,
    base: float = E,
    verbose: bool = False,
    degenerate: bool = False,
    dim: int = 1,
) -> float:
    """
    Compute conditional mutual information I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z).

    Args:
        X: First variable
        Y: Second variable
        Z: Conditioning variable
        method: Entropy estimation method
        nbins: Bins for histogram method
        k: Neighbors for k-NN methods
        base: Logarithmic base
        verbose: Print info
        degenerate: Handle degenerate cases
        dim: Data layout

    Returns:
        Conditional mutual information I(X;Y|Z)
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

    # I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
    xz_mat = np.vstack([mat_x, mat_z])
    yz_mat = np.vstack([mat_y, mat_z])
    xyz_mat = np.vstack([mat_x, mat_y, mat_z])

    ent_z = entropy(
        mat_z, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )
    ent_xz = entropy(
        xz_mat, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )
    ent_yz = entropy(
        yz_mat, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )
    ent_xyz = entropy(
        xyz_mat,
        method=method,
        nbins=nbins,
        k=k,
        base=base,
        degenerate=degenerate,
        dim=2,
    )

    return ent_xz + ent_yz - ent_xyz - ent_z


def normalized_mutual_information(
    X: NDArray,
    Y: NDArray,
    *,
    method: str = "inv",
    nbins: int = 10,
    k: int = 3,
    base: float = E,
    verbose: bool = False,
    degenerate: bool = False,
    dim: int = 1,
) -> float:
    """
    Compute normalized mutual information NMI = I(X;Y) / ((H(X) + H(Y)) / 2).

    Args:
        X: First variable
        Y: Second variable
        method: Entropy estimation method
        nbins: Bins for histogram method
        k: Neighbors for k-NN methods
        base: Logarithmic base
        verbose: Print info
        degenerate: Handle degenerate cases
        dim: Data layout

    Returns:
        Normalized mutual information in [0, 1]
    """
    mat_x = ensure_2d(X)
    mat_y = ensure_2d(Y)
    mat_x = ensure_columns_are_points(mat_x, dim)
    mat_y = ensure_columns_are_points(mat_y, dim)

    shape_x = get_shape(mat_x)
    shape_y = get_shape(mat_y)

    validate_same_num_points([shape_x, shape_y])
    validate_dimensions_equal_one([shape_x, shape_y])

    if verbose:
        total_dims = shape_x.num_dimensions + shape_y.num_dimensions
        print(f"Number of points: {shape_x.num_points}")
        print(f"Dimensions: {total_dims}")
        print(f"Base: {base}")

    # Compute entropies
    joint_mat = np.vstack([mat_x, mat_y])
    ent_x = entropy(
        mat_x, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )
    ent_y = entropy(
        mat_y, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )
    ent_joint = entropy(
        joint_mat,
        method=method,
        nbins=nbins,
        k=k,
        base=base,
        degenerate=degenerate,
        dim=2,
    )

    # NMI = I(X;Y) / avg(H(X), H(Y))
    # I(X;Y) = H(X) + H(Y) - H(X,Y), ensure non-negative
    mi = max(0.0, ent_x + ent_y - ent_joint)
    avg_entropy = (ent_x + ent_y) / 2.0

    return mi / avg_entropy


def interaction_information(
    X: NDArray,
    Y: NDArray,
    Z: NDArray,
    *,
    method: str = "inv",
    nbins: int = 10,
    k: int = 3,
    base: float = E,
    verbose: bool = False,
    degenerate: bool = False,
    dim: int = 1,
) -> float:
    """
    Compute interaction information (three-way interaction).

    II(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z)

    Args:
        X: First variable
        Y: Second variable
        Z: Third variable
        method: Entropy estimation method
        nbins: Bins for histogram method
        k: Neighbors for k-NN methods
        base: Logarithmic base
        verbose: Print info
        degenerate: Handle degenerate cases
        dim: Data layout

    Returns:
        Interaction information II(X;Y;Z)
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

    # Compute all required entropies
    xy_mat = np.vstack([mat_x, mat_y])
    xz_mat = np.vstack([mat_x, mat_z])
    yz_mat = np.vstack([mat_y, mat_z])
    xyz_mat = np.vstack([mat_x, mat_y, mat_z])

    ent_x = entropy(
        mat_x, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )
    ent_y = entropy(
        mat_y, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )
    ent_z = entropy(
        mat_z, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )
    ent_xy = entropy(
        xy_mat, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )
    ent_xz = entropy(
        xz_mat, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )
    ent_yz = entropy(
        yz_mat, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )
    ent_xyz = entropy(
        xyz_mat,
        method=method,
        nbins=nbins,
        k=k,
        base=base,
        degenerate=degenerate,
        dim=2,
    )

    # II = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z)
    return ent_x + ent_y + ent_z - ent_xy - ent_xz - ent_yz + ent_xyz


def information_quality_ratio(
    X: NDArray,
    Y: NDArray,
    *,
    method: str = "inv",
    nbins: int = 10,
    k: int = 3,
    base: float = E,
    verbose: bool = False,
    degenerate: bool = False,
    dim: int = 1,
) -> float:
    """
    Compute information quality ratio IQR = I(X;Y) / H(X).

    Args:
        X: First variable
        Y: Second variable
        method: Entropy estimation method
        nbins: Bins for histogram method
        k: Neighbors for k-NN methods
        base: Logarithmic base
        verbose: Print info
        degenerate: Handle degenerate cases
        dim: Data layout

    Returns:
        Information quality ratio
    """
    mat_x = ensure_2d(X)
    mat_y = ensure_2d(Y)
    mat_x = ensure_columns_are_points(mat_x, dim)
    mat_y = ensure_columns_are_points(mat_y, dim)

    shape_x = get_shape(mat_x)
    shape_y = get_shape(mat_y)

    validate_same_num_points([shape_x, shape_y])
    validate_dimensions_equal_one([shape_x, shape_y])

    if verbose:
        total_dims = shape_x.num_dimensions + shape_y.num_dimensions
        print(f"Number of points: {shape_x.num_points}")
        print(f"Dimensions: {total_dims}")
        print(f"Base: {base}")

    # Compute entropies
    joint_mat = np.vstack([mat_x, mat_y])
    ent_x = entropy(
        mat_x, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )
    ent_y = entropy(
        mat_y, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )
    ent_joint = entropy(
        joint_mat,
        method=method,
        nbins=nbins,
        k=k,
        base=base,
        degenerate=degenerate,
        dim=2,
    )

    # IQR = I(X;Y) / H(X)
    mi = ent_x + ent_y - ent_joint
    return mi / ent_x
