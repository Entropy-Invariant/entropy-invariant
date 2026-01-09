"""Partial Information Decomposition (PID) functions."""

from typing import Tuple
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
from entropy_invariant.mutual_information import mutual_information
from entropy_invariant.advanced import conditional_mutual_information


def redundancy(
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
    Compute redundancy R(X,Y;Z) = min(I(X;Z), I(Y;Z)).

    The shared information that both X and Y have about Z.

    Args:
        X: First source variable
        Y: Second source variable
        Z: Target variable
        method: Entropy estimation method
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
    method: str = "inv",
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

    Note: This function adds +1 to MI values before computing, which is
    documented technical debt to ensure positive unique information when
    redundancy equals MI.

    Args:
        X: First source variable
        Y: Second source variable
        Z: Target variable
        method: Entropy estimation method
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

    # Compute MI values with +1 correction (documented technical debt)
    mi_xz = mutual_information(
        mat_x, mat_z, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    ) + 1
    mi_yz = mutual_information(
        mat_y, mat_z, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    ) + 1

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
    method: str = "inv",
    nbins: int = 10,
    k: int = 3,
    base: float = E,
    verbose: bool = False,
    degenerate: bool = False,
    dim: int = 1,
) -> float:
    """
    Compute synergy S(X,Y;Z) = I(X,Y;Z) - U(X;Z) - U(Y;Z) - R(X,Y;Z).

    Information that X and Y jointly provide about Z beyond their
    individual contributions.

    Note: This function adds +1 to CMI and redundancy values, which is
    documented technical debt to maintain consistency with the unique()
    function's +1 corrections.

    Args:
        X: First source variable
        Y: Second source variable
        Z: Target variable
        method: Entropy estimation method
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

    # CMI with +1 correction (documented technical debt)
    cmi_xyz = conditional_mutual_information(
        mat_x, mat_y, mat_z, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    ) + 1

    # Get unique information
    unique_x, unique_y = unique(
        mat_x, mat_y, mat_z, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    )

    # Redundancy with +1 correction (documented technical debt)
    redundancy_xy_z = redundancy(
        mat_x, mat_y, mat_z, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
    ) + 1

    # S = I(X,Y;Z) - U(X;Z) - U(Y;Z) - R(X,Y;Z)
    return cmi_xyz - unique_x - unique_y - redundancy_xy_z
