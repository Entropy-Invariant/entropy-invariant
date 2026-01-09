"""Mutual information and conditional entropy functions."""

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


def conditional_entropy(
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
    Compute conditional entropy H(Y|X) = H(X,Y) - H(X).

    Args:
        X: First variable (conditioning variable)
        Y: Second variable
        method: Entropy estimation method
        nbins: Bins for histogram method
        k: Neighbors for k-NN methods
        base: Logarithmic base
        verbose: Print info
        degenerate: Handle degenerate cases
        dim: Data layout

    Returns:
        Conditional entropy H(Y|X)
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

    # H(Y|X) = H(X,Y) - H(X)
    joint_mat = np.vstack([mat_x, mat_y])
    ent_x = entropy(
        mat_x, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2
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

    return ent_joint - ent_x


def mutual_information(
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
    Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).

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
        Mutual information I(X;Y)
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

    # I(X;Y) = H(X) + H(Y) - H(X,Y)
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

    return ent_x + ent_y - ent_joint
