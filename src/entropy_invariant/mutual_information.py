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
from entropy_invariant.ksg import mutual_information_ksg


def conditional_entropy(
    X: NDArray,
    Y: NDArray,
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
    Compute conditional entropy H(Y|X) = H(X,Y) - H(X).

    Args:
        X: First variable (conditioning variable)
        Y: Second variable
        method: Entropy estimation method. `"inv_ksg"` (default) computes
            H(Y) - I(X;Y) using the plain invariant entropy for H(Y) and the
            bias-cancelling KSG estimator (see `mutual_information`) for I(X;Y).
            `"inv"`, `"knn"`, and `"histogram"` instead use the plug-in formula
            H(X,Y) - H(X), differencing two independently-estimated entropies.
        nbins: Bins for histogram method
        k: Neighbors for k-NN methods
        base: Logarithmic base
        verbose: Print info
        degenerate: Handle degenerate cases (ignored by method="inv_ksg")
        dim: Data layout

    Returns:
        Conditional entropy H(Y|X)
    """
    if method == "inv_ksg":
        ent_y = entropy(Y, method="inv", k=k, base=base, dim=dim)
        mi_xy = mutual_information_ksg(X, Y, k=k, base=base, verbose=verbose, dim=dim)
        return ent_y - mi_xy

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
    method: str = "inv_ksg",
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
        method: Entropy estimation method. `"inv_ksg"` (default) normalizes X
            and Y by their invariant measure and then applies the KSG
            (Kraskov et al. 2004) shared-radius estimator directly, which
            cancels the leading-order k-NN bias that the plug-in formula does
            not -- this matters most on outlier-contaminated or near-degenerate
            data. `"inv"`, `"knn"`, and `"histogram"` instead build MI from
            independently-estimated H(X), H(Y), H(X,Y) (plug-in formula); use
            these if you need the individual entropy terms, not just I(X;Y).
        nbins: Bins for histogram method
        k: Neighbors for k-NN methods
        base: Logarithmic base
        verbose: Print info
        degenerate: Handle degenerate cases (ignored by method="inv_ksg")
        dim: Data layout

    Returns:
        Mutual information I(X;Y)
    """
    if method == "inv_ksg":
        return mutual_information_ksg(X, Y, k=k, base=base, verbose=verbose, dim=dim)

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
