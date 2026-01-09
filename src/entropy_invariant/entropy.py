"""Core entropy estimation functions."""

from typing import Union
import math
import numpy as np
from numpy.typing import NDArray

from entropy_invariant._types import DataShape
from entropy_invariant._constants import E
from entropy_invariant.helpers.data import (
    ensure_columns_are_points,
    ensure_2d,
    get_shape,
)
from entropy_invariant.helpers.utility import (
    hist1d,
    hist2d,
    hist3d,
    log_computation_info,
)
from entropy_invariant.helpers.computation import (
    normalize_by_invariant_measure,
    compute_knn_distances,
    extract_nonzero_log_distances,
    compute_knn_entropy_nats,
    convert_to_base,
)


def entropy_hist(
    X: NDArray,
    *,
    nbins: int = 10,
    dim: int = 1,
    base: float = E,
    verbose: bool = False,
) -> float:
    """
    Compute entropy using histogram method.

    Args:
        X: Data array (1D or 2D)
        nbins: Number of histogram bins (default 10)
        dim: Data layout (1=rows are points, 2=cols are points)
        base: Logarithmic base (default e)
        verbose: Print computation info

    Returns:
        Entropy estimate

    Raises:
        ValueError: If data has more than 3 dimensions
    """
    mat = ensure_2d(X)
    mat = ensure_columns_are_points(mat, dim)
    shape = get_shape(mat)

    if verbose:
        log_computation_info(shape, base)

    if shape.num_dimensions > 3:
        raise ValueError("Maximum dimension for histogram method is 3")

    # Compute histogram
    if shape.num_dimensions == 1:
        weights = hist1d(mat.flatten(), nbins)
    elif shape.num_dimensions == 2:
        weights = hist2d(mat[0, :], mat[1, :], nbins)
    else:  # 3 dimensions
        weights = hist3d(mat[0, :], mat[1, :], mat[2, :], nbins)

    # Convert to probabilities
    probs = weights / weights.sum()

    # Compute entropy: H = -sum(p * log(p)) for p > 0
    nonzero_probs = probs[probs > 0]
    entropy_nats = -np.sum(nonzero_probs * np.log(nonzero_probs))

    return convert_to_base(entropy_nats, base)


def entropy_knn(
    X: NDArray,
    *,
    k: int = 3,
    base: float = E,
    verbose: bool = False,
    degenerate: bool = False,
    dim: int = 1,
) -> float:
    """
    Compute entropy using k-NN method (without invariant measure).

    Args:
        X: Data array (1D or 2D)
        k: Number of nearest neighbors (default 3)
        base: Logarithmic base (default e)
        verbose: Print computation info
        degenerate: Add +1 to distances for degenerate cases
        dim: Data layout

    Returns:
        Entropy estimate
    """
    mat = ensure_2d(X)
    mat = ensure_columns_are_points(mat, dim)
    shape = get_shape(mat)

    if verbose:
        log_computation_info(shape, base)

    noise = 1 if degenerate else 0
    knn_result = compute_knn_distances(mat, k)
    log_dists = extract_nonzero_log_distances(knn_result.kth_distances, noise)

    entropy_nats = compute_knn_entropy_nats(log_dists, shape.num_dimensions, k)
    return convert_to_base(entropy_nats, base)


def entropy_inv(
    X: NDArray,
    *,
    k: int = 3,
    base: float = E,
    verbose: bool = False,
    degenerate: bool = False,
    dim: int = 1,
) -> float:
    """
    Compute entropy using invariant method (default).

    This method normalizes by the invariant measure, ensuring scale/translation
    invariance and always-positive entropy values.

    Args:
        X: Data array (1D or 2D)
        k: Number of nearest neighbors (default 3)
        base: Logarithmic base (default e)
        verbose: Print computation info
        degenerate: Add +1 to distances for degenerate cases
        dim: Data layout

    Returns:
        Entropy estimate
    """
    mat = ensure_2d(X)
    mat = ensure_columns_are_points(mat, dim)
    shape = get_shape(mat)

    if verbose:
        log_computation_info(shape, base)

    # Invariant measure normalization - the key innovation
    normalized_mat = normalize_by_invariant_measure(mat)

    noise = 1 if degenerate else 0
    knn_result = compute_knn_distances(normalized_mat, k)
    log_dists = extract_nonzero_log_distances(knn_result.kth_distances, noise)

    entropy_nats = compute_knn_entropy_nats(log_dists, shape.num_dimensions, k)
    return convert_to_base(entropy_nats, base)


def entropy(
    X: NDArray,
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
    Unified entropy computation interface.

    Args:
        X: Data array (1D or 2D)
        method: Estimation method ("inv", "knn", or "histogram")
        nbins: Number of bins for histogram method
        k: Number of neighbors for k-NN/invariant methods
        base: Logarithmic base (default e)
        verbose: Print computation info
        degenerate: Add +1 for degenerate cases
        dim: Data layout (1=rows are points, 2=cols are points)

    Returns:
        Entropy estimate

    Raises:
        ValueError: If invalid method specified
    """
    if method == "knn":
        return entropy_knn(
            X, k=k, base=base, verbose=verbose, degenerate=degenerate, dim=dim
        )
    elif method == "histogram":
        return entropy_hist(X, nbins=nbins, base=base, verbose=verbose, dim=dim)
    elif method == "inv":
        return entropy_inv(
            X, k=k, base=base, verbose=verbose, degenerate=degenerate, dim=dim
        )
    else:
        raise ValueError(
            f"Invalid method: {method}. Choose 'inv', 'knn', or 'histogram'"
        )
