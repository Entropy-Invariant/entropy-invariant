"""Core computation helper functions for entropy estimation."""

import math
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.special import digamma

from entropy_invariant._types import DataShape, KNNResult
from entropy_invariant._constants import E, LOG_UNIT_BALL_VOLUMES
from entropy_invariant.helpers.utility import nn1


def compute_invariant_measure(data: NDArray[np.float64]) -> float:
    """
    Compute the invariant measure r_X for a 1D dataset.

    This is the core innovation solving Jaynes' limiting density problem.
    Formula: r_X = median(nearest_neighbor_distances) * num_points

    Args:
        data: 1D data array

    Returns:
        The invariant measure r_X
    """
    # Filter out zero values (sparse data handling)
    non_zero_data = data[data != 0]

    if len(non_zero_data) < 2:
        return 1.0  # Fallback for insufficient data

    sorted_data = np.sort(non_zero_data)
    nn_distances = nn1(sorted_data)
    median_distance = np.median(nn_distances)
    num_points = len(non_zero_data)
    return float(median_distance * num_points)


def normalize_by_invariant_measure(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalize each dimension of a matrix by its invariant measure.

    Args:
        mat: Data matrix in canonical format (rows=dimensions, cols=points)

    Returns:
        Normalized matrix where mat[i,:] /= r_X[i] for each dimension i
    """
    normalized = mat.copy()
    num_dims = mat.shape[0]

    for i in range(num_dims):
        measure = compute_invariant_measure(mat[i, :])
        normalized[i, :] /= measure

    return normalized


def compute_knn_distances(mat: NDArray[np.float64], k: int) -> KNNResult:
    """
    Compute k-nearest neighbor distances for all points.

    Uses scipy.spatial.cKDTree for efficient search.

    Args:
        mat: Data matrix in canonical format (rows=dimensions, cols=points)
        k: Number of nearest neighbors (excluding the point itself)

    Returns:
        KNNResult with indices, distances, and k-th distances
    """
    # cKDTree expects shape (n_points, n_dims), so transpose
    points = mat.T  # Shape: (num_points, num_dims)

    kdtree = cKDTree(points)

    # Query k+1 neighbors (includes the point itself at distance 0)
    distances, indices = kdtree.query(points, k=k + 1)

    # Extract k-th neighbor distance (index k, since 0 is self)
    kth_distances = distances[:, k]

    return KNNResult(
        indices=indices, all_distances=distances, kth_distances=kth_distances
    )


def extract_nonzero_log_distances(
    distances: NDArray[np.float64], noise: int = 0
) -> NDArray[np.float64]:
    """
    Extract logarithms of non-zero distances.

    Args:
        distances: Vector of distances
        noise: 0 for normal mode, 1 for degenerate mode (adds 1 before log)

    Returns:
        Log of non-zero distances
    """
    # Filter zeros and compute log
    nonzero_mask = distances != 0
    nonzero_dists = distances[nonzero_mask]
    return np.log(nonzero_dists + noise)


def compute_knn_entropy_nats(
    log_distances: NDArray[np.float64], dimension: int, k: int
) -> float:
    """
    Compute k-NN entropy estimate in nats using Kraskov formula.

    Formula: H = d * mean(log(rho_k)) + log(V_d) + psi(n) - psi(k)

    where:
        d = dimension
        rho_k = distance to k-th nearest neighbor
        V_d = volume of d-dimensional unit ball
        psi = digamma function
        n = number of points
        k = number of neighbors

    Args:
        log_distances: Logarithms of k-th nearest neighbor distances
        dimension: Dimensionality of the space (1, 2, or 3)
        k: Number of neighbors used

    Returns:
        Entropy estimate in nats (base e)

    Raises:
        ValueError: If dimension not in [1, 2, 3]
    """
    if dimension < 1 or dimension > 3:
        raise ValueError(
            f"Unit ball volume only available for dimensions 1-3, got {dimension}"
        )

    n = len(log_distances)
    mean_log_dist = np.mean(log_distances)
    log_volume = LOG_UNIT_BALL_VOLUMES[dimension - 1]  # 0-indexed

    entropy = dimension * mean_log_dist + log_volume + digamma(n) - digamma(k)

    return float(entropy)


def convert_to_base(entropy_nats: float, base: float) -> float:
    """
    Convert entropy from nats (base e) to specified logarithmic base.

    Conversion: H_base = H_nats * log_base(e) = H_nats / ln(base)

    Args:
        entropy_nats: Entropy in nats
        base: Target logarithmic base

    Returns:
        Entropy in specified base
    """
    # log_base(e) = 1 / ln(base)
    # But Julia uses log(base, e) which is ln(base), not 1/ln(base)
    # Let me check the Julia code again...
    # Julia: log(base, e) computes log_base(e) = 1/ln(base)
    # So: H_base = H_nats * log_base(e) = H_nats / ln(base)
    return entropy_nats / math.log(base)
