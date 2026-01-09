"""Type definitions for the entropy_invariant package."""

from dataclasses import dataclass
from typing import List
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DataShape:
    """
    Shape information for datasets.

    In canonical format (columns as points):
    - num_points: Number of data points (columns)
    - num_dimensions: Number of dimensions (rows)
    """

    num_points: int
    num_dimensions: int


@dataclass
class KNNResult:
    """
    Results from k-nearest neighbor computations.

    Attributes:
        indices: Indices of k-nearest neighbors for each point, shape (n_points, k+1)
        all_distances: All k-nearest neighbor distances for each point, shape (n_points, k+1)
        kth_distances: Distance to the k-th nearest neighbor for each point, shape (n_points,)
    """

    indices: NDArray[np.intp]
    all_distances: NDArray[np.float64]
    kth_distances: NDArray[np.float64]
