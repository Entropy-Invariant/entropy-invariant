"""Utility helper functions for histogram and nearest neighbor computations."""

from typing import Optional
import numpy as np
from numpy.typing import NDArray

from entropy_invariant._types import DataShape


def nn1(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate distances between neighboring elements in a sorted vector.

    For interior elements: minimum distance to left or right neighbor.
    For boundary elements: distance to the only neighbor.

    Args:
        array: A sorted vector of real numbers

    Returns:
        Vector of minimum distances to nearest neighbor for each element

    Raises:
        ValueError: If input array contains fewer than 2 elements
    """
    m = len(array)
    if m < 2:
        raise ValueError("Input array must contain more than one element")

    all_dist = np.zeros(m, dtype=np.float64)

    # Interior elements: min distance to left or right neighbor
    for i in range(1, m - 1):
        all_dist[i] = min(
            abs(array[i] - array[i - 1]), abs(array[i] - array[i + 1])
        )

    # Boundary elements
    all_dist[0] = abs(array[0] - array[1])
    all_dist[-1] = abs(array[-2] - array[-1])

    return all_dist


def hist1d(data: NDArray, nbins: int) -> NDArray[np.int64]:
    """
    Compute a 1D histogram matching Julia's implementation exactly.

    Bins are left-inclusive, right-exclusive: [edge_i, edge_{i+1})
    except the last bin which is inclusive on both ends: [edge_i, edge_{i+1}]

    Args:
        data: Vector of numeric values
        nbins: Number of bins

    Returns:
        Histogram counts
    """
    min_val, max_val = np.min(data), np.max(data)
    bin_edges = np.linspace(min_val, max_val, nbins + 1)
    counts = np.zeros(nbins, dtype=np.int64)

    for value in data:
        for i in range(nbins):
            # Last bin is inclusive on both ends
            if i == nbins - 1:
                if value >= bin_edges[i] and value <= bin_edges[i + 1]:
                    counts[i] += 1
                    break
            else:
                if value >= bin_edges[i] and value < bin_edges[i + 1]:
                    counts[i] += 1
                    break

    return counts


def hist2d(x: NDArray, y: NDArray, nbins: int) -> NDArray[np.int64]:
    """
    Compute a 2D histogram matching Julia's implementation exactly.

    Args:
        x: First dimension values
        y: Second dimension values
        nbins: Number of bins per dimension

    Returns:
        2D histogram counts
    """
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    bin_edges_x = np.linspace(min_x, max_x, nbins + 1)
    bin_edges_y = np.linspace(min_y, max_y, nbins + 1)

    counts = np.zeros((nbins, nbins), dtype=np.int64)

    for k in range(len(x)):
        # Find bin index for x[k]: first edge >= x[k], then subtract 1
        i = np.searchsorted(bin_edges_x, x[k], side="left") - 1
        j = np.searchsorted(bin_edges_y, y[k], side="left") - 1

        # Handle edge case where value equals first edge
        if x[k] == bin_edges_x[0]:
            i = 0
        if y[k] == bin_edges_y[0]:
            j = 0

        # Handle max values (last bin is inclusive on upper end)
        if x[k] == max_x:
            i = nbins - 1
        if y[k] == max_y:
            j = nbins - 1

        if 0 <= i < nbins and 0 <= j < nbins:
            counts[i, j] += 1

    return counts


def hist3d(x: NDArray, y: NDArray, z: NDArray, nbins: int) -> NDArray[np.int64]:
    """
    Compute a 3D histogram matching Julia's implementation exactly.

    Args:
        x, y, z: Dimension values
        nbins: Number of bins per dimension

    Returns:
        3D histogram counts
    """
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    min_z, max_z = np.min(z), np.max(z)

    bin_edges_x = np.linspace(min_x, max_x, nbins + 1)
    bin_edges_y = np.linspace(min_y, max_y, nbins + 1)
    bin_edges_z = np.linspace(min_z, max_z, nbins + 1)

    counts = np.zeros((nbins, nbins, nbins), dtype=np.int64)

    for k in range(len(x)):
        i = np.searchsorted(bin_edges_x, x[k], side="left") - 1
        j = np.searchsorted(bin_edges_y, y[k], side="left") - 1
        l = np.searchsorted(bin_edges_z, z[k], side="left") - 1

        # Handle edge case where value equals first edge
        if x[k] == bin_edges_x[0]:
            i = 0
        if y[k] == bin_edges_y[0]:
            j = 0
        if z[k] == bin_edges_z[0]:
            l = 0

        # Handle max values (last bin is inclusive on upper end)
        if x[k] == max_x:
            i = nbins - 1
        if y[k] == max_y:
            j = nbins - 1
        if z[k] == max_z:
            l = nbins - 1

        if 0 <= i < nbins and 0 <= j < nbins and 0 <= l < nbins:
            counts[i, j, l] += 1

    return counts


def log_computation_info(shape: DataShape, base: float, verbose: bool = True) -> None:
    """
    Print computation information for a dataset.

    Args:
        shape: Dataset shape information
        base: Logarithmic base being used
        verbose: Whether to print (default True)
    """
    if verbose:
        print(f"Number of points: {shape.num_points}")
        print(f"Dimensions: {shape.num_dimensions}")
        print(f"Base: {base}")
