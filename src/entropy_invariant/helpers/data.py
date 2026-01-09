"""Data layout and validation helper functions."""

from typing import List, Union
import numpy as np
from numpy.typing import NDArray

from entropy_invariant._types import DataShape


def get_shape(mat: NDArray[np.float64]) -> DataShape:
    """
    Extract shape information from a matrix in canonical format (dim=2).

    In canonical format:
    - Each column is a data point
    - Each row is a dimension

    Args:
        mat: Data matrix in canonical (dim=2) format, shape (num_dims, num_points)

    Returns:
        DataShape with num_points and num_dimensions
    """
    return DataShape(num_points=mat.shape[1], num_dimensions=mat.shape[0])


def ensure_columns_are_points(
    mat: NDArray[np.float64], dim: int = 1
) -> NDArray[np.float64]:
    """
    Convert matrix to canonical layout where each column is a data point.

    Args:
        mat: Input data matrix
        dim: Layout indicator
            - dim=1 (default): Input has points as rows -> transpose
            - dim=2: Input has points as columns -> no change

    Returns:
        Matrix in canonical format (rows=dimensions, cols=points)
    """
    if dim == 1:
        return np.asarray(mat.T, dtype=np.float64)
    else:
        return np.asarray(mat, dtype=np.float64)


def vector_to_matrix(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert a 1D vector to an n x 1 matrix for consistent processing.

    Args:
        vec: Input 1D array

    Returns:
        Column matrix with shape (n, 1)
    """
    return vec.reshape(-1, 1).astype(np.float64)


def ensure_2d(data: Union[NDArray, list]) -> NDArray[np.float64]:
    """
    Ensure input is 2D array, converting 1D to column vector if needed.

    Args:
        data: Input array (1D or 2D)

    Returns:
        2D array
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def validate_same_num_points(shapes: List[DataShape]) -> None:
    """
    Validate that all datasets have the same number of data points.

    Args:
        shapes: List of DataShape objects to validate

    Raises:
        ValueError: If datasets have different numbers of points
    """
    if len(shapes) < 2:
        return

    first_num_points = shapes[0].num_points
    for shape in shapes[1:]:
        if shape.num_points != first_num_points:
            raise ValueError("Input arrays must contain the same number of points")


def validate_dimensions_equal_one(shapes: List[DataShape]) -> None:
    """
    Validate that all datasets are 1-dimensional.

    Args:
        shapes: List of DataShape objects to validate

    Raises:
        ValueError: If any dataset has num_dimensions != 1
    """
    for shape in shapes:
        if shape.num_dimensions != 1:
            raise ValueError("Each input must be 1-dimensional")
