"""Helper functions for entropy computation."""

from entropy_invariant.helpers.utility import (
    nn1,
    hist1d,
    hist2d,
    hist3d,
    log_computation_info,
)
from entropy_invariant.helpers.data import (
    get_shape,
    ensure_columns_are_points,
    vector_to_matrix,
    ensure_2d,
    validate_same_num_points,
    validate_dimensions_equal_one,
)
from entropy_invariant.helpers.computation import (
    compute_invariant_measure,
    normalize_by_invariant_measure,
    compute_knn_distances,
    extract_nonzero_log_distances,
    compute_knn_entropy_nats,
    convert_to_base,
)

__all__ = [
    "nn1",
    "hist1d",
    "hist2d",
    "hist3d",
    "log_computation_info",
    "get_shape",
    "ensure_columns_are_points",
    "vector_to_matrix",
    "ensure_2d",
    "validate_same_num_points",
    "validate_dimensions_equal_one",
    "compute_invariant_measure",
    "normalize_by_invariant_measure",
    "compute_knn_distances",
    "extract_nonzero_log_distances",
    "compute_knn_entropy_nats",
    "convert_to_base",
]
