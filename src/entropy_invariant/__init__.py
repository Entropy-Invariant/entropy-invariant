"""
EntropyInvariant - Invariant entropy estimation using nearest neighbor methods.

This package implements an improved nearest neighbor method for estimating
differential entropy for continuous variables, solving Edwin Thompson Jaynes'
limiting density of discrete points problem.

The main innovation is the invariant measure m(x) based on the median value
of nearest-neighbor distances, which ensures:
- Invariance under change of variables (scaling and translation)
- Always positive entropy values

Example usage:
    >>> import numpy as np
    >>> from entropy_invariant import entropy, mutual_information
    >>>
    >>> # Generate random data
    >>> x = np.random.rand(1000)
    >>> y = 2 * x + np.random.rand(1000)
    >>>
    >>> # Compute entropy (invariant method)
    >>> h = entropy(x)
    >>>
    >>> # Entropy is scale-invariant
    >>> h_scaled = entropy(1e5 * x - 123.456)  # Same value!
    >>>
    >>> # Mutual information
    >>> mi = mutual_information(x, y)

Authors: Felix Truong, Alexandre Giuliani
"""

from entropy_invariant.entropy import (
    entropy,
    entropy_hist,
    entropy_inv,
    entropy_knn,
)
from entropy_invariant.mutual_information import (
    conditional_entropy,
    mutual_information,
)
from entropy_invariant.advanced import (
    conditional_mutual_information,
    information_quality_ratio,
    interaction_information,
    normalized_mutual_information,
)
from entropy_invariant.pid import (
    redundancy,
    synergy,
    unique,
)
from entropy_invariant.optimized import (
    MI,
    CMI,
)

__version__ = "1.1.0"
__all__ = [
    # Core entropy
    "entropy",
    "entropy_hist",
    "entropy_knn",
    "entropy_inv",
    # Basic information theory
    "conditional_entropy",
    "mutual_information",
    # Advanced information theory
    "conditional_mutual_information",
    "normalized_mutual_information",
    "interaction_information",
    "information_quality_ratio",
    # Partial Information Decomposition
    "redundancy",
    "unique",
    "synergy",
    # Optimized matrix functions
    "MI",
    "CMI",
]
