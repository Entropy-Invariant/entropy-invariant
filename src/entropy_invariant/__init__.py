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
from entropy_invariant.pid_lattice import (
    RedundancyLattice,
    coalition_mutual_information,
    iccs_redundancy,
    imin_redundancy,
    isotonic_repair,
    lattice_labels,
    mmi_redundancy,
    moebius_atoms,
    pid_lattice,
    redundancy_lattice,
    specific_information,
)
from entropy_invariant.ksg import (
    mutual_information_ksg,
    conditional_mutual_information_ksg,
)

__version__ = "2.2.0"
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
    # N-source PID: Williams & Beer redundancy lattice
    "pid_lattice",
    "redundancy_lattice",
    "RedundancyLattice",
    "lattice_labels",
    "moebius_atoms",
    "coalition_mutual_information",
    "isotonic_repair",
    "mmi_redundancy",
    "imin_redundancy",
    "iccs_redundancy",
    "specific_information",
    # Optimized matrix functions
    "MI",
    "CMI",
    # Invariant-measure + KSG/Frenzel-Pompe (bias-cancelling) estimators
    "mutual_information_ksg",
    "conditional_mutual_information_ksg",
]
