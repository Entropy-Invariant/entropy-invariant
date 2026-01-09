"""Tests for advanced information theory functions."""

import numpy as np
import pytest
from entropy_invariant import (
    conditional_mutual_information,
    normalized_mutual_information,
    interaction_information,
    information_quality_ratio,
    mutual_information,
)


class TestConditionalMutualInformation:
    """Tests for CMI."""

    def test_cmi_basic(self, simple_test_data):
        """Test basic CMI computation."""
        x, y = simple_test_data
        z = np.random.rand(len(x))

        cmi = conditional_mutual_information(x, y, z)
        assert isinstance(cmi, float)
        assert np.isfinite(cmi)

    def test_cmi_reduces_to_mi(self, simple_test_data):
        """Test that CMI with independent Z is close to MI."""
        np.random.seed(999)
        x, y = simple_test_data
        # Z is independent of X and Y
        z = np.random.rand(len(x))

        mi = mutual_information(x, y)
        cmi = conditional_mutual_information(x, y, z)

        # Should be somewhat similar (not exact due to estimation)
        assert abs(mi - cmi) < 1.0  # Allow some difference


class TestNormalizedMutualInformation:
    """Tests for NMI."""

    def test_nmi_basic(self, simple_test_data):
        """Test basic NMI computation."""
        x, y = simple_test_data
        nmi = normalized_mutual_information(x, y)
        assert isinstance(nmi, float)
        assert np.isfinite(nmi)

    def test_nmi_symmetric(self, simple_test_data):
        """Test NMI symmetry."""
        x, y = simple_test_data
        nmi_xy = normalized_mutual_information(x, y)
        nmi_yx = normalized_mutual_information(y, x)
        assert abs(nmi_xy - nmi_yx) < 1e-10


class TestInteractionInformation:
    """Tests for interaction information."""

    def test_ii_basic(self, simple_test_data):
        """Test basic II computation."""
        x, y = simple_test_data
        z = np.random.rand(len(x))

        ii = interaction_information(x, y, z)
        assert isinstance(ii, float)
        assert np.isfinite(ii)


class TestInformationQualityRatio:
    """Tests for IQR."""

    def test_iqr_basic(self, simple_test_data):
        """Test basic IQR computation."""
        x, y = simple_test_data
        iqr = information_quality_ratio(x, y)
        assert isinstance(iqr, float)
        assert np.isfinite(iqr)
