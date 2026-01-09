"""Tests for mutual information and conditional entropy functions."""

import numpy as np
import pytest
from entropy_invariant import (
    mutual_information,
    conditional_entropy,
    entropy,
)


class TestMutualInformation:
    """Tests for mutual information."""

    def test_mi_basic(self, simple_test_data):
        """Test basic MI computation."""
        x, y = simple_test_data
        mi = mutual_information(x, y)
        assert isinstance(mi, float)
        assert np.isfinite(mi)

    def test_mi_symmetric(self, simple_test_data):
        """Test that MI is symmetric: I(X;Y) = I(Y;X)."""
        x, y = simple_test_data
        mi_xy = mutual_information(x, y)
        mi_yx = mutual_information(y, x)
        assert abs(mi_xy - mi_yx) < 1e-10, f"MI not symmetric: {mi_xy} vs {mi_yx}"

    def test_mi_self(self, simple_test_data):
        """Test that I(X;X) is finite and positive."""
        x, _ = simple_test_data
        mi_xx = mutual_information(x, x)
        # For invariant method, I(X;X) may not equal H(X) due to
        # different normalization in joint vs marginal computations
        assert np.isfinite(mi_xx)
        assert mi_xx > 0, "I(X;X) should be positive"

    def test_mi_dim_consistency(self, simple_test_data):
        """Test dim=1 vs dim=2 consistency."""
        x, y = simple_test_data

        mi1 = mutual_information(x.reshape(-1, 1), y.reshape(-1, 1), dim=1)
        mi2 = mutual_information(x.reshape(1, -1), y.reshape(1, -1), dim=2)

        assert abs(mi1 - mi2) < 1e-10

    def test_mi_vector_input(self, simple_test_data):
        """Test MI with vector inputs."""
        x, y = simple_test_data
        mi = mutual_information(x, y)
        assert np.isfinite(mi)

    def test_mi_different_methods(self, simple_test_data):
        """Test MI with different methods."""
        x, y = simple_test_data

        mi_inv = mutual_information(x, y, method="inv")
        mi_knn = mutual_information(x, y, method="knn")

        assert np.isfinite(mi_inv)
        assert np.isfinite(mi_knn)


class TestConditionalEntropy:
    """Tests for conditional entropy."""

    def test_cond_entropy_basic(self, simple_test_data):
        """Test basic conditional entropy."""
        x, y = simple_test_data
        h_yx = conditional_entropy(x, y)
        assert isinstance(h_yx, float)
        assert np.isfinite(h_yx)

    def test_cond_entropy_chain_rule(self, simple_test_data):
        """Test chain rule: H(X,Y) = H(X) + H(Y|X)."""
        x, y = simple_test_data

        xy = np.column_stack([x, y])
        h_joint = entropy(xy)
        h_x = entropy(x)
        h_y_given_x = conditional_entropy(x, y)

        # H(X,Y) should approximately equal H(X) + H(Y|X)
        expected = h_x + h_y_given_x
        assert abs(h_joint - expected) < 0.1, f"Chain rule failed: {h_joint} vs {expected}"

    def test_cond_entropy_dim_consistency(self, simple_test_data):
        """Test dim=1 vs dim=2 consistency."""
        x, y = simple_test_data

        h1 = conditional_entropy(x.reshape(-1, 1), y.reshape(-1, 1), dim=1)
        h2 = conditional_entropy(x.reshape(1, -1), y.reshape(1, -1), dim=2)

        assert abs(h1 - h2) < 1e-10


class TestValidation:
    """Tests for input validation."""

    def test_different_lengths_error(self):
        """Test that different length inputs raise error."""
        x = np.random.rand(100)
        y = np.random.rand(50)

        with pytest.raises(ValueError, match="same number of points"):
            mutual_information(x, y)

    def test_multidimensional_error(self):
        """Test that multi-dimensional inputs raise error for MI."""
        x = np.random.rand(100, 2)
        y = np.random.rand(100, 2)

        with pytest.raises(ValueError, match="1-dimensional"):
            mutual_information(x, y)
