"""Tests for core entropy functions."""

import numpy as np
import pytest
from entropy_invariant import entropy, entropy_inv, entropy_knn, entropy_hist


class TestEntropy:
    """Tests for the entropy function."""

    def test_entropy_1d(self, simple_test_data):
        """Test entropy computation on 1D data."""
        x, _ = simple_test_data
        h = entropy(x)
        assert isinstance(h, float)
        assert np.isfinite(h)

    def test_entropy_methods(self, simple_test_data):
        """Test all entropy methods produce finite results."""
        x, _ = simple_test_data

        h_inv = entropy(x, method="inv")
        h_knn = entropy(x, method="knn")
        h_hist = entropy(x, method="histogram")

        assert np.isfinite(h_inv)
        assert np.isfinite(h_knn)
        assert np.isfinite(h_hist)

    def test_entropy_inv_scale_invariance(self, simple_test_data):
        """Test that invariant method is scale-invariant."""
        x, _ = simple_test_data

        h1 = entropy_inv(x, k=3)
        h2 = entropy_inv(1e5 * x, k=3)
        h3 = entropy_inv(1e-5 * x, k=3)

        # Should be approximately equal (scale invariant)
        assert abs(h1 - h2) < 0.1, f"Scale invariance failed: {h1} vs {h2}"
        assert abs(h1 - h3) < 0.1, f"Scale invariance failed: {h1} vs {h3}"

    def test_entropy_inv_translation_invariance(self, simple_test_data):
        """Test that invariant method is translation-invariant."""
        x, _ = simple_test_data

        h1 = entropy_inv(x, k=3)
        h2 = entropy_inv(x + 1000, k=3)
        h3 = entropy_inv(x - 500, k=3)

        # Should be approximately equal (translation invariant)
        assert abs(h1 - h2) < 0.01, f"Translation invariance failed: {h1} vs {h2}"
        assert abs(h1 - h3) < 0.01, f"Translation invariance failed: {h1} vs {h3}"

    def test_entropy_2d(self, simple_test_data):
        """Test entropy on 2D data (joint entropy)."""
        x, y = simple_test_data
        xy = np.column_stack([x, y])

        h = entropy(xy, method="inv")
        assert np.isfinite(h)

    def test_entropy_dim_consistency(self, simple_test_data):
        """Test that dim=1 and dim=2 produce consistent results."""
        x, y = simple_test_data
        xy = np.column_stack([x, y])  # Shape: (n, 2), rows are points

        h1 = entropy(xy, dim=1)  # rows are points
        h2 = entropy(xy.T, dim=2)  # cols are points

        assert abs(h1 - h2) < 1e-10, f"Dim consistency failed: {h1} vs {h2}"

    def test_entropy_vector_matrix_consistency(self, simple_test_data):
        """Test that vector and matrix inputs produce same results."""
        x, _ = simple_test_data

        h_vec = entropy(x)
        h_mat = entropy(x.reshape(-1, 1))

        assert abs(h_vec - h_mat) < 1e-10, f"Vector/matrix consistency failed: {h_vec} vs {h_mat}"

    def test_entropy_knn_different_from_inv(self, simple_test_data):
        """Test that knn and inv methods give different results (they should)."""
        x, _ = simple_test_data

        h_inv = entropy(x, method="inv")
        h_knn = entropy(x, method="knn")

        # These should generally be different
        assert h_inv != h_knn

    def test_entropy_hist_dimension_limit(self, simple_test_data):
        """Test that histogram method raises error for dim > 3."""
        x, y = simple_test_data

        # 4D data should raise error for histogram method
        data_4d = np.column_stack([x, y, x, y])

        with pytest.raises(ValueError, match="Maximum dimension"):
            entropy(data_4d, method="histogram")

    def test_entropy_invalid_method(self, simple_test_data):
        """Test that invalid method raises error."""
        x, _ = simple_test_data

        with pytest.raises(ValueError, match="Invalid method"):
            entropy(x, method="invalid")

    def test_entropy_base_conversion(self, simple_test_data):
        """Test entropy with different bases."""
        x, _ = simple_test_data

        h_nats = entropy(x, base=np.e)
        h_bits = entropy(x, base=2)

        # H_bits = H_nats / ln(2)
        expected_bits = h_nats / np.log(2)
        assert abs(h_bits - expected_bits) < 1e-10

    def test_entropy_degenerate_mode(self, simple_test_data):
        """Test degenerate mode doesn't crash."""
        x, _ = simple_test_data

        h_normal = entropy(x, degenerate=False)
        h_degenerate = entropy(x, degenerate=True)

        assert np.isfinite(h_normal)
        assert np.isfinite(h_degenerate)
