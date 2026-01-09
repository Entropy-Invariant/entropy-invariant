"""Tests for optimized matrix functions."""

import numpy as np
import pytest
from entropy_invariant import MI, CMI, mutual_information


class TestMIMatrix:
    """Tests for MI matrix function."""

    def test_mi_matrix_basic(self, simple_test_data):
        """Test basic MI matrix computation."""
        x, y = simple_test_data
        data = np.column_stack([x, y])

        mi_mat = MI(data)

        assert mi_mat.shape == (2, 2)
        assert np.isfinite(mi_mat).all()

    def test_mi_matrix_symmetric(self, simple_test_data):
        """Test that MI matrix is symmetric."""
        x, y = simple_test_data
        z = np.random.rand(len(x))
        data = np.column_stack([x, y, z])

        mi_mat = MI(data)

        assert mi_mat.shape == (3, 3)
        assert np.allclose(mi_mat, mi_mat.T)

    def test_mi_matrix_consistent_with_mi(self, simple_test_data):
        """Test that MI matrix gives similar results to pairwise MI."""
        x, y = simple_test_data
        data = np.column_stack([x, y])

        mi_mat = MI(data, k=3)
        mi_direct = mutual_information(x, y, k=3)

        # Should be approximately equal
        assert abs(mi_mat[0, 1] - mi_direct) < 0.1

    def test_mi_matrix_dim_consistency(self, simple_test_data):
        """Test dim=1 vs dim=2 consistency."""
        x, y = simple_test_data
        data = np.column_stack([x, y])

        mi1 = MI(data, dim=1)  # rows are points
        mi2 = MI(data.T, dim=2)  # cols are points

        assert np.allclose(mi1, mi2, atol=1e-10)


class TestCMIMatrix:
    """Tests for CMI matrix function."""

    def test_cmi_matrix_basic(self, simple_test_data):
        """Test basic CMI matrix computation."""
        x, y = simple_test_data
        z = np.random.rand(len(x))
        data = np.column_stack([x, y])

        cmi_mat = CMI(data, z)

        assert cmi_mat.shape == (2, 2)
        assert np.isfinite(cmi_mat).all()

    def test_cmi_matrix_symmetric(self, simple_test_data):
        """Test that CMI matrix is symmetric."""
        x, y = simple_test_data
        z = np.random.rand(len(x))
        data = np.column_stack([x, y])

        cmi_mat = CMI(data, z)

        assert np.allclose(cmi_mat, cmi_mat.T)

    def test_cmi_matrix_vector_z(self, simple_test_data):
        """Test CMI with vector conditioning variable."""
        x, y = simple_test_data
        z = np.random.rand(len(x))
        data = np.column_stack([x, y])

        cmi_mat = CMI(data, z)
        assert cmi_mat.shape == (2, 2)

    def test_cmi_matrix_column_z(self, simple_test_data):
        """Test CMI with column vector conditioning variable."""
        x, y = simple_test_data
        z = np.random.rand(len(x)).reshape(-1, 1)
        data = np.column_stack([x, y])

        cmi_mat = CMI(data, z)
        assert cmi_mat.shape == (2, 2)
