"""Tests for Partial Information Decomposition functions."""

import numpy as np
import pytest
from entropy_invariant import redundancy, unique, synergy


class TestPID:
    """Tests for PID functions."""

    def test_redundancy_basic(self, simple_test_data):
        """Test basic redundancy computation."""
        x, y = simple_test_data
        z = np.random.rand(len(x))

        r = redundancy(x, y, z)
        assert isinstance(r, float)
        assert np.isfinite(r)

    def test_unique_basic(self, simple_test_data):
        """Test basic unique information computation."""
        x, y = simple_test_data
        z = np.random.rand(len(x))

        ux, uy = unique(x, y, z)
        assert isinstance(ux, float)
        assert isinstance(uy, float)
        assert np.isfinite(ux)
        assert np.isfinite(uy)

    def test_synergy_basic(self, simple_test_data):
        """Test basic synergy computation."""
        x, y = simple_test_data
        z = np.random.rand(len(x))

        s = synergy(x, y, z)
        assert isinstance(s, float)
        assert np.isfinite(s)

    def test_unique_returns_tuple(self, simple_test_data):
        """Test that unique returns a tuple of two values."""
        x, y = simple_test_data
        z = np.random.rand(len(x))

        result = unique(x, y, z)
        assert isinstance(result, tuple)
        assert len(result) == 2
