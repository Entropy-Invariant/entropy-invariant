"""Tests for the invariant measure r_X and how estimators handle it."""

import numpy as np
import pytest

from entropy_invariant import (
    CMI,
    MI,
    coalition_mutual_information,
    conditional_entropy,
    conditional_mutual_information,
    entropy,
    mutual_information,
    redundancy,
    synergy,
)
from entropy_invariant.helpers.computation import compute_invariant_measure


class TestNoScaleIsNaN:
    """
    Fewer than two non-zero values: there is no spacing to measure, so no
    scale. This used to return 1.0, which left the column in its own units:
    entropy(w) and entropy(1000 * w) differed by exactly log(1000).
    """

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(4)
        n = 300
        w = np.zeros(n)
        w[6] = 3.0
        return w, rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)

    def test_measure_and_entropy(self, data):
        w, x, _, _ = data
        assert np.isnan(compute_invariant_measure(w))
        assert np.isnan(compute_invariant_measure(np.zeros(len(w))))
        assert np.isnan(entropy(w))
        assert np.isnan(entropy(1000 * w))
        assert np.isnan(entropy(np.column_stack([w, x])))

    @pytest.mark.parametrize("method", ["inv", "inv_ksg"])
    def test_every_estimator_returns_nan(self, data, method):
        w, x, y, z = data
        assert np.isnan(mutual_information(w, x, method=method))
        assert np.isnan(conditional_mutual_information(w, x, z, method=method))
        assert np.isnan(conditional_mutual_information(x, y, w, method=method))
        assert np.isnan(conditional_entropy(w, x, method=method))
        assert np.isnan(redundancy(w, x, z, method=method))
        assert np.isnan(synergy(w, x, z, method=method))

    @pytest.mark.parametrize("method", ["inv", "inv_ksg"])
    def test_matrices_nan_only_for_that_dimension(self, data, method):
        w, x, y, z = data
        mi_mat = MI(np.column_stack([w, x, y]), method=method)
        assert np.isnan(mi_mat[0, :]).all() and np.isnan(mi_mat[:, 0]).all()
        np.testing.assert_allclose(
            mi_mat[1:, 1:], MI(np.column_stack([x, y]), method=method), atol=1e-12
        )

        cmi_mat = CMI(np.column_stack([w, x, y]), z, method=method)
        assert np.isnan(cmi_mat[0, :]).all() and np.isnan(cmi_mat[:, 0]).all()
        np.testing.assert_allclose(
            cmi_mat[1:, 1:], CMI(np.column_stack([x, y]), z, method=method), atol=1e-12
        )
        assert np.isnan(CMI(np.column_stack([x, y]), w, method=method)).all()

    def test_parallel_matches_sequential(self, data):
        w, x, y, z = data
        cols = np.column_stack([w, x, y])
        np.testing.assert_array_equal(MI(cols, n_jobs=2), MI(cols, n_jobs=1))
        np.testing.assert_array_equal(CMI(cols, z, n_jobs=2), CMI(cols, z, n_jobs=1))

    def test_self_mi_and_coalitions(self, data):
        w, x, _, z = data
        assert np.isnan(mutual_information(w, w))
        coalitions = coalition_mutual_information(np.column_stack([w, x]), z)
        assert np.isnan(coalitions[0b01]) and np.isnan(coalitions[0b11])
        assert np.isfinite(coalitions[0b10])
