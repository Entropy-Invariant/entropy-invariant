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
from entropy_invariant.helpers.utility import nn1


class TestNoScaleIsNaN:
    """
    Fewer than two values that occur once: there is no spacing to measure, so
    no scale. This used to return 1.0, which left the column in its own units:
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


class TestRepeatedValues:
    """
    A repeated value has a nearest-neighbour distance of 0, whatever the
    value is. Setting aside the literal value 0 instead made the scale depend
    on location: 0:6 and 1:7 have the same spacing, but r was 6 and 7.
    """

    @staticmethod
    def make_col(rng, n, frac_nonzero):
        col = np.zeros(n)
        idx = rng.permutation(n)[: round(n * frac_nonzero)]
        col[idx] = rng.random(len(idx)) * 10 + 1.0
        return col

    def test_zero_is_not_special(self):
        x = np.arange(7.0)
        assert compute_invariant_measure(x) == 7.0
        assert compute_invariant_measure(x + 1) == 7.0
        assert abs(entropy(x, k=3) - entropy(x + 1, k=3)) < 1e-12

    @pytest.mark.parametrize("frac", [0.2, 0.98])
    def test_sparse_scale_unchanged(self, frac):
        # Sparse data, where the only duplicate is 0: same scale as setting
        # aside zeros, bit for bit.
        x = self.make_col(np.random.default_rng(5), 500, frac)
        nz = x[x != 0]
        assert compute_invariant_measure(x) == float(np.median(nn1(np.sort(nz))) * len(nz))

    def test_any_repeated_value_is_set_aside(self):
        # Here a saturation level as well as 0.
        x = self.make_col(np.random.default_rng(6), 500, 0.2)
        x[np.flatnonzero(x)[:10]] = 11.0
        once = x[(x != 0) & (x != 11.0)]
        assert compute_invariant_measure(x) == pytest.approx(
            np.median(nn1(np.sort(once))) * len(once), rel=1e-12
        )

    @pytest.mark.parametrize("a, b", [(1.0, 1.0), (-3.5, 2.0), (1e3, -7.0)])
    def test_affine_equivariance(self, a, b):
        # Including reflections, and shifts that move the sparse atom.
        rng = np.random.default_rng(7)
        saturated = self.make_col(rng, 500, 0.2)
        saturated[np.flatnonzero(saturated)[:10]] = 11.0
        lone_zero = np.append(rng.standard_normal(999), 0.0)
        for v in (saturated, self.make_col(rng, 500, 0.98), lone_zero, np.arange(7.0)):
            assert compute_invariant_measure(a * v + b) == pytest.approx(
                abs(a) * compute_invariant_measure(v), rel=1e-9
            )
            assert abs(entropy(a * v + b, k=3) - entropy(v, k=3)) < 1e-8

    def test_celsius_fahrenheit(self):
        # Celsius readings including a genuine 0.0 agree with Fahrenheit directly.
        rng = np.random.default_rng(8)
        celsius = np.append(rng.standard_normal(999) * 5 + 2, 0.0)
        assert abs(entropy(celsius, k=3) - entropy(1.8 * celsius + 32, k=3)) < 1e-8

    def test_negative_zero_equals_zero(self):
        assert compute_invariant_measure(np.array([-0.0, 0.0, 1.0, 2.5, 4.0])) == 3 * 1.5

    def test_spread_duplicates_fail_loudly(self):
        # Duplicates spread over many values (discrete or coarsely rounded
        # data) still fail loudly, wherever the data sits.
        rng = np.random.default_rng(9)
        discrete = rng.choice([1.0, 2.0, 3.0], size=200)
        for v in (discrete, discrete + 10, -2 * discrete, np.round(rng.standard_normal(10_000), 2)):
            with pytest.raises(ValueError, match="degenerate"):
                compute_invariant_measure(v)

    def test_too_few_single_values_is_nan_wherever_the_atom_sits(self):
        assert np.isnan(compute_invariant_measure(np.array([0.0, 0, 0, 0, 7])))
        assert np.isnan(compute_invariant_measure(np.array([5.0, 5, 5, 5, 7])))


class TestFactorN:
    """
    r_X = n * median(NN distance). The median alone shrinks like 1/n and would
    add log(n) to every entropy. For U(0,1), n * median(NN distance) tends to
    ln(2)/2, so the invariant entropy tends to -log(ln(2)/2) = 1.0597, the
    Uniform row of Table 2 (1.060).
    """

    def test_uniform_matches_published_table_2(self):
        x = np.random.default_rng(3).random(50_000)
        assert abs(entropy(x, k=3) - (-np.log(np.log(2) / 2))) < 0.03
