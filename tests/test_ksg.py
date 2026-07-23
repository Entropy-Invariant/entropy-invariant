"""Tests for the invariant-measure + KSG/Frenzel-Pompe estimators."""

import numpy as np
import pytest
from entropy_invariant import (
    mutual_information,
    mutual_information_ksg,
    conditional_mutual_information,
    conditional_mutual_information_ksg,
    conditional_entropy,
    normalized_mutual_information,
    interaction_information,
    information_quality_ratio,
    redundancy,
    unique,
    synergy,
    MI,
    CMI,
)


class TestMutualInformationKSG:
    """Tests for the KSG mutual information estimator."""

    def test_mi_ksg_basic(self, simple_test_data):
        x, y = simple_test_data
        mi = mutual_information_ksg(x, y)
        assert isinstance(mi, float)
        assert np.isfinite(mi)

    def test_mi_ksg_symmetric(self, simple_test_data):
        """I(X;Y) = I(Y;X)."""
        x, y = simple_test_data
        assert abs(mutual_information_ksg(x, y) - mutual_information_ksg(y, x)) < 1e-10

    def test_mi_ksg_scale_invariant(self, simple_test_data):
        """Rescaling either variable independently must not change MI."""
        x, y = simple_test_data
        mi = mutual_information_ksg(x, y)
        mi_scaled = mutual_information_ksg(1e6 * x - 99, 1e-6 * y + 5)
        assert abs(mi - mi_scaled) < 1e-9

    def test_mi_ksg_matches_plugin_roughly(self, simple_test_data):
        """KSG and the plug-in ('inv') estimator should agree in sign and rough magnitude."""
        x, y = simple_test_data
        mi_plugin = mutual_information(x, y, method="inv")
        mi_ksg = mutual_information_ksg(x, y)
        assert abs(mi_plugin - mi_ksg) < 0.5

    def test_mi_dispatch_via_method_string(self, simple_test_data):
        """mutual_information(..., method='inv_ksg') must match the direct call."""
        x, y = simple_test_data
        assert mutual_information(x, y, method="inv_ksg") == mutual_information_ksg(x, y)

    def test_mi_ksg_gaussian_closed_form(self):
        """Bivariate Gaussian: I(X;Y) = -0.5*log(1-rho^2)."""
        np.random.seed(1)
        n = 5000
        rho = 0.6
        cov = np.array([[1, rho], [rho, 1]])
        xy = np.random.multivariate_normal([0, 0], cov, n)
        true_mi = -0.5 * np.log(1 - rho**2)
        mi = mutual_information_ksg(xy[:, 0], xy[:, 1])
        assert abs(mi - true_mi) < 0.05

    def test_mi_ksg_independent_near_zero(self):
        np.random.seed(2)
        n = 3000
        x = np.random.randn(n)
        y = np.random.randn(n)
        assert abs(mutual_information_ksg(x, y)) < 0.1

    def test_mi_ksg_multidimensional_error(self):
        x = np.random.rand(100, 2)
        y = np.random.rand(100, 2)
        with pytest.raises(ValueError, match="1-dimensional"):
            mutual_information_ksg(x, y)


class TestConditionalMutualInformationKSG:
    """Tests for the Frenzel-Pompe conditional mutual information estimator."""

    def test_cmi_ksg_basic(self, simple_test_data):
        x, y = simple_test_data
        z = np.random.rand(len(x))
        cmi = conditional_mutual_information_ksg(x, y, z)
        assert isinstance(cmi, float)
        assert np.isfinite(cmi)

    def test_cmi_ksg_scale_invariant(self):
        """Rescaling X, Y, Z independently must not change CMI."""
        np.random.seed(0)
        n = 2000
        z = np.random.randn(n)
        x = z + 0.3 * np.random.randn(n)
        y = z + 0.3 * np.random.randn(n)

        cmi = conditional_mutual_information_ksg(x, y, z)
        cmi_scaled = conditional_mutual_information_ksg(1e6 * x - 99, 1e-6 * y, 1e3 * z)
        assert abs(cmi - cmi_scaled) < 1e-9

    def test_cmi_ksg_chain_near_zero(self):
        """Chain X -> Z -> Y: I(X;Y|Z) should be close to 0."""
        np.random.seed(42)
        n = 3000
        x = np.random.randn(n)
        z = x + 0.3 * np.random.randn(n)
        y = z + 0.3 * np.random.randn(n)
        assert abs(conditional_mutual_information_ksg(x, y, z)) < 0.1

    def test_cmi_ksg_collider_positive(self):
        """Collider X -> Z <- Y: I(X;Y|Z) should be clearly positive."""
        np.random.seed(42)
        n = 3000
        x = np.random.randn(n)
        y = np.random.randn(n)
        z = x + y + 0.3 * np.random.randn(n)
        assert conditional_mutual_information_ksg(x, y, z) > 0.3

    def test_cmi_dispatch_via_method_string(self):
        np.random.seed(0)
        n = 1000
        z = np.random.randn(n)
        x = z + 0.3 * np.random.randn(n)
        y = z + 0.3 * np.random.randn(n)
        assert conditional_mutual_information(
            x, y, z, method="inv_ksg"
        ) == conditional_mutual_information_ksg(x, y, z)

    def test_cmi_ksg_robust_to_outliers_vs_plugin(self):
        """On outlier-contaminated fork data (CMI should stay ~0), the KSG estimator
        should not drift as far from 0 as the naive plug-in formula does."""
        np.random.seed(42)
        n = 3000
        z_clean = np.random.randn(n)
        x_clean = z_clean + 0.3 * np.random.randn(n)
        y_clean = z_clean + 0.3 * np.random.randn(n)

        x, y, z = x_clean.copy(), y_clean.copy(), z_clean.copy()
        n_spikes = int(0.20 * n)
        for arr in [x, y, z]:
            idx = np.random.choice(n, n_spikes, replace=False)
            arr[idx] += 50 * np.random.choice([-1, 1], size=n_spikes)

        cmi_plugin = conditional_mutual_information(x, y, z, method="inv")
        cmi_ksg = conditional_mutual_information_ksg(x, y, z)
        assert abs(cmi_ksg) < abs(cmi_plugin)

    def test_cmi_ksg_multidimensional_error(self):
        x = np.random.rand(100, 2)
        y = np.random.rand(100)
        z = np.random.rand(100)
        with pytest.raises(ValueError, match="1-dimensional"):
            conditional_mutual_information_ksg(x, y, z)


class TestInvKsgIsDefault:
    """`inv_ksg` is the default `method` for every MI/CMI-derived quantity."""

    def test_mi_default_matches_ksg(self, simple_test_data):
        x, y = simple_test_data
        assert mutual_information(x, y) == mutual_information_ksg(x, y)

    def test_cmi_default_matches_ksg(self, simple_test_data):
        x, y = simple_test_data
        z = np.random.rand(len(x))
        assert conditional_mutual_information(x, y, z) == conditional_mutual_information_ksg(x, y, z)


class TestConditionalEntropyKSG:
    def test_matches_H_minus_MI(self, simple_test_data):
        """H(Y|X) via inv_ksg should equal H(Y) - I(X;Y)_ksg exactly (it's defined that way)."""
        from entropy_invariant import entropy

        x, y = simple_test_data
        h_yx = conditional_entropy(x, y)
        expected = entropy(y, method="inv") - mutual_information_ksg(x, y)
        assert abs(h_yx - expected) < 1e-10

    def test_scale_invariant(self, simple_test_data):
        x, y = simple_test_data
        h1 = conditional_entropy(x, y)
        h2 = conditional_entropy(1e6 * x - 5, 1e-6 * y + 3)
        assert abs(h1 - h2) < 1e-5


class TestNormalizedMutualInformationKSG:
    def test_symmetric(self, simple_test_data):
        x, y = simple_test_data
        assert abs(
            normalized_mutual_information(x, y) - normalized_mutual_information(y, x)
        ) < 1e-10

    def test_scale_invariant(self, simple_test_data):
        x, y = simple_test_data
        n1 = normalized_mutual_information(x, y)
        n2 = normalized_mutual_information(1e6 * x - 5, 1e-6 * y + 3)
        assert abs(n1 - n2) < 1e-5


class TestInteractionInformationKSG:
    def test_matches_MI_minus_CMI(self):
        """II(X;Y;Z) via inv_ksg should equal I(X;Y) - I(X;Y|Z) exactly."""
        np.random.seed(0)
        n = 1000
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = np.random.rand(n)

        ii = interaction_information(x, y, z)
        expected = mutual_information_ksg(x, y) - conditional_mutual_information_ksg(x, y, z)
        assert abs(ii - expected) < 1e-10

    def test_scale_invariant(self):
        np.random.seed(0)
        n = 1000
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = np.random.rand(n)

        ii1 = interaction_information(x, y, z)
        ii2 = interaction_information(1e6 * x - 5, 1e-6 * y + 3, 1e3 * z)
        assert abs(ii1 - ii2) < 1e-5


class TestInformationQualityRatioKSG:
    def test_scale_invariant(self, simple_test_data):
        x, y = simple_test_data
        i1 = information_quality_ratio(x, y)
        i2 = information_quality_ratio(1e6 * x - 5, 1e-6 * y + 3)
        assert abs(i1 - i2) < 1e-5


class TestPIDKSGDefault:
    """PID functions delegate to mutual_information / conditional_mutual_information,
    so they inherit the inv_ksg default automatically."""

    def test_redundancy_matches_ksg_mi(self, simple_test_data):
        x, y = simple_test_data
        z = np.random.rand(len(x))
        r = redundancy(x, y, z)
        expected = min(mutual_information_ksg(x, z), mutual_information_ksg(y, z))
        assert abs(r - expected) < 1e-10

    def test_synergy_finite_and_scale_invariant(self, simple_test_data):
        x, y = simple_test_data
        z = np.random.rand(len(x))
        s1 = synergy(x, y, z)
        assert np.isfinite(s1)
        s2 = synergy(1e6 * x - 5, 1e-6 * y + 3, 1e3 * z)
        assert abs(s1 - s2) < 1e-4


class TestMIMatrixKSG:
    def test_default_is_inv_ksg(self, simple_test_data):
        x, y = simple_test_data
        data = np.column_stack([x, y])
        mi_mat = MI(data)
        assert abs(mi_mat[0, 1] - mutual_information_ksg(x, y)) < 1e-10

    def test_symmetric_and_finite(self, simple_test_data):
        x, y = simple_test_data
        z = np.random.rand(len(x))
        data = np.column_stack([x, y, z])
        mi_mat = MI(data)
        assert np.isfinite(mi_mat).all()
        assert np.allclose(mi_mat, mi_mat.T)

    def test_inv_method_still_available(self, simple_test_data):
        x, y = simple_test_data
        data = np.column_stack([x, y])
        mi_mat = MI(data, method="inv")
        assert np.isfinite(mi_mat).all()

    def test_invalid_method_raises(self, simple_test_data):
        x, y = simple_test_data
        data = np.column_stack([x, y])
        with pytest.raises(ValueError, match="Invalid method"):
            MI(data, method="bogus")


class TestCMIMatrixKSG:
    def test_default_is_inv_ksg(self, simple_test_data):
        x, y = simple_test_data
        z = np.random.rand(len(x))
        data = np.column_stack([x, y])
        cmi_mat = CMI(data, z)
        assert abs(cmi_mat[0, 1] - conditional_mutual_information_ksg(x, y, z)) < 1e-10

    def test_symmetric_and_finite(self, simple_test_data):
        x, y = simple_test_data
        z = np.random.rand(len(x))
        data = np.column_stack([x, y])
        cmi_mat = CMI(data, z)
        assert np.isfinite(cmi_mat).all()
        assert np.allclose(cmi_mat, cmi_mat.T)

    def test_inv_method_still_available(self, simple_test_data):
        x, y = simple_test_data
        z = np.random.rand(len(x))
        data = np.column_stack([x, y])
        cmi_mat = CMI(data, z, method="inv")
        assert np.isfinite(cmi_mat).all()
