"""Tests for optimized matrix functions."""

import numpy as np
import pytest
from entropy_invariant import MI, CMI, mutual_information
from entropy_invariant.advanced import conditional_mutual_information
from entropy_invariant.helpers.computation import compute_invariant_measure


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


class TestMatrixScalarConsistency:
    """
    Regression tests for the bug where MI()/CMI() (matrix fast-path)
    duplicated the invariant-measure computation inline instead of reusing
    compute_invariant_measure(), silently skipping its zero-filtering. On
    sparse data (mostly zeros, common in real signals like mass-spec bins)
    this made the fast-path diverge from -- and eventually crash relative to
    -- the scalar mutual_information()/conditional_mutual_information()
    functions, which always went through the correct, zero-filtered helper.

    These tests assert the two APIs agree on sparse data for both methods,
    so any future reintroduction of duplicated/inconsistent normalization
    logic fails loudly here instead of shipping silently.
    """

    def test_mi_matrix_matches_scalar_on_sparse_data(self, sparse_test_data):
        # method="inv": no shared radius involved, so heavy (~80%) sparsity
        # on both x and y (no conditioning z to break ties) is fine here.
        x, y, _ = sparse_test_data
        data = np.column_stack([x, y])

        mi_mat = MI(data, method="inv", k=5)
        mi_direct = mutual_information(x, y, method="inv", k=5)

        assert np.isfinite(mi_mat).all()
        assert mi_mat[0, 1] == pytest.approx(mi_direct, abs=1e-9)

    def test_mi_matrix_matches_scalar_on_sparse_data_ksg(self, mildly_sparse_pair):
        # method="inv_ksg": with no conditioning z to break ties, x and y
        # both being heavily (~80%) sparse would make the shared KSG radius
        # legitimately degenerate (see TestDegenerateKsgRadius) -- that's
        # correct behavior, not a bug, so this uses lighter (~10%) sparsity
        # instead to test matrix/scalar consistency in the non-degenerate
        # regime.
        x, y = mildly_sparse_pair
        data = np.column_stack([x, y])

        mi_mat = MI(data, method="inv_ksg", k=5)
        mi_direct = mutual_information(x, y, method="inv_ksg", k=5)

        assert np.isfinite(mi_mat).all()
        assert mi_mat[0, 1] == pytest.approx(mi_direct, abs=1e-9)

    @pytest.mark.parametrize("method", ["inv", "inv_ksg"])
    def test_cmi_matrix_matches_scalar_on_sparse_data(self, sparse_test_data, method):
        x, y, z = sparse_test_data
        data = np.column_stack([x, y])

        cmi_mat = CMI(data, z, method=method, k=5)
        cmi_direct = conditional_mutual_information(x, y, z, method=method, k=5)

        assert np.isfinite(cmi_mat).all()
        assert cmi_mat[0, 1] == pytest.approx(cmi_direct, abs=1e-9)


class TestDegenerateInvariantMeasure:
    """Tests that a degenerate invariant measure fails loudly and clearly."""

    @pytest.fixture
    def duplicate_heavy_column(self):
        """A column where >=half the non-zero values are exact duplicates,
        so the median nearest-neighbor distance is exactly 0."""
        np.random.seed(0)
        return np.random.choice([1.0, 2.0, 3.0], size=200)

    def test_compute_invariant_measure_raises_on_degenerate_data(
        self, duplicate_heavy_column
    ):
        with pytest.raises(ValueError, match="degenerate"):
            compute_invariant_measure(duplicate_heavy_column)

    def test_mi_matrix_raises_clear_error_on_degenerate_column(
        self, duplicate_heavy_column, simple_test_data
    ):
        x, _ = simple_test_data
        data = np.column_stack([duplicate_heavy_column, x[: len(duplicate_heavy_column)]])

        with pytest.raises(ValueError, match="degenerate"):
            MI(data)

    def test_cmi_matrix_raises_clear_error_on_degenerate_column(
        self, duplicate_heavy_column, simple_test_data
    ):
        x, _ = simple_test_data
        n = len(duplicate_heavy_column)
        data = np.column_stack([duplicate_heavy_column, x[:n]])
        z = np.random.rand(n)

        with pytest.raises(ValueError, match="degenerate"):
            CMI(data, z)


class TestDegenerateKsgRadius:
    """
    Tests that a degenerate shared KSG radius fails loudly and clearly,
    instead of silently propagating digamma(0) = -inf into NaN.

    This happens when >=k+1 points coincide exactly in the joint space that
    the shared radius is computed over -- e.g. two variables that are both
    heavily sparse (many exact zeros) with no third, tie-breaking variable.
    In real usage this is rare for CMI (a conditioning variable such as a
    sum over many channels is virtually never exactly 0), but easy to hit
    for a plain two-variable MI on sparse data.
    """

    @pytest.fixture
    def doubly_sparse_pair(self):
        """x, y both ~80% zero at independently-chosen positions: by the
        pigeonhole principle, at least 300/500 points must be (0, 0)."""
        np.random.seed(3)
        n = 500

        def make_column():
            col = np.zeros(n)
            nonzero_idx = np.random.choice(n, size=n // 5, replace=False)
            col[nonzero_idx] = np.random.rand(len(nonzero_idx)) * 10 + 1.0
            return col

        return make_column(), make_column()

    def test_mi_ksg_raises_clear_error_on_degenerate_radius(self, doubly_sparse_pair):
        from entropy_invariant.ksg import mutual_information_ksg

        x, y = doubly_sparse_pair
        with pytest.raises(ValueError, match="degenerate"):
            mutual_information_ksg(x, y, k=5)

    def test_mi_matrix_raises_clear_error_on_degenerate_radius(self, doubly_sparse_pair):
        x, y = doubly_sparse_pair
        data = np.column_stack([x, y])

        with pytest.raises(ValueError, match="degenerate"):
            MI(data, method="inv_ksg", k=5)

    def test_cmi_matrix_raises_clear_error_on_degenerate_radius(self, doubly_sparse_pair):
        x, y = doubly_sparse_pair
        n = len(x)
        # z is also sparse here (unlike sparse_test_data's dense z), so the
        # full (x, y, z) joint space is degenerate too.
        z = np.zeros(n)
        z[np.random.choice(n, size=n // 5, replace=False)] = np.random.rand(n // 5) * 10 + 1.0
        data = np.column_stack([x, y])

        with pytest.raises(ValueError, match="degenerate"):
            CMI(data, z, method="inv_ksg", k=5)


class TestParallelExecution:
    """
    n_jobs (method="inv_ksg" only): multiprocessing.Pool over pairs instead
    of requiring users to hand-roll it (as we did ad hoc for the JASM
    analysis this generalizes). Regardless of n_jobs, each dimension's
    marginal/(Xi,Z) tree is built once and reused across all pairs -- so
    these tests also cover that the tree-reuse refactor didn't change
    results relative to the original per-pair tree rebuilding.
    """

    @pytest.fixture
    def matrix_test_data(self):
        np.random.seed(99)
        n = 300
        data = np.random.rand(n, 6)
        z = np.random.rand(n)
        return data, z

    @pytest.mark.parametrize("n_jobs", [2, -1])
    def test_mi_matrix_n_jobs_matches_sequential(self, matrix_test_data, n_jobs):
        data, _ = matrix_test_data
        mi_seq = MI(data, method="inv_ksg", k=4, n_jobs=1)
        mi_par = MI(data, method="inv_ksg", k=4, n_jobs=n_jobs)
        assert np.allclose(mi_seq, mi_par, atol=1e-10)

    @pytest.mark.parametrize("n_jobs", [2, -1])
    def test_cmi_matrix_n_jobs_matches_sequential(self, matrix_test_data, n_jobs):
        data, z = matrix_test_data
        cmi_seq = CMI(data, z, method="inv_ksg", k=4, n_jobs=1)
        cmi_par = CMI(data, z, method="inv_ksg", k=4, n_jobs=n_jobs)
        assert np.allclose(cmi_seq, cmi_par, atol=1e-10)

    def test_mi_matrix_n_jobs_ignored_for_inv(self, matrix_test_data):
        data, _ = matrix_test_data
        mi_seq = MI(data, method="inv", k=4, n_jobs=1)
        mi_par = MI(data, method="inv", k=4, n_jobs=2)
        assert np.allclose(mi_seq, mi_par, atol=1e-10)
