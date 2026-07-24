"""Tests for Partial Information Decomposition functions."""

import numpy as np
import pytest
from entropy_invariant import entropy, redundancy, unique, synergy
from entropy_invariant.pid import _joint_mutual_information


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

    def test_reconstruction_identity_inv(self, simple_test_data):
        """
        R + U_X + U_Y + Synergy must equal the joint mutual information
        I({X,Y};Z) -- that's the defining property of a partial information
        decomposition. This is the check that catches the historical bug
        where synergy() used conditional_mutual_information(X,Y,Z)
        (= I(X;Y|Z)) in place of the joint MI I({X,Y};Z); a
        finite/type-only check (as in test_synergy_basic above) cannot
        detect that kind of error, only a correctness check can.

        I_joint is computed independently here via the joint-entropy chain
        rule directly (not via synergy()'s own internal helper), so this
        check is not circular with the implementation.
        """
        x, y = simple_test_data
        rng = np.random.default_rng(0)
        z = rng.random(len(x))

        r = redundancy(x, y, z, method="inv")
        ux, uy = unique(x, y, z, method="inv")
        s = synergy(x, y, z, method="inv")

        xy_joint = np.vstack([x, y])
        z_row = z.reshape(1, -1)
        i_joint_independent = (
            entropy(xy_joint, method="inv", dim=2)
            + entropy(z_row, method="inv", dim=2)
            - entropy(np.vstack([xy_joint, z_row]), method="inv", dim=2)
        )

        assert abs((r + ux + uy + s) - i_joint_independent) < 1e-6

    def test_reconstruction_identity_inv_ksg(self, simple_test_data):
        """
        Same reconstruction identity as test_reconstruction_identity_inv,
        for the default method="inv_ksg". No independent public code path
        exists to compute I({X,Y};Z) for this method (that's exactly why
        synergy() needed a new internal helper), so this re-checks internal
        self-consistency rather than independence -- still catches any
        regression that breaks the identity, just not a bug shared between
        synergy() and its own helper.
        """
        x, y = simple_test_data
        rng = np.random.default_rng(0)
        z = rng.random(len(x))

        r = redundancy(x, y, z)
        ux, uy = unique(x, y, z)
        s = synergy(x, y, z)

        i_joint = _joint_mutual_information(
            x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1),
            method="inv_ksg", nbins=10, k=3, base=np.e, degenerate=False,
        )

        assert abs((r + ux + uy + s) - i_joint) < 1e-6
