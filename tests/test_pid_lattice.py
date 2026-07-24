"""
Tests for the N-source PID redundancy lattice.

The reconstruction identity ``sum(atoms) == I_cap(top)`` is NOT used as a test: Moebius
inversion makes it hold for any redundancy function, correct or not, so it cannot detect a
wrong lattice. These tests instead check published atom values on discrete toy
distributions, the Williams & Beer non-negativity theorem, and exact agreement with the
existing two-source ``redundancy``/``unique``.

Values are kept identical to the Julia package's test suite so the two ports stay in sync.
"""

import numpy as np
import pytest

from entropy_invariant import (
    coalition_mutual_information,
    iccs_redundancy,
    imin_redundancy,
    isotonic_repair,
    lattice_labels,
    mmi_redundancy,
    moebius_atoms,
    pid_lattice,
    redundancy,
    redundancy_lattice,
    unique,
)


def _pmf_and():
    p = np.zeros((2, 2, 2))
    p[0, 0, 0] = p[0, 1, 0] = p[1, 0, 0] = 0.25
    p[1, 1, 1] = 0.25
    return p


def _pmf_xor():
    p = np.zeros((2, 2, 2))
    p[0, 0, 0] = p[0, 1, 1] = p[1, 0, 1] = p[1, 1, 0] = 0.25
    return p


def _pmf_copy():
    p = np.zeros((2, 2, 4))
    p[0, 0, 0] = p[0, 1, 1] = p[1, 0, 2] = p[1, 1, 3] = 0.25
    return p


def _pmf_xor3():
    p = np.zeros((2, 2, 2, 2))
    for x in (0, 1):
        for y in (0, 1):
            for w in (0, 1):
                p[x, y, w, x ^ y ^ w] = 0.125
    return p


class TestLatticeStructure:
    """The combinatorics, independent of any data."""

    @pytest.mark.parametrize("n,expected", [(1, 1), (2, 4), (3, 18), (4, 166)])
    def test_node_counts(self, n, expected):
        assert len(redundancy_lattice(n).nodes) == expected

    def test_rejects_unsupported_sizes(self):
        with pytest.raises(ValueError):
            redundancy_lattice(5)
        with pytest.raises(ValueError):
            redundancy_lattice(0)

    def test_two_source_labels_and_order(self):
        lat = redundancy_lattice(2)
        labels = lattice_labels(lat, ["X", "Y"])
        assert sorted(labels) == sorted(["{X}{Y}", "{X}", "{Y}", "{XY}"])
        bottom = labels.index("{X}{Y}")
        top = labels.index("{XY}")
        assert lat.predecessors[bottom] == []       # nothing precedes the bottom
        assert len(lat.predecessors[top]) == 3      # everything precedes the top

    def test_label_separator_and_validation(self):
        lat = redundancy_lattice(2)
        labels = lattice_labels(lat, ["Ab", "Cd"], sep=".")
        assert "{Ab.Cd}" in labels
        with pytest.raises(ValueError):
            lattice_labels(lat, ["X"])


class TestIminPublishedValues:
    """Williams & Beer's own worked examples, in bits."""

    def test_and(self):
        a = pid_lattice(_pmf_and(), names=["X", "Y"])
        assert a["{X}{Y}"] == pytest.approx(0.311278, abs=1e-5)
        assert a["{X}"] == pytest.approx(0.0, abs=1e-9)
        assert a["{Y}"] == pytest.approx(0.0, abs=1e-9)
        assert a["{XY}"] == pytest.approx(0.5, abs=1e-5)
        assert sum(a.values()) == pytest.approx(0.811278, abs=1e-5)

    def test_xor_is_pure_synergy(self):
        a = pid_lattice(_pmf_xor(), names=["X", "Y"])
        assert a["{X}{Y}"] == pytest.approx(0.0, abs=1e-9)
        assert a["{X}"] == pytest.approx(0.0, abs=1e-9)
        assert a["{Y}"] == pytest.approx(0.0, abs=1e-9)
        assert a["{XY}"] == pytest.approx(1.0, abs=1e-9)

    def test_copy_reproduces_imins_known_flaw(self):
        """
        I_min's documented failure mode (Harder, Salge & Polani 2013): on two-bit COPY it
        calls two INDEPENDENT bits fully redundant, R = 1, where other measures give
        R = 0, U_X = U_Y = 1. Asserting the flaw confirms this really is I_min.
        """
        a = pid_lattice(_pmf_copy(), names=["X", "Y"])
        assert a["{X}{Y}"] == pytest.approx(1.0, abs=1e-9)
        assert a["{X}"] == pytest.approx(0.0, abs=1e-9)
        assert a["{Y}"] == pytest.approx(0.0, abs=1e-9)
        assert sum(a.values()) == pytest.approx(2.0, abs=1e-9)

    def test_three_way_xor(self):
        a = pid_lattice(_pmf_xor3(), names=["1", "2", "3"])
        assert len(a) == 18
        assert a["{123}"] == pytest.approx(1.0, abs=1e-9)
        assert max(abs(v) for k, v in a.items() if k != "{123}") < 1e-9

    def test_nonnegativity_theorem_on_random_distributions(self):
        """
        W&B proved I_min yields non-negative atoms. A wrong lattice order or Moebius
        traversal fails this almost surely, which makes it the strongest structural test.
        """
        rng = np.random.default_rng(20260725)
        lat = redundancy_lattice(3)
        worst = 0.0
        for _ in range(100):
            p = rng.random((2, 2, 2, 3))
            p /= p.sum()
            atoms = moebius_atoms(lat, imin_redundancy(p))
            worst = min(worst, atoms.min())
            # the top node's cumulative redundancy is I({all sources}; Z)
            assert atoms.sum() == pytest.approx(imin_redundancy(p)((7,)), abs=1e-9)
        assert worst > -1e-9


class TestIccs:
    """Ince (2017), pointwise common change in surprisal."""

    def test_and_exact(self):
        """
        The redundancy is carried entirely by the (x=0,y=0,z=0) realisation, where all
        three local terms equal log2(4/3); the two mixed realisations are discarded
        because the local source-target MIs disagree in sign. So R = 0.25*log2(4/3).
        """
        r_exact = 0.25 * np.log2(4.0 / 3.0)
        a = pid_lattice(_pmf_and(), measure="iccs", names=["X", "Y"])
        assert a["{X}{Y}"] == pytest.approx(r_exact, abs=1e-9)
        assert a["{X}"] == pytest.approx(0.3112781 - r_exact, abs=1e-6)
        assert a["{Y}"] == pytest.approx(0.3112781 - r_exact, abs=1e-6)
        assert sum(a.values()) == pytest.approx(0.8112781, abs=1e-6)

    def test_credits_less_redundancy_than_imin(self):
        ccs = pid_lattice(_pmf_and(), measure="iccs", names=["X", "Y"])
        imin = pid_lattice(_pmf_and(), measure="imin", names=["X", "Y"])
        assert ccs["{X}{Y}"] < imin["{X}{Y}"]

    def test_xor_is_pure_synergy(self):
        a = pid_lattice(_pmf_xor(), measure="iccs", names=["X", "Y"])
        assert a["{X}{Y}"] == pytest.approx(0.0, abs=1e-9)
        assert a["{XY}"] == pytest.approx(1.0, abs=1e-9)

    def test_copy_is_corrected(self):
        """
        This is why I_ccs is worth having. Two INDEPENDENT bits copied to the target must
        decompose as R = 0, U_X = U_Y = 1, Syn = 0. I_min instead reports them as fully
        redundant; I_ccs gets it right.
        """
        a = pid_lattice(_pmf_copy(), measure="iccs", names=["X", "Y"])
        assert a["{X}{Y}"] == pytest.approx(0.0, abs=1e-9)
        assert a["{X}"] == pytest.approx(1.0, abs=1e-9)
        assert a["{Y}"] == pytest.approx(1.0, abs=1e-9)
        assert a["{XY}"] == pytest.approx(0.0, abs=1e-9)
        imin = pid_lattice(_pmf_copy(), measure="imin", names=["X", "Y"])
        assert imin["{X}{Y}"] == pytest.approx(1.0, abs=1e-9)   # the measure it corrects

    def test_three_way_xor(self):
        a = pid_lattice(_pmf_xor3(), measure="iccs", names=["1", "2", "3"])
        assert len(a) == 18
        assert a["{123}"] == pytest.approx(1.0, abs=1e-9)
        assert max(abs(v) for k, v in a.items() if k != "{123}") < 1e-9

    def test_self_redundancy(self):
        """A single-coalition node must return I(X_A; Z) itself."""
        ccs = iccs_redundancy(_pmf_and())
        assert ccs((1,)) == pytest.approx(0.3112781, abs=1e-6)
        assert ccs((3,)) == pytest.approx(0.8112781, abs=1e-6)

    def test_rejects_unknown_measure(self):
        with pytest.raises(ValueError):
            pid_lattice(_pmf_and(), measure="nonsense")


class TestMmiAgainstExistingTwoSourceAPI:
    def test_matches_redundancy_and_unique_exactly(self):
        rng = np.random.default_rng(4242)
        n = 400
        z = rng.random(n)
        x = z + 0.5 * rng.random(n)      # correlated with the target
        y = z + 0.9 * rng.random(n)      # correlated, but less so
        a = pid_lattice([x, y], z, names=["X", "Y"], repair="none", k=3)
        r_pkg = redundancy(x, y, z, method="inv_ksg", k=3)
        ux_pkg, uy_pkg = unique(x, y, z, method="inv_ksg", k=3)
        assert a["{X}{Y}"] == pytest.approx(r_pkg, abs=1e-9)
        assert a["{X}"] == pytest.approx(ux_pkg, abs=1e-9)
        assert a["{Y}"] == pytest.approx(uy_pkg, abs=1e-9)
        cmi = coalition_mutual_information([x, y], z, k=3)
        assert sum(a.values()) == pytest.approx(cmi[3], abs=1e-9)


class TestIsotonicRepair:
    def test_enforces_monotonicity(self):
        # the pair is estimated BELOW one of its members, which true mutual information
        # can never be but kNN estimators regularly produce
        bad = {1: 0.50, 2: 0.20, 3: 0.30}
        for mode in ("isotonic", "majorant"):
            fixed = isotonic_repair(bad, mode=mode)
            assert fixed[3] >= fixed[1] - 1e-12
            assert fixed[3] >= fixed[2] - 1e-12

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            isotonic_repair({1: 0.1, 2: 0.2, 3: 0.3}, mode="nonsense")

    def test_leaves_monotone_input_untouched(self):
        good = {1: 0.10, 2: 0.20, 3: 0.35}
        fixed = isotonic_repair(good)
        for key, value in good.items():
            assert fixed[key] == pytest.approx(value, abs=1e-12)

    def test_does_not_clamp_negatives(self):
        """
        An estimate near a true zero is symmetric noise; truncating it would bias
        low-signal regions upward.
        """
        negs = {1: -0.05, 2: -0.03, 3: -0.01}
        fixed = isotonic_repair(negs)
        assert fixed[1] < 0.0
        assert fixed[3] >= fixed[1] - 1e-12 and fixed[3] >= fixed[2] - 1e-12

    def test_removes_impossible_negative_atoms(self):
        bad = {1: 0.50, 2: 0.20, 3: 0.30}
        lat = redundancy_lattice(2)
        labels = lattice_labels(lat, ["X", "Y"])
        atoms = moebius_atoms(lat, mmi_redundancy(isotonic_repair(bad)))
        d = dict(zip(labels, atoms))
        assert d["{X}"] >= -1e-12
        assert d["{Y}"] >= -1e-12
        assert d["{XY}"] >= -1e-12


class TestInputHandling:
    def test_sequence_and_array_forms_agree(self):
        rng = np.random.default_rng(7)
        n = 200
        x, y, z = rng.random(n), rng.random(n), rng.random(n)
        from_seq = coalition_mutual_information([x, y], z, k=3)
        from_arr = coalition_mutual_information(np.vstack([x, y]), z.reshape(1, -1), dim=2, k=3)
        for key in from_seq:
            assert from_seq[key] == pytest.approx(from_arr[key], abs=1e-12)

    def test_length_mismatch_rejected(self):
        rng = np.random.default_rng(1)
        x, z = rng.random(50), rng.random(50)
        with pytest.raises(ValueError):
            coalition_mutual_information([x, rng.random(49)], z)

    def test_plugin_method_rejects_too_many_dimensions(self):
        """"inv" cannot reach 4 total dimensions and must say so clearly."""
        rng = np.random.default_rng(2)
        n = 100
        with pytest.raises(ValueError):
            coalition_mutual_information(
                [rng.random(n), rng.random(n), rng.random(n)], rng.random(n), method="inv"
            )

    def test_measure_form_mismatches(self):
        rng = np.random.default_rng(3)
        x, y, z = rng.random(50), rng.random(50), rng.random(50)
        with pytest.raises(ValueError):
            pid_lattice([x, y], z, measure="imin")
        with pytest.raises(ValueError):
            pid_lattice(rng.random(4))
