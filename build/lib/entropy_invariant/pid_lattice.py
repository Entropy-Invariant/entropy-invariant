"""
N-source Partial Information Decomposition: the Williams & Beer redundancy lattice.

``redundancy`` / ``unique`` / ``synergy`` in ``pid.py`` decompose ``I({X,Y}; Z)`` into
exactly four terms and are hard-wired to two sources. This module provides the general
N-source decomposition of Williams & Beer (2010), "Nonnegative Decomposition of
Multivariate Information", in which ``I({X_1,...,X_N}; Z)`` splits into a lattice of
partial-information atoms: 4 nodes for N=2, 18 for N=3, 166 for N=4.

Three redundancy measures are provided, because they answer differently:

- ``"mmi"``  ``I_cap(a) = min over A in a of I(X_A; Z)``. Estimable from continuous data
  with any of the package's estimators, and reduces EXACTLY to ``redundancy``/``unique``
  at N=2. Because it is a function of the ``2^N - 1`` coalition MIs alone, its two-source
  unique atoms are ``max(0, I_X - I_Y)`` and ``max(0, I_Y - I_X)`` -- so exactly one is
  nonzero by construction. Fine for asking how information is divided, poor for asking
  how much each source contributes; prefer ``conditional_mutual_information`` for that.
- ``"imin"`` Williams & Beer's original measure, via specific information evaluated per
  target outcome and averaged. Guarantees non-negative atoms (which ``"mmi"`` does not
  for N >= 3) but needs a discrete target, so it takes a joint pmf. Known to over-credit
  redundancy (Harder, Salge & Polani 2013): on two-bit COPY it calls two independent bits
  fully redundant.
- ``"iccs"`` Ince's (2017) pointwise common change in surprisal. Also takes a joint pmf.
  Fixes the COPY problem and, unlike ``"mmi"``, lets both unique atoms be positive at
  once. Atoms may be negative; Ince argues these are meaningful.

NOTE ON VALIDATION. ``sum(atoms) == I_cap(top node)`` holds identically for ANY redundancy
function, right or wrong -- it is a property of the Moebius inversion, not evidence the
decomposition is correct. Correctness is established against published atom values on
discrete toy distributions (AND, XOR, two-bit COPY, three-way XOR), the Williams & Beer
non-negativity theorem, and exact agreement with ``redundancy``/``unique`` at N=2.
See ``tests/test_pid_lattice.py``.
"""

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from entropy_invariant._constants import E
from entropy_invariant.entropy import entropy
from entropy_invariant.helpers.computation import convert_to_base
from entropy_invariant.helpers.data import ensure_columns_are_points
from entropy_invariant.ksg import _mi_ksg_from_normalized
from entropy_invariant.pid import _invariant_normalize_columns

__all__ = [
    "RedundancyLattice",
    "redundancy_lattice",
    "lattice_labels",
    "moebius_atoms",
    "coalition_mutual_information",
    "isotonic_repair",
    "mmi_redundancy",
    "imin_redundancy",
    "iccs_redundancy",
    "specific_information",
    "pid_lattice",
]


@dataclass(frozen=True)
class RedundancyLattice:
    """
    The Williams & Beer redundancy lattice for a fixed number of sources.

    Attributes:
        n_sources: number of source variables.
        nodes: lattice nodes. Each node is an antichain -- a canonical (sorted) tuple of
            source coalitions, where a coalition is a bitmask over ``range(n_sources)``
            with bit ``i`` set iff source ``i`` belongs to it.
        predecessors: ``predecessors[i]`` lists the indices of every STRICT predecessor of
            ``nodes[i]`` (the full down-set, not just immediate covers, which is what the
            Moebius inversion requires).
        order: a topological order of node indices in which every strict predecessor
            appears before the node itself.
    """

    n_sources: int
    nodes: List[Tuple[int, ...]]
    predecessors: List[List[int]]
    order: List[int]


def _coalitions(n: int) -> List[int]:
    return list(range(1, 2 ** n))


def _issubset_mask(a: int, b: int) -> bool:
    return (a & b) == a


def _is_antichain(c: Sequence[int]) -> bool:
    for i, ci in enumerate(c):
        for j, cj in enumerate(c):
            if i != j and _issubset_mask(ci, cj):
                return False
    return True


def _antichains(n: int) -> List[Tuple[int, ...]]:
    """
    Every non-empty antichain of the non-empty subsets of ``range(n)``.

    Enumerated by filtering all ``2^(2^n - 1)`` collections, which is 128 candidates at
    n=3 and 32768 at n=4 -- immediate. n >= 5 has 7579 nodes and would need a dedicated
    antichain generator; it is rejected rather than attempted, since no estimator has the
    statistics to populate it.
    """
    if not 1 <= n <= 4:
        raise ValueError(
            f"redundancy_lattice supports 1 to 4 sources (n=5 has 7579 nodes and needs a "
            f"dedicated generator). Got n_sources={n}."
        )
    coals = _coalitions(n)
    out: List[Tuple[int, ...]] = []
    for sel in range(1, 2 ** len(coals)):
        c = [coals[i] for i in range(len(coals)) if (sel >> i) & 1]
        if _is_antichain(c):
            out.append(tuple(sorted(c)))
    return out


def _precedes(a: Sequence[int], b: Sequence[int]) -> bool:
    """
    The Williams & Beer lattice order: ``a <= b`` iff every coalition in ``b`` contains
    some coalition in ``a``. Reflexive.
    """
    return all(any(_issubset_mask(A, B) for A in a) for B in b)


def redundancy_lattice(n_sources: int) -> RedundancyLattice:
    """
    Build the Williams & Beer redundancy lattice for ``n_sources`` source variables:
    4 nodes for 2 sources, 18 for 3, 166 for 4.

    Example:
        >>> lat = redundancy_lattice(3)
        >>> len(lat.nodes)
        18
    """
    nodes = _antichains(n_sources)
    n = len(nodes)
    preds: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and _precedes(nodes[j], nodes[i]):
                preds[i].append(j)
    # In a partial order, b < a implies the down-set of b is a strict subset of that of a,
    # so ordering by down-set size is a valid topological order.
    order = sorted(range(n), key=lambda i: (len(preds[i]), i))
    return RedundancyLattice(n_sources, nodes, preds, order)


def lattice_labels(
    lat: RedundancyLattice, names: Optional[Sequence[str]] = None, *, sep: str = ""
) -> List[str]:
    """
    Human-readable labels for every lattice node, e.g. ``{X}{YZ}``.

    ``sep`` separates the members inside a coalition, worth setting when source names are
    multi-character (``sep="."`` gives ``{X}{Y.Z}``).
    """
    if names is None:
        names = [str(i + 1) for i in range(lat.n_sources)]
    if len(names) != lat.n_sources:
        raise ValueError(f"expected {lat.n_sources} names, got {len(names)}")
    labels = []
    for a in lat.nodes:
        labels.append(
            "".join(
                "{" + sep.join(names[i] for i in range(lat.n_sources) if (A >> i) & 1) + "}"
                for A in a
            )
        )
    return labels


def moebius_atoms(lat: RedundancyLattice, i_cap: Callable[[Tuple[int, ...]], float]) -> NDArray[np.float64]:
    """
    Partial-information atoms ``Pi(a) = i_cap(a) - sum over strict predecessors b < a of
    Pi(b)``, indexed like ``lat.nodes``.

    ``atoms.sum()`` equals ``i_cap`` at the top node identically, for any ``i_cap`` -- that
    is how Moebius inversion works and is not a check on correctness.
    """
    atoms = np.full(len(lat.nodes), np.nan)
    for i in lat.order:
        atoms[i] = i_cap(lat.nodes[i]) - sum(atoms[j] for j in lat.predecessors[i])
    return atoms


def coalition_mutual_information(
    sources: Union[NDArray, Sequence[NDArray]],
    target: NDArray,
    *,
    method: str = "inv_ksg",
    nbins: int = 10,
    k: int = 3,
    base: float = E,
    degenerate: bool = False,
    dim: int = 1,
) -> Dict[int, float]:
    """
    ``I(X_A; Z)`` for every non-empty coalition ``A`` of the source variables, keyed by the
    coalition's bitmask.

    ``sources`` is either an array whose rows (after ``dim`` canonicalisation) are the
    source variables, or a sequence of 1-D source arrays. ``target`` is 1-D or a
    single-row array.

    For ``method="inv_ksg"`` the coalition block is normalised per dimension and passed to
    the shared-radius KSG estimator, which is dimension-agnostic. Other methods go through
    the joint-entropy chain rule; note that the ``"inv"`` plug-in entropy is only defined
    up to 3 total dimensions, so coalitions large enough to exceed that are rejected with
    a clear error rather than a confusing one from deeper in the stack.
    """
    if isinstance(sources, (list, tuple)):
        arrs = [np.asarray(s, dtype=np.float64).ravel() for s in sources]
        tgt_v = np.asarray(target, dtype=np.float64).ravel()
        for i, s in enumerate(arrs):
            if s.shape[0] != tgt_v.shape[0]:
                raise ValueError(
                    f"source {i} has {s.shape[0]} points but target has {tgt_v.shape[0]}"
                )
        src = np.vstack(arrs)
        tgt = tgt_v.reshape(1, -1)
    else:
        src = ensure_columns_are_points(np.atleast_2d(np.asarray(sources, dtype=np.float64)), dim)
        t = np.asarray(target, dtype=np.float64)
        tgt = t.reshape(1, -1) if t.ndim == 1 else ensure_columns_are_points(t, dim)

    n_sources = src.shape[0]
    if src.shape[1] != tgt.shape[1]:
        raise ValueError(
            f"sources and target must have the same number of points, "
            f"got {src.shape[1]} and {tgt.shape[1]}"
        )
    if method != "inv_ksg" and (n_sources + tgt.shape[0]) > 3:
        raise ValueError(
            f'method="{method}" cannot evaluate a {n_sources}-source coalition against a '
            f"{tgt.shape[0]}-dimensional target: the plug-in entropy estimator is only "
            f'defined for 1-3 total dimensions. Use method="inv_ksg", which is '
            f"dimension-agnostic."
        )

    out: Dict[int, float] = {}
    for A in _coalitions(n_sources):
        rows = [i for i in range(n_sources) if (A >> i) & 1]
        block = src[rows, :]
        if method == "inv_ksg":
            out[A] = convert_to_base(
                _mi_ksg_from_normalized(
                    _invariant_normalize_columns(block.T),
                    _invariant_normalize_columns(tgt.T),
                    k,
                ),
                base,
            )
        else:
            kw = dict(method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
            out[A] = (
                entropy(block, **kw)
                + entropy(tgt, **kw)
                - entropy(np.vstack([block, tgt]), **kw)
            )
    return out


def isotonic_repair(
    coalition_mi: Dict[int, float],
    *,
    mode: str = "isotonic",
    iterations: int = 500,
    tol: float = 1e-12,
) -> Dict[int, float]:
    """
    Project estimated coalition mutual informations onto the monotone cone
    ``I(X_A; Z) <= I(X_B; Z)`` for ``A`` a subset of ``B``.

    True mutual information always satisfies this, but finite-sample kNN estimates
    measurably do not: adding a source that is nearly redundant with those already present
    lowers the estimate, because the neighbourhood is spread over an extra dimension.
    ``"mmi"`` takes a minimum over coalitions and the Moebius inversion takes differences
    between them, and both assume the ordering holds -- unrepaired estimates therefore
    produce negative unique and synergy atoms, which are mathematically impossible for
    true information.

    - ``"isotonic"`` -- least-squares projection onto the cone by Dykstra's algorithm over
      the pairwise constraints. Minimal distortion; the default.
    - ``"majorant"`` -- minimal monotone majorant,
      ``I(B) <- max(I(B), max over A subset B of I(A))``. Simpler and only ever raises
      values, but distorts more.

    This deliberately does NOT clamp to zero. Non-negativity is a constraint on the LEVEL
    of a single estimate, whereas monotonicity is a constraint BETWEEN estimates -- a
    violated pair is mutually inconsistent, so pooling loses nothing, but an estimate of
    -0.02 where the true value is near zero is an ordinary fluctuation of an estimator
    that is deliberately unbiased rather than truncated. Clamping it would bias every
    low-signal region upward and destroy the symmetry of the noise.

    Returns a new dict; the input is not modified.
    """
    keys = sorted(coalition_mi, key=lambda m: bin(m).count("1"))
    v = {A: float(coalition_mi[A]) for A in keys}

    if mode == "majorant":
        for B in keys:
            for A in keys:
                if A != B and _issubset_mask(A, B):
                    v[B] = max(v[B], v[A])
        return v
    if mode != "isotonic":
        raise ValueError(f'mode must be "isotonic" or "majorant", got "{mode}"')

    pairs = [(A, B) for A in keys for B in keys if A != B and _issubset_mask(A, B)]
    corr = {p: 0.0 for p in pairs}
    for _ in range(iterations):
        maxshift = 0.0
        for p in pairs:
            A, B = p
            a, b = v[A] - corr[p], v[B] + corr[p]
            if a > b:  # violated: pool to the midpoint
                mid = (a + b) / 2.0
                corr[p] = a - mid
                v[A] = v[B] = mid
                maxshift = max(maxshift, abs(corr[p]))
            else:
                v[A], v[B] = a, b
                corr[p] = 0.0
        if maxshift < tol:
            break
    return v


def mmi_redundancy(coalition_mi: Dict[int, float]) -> Callable[[Tuple[int, ...]], float]:
    """
    The minimum-mutual-information redundancy ``I_cap(a) = min over A in a of I(X_A; Z)``,
    as a function suitable for ``moebius_atoms``.

    Monotone on the lattice provided ``coalition_mi`` is monotone under subset inclusion --
    run ``isotonic_repair`` on estimated values first.
    """
    return lambda a: min(float(coalition_mi[A]) for A in a)


def specific_information(pmf: NDArray, coalition: int, *, base: float = 2.0) -> NDArray[np.float64]:
    """
    Williams & Beer's specific information
    ``I(Z=z; X_A) = sum_a p(a|z) log( p(z|a) / p(z) )``, one value per target outcome.

    ``pmf`` is the joint distribution over ``(X_1, ..., X_N, Z)`` with the target in the
    LAST axis. ``coalition`` is a bitmask over the sources.
    """
    p = np.asarray(pmf, dtype=np.float64)
    n_sources = p.ndim - 1
    members = [i for i in range(n_sources) if (coalition >> i) & 1]
    if not members:
        raise ValueError("coalition must be non-empty")
    drop = tuple(i for i in range(n_sources) if i not in members)
    m = p.sum(axis=drop) if drop else p
    nz = m.shape[-1]
    ma = m.reshape(-1, nz)
    pz = ma.sum(axis=0)
    pa = ma.sum(axis=1)
    out = np.zeros(nz)
    logb = np.log(base)
    for z in range(nz):
        if pz[z] <= 0:
            continue
        s = 0.0
        for a in range(ma.shape[0]):
            paz = ma[a, z]
            if paz <= 0 or pa[a] <= 0:
                continue
            s += (paz / pz[z]) * (np.log((paz / pa[a]) / pz[z]) / logb)
        out[z] = s
    return out


def imin_redundancy(pmf: NDArray, *, base: float = 2.0) -> Callable[[Tuple[int, ...]], float]:
    """
    Williams & Beer's original redundancy
    ``I_min(a) = sum_z p(z) * min over A in a of I(Z=z; X_A)``, as a function suitable for
    ``moebius_atoms``.

    Evaluated per target outcome and then averaged -- this is NOT a minimum over the
    coalitions' total mutual informations (that is ``mmi_redundancy``, a different measure
    that happens to coincide at the two-source bottom node). Guarantees non-negative
    atoms, and over-credits redundancy on two-bit COPY; both are properties of the measure.
    """
    p = np.asarray(pmf, dtype=np.float64)
    n_sources = p.ndim - 1
    pz = p.sum(axis=tuple(range(n_sources)))
    cache: Dict[int, NDArray[np.float64]] = {}

    def spec(A: int) -> NDArray[np.float64]:
        if A not in cache:
            cache[A] = specific_information(p, A, base=base)
        return cache[A]

    def i_cap(a: Tuple[int, ...]) -> float:
        s = np.vstack([spec(A) for A in a])
        return float(np.sum(pz * s.min(axis=0)))

    return i_cap


def _ccs_tables(p: NDArray[np.float64], masks: Sequence[int]):
    """
    For each coalition bitmask, the joint over ``(X_B, Z)`` reshaped to
    ``(joint source outcomes, target outcomes)``, the marginal over ``X_B``, and the
    member axes plus strides needed to map a full realisation to the right row.
    Built once per decomposition and shared by every node.
    """
    n_sources = p.ndim - 1
    dims = p.shape
    tables = {}
    for B in set(masks):
        members = [i for i in range(n_sources) if (B >> i) & 1]
        drop = tuple(i for i in range(n_sources) if i not in members)
        m = p.sum(axis=drop) if drop else p
        nz = m.shape[-1]
        joint = m.reshape(-1, nz)
        marg = joint.sum(axis=1)
        # C-order strides over the retained source axes
        strides = [1] * len(members)
        acc = 1
        for j in range(len(members) - 1, -1, -1):
            strides[j] = acc
            acc *= dims[members[j]]
        tables[B] = (joint, marg, members, strides)
    return tables


def _ccs_sign(v: float, tol: float) -> int:
    return 1 if v > tol else (-1 if v < -tol else 0)


def iccs_redundancy(
    pmf: NDArray, *, base: float = 2.0, tol: float = 1e-12
) -> Callable[[Tuple[int, ...]], float]:
    """
    Ince's (2017) ``I_ccs`` redundancy -- "common change in surprisal" -- as a function
    suitable for ``moebius_atoms``. ``pmf`` is the joint distribution over
    ``(X_1, ..., X_N, Z)`` with the target in the LAST axis.

    For a node ``{A_1, ..., A_k}`` it accumulates the pointwise co-information between the
    coalition variables and the target,

        c = sum over non-empty T of the A's of (-1)^(|T|+1) * i(x_{union of T}; z)

    where ``i(x_B; z) = log p(x_B, z) / (p(x_B) p(z))`` is the local mutual information,
    but counts a realisation ONLY when every ``i(x_{A_j}; z)`` and ``c`` itself share the
    same sign. That sign-agreement condition is the whole idea: it keeps only the
    surprisal change that all coalitions genuinely hold in common.

    Why it is worth having alongside ``"mmi"`` and ``"imin"``:

    - It fixes the two-bit COPY problem. For ``Z = (X, Y)`` with independent bits it gives
      ``R = 0``, ``U_X = U_Y = 1``, ``Syn = 0``, the decomposition most measures agree is
      right, where ``"imin"`` reports the two independent bits as fully redundant
      (``R = 1``).
    - Its unique atoms are not winner-take-all. ``"mmi"`` gives ``max(0, I_X - I_Y)`` and
      ``max(0, I_Y - I_X)``, so exactly one is nonzero by construction; ``I_ccs`` lets both
      be positive simultaneously, which matters whenever the question is how much each
      source contributes rather than how the total divides.
    - It is defined for any number of sources, unlike BROJA and the other
      optimisation-based measures, which do not extend cleanly past two.

    In exchange, ``I_ccs`` is not guaranteed monotone on the lattice, so atoms can come out
    negative. Ince argues these are meaningful rather than a defect; either way they are
    expected behaviour here and are not repaired away.
    """
    p = np.asarray(pmf, dtype=np.float64)
    n_sources = p.ndim - 1
    pz = p.sum(axis=tuple(range(n_sources)))
    logb = np.log(base)
    cache: Dict[Tuple[int, ...], float] = {}

    def i_cap(a: Tuple[int, ...]) -> float:
        if a in cache:
            return cache[a]
        k = len(a)
        # every union of a non-empty sub-collection of the node's coalitions
        subsets = []
        for sel in range(1, 2 ** k):
            B = 0
            cnt = 0
            for j in range(k):
                if (sel >> j) & 1:
                    B |= a[j]
                    cnt += 1
            subsets.append((B, cnt))
        tables = _ccs_tables(p, [b for b, _ in subsets] + list(a))

        def local_mi(B: int, idx) -> float:
            joint, marg, members, strides = tables[B]
            r = 0
            for j, ax in enumerate(members):
                r += idx[ax] * strides[j]
            if joint[r, idx[n_sources]] <= 0 or marg[r] <= 0:
                return -np.inf
            return float(np.log(joint[r, idx[n_sources]] / (marg[r] * pz[idx[n_sources]])) / logb)

        total = 0.0
        for idx in np.ndindex(*p.shape):
            w = p[idx]
            if w <= 0 or pz[idx[n_sources]] <= 0:
                continue
            c = 0.0
            ok = True
            for B, cnt in subsets:
                v = local_mi(B, idx)
                if not np.isfinite(v):
                    ok = False
                    break
                c += (1.0 if cnt % 2 else -1.0) * v
            if not ok:
                continue
            s0 = _ccs_sign(c, tol)
            agree = True
            for A in a:
                v = local_mi(A, idx)
                if not np.isfinite(v) or _ccs_sign(v, tol) != s0:
                    agree = False
                    break
            if agree:
                total += w * c
        cache[a] = total
        return total

    return i_cap


def pid_lattice(
    sources,
    target=None,
    *,
    measure: Optional[str] = None,
    repair: str = "isotonic",
    names: Optional[Sequence[str]] = None,
    method: str = "inv_ksg",
    nbins: int = 10,
    k: int = 3,
    base: Optional[float] = None,
    degenerate: bool = False,
    dim: int = 1,
) -> Dict[str, float]:
    """
    Full N-source partial information decomposition of ``I({X_1,...,X_N}; Z)`` over the
    Williams & Beer redundancy lattice, returning every atom keyed by its label
    (``"{1}{2}{3}"``, ``"{12}"``, ...).

    Two forms:

    - ``pid_lattice(sources, target)`` -- estimate from continuous data. Default
      ``measure="mmi"``, default ``base=e``.
    - ``pid_lattice(pmf)`` -- an explicit discrete joint distribution with the target in
      the last axis. Default ``measure="imin"``, default ``base=2``; ``measure="iccs"``
      selects Ince's measure.

    ``repair`` controls the monotonicity projection applied to the estimated coalition MIs
    before decomposing (``"isotonic"``, ``"majorant"`` or ``"none"``); see
    ``isotonic_repair`` for why this is not optional in practice. Pass ``names`` to label
    atoms with variable names instead of indices.

    Example:
        >>> # discrete, two-input AND -- Williams & Beer's published values in bits
        >>> pmf = np.zeros((2, 2, 2))
        >>> pmf[0, 0, 0] = pmf[0, 1, 0] = pmf[1, 0, 0] = 0.25
        >>> pmf[1, 1, 1] = 0.25
        >>> atoms = pid_lattice(pmf, names=["X", "Y"])
        >>> round(atoms["{X}{Y}"], 4), round(atoms["{XY}"], 4)
        (0.3113, 0.5)
    """
    if target is None:
        pmf = np.asarray(sources, dtype=np.float64)
        measure = "imin" if measure is None else measure
        if measure not in ("imin", "iccs"):
            raise ValueError(
                f'measure="{measure}" is not defined on a discrete joint distribution; '
                f'use "imin" or "iccs".'
            )
        if pmf.ndim < 2:
            raise ValueError(
                "pmf needs at least 2 axes (>=1 source plus the target in the last axis)"
            )
        b = 2.0 if base is None else base
        lat = redundancy_lattice(pmf.ndim - 1)
        i_cap = (
            imin_redundancy(pmf, base=b) if measure == "imin" else iccs_redundancy(pmf, base=b)
        )
    else:
        measure = "mmi" if measure is None else measure
        if measure != "mmi":
            raise ValueError(
                f'measure="{measure}" needs a discrete joint distribution; call '
                f"pid_lattice(pmf, measure=...) instead. Continuous data supports \"mmi\"."
            )
        b = E if base is None else base
        cmi = coalition_mutual_information(
            sources, target, method=method, nbins=nbins, k=k, base=b,
            degenerate=degenerate, dim=dim,
        )
        if repair != "none":
            cmi = isotonic_repair(cmi, mode=repair)
        lat = redundancy_lattice(int(round(np.log2(len(cmi) + 1))))
        i_cap = mmi_redundancy(cmi)

    labels = lattice_labels(lat, names)
    return {labels[i]: float(v) for i, v in enumerate(moebius_atoms(lat, i_cap))}
