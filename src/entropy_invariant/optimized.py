"""Optimized matrix functions for pairwise information theory computations."""

import math
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.special import digamma

from entropy_invariant._constants import E, LOG_UNIT_BALL_VOLUMES
from entropy_invariant.helpers.computation import compute_invariant_measure
from entropy_invariant.ksg import _mi_ksg_from_normalized, _cmi_fp_from_normalized


def MI(
    X: NDArray,
    *,
    method: str = "inv_ksg",
    k: int = 3,
    base: float = E,
    verbose: bool = False,
    degenerate: bool = False,
    dim: int = 1,
) -> NDArray[np.float64]:
    """
    Compute the pairwise mutual information (MI) matrix for all pairs of dimensions.

    Args:
        X: Data matrix where each column is a dimension (if dim=1, rows are points)
        method: `"inv_ksg"` (default) normalizes each dimension by its invariant
            measure and applies the KSG shared-radius estimator per pair (see
            `mutual_information_ksg`), cancelling the leading-order k-NN bias.
            `"inv"` instead uses the plug-in formula MI(Xi;Xj) = H(Xi)+H(Xj)-H(Xi,Xj)
            with independently-estimated k-NN entropies.
        k: Number of nearest neighbors (default 3)
        base: Logarithmic base (default e)
        verbose: Print computation info
        degenerate: Add +1 to distances for degenerate cases (ignored by method="inv_ksg")
        dim: Data layout (1=rows are points, 2=cols are points)

    Returns:
        Symmetric m x m matrix where M[i,j] is MI between dimensions i and j
    """
    # Convert to standard layout: rows are points, columns are dimensions
    a = np.asarray(X, dtype=np.float64)
    if dim == 2:
        a = a.T

    n = a.shape[0]  # number of points
    m = a.shape[1]  # number of dimensions

    if verbose:
        print(f"Number of points: {n}")
        print(f"Dimensions: {m}")
        print(f"Base: {base}")

    # Compute invariant measure for all dimensions and normalize once, shared
    # by both methods below.
    all_ri = np.array([compute_invariant_measure(a[:, i]) for i in range(m)])

    # Each element is shape (1, n) for KDTree
    all_a_ri = [a[:, i:i+1].T / all_ri[i] for i in range(m)]  # list of (1, n) arrays

    if method == "inv_ksg":
        log_base = math.log(base)
        all_mi_ij = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                mi_nats = _mi_ksg_from_normalized(
                    all_a_ri[i].T, all_a_ri[j].T, k
                )
                all_mi_ij[i, j] = mi_nats / log_base
                all_mi_ij[j, i] = all_mi_ij[i, j]
        return all_mi_ij

    if method != "inv":
        raise ValueError(f"Invalid method: {method}. Choose 'inv' or 'inv_ksg'")

    noise = 1 if degenerate else 0

    # Constants
    log_volume_unit_ball = LOG_UNIT_BALL_VOLUMES[:2]  # dims 1 and 2
    log_volume_1 = log_volume_unit_ball[0]
    log_volume_2 = log_volume_unit_ball[1]

    k_1 = k + 1
    dig_k = digamma(k)
    dig_n = digamma(n)

    # Compute marginal entropy for each dimension (1D)
    all_ent_i = np.zeros(m)
    for i in range(m):
        data_i = all_a_ri[i].T  # shape (n, 1) for KDTree
        kdtree = cKDTree(data_i)
        distances, _ = kdtree.query(data_i, k=k_1)
        dists_k = distances[:, k]  # k-th neighbor distance

        # Filter zeros and compute log
        log_dists_k = np.array([math.log(d + noise) for d in dists_k if d != 0])
        all_ent_i[i] = 1 * np.mean(log_dists_k) + log_volume_1 + dig_n - dig_k

    # Precompute all joint matrices (i, j) -> (2, n)
    all_ij = [[np.vstack([all_a_ri[i], all_a_ri[j]]) for j in range(m)] for i in range(m)]

    # Compute joint entropy for all pairs (2D)
    all_ent_ij = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):  # Only compute upper triangle
            data_ij = all_ij[i][j].T  # shape (n, 2) for KDTree
            kdtree = cKDTree(data_ij)
            distances, _ = kdtree.query(data_ij, k=k_1)
            dists_k = distances[:, k]

            log_dists_k = np.array([math.log(d + noise) for d in dists_k if d != 0])
            all_ent_ij[i, j] = 2 * np.mean(log_dists_k) + log_volume_2 + dig_n - dig_k
            all_ent_ij[j, i] = all_ent_ij[i, j]  # Symmetric

    # Compute MI matrix
    all_mi_ij = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            all_mi_ij[i, j] = all_ent_i[i] + all_ent_i[j] - all_ent_ij[i, j]
            all_mi_ij[j, i] = all_mi_ij[i, j]

    # Convert from nats to specified base
    return all_mi_ij / math.log(base)


def CMI(
    X: NDArray,
    Z: NDArray,
    *,
    method: str = "inv_ksg",
    base: float = E,
    k: int = 3,
    verbose: bool = False,
    degenerate: bool = False,
    dim: int = 1,
) -> NDArray[np.float64]:
    """
    Compute the conditional mutual information (CMI) matrix for all dimension pairs.

    Args:
        X: Data matrix where each column is a dimension
        Z: Conditioning variable (1D array or column vector)
        method: `"inv_ksg"` (default) normalizes each dimension (and Z) by its
            invariant measure and applies the Frenzel-Pompe shared-radius
            estimator per pair (see `conditional_mutual_information_ksg`),
            cancelling the leading-order k-NN bias. `"inv"` instead uses the
            plug-in formula CMI(Xi;Xj|Z) = H(Xi,Z)+H(Xj,Z)-H(Xi,Xj,Z)-H(Z).
        base: Logarithmic base (default e)
        k: Number of nearest neighbors (default 3)
        verbose: Print computation info
        degenerate: Add +1 to distances for degenerate cases (ignored by method="inv_ksg")
        dim: Data layout (1=rows are points, 2=cols are points)

    Returns:
        Symmetric m x m matrix where M[i,j] is CMI between dimensions i and j given Z
    """
    # Handle Z as either vector or matrix
    z = np.asarray(Z, dtype=np.float64)
    if z.ndim == 2:
        z = z.flatten()

    # Convert X to standard layout
    a = np.asarray(X, dtype=np.float64)
    if dim == 2:
        a = a.T

    n = a.shape[0]  # number of points
    m = a.shape[1]  # number of dimensions

    if len(z) != n:
        raise ValueError("Conditioning variable must have same number of points as X")

    if verbose:
        print(f"Number of points: {n}")
        print(f"Dimensions: {m}")
        print(f"Base: {base}")

    # Compute invariant measure for all dimensions of X and for Z, and
    # normalize once, shared by both methods below.
    all_ri = np.array([compute_invariant_measure(a[:, i]) for i in range(m)])
    rz = compute_invariant_measure(z)

    all_a_ri = [a[:, i:i+1].T / all_ri[i] for i in range(m)]  # list of (1, n) arrays
    b_rz = z.reshape(1, n) / rz  # shape (1, n)

    if method == "inv_ksg":
        log_base = math.log(base)
        z_col = b_rz.T  # shape (n, 1)
        all_cmi_ijz = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                cmi_nats = _cmi_fp_from_normalized(
                    all_a_ri[i].T, all_a_ri[j].T, z_col, k
                )
                all_cmi_ijz[i, j] = cmi_nats / log_base
                all_cmi_ijz[j, i] = all_cmi_ijz[i, j]
        return all_cmi_ijz

    if method != "inv":
        raise ValueError(f"Invalid method: {method}. Choose 'inv' or 'inv_ksg'")

    noise = 1 if degenerate else 0

    # Constants for dims 1, 2, 3
    log_volume_unit_ball = LOG_UNIT_BALL_VOLUMES[:3]
    log_volume_1 = log_volume_unit_ball[0]
    log_volume_2 = log_volume_unit_ball[1]
    log_volume_3 = log_volume_unit_ball[2]

    k_1 = k + 1
    dig_k = digamma(k)
    dig_n = digamma(n)

    # Compute entropy for Z (1D)
    data_z = b_rz.T  # shape (n, 1)
    kdtree_z = cKDTree(data_z)
    distances_z, _ = kdtree_z.query(data_z, k=k_1)
    dists_k_z = distances_z[:, k]
    log_dists_k_z = np.array([math.log(d + noise) for d in dists_k_z if d != 0])
    ent_z = 1 * np.mean(log_dists_k_z) + log_volume_1 + dig_n - dig_k

    # Compute marginal entropy for each dimension of X (1D)
    all_ent_i = np.zeros(m)
    for i in range(m):
        data_i = all_a_ri[i].T  # shape (n, 1)
        kdtree = cKDTree(data_i)
        distances, _ = kdtree.query(data_i, k=k_1)
        dists_k = distances[:, k]

        log_dists_k = np.array([math.log(d + noise) for d in dists_k if d != 0])
        all_ent_i[i] = 1 * np.mean(log_dists_k) + log_volume_1 + dig_n - dig_k

    # Compute joint entropy for (Xi, Z) pairs (2D)
    all_j_iz = [np.vstack([all_a_ri[i], b_rz]) for i in range(m)]  # list of (2, n)

    all_ent_iz = np.zeros(m)
    for i in range(m):
        data_iz = all_j_iz[i].T  # shape (n, 2)
        kdtree = cKDTree(data_iz)
        distances, _ = kdtree.query(data_iz, k=k_1)
        dists_k = distances[:, k]

        log_dists_k = np.array([math.log(d + noise) for d in dists_k if d != 0])
        all_ent_iz[i] = 2 * np.mean(log_dists_k) + log_volume_2 + dig_n - dig_k

    # Compute triple joint entropy for (Xi, Xj, Z) (3D)
    all_j_ijz = [[np.vstack([all_a_ri[i], all_a_ri[j], b_rz]) for j in range(m)] for i in range(m)]

    all_ent_ijz = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):  # Only compute upper triangle
            data_ijz = all_j_ijz[i][j].T  # shape (n, 3)
            kdtree = cKDTree(data_ijz)
            distances, _ = kdtree.query(data_ijz, k=k_1)
            dists_k = distances[:, k]

            log_dists_k = np.array([math.log(d + noise) for d in dists_k if d != 0])
            all_ent_ijz[i, j] = 3 * np.mean(log_dists_k) + log_volume_3 + dig_n - dig_k
            all_ent_ijz[j, i] = all_ent_ijz[i, j]  # Symmetric

    # Compute CMI matrix
    all_cmi_ijz = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            all_cmi_ijz[i, j] = all_ent_iz[i] + all_ent_iz[j] - all_ent_ijz[i, j] - ent_z
            all_cmi_ijz[j, i] = all_cmi_ijz[i, j]

    # Convert from nats to specified base
    return all_cmi_ijz / math.log(base)
