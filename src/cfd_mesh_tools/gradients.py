"""WLS gradient/Hessian coefficient builders, difference-form consumers, and slope limiters.

Coefficient builders live intact in ``cfd_mesh_tools.readers.MeshReader`` and are
re-exported here alongside the verbatim Numba gradient, Hessian, and slope-limiter
kernels originally defined in ``MeshAdaptation/Scripts/FunScripts/InterpSolution_Fun.py``.
"""

import time

import numpy as np
from numba import njit, prange

from .readers.MeshReader import (
    compute_wls_coefficients_deg2,
    compute_wls_coefficients_optimized,
    compute_wls_coefficients_pinv,
)


@njit(parallel=True, fastmath=True, cache=True)
def ComputeWLSGradients_Numba(fields, neighbors, NNeighbors, wls_coeffs, nDim):
    """
    Compute WLS gradients for multiple fields simultaneously.

    Parameters:
    -----------
    fields : np.ndarray, shape (n_fields, n_points)
        Multiple scalar fields stacked together
    neighbors : np.ndarray, shape (n_points, max_neighbors)
        Neighbor connectivity (-1 for invalid)
    wls_coeffs : np.ndarray, shape (n_points, max_neighbors, 3)
        Precomputed WLS coefficients
    nDim : int
        Active spatial dimensions (2 or 3)

    Returns:
    --------
    grads : np.ndarray, shape (n_fields, n_points, 3)
        Gradients for all fields (unused z column stays 0 in 2D)
    """
    n_fields, n_points = fields.shape
    max_neighbors = neighbors.shape[1]
    # Always allocate 3 columns for the spatial direction so consumers can
    # access [..., 2] safely in both 2D and 3D. The unused z column stays 0 in 2D.
    grads = np.zeros((n_fields, n_points, 3), dtype=np.float64)

    # Parallel loop over nodes and fields (flattened)
    for idx in prange(n_fields * n_points):
        field_id = idx // n_points
        node_id = idx % n_points

        phi_center = fields[field_id, node_id]

        # Accumulate gradient contributions
        for k in range(NNeighbors[node_id]):
            neighbor_idx = neighbors[node_id, k]

            d_phi = fields[field_id, neighbor_idx] - phi_center

            for d in range(nDim):
                grads[field_id, node_id, d] += d_phi * wls_coeffs[node_id, k, d]

    return grads


@njit(parallel=True, fastmath=True, cache=True)
def ComputeWLSHessians_Numba(fields, neighbors, NNeighbors, hess_coeffs, nH):
    """
    Apply precomputed degree-2 WLS Hessian coefficients to multiple fields:

        hess[var, node, comp] = sum_k hess_coeffs[node, k, comp] * (phi[nbr_k] - phi[node])

    with the symmetric Hessian packed upper-triangular (2D: [Hxx,Hxy,Hyy];
    3D: [Hxx,Hxy,Hxz,Hyy,Hyz,Hzz]). Mirrors ComputeWLSGradients_Numba. Nodes with
    a zero coefficient row (rank-deficient fallback) return a zero Hessian.
    """
    n_fields, n_points = fields.shape
    hess = np.zeros((n_fields, n_points, nH), dtype=np.float64)

    for idx in prange(n_fields * n_points):
        field_id = idx // n_points
        node_id = idx % n_points
        phi_center = fields[field_id, node_id]
        for k in range(NNeighbors[node_id]):
            neighbor_idx = neighbors[node_id, k]
            d_phi = fields[field_id, neighbor_idx] - phi_center
            for c in range(nH):
                hess[field_id, node_id, c] += d_phi * hess_coeffs[node_id, k, c]

    return hess


@njit(parallel=True, fastmath=True, cache=True)
def ComputeVenkatLimiter_Numba(fields, grads, points, neighbors, NNeighbors, nDim, eps_sq):
    """
    Venkatakrishnan smooth slope limiter (Nishikawa 2022 convention).

    The reconstruction is evaluated at the edge midpoint between node i and neighbor j:
        u_face = u_i + 0.5 * grad(f) . (r_j - r_i)
    so the "face increment" is d2_face = 0.5 * grad(f) . (r_j - r_i).
    """
    n_fields, n_points = fields.shape
    phi_out = np.ones((n_fields, n_points), dtype=np.float64)

    for idx in prange(n_fields * n_points):
        v = idx // n_points
        i = idx % n_points
        nk = NNeighbors[i]
        if nk == 0:
            continue

        fi = fields[v, i]
        fmax = fi
        fmin = fi
        for k in range(nk):
            j = neighbors[i, k]
            fj = fields[v, j]
            if fj > fmax:
                fmax = fj
            if fj < fmin:
                fmin = fj

        gx = grads[v, i, 0]
        gy = grads[v, i, 1]
        gz = grads[v, i, 2] if nDim == 3 else 0.0
        xi = points[i, 0]
        yi = points[i, 1]
        zi = points[i, 2] if nDim == 3 else 0.0

        phi_min = 1.0
        for k in range(nk):
            j = neighbors[i, k]
            dx = points[j, 0] - xi
            dy = points[j, 1] - yi
            dz = (points[j, 2] - zi) if nDim == 3 else 0.0
            d2_face = 0.5 * (gx * dx + gy * dy + gz * dz)
            if d2_face > 0.0:
                d1 = fmax - fi
            elif d2_face < 0.0:
                d1 = fmin - fi
            else:
                continue

            y = d1 / d2_face
            num = y * y + 2.0 * y + eps_sq
            den = y * y + y + 2.0 + eps_sq
            alpha = num / den if den > 0.0 else 1.0
            if alpha < 0.0:
                alpha = 0.0
            if alpha > 1.0:
                alpha = 1.0
            if alpha < phi_min:
                phi_min = alpha

        phi_out[v, i] = phi_min

    return phi_out


@njit(parallel=True, fastmath=True, cache=True)
def ComputeNishikawaRLimiter_Numba(fields, grads, points, neighbors, NNeighbors, nDim, R):
    """
    Nishikawa R-limiter family (AIAA SciTech 2022, Eqs 3.11-3.13), R in {3, 4, 5}.
    """
    n_fields, n_points = fields.shape
    phi_out = np.ones((n_fields, n_points), dtype=np.float64)

    for idx in prange(n_fields * n_points):
        v = idx // n_points
        i = idx % n_points
        nk = NNeighbors[i]
        if nk == 0:
            continue

        fi = fields[v, i]
        fmax = fi
        fmin = fi
        for k in range(nk):
            j = neighbors[i, k]
            fj = fields[v, j]
            if fj > fmax:
                fmax = fj
            if fj < fmin:
                fmin = fj

        gx = grads[v, i, 0]
        gy = grads[v, i, 1]
        gz = grads[v, i, 2] if nDim == 3 else 0.0
        xi = points[i, 0]
        yi = points[i, 1]
        zi = points[i, 2] if nDim == 3 else 0.0

        phi_min = 1.0
        for k in range(nk):
            j = neighbors[i, k]
            dx = points[j, 0] - xi
            dy = points[j, 1] - yi
            dz = (points[j, 2] - zi) if nDim == 3 else 0.0
            d2_face = 0.5 * (gx * dx + gy * dy + gz * dz)
            if d2_face > 0.0:
                d1 = fmax - fi
            elif d2_face < 0.0:
                d1 = fmin - fi
            else:
                continue

            y = d1 / d2_face
            if y > 2.0:
                alpha = 1.0
            else:
                if R == 3:
                    num = y * y * y + 4.0 * y
                    den = y * y * y + y * y + 4.0
                elif R == 4:
                    y2 = y * y
                    y3 = y2 * y
                    y4 = y3 * y
                    num = y4 + 2.0 * y3 - 4.0 * y2 + 8.0 * y
                    den = y4 + y3 + 2.0 * y2 - 4.0 * y + 8.0
                elif R == 5:
                    y2 = y * y
                    y3 = y2 * y
                    y4 = y3 * y
                    y5 = y4 * y
                    num = y5 + 8.0 * y3 - 16.0 * y2 + 16.0 * y
                    den = y5 + y4 + 8.0 * y2 - 16.0 * y + 16.0
                else:
                    num = y * y * y + 4.0 * y
                    den = y * y * y + y * y + 4.0
                alpha = num / den if den > 0.0 else 1.0
                if alpha < 0.0:
                    alpha = 0.0
                if alpha > 1.0:
                    alpha = 1.0
            if alpha < phi_min:
                phi_min = alpha

        phi_out[v, i] = phi_min

    return phi_out


@njit(parallel=True, fastmath=True, cache=True)
def ComputeBJLimiter_Numba(fields, grads, points, neighbors, NNeighbors, nDim):
    """Barth-Jespersen slope limiter. Returns phi in [0, 1] for each (field, node)."""
    n_fields, n_points = fields.shape
    phi_out = np.ones((n_fields, n_points), dtype=np.float64)

    for idx in prange(n_fields * n_points):
        v = idx // n_points
        i = idx % n_points
        nk = NNeighbors[i]
        if nk == 0:
            continue

        fi = fields[v, i]
        fmax = fi
        fmin = fi
        for k in range(nk):
            j = neighbors[i, k]
            fj = fields[v, j]
            if fj > fmax:
                fmax = fj
            if fj < fmin:
                fmin = fj

        gx = grads[v, i, 0]
        gy = grads[v, i, 1]
        gz = grads[v, i, 2] if nDim == 3 else 0.0
        xi = points[i, 0]
        yi = points[i, 1]
        zi = points[i, 2] if nDim == 3 else 0.0

        phi_min = 1.0
        for k in range(nk):
            j = neighbors[i, k]
            dx = points[j, 0] - xi
            dy = points[j, 1] - yi
            dz = (points[j, 2] - zi) if nDim == 3 else 0.0
            d2_face = 0.5 * (gx * dx + gy * dy + gz * dz)
            if d2_face > 0.0:
                num = fmax - fi
                if num <= 0.0:
                    alpha = 0.0
                else:
                    alpha = num / d2_face
                    if alpha > 1.0:
                        alpha = 1.0
            elif d2_face < 0.0:
                num = fmin - fi
                if num >= 0.0:
                    alpha = 0.0
                else:
                    alpha = num / d2_face
                    if alpha > 1.0:
                        alpha = 1.0
            else:
                alpha = 1.0
            if alpha < phi_min:
                phi_min = alpha

        phi_out[v, i] = phi_min

    return phi_out


def ApplyLimiterIfRequested(Gradients_From, dataFrom_fields, PointsFrom_3d,
                            NodesConnectedToNode_From, NumOfNodesConnectedToNode_From,
                            nDim, options, CVsAreas_From=None, Hessians_From=None):
    """Apply slope limiter in-place on Gradients_From (and optionally Hessians_From)."""
    if not options.get("UseLimiter", False):
        return Gradients_From
    LimiterType = options.get("LimiterType", "BJ")
    start = time.time()
    if LimiterType in ("BJ", "BarthJespersen"):
        print("Applying Barth-Jespersen limiter to gradients...")
        phi = ComputeBJLimiter_Numba(dataFrom_fields, Gradients_From, PointsFrom_3d,
                                     NodesConnectedToNode_From, NumOfNodesConnectedToNode_From, nDim)
    elif LimiterType in ("Venkat", "Venkatakrishnan"):
        K = options.get("VenkatK", 5.0)
        if CVsAreas_From is not None and CVsAreas_From.size > 0:
            mean_meas = float(np.mean(CVsAreas_From))
            h_ref = mean_meas ** (1.0 / nDim)
        else:
            h_ref = 1.0
        eps_sq = (K * h_ref) ** 3
        print(f"Applying Venkatakrishnan limiter (K={K}, h_ref={h_ref:.3e}, eps^2={eps_sq:.3e})...")
        phi = ComputeVenkatLimiter_Numba(dataFrom_fields, Gradients_From, PointsFrom_3d,
                                         NodesConnectedToNode_From, NumOfNodesConnectedToNode_From,
                                         nDim, eps_sq)
    elif LimiterType in ("NishikawaR3", "R3", "NishikawaR4", "R4", "NishikawaR5", "R5"):
        R = 3 if LimiterType.endswith("3") else (4 if LimiterType.endswith("4") else 5)
        print(f"Applying Nishikawa R{R} limiter (preserves up to order {R})...")
        phi = ComputeNishikawaRLimiter_Numba(dataFrom_fields, Gradients_From, PointsFrom_3d,
                                             NodesConnectedToNode_From, NumOfNodesConnectedToNode_From,
                                             nDim, R)
    else:
        print(f"WARNING: LimiterType '{LimiterType}' not implemented, falling back to BJ.")
        phi = ComputeBJLimiter_Numba(dataFrom_fields, Gradients_From, PointsFrom_3d,
                                     NodesConnectedToNode_From, NumOfNodesConnectedToNode_From, nDim)
    Gradients_From *= phi[..., np.newaxis]
    if Hessians_From is not None:
        Hessians_From *= phi[..., np.newaxis]
    print("Done limiter. Elapsed time ", time.time() - start, "sec  | mean phi =", float(np.mean(phi)),
          " min phi =", float(np.min(phi)))
    return Gradients_From


__all__ = [
    "compute_wls_coefficients_optimized",
    "compute_wls_coefficients_pinv",
    "compute_wls_coefficients_deg2",
    "ComputeWLSGradients_Numba",
    "ComputeWLSHessians_Numba",
    "ComputeVenkatLimiter_Numba",
    "ComputeNishikawaRLimiter_Numba",
    "ComputeBJLimiter_Numba",
    "ApplyLimiterIfRequested",
]
