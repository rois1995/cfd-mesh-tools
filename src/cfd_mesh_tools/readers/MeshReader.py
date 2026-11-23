from .SU2Reader import SU2Reader
from .CGNSReader import CGNSReader
from .ElementsUtilities import ElementsUtilities
from .Utils import setup_logging
import time
import numpy as np
from numba import njit, int32, prange, get_num_threads
from numba.typed import List as NumbaList
from numba.core import types
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from ..linalg.NumbaLinAlg import invert_2x2, invert_3x3

# Define the structured type for sorting 3x int64
vertex_dtype = np.dtype([('x', np.int64), ('y', np.int64), ('z', np.int64)])

# invert_2x2 / invert_3x3 now live in FunScripts.NumbaLinAlg (imported above).


# Outward-CCW (right-hand-normal points outward) faces for the supported 3D
# cells, in VTK/SU2 local-vertex ordering, with the number of tetrahedra each
# cell yields. Used by _cone_subdivide_to_tets below.
#   quad faces, triangle faces, n_tets
_CELL_FACES_3D = {
    'PYRA':  ([(0, 3, 2, 1)],
              [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)], 2),
    'PRISM': ([(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
              [(0, 2, 1), (3, 4, 5)], 3),
    'HEXA':  ([(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
               (1, 2, 6, 5), (2, 3, 7, 6), (0, 4, 7, 3)],
              [], 6),
}


def _cone_subdivide_to_tets(conn, quad_faces, tri_faces, n_tets):
    """
    Conforming subdivision of convex 3D cells (PYRA/PRISM/HEXA) into tetrahedra,
    coordinate-free (depends only on the global node indices).

    Each face is split by its own diagonal through the face's smallest-global-
    index vertex; the cell is then tetrahedralized by coning its globally
    smallest-index vertex `v` to the triangles of the faces that do NOT contain
    `v` (faces that do contain `v` are split through `v`, so coning would only
    produce degenerate tets -- they are dropped). Because every face's diagonal
    is fixed by the global ids on that face alone, two cells sharing a quad face
    split it identically -> the tetrahedralization is conforming.

    Verified exhaustively over every vertex-index ordering (PYRA 120, PRISM 720,
    HEXA 40320): correct tet count, all positively oriented, volumes summing to
    the cell volume, and min-vertex diagonals on every shared face.

    conn       : (n, V) int64 cell connectivity (VTK/SU2 local order).
    quad_faces : list of 4-tuples of local indices (outward CCW).
    tri_faces  : list of 3-tuples of local indices (outward CCW).
    n_tets     : tets produced per cell (PYRA 2, PRISM 3, HEXA 6).
    """
    n = conn.shape[0]
    vmin = conn.min(axis=1)                                  # global-min id per cell

    tri_list = []
    for (a, b, c, d) in quad_faces:
        ga, gb, gc, gd = conn[:, a], conn[:, b], conn[:, c], conn[:, d]
        use_ac = np.minimum(ga, gc) <= np.minimum(gb, gd)    # diagonal a-c vs b-d
        t1 = np.where(use_ac[:, None], np.stack([ga, gb, gc], 1), np.stack([gb, gc, gd], 1))
        t2 = np.where(use_ac[:, None], np.stack([ga, gc, gd], 1), np.stack([gb, gd, ga], 1))
        tri_list.append(t1)
        tri_list.append(t2)
    for (a, b, c) in tri_faces:
        tri_list.append(np.stack([conn[:, a], conn[:, b], conn[:, c]], 1))

    tris = np.stack(tri_list, axis=1)                        # (n, T, 3) global ids
    keep = ~(tris == vmin[:, None, None]).any(axis=2)        # drop triangles touching vmin
    # exactly n_tets triangles survive per cell -> safe reshape (row-major)
    kept = tris[keep].reshape(n, n_tets, 3)
    v = np.broadcast_to(vmin[:, None, None], (n, n_tets, 1))
    # (v, a, b, c) with outward-CCW (a,b,c) on a far face gives positive volume
    return np.concatenate([v, kept], axis=2).reshape(n * n_tets, 4)

@njit(fastmath={'reassoc', 'contract'}, cache=True)
def _normalized_wls_stencil(nodes, neighbors, i, nn, nDim, power, dxbuf, wbuf):
    """Cache dimensionless offsets/weights and return inverse mean spacing.

    Matches the quadratic builder's mean-distance normalization. Empty or
    entirely coincident stencils have zero coefficients; individual coincident
    neighbours contribute a zero constraint. Reject non-finite/unrepresentable
    spacing or weights with a negative return; the caller raises AFTER prange
    (exceptions inside parallel workers do not propagate reliably). The
    helper and its callers allow only reassociation/contraction: full fastmath
    assumes away NaN/Inf and defeats the finite checks.
    """
    if nn == 0:
        return 0.0
    if not np.isfinite(power):
        return -1.0
    h = 0.0
    for k in range(nn):
        j = neighbors[i, k]
        d2 = 0.0
        for d in range(nDim):
            dx = nodes[j, d] - nodes[i, d]
            if not np.isfinite(dx):
                return -1.0
            dxbuf[k, d] = dx
            d2 += dx * dx
        if d2 == 0.0:
            for d in range(nDim):
                if dxbuf[k, d] != 0.0:
                    return -1.0  # Distinct points whose squared distance underflows.
        wbuf[k] = d2  # Reuse the weight buffer to distinguish coincident points below.
        h += np.sqrt(d2)
    h /= nn
    if not np.isfinite(h):
        return -1.0
    if h == 0.0:
        return 0.0
    invh = 1.0 / h
    if not np.isfinite(invh):
        return -1.0
    for k in range(nn):
        d2 = 0.0
        for d in range(nDim):
            dxbuf[k, d] *= invh
            d2 += dxbuf[k, d] * dxbuf[k, d]
        if d2 == 0.0 and wbuf[k] != 0.0:
            return -1.0  # Normalization underflowed a nonzero separation.
        # A coincident point has no derivative information, for any power.
        wbuf[k] = 0.0 if d2 == 0.0 else 1.0 / (d2 ** (power / 2.0) + 1e-12)
        if not np.isfinite(wbuf[k]):
            return -1.0
    return invh


@njit(parallel=True, fastmath={'reassoc', 'contract'}, cache=True)
def compute_wls_coefficients_optimized(nodes, neighbors, nNeighbors, nDim, power=2.0):
    """
    Optimized WLS coefficient computation with Numba parallelization.

    Offsets/weights use local mean spacing h; coefficients are divided by h
    for physical gradient units in the neighbour-minus-centre consumer.
    Empty/all-coincident rows are zero. Singular moments retain the analytic
    inverse helpers' identity fallback, now in dimensionless coordinates: this
    is a finite degraded estimate, not an accurate full-dimensional gradient.
    
    Parameters:
    -----------
    nodes : np.ndarray, shape (n_nodes, 3)
        Node coordinates
    neighbors : np.ndarray, shape (n_nodes, max_neighbors)
        Neighbor connectivity (-1 for invalid)
    nNeighbors : np.ndarray, shape (n_nodes,)
        Number of neighbors for each node
    nDim : int
        Active dimensions (2 or 3)
    power : float
        Power for inverse distance weighting
        
    Returns:
    --------
    coeffs : np.ndarray, shape (n_nodes, max_neighbors, 3)
        WLS coefficients (only [:,:,:nDim] are non-zero)
    """
    n_nodes = nodes.shape[0]
    max_neighbors = neighbors.shape[1]
    coeffs = np.zeros((n_nodes, max_neighbors, 3), dtype=np.float64)
    invalid_rows = 0
    
    # Parallel loop over nodes - this is where parallelization happens
    for i in prange(n_nodes):
        # Local buffers for this node (reduces memory allocations)
        M = np.zeros((nDim, nDim), dtype=np.float64)
        M_inv = np.zeros((nDim, nDim), dtype=np.float64)
        nn = nNeighbors[i]
        # Cache normalized offsets and identical weights for both passes.
        dxbuf = np.empty((max_neighbors, nDim), dtype=np.float64)
        wbuf = np.empty(max_neighbors, dtype=np.float64)

        invh = _normalized_wls_stencil(
            nodes, neighbors, i, nn, nDim, power, dxbuf, wbuf)
        if invh < 0.0:
            invalid_rows += 1
            continue
        if invh == 0.0:
            continue

        # Build the dimensionless weighted least squares matrix M.
        for k in range(nn):
            weight = wbuf[k]
            # Accumulate weighted outer product into M
            for p in range(nDim):
                for q in range(nDim):
                    M[p, q] += weight * dxbuf[k, p] * dxbuf[k, q]

        # Invert M (dimension-specific analytical inversion)
        if nDim == 2:
            invert_2x2(M, M_inv)
        else:  # nDim == 3
            invert_3x3(M, M_inv)

        # Compute coefficients for each neighbor (dx/weight reused from the moment pass)
        for k in range(nn):
            weight = wbuf[k]
            for p in range(nDim):
                acc = 0.0
                for q in range(nDim):
                    acc += M_inv[p, q] * (weight * dxbuf[k, q])
                value = acc * invh
                if not np.isfinite(value):
                    invalid_rows += 1
                coeffs[i, k, p] = value

    if invalid_rows:
        raise ValueError("WLS requires finite power, displacements, spacing, weights and coefficients representable in float64")
    return coeffs


@njit(parallel=True, fastmath={'reassoc', 'contract'}, cache=True)
def compute_wls_coefficients_pinv(nodes, neighbors, nNeighbors, manifoldDim, ambientDim, power=2.0):
    """
    WLS gradient coefficients on a lower-dimensional manifold (a boundary curve
    in 2D, or a surface in 3D) embedded in `ambientDim` space.

    On a manifold the ambient moment matrix M = sum_k w_k dx_k (x) dx_k is
    rank-deficient (rank = manifoldDim), so the plain inverse is singular and a
    naive coordinate drop is wrong for any non-axis-aligned manifold. Instead we
    use the rank-truncated pseudo-inverse that keeps only the largest
    `manifoldDim` eigenvalues (the local tangent plane). The reconstructed
    gradient is the tangential gradient, expressed in ambient components.

    Returns coeffs (n_nodes, max_neighbors, 3) such that
        grad_ambient[i] = sum_k coeffs[i, k, :] * (phi[nbr] - phi[i])
    lies in the local tangent plane. Local mean-spacing normalization makes
    the weights dimensionless; division by h restores physical gradient units.
    Empty/all-coincident rows are zero. Eigenvalues below the existing relative
    tolerance are discarded, including any missing tangent directions.
    """
    n_nodes = nodes.shape[0]
    max_neighbors = neighbors.shape[1]
    coeffs = np.zeros((n_nodes, max_neighbors, 3), dtype=np.float64)
    invalid_rows = 0

    for i in prange(n_nodes):
        nn = nNeighbors[i]

        dxbuf = np.empty((max_neighbors, ambientDim), dtype=np.float64)
        wbuf = np.empty(max_neighbors, dtype=np.float64)
        invh = _normalized_wls_stencil(
            nodes, neighbors, i, nn, ambientDim, power, dxbuf, wbuf)
        if invh < 0.0:
            invalid_rows += 1
            continue
        if invh == 0.0:
            continue

        # Build the dimensionless weighted moment matrix.
        M = np.zeros((ambientDim, ambientDim), dtype=np.float64)
        for k in range(nn):
            w = wbuf[k]
            for p in range(ambientDim):
                for q in range(ambientDim):
                    M[p, q] += w * dxbuf[k, p] * dxbuf[k, q]

        # Symmetric eigendecomposition (eigenvalues ascending).
        evals, evecs = np.linalg.eigh(M)

        # Rank-truncated pseudo-inverse: keep the largest manifoldDim eigenvalues
        # (the tangent directions), discard the smallest (the normal direction).
        Mpinv = np.zeros((ambientDim, ambientDim), dtype=np.float64)
        lam_max = evals[ambientDim - 1]
        tol = 1e-12 * lam_max if lam_max > 0.0 else 0.0
        for e in range(ambientDim - manifoldDim, ambientDim):
            lam = evals[e]
            if lam > tol:
                inv = 1.0 / lam
                for p in range(ambientDim):
                    for q in range(ambientDim):
                        Mpinv[p, q] += inv * evecs[p, e] * evecs[q, e]

        # Reuse exactly the moment-pass offsets/weights; restore physical units.
        for k in range(nn):
            w = wbuf[k]
            for p in range(ambientDim):
                acc = 0.0
                for q in range(ambientDim):
                    acc += Mpinv[p, q] * w * dxbuf[k, q]
                value = acc * invh
                if not np.isfinite(value):
                    invalid_rows += 1
                coeffs[i, k, p] = value

    if invalid_rows:
        raise ValueError("WLS requires finite power, displacements, spacing, weights and coefficients representable in float64")
    return coeffs


@njit(parallel=True, fastmath=True, cache=True)
def compute_wls_coefficients_deg2(nodes, neighbors, nNeighbors, nDim, power, fallback_coeffs, want_hessian):
    """
    Robust degree-2 (quadratic) WLS gradient (and optionally Hessian) coefficients.

    Per node we fit  phi(r) ~ phi_0 + g.dr + 0.5 dr^T H dr  in the least-squares
    sense over the node's (2-ring) neighbours, from a SINGLE symmetric
    eigendecomposition of the normal matrix. We always return the gradient rows g;
    when `want_hessian` is set we ALSO return the Hessian rows H from that same
    eigendecomposition (this used to be a second, byte-identical builder that redid
    the whole A/W/M/eigh per node - now fused so the eigh runs once).

    The fit is exact for quadratic fields, so g is O(h^2) accurate (vs O(h) for the
    linear/naive builder). Because least squares is linear in the data, both the
    gradient and the Hessian are linear functionals of the neighbour value-diffs:

        grad[i]       = sum_k coeffs[i, k, :nDim] * (phi[nbr_k] - phi[i])
        H[i, comp]    = sum_k hcoeffs[i, k, comp] * (phi[nbr_k] - phi[i])

    so `coeffs` has the identical meaning/shape as the degree-1 coeffs (consumed by
    `ComputeWLSGradients_Numba` unchanged), and `hcoeffs` packs the symmetric Hessian
    upper-triangular:  2D (nH=3) [Hxx, Hxy, Hyy];  3D (nH=6) [Hxx, Hxy, Hxz, Hyy,
    Hyz, Hzz]  (consumed by `ComputeWLSHessians_Numba`).

    Returns (coeffs, hcoeffs). When want_hessian is False, hcoeffs is a (1,1,1) dummy
    (never allocated at full size, so Second-order runs pay no extra RAM).

    Robustness:
      * offsets are normalised by the local mean spacing h so the linear (~h) and
        quadratic (~h^2) basis columns are comparably scaled (keeps M well
        conditioned on fine meshes); the recovered gradient is rescaled by 1/h,
        the Hessian by 1/h^2.
      * M = A^T W A (m x m, m = 5 in 2D / 9 in 3D) is inverted via a symmetric
        eigendecomposition with small-eigenvalue truncation.
      * if a node has too few neighbours (nn < m) or the patch is rank deficient
        (degenerate / coplanar), the gradient falls back to that node's degree-1
        coefficients in `fallback_coeffs` (so degree-2 is never worse than naive)
        and the Hessian is left ZERO (local linear / 2nd-order reconstruction).
    """
    n_nodes = nodes.shape[0]
    max_neighbors = neighbors.shape[1]
    coeffs = np.zeros((n_nodes, max_neighbors, 3), dtype=np.float64)

    # Number of polynomial basis columns (value-differences, so no constant term):
    # 2D -> [sx, sy, 0.5 sx^2, sx sy, 0.5 sy^2]                              (m = 5)
    # 3D -> [sx, sy, sz, 0.5 sx^2, sx sy, sx sz, 0.5 sy^2, sy sz, 0.5 sz^2]  (m = 9)
    m = 5 if nDim == 2 else 9
    nH = m - nDim  # 3 in 2D, 6 in 3D
    if want_hessian:
        hcoeffs = np.zeros((n_nodes, max_neighbors, nH), dtype=np.float64)
    else:
        hcoeffs = np.zeros((1, 1, 1), dtype=np.float64)

    for i in prange(n_nodes):
        nn = nNeighbors[i]

        # Not enough neighbours to fit the quadratic -> degree-1 grad, zero Hessian.
        if nn < m:
            for k in range(nn):                 # k>=nn: coeffs already 0 == fallback_coeffs (0)
                for d in range(nDim):
                    coeffs[i, k, d] = fallback_coeffs[i, k, d]
            continue

        # Local mean spacing for column normalisation.
        h = 0.0
        for k in range(nn):
            j = neighbors[i, k]
            d2 = 0.0
            for d in range(nDim):
                dd = nodes[j, d] - nodes[i, d]
                d2 += dd * dd
            h += np.sqrt(d2)
        h = h / nn
        if h <= 0.0:
            for k in range(nn):                 # k>=nn: coeffs already 0 == fallback_coeffs (0)
                for d in range(nDim):
                    coeffs[i, k, d] = fallback_coeffs[i, k, d]
            continue
        invh = 1.0 / h
        invh2 = invh * invh

        # Build the (normalised) design matrix rows and weights.
        A = np.zeros((nn, m), dtype=np.float64)
        w = np.zeros(nn, dtype=np.float64)
        for k in range(nn):
            j = neighbors[i, k]
            dx = np.zeros(nDim, dtype=np.float64)
            d2 = 0.0
            for d in range(nDim):
                dx[d] = (nodes[j, d] - nodes[i, d]) * invh
                d2 += dx[d] * dx[d]
            # Dimensionless inverse-distance weight on the normalized offsets.
            w[k] = 1.0 / (d2 ** (power / 2.0) + 1e-12)
            if nDim == 2:
                A[k, 0] = dx[0]
                A[k, 1] = dx[1]
                A[k, 2] = 0.5 * dx[0] * dx[0]
                A[k, 3] = dx[0] * dx[1]
                A[k, 4] = 0.5 * dx[1] * dx[1]
            else:
                A[k, 0] = dx[0]
                A[k, 1] = dx[1]
                A[k, 2] = dx[2]
                A[k, 3] = 0.5 * dx[0] * dx[0]
                A[k, 4] = dx[0] * dx[1]
                A[k, 5] = dx[0] * dx[2]
                A[k, 6] = 0.5 * dx[1] * dx[1]
                A[k, 7] = dx[1] * dx[2]
                A[k, 8] = 0.5 * dx[2] * dx[2]

        # Weighted normal matrix M = A^T W A (symmetric, m x m).
        M = np.zeros((m, m), dtype=np.float64)
        for k in range(nn):
            wk = w[k]
            for p in range(m):
                akp = A[k, p] * wk
                for q in range(m):
                    M[p, q] += akp * A[k, q]

        # Symmetric eigendecomposition (ascending eigenvalues) - ONE per node,
        # shared by the gradient and Hessian row extraction below.
        evals, evecs = np.linalg.eigh(M)
        lam_max = evals[m - 1]
        tol = 1e-10 * lam_max if lam_max > 0.0 else 0.0

        # Rank check: degenerate patch -> degree-1 grad, zero Hessian.
        rank = 0
        for e in range(m):
            if evals[e] > tol:
                rank += 1
        if rank < m:
            for k in range(nn):                 # k>=nn: coeffs already 0 == fallback_coeffs (0)
                for d in range(nDim):
                    coeffs[i, k, d] = fallback_coeffs[i, k, d]
            continue

        # Truncated inverse rows we need: gradient rows [0..nDim), and (if wanted)
        # Hessian rows [nDim..m). Extracted from the single eigendecomposition.
        nrows = m if want_hessian else nDim
        Minv = np.zeros((nrows, m), dtype=np.float64)
        for e in range(m):
            lam = evals[e]
            if lam > tol:
                inv = 1.0 / lam
                for p in range(nrows):
                    vpe = evecs[p, e] * inv
                    for q in range(m):
                        Minv[p, q] += vpe * evecs[q, e]

        # Gradient coeffs = (Minv @ A^T W)[:nDim], rescaled to physical units.
        for k in range(nn):
            wk = w[k]
            for p in range(nDim):
                acc = 0.0
                for q in range(m):
                    acc += Minv[p, q] * A[k, q]
                coeffs[i, k, p] = acc * wk * invh

        # Hessian coeffs = (Minv @ A^T W)[nDim..m], rescaled by invh^2. The basis
        # columns carry the 0.5 factor for the pure-square terms, so the solved
        # coefficient of that column IS H_dd (not 2*H_dd); cross terms map to H_ij
        # directly. Rows are already in upper-triangular order.
        if want_hessian:
            for k in range(nn):
                wk = w[k]
                for c in range(nH):
                    acc = 0.0
                    for q in range(m):
                        acc += Minv[nDim + c, q] * A[k, q]
                    hcoeffs[i, k, c] = acc * wk * invh2

    return coeffs, hcoeffs


@njit(parallel=True, cache=True)
def build_inverse_map_local(sort_idx, group_ids, n_local):
    """Builds map 0..M -> 0..U for the local slice."""
    mapping = np.empty(n_local, dtype=np.int64)
    for i in prange(n_local):
        mapping[sort_idx[i]] = group_ids[i]
    return mapping

# --- The New Logic: Split Mapping ---
@njit(parallel=True, cache=True)
def apply_map_with_offset(cells, local_map, n_fixed):
    """
    Applies mapping logic:
    - If ID < n_fixed: Keep it (it's a fixed grid node).
    - If ID >= n_fixed: It's a generated point. Use the local map + offset.
    """
    n_cells = cells.shape[0]
    n_nodes = cells.shape[1]
    out_cells = np.empty_like(cells)
    
    for i in prange(n_cells):
        for j in range(n_nodes):
            original_id = cells[i, j]
            
            if original_id < n_fixed:
                # Preserved Fixed Point
                out_cells[i, j] = original_id
            else:
                # Generated Point (Midpoint/Centroid)
                # 1. Convert to local index
                local_id = original_id - n_fixed
                # 2. Find unique ID in local space
                unique_local_id = local_map[local_id]
                # 3. Shift back to global space (after the fixed block)
                out_cells[i, j] = n_fixed + unique_local_id
                
    return out_cells

@njit(parallel=True, cache=True)
def quantize_parallel(coords, min_val, inv_tol, out):
    """
    Quantizes coordinates directly into the output array to save memory.
    Parallelized for performance.
    """
    n = coords.shape[0]
    # Loop over flattened or rows
    for i in prange(n):
        # Unroll for 3D specifically (or generic loop)
        out[i, 0] = int(round((coords[i, 0] - min_val[0]) * inv_tol))
        out[i, 1] = int(round((coords[i, 1] - min_val[1]) * inv_tol))
        out[i, 2] = int(round((coords[i, 2] - min_val[2]) * inv_tol))

@njit(parallel=True, cache=True)
def compute_unique_mask(sorted_coords, sort_idx):
    """
    Scans sorted coordinates to identify unique vertices.
    Returns a boolean mask where True indicates the start of a new unique vertex.
    """
    n = len(sort_idx)
    mask = np.empty(n, dtype=np.bool_)
    mask[0] = True
    
    # Parallel check of neighbors
    # We compare index i with i-1.
    for i in prange(1, n):
        curr_idx = sort_idx[i]
        prev_idx = sort_idx[i-1]
        
        # Check for equality across all 3 dimensions
        row_match = (
            sorted_coords[curr_idx, 0] == sorted_coords[prev_idx, 0] and
            sorted_coords[curr_idx, 1] == sorted_coords[prev_idx, 1] and
            sorted_coords[curr_idx, 2] == sorted_coords[prev_idx, 2]
        )
        mask[i] = not row_match
        
    return mask


def deadCode(fn):
    """DEAD-CODE marker: no caller anywhere in the codebase (kept for reference).
    No-op passthrough; grep '@deadCode' to list flagged functions."""
    return fn

@deadCode
@njit(parallel=True, cache=True)
def build_inverse_map(sort_idx, group_ids, n_points):
    """
    Scatters the compressed group IDs back to the original vertex indices.
    """
    mapping = np.empty(n_points, dtype=np.int64)
    for i in prange(n_points):
        # The vertex at original index `sort_idx[i]` belongs to `group_ids[i]`
        mapping[sort_idx[i]] = group_ids[i]
    return mapping

@deadCode
@njit(parallel=True, cache=True)
def map_cells(cells, mapping):
    """
    Updates the cell connectivity with new vertex indices.
    """
    n_cells = cells.shape[0]
    n_nodes = cells.shape[1]
    new_cells = np.empty_like(cells)
    
    for i in prange(n_cells):
        for j in range(n_nodes):
            new_cells[i, j] = mapping[cells[i, j]]
    return new_cells


@njit(parallel=True, cache=True)
def apply_map_ckdtree(cells, local_map, n_fixed, nDim):
    """
    Applies the suffix map to the cells.
    """
    n_cells = cells.shape[0]
    out_cells = np.empty_like(cells)
    
    for i in prange(n_cells):
        for j in range(nDim + 1):
            val = cells[i, j]
            if val < n_fixed:
                out_cells[i, j] = val
            else:
                # Map global index to local suffix index
                local_idx = val - n_fixed
                # Get new local unique index
                new_local = local_map[local_idx]
                # Map back to global
                out_cells[i, j] = n_fixed + new_local
                
    return out_cells


@njit(parallel=True, cache=True)
def fill_padded_cells_and_mask(indices, indptr, n_nodes, max_degree):
    """
    Parallel Numba kernel to fill:
    1. The padded connectivity array (with -1 padding)
    2. The boolean mask (True where data exists, False where padding)
    """
    # Initialize arrays
    padded_conn = np.full((n_nodes, max_degree), -1, dtype=np.int32)
    mask = np.zeros((n_nodes, max_degree), dtype=np.bool_)
    
    for i in prange(n_nodes):
        start = indptr[i]
        end = indptr[i+1]
        length = end - start
        
        if length > 0:
            # Clamp to max_degree just in case
            copy_len = min(length, max_degree)
            
            # 1. Copy indices
            padded_conn[i, :copy_len] = indices[start : start + copy_len]
            
            # 2. Set mask to True for valid entries
            mask[i, :copy_len] = True
            
    return padded_conn, mask


# ---------------------------------------------------------------------------------------------------
# Fully-numba node->node adjacency (WLS operator connectivity) from an edge list, used by
# SetNodesConnectedToNode for LARGE meshes (>= _NODESCONN_NUMBA_MIN nodes). Mirrors the scipy
# CSR + A@A build but with per-chunk stamped-dedup CSR kernels (like build_cell_adjacency in the
# interpolation code) -- MEASURED ~2.6x @64^3 / ~4.8x @128^3 and lower peak, IDENTICAL adjacency;
# scipy stays faster on small/2D meshes (dispatch below). Self-contained here to avoid a circular
# import (FunScripts.InterpSolution_Fun imports this module).
# ---------------------------------------------------------------------------------------------------
_NODESCONN_NUMBA_MIN = 100_000


@njit(parallel=True, cache=True)
def _nc_edge_count(src, nchunks, hist):
    """Per-chunk histogram of the edge source nodes (race-free: each chunk writes its own row)."""
    m = src.shape[0]
    for ci in prange(nchunks):
        s0 = (m * ci) // nchunks; s1 = (m * (ci + 1)) // nchunks
        h = hist[ci]
        for i in range(s0, s1):
            h[src[i]] += 1


@njit(parallel=True, cache=True)
def _nc_edge_chunkoff(hist, offsets, nchunks, n_nodes, chunk_off):
    for u in prange(n_nodes):
        acc = offsets[u]
        for ci in range(nchunks):
            chunk_off[ci, u] = acc
            acc += hist[ci, u]


@njit(parallel=True, cache=True)
def _nc_edge_scatter(src, chunk_off, edge_idx, nchunks):
    """Group edge indices by their source node (node -> its edges CSR), per-chunk cursors -> no races."""
    m = src.shape[0]
    for ci in prange(nchunks):
        cur = chunk_off[ci].copy()
        s0 = (m * ci) // nchunks; s1 = (m * (ci + 1)) // nchunks
        for i in range(s0, s1):
            u = src[i]; edge_idx[cur[u]] = i; cur[u] += 1


@njit(parallel=True, cache=True)
def _nc_distinct(edge_off, edge_idx, dst, n_nodes, nchunks, vis, out_off, out_idx, do_fill):
    """1-ring: distinct neighbour (dst) per node, stamped dedup (stamp = node index -> no clearing)."""
    for ci in prange(nchunks):
        s0 = (n_nodes * ci) // nchunks; s1 = (n_nodes * (ci + 1)) // nchunks
        v = vis[ci]
        for u in range(s0, s1):
            c = 0
            base = out_off[u] if do_fill == 1 else 0
            for k in range(edge_off[u], edge_off[u + 1]):
                w = dst[edge_idx[k]]
                if w != u and v[w] != u:
                    v[w] = u
                    if do_fill == 1:
                        out_idx[base + c] = w
                    c += 1
            if do_fill == 0:
                out_off[u + 1] = c


@njit(parallel=True, cache=True)
def _nc_ring2(off1, idx1, n_nodes, nchunks, vis, out_off, out_idx, do_fill):
    """2-ring: the 1-ring plus the 1-ring of each 1-ring member (<= 2 hops), excl self. Stamped dedup."""
    for ci in prange(nchunks):
        s0 = (n_nodes * ci) // nchunks; s1 = (n_nodes * (ci + 1)) // nchunks
        v = vis[ci]
        for u in range(s0, s1):
            c = 0
            base = out_off[u] if do_fill == 1 else 0
            for k1 in range(off1[u], off1[u + 1]):
                m = idx1[k1]
                if v[m] != u:
                    v[m] = u
                    if do_fill == 1:
                        out_idx[base + c] = m
                    c += 1
                for k2 in range(off1[m], off1[m + 1]):
                    p = idx1[k2]
                    if p != u and v[p] != u:
                        v[p] = u
                        if do_fill == 1:
                            out_idx[base + c] = p
                        c += 1
            if do_fill == 0:
                out_off[u + 1] = c


def build_nodes_connected_numba(src, dst, n_nodes, degree):
    """Node->node adjacency CSR (offsets, indices) from an edge list, fully numba. degree>=2 -> 2-ring.
    Returns (offsets int64, indices int32, degrees int32, maxdeg). See _NODESCONN_NUMBA_MIN dispatch."""
    src = np.ascontiguousarray(src.astype(np.int32)); dst = np.ascontiguousarray(dst.astype(np.int32))
    nch = max(1, get_num_threads())
    # node -> edges CSR
    hist = np.zeros((nch, n_nodes), dtype=np.int64)
    _nc_edge_count(src, nch, hist)
    edge_off = np.zeros(n_nodes + 1, dtype=np.int64)
    edge_off[1:] = np.cumsum(hist.sum(axis=0))
    chunk_off = np.empty((nch, n_nodes), dtype=np.int64)
    _nc_edge_chunkoff(hist, edge_off, nch, n_nodes, chunk_off)
    edge_idx = np.empty(src.shape[0], dtype=np.int32)
    _nc_edge_scatter(src, chunk_off, edge_idx, nch)
    # 1-ring (distinct neighbours)
    vis = np.full((nch, n_nodes), -1, dtype=np.int64)
    off1 = np.zeros(n_nodes + 1, dtype=np.int64)
    _nc_distinct(edge_off, edge_idx, dst, n_nodes, nch, vis, off1, np.empty(0, np.int32), 0)
    off1 = np.cumsum(off1)
    idx1 = np.empty(int(off1[-1]), dtype=np.int32); vis[:] = -1
    _nc_distinct(edge_off, edge_idx, dst, n_nodes, nch, vis, off1, idx1, 1)
    if degree >= 2:
        vis[:] = -1
        off2 = np.zeros(n_nodes + 1, dtype=np.int64)
        _nc_ring2(off1, idx1, n_nodes, nch, vis, off2, np.empty(0, np.int32), 0)
        off2 = np.cumsum(off2)
        idx2 = np.empty(int(off2[-1]), dtype=np.int32); vis[:] = -1
        _nc_ring2(off1, idx1, n_nodes, nch, vis, off2, idx2, 1)
        off1, idx1 = off2, idx2
    degrees = np.diff(off1).astype(np.int32)
    maxdeg = int(degrees.max()) if degrees.size else 0
    return off1, idx1, degrees, maxdeg


class MeshReader:

    def __init__(self, fileName, meshType="SU2", nDim=2):

        self.Logger = setup_logging(filenameOut='output.log', filenameErr='error.log', verbose=True)

        if meshType == "SU2":
            self.Mesh = SU2Reader(fileName, self.Logger, nDim)
        elif meshType == "CGNS":
            self.Mesh = CGNSReader(fileName, self.Logger, nDim)
        elif meshType == "VTU":
            from .VTUReader import VTUReader
            self.Mesh = VTUReader(fileName, self.Logger, nDim)
        else:
            print("ERROR! Mesh type", meshType, "not supported!")
            exit(1)

        self.Nodes = None
        self.Elements = None
        self.cell_types = None
        self.ElementsUtilities = ElementsUtilities(self.Logger)
        self.nDim = nDim
        self.meshType = meshType
        self.WLS_Coeffs = None
        # Degree-2 WLS Hessian coefficients (upper-tri packed), built in the SAME
        # eigendecomposition as WLS_Coeffs when a "Third"-order path asks for it;
        # None otherwise. Consumed by ComputeWLSHessians_Numba.
        self.WLS_Hess_Coeffs = None
        # WLS gradient reconstruction degree: 1 = linear (naive, exact-for-linear,
        # O(h) gradient on quadratics), 2 = quadratic (robust, exact-for-quadratics,
        # O(h^2) gradient). Degree 2 uses a 2-ring neighbourhood and a per-node
        # rank fallback to degree 1. Set from options['WLSDegree'].
        self.WLSDegree = 1
        self.NodesConnectedToNode, self.NumOfNodesConnectedToNode, self.NodesConnectedToNodeMask = None, None, None
        self.CellsConnectedToNode, self.NumOfCellsConnectedToNode, self.CellsConnectedToNodeMask = None, None, None

    def extractMarkersIndices(self, Markers2Extract=None):
        return self.Mesh.extractMarkersIndices(Markers2Extract)

    def extractMarkerElements(self, Markers2Extract=None):
        return self.Mesh.extractMarkerElements(Markers2Extract)

    def extractCoordsFromIDs(self, IDs2Extract):
        return self.Mesh.extractCoordsFromIDs(IDs2Extract)

    def extractAllCoords(self, zone2Read=None):
        self.Nodes = self.Mesh.extractAllCoords(zone2Read=zone2Read)
        return self.Nodes

    def SetCellsConnectedToNode(self):
        n_nodes = len(self.Nodes)
        
        # --- Step 1: Gather all Node -> Cell connections ---
        
        # Pre-calculate total size to allocate once
        total_entries = 0
        work_items = []
        
        for section_name, elem_data in self.Elements.items():
            connectivity = elem_data['connectivity']
            start_idx, end_idx = elem_data['range']
            
            total_entries += connectivity.size
            work_items.append((connectivity, start_idx, end_idx))
            
        # Allocate monolithic arrays
        full_node_indices = np.empty(total_entries, dtype=np.int32)
        full_elem_indices = np.empty(total_entries, dtype=np.int32)
        
        cursor = 0
        
        # Fill arrays (Vectorized)
        for connectivity, start_idx, end_idx in work_items:
            n_elems, n_nodes_per_elem = connectivity.shape
            block_size = connectivity.size
            
            # Create global element IDs: [100, 101, ...]
            global_elem_ids = np.arange(start_idx, end_idx, dtype=np.int32)
            
            # Repeat them to match connectivity: [100, 100, ..., 101, 101, ...]
            elem_ids_repeated = np.repeat(global_elem_ids, n_nodes_per_elem)
            
            # Flatten nodes: [n1, n2, ..., n1, n2, ...]
            node_ids_flat = connectivity.flatten()
            
            full_node_indices[cursor : cursor + block_size] = node_ids_flat
            full_elem_indices[cursor : cursor + block_size] = elem_ids_repeated
            
            cursor += block_size
            
        # --- Step 2: Build CSR Matrix for Grouping ---
        
        # Rows=Nodes, Cols=Elements. 
        # This groups all elements connected to Node i into row i.
        data = np.ones(total_entries, dtype=bool)
        max_elem_id = np.max(full_elem_indices) if total_entries > 0 else 0
        
        cell_matrix = sp.csr_matrix((data, (full_node_indices, full_elem_indices)), 
                                    shape=(n_nodes, max_elem_id + 1))
        
        # --- Step 3: Generate Outputs ---
        
        indptr = cell_matrix.indptr
        indices = cell_matrix.indices
        
        # Output A: Number of cells per node
        # This is exactly the difference between adjacent pointers in CSR
        self.NumOfCellsConnectedToNode = np.diff(indptr).astype(np.int32)
        
        # Calculate max degree to size the rectangular arrays
        max_degree = self.NumOfCellsConnectedToNode.max() if len(self.NumOfCellsConnectedToNode) > 0 else 0
        
        # Output B & C: Rectangular Connectivity & Boolean Mask
        # Use Numba to fill both in one parallel pass
        self.CellsConnectedToNode, self.CellsConnectedToNodeMask = fill_padded_cells_and_mask(
            indices, indptr, n_nodes, max_degree
        )
        
    
    def SetNodesConnectedToNode(self):
        n_nodes = len(self.Nodes)
        
        # Lists to collect all edges (u -> v) before building the graph
        all_src_indices = []
        all_dst_indices = []

        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity'] # Shape: (n_elems, n_nodes_per_elem)
            
            # Mapping: local_node_idx -> [local_neighbor_1, local_neighbor_2, ...]
            # This is your NODESCONNECTIONS_PER_ELEMENT for this type
            local_conn_map = self.ElementsUtilities.NODESCONNECTIONS_PER_ELEMENT[self.meshType].get(elem_type)
            
            if local_conn_map is None:
                continue

            # VECTORIZATION: Process each local node column for ALL elements at once
            for local_node_idx, local_neighbors in enumerate(local_conn_map):
                if not local_neighbors:
                    continue
                    
                # 1. Get the column of global IDs for this local node across all elements
                # Shape: (n_elems,)
                src_nodes = connectivity[:, local_node_idx]
                
                # 2. Get columns of neighbor global IDs
                # Shape: (n_elems, n_local_neighbors)
                dst_nodes_block = connectivity[:, local_neighbors]
                
                # 3. Repeat source nodes to match the number of neighbors
                # If node I has 3 neighbors, we repeat I three times: [I, I, I]
                n_neighbors = len(local_neighbors)
                src_repeated = np.repeat(src_nodes, n_neighbors)
                
                # 4. Flatten the destination block to match
                dst_flattened = dst_nodes_block.flatten() # Row-major: (elem0_n1, elem0_n2, elem1_n1...)
                
                all_src_indices.append(src_repeated)
                all_dst_indices.append(dst_flattened)

        # Concatenate all collected edges from all element types
        if not all_src_indices:
            self.NodesConnectedToNode = np.full((n_nodes, 0), -1, dtype=np.int32)
            return

        full_src = np.concatenate(all_src_indices)
        full_dst = np.concatenate(all_dst_indices)

        # SIZE DISPATCH: large meshes (>= _NODESCONN_NUMBA_MIN nodes, i.e. 64^3+) use the fully-numba
        # edge-list adjacency build (MEASURED ~2.6x @64^3 / ~4.8x @128^3, identical output); small/2D
        # meshes keep the scipy CSR + A@A build below, which is faster at that scale and avoids the
        # numba per-chunk overhead. Both produce the identical (padded) neighbour array + mask.
        if n_nodes >= _NODESCONN_NUMBA_MIN:
            off, indices, degrees, max_degree = build_nodes_connected_numba(
                full_src, full_dst, n_nodes, self.WLSDegree)
            self.NumOfNodesConnectedToNode = degrees
            self.NodesConnectedToNode, self.NodesConnectedToNodeMask = fill_padded_cells_and_mask(
                indices, off, n_nodes, max_degree)
            return

        # Build Sparse Matrix (CSR) to handle aggregation and deduplication automatically
        # Duplicate entries (e.g., edge 2-3 appearing in multiple elements) are merged
        # data=True/1 makes it structural
        data = np.ones(len(full_src), dtype=bool)
        adj_matrix = sp.csr_matrix((data, (full_src, full_dst)), shape=(n_nodes, n_nodes))

        # Degree-2 (quadratic) WLS needs >= 5 (2D) / 9 (3D) well-distributed
        # neighbours per node; the 1-ring is too small on boundary/corner nodes.
        # Expand to the 2-ring (neighbours-of-neighbours) by squaring the boolean
        # adjacency: A2 = A | A@A, with the diagonal removed. Only when WLSDegree>=2
        # so the degree-1 path keeps its exact 1-ring behaviour.
        if self.WLSDegree >= 2:
            adj_matrix = (adj_matrix + adj_matrix.dot(adj_matrix)).astype(bool).tocsr()
            adj_matrix.setdiag(False)
            adj_matrix.eliminate_zeros()

        # --- Convert CSR to your Padded Rectangular Array Format ---
        
        # indptr points to the start of each row's indices
        indptr = adj_matrix.indptr
        indices = adj_matrix.indices
        
        # Calculate degrees (number of neighbors) for each node
        degrees = np.diff(indptr)
        self.NumOfNodesConnectedToNode = degrees
        
        max_degree = degrees.max() if len(degrees) > 0 else 0

        # Fill the padded rectangular neighbour array (-1 padded) and its mask via the
        # SAME parallel Numba kernel the cell path uses (SetCellsConnectedToNode above).
        # This replaces the previous pure-Python O(n_nodes) row-copy loop, which on the
        # augmented 2-ring mesh was millions of Python-level iterations run for every WLS
        # build. Result is bit-identical to the old loop (max_degree = degrees.max(), so
        # no row is ever clamped); args mirror the cell path for a shared cached compile.
        self.NodesConnectedToNode, self.NodesConnectedToNodeMask = fill_padded_cells_and_mask(
            indices, indptr, n_nodes, max_degree)

    def ComputeWLSCoefficients_Numba(self, power=2.0, want_hessian=False):
        """
        Wrapper method for optimized WLS coefficient computation.

        WLSDegree == 1 -> linear (naive) builder.
        WLSDegree == 2 -> robust quadratic builder, with the linear coefficients
        supplied as the per-node rank-deficient fallback. Both store the same
        (n_nodes, max_neighbors, 3) WLS_Coeffs consumed by ComputeWLSGradients_Numba.

        want_hessian (degree-2 only) additionally builds WLS_Hess_Coeffs from the
        SAME per-node eigendecomposition (no second pass); used by "Third"-order
        Hessian reconstruction. Ignored at degree 1 (no quadratic fit -> no Hessian).
        """
        deg1 = compute_wls_coefficients_optimized(
            self.Nodes,
            self.NodesConnectedToNode,
            self.NumOfNodesConnectedToNode,
            self.nDim,
            power
        )
        if self.WLSDegree >= 2:
            self.WLS_Coeffs, hcoeffs = compute_wls_coefficients_deg2(
                self.Nodes,
                self.NodesConnectedToNode,
                self.NumOfNodesConnectedToNode,
                self.nDim,
                power,
                deg1,
                want_hessian
            )
            self.WLS_Hess_Coeffs = hcoeffs if want_hessian else None
        else:
            self.WLS_Coeffs = deg1
            self.WLS_Hess_Coeffs = None

    def ComputeWLSCoefficients_Surface_Numba(self, manifoldDim, power=2.0):
        """
        WLS coefficients for a boundary manifold (curve in 2D, surface in 3D)
        embedded in self.nDim ambient space. Uses the tangent-plane (rank-
        truncated pseudo-inverse) reconstruction. manifoldDim = self.nDim - 1.
        """
        self.WLS_Coeffs = compute_wls_coefficients_pinv(
            self.Nodes,
            self.NodesConnectedToNode,
            self.NumOfNodesConnectedToNode,
            manifoldDim,
            self.nDim,
            power
        )

    def build_DualControlVolumes_2D(self, origData, modifyOrigData=False):

        origData = np.transpose(origData)
        n_nodes = len(self.Nodes)

        # Let's start from computing the control volumes associated 
        # to each node and their elements.

        start = time.time()
        self.Logger.info("Initializing ControlVolumes per Node dictionary...")
        
        maxNPoissibleCVs = 0
        maxNNodes = 0
        
        # I want to perform two passages. In the first one I assign indices to the Cell Centers and to the Edges MidPoints
        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']
            edges_per_elem = self.ElementsUtilities.EDGES_PER_ELEMENT[self.meshType].get(elem_type, 0)
            true_edges_per_elem = edges_per_elem[0:int(len(edges_per_elem)/2)]
            
            maxNPoissibleCVs += connectivity.shape[0]*len(true_edges_per_elem)*2
            maxNNodes += connectivity.shape[0]*len(true_edges_per_elem)

        
        CVElements = np.zeros((maxNPoissibleCVs, 3), dtype=np.int32)
        CVAssociatedToOrigPoints = np.zeros((maxNPoissibleCVs, ), dtype=np.int32)

        # Then I have to augment the nodes of the grid with the cell centers and the edge mid points
        # I am not sure that maxNPoissibleCVs is the correct estimate for the number of edge mid-points
        # I have to make sure that if there are redundant nodes then I have to remove them
        NewNodes = np.zeros((n_nodes+len(self.cell_types)+maxNNodes, 3), dtype=np.float64)
        NewNodes[0:n_nodes, :] = self.Nodes
        NewData = None
        if modifyOrigData:
            NewData = np.zeros((n_nodes+len(self.cell_types)+maxNNodes, origData.shape[1]), dtype=np.float64)
            NewData[0:n_nodes, :] = origData

        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")

        StartCellCenterIndex = n_nodes
        StartEdgeMidPointIndex = n_nodes+len(self.cell_types)
        CellCentersRangesIndices = np.zeros((18, 2), dtype=np.int32)
        EdgesMidpointsRangesIndices = np.zeros((18, 2), dtype=np.int32)

        maxNPoissibleCVs = 0
        
        # I want to perform two passages. In the first one I assign indices to the Cell Centers and to the Edges MidPoints
        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']

            ElementNodes = self.Nodes[connectivity]

            CellCenters = np.mean(ElementNodes, axis=1)

            CellCentersRangesIndices[elem_type, 0] = StartCellCenterIndex
            CellCentersRangesIndices[elem_type, 1] = StartCellCenterIndex+CellCenters.shape[0]
            NewNodes[StartCellCenterIndex:(StartCellCenterIndex+CellCenters.shape[0]), :] = CellCenters

            StartCellCenterIndex = StartCellCenterIndex+connectivity.shape[0]

            edges_per_elem = self.ElementsUtilities.EDGES_PER_ELEMENT[self.meshType].get(elem_type, 0)
            true_edges_per_elem = edges_per_elem[0:int(len(edges_per_elem)/2)]
            edgeCentroids = np.reshape(np.mean(ElementNodes[:, true_edges_per_elem, :], axis=2), (connectivity.shape[0]*len(true_edges_per_elem), 3))

            EdgesMidpointsRangesIndices[elem_type, 0] = StartEdgeMidPointIndex
            EdgesMidpointsRangesIndices[elem_type, 1] = StartEdgeMidPointIndex+edgeCentroids.shape[0]
            NewNodes[StartEdgeMidPointIndex:(StartEdgeMidPointIndex+edgeCentroids.shape[0]), :] = edgeCentroids
            StartEdgeMidPointIndex = StartEdgeMidPointIndex+edgeCentroids.shape[0]



        start = time.time()
        self.Logger.info("Computing the edges (and faces in 3D) centroids...")
        self.Logger.info("Computing Control Volume and face area between node i and j...")

        # Cycle on each element and pre-compute edges mid-points and faces centers if 3D
        startIndex = 0
        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']

            edges_per_elem = self.ElementsUtilities.EDGES_PER_ELEMENT[self.meshType].get(elem_type, 0)
            true_edges_per_elem = edges_per_elem[0:int(len(edges_per_elem)/2)]

            StartCellCenterIndex = CellCentersRangesIndices[elem_type, 0]
            EndCellCenterIndex   = CellCentersRangesIndices[elem_type, 1]

            StartEdgesMidpointsIndex = EdgesMidpointsRangesIndices[elem_type, 0]
            EndEdgesMidpointsIndex   = EdgesMidpointsRangesIndices[elem_type, 1]

            CellCenters = np.mean(self.Nodes[connectivity], axis=1)

            if modifyOrigData:
                OrigDataAtCellCenters = np.mean(origData[connectivity, self.nDim:], axis=1)
                NewData[StartCellCenterIndex:EndCellCenterIndex, 0:self.nDim] = CellCenters[:, 0:self.nDim]
                NewData[StartCellCenterIndex:EndCellCenterIndex, self.nDim:] = OrigDataAtCellCenters
            
            # Compute the median of all the edges
            ElementNodes = self.Nodes[connectivity]

            start= time.time()
            self.Logger.info(f"Computing edge midpoints...")

            edgeCentroids = np.mean(ElementNodes[:, true_edges_per_elem, :], axis=2)
            if modifyOrigData:
                OrigDataAtCentroids = np.mean(origData[np.reshape(connectivity[:, true_edges_per_elem], (connectivity.shape[0]*len(true_edges_per_elem),2)), self.nDim:], axis=1)
                NewData[StartEdgesMidpointsIndex:EndEdgesMidpointsIndex, 0:self.nDim] = np.reshape(edgeCentroids, (connectivity.shape[0]*len(true_edges_per_elem),3))[:, 0:self.nDim]
                NewData[StartEdgesMidpointsIndex:EndEdgesMidpointsIndex, self.nDim:] = OrigDataAtCentroids

            self.Logger.info(f"Done computing edge midpoints. Elapsed time {time.time()-start} s.")

            CellCentersIDs = np.arange(StartCellCenterIndex, EndCellCenterIndex)
            EdgesMidpointsIDs = np.reshape(np.arange(StartEdgesMidpointsIndex, EndEdgesMidpointsIndex), (connectivity.shape[0], len(true_edges_per_elem)))

            # Now I compute the areas of each edge midpoint
            # Cycle on each edge
            start= time.time()
            self.Logger.info(f"Computing edges normals and faces...")
            CVs_0To1 = np.zeros((connectivity.shape[0], 3), dtype=np.int32)
            CVs_1To0 = np.zeros((connectivity.shape[0], 3), dtype=np.int32)

            CVs_0To1[:, 0] = CellCentersIDs
            CVs_1To0[:, 0] = CellCentersIDs

            for iEdge, edge in enumerate(true_edges_per_elem):


                CVs_0To1[:, 2] = EdgesMidpointsIDs[:, iEdge]
                CVs_1To0[:, 1] = EdgesMidpointsIDs[:, iEdge]

                
                # It substitutes the for loop right after that is commented out
                CVs_0To1[:, 1] = connectivity[:, edge[0]]
                CVs_1To0[:, 2] = connectivity[:, edge[1]]

                # print(CVs_0To1.shape)

                CVElements[startIndex:(startIndex+connectivity.shape[0]), :] = CVs_0To1
                CVAssociatedToOrigPoints[startIndex:(startIndex+connectivity.shape[0])] = connectivity[:, edge[0]]
                CVElements[(startIndex+connectivity.shape[0]):(startIndex+2*connectivity.shape[0]), :] = CVs_1To0
                CVAssociatedToOrigPoints[(startIndex+connectivity.shape[0]):(startIndex+2*connectivity.shape[0])] = connectivity[:, edge[1]]
                startIndex = startIndex+2*connectivity.shape[0]

            self.Logger.info(f"Done computing edges normals and faces. Elapsed time {time.time()-start} s.")

            
        
        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")

        start = time.time()
        self.Logger.info("Checking orientation of every CV to be consistent")
 
        trisq = NewNodes[CVElements, 0:2]
        # signedArea = (x2−x1)(y3−y1)−(x3−x1)(y2−y1)
        signedArea = (trisq[:, 1, 0]-trisq[:, 0, 0])*(trisq[:, 2, 1]-trisq[:, 0, 1]) - (trisq[:, 2, 0]-trisq[:, 0, 0])*(trisq[:, 1, 1]-trisq[:, 0, 1])
        Has2BeInverted = np.where(signedArea < 0.0)[0]
        if len(Has2BeInverted) > 0:
            self.Logger.info(f"{str(len(Has2BeInverted))} CVs out of {str(CVElements.shape[0])} have to be inverted")
            Tria2BeInverted = CVElements[Has2BeInverted, :]
            a = Tria2BeInverted[:, 2].copy()
            Tria2BeInverted[:, 2] = Tria2BeInverted[:, 1].copy()
            Tria2BeInverted[:, 1] = a
            CVElements[Has2BeInverted, :] = Tria2BeInverted

        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")
        
        if not modifyOrigData:
            NewData = origData # Remember that it was transposed

        NewData = np.transpose(NewData)

        return NewNodes, CVElements, NewData, CVAssociatedToOrigPoints
    
    def build_DualControlVolumes_2D_Optimized(self, origData, modifyOrigData=False):
        """
        Optimized generation of Median-Dual Control Volumes.
        Avoids duplicate points by resolving topology first.
        """
        n_nodes = len(self.Nodes)
        n_dim = 2
        
        # --- PHASE 1: Collect All Topology ---
        # We flatten all local edges into massive arrays.
        # We keep track of ranges to map them back to specific element blocks.
        
        all_local_edges = []
        
        # Store ranges to map back later: section -> (start_edge_idx, end_edge_idx)
        section_ranges = {} 
        
        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']
            n_elems = connectivity.shape[0]
            
            # 1. Get Definitions
            # EDGES: (N_edges_per_elem, 2) indices
            edges_def = self.ElementsUtilities.EDGES_PER_ELEMENT[self.meshType].get(elem_type)
            # We assume the utility returns ALL edges. If it duplicates (0-1 and 1-0), 
            # take only the first half like your code did:
            true_edges_def = np.array(edges_def[0:int(len(edges_def)/2)])
            n_edges_per_elem = len(true_edges_def)
            
            # 2. Expand Local Topology to Global Node IDs
            # Edges: Shape (N_elems * N_edges_per_elem, 2)
            # Use numpy broadcasting to index connectivity
            # connectivity[:, true_edges_def[:, 0]] gets first node of every edge for all elems
            local_edges = np.stack([
                connectivity[:, true_edges_def[:, 0]].ravel(),
                connectivity[:, true_edges_def[:, 1]].ravel()
            ], axis=1)
            
            # Sort node indices per edge (u < v) to ensure uniqueness
            local_edges.sort(axis=1)
            all_local_edges.append(local_edges)

            # Pre-allocate for this section
            n_elems = connectivity.shape[0]
            
            
            # Record where this section lives in the flattened arrays
            section_ranges[section_name] = {
                'n_elems': n_elems,
                'connectivity': connectivity,
                'elem_type': elem_type,
                'true_edges_def': true_edges_def, # Store defs for phase 3
                # We will fill indices here after uniqueness check
            }

        # --- PHASE 2: Resolve Uniqueness (The "Magic" Step) ---
        
        # 1. Stack everything
        big_edges_array = np.vstack(all_local_edges)
        
        # 2. Find Unique Edges
        # unique_edges: The actual (u, v) pairs
        # edge_inverse_map: Array of same size as big_edges_array mapping original->unique
        print(" resolving unique edges...")
        start = time.time()
        # Each row is already sorted (u <= v), so encode the pair as a single
        # int64 key u*n_nodes + v and uniquify the scalar keys. np.unique on 1D
        # integers is several times faster than np.unique(axis=0), which builds a
        # void-dtype view and lexsorts the rows. The key order equals the (u, v)
        # lexicographic order, so unique_edges and edge_inverse_map come out
        # byte-identical to the axis=0 path (verified on tri meshes up to 1.3M
        # nodes, ~5x faster).
        edge_keys = big_edges_array[:, 0].astype(np.int64) * np.int64(n_nodes) + big_edges_array[:, 1]
        unique_keys, edge_inverse_map = np.unique(edge_keys, return_inverse=True)
        edge_inverse_map = edge_inverse_map.reshape(-1)        # guard against numpy axis-inverse shape quirks
        unique_edges = np.empty((unique_keys.shape[0], 2), dtype=big_edges_array.dtype)
        unique_edges[:, 0] = unique_keys // n_nodes
        unique_edges[:, 1] = unique_keys %  n_nodes
        n_unique_edges = len(unique_edges)
        print(" Done! Elapsed time", time.time()-start, "s. Found", n_unique_edges, "unique edges.")

        # --- PHASE 3: Compute Geometry (Once per unique entity) ---
        
        # 1. Cell Centers (One per element - no global uniqueness check needed)
        # We can compute these on the fly or pre-compute. Let's listify to stack.
        print(" Computing cell centers...")
        start = time.time()
        all_cell_centers = []
        total_elems = 0
        for section_name, info in section_ranges.items():
            conn = info['connectivity']
            # summed gather over the nv vertices; avoids the (N,nv,3) intermediate np.mean builds
            # (nv <= 8 < numpy pairwise threshold -> naive-sequential sum, same value)
            centers = self.Nodes[conn[:, 0]].astype(np.float64).copy()
            for _k in range(1, conn.shape[1]):
                centers += self.Nodes[conn[:, _k]]
            centers /= conn.shape[1]
            all_cell_centers.append(centers)
            total_elems += info['n_elems']
        
        unique_cell_centers = np.vstack(all_cell_centers)
        n_cells = len(unique_cell_centers)
        print(" Done! Elapsed time", time.time()-start, "s.")
        
        # 3. Edge Midpoints
        print(" Computing edge midpoints...")
        start = time.time()
        # unique_edges is (N_unique, 2). 0.5*(P_u + P_v) avoids the
        # (N_unique, 2, 3) intermediate that np.mean(self.Nodes[unique_edges])
        # materialises -- ~2.6x faster at multi-million edge counts.
        unique_edge_midpoints = 0.5 * (self.Nodes[unique_edges[:, 0]] + self.Nodes[unique_edges[:, 1]])
        print(" Done! Elapsed time", time.time()-start, "s.")
        # print(unique_edge_midpoints)
        
        # --- PHASE 4: Assemble Global Arrays ---
        
        # Construct NewNodes Array
        print(" Construct NewNodes Array...")
        start = time.time()
        # Structure: [Original | Cells | Edges]
        start_cells = n_nodes
        start_edges = start_cells + n_cells
        total_new_nodes = start_edges + n_unique_edges
        
        NewNodes = np.zeros((total_new_nodes, 3), dtype=np.float64)
        NewNodes[0:n_nodes] = self.Nodes
        NewNodes[start_cells:start_edges] = unique_cell_centers
        NewNodes[start_edges:]            = unique_edge_midpoints
        print(" Done! Elapsed time", time.time()-start, "s.")

        # Handle Data Interpolation (ModifyOrigData)
        NewData = None
        if modifyOrigData:
            print(" Handle Data Interpolation...")
            start = time.time()

            # The field gathers below read per-node field vectors (rows) from this
            # transpose. On the F-order VIEW those reads are strided; a
            # C-contiguous copy speeds the cell-centre gather but costs one full
            # copy of the data. Measured crossover on a 1.33M-node mesh is ~8
            # solution-field rows: below it the view wins (copy not amortised),
            # above it the copy wins. Gate on the field count to get both regimes.
            origData = np.transpose(origData)
            if origData.shape[1] - self.nDim >= 8:
                origData = np.ascontiguousarray(origData)

            # Interpolate logic similar to coords...
            # Just repeat the mean() logic on origData instead of self.Nodes
            NewData = np.zeros((NewNodes.shape[0], origData.shape[1]), dtype=np.float64)
            NewData[0:n_nodes, :] = origData
            NewData[start_cells:start_edges, 0:self.nDim] = unique_cell_centers[:, 0:self.nDim]
            NewData[start_edges:, 0:self.nDim] = unique_edge_midpoints[:, 0:self.nDim]

            cell_data_offset = 0
            for section_name, info in section_ranges.items():
                connectivity = info['connectivity']
                n_e = info['n_elems']
                OrigDataAtCellCenters = np.mean(origData[connectivity, self.nDim:], axis=1)
                NewData[start_cells + cell_data_offset : start_cells + cell_data_offset + n_e, self.nDim:] = OrigDataAtCellCenters
                cell_data_offset += n_e

            # Then edges: 0.5*(data_u + data_v) avoids the (N_unique, 2, ndata)
            # intermediate that np.mean(origData[unique_edges, ...]) builds --
            # ~2.4x faster on the real mesh, regardless of the view/copy choice.
            OrigDataAtEdgeMidPoints = 0.5 * (origData[unique_edges[:, 0], self.nDim:]
                                             + origData[unique_edges[:, 1], self.nDim:])
            NewData[start_edges:, self.nDim:] = OrigDataAtEdgeMidPoints

            NewData = np.transpose(NewData)
            print(" Done! Elapsed time", time.time()-start, "s.")
        else:
            NewData = origData

        # --- PHASE 5: Build Connectivity (CVElements) ---
        
        # Total CVs = Sum(N_elems * N_edges_per_elem * 4)
        print(" Building Connectivity (CVElements)...")
        start = time.time()
        total_cvs = 0
        for section_name, info in section_ranges.items():
            n_e = info['n_elems']
            n_edges = len(info['true_edges_def'])
            total_cvs += n_e * n_edges * 2 # 2 CVs per edge. One connected to edge[0] and one connected to edge[1]

        print(" Done! Elapsed time", time.time()-start, "s.")
        CVElements = np.zeros((total_cvs, self.nDim+1), dtype=np.int32)
        CVAssociatedToOrigPoints = np.zeros(total_cvs, dtype=np.int32)
        
        # Pointers for slicing the big inverse maps
        curr_edge_map_idx = 0
        curr_cell_idx_offset = 0 # Offset into unique_cell_centers array
        cv_row_idx = 0
        
        for section_name, info in section_ranges.items():
            conn = info['connectivity']
            n_e = info['n_elems']
            true_edges = info['true_edges_def'] # (N_edges_local, 2)
            n_edges_local = len(true_edges)
            
            # 1. Get Global IDs for this section's Cell Centers
            # These are contiguous in the NewNodes array block for Cells
            global_cell_ids = np.arange(start_cells + curr_cell_idx_offset, 
                                        start_cells + curr_cell_idx_offset + n_e)
            curr_cell_idx_offset += n_e
            
            # 2. Get Global IDs for this section's Edges
            # Slice from the big edge_inverse_map
            # The slice length is n_e * n_edges_local
            n_total_edges_section = n_e * n_edges_local
            section_edge_indices = edge_inverse_map[curr_edge_map_idx : curr_edge_map_idx + n_total_edges_section]
            curr_edge_map_idx += n_total_edges_section
            
            # Reshape to (N_elems, N_edges_local)
            # Now global_edge_ids[e, i] is the UniqueID of i-th edge of element e
            global_edge_ids = section_edge_indices.reshape(n_e, n_edges_local) + start_edges

            # Parent element winding (CCW>0), one shoelace sign per element. A
            # median-dual sub-triangle (cell-center, edge-mid, node) always winds
            # OPPOSITE to its parent polygon (verified over tris and quads, both
            # windings), so we emit each CV pre-oriented CCW below and skip the
            # old global per-CV signed-area post-check entirely.
            poly = self.Nodes[conn, 0:2]
            parent_ccw = (np.sum(poly[:, :, 0] * np.roll(poly[:, :, 1], -1, axis=1)
                                 - np.roll(poly[:, :, 0], -1, axis=1) * poly[:, :, 1],
                                 axis=1) > 0.0)

            # 4. Fill CVElements Vectorized

            # Iterate edges (small loop, e.g. 6 times)
            for i_edge_local in range(n_edges_local):
                edge_def = true_edges[i_edge_local] # [u, v] local node indices

                # Global Indices for this specific edge across all elements in section
                g_edges = global_edge_ids[:, i_edge_local]

                # Nodes u and v (Global node indices from connectivity)
                g_node_u = conn[:, edge_def[0]]
                g_node_v = conn[:, edge_def[1]]
                
                # -- Build CVs --
                # We build them in blocks of size N_elems
                    
                # CV Part A: Cell -> Edge -> Node U
                # Vectorized fill
                count = n_e
                 
                # Indices in CVElements
                idx_start = cv_row_idx
                idx_end   = cv_row_idx + count
                
                # Pre-oriented CCW: parent CCW -> (C, U, M), else (C, M, U).
                CVElements[idx_start:idx_end, 0] = global_cell_ids
                CVElements[idx_start:idx_end, 1] = np.where(parent_ccw, g_node_u, g_edges)
                CVElements[idx_start:idx_end, 2] = np.where(parent_ccw, g_edges, g_node_u)

                CVAssociatedToOrigPoints[idx_start:idx_end] = g_node_u
                cv_row_idx += count

                # CV Part B: Cell -> Edge -> Node V
                idx_start = cv_row_idx
                idx_end   = cv_row_idx + count

                # Pre-oriented CCW: parent CCW -> (C, M, V), else (C, V, M).
                CVElements[idx_start:idx_end, 0] = global_cell_ids
                CVElements[idx_start:idx_end, 1] = np.where(parent_ccw, g_edges, g_node_v)
                CVElements[idx_start:idx_end, 2] = np.where(parent_ccw, g_node_v, g_edges)

                CVAssociatedToOrigPoints[idx_start:idx_end] = g_node_v
                cv_row_idx += count
        
        # CVs are emitted pre-oriented CCW from the parent-winding rule above, so
        # the former global per-CV signed-area check + flip pass is no longer
        # needed (it also materialised an (N_cv, 3, 2) temp and printed the whole
        # inverted-index array). Guard it behind debug only.
        if getattr(self, "debug", False):
            x = NewNodes[:, 0]; y = NewNodes[:, 1]
            c0, c1, c2 = CVElements[:, 0], CVElements[:, 1], CVElements[:, 2]
            signedArea = (x[c1]-x[c0])*(y[c2]-y[c0]) - (x[c2]-x[c0])*(y[c1]-y[c0])
            nBad = int(np.count_nonzero(signedArea < 0.0))
            if nBad:
                self.Logger.info(f"WARNING: {nBad}/{CVElements.shape[0]} CVs still negatively oriented after construction")

        return NewNodes, CVElements, NewData, CVAssociatedToOrigPoints


    def build_DualControlVolumes_3D(self, origData, modifyOrigData=False):

        origData = np.transpose(origData)
        n_nodes = len(self.Nodes)

        # Let's start from computing the control volumes associated 
        # to each node and their elements.

        start = time.time()
        self.Logger.info("Initializing ControlVolumes per Node dictionary...")
        
        maxNPoissibleCVs = 0
        maxNNodes = 0
        NFaces = 0
        
        # I want to perform two passages. In the first one I assign indices to the Cell Centers and to the Edges MidPoints
        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']
            # Surface sections (boundary TRI_3/QUAD_4 stored in the zone) own no dual CV: skip
            if not self.ElementsUtilities.FACESOFEDGES_PER_ELEMENT[self.meshType].get(elem_type):
                continue
            edges_per_elem = self.ElementsUtilities.EDGES_PER_ELEMENT[self.meshType].get(elem_type, 0)
            true_edges_per_elem = edges_per_elem[0:int(len(edges_per_elem)/2)]

            maxNPoissibleCVs += connectivity.shape[0]*len(true_edges_per_elem)*4
            maxNNodes += connectivity.shape[0]*len(true_edges_per_elem)

            faces_per_elem = self.ElementsUtilities.FACES_PER_ELEMENT[self.meshType].get(elem_type, 0)
            NFaces += connectivity.shape[0]*len(faces_per_elem)
            maxNNodes += connectivity.shape[0]*len(faces_per_elem)

        CVElements = np.zeros((maxNPoissibleCVs, 4), dtype=np.int32)
        CVAssociatedToOrigPoints = np.zeros((maxNPoissibleCVs, ), dtype=np.int32)

        # Then I have to augment the nodes of the grid with the cell centers and the edge mid points
        # I am not sure that maxNPoissibleCVs is the correct estimate for the number of edge mid-points
        # I have to make sure that if there are redundant nodes then I have to remove them
        NewNodes = np.zeros((n_nodes+len(self.cell_types)+maxNNodes, 3), dtype=np.float64)
        NewNodes[0:n_nodes, :] = self.Nodes
        NewData = None
        if modifyOrigData:
            NewData = np.zeros((n_nodes+len(self.cell_types)+maxNNodes, origData.shape[1]), dtype=np.float64)
            NewData[0:n_nodes, :] = origData

        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")

        StartCellCenterIndex = n_nodes
        StartFacesMidPointIndex = n_nodes+len(self.cell_types)
        StartEdgeMidPointIndex = n_nodes+len(self.cell_types)+NFaces
        CellCentersRangesIndices = np.zeros((18, 2), dtype=np.int32)
        FacesMidpointsRangesIndices = np.zeros((18, 2), dtype=np.int32)
        EdgesMidpointsRangesIndices = np.zeros((18, 2), dtype=np.int32)

        maxNPoissibleCVs = 0
        
        # I want to perform two passages. In the first one I assign indices to the Cell Centers and to the Edges MidPoints
        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']
            # Surface sections (boundary TRI_3/QUAD_4 stored in the zone) own no dual CV: skip
            if not self.ElementsUtilities.FACESOFEDGES_PER_ELEMENT[self.meshType].get(elem_type):
                continue

            ElementNodes = self.Nodes[connectivity]

            CellCenters = np.mean(ElementNodes, axis=1)

            CellCentersRangesIndices[elem_type, 0] = StartCellCenterIndex
            CellCentersRangesIndices[elem_type, 1] = StartCellCenterIndex+CellCenters.shape[0]
            NewNodes[StartCellCenterIndex:(StartCellCenterIndex+CellCenters.shape[0]), :] = CellCenters

            StartCellCenterIndex = StartCellCenterIndex+connectivity.shape[0]

            faces_per_elem = self.ElementsUtilities.FACES_PER_ELEMENT[self.meshType].get(elem_type, 0)
            faceCentroids = np.zeros((connectivity.shape[0], len(faces_per_elem), 3), dtype=np.float64)
            for iFace, face in enumerate(faces_per_elem):
                faceCentroids[:, iFace, :] = np.mean(ElementNodes[:, face, :], axis=1)
            faceCentroids = np.reshape(faceCentroids, (connectivity.shape[0]*len(faces_per_elem), 3))

            FacesMidpointsRangesIndices[elem_type, 0] = StartFacesMidPointIndex
            FacesMidpointsRangesIndices[elem_type, 1] = StartFacesMidPointIndex+faceCentroids.shape[0]
            NewNodes[StartFacesMidPointIndex:(StartFacesMidPointIndex+faceCentroids.shape[0]), :] = faceCentroids
            StartFacesMidPointIndex = StartFacesMidPointIndex+faceCentroids.shape[0]

            edges_per_elem = self.ElementsUtilities.EDGES_PER_ELEMENT[self.meshType].get(elem_type, 0)
            true_edges_per_elem = edges_per_elem[0:int(len(edges_per_elem)/2)]
            edgeCentroids = np.reshape(np.mean(ElementNodes[:, true_edges_per_elem, :], axis=2), (connectivity.shape[0]*len(true_edges_per_elem), 3))

            EdgesMidpointsRangesIndices[elem_type, 0] = StartEdgeMidPointIndex
            EdgesMidpointsRangesIndices[elem_type, 1] = StartEdgeMidPointIndex+edgeCentroids.shape[0]
            NewNodes[StartEdgeMidPointIndex:(StartEdgeMidPointIndex+edgeCentroids.shape[0]), :] = edgeCentroids
            StartEdgeMidPointIndex = StartEdgeMidPointIndex+edgeCentroids.shape[0]

        start = time.time()
        self.Logger.info("Computing the edges (and faces in 3D) centroids...")
        self.Logger.info("Computing Control Volume and face area between node i and j...")

        # Cycle on each element and pre-compute edges mid-points and faces centers if 3D
        startIndex = 0
        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']
            # Surface sections (boundary TRI_3/QUAD_4 stored in the zone) own no dual CV: skip
            if not self.ElementsUtilities.FACESOFEDGES_PER_ELEMENT[self.meshType].get(elem_type):
                continue

            edges_per_elem = self.ElementsUtilities.EDGES_PER_ELEMENT[self.meshType].get(elem_type, 0)
            true_edges_per_elem = edges_per_elem[0:int(len(edges_per_elem)/2)]

            StartCellCenterIndex = CellCentersRangesIndices[elem_type, 0]
            EndCellCenterIndex   = CellCentersRangesIndices[elem_type, 1]

            StartFacesMidpointsIndex = FacesMidpointsRangesIndices[elem_type, 0]
            EndFacesMidpointsIndex   = FacesMidpointsRangesIndices[elem_type, 1]

            StartEdgesMidpointsIndex = EdgesMidpointsRangesIndices[elem_type, 0]
            EndEdgesMidpointsIndex   = EdgesMidpointsRangesIndices[elem_type, 1]

            CellCenters = np.mean(self.Nodes[connectivity], axis=1)
            if modifyOrigData:
                OrigDataAtCellCenters = np.mean(origData[connectivity, self.nDim:], axis=1)
                NewData[StartCellCenterIndex:EndCellCenterIndex, 0:self.nDim] = CellCenters[:, 0:self.nDim]
                NewData[StartCellCenterIndex:EndCellCenterIndex, self.nDim:] = OrigDataAtCellCenters
            
            # Compute the median of all the edges
            ElementNodes = self.Nodes[connectivity]

            start= time.time()
            self.Logger.info(f"Computing edge midpoints...")

            faces_per_elem = self.ElementsUtilities.FACES_PER_ELEMENT[self.meshType].get(elem_type, 0)
            faceCentroids = np.zeros((connectivity.shape[0], len(faces_per_elem), 3), dtype=np.float64)
            for iFace, face in enumerate(faces_per_elem):
                faceCentroids[:, iFace, :] = np.mean(ElementNodes[:, face, :], axis=1)
            if modifyOrigData:
                OrigDataAtFaceCentroids = np.zeros((connectivity.shape[0], len(faces_per_elem), origData.shape[1]-self.nDim), dtype=np.float64)
                for iFace, face in enumerate(faces_per_elem):
                    OrigDataAtFaceCentroids[:, iFace, :] = np.mean(origData[connectivity[:, face], self.nDim:], axis=1)
                OrigDataAtFaceCentroids = np.reshape(OrigDataAtFaceCentroids, (connectivity.shape[0]*len(faces_per_elem), origData.shape[1]-self.nDim))
                NewData[StartFacesMidpointsIndex:EndFacesMidpointsIndex, 0:self.nDim] = np.reshape(faceCentroids, (connectivity.shape[0]*len(faces_per_elem), 3))[:, 0:self.nDim]
                NewData[StartFacesMidpointsIndex:EndFacesMidpointsIndex, self.nDim:] = OrigDataAtFaceCentroids

            edgeCentroids = np.mean(ElementNodes[:, true_edges_per_elem, :], axis=2)
            if modifyOrigData:
                OrigDataAtCentroids = np.mean(origData[np.reshape(connectivity[:, true_edges_per_elem], (connectivity.shape[0]*len(true_edges_per_elem),2)), self.nDim:], axis=1)
                NewData[StartEdgesMidpointsIndex:EndEdgesMidpointsIndex, 0:self.nDim] = np.reshape(edgeCentroids, (connectivity.shape[0]*len(true_edges_per_elem),3))[:, 0:self.nDim]
                NewData[StartEdgesMidpointsIndex:EndEdgesMidpointsIndex, self.nDim:] = OrigDataAtCentroids

            self.Logger.info(f"Done computing edge midpoints. Elapsed time {time.time()-start} s.")

            CellCentersIDs = np.arange(StartCellCenterIndex, EndCellCenterIndex)
            FacesMidpointsIDs = np.reshape(np.arange(StartFacesMidpointsIndex, EndFacesMidpointsIndex), (connectivity.shape[0], len(faces_per_elem)))
            EdgesMidpointsIDs = np.reshape(np.arange(StartEdgesMidpointsIndex, EndEdgesMidpointsIndex), (connectivity.shape[0], len(true_edges_per_elem)))
            
            facesOfEdges = self.ElementsUtilities.FACESOFEDGES_PER_ELEMENT[self.meshType].get(elem_type, 0)

            # Now I compute the areas of each edge midpoint
            # Cycle on each edge
            start= time.time()

            CVs_0To1 = np.zeros((connectivity.shape[0], 4), dtype=np.int32)
            CVs_1To0 = np.zeros((connectivity.shape[0], 4), dtype=np.int32)
            CVs_0To1[:, 0] = CellCentersIDs
            CVs_1To0[:, 0] = CellCentersIDs

            self.Logger.info(f"Computing edges normals and faces...")
            for iEdge, edge in enumerate(true_edges_per_elem):

                CVs_0To1[:, 1] = EdgesMidpointsIDs[:, iEdge]
                CVs_1To0[:, 1] = EdgesMidpointsIDs[:, iEdge]
                
                # It substitutes the for loop right after that is commented out
                CVs_0To1[:, 2] = connectivity[:, edge[0]]
                CVs_1To0[:, 2] = connectivity[:, edge[1]]

                for face in facesOfEdges[iEdge]:
                    CVs_0To1[:, 3] = FacesMidpointsIDs[:, face]
                    CVs_1To0[:, 3] = FacesMidpointsIDs[:, face]

                    CVElements[startIndex:(startIndex+connectivity.shape[0]), :] = CVs_0To1
                    CVAssociatedToOrigPoints[startIndex:(startIndex+connectivity.shape[0])] = connectivity[:, edge[0]]
                    CVElements[(startIndex+connectivity.shape[0]):(startIndex+2*connectivity.shape[0]), :] = CVs_1To0
                    CVAssociatedToOrigPoints[(startIndex+connectivity.shape[0]):(startIndex+2*connectivity.shape[0])] = connectivity[:, edge[1]]
                    startIndex = startIndex+2*connectivity.shape[0]

            self.Logger.info(f"Done computing edges normals and faces. Elapsed time {time.time()-start} s.")

            
        
        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")

        start = time.time()
        self.Logger.info("Collecting the edge face area from every element")
        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")
        

        if not modifyOrigData:
            NewData = origData # Remember that it was transposed
        NewData = np.transpose(NewData)

        return NewNodes, CVElements, NewData, CVAssociatedToOrigPoints


    @staticmethod
    def _unique_rows(rows):
        """
        Unique rows of an integer array with the inverse map. Equivalent to
        np.unique(rows, axis=0, return_inverse=True) -- same lexicographically
        sorted unique rows and the same inverse.

        Two-column rows (edges) take a fast path: the pair (c0,c1) is packed into a
        single int64 key c0*(maxid+1)+c1 and deduplicated with 1-D np.unique -- ~2x
        faster and ~half the transient RAM of the axis=0 path, and result-identical.
        Wider rows (3-/4-id faces) cannot use that key -- a 4-id quad-face key exceeds
        2^63 past only ~55k nodes -- so they fall back to lexsort+diff, which has no
        such ceiling and still beats np.unique(axis=0)'s void-dtype view.

        Works for any column count (2/3/4 here) and any node count.
        """
        rows = np.ascontiguousarray(rows)
        n = rows.shape[0]
        if n == 0:
            return rows.copy(), np.empty(0, dtype=np.int64)
        # Fast path for 2-id rows (edges): pack (c0,c1) into a single int64 key and use 1-D
        # np.unique. A pure lexicographic key c0*(maxid+1)+c1 gives the SAME (c0,c1)-sorted
        # unique rows and the same inverse as np.unique(axis=0), but is ~2x faster and uses
        # ~half the transient RAM (no full rows[order] copy). Guard against overflow: the key
        # is < (maxid+1)^2, safe while maxid < 2^31 (key < 2^62 < 2^63). Wider rows keep lexsort.
        if rows.shape[1] == 2:
            maxid = int(rows.max())
            if maxid < (1 << 31):
                base = maxid + 1
                key = rows[:, 0].astype(np.int64) * base + rows[:, 1].astype(np.int64)
                ukey, inv = np.unique(key, return_inverse=True)
                inv = np.asarray(inv).reshape(-1).astype(np.int64)
                unique_rows = np.empty((ukey.shape[0], 2), dtype=rows.dtype)
                unique_rows[:, 0] = ukey // base
                unique_rows[:, 1] = ukey - unique_rows[:, 0].astype(np.int64) * base
                return unique_rows, inv
        # np.lexsort uses the LAST key as primary; rows.T[::-1] puts column 0 last
        # -> primary sort on column 0, then 1, ... == np.unique(axis=0) ordering.
        order = np.lexsort(rows.T[::-1])
        srt = rows[order]
        isnew = np.ones(n, dtype=bool)                 # first row always starts a group
        if n > 1:
            np.any(srt[1:] != srt[:-1], axis=1, out=isnew[1:])
        unique_rows = srt[isnew]
        inv = np.empty(n, dtype=np.int64)
        inv[order] = np.cumsum(isnew) - 1              # original-row -> unique index
        return unique_rows, inv

    def build_DualControlVolumes_3D_Optimized(self, origData, modifyOrigData=False):
        """
        Optimized generation of Median-Dual Control Volumes.
        Avoids duplicate points by resolving topology first.
        """
        n_nodes = len(self.Nodes)
        n_dim = 3
        
        # --- PHASE 1: Collect All Topology ---
        # We flatten all local edges and faces into massive arrays.
        # We keep track of ranges to map them back to specific element blocks.
        
        all_local_edges = []
        all_local_tri_faces = []
        all_local_quad_faces = []

        # Store ranges to map back later: section -> (start_edge_idx, end_edge_idx)
        section_ranges = {}

        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']
            n_elems = connectivity.shape[0]

            # Skip boundary/surface sections (TRI_3 / QUAD_4 faces stored alongside the
            # volume elements, e.g. CGNS marker sections): they own no dual control
            # volume and would desync the flattened edge/face offsets in phase 3.
            if not self.ElementsUtilities.FACESOFEDGES_PER_ELEMENT[self.meshType].get(elem_type):
                print(f"  Skipping surface section '{section_name}' (element type {elem_type}) in 3D CV construction")
                continue

            # 1. Get Definitions
            # EDGES: (N_edges_per_elem, 2) indices
            edges_def = self.ElementsUtilities.EDGES_PER_ELEMENT[self.meshType].get(elem_type)
            # We assume the utility returns ALL edges. If it duplicates (0-1 and 1-0), 
            # take only the first half like your code did:
            true_edges_def = np.array(edges_def[0:int(len(edges_def)/2)])
            n_edges_per_elem = len(true_edges_def)
            
            # 2. Expand Local Topology to Global Node IDs
            # Edges: Shape (N_elems * N_edges_per_elem, 2)
            # Use numpy broadcasting to index connectivity
            # connectivity[:, true_edges_def[:, 0]] gets first node of every edge for all elems
            local_edges = np.stack([
                connectivity[:, true_edges_def[:, 0]].ravel(),
                connectivity[:, true_edges_def[:, 1]].ravel()
            ], axis=1)
            
            # Sort node indices per edge (u < v) to ensure uniqueness
            local_edges.sort(axis=1)
            all_local_edges.append(local_edges)
            
            # FACES: (N_faces_per_elem, N_nodes_per_face)
            faces_def = self.ElementsUtilities.FACES_PER_ELEMENT[self.meshType].get(elem_type)
            n_faces_per_elem = len(faces_def)

            # Faces: Shape (N_elems * N_faces_per_elem, 3 or 4) 
            # We need to extract them. Since a prism has 2 tris and 3 quads, 
            # we can't just do one big array for the element type.
            # We must iterate the face definitions.

            # Pre-allocate for this section
            n_elems = connectivity.shape[0]

            # These lists track which global index (in the Tri or Quad unique list)
            # each face of this element maps to.
            # We store: (is_quad, local_face_index_in_section_array)
            face_mapping_info = [] 
            
            for i_face, face_indices in enumerate(faces_def):
                n_nodes_face = len(face_indices)
                
                # Extract face connectivity for ALL elements of this type
                # Shape (N_elems, n_nodes_face)
                # connectivity[:, face_indices] 
                
                # Use fancy indexing to get nodes
                current_face_nodes = connectivity[:, face_indices]
                
                # Sort to ensure uniqueness (Set of nodes)
                current_face_nodes.sort(axis=1)
                
                if n_nodes_face == 3:
                    # It's a Triangle
                    start_idx = len(all_local_tri_faces) # tracking where this block starts
                    all_local_tri_faces.append(current_face_nodes)
                    face_mapping_info.append(('tri', start_idx)) 
                elif n_nodes_face == 4:
                    # It's a Quad
                    start_idx = len(all_local_quad_faces)
                    all_local_quad_faces.append(current_face_nodes)
                    face_mapping_info.append(('quad', start_idx))
                    
            
            # Record where this section lives in the flattened arrays
            section_ranges[section_name] = {
                'n_elems': n_elems,
                'connectivity': connectivity,
                'elem_type': elem_type,
                'true_edges_def': true_edges_def, # Store defs for phase 3
                'face_mapping_info': face_mapping_info
                # We will fill indices here after uniqueness check
            }

        # --- PHASE 2: Resolve Uniqueness (The "Magic" Step) ---
        
        # 1. Stack everything
        big_edges_array = np.vstack(all_local_edges)
        
        # 2. Find Unique Edges
        # unique_edges: The actual (u, v) pairs
        # edge_inverse_map: Array of same size as big_edges_array mapping original->unique
        print(" resolving unique edges...")
        # lexsort-based unique rows: same result as np.unique(axis=0) but faster
        # and, unlike int64 key-packing, with no overflow ceiling (a 4-id quad
        # key overflows int64 past ~55k nodes). See _unique_rows.
        unique_edges, edge_inverse_map = self._unique_rows(big_edges_array)
        n_unique_edges = len(unique_edges)

        # 3. Find Unique Faces and compute centroids
        print(" resolving unique faces and compute their centroids...")
        # 1. Triangles
        unique_tris = None
        unique_quads = None
        if len(all_local_tri_faces) > 0:
            big_tri_array = np.vstack(all_local_tri_faces)
            unique_tris, tri_inverse_map = self._unique_rows(big_tri_array)
            # Compute centroids
            # summed gather avoids the (N,3,3) intermediate np.mean(self.Nodes[unique_tris]) builds
            # (same value: k=3 < numpy's pairwise-sum threshold, so this is naive-sequential too).
            tri_centroids = (self.Nodes[unique_tris[:, 0]] + self.Nodes[unique_tris[:, 1]]
                             + self.Nodes[unique_tris[:, 2]]) / 3.0
        else:
            tri_centroids = np.empty((0,3))
            tri_inverse_map = np.empty(0, dtype=int)

        # 2. Quads
        if len(all_local_quad_faces) > 0:
            big_quad_array = np.vstack(all_local_quad_faces)
            unique_quads, quad_inverse_map = self._unique_rows(big_quad_array)
            # Compute centroids (summed gather; avoids the (N,4,3) intermediate, same value)
            quad_centroids = (self.Nodes[unique_quads[:, 0]] + self.Nodes[unique_quads[:, 1]]
                              + self.Nodes[unique_quads[:, 2]] + self.Nodes[unique_quads[:, 3]]) / 4.0
        else:
            quad_centroids = np.empty((0,3))
            quad_inverse_map = np.empty(0, dtype=int)

        # 3. Stack Centroids to get Global Face Nodes
        # Order: [Triangles | Quads]
        unique_face_centroids = np.vstack([tri_centroids, quad_centroids])
        n_tri_faces = len(tri_centroids)
        n_quad_faces = len(quad_centroids)
        
        # Offset for Quads (Triangles occupy 0 ... N_tri-1)
        quad_offset_start = len(tri_centroids)
        
        # --- PHASE 3: Compute Geometry (Once per unique entity) ---
        
        # 1. Cell Centers (One per element - no global uniqueness check needed)
        # We can compute these on the fly or pre-compute. Let's listify to stack.
        all_cell_centers = []
        total_elems = 0
        for section_name, info in section_ranges.items():
            conn = info['connectivity']
            # summed gather over the nv vertices; avoids the (N,nv,3) intermediate np.mean builds
            # (nv <= 8 < numpy pairwise threshold -> naive-sequential sum, same value)
            centers = self.Nodes[conn[:, 0]].astype(np.float64).copy()
            for _k in range(1, conn.shape[1]):
                centers += self.Nodes[conn[:, _k]]
            centers /= conn.shape[1]
            all_cell_centers.append(centers)
            total_elems += info['n_elems']
        
        unique_cell_centers = np.vstack(all_cell_centers)
        n_cells = len(unique_cell_centers)
        
        # 3. Edge Midpoints. 0.5*(P_u + P_v) avoids the (N_unique, 2, 3)
        # intermediate that np.mean(self.Nodes[unique_edges]) materialises
        # (~2.6x faster at multi-million edge counts; see the 2D builder).
        unique_edge_midpoints = 0.5 * (self.Nodes[unique_edges[:, 0]] + self.Nodes[unique_edges[:, 1]])
        
        # --- PHASE 4: Assemble Global Arrays ---
        
        # Construct NewNodes Array
        # Structure: [Original | Cells | Faces | Edges]
        start_cells = n_nodes
        start_faces_tri = start_cells + n_cells
        start_faces_quad = start_faces_tri + n_tri_faces
        start_edges = start_faces_quad + n_quad_faces
        total_new_nodes = start_edges + n_unique_edges
        
        NewNodes = np.zeros((total_new_nodes, 3), dtype=np.float64)
        NewNodes[0:n_nodes] = self.Nodes
        NewNodes[start_cells:start_faces_tri] = unique_cell_centers
        NewNodes[start_faces_tri:start_edges] = unique_face_centroids
        NewNodes[start_edges:]                = unique_edge_midpoints
        
        # Handle Data Interpolation (ModifyOrigData)
        NewData = None
        if modifyOrigData:

            # F-order transpose view by default; materialise a C-contiguous copy
            # only when there are enough solution-field rows to amortise it (the
            # cell / tri-face / quad-face gathers benefit). Same field-count gate
            # as the 2D builder (~8 data rows); this path has 4 gathers off the
            # transpose, so the copy pays off slightly sooner.
            origData = np.transpose(origData)
            if origData.shape[1] - self.nDim >= 8:
                origData = np.ascontiguousarray(origData)

            # Interpolate logic similar to coords...
            # Just repeat the mean() logic on origData instead of self.Nodes
            NewData = np.zeros((NewNodes.shape[0], origData.shape[1]), dtype=np.float64)
            NewData[0:n_nodes, :] = origData
            NewData[start_cells:start_faces_tri, 0:self.nDim] = unique_cell_centers
            NewData[start_faces_tri:start_edges, 0:self.nDim] = unique_face_centroids[:, 0:self.nDim]
            NewData[start_edges:, 0:self.nDim] = unique_edge_midpoints[:, 0:self.nDim]
            
            cell_data_offset = 0
            for section_name, info in section_ranges.items():
                connectivity = info['connectivity']
                n_e = info['n_elems']
                OrigDataAtCellCenters = np.mean(origData[connectivity, self.nDim:], axis=1)
                NewData[start_cells + cell_data_offset : start_cells + cell_data_offset + n_e, self.nDim:] = OrigDataAtCellCenters
                cell_data_offset += n_e
                
            # Then interpolate at faces centers
            # First tria faces
            if isinstance(unique_tris, np.ndarray):
                OrigDataAtTriafaceCenters = np.mean(origData[unique_tris, self.nDim:], axis=1)
                NewData[start_faces_tri:start_faces_quad, self.nDim:] = OrigDataAtTriafaceCenters

            # Then quad faces
            if isinstance(unique_quads, np.ndarray):
                OrigDataAtQuadfaceCenters = np.mean(origData[unique_quads, self.nDim:], axis=1)
                NewData[start_faces_quad:start_edges, self.nDim:] = OrigDataAtQuadfaceCenters

            # Then edges: pairwise 0.5*(data_u + data_v) avoids the
            # (N_unique, 2, ndata) intermediate np.mean would build (~2.4x).
            OrigDataAtEdgeMidPoints = 0.5 * (origData[unique_edges[:, 0], self.nDim:]
                                             + origData[unique_edges[:, 1], self.nDim:])
            NewData[start_edges:, self.nDim:] = OrigDataAtEdgeMidPoints

            NewData = np.transpose(NewData)
        else:
            NewData = origData

        # --- PHASE 5: Build Connectivity (CVElements) ---
        
        # Total CVs = Sum(N_elems * N_edges_per_elem * 4)
        total_cvs = 0
        for section_name, info in section_ranges.items():
            n_e = info['n_elems']
            n_edges = len(info['true_edges_def'])
            total_cvs += n_e * n_edges * 4 # 4 CVs per edge. Two connected to edge[0] and two connected to edge[1]
            
        CVElements = np.zeros((total_cvs, 4), dtype=np.int32)
        CVAssociatedToOrigPoints = np.zeros(total_cvs, dtype=np.int32)
        
        # Pointers for slicing the big inverse maps
        curr_edge_map_idx = 0
        curr_tri_map_idx = 0
        curr_quad_map_idx = 0
        curr_cell_idx_offset = 0 # Offset into unique_cell_centers array
        cv_row_idx = 0
        
        for section_name, info in section_ranges.items():
            conn = info['connectivity']
            n_e = info['n_elems']
            true_edges = info['true_edges_def'] # (N_edges_local, 2)
            n_edges_local = len(true_edges)
            
            # 1. Get Global IDs for this section's Cell Centers
            # These are contiguous in the NewNodes array block for Cells
            global_cell_ids = np.arange(start_cells + curr_cell_idx_offset, 
                                        start_cells + curr_cell_idx_offset + n_e)
            curr_cell_idx_offset += n_e
            
            # 2. Get Global IDs for this section's Edges
            # Slice from the big edge_inverse_map
            # The slice length is n_e * n_edges_local
            n_total_edges_section = n_e * n_edges_local
            section_edge_indices = edge_inverse_map[curr_edge_map_idx : curr_edge_map_idx + n_total_edges_section]
            curr_edge_map_idx += n_total_edges_section
            
            # Reshape to (N_elems, N_edges_local)
            # Now global_edge_ids[e, i] is the UniqueID of i-th edge of element e
            global_edge_ids = section_edge_indices.reshape(n_e, n_edges_local) + start_edges
            
            # 3. Get Global IDs for this section's Faces
            # Reconstruct Global Face IDs for this section
            # We need an array (N_elems, N_faces_local)
            n_faces_local = len(info['face_mapping_info'])
            global_face_ids = np.zeros((n_e, n_faces_local), dtype=np.int32) 

            for i_face_local, (f_type, _) in enumerate(info['face_mapping_info']):
            
                if f_type == 'tri':
                    # Slice from Tri map
                    # Size of slice is n_e (since we added 1 block of size n_e)
                    indices = tri_inverse_map[curr_tri_map_idx : curr_tri_map_idx + n_e]
                    curr_tri_map_idx += n_e
                    
                    # Map to global NewNodes index
                    global_face_ids[:, i_face_local] = start_faces_tri + indices
                    
                elif f_type == 'quad':
                    # Slice from Quad map
                    indices = quad_inverse_map[curr_quad_map_idx : curr_quad_map_idx + n_e]
                    curr_quad_map_idx += n_e
                    
                    # Map to global NewNodes index (Offset by Tri count!)
                    global_face_ids[:, i_face_local] = start_faces_tri + quad_offset_start + indices
            
            # 4. Fill CVElements Vectorized
            # We need to map: "Edge i connects to Face A and Face B"
            facesOfEdges = self.ElementsUtilities.FACESOFEDGES_PER_ELEMENT[self.meshType].get(info['elem_type'])
            
            # Iterate edges (small loop, e.g. 6 times)
            for i_edge_local in range(n_edges_local):
                edge_def = true_edges[i_edge_local] # [u, v] local node indices
                
                # Global Indices for this specific edge across all elements in section
                g_edges = global_edge_ids[:, i_edge_local]
                
                # Nodes u and v (Global node indices from connectivity)
                g_node_u = conn[:, edge_def[0]]
                g_node_v = conn[:, edge_def[1]]
                
                # -- Build CVs --
                # One control volume per (face adjacent to this edge) x (node of
                # the edge), filled in blocks of n_e elements at a time.
                for f_local_idx in facesOfEdges[i_edge_local]:
                    g_face = global_face_ids[:, f_local_idx]
                    
                    # CV Part A: Cell -> Edge -> Node U -> Face
                    # Vectorized fill
                    count = n_e
                    
                    # Indices in CVElements
                    idx_start = cv_row_idx
                    idx_end   = cv_row_idx + count
                    
                    # Column 0: Cell Center
                    CVElements[idx_start:idx_end, 0] = global_cell_ids
                    # Column 1: Edge Midpoint
                    CVElements[idx_start:idx_end, 1] = g_edges
                    # Column 2: Node U
                    CVElements[idx_start:idx_end, 2] = g_node_u
                    # Column 3: Face Centroid
                    CVElements[idx_start:idx_end, 3] = g_face
                    
                    CVAssociatedToOrigPoints[idx_start:idx_end] = g_node_u
                    cv_row_idx += count
                    
                    # CV Part B: Cell -> Edge -> Node V -> Face
                    idx_start = cv_row_idx
                    idx_end   = cv_row_idx + count
                    
                    CVElements[idx_start:idx_end, 0] = global_cell_ids
                    CVElements[idx_start:idx_end, 1] = g_edges
                    CVElements[idx_start:idx_end, 2] = g_node_v
                    CVElements[idx_start:idx_end, 3] = g_face
                    
                    CVAssociatedToOrigPoints[idx_start:idx_end] = g_node_v
                    cv_row_idx += count
        
        NanValues = np.isnan(NewData[self.nDim, :])
        if np.sum(NanValues) > 0:
            print("Number of points with NaN values in original restart = ", np.sum(NanValues))
            print("Something is wrong!")
            exit(1)

        return NewNodes, CVElements, NewData, CVAssociatedToOrigPoints
    

    def extractCVStructure(self, origData, coords, useOptimizedCVBuilder, InterpolationOrder="First", modifyOrigData=False, WLSOnAugmentedGrid=False, wantHessianOnAugmentedGrid=False):
        """Build the median-dual control-volume structure.

        useOptimizedCVBuilder selects WHICH control-volume builder runs:

          True  -> build_DualControlVolumes_{2D,3D}_Optimized, which resolves the
                   unique edges and faces first and therefore emits no duplicate
                   nodes. Used for the SOURCE mesh, whose CVs feed the supermesh
                   intersection and the conservative assembly -- duplicated nodes
                   would corrupt that connectivity.
          False -> build_DualControlVolumes_{2D,3D}, the plain builder. Used for
                   the TARGET mesh, whose CVs only supply volumes and point
                   location, so duplicates are harmless and the topology
                   resolution can be skipped.

        This argument used to be called `needMeshWelding`, from a design where the
        CVs were built with duplicated nodes and then welded shut afterwards by
        weld_generated_portion / weld_with_ckdtree. That pass was commented out
        when the *_Optimized builders landed and it de-duplicates by construction,
        so nothing welds anything any more -- those functions are unreachable. The
        old name read as "skip the welding, is that safe?" when the question is
        only which builder to use.
        """

        if not isinstance(self.Elements, dict):
            self.Elements, self.cell_types = self.Mesh.extractMeshStructure()

        # First of all I have to take the coordinates from the restart file
        self.Nodes = np.zeros((coords.shape[0], 3), dtype=np.float64)
        self.Nodes[:, 0:self.nDim] = coords[:, 0:self.nDim]

        # Two valid configurations exist:
        #   - WLSOnAugmentedGrid=False (Conservative path): WLS operator built on the original
        #     mesh (n_orig nodes). dataFrom passed to ComputeWLSGradients_Numba later has n_orig
        #     columns (modifyOrigData=False), so the original-mesh stencil matches.
        #   - WLSOnAugmentedGrid=True (Barycentric path): WLS operator built on the augmented
        #     dual-CV mesh (n_orig + edge midpoints + cell centers). dataFrom is augmented too
        #     (modifyOrigData=True), so gradients are defined at every CV-simplex vertex, which
        #     is what the barycentric 2nd-order reconstruction indexes into.

        if not WLSOnAugmentedGrid:
            # The original-mesh adjacency feeds the WLS gradient operator, consumed by
            # the 2nd-order reconstruction and (with the degree-2 builder + Hessian) the
            # 3rd-order conservative reconstruction. In First order (and NN) the gradients
            # are never added, so skip the neighbour walk and WLS operator and return None.
            if InterpolationOrder in ("Second", "Third"):
                startTime = time.time()
                self.Logger.info(f"Constructing NodesConnectedToNode structure (original mesh)...")
                self.SetNodesConnectedToNode()
                self.Logger.info(f"Finished construction of NodesConnectedToNode structure. Elapsed time {time.time()-startTime} s.")

                startTime = time.time()
                self.Logger.info(f"Pre-computing WLS Coefficients (original mesh)...")
                # "Third"-order Conservative needs source Hessians (fused into the same
                # eigendecomposition as the gradients).
                self.ComputeWLSCoefficients_Numba(want_hessian=(InterpolationOrder == "Third"))
                self.Logger.info(f"Finished pre-computation of WLS Coefficients. Elapsed time {time.time()-startTime} s.")
            else:
                self.NodesConnectedToNode = None
                self.NumOfNodesConnectedToNode = None
                self.WLS_Coeffs = None
                self.WLS_Hess_Coeffs = None

        startTime = time.time()
        self.Logger.info(f"Constructing Control Volumes structure...")
        if self.nDim == 2:
            if useOptimizedCVBuilder:
                NewNodes, CVElements, NewData, CVAssociatedToOrigPoints = self.build_DualControlVolumes_2D_Optimized(origData, modifyOrigData)
            else:
                NewNodes, CVElements, NewData, CVAssociatedToOrigPoints = self.build_DualControlVolumes_2D(origData, modifyOrigData)
        else:
            if useOptimizedCVBuilder:
                NewNodes, CVElements, NewData, CVAssociatedToOrigPoints = self.build_DualControlVolumes_3D_Optimized(origData, modifyOrigData)
            else:
                NewNodes, CVElements, NewData, CVAssociatedToOrigPoints = self.build_DualControlVolumes_3D(origData, modifyOrigData)
        self.Logger.info(f"Finished construction of Control Volumes structure. Elapsed time {time.time()-startTime} s.")

        if WLSOnAugmentedGrid:
            # Swap MeshReader state to the augmented mesh before building the WLS operator.
            self.Nodes = NewNodes
            cv_elem_type = self.ElementsUtilities.ELEMENT_TYPES_FROM_NAMES[self.meshType]['TRI' if self.nDim == 2 else 'TET']
            self.Elements = {'CV_simplices': {'type': cv_elem_type, 'connectivity': CVElements}}

            # Augmented-mesh adjacency feeds the Second-order WLS operator and the
            # Barycentric "Third" half-gradient; Barycentric-First never indexes gradients.
            if InterpolationOrder in ("Second", "Third"):
                startTime = time.time()
                self.Logger.info(f"Constructing NodesConnectedToNode structure (augmented mesh)...")
                self.SetNodesConnectedToNode()
                self.Logger.info(f"Finished construction of NodesConnectedToNode structure. Elapsed time {time.time()-startTime} s.")

                startTime = time.time()
                self.Logger.info(f"Pre-computing WLS Coefficients (augmented mesh)...")
                # Only NN "Third" consumes Hessians on the augmented grid; Barycentric "Third"
                # uses the half-gradient alone, so it must not pay for the Hessian build.
                self.ComputeWLSCoefficients_Numba(want_hessian=wantHessianOnAugmentedGrid)
                self.Logger.info(f"Finished pre-computation of WLS Coefficients. Elapsed time {time.time()-startTime} s.")
            else:
                self.NodesConnectedToNode = None
                self.NumOfNodesConnectedToNode = None
                self.WLS_Coeffs = None
                self.WLS_Hess_Coeffs = None

        return NewNodes, CVElements, NewData, CVAssociatedToOrigPoints, self.NodesConnectedToNode, self.NumOfNodesConnectedToNode, self.WLS_Coeffs

    def buildNodesConnections(self, zone2Read=None):
        
        if not isinstance(self.Nodes, np.ndarray):
            self.Nodes = self.Mesh.extractAllCoords(zone2Read=zone2Read)

        if self.Elements == None:
            self.Elements, self.cell_types = self.Mesh.extractMeshStructure(zone2Read=zone2Read)

        # print(self.NodesConnectedToNode[node2check])
        startTime = time.time()
        self.Logger.info(f"Constructing NodesConnectedToNode structure...")
        self.SetNodesConnectedToNode()
        self.Logger.info(f"Finished construction of NodesConnectedToNode structure. Elapsed time {time.time()-startTime} s.")

        return self.NodesConnectedToNode, self.NumOfNodesConnectedToNode, self.NodesConnectedToNodeMask
    
    def buildCellsOfPoints(self, zone2Read=None):

        if not isinstance(self.Nodes, np.ndarray):
            self.Nodes = self.Mesh.extractAllCoords(zone2Read=zone2Read)

        if self.Elements == None:
            self.Elements, self.cell_types = self.Mesh.extractMeshStructure(zone2Read=zone2Read)

        # print(self.NodesConnectedToNode[node2check])
        startTime = time.time()
        self.Logger.info(f"Constructing CellsOfPoints structure...")
        self.SetCellsConnectedToNode()
        self.Logger.info(f"Finished construction of CellsOfPoints structure. Elapsed time {time.time()-startTime} s.")

        return self.CellsConnectedToNode, self.NumOfCellsConnectedToNode, self.CellsConnectedToNodeMask
    
    def buildWLSCoeffs(self, want_hessian=False):

        if not isinstance(self.Elements, dict):
            self.Elements, self.cell_types = self.Mesh.extractMeshStructure()

        if not isinstance(self.NodesConnectedToNode, np.ndarray):
            startTime = time.time()
            self.Logger.info(f"Constructing NodesConnectedToNode structure...")
            self.SetNodesConnectedToNode()
            self.Logger.info(f"Finished construction of NodesConnectedToNode structure. Elapsed time {time.time()-startTime} s.")

        startTime = time.time()
        self.Logger.info(f"Pre-computing WLS Coefficients...")
        # want_hessian additionally builds WLS_Hess_Coeffs (NN "Third") in the same
        # per-node eigendecomposition as the gradients.
        self.ComputeWLSCoefficients_Numba(want_hessian=want_hessian)
        self.Logger.info(f"Finished pre-computation of WLS Coefficients. Elapsed time {time.time()-startTime} s.")

        return self.WLS_Coeffs, self.NodesConnectedToNode
    
    def extractConnectivity(self, zone2Read=None):
        self.Elements, self.cell_types = self.Mesh.extractMeshStructure(zone2Read=zone2Read)
        return self.Elements, self.cell_types

    # Canonical element-type names across the supported mesh formats (SU2 and
    # CGNS use different section names: TET/TETRA_4, PRISM/PENTA_6, ...).
    _CANONICAL_ELEMENT_NAME = {
        'TRI': 'TRI',   'TRI_3':   'TRI',
        'QUAD': 'QUAD', 'QUAD_4':  'QUAD',
        'TET': 'TET',   'TETRA_4': 'TET',
        'PYRA': 'PYRA', 'PYRA_5':  'PYRA',
        'PRISM': 'PRISM', 'PENTA_6': 'PRISM',
        'HEXA': 'HEXA', 'HEXA_8':  'HEXA',
        'LINE': 'LINE', 'BAR_2':   'LINE',
    }

    def triangulateOrTetrahedralize(self):
        """
        In-process simplicial subdivision of the mesh, replacing the ParaView
        'Tetrahedralize' filter + PointIndices.csv round-trip.

        Each cell is split into simplices over the SAME vertex set (no Steiner
        points added), returning an (n_simplices, nDim+1) int32 connectivity
        that indexes the original nodes -- the array consumed by
        LinInterp_Mine_New as Connectivity_From.

        2D:  TRI  -> itself ;  QUAD -> 2 triangles (interior diagonal, always
             conforming since quad edges are never split). LINE markers skipped.
        3D:  TET  -> itself ;  PYRA/PRISM/HEXA -> 2/3/6 tetrahedra via
             _cone_subdivide_to_tets, which splits every shared quad face through
             its min-global-index vertex so the tetrahedralization is conforming.
             Boundary TRI/QUAD/LINE faces are skipped.

        Works for both SU2 and CGNS section naming/numbering conventions.

        Returns
        -------
        (M, nDim+1) int32 array of simplex connectivity into the original nodes.
        """
        if not isinstance(self.Elements, dict):
            self.Elements, self.cell_types = self.Mesh.extractMeshStructure()

        code2name = self.ElementsUtilities.ELEMENT_NAMES_FROM_TYPES.get(self.meshType, {})

        simplex_blocks = []
        for key, section in self.Elements.items():
            # Resolve a canonical element name from the section key or numeric code.
            etype = self._CANONICAL_ELEMENT_NAME.get(key) if isinstance(key, str) else None
            if etype is None:
                raw = section.get('type')
                nm = raw if isinstance(raw, str) else code2name.get(str(int(raw)), '')
                etype = self._CANONICAL_ELEMENT_NAME.get(nm)

            conn = np.asarray(section['connectivity'], dtype=np.int64)
            if conn.size == 0:
                continue

            if self.nDim == 2:
                if etype == 'TRI':
                    simplex_blocks.append(conn[:, :3])
                elif etype == 'QUAD':
                    n = conn.shape[0]
                    tris = np.empty((n * 2, 3), dtype=np.int64)
                    tris[0::2] = conn[:, [0, 1, 2]]
                    tris[1::2] = conn[:, [0, 2, 3]]
                    simplex_blocks.append(tris)
                elif etype == 'LINE':
                    continue  # boundary edge, not an area cell
                else:
                    raise NotImplementedError(
                        f"triangulateOrTetrahedralize: unsupported 2D element "
                        f"'{key}' (type {section.get('type')}); expected TRI or QUAD.")
            elif self.nDim == 3:
                if etype == 'TET':
                    simplex_blocks.append(conn[:, :4])
                elif etype in ('PYRA', 'PRISM', 'HEXA'):
                    quad_faces, tri_faces, n_tets = _CELL_FACES_3D[etype]
                    simplex_blocks.append(
                        _cone_subdivide_to_tets(conn, quad_faces, tri_faces, n_tets))
                elif etype in ('TRI', 'QUAD', 'LINE'):
                    continue  # boundary face/edge, not a volume cell
                else:
                    raise NotImplementedError(
                        f"triangulateOrTetrahedralize: unsupported 3D element "
                        f"'{key}' (type {section.get('type')}); expected TET, PYRA, "
                        f"PRISM or HEXA.")
            else:
                raise ValueError(
                    f"triangulateOrTetrahedralize: nDim must be 2 or 3, got {self.nDim}")

        if not simplex_blocks:
            raise ValueError("triangulateOrTetrahedralize: no supported volume/area elements found.")

        return np.ascontiguousarray(np.vstack(simplex_blocks).astype(np.int32))
    
    def extractUnpackedMeshStructure(self, zone2Read=None):
        return self.Mesh.extractUnpackedMeshStructure(zone2Read=zone2Read)
    
    def weld_generated_portion(self, coords, cells, n_fixed, tol=1e-11):
        """
        Welds ONLY the points from n_fixed onwards.
        Assumes coords[:n_fixed] are unique and disjoint from the rest.
        """
        # 1. Slice the Data (Zero Copy View)
        # We only process the "New" potentially duplicate points
        coords_new = coords[n_fixed:]
        n_new = coords_new.shape[0]
        
        print(f"Processing Suffix: {n_new} points (Fixed Prefix: {n_fixed})")
        
        # 2. Quantize ONLY the new part
        min_val = np.min(coords_new, axis=0)
        inv_tol = 1.0 / tol
        coords_int = np.empty((n_new, 3), dtype=np.int64)
        quantize_parallel(coords_new, min_val, inv_tol, coords_int)
        
        # 3. Sort ONLY the new part
        # This is much faster than sorting the full billions of points
        sort_idx = np.lexsort((coords_int[:, 2], coords_int[:, 1], coords_int[:, 0]))
        
        # 4. Identify Uniques
        unique_mask = compute_unique_mask(coords_int, sort_idx)
        
        # 5. Build Local Mapping (New_Index -> Unique_New_Index)
        group_ids = np.cumsum(unique_mask) - 1
        local_map = build_inverse_map_local(sort_idx, group_ids, n_new)
        
        n_unique_new = group_ids[-1] + 1
        
        # 6. Construct Final Coordinate Array
        total_unique = n_fixed + n_unique_new
        new_coords = np.empty((total_unique, 3), dtype=coords.dtype)
        
        # A. Copy Fixed Part (Bulk copy, fast)
        new_coords[:n_fixed] = coords[:n_fixed]
        
        # B. Copy Unique New Part
        # Get the indices of the unique points relative to the slice
        unique_slice_indices = sort_idx[unique_mask]
        new_coords[n_fixed:] = coords_new[unique_slice_indices]
        
        # 7. Map Cells
        new_cells = apply_map_with_offset(cells, local_map, n_fixed)
        
        print(f"Welding Report:")
        print(f"  Fixed Preserved: {n_fixed}")
        print(f"  New Original:    {n_new}")
        print(f"  New Unique:      {n_unique_new}")
        print(f"  Duplicates RM:   {n_new - n_unique_new}")
        
        return new_coords, new_cells
        
    def check_welding_quality(self, points, tolerance=1e-6):
        """
        Checks if there are duplicate points remaining in the mesh.
        
        Parameters:
        -----------
        points : (N, 3) float array of vertex coordinates.
        tolerance : float, the welding threshold used previously.
        
        Returns:
        --------
        duplicates : list of (i, j) tuples where dist(p[i], p[j]) < tolerance
        """
        tree = cKDTree(points)
        
        # Query all pairs within tolerance
        # returns set of (i, j) indices
        pairs = tree.query_pairs(r=tolerance)
        
        if len(pairs) == 0:
            print("✅ Welding successful: No duplicate points found.")
            return []
        else:
            print(f"❌ Welding failed: Found {len(pairs)} pairs of non-unique points.")
            # Print first few for inspection
            for (i, j) in list(pairs)[:5]:
                dist = np.linalg.norm(points[i] - points[j])
                print(f"   - Point {i} and {j} are distinct but dist={dist:.2e}")
            return list(pairs)
        
    def weld_with_ckdtree(self, coords, cells, n_fixed, tol=1e-11):
        """
        Fallback welding strategy using KD-Tree.
        Applies ONLY to coords[n_fixed:].
        Robust against grid alignment issues.
        """
        # 1. Slice New Points
        coords_new = coords[n_fixed:]
        n_new = coords_new.shape[0]
        
        print(f"KD-Tree Welding on {n_new} points...")
        
        # 2. Build Tree & Query Pairs
        # compact_nodes=False speeds up build for query-heavy tasks, 
        # but True is fine. balanced_tree=False is faster to build.
        tree = cKDTree(coords_new, compact_nodes=False, balanced_tree=False)
        
        # query_pairs returns a set of (i, j) tuples where i < j and dist < tol
        pairs = tree.query_pairs(r=tol)
        
        if len(pairs) == 0:
            print("KD-Tree found no duplicates.")
            return coords, cells
            
        print(f"KD-Tree found {len(pairs)} duplicate pairs. Merging...")
        
        # 3. Resolve Groups (Transitive Closure)
        # If A close to B, and B close to C, then A-B-C form a group.
        # We use a sparse graph to find connected components.
        
        # Convert pairs to arrays for sparse matrix construction
        pairs_list = np.array(list(pairs))
        row = pairs_list[:, 0]
        col = pairs_list[:, 1]
        data = np.ones(len(pairs), dtype=np.int32)
        
        # Adjacency matrix of duplicates
        adj = coo_matrix((data, (row, col)), shape=(n_new, n_new))
        
        # Find connected components (groups of duplicate points)
        # n_comps: number of unique resulting points (from those involved in duplicates)
        # labels: array of length n_new. labels[i] is the Group ID of point i.
        # BUT: unconnected points get their own label. We need to be careful.
        n_comps, labels = connected_components(adj, directed=False)
        
        # 4. Build the Mapping (Old -> New)
        # We need to map every point i to the "representative" of its component.
        # A simple way: map every point to the FIRST index in its component.
        
        # We can use the 'labels' to group them.
        # labels gives us a group ID for everyone.
        # We want to map: local_map[i] = representative_of_group[labels[i]]
        
        # Find the representative (min index) for each label
        # This is efficiently done by updating an array with min index
        
        # Initialize reps with a large number
        reps = np.full(n_comps, n_new + 1, dtype=np.int32)
        
        # Min index per component -- vectorized (identical to the per-point min loop).
        np.minimum.at(reps, labels, np.arange(n_new, dtype=np.int32))

        # Now build local map: point i maps to reps[labels[i]]
        local_map = reps[labels]
        
        # 5. Compress the array (remove gaps)
        # 'local_map' points at indices in the original (sparse) array: it maps
        # every old node to the representative of its component. Renumber those
        # survivors to 0..N_unique-1.
        #
        # return_inverse gives both halves in one pass: 'survivors' is the sorted
        # set of surviving original indices, and 'inverse[i]' is already the
        # dense index of the point that node i maps to.
        survivors, inverse = np.unique(local_map, return_inverse=True)

        final_map = inverse
        n_unique_new = len(survivors)
        
        # 6. Construct Output Arrays
        
        # A. Coords
        # Only keep the survivors
        unique_coords_new = coords_new[survivors]
        
        total_unique = n_fixed + n_unique_new
        final_coords = np.empty((total_unique, 3), dtype=coords.dtype)
        final_coords[:n_fixed] = coords[:n_fixed]
        final_coords[n_fixed:] = unique_coords_new
        
        # B. Cells
        # We need to update cells. Indices < n_fixed stay same.
        # Indices >= n_fixed need to be mapped.
        # Map: old_global_idx -> new_global_idx
        
        # The map 'final_map' is for the suffix (0 to n_new-1).
        # We need to apply it to cells that point to (n_fixed + 0 ... n_fixed + n_new-1).
        
        # Use Numba for fast application if cells is large
        final_cells = apply_map_ckdtree(cells, final_map, n_fixed, nDim=self.nDim)
        
        return final_coords, final_cells
    
    def extractFields(self, zone2Read, fields2Read = None):

        # Should work only if CGNS
        Fields = self.Mesh.extractFields(zone2Read, fields2Read)
        return Fields
    
    def appendFields(self, zone2Read, fields2Append):

        # Should work only if CGNS
        self.Mesh.appendFields(zone2Read, fields2Append)

    