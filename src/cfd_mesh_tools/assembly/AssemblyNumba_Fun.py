"""Numba assembly kernel for the native mixed-element elasticity
(phase-5 follow-up item: the measured target is the last super-linear /
scatter-bound stage; baseline 137 s at 1.4M nodes with chunked-CSR merging,
and the NumPy preallocated variant was np.add.at-bound at the same cost).

Design:
  - the CSR PATTERN comes from DeformMesh_Fun.buildAssemblyPattern (built
    once per connectivity, reused across increments and retries);
  - elements are GRAPH-COLORED once per connectivity: no two elements of the
    same color share a node, so a `prange` over one color scatters into the
    shared data array with no races and no atomics;
  - each element's matrix is computed inside the kernel (quadrature loops on
    the same dshape tables as the reference path) and added into its
    precomputed CSR slots via binary search on the node-pair keys.

AssemblyMode="numba" selects this path; "merge" remains the reference.
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _elementVolumes(X, conn, D, w, nDim):
    volume = np.empty(conn.shape[0])
    for e in prange(conn.shape[0]):
        value = 0.0
        for g in range(w.size):
            J = np.zeros((nDim, nDim))
            for a in range(conn.shape[1]):
                node = conn[e, a]
                for r in range(nDim):
                    for d in range(nDim):
                        J[r, d] += D[g, a, r] * X[node, d]
            if nDim == 2:
                det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            else:
                det = (J[0, 0] * (J[1, 1] * J[2, 2]
                                   - J[1, 2] * J[2, 1])
                       - J[0, 1] * (J[1, 0] * J[2, 2]
                                    - J[1, 2] * J[2, 0])
                       + J[0, 2] * (J[1, 0] * J[2, 1]
                                    - J[1, 1] * J[2, 0]))
            value += w[g] * abs(det)
        volume[e] = value
    return volume


def elementVolumesNumba(P, cellsByType, nDim):
    """Parallel determinant-only volumes for mixed 2D/3D element blocks."""
    from ..quadrature import _quadRule
    X = np.ascontiguousarray(np.asarray(P, dtype=np.float64)[:, :nDim])
    volumes = []
    for cellType, conn in cellsByType:
        D, w = _quadRule(cellType)
        volumes.append(_elementVolumes(
            X, np.ascontiguousarray(conn),
            np.ascontiguousarray(D), np.ascontiguousarray(w), nDim))
    return np.concatenate(volumes)


@njit(cache=True)
def colorElements(conn, nPts):
    """Greedy element coloring: same-color elements share no node.
    Returns (colorOfElement, nColors). Uses a per-node bitmask of the colors
    already taken by incident elements (an int64 supports 64 colors; typical
    meshes need ~a node's max valence, well under that)."""
    nE, nN = conn.shape
    nodeMask = np.zeros(nPts, dtype=np.int64)
    color = np.empty(nE, dtype=np.int32)
    nColors = 0
    for e in range(nE):
        used = np.int64(0)
        for a in range(nN):
            used |= nodeMask[conn[e, a]]
        c = 0
        while c < 63 and ((used >> c) & 1):
            c += 1
        if c >= 63:
            c = 63                       # overflow color: elements here may
                                         # share nodes; the CALLER assembles
                                         # color 63 SERIALLY, so it stays
                                         # race-free
        color[e] = c
        bit = np.int64(1) << c
        for a in range(nN):
            nodeMask[conn[e, a]] |= bit
        if c + 1 > nColors:
            nColors = c + 1
    return color, nColors


@njit(parallel=True, cache=True)
def _assembleColor(X, conn, elems, Ee, lam_f, mu_f, D, w, nDim,
                   nodeKeys, nIndptr, dofIndptr, data, nPts, blockMode):
    """Assemble and scatter all elements of ONE color (race-free prange)."""
    nG, nN = D.shape[0], D.shape[1]
    nd = nDim
    for ie in prange(elems.size):
        e = elems[ie]
        # gather element coordinates
        Xe = np.empty((nN, nd))
        for a in range(nN):
            for d_ in range(nd):
                Xe[a, d_] = X[conn[e, a], d_]
        lam = lam_f[e]
        mu = mu_f[e]
        Ke = np.zeros((nN * nd, nN * nd))
        J = np.empty((nd, nd))
        Ji = np.empty((nd, nd))
        G = np.empty((nN, nd))
        for g in range(nG):
            # J[r, c] = sum_a D[g, a, r] * Xe[a, c]
            for r in range(nd):
                for c in range(nd):
                    s = 0.0
                    for a in range(nN):
                        s += D[g, a, r] * Xe[a, c]
                    J[r, c] = s
            if nd == 2:
                det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                inv = 1.0 / det
                Ji[0, 0] = J[1, 1] * inv
                Ji[0, 1] = -J[0, 1] * inv
                Ji[1, 0] = -J[1, 0] * inv
                Ji[1, 1] = J[0, 0] * inv
            else:
                det = (J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
                       - J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0])
                       + J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]))
                inv = 1.0 / det
                Ji[0, 0] = (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) * inv
                Ji[0, 1] = (J[0, 2] * J[2, 1] - J[0, 1] * J[2, 2]) * inv
                Ji[0, 2] = (J[0, 1] * J[1, 2] - J[0, 2] * J[1, 1]) * inv
                Ji[1, 0] = (J[1, 2] * J[2, 0] - J[1, 0] * J[2, 2]) * inv
                Ji[1, 1] = (J[0, 0] * J[2, 2] - J[0, 2] * J[2, 0]) * inv
                Ji[1, 2] = (J[0, 2] * J[1, 0] - J[0, 0] * J[1, 2]) * inv
                Ji[2, 0] = (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]) * inv
                Ji[2, 1] = (J[0, 1] * J[2, 0] - J[0, 0] * J[2, 1]) * inv
                Ji[2, 2] = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]) * inv
            # dN/dx_d = sum_r D[g, a, r] * Ji[d, r]   (TRANSPOSED inverse --
            # the same contraction the prism patch test enforces)
            for a in range(nN):
                for d_ in range(nd):
                    s = 0.0
                    for r in range(nd):
                        s += D[g, a, r] * Ji[d_, r]
                    G[a, d_] = s
            wd = w[g] * abs(det)
            for a in range(nN):
                for b in range(nN):
                    gg = 0.0
                    for k in range(nd):
                        gg += G[a, k] * G[b, k]
                    for i in range(nd):
                        for j in range(nd):
                            v = (wd * lam) * G[a, i] * G[b, j] \
                                + (wd * mu) * G[a, j] * G[b, i]
                            if i == j:
                                v += (wd * mu) * gg
                            Ke[a * nd + i, b * nd + j] += v
        # scatter into the CSR slots (binary search on node-pair keys)
        for a in range(nN):
            na = conn[e, a]
            for b in range(nN):
                nb = conn[e, b]
                if blockMode:
                    start = nIndptr[na]
                    slot = start + np.searchsorted(
                        nodeKeys[start:nIndptr[na + 1]], nb)
                else:
                    slot = np.searchsorted(nodeKeys, na * nPts + nb)
                pos = slot - nIndptr[na]
                for i in range(nd):
                    if blockMode:
                        base = slot * nd * nd + i * nd
                    else:
                        base = dofIndptr[na * nd + i] + pos * nd
                    for j in range(nd):
                        data[base + j] += Ke[a * nd + i, b * nd + j]


def assembleElasticityNumba(P, cellsByType, E, Nu, nDim, pattern, colorInfo):
    """Numba-parallel assembly into the preallocated pattern. `colorInfo` is
    {blockIndex: (colorOfElement, nColors)}, cached with the pattern."""
    from scipy.sparse import csr_matrix
    from ..quadrature import _quadRule
    X = np.ascontiguousarray(np.asarray(P, dtype=np.float64)[:, :nDim])
    nPts = pattern['nPts']
    data = np.zeros(pattern['dofIndices'].size)
    base = 0
    for bi, (t, connAll) in enumerate(cellsByType):
        C = np.ascontiguousarray(connAll)
        nE = C.shape[0]
        Ee = np.asarray(E[base:base + nE], dtype=np.float64)
        base += nE
        lam = Nu * Ee / ((1.0 + Nu) * (1.0 - 2.0 * Nu))
        mu = Ee / (2.0 * (1.0 + Nu))
        D, w = _quadRule(t)
        D = np.ascontiguousarray(D)
        w = np.ascontiguousarray(w)
        color, nColors = colorInfo[bi]
        for c in range(nColors):
            elems = np.where(color == c)[0]
            if not elems.size:
                continue
            if c == 63:
                # overflow color: serial, one element at a time (rare)
                for e in elems:
                    _assembleColor(X, C, np.array([e], dtype=np.int64),
                                   Ee, lam, mu, D, w, nDim,
                                   pattern['nodeKeys'], pattern['nIndptr'],
                                   pattern['dofIndptr'], data, nPts, False)
            else:
                _assembleColor(X, C, elems, Ee, lam, mu, D, w, nDim,
                               pattern['nodeKeys'], pattern['nIndptr'],
                               pattern['dofIndptr'], data, nPts, False)
    return csr_matrix((data, pattern['dofIndices'], pattern['dofIndptr']),
                      shape=(nPts * nDim, nPts * nDim))


def assembleElasticityBSRNumba(P, cellsByType, E, Nu, nDim, pattern,
                               colorInfo):
    """Assemble directly into dense nodal blocks; no scalar CSR is formed."""
    from scipy.sparse import bsr_matrix
    from ..quadrature import _quadRule
    X = np.ascontiguousarray(np.asarray(P, dtype=np.float64)[:, :nDim])
    nPts = pattern['nPts']
    data = np.zeros(pattern['nIndices'].size * nDim * nDim)
    dummy = np.empty(0, dtype=pattern['nIndptr'].dtype)
    base = 0
    for bi, (t, connAll) in enumerate(cellsByType):
        C = np.ascontiguousarray(connAll)
        nE = C.shape[0]
        Ee = np.asarray(E[base:base + nE], dtype=np.float64)
        base += nE
        lam = Nu * Ee / ((1.0 + Nu) * (1.0 - 2.0 * Nu))
        mu = Ee / (2.0 * (1.0 + Nu))
        D, w = _quadRule(t)
        D, w = np.ascontiguousarray(D), np.ascontiguousarray(w)
        color, nColors = colorInfo[bi]
        for c in range(nColors):
            elems = np.where(color == c)[0]
            if c == 63:
                for e in elems:
                    _assembleColor(X, C, np.asarray([e]), Ee, lam, mu, D, w,
                                   nDim, pattern['nIndices'],
                                   pattern['nIndptr'], dummy, data, nPts, True)
            elif elems.size:
                _assembleColor(X, C, elems, Ee, lam, mu, D, w, nDim,
                               pattern['nIndices'], pattern['nIndptr'], dummy,
                               data, nPts, True)
    blocks = data.reshape(-1, nDim, nDim)
    return bsr_matrix((blocks, pattern['nIndices'], pattern['nIndptr']),
                      shape=(nPts * nDim, nPts * nDim))


@njit(parallel=True, cache=True)
def _transformBlocks(data, indices, indptr, bases):
    nDim = data.shape[1]
    for row in prange(indptr.size - 1):
        left = bases[row]
        for slot in range(indptr[row], indptr[row + 1]):
            right = bases[indices[slot]]
            old = data[slot].copy()
            for i in range(nDim):
                for j in range(nDim):
                    value = 0.0
                    for a in range(nDim):
                        for b in range(nDim):
                            value += left[a, i] * old[a, b] * right[b, j]
                    data[slot, i, j] = value


@njit(parallel=True, cache=True)
def _constrainBlocks(data, indices, indptr, free):
    nDim = data.shape[1]
    for row in prange(indptr.size - 1):
        for slot in range(indptr[row], indptr[row + 1]):
            col = indices[slot]
            for i in range(nDim):
                for j in range(nDim):
                    if not free[row, i] or not free[col, j]:
                        data[slot, i, j] = 0.0
            if col == row:
                for i in range(nDim):
                    if not free[row, i]:
                        data[slot, i, i] = 1.0


def transformElasticityBSR(K, bases):
    """In-place nodal basis transform ``Q.T K Q``."""
    _transformBlocks(K.data, K.indices, K.indptr,
                     np.ascontiguousarray(bases, dtype=np.float64))


def constrainElasticityBSR(K, free):
    """In-place symmetric identity-row Dirichlet constraints."""
    _constrainBlocks(K.data, K.indices, K.indptr,
                     np.ascontiguousarray(free, dtype=np.bool_))


def buildColorInfo(cellsByType, nPts):
    """{blockIndex: (colorOfElement, nColors)} — connectivity-only, cached
    with the assembly pattern across increments and retries."""
    return {bi: colorElements(np.ascontiguousarray(
                np.asarray(conn)), nPts)
            for bi, (t, conn) in enumerate(cellsByType)}


@njit(cache=True)
def _bucketElements(color, nColors):
    """Counting-sort element ids by color once for repeated matrix-free use."""
    counts = np.zeros(nColors, dtype=np.int64)
    for e in range(color.size):
        counts[color[e]] += 1
    offsets = np.empty(nColors + 1, dtype=np.int64)
    offsets[0] = 0
    for c in range(nColors):
        offsets[c + 1] = offsets[c] + counts[c]
    cursor = offsets[:-1].copy()
    order = np.empty(color.size, dtype=np.int32)
    for e in range(color.size):
        c = color[e]
        order[cursor[c]] = e
        cursor[c] += 1
    return order, offsets


def buildColorBuckets(colorInfo):
    """Connectivity-only color buckets reused by every operator application."""
    return {bi: _bucketElements(color, nColors)
            for bi, (color, nColors) in colorInfo.items()}


@njit(parallel=True, cache=True)
def _applyColor(X, conn, elems, Ee, Nu, D, w, x, y, nDim):
    """Apply element stiffness for one race-free color without forming Ke."""
    nG, nN = D.shape[0], D.shape[1]
    nd = nDim
    for ie in prange(elems.size):
        e = elems[ie]
        Xe = np.empty((nN, nd))
        ue = np.empty((nN, nd))
        for a in range(nN):
            node = conn[e, a]
            for d_ in range(nd):
                Xe[a, d_] = X[node, d_]
                ue[a, d_] = x[node, d_]
        lam = Nu * Ee[e] / ((1.0 + Nu) * (1.0 - 2.0 * Nu))
        mu = Ee[e] / (2.0 * (1.0 + Nu))
        J = np.empty((nd, nd))
        Ji = np.empty((nd, nd))
        G = np.empty((nN, nd))
        grad = np.empty((nd, nd))
        stress = np.empty((nd, nd))
        for g in range(nG):
            for r in range(nd):
                for c in range(nd):
                    value = 0.0
                    for a in range(nN):
                        value += D[g, a, r] * Xe[a, c]
                    J[r, c] = value
            if nd == 2:
                det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                inv = 1.0 / det
                Ji[0, 0] = J[1, 1] * inv
                Ji[0, 1] = -J[0, 1] * inv
                Ji[1, 0] = -J[1, 0] * inv
                Ji[1, 1] = J[0, 0] * inv
            else:
                det = (J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
                       - J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0])
                       + J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]))
                inv = 1.0 / det
                Ji[0, 0] = (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) * inv
                Ji[0, 1] = (J[0, 2] * J[2, 1] - J[0, 1] * J[2, 2]) * inv
                Ji[0, 2] = (J[0, 1] * J[1, 2] - J[0, 2] * J[1, 1]) * inv
                Ji[1, 0] = (J[1, 2] * J[2, 0] - J[1, 0] * J[2, 2]) * inv
                Ji[1, 1] = (J[0, 0] * J[2, 2] - J[0, 2] * J[2, 0]) * inv
                Ji[1, 2] = (J[0, 2] * J[1, 0] - J[0, 0] * J[1, 2]) * inv
                Ji[2, 0] = (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]) * inv
                Ji[2, 1] = (J[0, 1] * J[2, 0] - J[0, 0] * J[2, 1]) * inv
                Ji[2, 2] = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]) * inv
            for a in range(nN):
                for d_ in range(nd):
                    value = 0.0
                    for r in range(nd):
                        value += D[g, a, r] * Ji[d_, r]
                    G[a, d_] = value
            for i in range(nd):
                for j in range(nd):
                    value = 0.0
                    for a in range(nN):
                        value += ue[a, i] * G[a, j]
                    grad[i, j] = value
            divu = 0.0
            for i in range(nd):
                divu += grad[i, i]
            for i in range(nd):
                for j in range(nd):
                    stress[i, j] = mu * (grad[i, j] + grad[j, i])
                    if i == j:
                        stress[i, j] += lam * divu
            wd = w[g] * abs(det)
            for a in range(nN):
                node = conn[e, a]
                for i in range(nd):
                    value = 0.0
                    for j in range(nd):
                        value += G[a, j] * stress[i, j]
                    y[node, i] += wd * value


@njit(parallel=True, cache=True)
def _blockDiagonalColor(X, conn, elems, Ee, Nu, D, w, blocks, nDim):
    """Accumulate exact nodal diagonal blocks for one race-free color."""
    nG, nN = D.shape[0], D.shape[1]
    nd = nDim
    for ie in prange(elems.size):
        e = elems[ie]
        Xe = np.empty((nN, nd))
        for a in range(nN):
            for d_ in range(nd):
                Xe[a, d_] = X[conn[e, a], d_]
        lam = Nu * Ee[e] / ((1.0 + Nu) * (1.0 - 2.0 * Nu))
        mu = Ee[e] / (2.0 * (1.0 + Nu))
        J = np.empty((nd, nd))
        Ji = np.empty((nd, nd))
        G = np.empty((nN, nd))
        for g in range(nG):
            for r in range(nd):
                for c in range(nd):
                    value = 0.0
                    for a in range(nN):
                        value += D[g, a, r] * Xe[a, c]
                    J[r, c] = value
            if nd == 2:
                det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                inv = 1.0 / det
                Ji[0, 0] = J[1, 1] * inv
                Ji[0, 1] = -J[0, 1] * inv
                Ji[1, 0] = -J[1, 0] * inv
                Ji[1, 1] = J[0, 0] * inv
            else:
                det = (J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
                       - J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0])
                       + J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]))
                inv = 1.0 / det
                Ji[0, 0] = (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) * inv
                Ji[0, 1] = (J[0, 2] * J[2, 1] - J[0, 1] * J[2, 2]) * inv
                Ji[0, 2] = (J[0, 1] * J[1, 2] - J[0, 2] * J[1, 1]) * inv
                Ji[1, 0] = (J[1, 2] * J[2, 0] - J[1, 0] * J[2, 2]) * inv
                Ji[1, 1] = (J[0, 0] * J[2, 2] - J[0, 2] * J[2, 0]) * inv
                Ji[1, 2] = (J[0, 2] * J[1, 0] - J[0, 0] * J[1, 2]) * inv
                Ji[2, 0] = (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]) * inv
                Ji[2, 1] = (J[0, 1] * J[2, 0] - J[0, 0] * J[2, 1]) * inv
                Ji[2, 2] = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]) * inv
            for a in range(nN):
                for d_ in range(nd):
                    value = 0.0
                    for r in range(nd):
                        value += D[g, a, r] * Ji[d_, r]
                    G[a, d_] = value
            wd = w[g] * abs(det)
            for a in range(nN):
                gg = 0.0
                for k in range(nd):
                    gg += G[a, k] * G[a, k]
                node = conn[e, a]
                for i in range(nd):
                    for j in range(nd):
                        blocks[node, i, j] += wd * (lam + mu) \
                            * G[a, i] * G[a, j]
                        if i == j:
                            blocks[node, i, j] += wd * mu * gg


def _forEachColor(P, cellsByType, E, Nu, nDim, colorInfo, colorBuckets,
                  kernel, output):
    """Shared type/color traversal for matrix-free action and diagonal."""
    from ..quadrature import _quadRule
    from numba import get_num_threads, set_num_threads
    X = np.ascontiguousarray(np.asarray(P, dtype=np.float64)[:, :nDim])
    serial = sum(len(conn) for _t, conn in cellsByType) < 100_000
    oldThreads = get_num_threads()
    if serial:
        set_num_threads(1)
    base = 0
    try:
        for bi, (t, connAll) in enumerate(cellsByType):
            C = np.ascontiguousarray(connAll)
            nE = C.shape[0]
            Ee = np.ascontiguousarray(np.asarray(E[base:base + nE],
                                                 dtype=np.float64))
            base += nE
            D, w = _quadRule(t)
            D, w = np.ascontiguousarray(D), np.ascontiguousarray(w)
            if serial:
                kernel(X, C, np.arange(nE, dtype=np.int32), Ee, Nu, D, w,
                       output, nDim)
                continue
            order, offsets = colorBuckets[bi]
            nColors = colorInfo[bi][1]
            for c in range(nColors):
                elems = order[offsets[c]:offsets[c + 1]]
                if c == 63:
                    for e in elems:
                        kernel(X, C, np.asarray([e], dtype=np.int32), Ee, Nu,
                               D, w, output, nDim)
                elif elems.size:
                    kernel(X, C, elems, Ee, Nu, D, w, output, nDim)
    finally:
        if serial:
            set_num_threads(oldThreads)


def applyElasticityMatrixFree(P, cellsByType, E, Nu, nDim, colorInfo,
                              colorBuckets, vector):
    """Exact ``K @ vector`` from element actions; no global matrix or Ke."""
    x = np.ascontiguousarray(np.asarray(vector, dtype=np.float64)
                             .reshape(-1, nDim))
    y = np.zeros_like(x)

    def apply(X, C, elems, Ee, nu, D, w, output, nd):
        _applyColor(X, C, elems, Ee, nu, D, w, x, output, nd)

    _forEachColor(P, cellsByType, E, Nu, nDim, colorInfo, colorBuckets,
                  apply, y)
    return y.ravel()


def elasticityDiagonalMatrixFree(P, cellsByType, E, Nu, nDim, colorInfo,
                                 colorBuckets):
    """Exact diagonal of the unassembled elasticity operator."""
    blocks = elasticityBlockDiagonalMatrixFree(
        P, cellsByType, E, Nu, nDim, colorInfo, colorBuckets)
    return np.diagonal(blocks, axis1=1, axis2=2).copy().ravel()


def elasticityBlockDiagonalMatrixFree(P, cellsByType, E, Nu, nDim,
                                      colorInfo, colorBuckets):
    """Exact nodal block diagonal of the unassembled elasticity operator."""
    blocks = np.zeros((np.asarray(P).shape[0], nDim, nDim), dtype=np.float64)
    _forEachColor(P, cellsByType, E, Nu, nDim, colorInfo, colorBuckets,
                  _blockDiagonalColor, blocks)
    return blocks
