"""SU2-compatible reference-element quadrature rules and shape-function derivatives.

Extracted verbatim from ``MeshAdaptation/Scripts/FunScripts/DeformMesh_Fun.py``.
"""

import numpy as np

from .readers.MeshWriter import HEX, LINE, PRISM, PYRA, QUAD, TET, TRI

_ASSEMBLY_CHUNK = 50_000


def _quadRule(t):
    """(dshape (nG, nNodes, rdim), weights (nG,)) -- SU2's rules."""
    if t == TRI:                                            # 1 pt, w = 1/2
        return np.array([[[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]]), np.array([0.5])
    if t == QUAD:                                           # 2x2 Gauss, w = 1
        g = 0.577350269189626
        pts = [(-g, -g), (g, -g), (g, g), (-g, g)]
        return np.stack([_dQuad(x, e) for x, e in pts]), np.ones(4)
    if t == TET:                                            # 1 pt, w = 1/6
        d = np.array([[[-1.0, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        return d, np.array([1.0 / 6.0])
    if t == HEX:                                            # 2x2x2 Gauss, w = 1
        g = 0.577350269189626
        pts = [(sx * g, sy * g, sz * g)
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        return np.stack([_dHex(*p) for p in pts]), np.ones(8)
    if t == PRISM:                                          # SU2 6-pt, w = 1/6
        g = 0.577350269189626
        tp = [(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)]
        pts = [(x, e, z) for x in (-g, g) for (e, z) in tp]
        return np.stack([_dPrismSU2(*p) for p in pts]), np.full(6, 1.0 / 6.0)
    if t == PYRA:                                           # SU2 5-pt rule
        z1, z2, w = 0.1531754163448146, 0.6372983346207416, 0.133333333333333
        pts = [(0.5, 0.0, z1), (0.0, 0.5, z1), (-0.5, 0.0, z1),
               (0.0, -0.5, z1), (0.0, 0.0, z2)]
        return np.stack([_dPyramSU2(*p) for p in pts]), np.full(5, w)
    raise ValueError(f"no quadrature rule for element type {t}")


def _dQuad(xi, eta):
    return 0.25 * np.array([[-(1 - eta), -(1 - xi)], [+(1 - eta), -(1 + xi)],
                            [+(1 + eta), +(1 + xi)], [-(1 + eta), +(1 - xi)]])


def _dHex(xi, eta, ze):
    sx = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float)
    sy = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=float)
    sz = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float)
    d = np.empty((8, 3))
    d[:, 0] = 0.125 * sx * (1 + sy * eta) * (1 + sz * ze)
    d[:, 1] = 0.125 * sy * (1 + sx * xi) * (1 + sz * ze)
    d[:, 2] = 0.125 * sz * (1 + sx * xi) * (1 + sy * eta)
    return d


def _dPrismSU2(xi, eta, ze):
    # SU2 wedge: extrusion along Xi, triangle coords (Eta, Zeta)
    d = np.empty((6, 3))
    d[0] = [-0.5 * eta, 0.5 * (1 - xi), 0.0]
    d[1] = [-0.5 * ze, 0.0, 0.5 * (1 - xi)]
    d[2] = [-0.5 * (1 - eta - ze), -0.5 * (1 - xi), -0.5 * (1 - xi)]
    d[3] = [0.5 * eta, 0.5 * (xi + 1), 0.0]
    d[4] = [0.5 * ze, 0.0, 0.5 * (xi + 1)]
    d[5] = [0.5 * (1 - eta - ze), -0.5 * (xi + 1), -0.5 * (xi + 1)]
    return d


def _dPyramSU2(xi, eta, ze):
    # SU2 rational pyramid: base at Zeta=0, apex node 4 at Zeta=1
    d = np.empty((5, 3))
    zm = ze - 1.0
    d[0, 0] = 0.5 * (ze - xi - 1.0) / zm
    d[1, 0] = 0.5 * xi / zm
    d[2, 0] = 0.5 * (1.0 - ze - xi) / zm
    d[3, 0] = d[1, 0]
    d[4, 0] = 0.0
    d[0, 1] = 0.5 * eta / zm
    d[1, 1] = 0.5 * (ze - eta - 1.0) / zm
    d[2, 1] = d[0, 1]
    d[3, 1] = 0.5 * (1.0 - ze - eta) / zm
    d[4, 1] = 0.0
    zm2 = (1.0 - ze) ** 2
    d[0, 2] = 0.25 * (-1.0 + 2.0 * ze - ze * ze - eta * eta + xi * xi) / zm2
    d[1, 2] = 0.25 * (-1.0 + 2.0 * ze - ze * ze + eta * eta - xi * xi) / zm2
    d[2, 2] = d[0, 2]
    d[3, 2] = d[1, 2]
    d[4, 2] = 1.0
    return d


def _invDet(J):
    """Vectorized inverse and determinant of (..., d, d), d in (2, 3)."""
    if J.shape[-1] == 2:
        det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
        inv = np.empty_like(J)
        inv[..., 0, 0] = J[..., 1, 1]
        inv[..., 0, 1] = -J[..., 0, 1]
        inv[..., 1, 0] = -J[..., 1, 0]
        inv[..., 1, 1] = J[..., 0, 0]
        return inv / det[..., None, None], det
    det = np.linalg.det(J)
    return np.linalg.inv(J), det


def elementVolumes(P, cellsByType, nDim):
    """Per-cell volume by the operator's own quadrature (concatenated order)."""
    out = []
    X = np.asarray(P, dtype=np.float64)[:, :nDim]
    for t, conn in cellsByType:
        D, w = _quadRule(t)
        C = np.asarray(conn, dtype=np.int64)
        volume = np.empty(C.shape[0], dtype=np.float64)
        for start in range(0, len(C), _ASSEMBLY_CHUNK):
            stop = min(start + _ASSEMBLY_CHUNK, len(C))
            J = np.einsum('gar,ead->egrd', D, X[C[start:stop]])
            if nDim == 2:
                det = J[..., 0, 0] * J[..., 1, 1] \
                    - J[..., 0, 1] * J[..., 1, 0]
            else:
                det = (J[..., 0, 0]
                       * (J[..., 1, 1] * J[..., 2, 2]
                          - J[..., 1, 2] * J[..., 2, 1])
                       - J[..., 0, 1]
                       * (J[..., 1, 0] * J[..., 2, 2]
                          - J[..., 1, 2] * J[..., 2, 0])
                       + J[..., 0, 2]
                       * (J[..., 1, 0] * J[..., 2, 1]
                          - J[..., 1, 1] * J[..., 2, 0]))
            volume[start:stop] = np.abs(det) @ w
        out.append(volume)
    return np.concatenate(out)


__all__ = [
    "LINE",
    "TRI",
    "QUAD",
    "TET",
    "HEX",
    "PRISM",
    "PYRA",
    "_quadRule",
    "_dQuad",
    "_dHex",
    "_dPrismSU2",
    "_dPyramSU2",
    "_invDet",
    "elementVolumes",
]
