"""
Shared numba (@njit) linear-algebra primitives -- small, register-level, fully
reusable building blocks so the same routine is never rewritten across modules.

Contents:
  * Scalar 3-vector ops:  cross_product, dot_product, sub, norm3, normalize3
  * In-place inverses:    invert_2x2, invert_3x3   (M, M_inv)
  * 2x2 symmetric eigen:  eig2, from_eig2, pow2, symprod2
  * 3x3 symmetric eigen:  eig3 (cyclic Jacobi), from_eig3, pow3, msym_triple3

These are intentionally domain-agnostic (no metric/mesh semantics). Higher-level
operations that carry meaning -- e.g. metric intersection (clamp eigenvalues to
>= 1) -- live with their domain (FunScripts/AnisotropicMetric.py) and call these.

This file imports only math/numpy/numba, so it can be imported from both
Readers/* and FunScripts/* without creating cycles.
"""

import math
import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Scalar 3-vector helpers
# ---------------------------------------------------------------------------
@njit(inline='always')
def cross_product(ax, ay, az, bx, by, bz):
    return (ay * bz - az * by), (az * bx - ax * bz), (ax * by - ay * bx)


@njit(inline='always')
def dot_product(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


@njit(inline='always')
def sub(ax, ay, az, bx, by, bz):
    return ax - bx, ay - by, az - bz


@njit(inline='always')
def norm3(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


@njit(inline='always')
def normalize3(x, y, z):
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-300:
        return 0.0, 0.0, 0.0
    return x / n, y / n, z / n


# ---------------------------------------------------------------------------
# In-place analytic matrix inverses (M, M_inv passed in; M_inv filled)
# ---------------------------------------------------------------------------
@njit(fastmath=True)
def invert_2x2(M, M_inv):
    """In-place 2x2 matrix inversion"""
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) < 1e-12:
        M_inv[0, 0] = 1.0
        M_inv[0, 1] = 0.0
        M_inv[1, 0] = 0.0
        M_inv[1, 1] = 1.0
    else:
        det_inv = 1.0 / det
        M_inv[0, 0] = M[1, 1] * det_inv
        M_inv[0, 1] = -M[0, 1] * det_inv
        M_inv[1, 0] = -M[1, 0] * det_inv
        M_inv[1, 1] = M[0, 0] * det_inv


@njit(fastmath=True)
def invert_3x3(M, M_inv):
    """In-place 3x3 matrix inversion using analytical formula"""
    det = (M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) -
           M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0]) +
           M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]))

    if abs(det) < 1e-12:
        for i in range(3):
            for j in range(3):
                M_inv[i, j] = 1.0 if i == j else 0.0
    else:
        det_inv = 1.0 / det
        M_inv[0, 0] = (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) * det_inv
        M_inv[0, 1] = (M[0, 2] * M[2, 1] - M[0, 1] * M[2, 2]) * det_inv
        M_inv[0, 2] = (M[0, 1] * M[1, 2] - M[0, 2] * M[1, 1]) * det_inv
        M_inv[1, 0] = (M[1, 2] * M[2, 0] - M[1, 0] * M[2, 2]) * det_inv
        M_inv[1, 1] = (M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0]) * det_inv
        M_inv[1, 2] = (M[0, 2] * M[1, 0] - M[0, 0] * M[1, 2]) * det_inv
        M_inv[2, 0] = (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]) * det_inv
        M_inv[2, 1] = (M[0, 1] * M[2, 0] - M[0, 0] * M[2, 1]) * det_inv
        M_inv[2, 2] = (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) * det_inv


# ---------------------------------------------------------------------------
# 2x2 symmetric eigen-algebra (matrix as scalars a,b,c = [[a,b],[b,c]])
# ---------------------------------------------------------------------------
@njit(inline='always')
def eig2(a, b, c):
    """Symmetric [[a,b],[b,c]] -> (l1>=l2, unit eigenvector (cx,cy) for l1)."""
    tr = 0.5 * (a + c)
    df = 0.5 * (a - c)
    rad = math.sqrt(df * df + b * b)
    l1 = tr + rad
    l2 = tr - rad
    if rad < 1e-300:
        return l1, l2, 1.0, 0.0
    # Use the row whose diagonal subtraction has the larger magnitude. When
    # c dominates, l1-c can be pure roundoff; normalizing that residual would
    # select the x axis for a diagonal matrix whose principal direction is y.
    # The alternative (b,l1-a) avoids this cancellation, including at walls.
    vx = l1-c if a >= c else b
    vy = b if a >= c else l1-a
    nrm = math.sqrt(vx * vx + vy * vy)
    if nrm < 1e-300:
        vx = b
        vy = l1 - a
        nrm = math.sqrt(vx * vx + vy * vy)
        if nrm < 1e-300:
            return l1, l2, 1.0, 0.0
    return l1, l2, vx / nrm, vy / nrm


@njit(inline='always')
def from_eig2(l1, l2, cx, cy):
    """Rebuild symmetric 2x2 from eigenpairs: v1=(cx,cy), v2=(-cy,cx)."""
    a = l1 * cx * cx + l2 * cy * cy
    b = (l1 - l2) * cx * cy
    c = l1 * cy * cy + l2 * cx * cx
    return a, b, c


@njit(inline='always')
def pow2(a, b, c, p):
    """Matrix power of a symmetric 2x2 (eigenvalues raised to p)."""
    l1, l2, cx, cy = eig2(a, b, c)
    p1 = l1 ** p if l1 > 0.0 else 0.0
    p2 = l2 ** p if l2 > 0.0 else 0.0
    return from_eig2(p1, p2, cx, cy)


@njit(inline='always')
def symprod2(a1, b1, c1, a2, b2, c2):
    """Product S1 @ S2 of two symmetric 2x2 matrices as a full 2x2 (m00,m01,m10,m11)."""
    m00 = a1 * a2 + b1 * b2
    m01 = a1 * b2 + b1 * c2
    m10 = b1 * a2 + c1 * b2
    m11 = b1 * b2 + c1 * c2
    return m00, m01, m10, m11


# ---------------------------------------------------------------------------
# 3x3 symmetric eigen-algebra (matrix as 6 scalars a00,a01,a02,a11,a12,a22)
# ---------------------------------------------------------------------------
@njit
def eig3(a00, a01, a02, a11, a12, a22):
    """
    Eigenvalues/eigenvectors of a symmetric 3x3 by cyclic Jacobi iteration.
    Returns (l0,l1,l2, v0(3), v1(3), v2(3)) -- eigenvalues and the eigenvectors
    as columns of the accumulated rotation V. Converges to machine precision in
    a few sweeps and handles repeated eigenvalues robustly.
    """
    a, b, c = a00, a11, a22          # diagonal
    p, q, r = a01, a02, a12          # off-diagonal (01, 02, 12)
    v00 = 1.0; v01 = 0.0; v02 = 0.0
    v10 = 0.0; v11 = 1.0; v12 = 0.0
    v20 = 0.0; v21 = 0.0; v22 = 1.0

    for _ in range(12):
        if abs(p) + abs(q) + abs(r) < 1e-30:
            break
        # --- rotate (0,1): zero p=a_01 ---
        if abs(p) > 1e-300:
            tau = (b - a) / (2.0 * p)
            t = (1.0 if tau >= 0.0 else -1.0) / (abs(tau) + math.sqrt(1.0 + tau * tau))
            cs = 1.0 / math.sqrt(1.0 + t * t); sn = t * cs
            na = a - t * p; nb = b + t * p
            nq = cs * q - sn * r
            nr = sn * q + cs * r
            a, b, p, q, r = na, nb, 0.0, nq, nr
            for row in range(3):
                if row == 0: x0, x1 = v00, v01
                elif row == 1: x0, x1 = v10, v11
                else: x0, x1 = v20, v21
                n0 = cs * x0 - sn * x1; n1 = sn * x0 + cs * x1
                if row == 0: v00, v01 = n0, n1
                elif row == 1: v10, v11 = n0, n1
                else: v20, v21 = n0, n1
        # --- rotate (0,2): zero q=a_02 ---
        if abs(q) > 1e-300:
            tau = (c - a) / (2.0 * q)
            t = (1.0 if tau >= 0.0 else -1.0) / (abs(tau) + math.sqrt(1.0 + tau * tau))
            cs = 1.0 / math.sqrt(1.0 + t * t); sn = t * cs
            na = a - t * q; nc = c + t * q
            np_ = cs * p - sn * r
            nr = sn * p + cs * r
            a, c, q, p, r = na, nc, 0.0, np_, nr
            for row in range(3):
                if row == 0: x0, x2 = v00, v02
                elif row == 1: x0, x2 = v10, v12
                else: x0, x2 = v20, v22
                n0 = cs * x0 - sn * x2; n2 = sn * x0 + cs * x2
                if row == 0: v00, v02 = n0, n2
                elif row == 1: v10, v12 = n0, n2
                else: v20, v22 = n0, n2
        # --- rotate (1,2): zero r=a_12 ---
        if abs(r) > 1e-300:
            tau = (c - b) / (2.0 * r)
            t = (1.0 if tau >= 0.0 else -1.0) / (abs(tau) + math.sqrt(1.0 + tau * tau))
            cs = 1.0 / math.sqrt(1.0 + t * t); sn = t * cs
            nb = b - t * r; nc = c + t * r
            np_ = cs * p - sn * q
            nq = sn * p + cs * q
            b, c, r, p, q = nb, nc, 0.0, np_, nq
            for row in range(3):
                if row == 0: x1, x2 = v01, v02
                elif row == 1: x1, x2 = v11, v12
                else: x1, x2 = v21, v22
                n1 = cs * x1 - sn * x2; n2 = sn * x1 + cs * x2
                if row == 0: v01, v02 = n1, n2
                elif row == 1: v11, v12 = n1, n2
                else: v21, v22 = n1, n2

    return (a, b, c,
            v00, v10, v20,    # column 0
            v01, v11, v21,    # column 1
            v02, v12, v22)    # column 2


@njit
def from_eig3(l0, l1, l2, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z):
    """Rebuild symmetric 3x3 = sum_k lam_k v_k v_k^T -> (a00,a01,a02,a11,a12,a22)."""
    a00 = l0 * v0x * v0x + l1 * v1x * v1x + l2 * v2x * v2x
    a11 = l0 * v0y * v0y + l1 * v1y * v1y + l2 * v2y * v2y
    a22 = l0 * v0z * v0z + l1 * v1z * v1z + l2 * v2z * v2z
    a01 = l0 * v0x * v0y + l1 * v1x * v1y + l2 * v2x * v2y
    a02 = l0 * v0x * v0z + l1 * v1x * v1z + l2 * v2x * v2z
    a12 = l0 * v0y * v0z + l1 * v1y * v1z + l2 * v2y * v2z
    return a00, a01, a02, a11, a12, a22


@njit
def pow3(a00, a01, a02, a11, a12, a22, pw):
    """Matrix power of a symmetric 3x3 (eigenvalues raised to pw)."""
    l0, l1, l2, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z = eig3(a00, a01, a02, a11, a12, a22)
    p0 = l0 ** pw if l0 > 0.0 else 0.0
    p1 = l1 ** pw if l1 > 0.0 else 0.0
    p2 = l2 ** pw if l2 > 0.0 else 0.0
    return from_eig3(p0, p1, p2, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z)


@njit
def msym_triple3(x00, x01, x02, x11, x12, x22, b00, b01, b02, b11, b12, b22):
    """Symmetric triple product X B X for symmetric 3x3 X, B (returns 6 sym comps)."""
    p00 = x00 * b00 + x01 * b01 + x02 * b02
    p01 = x00 * b01 + x01 * b11 + x02 * b12
    p02 = x00 * b02 + x01 * b12 + x02 * b22
    p10 = x01 * b00 + x11 * b01 + x12 * b02
    p11 = x01 * b01 + x11 * b11 + x12 * b12
    p12 = x01 * b02 + x11 * b12 + x12 * b22
    p20 = x02 * b00 + x12 * b01 + x22 * b02
    p21 = x02 * b01 + x12 * b11 + x22 * b12
    p22 = x02 * b02 + x12 * b12 + x22 * b22
    r00 = p00 * x00 + p01 * x01 + p02 * x02
    r01 = p00 * x01 + p01 * x11 + p02 * x12
    r02 = p00 * x02 + p01 * x12 + p02 * x22
    r11 = p10 * x01 + p11 * x11 + p12 * x12
    r12 = p10 * x02 + p11 * x12 + p12 * x22
    r22 = p20 * x02 + p21 * x12 + p22 * x22
    return r00, r01, r02, r11, r12, r22
