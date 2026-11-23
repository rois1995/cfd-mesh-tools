"""API contract and numerical oracle tests for cfd-mesh-tools.

These tests lock down the exact function signatures, array shapes, dtypes,
packing conventions, and numerical accuracy expected by downstream consumers
(LaplaceSolver_V2 and MeshAdaptation).

Run standalone:
    python3 packages/cfd_mesh_tools/tests/test_contract.py
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path
import struct
import tempfile

import numpy as np


def _grid_neighbors(n: int, ndim: int, *, two_ring: bool = False):
    """Build a regular n^ndim grid in [0, 1]^ndim and its padded neighbor table."""
    axes = [np.linspace(0.0, 1.0, n) for _ in range(ndim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.zeros((n ** ndim, 3), dtype=np.float64)
    for d in range(ndim):
        pts[:, d] = mesh[d].ravel()

    # Build 1-ring (all cell neighbors within Chebyshev distance 1, or 2 for two_ring)
    rad = 2 if two_ring else 1
    coords_i = np.stack([g.ravel() for g in np.meshgrid(*[np.arange(n)] * ndim, indexing="ij")], axis=1)
    idx_map = {tuple(c): i for i, c in enumerate(coords_i)}
    rows = []
    for i, c in enumerate(coords_i):
        nbrs = []
        ranges = [range(max(0, c[d] - rad), min(n, c[d] + rad + 1)) for d in range(ndim)]
        if ndim == 2:
            for a in ranges[0]:
                for b in ranges[1]:
                    j = idx_map[(a, b)]
                    if j != i:
                        nbrs.append(j)
        else:
            for a in ranges[0]:
                for b in ranges[1]:
                    for k in ranges[2]:
                        j = idx_map[(a, b, k)]
                        if j != i:
                            nbrs.append(j)
        rows.append(nbrs)
    max_nb = max(len(r) for r in rows)
    neighbors = np.full((len(pts), max_nb), -1, dtype=np.int64)
    counts = np.zeros(len(pts), dtype=np.int64)
    for i, r in enumerate(rows):
        neighbors[i, :len(r)] = r
        counts[i] = len(r)
    return pts, neighbors, counts


def test_public_api_and_signatures():
    """Lock down exported names and parameter signatures expected by consumers."""
    import cfd_mesh_tools as cmt
    from cfd_mesh_tools import gradients, quadrature, linalg, assembly, readers

    assert hasattr(cmt, "__version__")
    expected_sigs = {
        gradients.compute_wls_coefficients_optimized: ["nodes", "neighbors", "nNeighbors", "nDim", "power"],
        gradients.compute_wls_coefficients_pinv: ["nodes", "neighbors", "nNeighbors", "manifoldDim", "ambientDim", "power"],
        gradients.compute_wls_coefficients_deg2: ["nodes", "neighbors", "nNeighbors", "nDim", "power", "fallback_coeffs", "want_hessian"],
        gradients.ComputeWLSGradients_Numba: ["fields", "neighbors", "NNeighbors", "wls_coeffs", "nDim"],
        gradients.ComputeWLSHessians_Numba: ["fields", "neighbors", "NNeighbors", "hess_coeffs", "nH"],
        gradients.ComputeBJLimiter_Numba: ["fields", "grads", "points", "neighbors", "NNeighbors", "nDim"],
        gradients.ComputeVenkatLimiter_Numba: ["fields", "grads", "points", "neighbors", "NNeighbors", "nDim", "eps_sq"],
        gradients.ComputeNishikawaRLimiter_Numba: ["fields", "grads", "points", "neighbors", "NNeighbors", "nDim", "R"],
        quadrature._quadRule: ["t"],
        assembly.elementVolumesNumba: ["P", "cellsByType", "nDim"],
    }
    for fn, params in expected_sigs.items():
        actual = list(inspect.signature(fn).parameters.keys())
        assert actual == params, f"{fn.__name__}: got {actual}, expected {params}"

    # All 9 intact reader modules must remain importable
    for name in (
        "BinaryRestartReader", "CGNSReader", "CSVReader", "ElementsUtilities",
        "MeshReader", "MeshWriter", "SU2Reader", "VTUReader", "setup_logging",
    ):
        assert hasattr(readers, name), f"missing readers.{name}"
    assert hasattr(linalg, "invert_2x2") and hasattr(linalg, "invert_3x3")
    print("[PASS] public API and signature contract")


def test_wls_degree1_and_scale_invariance():
    """Degree-1 WLS recovers affine fields exactly in 2D/3D across 12 orders of magnitude."""
    from cfd_mesh_tools.gradients import (
        ComputeWLSGradients_Numba,
        compute_wls_coefficients_optimized,
    )

    for ndim in (2, 3):
        pts0, nb, cnt = _grid_neighbors(4, ndim)
        for scale in (1e-6, 1.0, 1e6):
            pts = pts0 * scale + np.array([1.2, -0.7, 0.4]) * scale
            coeffs = compute_wls_coefficients_optimized(pts, nb, cnt, ndim, 2.0)
            assert coeffs.shape == (len(pts), nb.shape[1], 3)
            assert coeffs.dtype == np.float64
            if ndim == 2:
                assert np.all(coeffs[:, :, 2] == 0.0)

            b = np.array([[2.5, -1.3, 0.9], [-0.4, 3.1, -2.2]])[:, :ndim]
            f = np.array([0.7 * scale + pts[:, :ndim] @ b[0], -1.1 * scale + pts[:, :ndim] @ b[1]])
            grads = ComputeWLSGradients_Numba(f, nb, cnt, coeffs, ndim)
            assert grads.shape == (2, len(pts), 3)
            assert grads.dtype == np.float64
            if ndim == 2:
                assert np.all(grads[:, :, 2] == 0.0)
            for k in range(2):
                expected = np.broadcast_to(b[k], (len(pts), ndim))
                np.testing.assert_allclose(grads[k, :, :ndim], expected, rtol=0, atol=1e-12)
    print("[PASS] WLS degree-1 2D/3D affine exactness & coordinate scale invariance")


def test_wls_degree2_gradient_hessian_and_fallback():
    """Degree-2 WLS recovers quadratic gradients and packed upper-triangular Hessians."""
    from cfd_mesh_tools.gradients import (
        ComputeWLSGradients_Numba,
        ComputeWLSHessians_Numba,
        compute_wls_coefficients_deg2,
        compute_wls_coefficients_optimized,
    )

    # 2D quadratic: f(x, y) = 1 + 2x - 3y + 0.5*(4 x^2 + 2*(-1.5) x y + 5 y^2)
    pts2, nb2, cnt2 = _grid_neighbors(5, 2, two_ring=True)
    c1 = compute_wls_coefficients_optimized(pts2, nb2, cnt2, 2, 2.0)
    c2_no_h, h_dummy = compute_wls_coefficients_deg2(pts2, nb2, cnt2, 2, 2.0, c1, False)
    assert h_dummy.shape == (1, 1, 1)
    c2, hc2 = compute_wls_coefficients_deg2(pts2, nb2, cnt2, 2, 2.0, c1, True)
    assert c2.shape == c1.shape and hc2.shape == (len(pts2), nb2.shape[1], 3)
    np.testing.assert_allclose(c2, c2_no_h, atol=1e-14)

    x, y = pts2[:, 0], pts2[:, 1]
    f2 = (1.0 + 2.0 * x - 3.0 * y + 2.0 * x * x - 1.5 * x * y + 2.5 * y * y)[None, :]
    g2 = ComputeWLSGradients_Numba(f2, nb2, cnt2, c2, 2)
    h2 = ComputeWLSHessians_Numba(f2, nb2, cnt2, hc2, 3)
    np.testing.assert_allclose(g2[0, :, 0], 2.0 + 4.0 * x - 1.5 * y, atol=1e-10)
    np.testing.assert_allclose(g2[0, :, 1], -3.0 - 1.5 * x + 5.0 * y, atol=1e-10)
    # Packed 2D Hessian order: [Hxx, Hxy, Hyy] = [4.0, -1.5, 5.0]
    np.testing.assert_allclose(h2[0, :, 0], 4.0, atol=1e-9)
    np.testing.assert_allclose(h2[0, :, 1], -1.5, atol=1e-9)
    np.testing.assert_allclose(h2[0, :, 2], 5.0, atol=1e-9)

    # 3D quadratic: packed 3D Hessian order [Hxx, Hxy, Hxz, Hyy, Hyz, Hzz]
    pts3, nb3, cnt3 = _grid_neighbors(4, 3, two_ring=True)
    c1_3 = compute_wls_coefficients_optimized(pts3, nb3, cnt3, 3, 2.0)
    c2_3, hc3 = compute_wls_coefficients_deg2(pts3, nb3, cnt3, 3, 2.0, c1_3, True)
    assert hc3.shape == (len(pts3), nb3.shape[1], 6)
    x, y, z = pts3[:, 0], pts3[:, 1], pts3[:, 2]
    H_true = [2.0, -0.5, 0.7, -1.4, 0.9, 3.2]  # Hxx, Hxy, Hxz, Hyy, Hyz, Hzz
    f3 = (0.3 - x + 2.0 * y - 0.4 * z
          + 0.5 * H_true[0] * x * x + H_true[1] * x * y + H_true[2] * x * z
          + 0.5 * H_true[3] * y * y + H_true[4] * y * z + 0.5 * H_true[5] * z * z)[None, :]
    g3 = ComputeWLSGradients_Numba(f3, nb3, cnt3, c2_3, 3)
    h3 = ComputeWLSHessians_Numba(f3, nb3, cnt3, hc3, 6)
    np.testing.assert_allclose(g3[0, :, 0], -1.0 + H_true[0] * x + H_true[1] * y + H_true[2] * z, atol=1e-10)
    np.testing.assert_allclose(g3[0, :, 1], 2.0 + H_true[1] * x + H_true[3] * y + H_true[4] * z, atol=1e-10)
    np.testing.assert_allclose(g3[0, :, 2], -0.4 + H_true[2] * x + H_true[4] * y + H_true[5] * z, atol=1e-10)
    for comp, val in enumerate(H_true):
        np.testing.assert_allclose(h3[0, :, comp], val, atol=1e-9)

    # Underdetermined row (nn < m=5 in 2D) must fall back to degree-1 gradient and zero Hessian
    cnt_short = cnt2.copy()
    cnt_short[0] = 3
    c2_fb, hc_fb = compute_wls_coefficients_deg2(pts2, nb2, cnt_short, 2, 2.0, c1, True)
    np.testing.assert_allclose(c2_fb[0, :3, :2], c1[0, :3, :2], atol=0)
    np.testing.assert_allclose(hc_fb[0], 0.0, atol=0)
    print("[PASS] WLS degree-2 gradient, packed Hessian (2D/3D), and fallback contract")


def test_wls_manifold_pinv_and_limiters():
    """Manifold pinv recovers tangential gradient; slope limiters stay in [0, 1]."""
    from cfd_mesh_tools.gradients import (
        ComputeBJLimiter_Numba,
        ComputeNishikawaRLimiter_Numba,
        ComputeVenkatLimiter_Numba,
        ComputeWLSGradients_Numba,
        compute_wls_coefficients_optimized,
        compute_wls_coefficients_pinv,
    )

    # 2D plane tilted in 3D: z = 0.5 x - 0.25 y, normal n_hat
    pts2, nb, cnt = _grid_neighbors(4, 2)
    pts3 = pts2.copy()
    pts3[:, 2] = 0.5 * pts3[:, 0] - 0.25 * pts3[:, 1]
    c_surf = compute_wls_coefficients_pinv(pts3, nb, cnt, 2, 3, 2.0)
    a = np.array([1.2, -0.8, 2.0])
    n = np.array([-0.5, 0.25, 1.0])
    n /= np.linalg.norm(n)
    a_tan = a - np.dot(a, n) * n
    f = (pts3 @ a)[None, :]
    g = ComputeWLSGradients_Numba(f, nb, cnt, c_surf, 3)
    np.testing.assert_allclose(g[0], np.broadcast_to(a_tan, (len(pts3), 3)), atol=1e-11)

    # Slope limiters on a 2D step + smooth field
    c1 = compute_wls_coefficients_optimized(pts2, nb, cnt, 2, 2.0)
    f_step = (pts2[:, 0] > 0.5).astype(np.float64)[None, :]
    g_step = ComputeWLSGradients_Numba(f_step, nb, cnt, c1, 2)
    for phi in (
        ComputeBJLimiter_Numba(f_step, g_step, pts2, nb, cnt, 2),
        ComputeVenkatLimiter_Numba(f_step, g_step, pts2, nb, cnt, 2, 1e-4),
        ComputeNishikawaRLimiter_Numba(f_step, g_step, pts2, nb, cnt, 2, 3),
    ):
        assert phi.shape == (1, len(pts2))
        assert np.all((phi >= 0.0) & (phi <= 1.0))
    print("[PASS] WLS manifold pinv and slope limiters contract")


def test_quadrature_and_element_volumes():
    """Verify _quadRule shapes, partition-of-unity, and elementVolumesNumba on all 6 element types."""
    from cfd_mesh_tools.assembly import elementVolumesNumba
    from cfd_mesh_tools.quadrature import HEX, PRISM, PYRA, QUAD, TET, TRI, _quadRule, elementVolumes

    expected_shapes = {
        TRI: ((1, 3, 2), (1,)),
        QUAD: ((4, 4, 2), (4,)),
        TET: ((1, 4, 3), (1,)),
        HEX: ((8, 8, 3), (8,)),
        PRISM: ((6, 6, 3), (6,)),
        PYRA: ((5, 5, 3), (5,)),
    }
    for code, (dshape, wshape) in expected_shapes.items():
        D, w = _quadRule(code)
        assert D.shape == dshape and w.shape == wshape, (code, D.shape, w.shape)
        # Partition of unity: sum_a dN_a / dxi_r == 0 at every Gauss point
        np.testing.assert_allclose(D.sum(axis=1), 0.0, atol=1e-13)

    # Unit reference elements: exact volumes
    # 2D: unit triangle (area 0.5) + unit square (area 1.0)
    P2 = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    cells2 = [(TRI, np.array([[0, 1, 2]])), (QUAD, np.array([[0, 1, 3, 2]]))]
    v2_numba = elementVolumesNumba(P2, cells2, 2)
    v2_np = elementVolumes(P2, cells2, 2)
    np.testing.assert_allclose(v2_numba, [0.5, 1.0], atol=1e-13)
    np.testing.assert_allclose(v2_np, [0.5, 1.0], atol=1e-13)

    # 3D: unit tet (1/6), unit cube hex (1.0), unit prism (0.5), unit pyramid (1/3)
    P3 = np.array([
        [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
        [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
        [0.5, 0.5, 1.0],
    ])
    cells3 = [
        (TET, np.array([[0, 1, 3, 4]])),
        (HEX, np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
        (PRISM, np.array([[0, 1, 3, 4, 5, 7]])),
        (PYRA, np.array([[0, 1, 2, 3, 8]])),
    ]
    v3_numba = elementVolumesNumba(P3, cells3, 3)
    v3_np = elementVolumes(P3, cells3, 3)
    np.testing.assert_allclose(v3_numba, [1.0 / 6.0, 1.0, 0.5, 1.0 / 3.0], atol=1e-13)
    np.testing.assert_allclose(v3_np, [1.0 / 6.0, 1.0, 0.5, 1.0 / 3.0], atol=1e-13)
    print("[PASS] quadrature rules and elementVolumesNumba contract (all 6 types)")


def test_linalg_inverses():
    """Verify invert_2x2 and invert_3x3 exact inverses and singular identity fallback."""
    from cfd_mesh_tools.linalg import invert_2x2, invert_3x3

    M2 = np.array([[4.0, 1.0], [2.0, 3.0]])
    inv2 = np.zeros((2, 2))
    invert_2x2(M2, inv2)
    np.testing.assert_allclose(M2 @ inv2, np.eye(2), atol=1e-14)
    # Singular fallback -> identity
    invert_2x2(np.zeros((2, 2)), inv2)
    np.testing.assert_allclose(inv2, np.eye(2), atol=0)

    M3 = np.array([[3.0, 1.0, 0.5], [1.0, 4.0, -0.2], [0.5, -0.2, 2.5]])
    inv3 = np.zeros((3, 3))
    invert_3x3(M3, inv3)
    np.testing.assert_allclose(M3 @ inv3, np.eye(3), atol=1e-14)
    invert_3x3(np.zeros((3, 3)), inv3)
    np.testing.assert_allclose(inv3, np.eye(3), atol=0)
    print("[PASS] NumbaLinAlg invert_2x2 / invert_3x3 contract")


def test_readers_roundtrip():
    """Verify SU2Reader, BinaryRestartReader, CSVReader, and MeshReader contracts."""
    from cfd_mesh_tools.readers import (
        BinaryRestartReader,
        CSVReader,
        MeshReader,
        SU2Reader,
    )

    logger = logging.getLogger("cfd_mesh_tools_contract")
    logger.addHandler(logging.NullHandler())

    with tempfile.TemporaryDirectory(prefix="cmt-contract-") as td:
        root = Path(td)

        # 1. SU2Reader + MeshReader on a 2D unit square (2 triangles)
        su2 = root / "sq.su2"
        su2.write_text(
            "NDIME= 2\n"
            "NELEM= 2\n"
            "5 0 1 2 0\n"
            "5 0 2 3 1\n"
            "NPOIN= 4\n"
            "0.0 0.0 0\n"
            "1.0 0.0 1\n"
            "1.0 1.0 2\n"
            "0.0 1.0 3\n"
            "NMARK= 1\n"
            "MARKER_TAG= Wall\n"
            "MARKER_ELEMS= 4\n"
            "3 0 1\n"
            "3 1 2\n"
            "3 2 3\n"
            "3 3 0\n"
        )
        sr = SU2Reader(str(su2), logger, nDim=2)
        coords = sr.extractAllCoords()
        assert coords.shape == (4, 3) and np.all(coords[:, 2] == 0.0)
        elems, ctypes = sr.extractMeshStructure()
        assert len(ctypes) == 2 and "TRI" in elems
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(root)
            mr = MeshReader(str(su2), meshType="SU2", nDim=2)
            assert mr.Mesh.extractAllCoords().shape == (4, 3)
        finally:
            os.chdir(old_cwd)

        # 2. BinaryRestartReader on a synthetic SU2 binary restart file
        # Format: 5 int32 header (magic=535532, nFields, nPoints, nExtra1, nExtra2),
        # then nFields * 33-byte ASCII names, then nPoints * nFields float64 values (Fortran order).
        restart = root / "restart.dat"
        names = ["PointID", "x", "y", "Pressure"]
        vals = np.array([
            [0.0, 0.0, 0.0, 101.0],
            [1.0, 1.0, 0.0, 102.0],
        ], dtype="<f8")  # (nPoints, nFields) -> Fortran reshape (nFields, nPoints) is vals.T
        with open(restart, "wb") as fh:
            fh.write(struct.pack("<5i", 535532, len(names), len(vals), 0, 0))
            for name in names:
                raw = name.encode("ascii")
                fh.write(raw + b"\x00" * (33 - len(raw)))
            fh.write(vals.tobytes())
        br = BinaryRestartReader(str(restart))
        assert br.read("FIELD_NAMES") == names
        data, fields = br.read()
        assert fields == names and data.shape == (4, 2)
        np.testing.assert_allclose(data, vals.T)

        # 3. CSVReader on a small CSV
        csv_path = root / "data.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n")
        cr = CSVReader(str(csv_path), separator=",", type=int)
        arr = cr.read()
        np.testing.assert_array_equal(arr, [[1, 2], [3, 4]])
    print("[PASS] Readers contract (SU2Reader, MeshReader, BinaryRestartReader, CSVReader)")


def main():
    test_public_api_and_signatures()
    test_wls_degree1_and_scale_invariance()
    test_wls_degree2_gradient_hessian_and_fallback()
    test_wls_manifold_pinv_and_limiters()
    test_quadrature_and_element_volumes()
    test_linalg_inverses()
    test_readers_roundtrip()
    print("\nALL cfd-mesh-tools CONTRACT TESTS PASS")


if __name__ == "__main__":
    main()
