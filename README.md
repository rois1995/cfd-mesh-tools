# cfd-mesh-tools

Shared, versioned Python library providing CFD mesh/solution readers, Numba-accelerated WLS gradient/Hessian reconstruction, slope limiters, small-matrix linear algebra, and SU2-compatible FEM quadrature rules for `LaplaceSolver_V2` and `MeshAdaptation`.

## Installation

Install a pinned release directly from GitHub:

```bash
pip install "cfd-mesh-tools @ git+https://github.com/rois1995/cfd-mesh-tools.git@v0.1.0"
```

Or pin in `requirements.txt` / `pyproject.toml`:

```text
cfd-mesh-tools @ git+https://github.com/rois1995/cfd-mesh-tools.git@v0.1.0
```

Editable local install (during development):

```bash
pip install --no-deps -e .
```

Core installation requires only `numpy`, `numba`, and `scipy` (suitable for HPC compute nodes). Optional reader extras:

- `cfd-mesh-tools[cgns]` — `h5py` for `CGNSReader` / `MeshWriter`
- `cfd-mesh-tools[vtu]` — `vtk` for `VTUReader`
- `cfd-mesh-tools[csv]` — `polars` for fast `CSVReader`
- `cfd-mesh-tools[readers]` — all optional reader dependencies

## Package Layout

- `cfd_mesh_tools.readers` — Intact mesh & flow readers (`MeshReader`, `SU2Reader`, `CGNSReader`, `VTUReader`, `BinaryRestartReader`, `CSVReader`, `MeshWriter`, `ElementsUtilities`, `Utils`)
- `cfd_mesh_tools.gradients` — WLS coefficient builders (`compute_wls_coefficients_optimized`, `compute_wls_coefficients_pinv`, `compute_wls_coefficients_deg2`), difference-form consumers (`ComputeWLSGradients_Numba`, `ComputeWLSHessians_Numba`), and slope limiters (`ComputeBJLimiter_Numba`, `ComputeVenkatLimiter_Numba`, `ComputeNishikawaRLimiter_Numba`, `ApplyLimiterIfRequested`)
- `cfd_mesh_tools.quadrature` — Reference-element quadrature rules (`_quadRule`, `_dQuad`, `_dHex`, `_dPrismSU2`, `_dPyramSU2`, `_invDet`, `elementVolumes`, element type codes)
- `cfd_mesh_tools.linalg` — `NumbaLinAlg` (`invert_2x2`, `invert_3x3`, `eig2`, `eig3`, etc.)
- `cfd_mesh_tools.assembly` — `AssemblyNumba_Fun` (`elementVolumesNumba`, `colorElements`, Numba FEM assembly)

## Contract Tests

Run the interface and numerical contract test suite before tagging any new release:

```bash
python3 tests/test_contract.py
```
