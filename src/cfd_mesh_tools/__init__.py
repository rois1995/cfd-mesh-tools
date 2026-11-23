"""Shared CFD mesh/solution readers, WLS gradient/Hessian kernels, and FEM quadrature."""

__version__ = "0.1.0"

from .gradients import (
    ApplyLimiterIfRequested,
    ComputeBJLimiter_Numba,
    ComputeNishikawaRLimiter_Numba,
    ComputeVenkatLimiter_Numba,
    ComputeWLSGradients_Numba,
    ComputeWLSHessians_Numba,
    compute_wls_coefficients_deg2,
    compute_wls_coefficients_optimized,
    compute_wls_coefficients_pinv,
)
from .quadrature import (
    HEX,
    LINE,
    PRISM,
    PYRA,
    QUAD,
    TET,
    TRI,
    _quadRule,
    elementVolumes,
)

__all__ = [
    "__version__",
    "compute_wls_coefficients_optimized",
    "compute_wls_coefficients_pinv",
    "compute_wls_coefficients_deg2",
    "ComputeWLSGradients_Numba",
    "ComputeWLSHessians_Numba",
    "ComputeBJLimiter_Numba",
    "ComputeVenkatLimiter_Numba",
    "ComputeNishikawaRLimiter_Numba",
    "ApplyLimiterIfRequested",
    "_quadRule",
    "elementVolumes",
    "LINE",
    "TRI",
    "QUAD",
    "TET",
    "HEX",
    "PRISM",
    "PYRA",
]
