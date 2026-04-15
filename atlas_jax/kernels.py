"""Optional GPU kernel dispatch.

Centralizes the `try: import triton/pallas kernel; except ImportError: fall back`
logic that used to live at the top of `model.py` and `memory_layer.py`. Consumers
import from here once and branch on the `HAS_*` flags when they need to know
whether a faster path is available.

Three kernel families:
    * fused_chunk_scan — FlashATLAS end-to-end chunk kernel (Triton, with Pallas
      fallback). `HAS_FUSED_CHUNK` is True iff one of the backends imported
      cleanly *and* its runtime availability probe succeeded.
    * triton_polar_express / triton_polar_express_ste — fused Newton-Schulz
      iteration in a single Triton kernel. No fallback here; callers use
      `HAS_TRITON_PE` to decide whether to prefer it over the Python version.
    * triton_linear_scan — fused linear-recurrence kernel. Callers that don't
      need the non-Triton fallback (associative_scan) should gate on
      `HAS_TRITON_SCAN`.

The underlying modules (`atlas_jax.fused_chunk`, `atlas_jax.pallas_fused`,
`atlas_jax.triton_pe`, `atlas_jax.triton_scan`) remain importable directly for
tests and micro-benchmarks that target a specific backend.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# FlashATLAS fused chunk scan — Triton preferred, Pallas fallback
# ---------------------------------------------------------------------------

fused_chunk_scan = None
HAS_FUSED_CHUNK = False

try:
    from atlas_jax.fused_chunk import (
        fused_chunk_scan as _fcs_triton,
        fused_chunk_available as _fcs_triton_available,
    )
    if _fcs_triton_available():
        fused_chunk_scan = _fcs_triton
        HAS_FUSED_CHUNK = True
except ImportError:
    pass

if not HAS_FUSED_CHUNK:
    try:
        from atlas_jax.pallas_fused import (
            fused_chunk_scan as _fcs_pallas,
            pallas_available as _pallas_available,
        )
        if _pallas_available():
            fused_chunk_scan = _fcs_pallas
            HAS_FUSED_CHUNK = True
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Fused Triton Polar Express (single kernel for all NS iterations)
# ---------------------------------------------------------------------------

triton_polar_express = None
triton_polar_express_ste = None
HAS_TRITON_PE = False

try:
    from atlas_jax.triton_pe import (
        triton_polar_express as _tpe,
        triton_polar_express_ste as _tpe_ste,
    )
    triton_polar_express = _tpe
    triton_polar_express_ste = _tpe_ste
    HAS_TRITON_PE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Fused Triton linear scan (linear recurrence)
# ---------------------------------------------------------------------------

triton_linear_scan = None
HAS_TRITON_SCAN = False

try:
    from atlas_jax.triton_scan import triton_linear_scan as _tls
    triton_linear_scan = _tls
    HAS_TRITON_SCAN = True
except ImportError:
    pass


__all__ = [
    "fused_chunk_scan",
    "HAS_FUSED_CHUNK",
    "triton_polar_express",
    "triton_polar_express_ste",
    "HAS_TRITON_PE",
    "triton_linear_scan",
    "HAS_TRITON_SCAN",
]
