"""TDD RED Phase: Tests for C kernel implementations via ctypes.

Validates C code against Python reference implementations.
"""
import numpy as np
import pytest
import ctypes
import os
import subprocess

# Path to compiled shared library
LIB_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'libses_kernels.so')


def _build_if_needed():
    """Build the C library if it doesn't exist."""
    if not os.path.exists(LIB_PATH):
        src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
        result = subprocess.run(
            ['make', '-f', 'Makefile.linux', 'libses_kernels.so'],
            cwd=src_dir, capture_output=True, text=True
        )
        if result.returncode != 0:
            pytest.skip(f"C library build failed: {result.stderr}")


@pytest.fixture(autouse=True)
def ensure_lib():
    _build_if_needed()
    if not os.path.exists(LIB_PATH):
        pytest.skip("C library not available")


def load_lib():
    return ctypes.CDLL(LIB_PATH)


class TestCRMSNorm:
    """C rms_norm matches Python reference."""

    def test_matches_python(self):
        from ses.src.cpu_kernels import rms_norm as py_rms_norm
        lib = load_lib()

        np.random.seed(42)
        x = np.random.randn(256).astype(np.float32)
        w = np.random.randn(256).astype(np.float32)
        out_c = np.zeros(256, dtype=np.float32)

        lib.c_rms_norm(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(256),
            ctypes.c_float(1e-6)
        )

        out_py = py_rms_norm(x, w, eps=1e-6)
        np.testing.assert_allclose(out_c, out_py, atol=1e-5)


class TestCSwiGLU:
    """C swiglu matches Python reference."""

    def test_matches_python(self):
        from ses.src.cpu_kernels import swiglu as py_swiglu
        lib = load_lib()

        np.random.seed(42)
        gate = np.random.randn(128).astype(np.float32)
        up = np.random.randn(128).astype(np.float32)
        out_c = np.zeros(128, dtype=np.float32)

        lib.c_swiglu(
            gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            up.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(128)
        )

        out_py = py_swiglu(gate, up)
        np.testing.assert_allclose(out_c, out_py, atol=1e-5)


class TestCSwiGLUEdgeCases:
    """C swiglu with extreme inputs (clipping boundary)."""

    def test_extreme_positive_matches_python(self):
        from ses.src.cpu_kernels import swiglu as py_swiglu
        lib = load_lib()
        gate = np.array([100.0, -100.0, 88.0, -88.0, 0.0], dtype=np.float32)
        up = np.ones(5, dtype=np.float32)
        out_c = np.zeros(5, dtype=np.float32)
        lib.c_swiglu(
            gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            up.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(5)
        )
        out_py = py_swiglu(gate, up)
        np.testing.assert_allclose(out_c, out_py, atol=1e-3)


class TestCMoECombine:
    """C moe_combine matches Python reference."""

    def test_matches_python(self):
        from ses.src.cpu_kernels import moe_combine as py_moe_combine
        lib = load_lib()

        np.random.seed(42)
        K = 4
        dim = 64
        experts = [np.random.randn(dim).astype(np.float32) for _ in range(K)]
        weights = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        residual = np.random.randn(dim).astype(np.float32)
        out_c = residual.copy()

        # Pack experts into contiguous array for C
        experts_flat = np.concatenate(experts)

        lib.c_moe_combine(
            experts_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(K),
            ctypes.c_int(dim)
        )

        out_py = py_moe_combine(experts, weights, residual)
        np.testing.assert_allclose(out_c, out_py, atol=1e-5)
