"""TDD RED Phase: Tests for CPU inference kernels.

These serve as ground-truth for C kernel validation.
Python reference implementations tested against numpy.
"""
import numpy as np
import pytest


class TestRMSNorm:
    """RMS normalization: x * w / sqrt(mean(x^2) + eps)."""

    def test_output_shape(self):
        from ses.src.cpu_kernels import rms_norm
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        out = rms_norm(x, w, eps=1e-6)
        assert out.shape == (4,)

    def test_unit_weights_normalizes(self):
        from ses.src.cpu_kernels import rms_norm
        x = np.array([3.0, 4.0], dtype=np.float32)
        w = np.ones(2, dtype=np.float32)
        out = rms_norm(x, w, eps=1e-6)
        # RMS = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        # out = x / RMS = [0.849, 1.131]
        rms = np.sqrt(np.mean(x ** 2) + 1e-6)
        expected = x / rms
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_with_weights(self):
        from ses.src.cpu_kernels import rms_norm
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        w = np.array([2.0, 0.5, 1.0], dtype=np.float32)
        out = rms_norm(x, w, eps=1e-6)
        rms = np.sqrt(np.mean(x ** 2) + 1e-6)
        expected = x / rms * w
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_zero_input(self):
        from ses.src.cpu_kernels import rms_norm
        x = np.zeros(4, dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        out = rms_norm(x, w, eps=1e-6)
        np.testing.assert_allclose(out, np.zeros(4), atol=1e-3)


class TestSwiGLU:
    """SwiGLU activation: silu(gate) * up = gate * sigmoid(gate) * up."""

    def test_output_shape(self):
        from ses.src.cpu_kernels import swiglu
        gate = np.array([1.0, -1.0, 0.0], dtype=np.float32)
        up = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        out = swiglu(gate, up)
        assert out.shape == (3,)

    def test_zero_gate(self):
        from ses.src.cpu_kernels import swiglu
        gate = np.zeros(4, dtype=np.float32)
        up = np.ones(4, dtype=np.float32)
        out = swiglu(gate, up)
        # silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        np.testing.assert_allclose(out, np.zeros(4), atol=1e-6)

    def test_positive_gate(self):
        from ses.src.cpu_kernels import swiglu
        gate = np.array([5.0], dtype=np.float32)
        up = np.array([1.0], dtype=np.float32)
        out = swiglu(gate, up)
        # silu(5) = 5 * sigmoid(5) ≈ 5 * 0.9933 = 4.9665
        expected = 5.0 * (1.0 / (1.0 + np.exp(-5.0)))
        np.testing.assert_allclose(out, [expected], atol=1e-4)

    def test_matches_numpy_reference(self):
        from ses.src.cpu_kernels import swiglu
        np.random.seed(42)
        gate = np.random.randn(128).astype(np.float32)
        up = np.random.randn(128).astype(np.float32)
        out = swiglu(gate, up)
        sigmoid = 1.0 / (1.0 + np.exp(-gate))
        expected = gate * sigmoid * up
        np.testing.assert_allclose(out, expected, atol=1e-5)


class TestMoECombine:
    """Weighted combination of expert outputs + residual."""

    def test_single_expert(self):
        from ses.src.cpu_kernels import moe_combine
        expert_outputs = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        weights = np.array([1.0], dtype=np.float32)
        residual = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        out = moe_combine(expert_outputs, weights, residual)
        expected = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_multiple_experts_weighted(self):
        from ses.src.cpu_kernels import moe_combine
        e1 = np.array([1.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0], dtype=np.float32)
        weights = np.array([0.7, 0.3], dtype=np.float32)
        residual = np.zeros(2, dtype=np.float32)
        out = moe_combine([e1, e2], weights, residual)
        expected = np.array([0.7, 0.3], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_four_experts_with_residual(self):
        from ses.src.cpu_kernels import moe_combine
        np.random.seed(42)
        experts = [np.random.randn(64).astype(np.float32) for _ in range(4)]
        weights = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        residual = np.random.randn(64).astype(np.float32)
        out = moe_combine(experts, weights, residual)
        expected = residual.copy()
        for k in range(4):
            expected += weights[k] * experts[k]
        np.testing.assert_allclose(out, expected, atol=1e-5)


class TestExpertForwardIntegrated:
    """Full expert forward: dequant → gate/up matvec → SwiGLU → down matvec."""

    def test_full_expert_pipeline(self):
        from ses.src.cpu_kernels import expert_forward_pipeline
        np.random.seed(42)
        gate_w = np.random.randn(128, 256).astype(np.float32) * 0.02
        up_w = np.random.randn(128, 256).astype(np.float32) * 0.02
        down_w = np.random.randn(256, 128).astype(np.float32) * 0.02
        x = np.random.randn(256).astype(np.float32)

        out = expert_forward_pipeline(gate_w, up_w, down_w, x)
        assert out.shape == (256,)
        # Should be finite
        assert np.all(np.isfinite(out))

    def test_pipeline_matches_manual(self):
        from ses.src.cpu_kernels import expert_forward_pipeline, swiglu
        np.random.seed(123)
        gate_w = np.random.randn(64, 128).astype(np.float32) * 0.02
        up_w = np.random.randn(64, 128).astype(np.float32) * 0.02
        down_w = np.random.randn(128, 64).astype(np.float32) * 0.02
        x = np.random.randn(128).astype(np.float32)

        out = expert_forward_pipeline(gate_w, up_w, down_w, x)

        # Manual computation
        gate_out = gate_w @ x
        up_out = up_w @ x
        act = swiglu(gate_out, up_out)
        expected = down_w @ act
        np.testing.assert_allclose(out, expected, atol=1e-4)
