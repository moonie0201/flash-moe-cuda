"""TDD RED Phase: Tests for SES expert repacking (Base/Residual split)."""
import numpy as np
import pytest
import tempfile
import os


class TestQuantize4Bit:
    """Re-quantize FP32 matrix back to 4-bit format."""

    def test_quantize_roundtrip(self):
        from ses.src.repack_ses import quantize_4bit
        from ses.src.dequant import dequant_4bit
        # Create a matrix with values in 4-bit range
        np.random.seed(42)
        original = np.random.uniform(0, 15, (4, 16)).astype(np.float32)
        weight, scales, biases = quantize_4bit(original, group_size=8)
        reconstructed = dequant_4bit(weight, scales, biases,
                                     out_dim=4, in_dim=16, group_size=8)
        # Should be close (quantization error expected)
        np.testing.assert_allclose(reconstructed, original, atol=1.5)

    def test_quantize_output_shapes(self):
        from ses.src.repack_ses import quantize_4bit
        matrix = np.random.randn(8, 32).astype(np.float32)
        weight, scales, biases = quantize_4bit(matrix, group_size=8)
        assert weight.dtype == np.uint32
        assert scales.dtype == np.uint16
        assert biases.dtype == np.uint16
        assert weight.shape == (8, 32 // 8)  # packed
        assert scales.shape == (8, 32 // 8)  # num_groups
        assert biases.shape == (8, 32 // 8)


class TestSVDFactorStorage:
    """Store/load SVD factors (U, S, Vt) in compact binary format."""

    def test_save_and_load_factors(self):
        from ses.src.repack_ses import save_svd_factors, load_svd_factors
        np.random.seed(42)
        U = np.random.randn(64, 16).astype(np.float32)
        S = np.random.randn(16).astype(np.float32)
        Vt = np.random.randn(16, 128).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            path = f.name
        try:
            save_svd_factors(path, U, S, Vt)
            U2, S2, Vt2 = load_svd_factors(path, m=64, rank=16, n=128)
            np.testing.assert_allclose(U, U2, atol=1e-6)
            np.testing.assert_allclose(S, S2, atol=1e-6)
            np.testing.assert_allclose(Vt, Vt2, atol=1e-6)
        finally:
            os.unlink(path)

    def test_factor_file_size(self):
        from ses.src.repack_ses import save_svd_factors
        np.random.seed(42)
        U = np.random.randn(1024, 64).astype(np.float32)
        S = np.random.randn(64).astype(np.float32)
        Vt = np.random.randn(64, 4096).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            path = f.name
        try:
            save_svd_factors(path, U, S, Vt)
            size = os.path.getsize(path)
            expected = (1024 * 64 + 64 + 64 * 4096) * 4  # FP32
            assert size == expected
        finally:
            os.unlink(path)


class TestRepackExpertSES:
    """Full repacking: expert binary → Base factors + Residual quantized."""

    def test_repack_single_expert_creates_files(self):
        from ses.src.repack_ses import repack_expert_to_ses
        np.random.seed(42)
        # Create synthetic expert data matching flash-moe format
        gate_w = np.random.randn(1024, 4096).astype(np.float32) * 0.01
        up_w = np.random.randn(1024, 4096).astype(np.float32) * 0.01
        down_w = np.random.randn(4096, 1024).astype(np.float32) * 0.01

        with tempfile.TemporaryDirectory() as tmpdir:
            repack_expert_to_ses(
                gate_w, up_w, down_w,
                output_dir=tmpdir, layer=0, expert_id=0, rank=64
            )
            # Check base factors exist
            assert os.path.exists(os.path.join(tmpdir, 'layer_00_expert_000_base.bin'))
            # Check residual exists
            assert os.path.exists(os.path.join(tmpdir, 'layer_00_expert_000_residual.bin'))

    def test_base_factors_are_smaller(self):
        from ses.src.repack_ses import repack_expert_to_ses
        np.random.seed(42)
        gate_w = np.random.randn(256, 512).astype(np.float32) * 0.01
        up_w = np.random.randn(256, 512).astype(np.float32) * 0.01
        down_w = np.random.randn(512, 256).astype(np.float32) * 0.01

        with tempfile.TemporaryDirectory() as tmpdir:
            repack_expert_to_ses(
                gate_w, up_w, down_w,
                output_dir=tmpdir, layer=0, expert_id=0, rank=32
            )
            base_size = os.path.getsize(
                os.path.join(tmpdir, 'layer_00_expert_000_base.bin'))
            res_size = os.path.getsize(
                os.path.join(tmpdir, 'layer_00_expert_000_residual.bin'))
            total_original = (256*512 + 256*512 + 512*256) * 4
            assert base_size < total_original * 0.4
            assert res_size > 0
