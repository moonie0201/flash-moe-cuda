"""TDD RED Phase: Tests for 4-bit dequantization.

Tests written BEFORE implementation. Must FAIL first.
"""
import numpy as np
import pytest
import struct


class TestBF16ToFP32:
    """BF16 (bfloat16) to FP32 conversion."""

    def test_bf16_zero(self):
        from ses.src.dequant import bf16_to_f32
        assert bf16_to_f32(0x0000) == 0.0

    def test_bf16_one(self):
        from ses.src.dequant import bf16_to_f32
        # BF16 for 1.0: sign=0, exp=127 (0x3F), mantissa=0 → 0x3F80
        assert bf16_to_f32(0x3F80) == pytest.approx(1.0)

    def test_bf16_negative(self):
        from ses.src.dequant import bf16_to_f32
        # BF16 for -1.0: 0xBF80
        assert bf16_to_f32(0xBF80) == pytest.approx(-1.0)

    def test_bf16_small_value(self):
        from ses.src.dequant import bf16_to_f32
        # BF16 for 0.5: 0x3F00
        assert bf16_to_f32(0x3F00) == pytest.approx(0.5)

    def test_bf16_array(self):
        from ses.src.dequant import bf16_array_to_f32
        raw = np.array([0x3F80, 0xBF80, 0x0000], dtype=np.uint16)
        result = bf16_array_to_f32(raw)
        np.testing.assert_allclose(result, [1.0, -1.0, 0.0], atol=1e-6)


class TestPack4Bit:
    """4-bit packing/unpacking utilities."""

    def test_unpack_single_uint32(self):
        from ses.src.dequant import unpack_nibbles
        # 0x76543210 → nibbles [0,1,2,3,4,5,6,7]
        packed = np.uint32(0x76543210)
        nibbles = unpack_nibbles(packed)
        np.testing.assert_array_equal(nibbles, [0, 1, 2, 3, 4, 5, 6, 7])

    def test_unpack_all_zeros(self):
        from ses.src.dequant import unpack_nibbles
        nibbles = unpack_nibbles(np.uint32(0x00000000))
        np.testing.assert_array_equal(nibbles, [0, 0, 0, 0, 0, 0, 0, 0])

    def test_unpack_all_fifteen(self):
        from ses.src.dequant import unpack_nibbles
        nibbles = unpack_nibbles(np.uint32(0xFFFFFFFF))
        np.testing.assert_array_equal(nibbles, [15, 15, 15, 15, 15, 15, 15, 15])


class TestDequant4Bit:
    """Full 4-bit dequantized matrix reconstruction."""

    def test_small_matrix_shape(self):
        from ses.src.dequant import dequant_4bit
        # 2 rows, 8 columns (1 uint32 per row), group_size=8
        weight = np.array([0x76543210, 0xFEDCBA98], dtype=np.uint32)
        scales = np.array([0x3F80, 0x4000], dtype=np.uint16)  # [1.0, 2.0]
        biases = np.array([0x0000, 0x0000], dtype=np.uint16)  # [0.0, 0.0]
        result = dequant_4bit(weight, scales, biases, out_dim=2, in_dim=8, group_size=8)
        assert result.shape == (2, 8)

    def test_dequant_identity_scale(self):
        from ses.src.dequant import dequant_4bit
        # scale=1.0, bias=0.0 → output = nibble values directly
        weight = np.array([0x76543210], dtype=np.uint32)  # 1 row, 8 cols
        scales = np.array([0x3F80], dtype=np.uint16)  # 1.0
        biases = np.array([0x0000], dtype=np.uint16)  # 0.0
        result = dequant_4bit(weight, scales, biases, out_dim=1, in_dim=8, group_size=8)
        expected = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_dequant_with_bias(self):
        from ses.src.dequant import dequant_4bit
        # scale=1.0, bias=10.0 → output = nibble + 10
        weight = np.array([0x00000000], dtype=np.uint32)  # all zeros
        scales = np.array([0x3F80], dtype=np.uint16)  # 1.0
        biases = np.array([0x4120], dtype=np.uint16)  # 10.0 in bf16
        result = dequant_4bit(weight, scales, biases, out_dim=1, in_dim=8, group_size=8)
        # 0 * 1.0 + 10.0 = 10.0 for all elements
        expected = np.full((1, 8), 10.0, dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=0.1)

    def test_dequant_multi_group(self):
        from ses.src.dequant import dequant_4bit
        # 1 row, 16 cols = 2 uint32, 2 groups of 8
        weight = np.array([0x00000000, 0x11111111], dtype=np.uint32)
        # group 0: scale=1.0, bias=0.0 → all 0s
        # group 1: scale=2.0, bias=1.0 → 1*2+1 = 3 for all
        scales = np.array([0x3F80, 0x4000], dtype=np.uint16)  # [1.0, 2.0]
        biases = np.array([0x0000, 0x3F80], dtype=np.uint16)  # [0.0, 1.0]
        result = dequant_4bit(weight, scales, biases, out_dim=1, in_dim=16, group_size=8)
        assert result.shape == (1, 16)
        # First 8: nibble=0, 0*1.0+0.0=0.0
        np.testing.assert_allclose(result[0, :8], 0.0, atol=1e-5)
        # Last 8: nibble=1, 1*2.0+1.0=3.0
        np.testing.assert_allclose(result[0, 8:16], 3.0, atol=0.1)


class TestDequantMatvec:
    """4-bit dequantized matrix-vector multiply."""

    def test_matvec_identity_like(self):
        from ses.src.dequant import dequant_matvec
        # Simple: 2x8 matrix @ 8-vec
        weight = np.array([0x76543210, 0x00000000], dtype=np.uint32)
        scales = np.array([0x3F80, 0x3F80], dtype=np.uint16)
        biases = np.array([0x0000, 0x0000], dtype=np.uint16)
        x = np.ones(8, dtype=np.float32)
        result = dequant_matvec(weight, scales, biases, x,
                                out_dim=2, in_dim=8, group_size=8)
        assert result.shape == (2,)
        # Row 0: [0,1,2,3,4,5,6,7] @ [1,1,...,1] = 28
        assert result[0] == pytest.approx(28.0, abs=0.1)
        # Row 1: [0,0,0,0,0,0,0,0] @ [1,1,...,1] = 0
        assert result[1] == pytest.approx(0.0, abs=0.1)
