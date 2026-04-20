"""4-bit dequantization utilities for MoE expert weights.

Implements the MLX affine 4-bit quantization format used by flash-moe:
- weight: uint32 packed (8 nibbles per word, LSB first)
- scales: bf16 per group
- biases: bf16 per group
- group_size: typically 64 (8 for tests)
"""
import numpy as np


def bf16_to_f32(raw: int) -> float:
    """Convert a single BF16 value (as uint16) to FP32."""
    f32_bits = raw << 16
    return np.frombuffer(np.uint32(f32_bits).tobytes(), dtype=np.float32)[0]


def bf16_array_to_f32(raw: np.ndarray) -> np.ndarray:
    """Convert array of BF16 uint16 values to FP32."""
    f32_bits = raw.astype(np.uint32) << 16
    return np.frombuffer(f32_bits.tobytes(), dtype=np.float32).copy()


def unpack_nibbles(packed: np.uint32) -> np.ndarray:
    """Unpack a uint32 into 8 4-bit nibbles (LSB first)."""
    val = int(packed)
    return np.array([(val >> (i * 4)) & 0xF for i in range(8)], dtype=np.float32)


def dequant_4bit(weight: np.ndarray, scales: np.ndarray, biases: np.ndarray,
                 out_dim: int, in_dim: int, group_size: int = 64) -> np.ndarray:
    """Dequantize 4-bit packed weight matrix to FP32.

    Args:
        weight: [out_dim, in_dim/8] uint32 packed nibbles
        scales: [out_dim, in_dim/group_size] bf16 as uint16
        biases: [out_dim, in_dim/group_size] bf16 as uint16
        out_dim: number of output rows
        in_dim: number of input columns (unpacked)
        group_size: quantization group size
    """
    packed_cols = in_dim // 8
    num_groups = in_dim // group_size

    scales_f32 = bf16_array_to_f32(scales).reshape(out_dim, num_groups)
    biases_f32 = bf16_array_to_f32(biases).reshape(out_dim, num_groups)
    weight_flat = weight.reshape(out_dim, packed_cols)

    result = np.zeros((out_dim, in_dim), dtype=np.float32)

    for row in range(out_dim):
        for p in range(packed_cols):
            nibbles = unpack_nibbles(weight_flat[row, p])
            base_col = p * 8
            for n in range(8):
                col = base_col + n
                group = col // group_size
                s = scales_f32[row, group]
                b = biases_f32[row, group]
                result[row, col] = nibbles[n] * s + b

    return result


def dequant_matvec(weight: np.ndarray, scales: np.ndarray, biases: np.ndarray,
                   x: np.ndarray, out_dim: int, in_dim: int,
                   group_size: int = 64) -> np.ndarray:
    """4-bit dequantized matrix-vector multiply: y = dequant(W) @ x."""
    matrix = dequant_4bit(weight, scales, biases, out_dim, in_dim, group_size)
    return matrix @ x
