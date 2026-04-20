"""SES expert repacking: split experts into Base (SVD factors) + Residual.

Creates per-expert files:
  - layer_XX_expert_YYY_base.bin: SVD factors (U, S, Vt) for 3 projections
  - layer_XX_expert_YYY_residual.bin: quantized residual for 3 projections
"""
import numpy as np
import os
from .svd_decompose import svd_decompose


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Convert FP32 array to BF16 (as uint16)."""
    f32_bytes = values.astype(np.float32).tobytes()
    u32 = np.frombuffer(f32_bytes, dtype=np.uint32)
    return (u32 >> 16).astype(np.uint16)


def quantize_4bit(matrix: np.ndarray, group_size: int = 8):
    """Quantize FP32 matrix to 4-bit affine format.

    Args:
        matrix: [out_dim, in_dim] FP32
        group_size: quantization group size

    Returns:
        weight: [out_dim, in_dim//8] uint32 packed nibbles
        scales: [out_dim, num_groups] bf16 as uint16
        biases: [out_dim, num_groups] bf16 as uint16
    """
    out_dim, in_dim = matrix.shape
    assert group_size % 8 == 0, f"group_size must be multiple of 8, got {group_size}"
    num_groups = in_dim // group_size

    weight_packed = np.zeros((out_dim, in_dim // 8), dtype=np.uint32)
    scales = np.zeros((out_dim, num_groups), dtype=np.uint16)
    biases = np.zeros((out_dim, num_groups), dtype=np.uint16)

    for row in range(out_dim):
        for g in range(num_groups):
            col_start = g * group_size
            col_end = col_start + group_size
            group_vals = matrix[row, col_start:col_end]

            vmin = float(np.min(group_vals))
            vmax = float(np.max(group_vals))

            if vmax - vmin < 1e-10:
                scale = 1.0
                bias = vmin
            else:
                scale = (vmax - vmin) / 15.0
                bias = vmin

            scales[row, g] = f32_to_bf16(np.float32(scale))[0]
            biases[row, g] = f32_to_bf16(np.float32(bias))[0]

            # Quantize to 4-bit nibbles
            for i in range(group_size):
                col = col_start + i
                val = group_vals[i]
                nibble = int(np.clip(np.round((val - bias) / (scale + 1e-10)), 0, 15))
                pack_idx = col // 8
                nibble_pos = col % 8
                weight_packed[row, pack_idx] |= np.uint32(nibble << (nibble_pos * 4))

    return weight_packed, scales, biases


def save_svd_factors(path: str, U: np.ndarray, S: np.ndarray, Vt: np.ndarray):
    """Save SVD factors (U, S, Vt) as contiguous FP32 binary."""
    with open(path, 'wb') as f:
        f.write(U.astype(np.float32).tobytes())
        f.write(S.astype(np.float32).tobytes())
        f.write(Vt.astype(np.float32).tobytes())


def load_svd_factors(path: str, m: int, rank: int, n: int):
    """Load SVD factors from binary file."""
    with open(path, 'rb') as f:
        U = np.frombuffer(f.read(m * rank * 4), dtype=np.float32).reshape(m, rank)
        S = np.frombuffer(f.read(rank * 4), dtype=np.float32).copy()
        Vt = np.frombuffer(f.read(rank * n * 4), dtype=np.float32).reshape(rank, n)
    return U, S, Vt


def repack_expert_to_ses(gate_w: np.ndarray, up_w: np.ndarray,
                         down_w: np.ndarray, output_dir: str,
                         layer: int, expert_id: int, rank: int = 64):
    """Decompose and save one expert as Base factors + Residual.

    Creates:
      - {output_dir}/layer_XX_expert_YYY_base.bin: concatenated SVD factors
      - {output_dir}/layer_XX_expert_YYY_residual.bin: concatenated residual matrices
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"layer_{layer:02d}_expert_{expert_id:03d}"

    base_path = os.path.join(output_dir, f"{prefix}_base.bin")
    res_path = os.path.join(output_dir, f"{prefix}_residual.bin")

    with open(base_path, 'wb') as bf, open(res_path, 'wb') as rf:
        for W in [gate_w, up_w, down_w]:
            U, S_vals, Vt = np.linalg.svd(W, full_matrices=False)

            # Base: rank-r factors
            bf.write(U[:, :rank].astype(np.float32).tobytes())
            bf.write(S_vals[:rank].astype(np.float32).tobytes())
            bf.write(Vt[:rank, :].astype(np.float32).tobytes())

            # Residual: W - U_r @ diag(S_r) @ Vt_r, re-quantized to 4-bit
            base_approx = U[:, :rank] @ np.diag(S_vals[:rank]) @ Vt[:rank, :]
            residual = W - base_approx
            res_w, res_s, res_b = quantize_4bit(residual, group_size=64)
            rf.write(res_w.tobytes())
            rf.write(res_s.tobytes())
            rf.write(res_b.tobytes())
