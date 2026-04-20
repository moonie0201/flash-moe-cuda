"""SVD-based expert decomposition for Speculative Expert Streaming (SES).

Decomposes expert weight matrices into Base (low-rank) + Residual,
enabling hierarchical loading and speculative computation.
"""
import numpy as np


def svd_decompose(matrix: np.ndarray, rank: int):
    """Decompose matrix into rank-r Base + Residual via SVD.

    Returns:
        base: rank-r approximation
        residual: matrix - base
        metrics: dict with energy_ratio, relative_error, rank
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    base = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    residual = matrix - base

    total_energy = np.sum(S ** 2)
    captured_energy = np.sum(S[:rank] ** 2)
    energy_ratio = float(captured_energy / total_energy) if total_energy > 0 else 1.0

    rel_error = float(np.linalg.norm(residual, 'fro') / (np.linalg.norm(matrix, 'fro') + 1e-10))

    return base, residual, {
        'rank': rank,
        'energy_ratio': energy_ratio,
        'relative_error': rel_error,
    }


def decompose_expert_matrices(expert: dict, rank: int) -> dict:
    """Decompose all projections of an expert into Base + Residual.

    Args:
        expert: {'gate_proj': ndarray, 'up_proj': ndarray, 'down_proj': ndarray}
        rank: SVD rank for approximation
    """
    result = {}
    for name, matrix in expert.items():
        base, residual, metrics = svd_decompose(matrix, rank)
        result[name] = {
            'base': base,
            'residual': residual,
            'metrics': metrics,
        }
    return result


def base_size_bytes(decomposed: dict, rank: int) -> int:
    """Estimate storage size for Base factors (U, S, Vt) in FP32."""
    total = 0
    for proj in decomposed.values():
        base = proj['base']
        m, n = base.shape
        # Stored as U[m, rank] + S[rank] + Vt[rank, n] in FP32
        total += (m * rank + rank + rank * n) * 4
    return total


def _swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """SwiGLU activation: silu(gate) * up."""
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(gate, -88, 88)))
    return gate * sigmoid * up


def expert_forward(gate_w: np.ndarray, up_w: np.ndarray, down_w: np.ndarray,
                   x: np.ndarray) -> np.ndarray:
    """Standard expert forward: down(SwiGLU(gate(x), up(x)))."""
    gate_out = gate_w @ x
    up_out = up_w @ x
    act = _swiglu(gate_out, up_out)
    return down_w @ act


def _low_rank_matvec(matrix: np.ndarray, x: np.ndarray, rank: int) -> np.ndarray:
    """Low-rank matvec from full matrix (reference only, re-runs SVD)."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    tmp = Vt[:rank, :] @ x         # [rank]
    tmp = tmp * S[:rank]            # [rank]
    return U[:, :rank] @ tmp        # [out_dim]


def precomputed_low_rank_matvec(U: np.ndarray, S: np.ndarray,
                                Vt: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Efficient low-rank matvec from pre-computed SVD factors.

    Args:
        U: [m, rank], S: [rank], Vt: [rank, n], x: [n]
    Returns:
        [m] vector
    """
    tmp = Vt @ x          # [rank]
    tmp = tmp * S          # [rank]
    return U @ tmp         # [m]


def base_expert_forward(gate_w: np.ndarray, up_w: np.ndarray,
                        down_w: np.ndarray, x: np.ndarray,
                        rank: int) -> np.ndarray:
    """Expert forward using only Base (reference, re-computes SVD)."""
    gate_out = _low_rank_matvec(gate_w, x, rank)
    up_out = _low_rank_matvec(up_w, x, rank)
    act = _swiglu(gate_out, up_out)
    return _low_rank_matvec(down_w, act, rank)


def base_expert_forward_precomputed(factors: dict, x: np.ndarray) -> np.ndarray:
    """Expert forward using pre-computed SVD factors (production path).

    Args:
        factors: dict with keys 'gate_proj', 'up_proj', 'down_proj',
                 each containing {'U': ndarray, 'S': ndarray, 'Vt': ndarray}
        x: input vector [hidden_dim]
    """
    gate_out = precomputed_low_rank_matvec(
        factors['gate_proj']['U'], factors['gate_proj']['S'],
        factors['gate_proj']['Vt'], x)
    up_out = precomputed_low_rank_matvec(
        factors['up_proj']['U'], factors['up_proj']['S'],
        factors['up_proj']['Vt'], x)
    act = _swiglu(gate_out, up_out)
    return precomputed_low_rank_matvec(
        factors['down_proj']['U'], factors['down_proj']['S'],
        factors['down_proj']['Vt'], act)


def speculative_expert_forward(gate_w: np.ndarray, up_w: np.ndarray,
                               down_w: np.ndarray, x: np.ndarray,
                               rank: int,
                               residual_available: bool = True) -> np.ndarray:
    """Speculative expert forward: Base first, refine with Residual if available.

    When residual_available=True: equivalent to full forward (Base + Residual).
    When residual_available=False: Base-only (graceful degradation).
    """
    if not residual_available:
        return base_expert_forward(gate_w, up_w, down_w, x, rank)

    # Full forward (Base + Residual = original)
    return expert_forward(gate_w, up_w, down_w, x)
