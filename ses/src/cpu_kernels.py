"""CPU kernel reference implementations (Python/NumPy).

These serve as ground truth for C kernel validation.
Each function mirrors the planned C implementation.
"""
import numpy as np


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMS normalization: x * weight / sqrt(mean(x^2) + eps)."""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return x / rms * weight


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """SwiGLU activation: silu(gate) * up = clipped_gate * sigmoid(clipped_gate) * up."""
    g = np.clip(gate, -88, 88)
    sigmoid = 1.0 / (1.0 + np.exp(-g))
    return g * sigmoid * up


def moe_combine(expert_outputs: list, weights: np.ndarray,
                residual: np.ndarray) -> np.ndarray:
    """Weighted sum of expert outputs + residual connection."""
    result = residual.copy()
    for k, expert_out in enumerate(expert_outputs):
        result += weights[k] * expert_out
    return result


def expert_forward_pipeline(gate_w: np.ndarray, up_w: np.ndarray,
                            down_w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Full expert forward: gate/up projections → SwiGLU → down projection."""
    gate_out = gate_w @ x
    up_out = up_w @ x
    act = swiglu(gate_out, up_out)
    return down_w @ act
