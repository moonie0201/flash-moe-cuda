"""Confidence-aware expert prediction for Speculative Expert Streaming.

Pillar 2: Analyze activation landscape, classify confidence,
and generate adaptive prefetch strategies.
"""
import numpy as np


def softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)


def topk(probs: np.ndarray, k: int):
    """Return top-k indices and their normalized weights from probabilities.

    Args:
        probs: non-negative probability/score array (e.g. post-softmax)
        k: number of top entries to return

    Returns:
        indices: array of k indices (highest first)
        weights: array of k normalized weights (sum to 1)
    """
    probs = np.maximum(probs, 0.0)  # guard against negative inputs
    indices = np.argsort(probs)[-k:][::-1]
    raw_weights = probs[indices]
    weights = raw_weights / (np.sum(raw_weights) + 1e-10)
    return indices, weights


def classify_confidence(scores: np.ndarray):
    """Classify prediction confidence from raw gate scores.

    Args:
        scores: raw gate scores [num_experts] (pre-softmax)

    Returns:
        level: 'HIGH', 'MEDIUM', or 'LOW'
        metrics: dict with entropy, normalized_entropy, top4_mass
    """
    probs = softmax(scores)
    num_experts = len(probs)

    # Entropy
    safe_probs = np.clip(probs, 1e-10, 1.0)
    entropy = float(-np.sum(safe_probs * np.log(safe_probs)))
    max_entropy = np.log(num_experts)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Top-4 probability mass
    sorted_probs = np.sort(probs)[::-1]
    top4_mass = float(np.sum(sorted_probs[:4]))

    metrics = {
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'top4_mass': top4_mass,
    }

    if top4_mass > 0.8 and normalized_entropy < 0.3:
        return 'HIGH', metrics
    elif top4_mass > 0.5 or normalized_entropy < 0.7:
        return 'MEDIUM', metrics
    else:
        return 'LOW', metrics


def adaptive_prefetch_plan(confidence: str, top_indices: list,
                           top_scores: np.ndarray) -> dict:
    """Generate prefetch plan based on confidence level.

    Args:
        confidence: 'HIGH', 'MEDIUM', or 'LOW'
        top_indices: sorted expert indices (highest score first), at least 32
        top_scores: corresponding scores

    Returns:
        dict with full_ids, base_ids, num_full, num_base
    """
    if confidence == 'HIGH':
        num_full = 4
        num_base = 0
    elif confidence == 'MEDIUM':
        num_full = 4
        num_base = 12
    else:  # LOW
        num_full = 4
        num_base = 28

    full_ids = top_indices[:num_full]
    base_ids = top_indices[num_full:num_full + num_base]

    return {
        'full_ids': list(full_ids),
        'base_ids': list(base_ids),
        'num_full': num_full,
        'num_base': num_base,
    }


def ses_predict_and_plan(gate_scores: np.ndarray) -> dict:
    """Full SES prediction pipeline: scores → confidence → prefetch plan.

    Args:
        gate_scores: raw gate scores [num_experts]

    Returns:
        dict with confidence, plan, actual_top4
    """
    probs = softmax(gate_scores)
    confidence, metrics = classify_confidence(gate_scores)

    # Get top-32 for prefetch candidates
    top32_indices = np.argsort(probs)[-32:][::-1].tolist()
    top32_scores = probs[top32_indices]

    plan = adaptive_prefetch_plan(confidence, top32_indices, top32_scores)

    # Actual top-4 routing
    actual_indices = np.argsort(probs)[-4:][::-1]
    actual_weights = probs[actual_indices]
    actual_weights = actual_weights / (np.sum(actual_weights) + 1e-10)

    return {
        'confidence': confidence,
        'metrics': metrics,
        'plan': plan,
        'actual_top4': {
            'indices': actual_indices.tolist(),
            'weights': actual_weights.tolist(),
        },
    }


def ses_calculate_hit_rate(predicted_full: list, predicted_base: list,
                           actual: list) -> dict:
    """Calculate hit rate for SES prediction.

    Returns:
        dict with full_hits, base_hits, misses
    """
    full_set = set(predicted_full)
    base_set = set(predicted_base)

    full_hits = 0
    base_hits = 0
    misses = 0

    for eid in actual:
        if eid in full_set:
            full_hits += 1
        elif eid in base_set:
            base_hits += 1
        else:
            misses += 1

    return {
        'full_hits': full_hits,
        'base_hits': base_hits,
        'misses': misses,
    }
