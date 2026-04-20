"""TDD RED Phase: Tests for confidence-aware expert prediction.

Pillar 2: Classify activation landscape and adapt prefetch strategy.
"""
import numpy as np
import pytest


class TestSoftmaxTopK:
    """Basic routing utilities."""

    def test_softmax_sums_to_one(self):
        from ses.src.confidence import softmax
        scores = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        probs = softmax(scores)
        assert probs.shape == (4,)
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_softmax_largest_has_highest_prob(self):
        from ses.src.confidence import softmax
        scores = np.array([1.0, 5.0, 2.0], dtype=np.float32)
        probs = softmax(scores)
        assert np.argmax(probs) == 1

    def test_topk_returns_correct_indices(self):
        from ses.src.confidence import topk
        scores = np.array([0.1, 0.5, 0.2, 0.8, 0.3], dtype=np.float32)
        indices, weights = topk(scores, k=2)
        assert set(indices) == {1, 3}
        assert len(weights) == 2

    def test_topk_weights_normalized(self):
        from ses.src.confidence import topk
        scores = np.array([0.1, 0.5, 0.2, 0.8, 0.3], dtype=np.float32)
        _, weights = topk(scores, k=3)
        assert abs(np.sum(weights) - 1.0) < 1e-5

    def test_topk_with_negative_inputs(self):
        from ses.src.confidence import topk
        scores = np.array([-1.0, -0.5, 0.2, -0.8, 0.3], dtype=np.float32)
        indices, weights = topk(scores, k=2)
        assert set(indices) == {2, 4}
        assert np.all(weights >= 0)
        assert abs(np.sum(weights) - 1.0) < 1e-5


class TestConfidenceClassification:
    """Classify prediction confidence from gate score distribution."""

    def test_peaked_distribution_is_high(self):
        from ses.src.confidence import classify_confidence
        # One dominant expert
        scores = np.zeros(512, dtype=np.float32)
        scores[0] = 10.0
        scores[1] = 9.0
        scores[2] = 8.5
        scores[3] = 8.0
        level, metrics = classify_confidence(scores)
        assert level == 'HIGH'

    def test_flat_distribution_is_low(self):
        from ses.src.confidence import classify_confidence
        # Nearly uniform
        scores = np.random.randn(512).astype(np.float32) * 0.01
        level, metrics = classify_confidence(scores)
        assert level == 'LOW'

    def test_medium_distribution(self):
        from ses.src.confidence import classify_confidence
        # Some concentration but not extreme — top-4 clearly above rest
        scores = np.zeros(512, dtype=np.float32) - 5.0  # baseline low
        scores[:4] = 5.0    # top 4 strongly dominant
        scores[4:20] = 0.0  # middle tier
        level, metrics = classify_confidence(scores)
        assert level in ('MEDIUM', 'HIGH'), \
            f"Expected MEDIUM/HIGH, got {level}, top4_mass={metrics['top4_mass']:.4f}"

    def test_metrics_contain_entropy_and_mass(self):
        from ses.src.confidence import classify_confidence
        scores = np.random.randn(512).astype(np.float32)
        _, metrics = classify_confidence(scores)
        assert 'entropy' in metrics
        assert 'normalized_entropy' in metrics
        assert 'top4_mass' in metrics
        assert 0.0 <= metrics['normalized_entropy'] <= 1.0
        assert 0.0 <= metrics['top4_mass'] <= 1.0


class TestAdaptivePrefetchStrategy:
    """Generate prefetch plan based on confidence level."""

    def test_high_confidence_prefetches_full_only(self):
        from ses.src.confidence import adaptive_prefetch_plan
        plan = adaptive_prefetch_plan('HIGH', top_indices=list(range(32)),
                                      top_scores=np.ones(32, dtype=np.float32))
        assert plan['num_full'] == 4
        assert plan['num_base'] == 0
        assert len(plan['full_ids']) == 4
        assert len(plan['base_ids']) == 0

    def test_medium_confidence_adds_base_prefetch(self):
        from ses.src.confidence import adaptive_prefetch_plan
        plan = adaptive_prefetch_plan('MEDIUM', top_indices=list(range(32)),
                                      top_scores=np.ones(32, dtype=np.float32))
        assert plan['num_full'] == 4
        assert plan['num_base'] == 12
        assert len(plan['full_ids']) == 4
        assert len(plan['base_ids']) == 12

    def test_low_confidence_wide_base_prefetch(self):
        from ses.src.confidence import adaptive_prefetch_plan
        plan = adaptive_prefetch_plan('LOW', top_indices=list(range(32)),
                                      top_scores=np.ones(32, dtype=np.float32))
        assert plan['num_full'] == 4
        assert plan['num_base'] == 28
        assert len(plan['base_ids']) == 28

    def test_full_and_base_ids_dont_overlap(self):
        from ses.src.confidence import adaptive_prefetch_plan
        plan = adaptive_prefetch_plan('LOW', top_indices=list(range(32)),
                                      top_scores=np.ones(32, dtype=np.float32))
        full_set = set(plan['full_ids'])
        base_set = set(plan['base_ids'])
        assert len(full_set & base_set) == 0


class TestSESPipeline:
    """End-to-end SES prediction + prefetch simulation."""

    def test_predict_and_plan_returns_valid_structure(self):
        from ses.src.confidence import ses_predict_and_plan
        gate_scores = np.random.randn(512).astype(np.float32)
        result = ses_predict_and_plan(gate_scores)
        assert 'confidence' in result
        assert 'plan' in result
        assert 'actual_top4' in result
        assert result['confidence'] in ('HIGH', 'MEDIUM', 'LOW')
        assert len(result['actual_top4']['indices']) == 4

    def test_ses_hit_rate_calculation(self):
        from ses.src.confidence import ses_calculate_hit_rate
        predicted_full = [0, 1, 2, 3]
        predicted_base = [4, 5, 6, 7]
        actual = [0, 2, 5, 99]
        hits = ses_calculate_hit_rate(predicted_full, predicted_base, actual)
        assert hits['full_hits'] == 2    # 0, 2
        assert hits['base_hits'] == 1    # 5
        assert hits['misses'] == 1       # 99
