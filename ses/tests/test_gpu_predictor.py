"""Tests for GPU predictor and expert cache."""
import numpy as np
import pytest


class TestGPUPredictor:
    """GPU expert prediction."""

    def _make_predictor(self):
        from ses.src.gpu_predictor import GPUPredictor
        np.random.seed(42)
        gate_weights = {
            i: np.random.randn(256, 2048).astype(np.float32) * 0.1
            for i in range(5)
        }
        return GPUPredictor(gate_weights, top_k=8)

    def test_predict_returns_valid_structure(self):
        pred = self._make_predictor()
        x = np.random.randn(2048).astype(np.float32)
        result = pred.predict(x, layer=0)
        assert 'expert_ids' in result
        assert 'scores' in result
        assert 'confidence' in result
        assert len(result['expert_ids']) == 8
        assert result['confidence'] in ('HIGH', 'MEDIUM', 'LOW')

    def test_predict_expert_ids_in_range(self):
        pred = self._make_predictor()
        x = np.random.randn(2048).astype(np.float32)
        result = pred.predict(x, layer=0)
        assert all(0 <= eid < 256 for eid in result['expert_ids'])

    def test_predict_scores_sum_reasonable(self):
        pred = self._make_predictor()
        x = np.random.randn(2048).astype(np.float32)
        result = pred.predict(x, layer=0)
        assert all(s > 0 for s in result['scores'])

    def test_predict_different_inputs_different_experts(self):
        pred = self._make_predictor()
        x1 = np.random.randn(2048).astype(np.float32)
        x2 = np.random.randn(2048).astype(np.float32) * 5
        r1 = pred.predict(x1, layer=0)
        r2 = pred.predict(x2, layer=0)
        # Not guaranteed to be different but very likely
        assert not np.array_equal(r1['expert_ids'], r2['expert_ids'])

    def test_predict_batch(self):
        pred = self._make_predictor()
        x = np.random.randn(2048).astype(np.float32)
        results = pred.predict_batch(x, layers=[0, 1, 2])
        assert len(results) == 3
        for r in results:
            assert 'expert_ids' in r

    def test_confidence_peaked_distribution(self):
        """Peaked gate weights should give HIGH confidence."""
        from ses.src.gpu_predictor import GPUPredictor
        # Create gate weights where one expert dominates
        W = np.zeros((256, 2048), dtype=np.float32)
        W[0, :] = 10.0  # expert 0 always wins
        W[1, :] = 9.0
        W[2, :] = 8.5
        W[3, :] = 8.0
        pred = GPUPredictor({0: W}, top_k=8)
        x = np.ones(2048, dtype=np.float32)
        result = pred.predict(x, layer=0)
        assert result['confidence'] in ('HIGH', 'MEDIUM')
        assert result['top4_mass'] > 0.5


class TestExpertCache:
    """Frequency-aware expert cache."""

    def _make_cache(self):
        from ses.src.gpu_predictor import ExpertCache
        def loader(layer, expert_id):
            return np.random.randn(1024).astype(np.float32)
        profile = {0: {'hot_count': 5}}
        return ExpertCache(profile, loader, cache_budget_gb=0.001)

    def test_warmup_loads_experts(self):
        cache = self._make_cache()
        cache.warmup({0: [0, 1, 2]})
        assert len(cache.cache) == 3

    def test_cached_hit(self):
        cache = self._make_cache()
        cache.warmup({0: [42]})
        data, is_cached = cache.get(0, 42)
        assert is_cached is True
        assert data.shape == (1024,)

    def test_uncached_miss(self):
        cache = self._make_cache()
        cache.warmup({0: [42]})
        data, is_cached = cache.get(0, 99)
        assert is_cached is False
        assert data.shape == (1024,)

    def test_hit_rate(self):
        cache = self._make_cache()
        cache.warmup({0: [0, 1, 2]})
        cache.get(0, 0)   # hit
        cache.get(0, 1)   # hit
        cache.get(0, 99)  # miss
        assert cache.hit_rate() == pytest.approx(2/3)
