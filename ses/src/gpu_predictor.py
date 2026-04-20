#!/usr/bin/env python3
"""GPU-based expert prediction and confidence analysis.

Uses GPU 0 to run gate projections for upcoming layers,
predicting which experts will be activated before the CPU needs them.

Requires PyTorch with CUDA support.
"""
import numpy as np
import time

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


class GPUPredictor:
    """Dedicated GPU expert predictor.

    Loads all gate weights onto GPU 0 and runs gate projections
    to predict expert routing for upcoming layers.
    """

    def __init__(self, gate_weights: dict, device_id: int = 0, top_k: int = 8):
        """
        Args:
            gate_weights: {layer_idx: numpy array [num_experts, hidden_dim]}
            device_id: GPU device for prediction
            top_k: number of experts per token
        """
        self.top_k = top_k
        self.num_layers = len(gate_weights)

        if HAS_CUDA:
            self.device = torch.device(f'cuda:{device_id}')
            self.gate_tensors = {}
            total_bytes = 0
            for layer, W in gate_weights.items():
                t = torch.from_numpy(W).to(self.device)
                self.gate_tensors[layer] = t
                total_bytes += t.nelement() * t.element_size()
            self.gpu_mode = True
            print(f"GPUPredictor: {len(self.gate_tensors)} layers on GPU {device_id} "
                  f"({total_bytes/1e6:.1f} MB)")
        else:
            # CPU fallback
            self.gate_np = gate_weights
            self.gpu_mode = False
            print(f"GPUPredictor: CPU mode ({len(gate_weights)} layers)")

    def predict(self, hidden_state: np.ndarray, layer: int):
        """Predict top-K experts and confidence for a layer.

        Args:
            hidden_state: [hidden_dim] current hidden state
            layer: target layer index

        Returns:
            dict with:
                expert_ids: top-K predicted expert indices
                scores: softmax probabilities for top-K
                confidence: 'HIGH', 'MEDIUM', or 'LOW'
                top4_mass: probability mass of top-4
                entropy: normalized entropy of distribution
        """
        if self.gpu_mode:
            return self._predict_gpu(hidden_state, layer)
        else:
            return self._predict_cpu(hidden_state, layer)

    def _predict_gpu(self, hidden_state, layer):
        with torch.no_grad():
            x = torch.from_numpy(hidden_state).to(self.device)
            W = self.gate_tensors[layer]
            scores = W @ x  # [num_experts]
            probs = torch.softmax(scores, dim=0)

            topk_vals, topk_ids = torch.topk(probs, self.top_k)

            # Confidence analysis
            top4_mass = float(torch.topk(probs, 4).values.sum())
            entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)))
            max_entropy = float(torch.log(torch.tensor(float(len(probs)))))
            norm_entropy = entropy / max_entropy

            return {
                'expert_ids': topk_ids.cpu().numpy(),
                'scores': topk_vals.cpu().numpy(),
                'confidence': self._classify(top4_mass, norm_entropy),
                'top4_mass': top4_mass,
                'entropy': norm_entropy,
            }

    def _predict_cpu(self, hidden_state, layer):
        W = self.gate_np[layer]
        scores = W @ hidden_state
        scores_shifted = scores - scores.max()
        probs = np.exp(scores_shifted) / np.exp(scores_shifted).sum()

        topk_ids = np.argsort(probs)[-self.top_k:][::-1]
        topk_vals = probs[topk_ids]

        top4_mass = float(np.sort(probs)[-4:].sum())
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = np.log(len(probs))
        norm_entropy = entropy / max_entropy

        return {
            'expert_ids': topk_ids,
            'scores': topk_vals,
            'confidence': self._classify(top4_mass, norm_entropy),
            'top4_mass': top4_mass,
            'entropy': norm_entropy,
        }

    @staticmethod
    def _classify(top4_mass, norm_entropy):
        if top4_mass > 0.8 and norm_entropy < 0.3:
            return 'HIGH'
        elif top4_mass > 0.5 or norm_entropy < 0.7:
            return 'MEDIUM'
        else:
            return 'LOW'

    def predict_batch(self, hidden_state: np.ndarray, layers: list):
        """Predict multiple layers at once (for lookahead)."""
        return [self.predict(hidden_state, l) for l in layers]

    def benchmark(self, hidden_dim: int, num_iters: int = 1000):
        """Measure prediction latency."""
        x = np.random.randn(hidden_dim).astype(np.float32)
        layers = list(range(min(5, self.num_layers)))

        # Warmup
        for _ in range(10):
            for l in layers:
                self.predict(x, l)

        # Benchmark
        t0 = time.time()
        for _ in range(num_iters):
            for l in layers:
                self.predict(x, l)
        elapsed = time.time() - t0
        total_predictions = num_iters * len(layers)
        per_prediction_us = elapsed / total_predictions * 1e6

        return {
            'total_predictions': total_predictions,
            'elapsed_s': elapsed,
            'per_prediction_us': per_prediction_us,
        }


class ExpertCache:
    """Frequency-aware expert cache.

    HOT experts are pinned in memory, WARM/COLD loaded on demand.
    """

    def __init__(self, profile: dict, expert_loader, cache_budget_gb: float = 50.0):
        """
        Args:
            profile: {layer: {hot_count, warm_count, cold_count, ...}}
            expert_loader: callable(layer, expert_id) -> numpy array
            cache_budget_gb: maximum RAM for expert cache
        """
        self.profile = profile
        self.loader = expert_loader
        self.budget = cache_budget_gb * 1e9
        self.cache = {}  # (layer, expert_id) -> numpy array
        self.stats = {'hits': 0, 'misses': 0}

    def warmup(self, hot_expert_ids: dict):
        """Pre-load HOT experts into cache.

        Args:
            hot_expert_ids: {layer: [expert_id, ...]}
        """
        total_bytes = 0
        for layer, expert_ids in hot_expert_ids.items():
            for eid in expert_ids:
                data = self.loader(layer, eid)
                self.cache[(layer, eid)] = data
                total_bytes += data.nbytes
                if total_bytes > self.budget:
                    print(f"Cache budget exhausted at {total_bytes/1e9:.1f} GB")
                    return
        print(f"Warmed up {len(self.cache)} experts ({total_bytes/1e9:.1f} GB)")

    def get(self, layer: int, expert_id: int):
        """Get expert data, from cache if available."""
        key = (layer, expert_id)
        if key in self.cache:
            self.stats['hits'] += 1
            return self.cache[key], True  # data, is_cached
        else:
            self.stats['misses'] += 1
            data = self.loader(layer, expert_id)
            return data, False

    def hit_rate(self):
        total = self.stats['hits'] + self.stats['misses']
        return self.stats['hits'] / total if total > 0 else 0.0
