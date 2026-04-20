#!/usr/bin/env python3
"""
train_cross_predictor.py — Train a cross-layer expert routing predictor.

Input:  hidden state at layer L  (4096-dim)
Output: predicted expert IDs at layer L+1  (512-way multilabel)

Training data is collected by run_397b_ssd.py --save-hs, which saves
[(layer_idx, h_normed_np float32[4096], expert_ids_list), ...] via pickle.

Usage:
    python train_cross_predictor.py --input /tmp/hs_trace_397b.npy \
                                    --output ses/src/cross_predictor_397b.pt
    python train_cross_predictor.py --eval cross_predictor_397b.pt \
                                    --input /tmp/hs_trace_397b.npy
"""
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_DIM = 4096
NUM_EXPERTS = 512
NUM_LAYERS = 60
LAYER_EMB_DIM = 32


class CrossLayerExpertPredictor(nn.Module):
    """Small MLP: (h_normed[L], layer_L) → logits[L+1 experts]."""

    def __init__(self, hidden_dim=HIDDEN_DIM, num_experts=NUM_EXPERTS,
                 num_layers=NUM_LAYERS, layer_emb_dim=LAYER_EMB_DIM):
        super().__init__()
        self.layer_emb = nn.Embedding(num_layers, layer_emb_dim)
        in_dim = hidden_dim + layer_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_experts),
        )

    def forward(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """h: float32[hidden_dim] on CPU. Returns logits[num_experts]."""
        emb = self.layer_emb(torch.tensor(layer_idx))
        x = torch.cat([h, emb])
        return self.net(x)

    def forward_batch(self, h_batch: torch.Tensor,
                      layer_ids: torch.Tensor) -> torch.Tensor:
        """h_batch: [N, hidden_dim], layer_ids: [N] int64."""
        emb = self.layer_emb(layer_ids)
        x = torch.cat([h_batch, emb], dim=1)
        return self.net(x)


def load_trace(path: str):
    """Load pickle trace: list of (layer_idx, h_np float32[4096], expert_ids list)."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_cross_layer_pairs(trace):
    """
    Build (h[L], layer_L, target_experts[L+1]) training pairs.

    Trace is ordered: layer 0..59 for token 0, then 0..59 for token 1, ...
    We detect token boundaries by layer_idx going from non-zero back to 0.
    """
    # Group into token sequences
    tokens = []
    cur = {}
    for layer_idx, h_np, expert_ids in trace:
        if layer_idx == 0 and cur:
            tokens.append(cur)
            cur = {}
        cur[layer_idx] = (h_np, expert_ids)
    if cur:
        tokens.append(cur)

    h_list, layer_list, target_list = [], [], []
    for tok in tokens:
        for layer in range(NUM_LAYERS - 1):
            if layer in tok and (layer + 1) in tok:
                h_np, _ = tok[layer]
                _, next_eids = tok[layer + 1]
                target = np.zeros(NUM_EXPERTS, dtype=np.float32)
                for eid in next_eids:
                    if 0 <= eid < NUM_EXPERTS:
                        target[eid] = 1.0
                h_list.append(h_np)
                layer_list.append(layer)
                target_list.append(target)

    h_arr = np.stack(h_list)          # [N, 4096]
    layer_arr = np.array(layer_list, dtype=np.int64)  # [N]
    target_arr = np.stack(target_list)  # [N, 512]
    return h_arr, layer_arr, target_arr


def compute_hit_rate(logits: torch.Tensor, targets: torch.Tensor, K: int) -> float:
    topk = logits.topk(K, dim=1).indices  # [N, K]
    hits = 0
    total = 0
    for i in range(topk.shape[0]):
        pred_set = set(topk[i].tolist())
        actual = targets[i].nonzero(as_tuple=False).squeeze(1).tolist()
        hits += len(pred_set & set(actual))
        total += len(actual)
    return hits / total if total else 0.0


def train(input_path: str, output_path: str, epochs: int = 50,
          lr: float = 1e-3, batch_size: int = 256):
    print(f"Loading trace from {input_path}...")
    trace = load_trace(input_path)
    print(f"  {len(trace)} samples collected")

    print("Building cross-layer training pairs...")
    h_arr, layer_arr, target_arr = build_cross_layer_pairs(trace)
    N = len(h_arr)
    print(f"  {N} training pairs (layer L → L+1)")

    # Train/val split (80/20 by position, not random, to avoid temporal leakage)
    split = int(N * 0.8)
    h_train = torch.from_numpy(h_arr[:split])
    l_train = torch.from_numpy(layer_arr[:split])
    t_train = torch.from_numpy(target_arr[:split])
    h_val = torch.from_numpy(h_arr[split:])
    l_val = torch.from_numpy(layer_arr[split:])
    t_val = torch.from_numpy(target_arr[split:])

    model = CrossLayerExpertPredictor()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e3:.0f}K params")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"\nTraining {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(split)
        total_loss = 0.0
        for start in range(0, split, batch_size):
            b = idx[start:start + batch_size]
            logits = model.forward_batch(h_train[b], l_train[b])
            loss = loss_fn(logits, t_train[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b)

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model.forward_batch(h_val, l_val)
                val_loss = loss_fn(val_logits, t_val).item()
                hr5  = compute_hit_rate(val_logits, t_val, K=5)
                hr10 = compute_hit_rate(val_logits, t_val, K=10)
                hr20 = compute_hit_rate(val_logits, t_val, K=20)
            print(f"  epoch {epoch:3d}: loss={total_loss/split:.4f}  val={val_loss:.4f}  "
                  f"hit@5={hr5:.1%}  hit@10={hr10:.1%}  hit@20={hr20:.1%}")

    torch.save(model.state_dict(), output_path)
    print(f"\nSaved → {output_path}")

    # Final eval on full validation set
    model.eval()
    with torch.no_grad():
        val_logits = model.forward_batch(h_val, l_val)
        print("\nFinal hit rates (validation):")
        for K in [5, 10, 15, 20]:
            hr = compute_hit_rate(val_logits, t_val, K=K)
            print(f"  top-{K:2d}: {hr:.1%}")

    return model


def evaluate(model_path: str, input_path: str):
    trace = load_trace(input_path)
    h_arr, layer_arr, target_arr = build_cross_layer_pairs(trace)
    model = CrossLayerExpertPredictor()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    h_t = torch.from_numpy(h_arr)
    l_t = torch.from_numpy(layer_arr)
    t_t = torch.from_numpy(target_arr)
    with torch.no_grad():
        logits = model.forward_batch(h_t, l_t)
    print(f"Evaluating {model_path} on {len(h_arr)} pairs:")
    for K in [5, 10, 15, 20]:
        hr = compute_hit_rate(logits, t_t, K=K)
        print(f"  top-{K:2d}: {hr:.1%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Pickle trace file from --save-hs')
    parser.add_argument('--output', default='cross_predictor_397b.pt')
    parser.add_argument('--eval', default=None, help='Evaluate existing model (skip training)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval, args.input)
    else:
        train(args.input, args.output, epochs=args.epochs,
              lr=args.lr, batch_size=args.batch_size)
