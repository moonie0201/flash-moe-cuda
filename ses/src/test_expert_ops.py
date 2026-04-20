"""Correctness + timing test for expert_ops CUDA extension.

Compares batched CUDA kernel output against reference serial Python path
(TwoBitLoader dequant + F.linear x2 + SiLU).
"""
import sys, os, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PACKED_2BIT_DIR = '/home/mh/models/Qwen3.5-397B-A17B/packed_experts_2bit'
DEVICE = torch.device('cuda:0')

# Model constants
GU_ROWS, GU_COLS = 2048, 4096
DN_ROWS, DN_COLS = 4096, 1024
GROUP_SIZE = 64

def load_expert_ops():
    import expert_ops
    return expert_ops

def reference_forward(loader, layer, eids, x, weights):
    """Serial Python reference: dequant + F.linear x2 + SiLU for each expert."""
    out = torch.zeros(DN_ROWS, dtype=torch.bfloat16, device=DEVICE)
    for eid, w in zip(eids, weights):
        gu, dw = loader.load_expert(layer, eid, DEVICE)
        gate_up = F.linear(x.unsqueeze(0), gu).squeeze(0)   # [2048]
        gate, up = gate_up.chunk(2)                          # [1024] each
        h = F.silu(gate) * up                                # [1024]
        expert_out = F.linear(h.unsqueeze(0), dw).squeeze(0) # [4096]
        out = out + expert_out * w
    return out

def batched_forward(loader, ops, layer, eids, x, weights):
    """CUDA batched path: stack raw bytes → single kernel call."""
    raw_list = []
    for eid in eids:
        raw_np = loader.load_expert_raw_bytes(layer, eid)
        raw_list.append(torch.from_numpy(raw_np).to(DEVICE))
    raw_batch = torch.stack(raw_list)   # [N, expert_size]

    routing_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    return ops.expert_batch_forward_2bit(
        raw_batch, x, routing_weights,
        GU_ROWS, GU_COLS, DN_ROWS, DN_COLS, GROUP_SIZE,
    )

def main():
    from two_bit_loader import TwoBitLoader

    print("Loading expert_ops...")
    ops = load_expert_ops()
    print(f"  expert_ops loaded: {ops}")

    print(f"Loading TwoBitLoader from {PACKED_2BIT_DIR}...")
    loader = TwoBitLoader(PACKED_2BIT_DIR, DEVICE)

    layer = 0
    eids = [0, 3, 7, 12, 25]       # 5 representative experts
    weights = [0.3, 0.2, 0.2, 0.15, 0.15]

    torch.manual_seed(42)
    x = torch.randn(GU_COLS, dtype=torch.bfloat16, device=DEVICE)

    print(f"\nLayer {layer}, experts={eids}, weights={weights}")

    # Warmup (first call has JIT overhead)
    _ = batched_forward(loader, ops, layer, eids, x, weights)
    torch.cuda.synchronize()

    print("\n--- Correctness check ---")
    ref = reference_forward(loader, layer, eids, x, weights)
    cuda_out = batched_forward(loader, ops, layer, eids, x, weights)
    torch.cuda.synchronize()

    max_diff = (ref.float() - cuda_out.float()).abs().max().item()
    mean_diff = (ref.float() - cuda_out.float()).abs().mean().item()
    ref_norm = ref.float().abs().mean().item()
    rel_err = max_diff / (ref_norm + 1e-9)

    print(f"  Reference output[:8]:  {ref[:8].tolist()}")
    print(f"  CUDA output[:8]:       {cuda_out[:8].tolist()}")
    print(f"  max_abs_diff:  {max_diff:.6f}")
    print(f"  mean_abs_diff: {mean_diff:.6f}")
    print(f"  relative_err:  {rel_err:.4%}")

    # Kernel accumulates in float32; reference uses bf16 F.linear. Small numerical differences
    # are expected and acceptable for 2-bit quantized inference (the quantization itself
    # is far noisier). Gate on absolute error, not relative (outputs near zero inflate rel%).
    if max_diff < 0.001:
        print(f"  ✓ PASSED (max_abs_diff={max_diff:.2e} — f32 vs bf16 rounding)")
    else:
        print(f"  ✗ FAILED (max_abs_diff={max_diff:.2e} too large!)")

    print("\n--- Timing (layer=0, N=10 experts, 100 iterations) ---")
    eids10 = list(range(10))
    weights10 = [0.1] * 10
    x10 = torch.randn(GU_COLS, dtype=torch.bfloat16, device=DEVICE)

    # Pre-load raw bytes to GPU to isolate kernel time
    raw_list = [torch.from_numpy(loader.load_expert_raw_bytes(layer, eid)).to(DEVICE)
                for eid in eids10]
    raw_batch = torch.stack(raw_list)
    routing_weights = torch.tensor(weights10, dtype=torch.float32, device=DEVICE)

    # Warmup
    for _ in range(5):
        ops.expert_batch_forward_2bit(raw_batch, x10, routing_weights,
                                       GU_ROWS, GU_COLS, DN_ROWS, DN_COLS, GROUP_SIZE)
    torch.cuda.synchronize()

    N_ITER = 100
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        ops.expert_batch_forward_2bit(raw_batch, x10, routing_weights,
                                       GU_ROWS, GU_COLS, DN_ROWS, DN_COLS, GROUP_SIZE)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / N_ITER * 1000

    print(f"  Batched CUDA (N=10, raw on GPU): {elapsed:.3f} ms per layer")
    print(f"  Projected (60 layers, kernel only): {elapsed*60:.1f} ms/token → "
          f"{1000/(elapsed*60):.1f} tok/s kernel bound")

    # Reference serial timing
    t0 = time.perf_counter()
    for _ in range(20):
        reference_forward(loader, layer, eids10, x10, [0.1]*10)
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"  Serial Python reference (N=10):   {ref_ms:.3f} ms per layer")
    print(f"  Speedup:  {ref_ms/elapsed:.1f}×")

    loader.close()

if __name__ == '__main__':
    main()
