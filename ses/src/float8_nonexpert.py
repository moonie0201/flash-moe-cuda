"""Float8 non-expert weight quantization for 397B.

Stores all non-expert Linear weights as float8_e4m3fn (1 byte/param vs 2 bytes bf16).
Halves VRAM for attention/norm weights, freeing space for GPU-resident expert cache.

Usage (from run_397b_ssd.py after load_non_expert_weights):
    from float8_nonexpert import apply_float8_nonexpert
    saved_gb = apply_float8_nonexpert(model)
    print(f"Float8: freed {saved_gb:.1f}GB VRAM")
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_float8_available():
    if not hasattr(torch, 'float8_e4m3fn'):
        raise RuntimeError(
            "torch.float8_e4m3fn not available — need PyTorch ≥ 2.1. "
            f"Current: {torch.__version__}"
        )


class F8Linear(nn.Module):
    """Linear layer: float8_e4m3fn weight storage, bf16/fp32 compute.

    Per-tensor scale: w_f8 = clamp(w / scale, -448, 448).to(f8)
    Forward:          y = F.linear(x, w_f8.to(x.dtype) * scale, bias)
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None,
                 target_device=None):
        super().__init__()
        _check_float8_available()
        # Convert on CPU to avoid GPU OOM (GPU already full with model weights)
        target_dev = target_device or weight.device
        w_cpu = weight.cpu().float()
        scale = w_cpu.abs().max().clamp(min=1e-30) / 448.0
        w_f8 = (w_cpu / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        del w_cpu
        self.register_buffer('weight_f8', w_f8.to(target_dev))
        self.register_buffer('w_scale', scale.to(torch.float32).reshape([]))
        if bias is not None:
            self.register_buffer('bias', bias.to(target_dev) if bias.device != target_dev else bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_f8.to(x.dtype) * self.w_scale.to(x.dtype)
        return F.linear(x, w, self.bias)

    @property
    def weight(self):
        return self.weight_f8.to(torch.bfloat16) * self.w_scale.to(torch.bfloat16)

    def extra_repr(self):
        out, inp = self.weight_f8.shape
        return f'in={inp}, out={out}, dtype=f8e4m3fn'


def _set_nested(parent, attr_path: str, new_module: nn.Module):
    """Set a nested attribute like 'self_attn.q_proj' on parent."""
    parts = attr_path.split('.')
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def apply_float8_nonexpert(model: nn.Module, verbose: bool = True) -> float:
    """Replace all non-expert Linear weights with F8Linear.

    Skips:
      - Any Linear inside '.mlp.experts.' (expert weights, handled separately)
      - Any Linear with no weight (unusual)

    Returns total VRAM freed in GB.
    """
    _check_float8_available()

    replaced = 0
    saved_bytes = 0

    # Collect (parent_path, attr_name, module) before iterating to avoid mutation issues
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if '.mlp.experts.' in name:
            continue
        if module.weight is None:
            continue
        # Skip lm_head and embed_tokens: too large (1.2GB+), dequant causes OOM during inference
        if name in ('lm_head', 'model.embed_tokens', 'embed_tokens'):
            continue
        targets.append(name)

    for name in targets:
        # Walk to parent
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]
        module = getattr(parent, attr)

        w = module.weight.data
        b = module.bias.data if module.bias is not None else None
        saved_bytes += w.nelement() * w.element_size() // 2  # bf16→f8 saves half
        dev = w.device
        # Move weight to CPU, free GPU bf16 copy, then convert+upload as f8
        w_cpu = w.cpu()
        b_cpu = b.cpu() if b is not None else None
        del w, b
        module.weight = None
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        f8 = F8Linear(w_cpu, b_cpu, target_device=dev)
        del w_cpu, b_cpu
        setattr(parent, attr, f8)
        replaced += 1

    saved_gb = saved_bytes / 1e9
    if verbose:
        print(f"  Float8: replaced {replaced} Linear layers, "
              f"freed ~{saved_gb:.2f}GB VRAM (bf16→f8)")
    return saved_gb


def float8_vram_estimate(model: nn.Module) -> dict:
    """Estimate VRAM savings from float8 conversion (without applying it)."""
    total_params = 0
    expert_params = 0
    nonexpert_params = 0

    for name, param in model.named_parameters():
        n = param.nelement()
        total_params += n
        if '.mlp.experts.' in name:
            expert_params += n
        else:
            nonexpert_params += n

    return {
        'total_params': total_params,
        'nonexpert_params': nonexpert_params,
        'bf16_gb': nonexpert_params * 2 / 1e9,
        'f8_gb':   nonexpert_params * 1 / 1e9,
        'saving_gb': nonexpert_params * 1 / 1e9,
    }
