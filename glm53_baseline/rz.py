
import sys

import torch

ENABLED = True
MAX_ROWS = 16
N, K = 256, 6144

_NATIVE: list = []
_announced: set = set()


def _announce(tag: str, detail: str) -> None:
    if tag not in _announced:
        _announced.add(tag)
        print(f"CACHEON_RZ {tag} {detail}", file=sys.stderr, flush=True)


def _native():
    if not _NATIVE:
        import route8s

        _NATIVE.append(route8s)
    return _NATIVE[0]


def _stock_is_tiny(gate, rows: int) -> bool:
    from sglang.srt.models import deepseek_v2

    if rows > getattr(gate, "tiny_router_gemm_max_tokens", 0):
        return False
    if deepseek_v2.use_intel_amx_backend(gate):
        return False
    return not deepseek_v2.get_exec().deterministic.enable_deterministic_inference


def logits(gate, x: torch.Tensor, zero: torch.Tensor):
    w = getattr(gate, "weight", None)
    rows = x.shape[0] if x.dim() == 2 else 0
    if (not ENABLED or w is None or not 1 <= rows <= MAX_ROWS or x.shape[1] != K or x.dtype != torch.bfloat16
            or not x.is_contiguous() or x.data_ptr() % 32 or w.dtype != torch.bfloat16 or tuple(w.shape) != (N, K)
            or not w.is_contiguous() or w.data_ptr() % 32 or zero.dtype != torch.uint8 or zero.numel() < MAX_ROWS
            or zero.data_ptr() % 16 or zero.device != x.device or not _stock_is_tiny(gate, rows)):
        return None
    out = torch.empty((rows, N), dtype=torch.float32, device=x.device)
    _native().rz_run(x, w, zero, out, True, True)
    _announce("served", f"{rows} gathered rows, padding requests skipped")
    return out
