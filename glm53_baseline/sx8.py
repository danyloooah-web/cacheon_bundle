
from __future__ import annotations

import sys
from functools import lru_cache

import torch

ENABLED = True
MAX_ROWS = 128
GATE_UP_MAX_ROWS = 32
MIN_LAYER = 3
_SG = 2
_SUMS: dict = {}
_announced: set = set()


def _announce(tag: str, detail: str) -> None:
    if tag not in _announced:
        _announced.add(tag)
        print(f"CACHEON_SX8 {tag} {detail}", file=sys.stderr, flush=True)


@lru_cache(maxsize=1)
def _native():
    import sx8_native

    return sx8_native


def quantize(w: torch.Tensor, sg: int = _SG):
    b, n, k = w.shape
    g = 64 * sg
    wf = w.float().reshape(b, n, k // g, g)
    s = (wf.abs().amax(-1) / 127.0).to(torch.float16)
    sf = s.float()
    q = torch.round(wf / torch.where(sf > 0, sf, torch.ones_like(sf)).unsqueeze(-1)).clamp_(-127, 127)
    del wf
    u = (q.reshape(b, n, k) + 128.0).to(torch.uint8)
    del q
    frag = u.view(b, n // 8, 8, k // 64, 2, 4, 8).permute(0, 1, 3, 2, 5, 4, 6).contiguous().reshape(-1)
    scales = s.view(b, n // 16, 2, 8, k // g).permute(0, 1, 4, 3, 2).contiguous()
    return frag, scales


def interleave_gate_up(w: torch.Tensor) -> torch.Tensor:
    two_i, k = w.shape
    i = two_i // 2
    return torch.stack([w[:i].view(i // 8, 8, k), w[i:].view(i // 8, 8, k)], dim=1).reshape(two_i, k).contiguous()


def sums_buffer(device):
    b = _SUMS.get(device)
    if b is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("sx8: the sums buffer must exist before capture")
        b = _SUMS[device] = torch.zeros((MAX_ROWS, 1024), dtype=torch.float32, device=device)
    return b


class SharedI8:
    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("sx8: INT8 copies are made in prepare, never under stream capture")
        self.k = gate_up.shape[1]
        self.gate_up = quantize(interleave_gate_up(gate_up).unsqueeze(0))
        self.down = quantize(down.unsqueeze(0))
        self.sums = sums_buffer(gate_up.device)

    def gate_up_act(self, x: torch.Tensor) -> torch.Tensor:
        act = torch.empty((x.shape[0], 512), dtype=torch.bfloat16, device=x.device)
        _native().gate_up_sums(x.unsqueeze(0), self.gate_up[0], self.gate_up[1], self.sums, True)
        _native().act_rezero(self.sums, act)
        return act

    def act_down(self, act: torch.Tensor) -> torch.Tensor:
        out = torch.empty((act.shape[0], 6144), dtype=torch.bfloat16, device=act.device)
        _native().down(act.unsqueeze(0), self.down[0], self.down[1], out.unsqueeze(0), True)
        return out


def _local_only(down) -> bool:
    from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
    from sglang.srt.runtime_context import get_forward

    tp = int(getattr(down, "tp_size", 1))
    if tp > 1 and get_forward().sp_active:
        return False
    reduces = (getattr(down, "reduce_results", False) and tp > 1) or getattr(down, "use_decode_attn_tp", False)
    return not reduces or should_skip_mlp_all_reduce()


def install(mlp, layer_id: int) -> None:
    from sglang.srt.models.deepseek_v2 import DeepseekV2MLP

    shared_mlp = getattr(mlp, "shared_experts", None)
    if not ENABLED or layer_id < MIN_LAYER or not isinstance(shared_mlp, DeepseekV2MLP):
        return
    gu, dn = getattr(shared_mlp.gate_up_proj, "weight", None), getattr(shared_mlp.down_proj, "weight", None)
    if (not isinstance(gu, torch.Tensor) or not isinstance(dn, torch.Tensor) or gu.dtype != torch.bfloat16
            or dn.dtype != torch.bfloat16 or tuple(gu.shape) != (1024, 6144) or tuple(dn.shape) != (6144, 512)
            or not gu.is_contiguous() or not dn.is_contiguous() or not gu.is_cuda
            or getattr(shared_mlp.gate_up_proj, "bias", None) is not None
            or getattr(shared_mlp.down_proj, "bias", None) is not None):
        _announce("static_stock_shared", f"gate_up {getattr(gu, 'dtype', None)} {tuple(getattr(gu, 'shape', ()))}")
        return
    _native()
    shared_mlp._cacheon_sx8 = SharedI8(gu, dn)


def shared(mlp, x):
    state = getattr(mlp, "_cacheon_sx8", None)
    if (state is None or not isinstance(x, torch.Tensor) or x.dim() != 2 or not 1 <= x.shape[0] <= MAX_ROWS
            or x.shape[1] != state.k or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16
            or torch.cuda.is_current_stream_capturing() and state.sums is None or not _local_only(mlp.down_proj)):
        return None
    _announce("served_shared", f"<= {MAX_ROWS} gathered rows")
    return state
