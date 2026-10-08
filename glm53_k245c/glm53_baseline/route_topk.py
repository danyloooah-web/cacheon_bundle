
import torch
import triton
import triton.language as tl

from glm53_baseline.fused_add_rmsnorm import _pdl_trigger, _pdl_wait

TOP_K = 8


@triton.jit
def _route(L, BIAS, IDS, W, scale, E: tl.constexpr, K: tl.constexpr, USE_PDL: tl.constexpr):
    row = tl.program_id(0)
    if USE_PDL:
        row = _pdl_wait(row)
        row = _pdl_trigger(row)
    cols = tl.arange(0, E)
    logits = tl.load(L + row * E + cols)
    ks = tl.arange(0, K)
    if tl.max(tl.abs(logits), axis=0) == 0.0:
        tl.store(IDS + row * K + ks, tl.full([K], -1, tl.int32))
        tl.store(W + row * K + ks, tl.zeros([K], tl.float32))
    else:
        score = 1.0 / (1.0 + tl.exp(-logits))
        key = score + tl.load(BIAS + cols).to(tl.float32)
        ids = tl.zeros([K], tl.int32)
        picked = tl.zeros([K], tl.float32)
        for k in tl.static_range(K):
            best = tl.max(key, axis=0)
            at = tl.min(tl.where(key == best, cols, E), axis=0)
            ids = tl.where(ks == k, at, ids)
            picked = tl.where(ks == k, tl.sum(tl.where(cols == at, score, 0.0), axis=0), picked)
            key = tl.where(cols == at, float("-inf"), key)
        total = tl.sum(picked, axis=0)
        tl.store(IDS + row * K + ks, ids)
        tl.store(W + row * K + ks, picked / total * scale)


def route(logits: torch.Tensor, bias: torch.Tensor, scale: float, pdl: bool = True):
    rows, experts = logits.shape
    if (logits.dtype != torch.float32 or not logits.is_contiguous() or experts & (experts - 1)
            or bias is None or bias.numel() != experts or not bias.is_contiguous()):
        raise ValueError(f"route_topk: logits {tuple(logits.shape)} {logits.dtype}, bias "
                         f"{None if bias is None else (tuple(bias.shape), bias.dtype)}")
    ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=logits.device)
    weights = torch.empty((rows, TOP_K), dtype=torch.float32, device=logits.device)
    _route[(rows,)](logits, bias, ids, weights, float(scale), experts, TOP_K, pdl, num_warps=4, launch_pdl=pdl)
    return ids, weights


_R8: list = []


def _route8():
    if not _R8:
        import route8

        _R8.append(route8)
    return _R8[0]


_route_triton = route


PRUNE_MAX_COUNT = 1
PRUNE_BUDGET = 0.25
PRUNE_PAIR = True
PRUNE_POS_MULT = (1.0, 1.0, 1.0, 1.0)
PRUNE_RENORM = False
_COUNTS: dict = {}


def route(logits: torch.Tensor, bias: torch.Tensor, scale: float, pdl: bool = True, rezero: bool = False):
    if rezero and not (PRUNE_BUDGET > 0.0 and PRUNE_MAX_COUNT > 0):
        raise ValueError("router_tri's atomic logits need the pruning route (route8.route_prune) to re-zero them")
    if logits.dim() == 3:
        if not (PRUNE_BUDGET > 0.0 and PRUNE_MAX_COUNT > 0):
            raise ValueError("router slabs need the pruning route (route8.route_prune)")
    elif not (logits.dim() == 2 and logits.shape[1] == 256):
        return _route_triton(logits, bias, scale, pdl)
    if (logits.dtype == torch.float32 and logits.shape[-1] == 256 and logits.is_contiguous()
            and bias is not None and bias.dtype == torch.float32 and bias.numel() == 256 and bias.is_contiguous()):
        rows = logits.shape[-2]
        ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=logits.device)
        weights = torch.empty((rows, TOP_K), dtype=torch.float32, device=logits.device)
        if PRUNE_BUDGET > 0.0 and PRUNE_MAX_COUNT > 0:
            counts = _COUNTS.get(logits.device)
            if counts is None:
                counts = _COUNTS[logits.device] = torch.zeros(1, dtype=torch.int32, device=logits.device)
            _route8().route_prune(logits, bias, ids, weights, float(scale), pdl, counts, PRUNE_MAX_COUNT, PRUNE_BUDGET,
                                  PRUNE_PAIR, list(PRUNE_POS_MULT), PRUNE_RENORM, rezero)
        else:
            _route8().route(logits, bias, ids, weights, float(scale), pdl)
        return ids, weights
    return _route_triton(logits, bias, scale, pdl)
