
from functools import lru_cache

import torch

ENABLED = True
MAX_SEPARATE = 18
PER = 4
MAX_DMLA8 = 37


@lru_cache(maxsize=1)
def _native8():
    import dmla8

    return dmla8


@lru_cache(maxsize=4)
def _cluster_limit8(index: int):
    return int(_native8().max_clusters(16 // PER))


@lru_cache(maxsize=1)
def _native():
    import dmla4

    return dmla4


@lru_cache(maxsize=4)
def _cluster_limits(index: int):
    native = _native()
    return int(native.max_clusters(16)), int(native.max_clusters(8))


def _plan(t: int, index: int):
    c16, c8 = _cluster_limits(index)
    if t <= c16:
        return 4, 16, True
    if t <= c8:
        return 4, 8, True
    if t <= MAX_SEPARATE:
        return 4, 8, False
    if t <= MAX_DMLA8:
        return 8, PER, t <= _cluster_limit8(index)
    return None


def decode(query, k_cache, page_table, seq_lens, bmm1_scale):
    if not ENABLED:
        return None
    t = query.shape[0]
    if (query.dim() != 4 or tuple(query.shape[1:]) != (1, 64, 576) or query.dtype != torch.float8_e4m3fn
            or not query.is_contiguous() or k_cache.dtype != torch.float8_e4m3fn or k_cache.shape[-1] != 576
            or not k_cache.is_contiguous() or page_table.dim() != 2 or page_table.shape[0] != t
            or page_table.shape[1] < 2048 or page_table.dtype != torch.int32 or page_table.stride(1) != 1
            or page_table.stride(0) % 4 or page_table.data_ptr() % 16 or seq_lens.dtype != torch.int32
            or seq_lens.numel() != t or not seq_lens.is_contiguous()):
        return None
    dev = query.device
    plan = _plan(t, dev.index if dev.index is not None else 0) if t >= 1 else None
    if plan is None:
        return None
    kernel, splits, fused = plan
    out = torch.empty((t, 1, 64, 512), dtype=torch.bfloat16, device=dev)
    if kernel == 8:
        rows = int(_native8().parts_rows(t, splits))
        parts = torch.empty(rows, 512 * 64, dtype=torch.bfloat16, device=dev)
        lse = torch.empty(rows, 64, dtype=torch.float32, device=dev)
        _native8().decode8(query.view(t, 64, 576), k_cache.view(-1, 576), page_table, seq_lens, float(bmm1_scale),
                           splits, parts, lse, out.view(t, 64, 512), True, True, None, 0, 1 if fused else 0, None)
        return out
    parts = torch.empty(t, splits, 512 * 64, dtype=torch.bfloat16, device=dev)
    lse = torch.empty(t, splits, 64, dtype=torch.float32, device=dev)
    _native().decode(query.view(t, 64, 576), k_cache.view(-1, 576), page_table, seq_lens, float(bmm1_scale), splits,
                     parts, lse, out.view(t, 64, 512), True, True, None, fused)
    return out
