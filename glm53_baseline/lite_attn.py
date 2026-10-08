
from __future__ import annotations

import sys
from functools import lru_cache

import torch

from glm53_baseline import prefill_levers

_TOPK = 2048
_HEADS, _DQK, _DV = 64, 576, 512
_announced: set = set()
_stock: list = []
current_topk: list = []
_union_cache: list = []
_MUST_BE_NONE = ("sinks", "skip_softmax_threshold_scale_factor", "sparse_mla_top_k_lens", "lse",
                 "cum_seq_lens_q", "max_q_len", "use_fp16_softmax")


def _announce(tag: str, detail: str) -> None:
    if tag not in _announced:
        _announced.add(tag)
        print(f"CACHEON_LITEDSA {tag} {detail}", file=sys.stderr, flush=True)


@lru_cache(maxsize=1)
def _native():
    import litedsa_native

    return litedsa_native


def capture_stock():
    import flashinfer.decode

    if not _stock:
        _stock.append(flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla)
    return _stock[0]


def _reason(query, kv_cache, block_tables, seq_lens, sparse_mla_top_k, out, bmm1_scale, bmm2_scale,
            extra) -> str | None:
    if (query.dtype != torch.float8_e4m3fn or query.dim() != 4 or tuple(query.shape[1:]) != (1, _HEADS, _DQK)
            or not query.is_contiguous()):
        return f"query {query.dtype} {tuple(query.shape)} is not a contiguous fp8 [T, 1, 64, 576]"
    if query.shape[0] < 2:
        return "fewer than two rows"
    if kv_cache.dtype != torch.float8_e4m3fn or kv_cache.shape[-1] != _DQK or not kv_cache.is_contiguous():
        return f"kv cache {kv_cache.dtype} {tuple(kv_cache.shape)} is not a contiguous fp8 576-wide pool"
    if (block_tables.dtype != torch.int32 or block_tables.dim() != 3 or block_tables.shape[1] != 1
            or block_tables.shape[0] != query.shape[0] or block_tables.shape[2] < _TOPK
            or block_tables.stride(2) != 1 or block_tables.stride(0) % 4 or block_tables.data_ptr() % 16):
        return (f"block tables {block_tables.dtype} {tuple(block_tables.shape)} {tuple(block_tables.stride())} "
                f"are not 16-byte aligned [T, 1, >=2048] int32 rows")
    if (seq_lens is None or seq_lens.dtype != torch.int32 or seq_lens.dim() != 1
            or seq_lens.shape[0] != query.shape[0] or not seq_lens.is_contiguous()):
        return "seq_lens are not a contiguous int32 [T]"
    if sparse_mla_top_k != _TOPK:
        return f"sparse_mla_top_k={sparse_mla_top_k}"
    if out is not None or extra.get("return_lse"):
        return "caller-provided out / lse"
    if isinstance(bmm1_scale, torch.Tensor) or isinstance(bmm2_scale, torch.Tensor):
        return "tensor scales"
    for name in _MUST_BE_NONE:
        if extra.get(name) is not None:
            return f"{name} is set"
    return None


def attention(query, kv_cache, workspace_buffer, qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim,
              block_tables, seq_lens, max_seq_len, sparse_mla_top_k=0, out=None, bmm1_scale=1.0,
              bmm2_scale=1.0, **extra):
    stock = _stock[0]
    why = _reason(query, kv_cache, block_tables, seq_lens, sparse_mla_top_k, out, bmm1_scale, bmm2_scale, extra)
    if why is None and prefill_levers.lds_stock(query.shape[0]):
        return stock(query=query, kv_cache=kv_cache, workspace_buffer=workspace_buffer,
                     qk_nope_head_dim=qk_nope_head_dim, kv_lora_rank=kv_lora_rank,
                     qk_rope_head_dim=qk_rope_head_dim, block_tables=block_tables, seq_lens=seq_lens,
                     max_seq_len=max_seq_len, sparse_mla_top_k=sparse_mla_top_k, out=out, bmm1_scale=bmm1_scale,
                     bmm2_scale=bmm2_scale, **extra)
    if why is not None:
        _announce("stock", why)
        return stock(query=query, kv_cache=kv_cache, workspace_buffer=workspace_buffer,
                     qk_nope_head_dim=qk_nope_head_dim, kv_lora_rank=kv_lora_rank,
                     qk_rope_head_dim=qk_rope_head_dim, block_tables=block_tables, seq_lens=seq_lens,
                     max_seq_len=max_seq_len, sparse_mla_top_k=sparse_mla_top_k, out=out, bmm1_scale=bmm1_scale,
                     bmm2_scale=bmm2_scale, **extra)
    rows = query.shape[0]
    even = rows - rows % 2
    pairs = even // 2
    dev = query.device
    native = _native()
    tables = block_tables.view(rows, block_tables.shape[2])
    key = current_topk[0] if current_topk else None
    shape = (rows, block_tables.shape[2])
    if (key is not None and _union_cache and _union_cache[0] is key and _union_cache[1] == key._version
            and _union_cache[2] == shape):
        union, counts, memb = _union_cache[3:]
    else:
        union = torch.empty(pairs, 2 * _TOPK, dtype=torch.int32, device=dev)
        counts = torch.empty(pairs, dtype=torch.int32, device=dev)
        memb = torch.empty(pairs, 2, 2 * _TOPK // 32, dtype=torch.int32, device=dev)
        native.pair_union(tables[:even], seq_lens[:even], union, counts, memb)
        _union_cache[:] = [key, key._version, shape, union, counts, memb] if key is not None else []
    result = torch.empty(rows, 1, _HEADS, _DV, dtype=torch.bfloat16, device=dev)
    scratch = torch.empty(2, pairs, 128, dtype=torch.float32, device=dev)
    native.masked_mla(query[:even].view(pairs, 128, _DQK), kv_cache.view(-1, 1, _DQK), union,
                      float(bmm1_scale), float(bmm2_scale), counts, memb, result[:even].view(pairs, 128, _DV),
                      scratch[0], scratch[1])
    if rows != even:
        result[even:] = stock(query=query[even:], kv_cache=kv_cache, workspace_buffer=workspace_buffer,
                              qk_nope_head_dim=qk_nope_head_dim, kv_lora_rank=kv_lora_rank,
                              qk_rope_head_dim=qk_rope_head_dim, block_tables=block_tables[even:],
                              seq_lens=seq_lens[even:], max_seq_len=max_seq_len,
                              sparse_mla_top_k=sparse_mla_top_k, bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale,
                              **extra)
    _announce("pairs", f"prefill sparse MLA through LiteDSA token pairs, rows={rows}")
    return result
