from __future__ import annotations

import bisect
import itertools
import sys

import torch

LDS = True
LDS_CTX = 10240
MEMFIX = True
NOGATHER = True

_said: set = set()
_DSA_MODULE = "sglang.srt.layers.attention.dsa.dsa_indexer"


def _say(tag, line: str) -> None:
    if tag not in _said:
        _said.add(tag)
        print(f"CACHEON_PREFILL {line}", file=sys.stderr, flush=True)


def _rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
    except Exception:
        pass
    return -1


_lds_call: list = []


def lds_sk_eff(prefix_lens, extend_lens):
    if prefix_lens is None or extend_lens is None:
        return None
    try:
        pre = [int(p) for p in prefix_lens]
        ext = [int(e) for e in extend_lens]
    except (TypeError, ValueError):
        return None
    if not ext or len(pre) != len(ext) or min(ext) < 0 or min(pre) < 0:
        return None
    rows = sum(ext)
    if rows <= 0:
        return None
    return rows, sum(e * (p + e) for p, e in zip(pre, ext)) / rows


def lds_arm(forward_batch) -> None:
    _lds_call.clear()
    if not LDS:
        return
    got = lds_sk_eff(getattr(forward_batch, "extend_prefix_lens_cpu", None),
                     getattr(forward_batch, "extend_seq_lens_cpu", None))
    if got is not None:
        _lds_call[:] = list(got)


def lds_clear() -> None:
    _lds_call.clear()


def lds_stock(rows: int) -> bool:
    if not LDS or not _lds_call or _lds_call[0] != int(rows):
        return False
    sk = _lds_call[1]
    if sk < LDS_CTX:
        _say("lds_pairs", f"lds pairs rank={_rank()} rows={rows} sk_eff={sk:.0f} < {LDS_CTX}: LiteDSA token pairs")
        return False
    _say("lds_stock", f"lds stock rank={_rank()} rows={rows} sk_eff={sk:.0f} >= {LDS_CTX}: trtllm-gen sparse MLA")
    return True


def _release(chunk) -> int:
    if not isinstance(chunk, torch.Tensor) or not chunk.is_cuda or chunk.dim() != 2 or chunk.storage_offset() != 0:
        return 0
    storage = chunk.untyped_storage()
    size = int(storage.nbytes())
    if size == 0 or size != int(chunk.shape[0]) * int(chunk.stride(0)) * chunk.element_size():
        return 0
    if not storage.resizable():
        _say("memfix_keep", f"memfix keep rank={_rank()} logits chunk storage not resizable: stock lifetime")
        return 0
    try:
        storage.resize_(0)
    except RuntimeError as exc:
        _say("memfix_keep", f"memfix keep rank={_rank()} resize: {type(exc).__name__}: {exc}")
        return 0
    return size


def _memfix_install(indexer) -> None:
    cls = type(indexer)
    if not (callable(getattr(cls, "_get_mqa_logits_budget_bytes", None))
            and callable(getattr(cls, "_get_topk_ragged", None))
            and callable(getattr(cls, "_pad_heads_for_deep_gemm", None))
            and callable(getattr(cls, "_mask_init_and_local_tokens", None))):
        _say(("memfix_off", cls.__name__), f"memfix off rank={_rank()} {cls.__name__}: no ragged hooks")
        return
    from glm53_baseline import qabsorb

    cap = int(qabsorb._LOGITS_CAP)
    if not isinstance(getattr(indexer, "_mqa_logits_budget_bytes", None), qabsorb._CappedBudget):
        indexer._mqa_logits_budget_bytes = qabsorb._CappedBudget()
    stock_budget = indexer._get_mqa_logits_budget_bytes
    stock_ragged = indexer._get_topk_ragged
    stock_pad = indexer._pad_heads_for_deep_gemm
    stock_mask = indexer._mask_init_and_local_tokens
    state = {"on": False, "chunk": None}

    def budget(device_index):
        return min(int(stock_budget(device_index)), cap)

    def ragged(*args, **kwargs):
        if state["on"]:
            return stock_ragged(*args, **kwargs)
        state["on"], state["chunk"] = True, None
        try:
            return stock_ragged(*args, **kwargs)
        finally:
            state["on"], state["chunk"] = False, None

    def mask(logits, *args, **kwargs):
        if state["on"]:
            state["chunk"] = logits
        return stock_mask(logits, *args, **kwargs)

    def pad(q_fp8, weights):
        prev = state["chunk"]
        if prev is not None:
            state["chunk"] = None
            if not torch.cuda.is_current_stream_capturing():
                freed = _release(prev)
                if freed:
                    _say("memfix_release", f"memfix release rank={_rank()} bytes={freed} (previous logits chunk)")
        return stock_pad(q_fp8, weights)

    indexer._get_mqa_logits_budget_bytes = budget
    indexer._get_topk_ragged = ragged
    indexer._pad_heads_for_deep_gemm = pad
    indexer._mask_init_and_local_tokens = mask
    _say("memfix_install", f"memfix install rank={_rank()} cap={cap} ({cls.__name__})")


def _req_cu(indexer, metadata, q_offset: int):
    if getattr(indexer, "dsa_enable_prefill_cp", False):
        return None
    try:
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import DSAIndexerMetadata
    except ImportError:
        return None
    if not (isinstance(metadata, DSAIndexerMetadata) and metadata.topk_backend.is_sgl_kernel()
            and not metadata.force_unfused_topk and envs.SGLANG_DSA_FUSE_TOPK.get()):
        return None
    attn = metadata.attn_metadata
    lens = metadata.get_dsa_extend_len_cpu()
    if (not lens or attn.page_table_1 is None or attn.page_table_1.shape[0] != len(lens)
            or attn.cu_seqlens_q is None or attn.cu_seqlens_q.shape[0] != len(lens) + 1
            or min(lens) <= 0 or sum(lens) != q_offset):
        return None
    return list(itertools.accumulate(lens, initial=0))


def _chunk_rows(req_cu_cpu, req_cu, start: int, end: int):
    first = bisect.bisect_right(req_cu_cpu, start) - 1
    last = bisect.bisect_right(req_cu_cpu, end - 1) - 1
    chunk_cu = (req_cu[first:last + 2] - start).clamp_(min=0, max=end - start)
    return chunk_cu[1:] - chunk_cu[:-1], slice(first, last + 1)


def _nogather_ragged(indexer, prior, enable_dual_stream, forward_batch, layer_id, q_fp8, weights, metadata,
                     topk_result=None):
    dsa = sys.modules.get(_DSA_MODULE)
    if (dsa is None or getattr(dsa, "_is_hip", True) or getattr(dsa, "_is_xpu", True)
            or getattr(dsa, "_is_fp8_fnuz", True) or not callable(getattr(getattr(dsa, "deep_gemm", None),
                                                                          "fp8_mqa_logits", None))
            or (q_fp8.is_cuda and torch.cuda.is_current_stream_capturing())
            or not forward_batch.forward_mode.is_extend_without_speculative()
            or forward_batch.seq_lens_cpu is None or forward_batch.extend_seq_lens_cpu is None
            or weights.dim() != 3 or q_fp8.dim() != 3 or dsa.get_token_to_kv_pool().page_size != 64
            or metadata.attn_metadata.topk_indices_offset is not None):
        return prior(enable_dual_stream, forward_batch, layer_id, q_fp8, weights, metadata, topk_result)
    block_tables = metadata.get_page_table_64()
    ks, ke = metadata.get_indexer_kvcache_range()
    q_offset = int(ks.shape[0])
    seq_cpu = metadata.get_indexer_seq_len_cpu()
    if len(block_tables) == 0 or q_offset == 0 or seq_cpu is None or seq_cpu.numel() == 0:
        return prior(enable_dual_stream, forward_batch, layer_id, q_fp8, weights, metadata, topk_result)
    seq_len_sum = torch.sum(seq_cpu).item()
    max_seq_len = torch.max(seq_cpu).item()
    device = q_fp8.device
    need_chunk, _ = indexer._should_chunk_mqa_logits(q_offset, int(seq_len_sum), device.index)
    req_cu = _req_cu(indexer, metadata, q_offset) if need_chunk else None
    if req_cu is None:
        return prior(enable_dual_stream, forward_batch, layer_id, q_fp8, weights, metadata, topk_result)

    dg = dsa.deep_gemm
    weights = weights.squeeze(-1)
    if topk_result is None:
        topk_result = torch.full((q_fp8.shape[0], indexer.index_topk), -1, device=device, dtype=torch.int32)
    k_fp8, k_scale = dsa.get_token_to_kv_pool().get_index_k_scale_buffer(
        layer_id, metadata.get_indexer_seq_len(), block_tables, seq_len_sum, max_seq_len)
    kv_fp8 = (k_fp8.view(torch.float8_e4m3fn), k_scale.view(torch.float32).squeeze(-1))
    seq_lens_expanded = metadata.get_seqlens_expanded()
    k_offset = int(kv_fp8[0].shape[0])
    need_chunk, budget = indexer._should_chunk_mqa_logits(q_offset, k_offset, device.index)
    if not need_chunk:
        return prior(enable_dual_stream, forward_batch, layer_id, q_fp8, weights.unsqueeze(-1), metadata,
                     topk_result)
    rows_per = min(max(1, int(budget // max(k_offset * indexer._MQA_LOGITS_BYTES_PER_ELEM, 1))), q_offset)
    assert seq_lens_expanded.shape[0] == q_offset, (
        f"seq_lens_expanded length mismatch: {seq_lens_expanded.shape[0]} != {q_offset}")
    _say("nogather_on", f"nogather on rank={_rank()} layer={layer_id} rows={q_offset} requests={len(req_cu) - 1} "
                        f"chunk_rows={rows_per}")
    start = 0
    logits_chunk = None
    while start < q_offset:
        end = min(start + rows_per, q_offset)
        logits_chunk = None
        with indexer._with_real_sm_count():
            q_padded, w_padded, _ = indexer._pad_heads_for_deep_gemm(q_fp8[start:end], weights[start:end])
            logits_chunk = dg.fp8_mqa_logits(q_padded, kv_fp8, w_padded, ks[start:end], ke[start:end],
                                             clean_logits=False)
        lengths_chunk = seq_lens_expanded[start:end]
        indexer._mask_init_and_local_tokens(logits_chunk, lengths_chunk, ks[start:end])
        cu_chunk, batch_chunk = _chunk_rows(req_cu, metadata.attn_metadata.cu_seqlens_q, start, end)
        topk_result[start:end] = metadata.topk_transform(
            logits_chunk, indexer.index_topk, ks=ks[start:end], cu_seqlens_q=cu_chunk, ke_offset=lengths_chunk,
            batch_idx_list=batch_chunk, topk_indices_offset_override=None)
        start = end
    return topk_result


def install(indexer) -> None:
    if indexer is None or getattr(indexer, "_cacheon_prefill_levers_id", None) == id(indexer):
        return
    if getattr(indexer, "_cacheon_prefill_levers_id", None) is not None:
        if MEMFIX:
            del indexer._get_mqa_logits_budget_bytes
            del indexer._pad_heads_for_deep_gemm
            del indexer._mask_init_and_local_tokens
        if MEMFIX or NOGATHER:
            del indexer._get_topk_ragged
    if MEMFIX:
        _memfix_install(indexer)
    cls = type(indexer)
    if NOGATHER and (callable(getattr(cls, "_get_topk_ragged", None))
                     and callable(getattr(cls, "_pad_heads_for_deep_gemm", None))
                     and callable(getattr(cls, "_mask_init_and_local_tokens", None))
                     and callable(getattr(cls, "_get_mqa_logits_budget_bytes", None))
                     and callable(getattr(cls, "_should_chunk_mqa_logits", None))
                     and callable(getattr(cls, "_with_real_sm_count", None))):
        prior = indexer._get_topk_ragged

        def ragged(enable_dual_stream, forward_batch, layer_id, q_fp8, weights, metadata, topk_result=None):
            return _nogather_ragged(indexer, prior, enable_dual_stream, forward_batch, layer_id, q_fp8, weights,
                                    metadata, topk_result)

        indexer._get_topk_ragged = ragged
    indexer._cacheon_prefill_levers_id = id(indexer)
    _say(("install", cls.__name__), f"install rank={_rank()} memfix={int(MEMFIX)} nogather={int(NOGATHER)} "
                                    f"lds={int(LDS)}:{LDS_CTX} ({cls.__name__})")
