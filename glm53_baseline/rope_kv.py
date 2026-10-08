
import sys

import torch
import triton
import triton.language as tl

from glm53_baseline import dqab
from glm53_baseline.fused_add_rmsnorm import _pdl_trigger, _pdl_wait

FMA_FORM = 1
FUSED_MAX_ROWS = 8
WIDE_MAX = 64
WIDE_HB = 4
WIDE_WARPS = 2
DMLA_WIDE = True
_KWARGS = frozenset({"q_rope", "k_rope", "cos_sin_cache", "is_neox", "llama_4_scaling", "topk_indices"})
_announced: set = set()


def _announce(tag: str, detail: str) -> None:
    if tag not in _announced:
        _announced.add(tag)
        print(f"CACHEON_ROPE_KV {tag} {detail}", file=sys.stderr, flush=True)


@triton.jit
def _e4m3(x):
    return tl.minimum(tl.maximum(x, -448.0), 448.0).to(tl.float8e4nv)


@triton.jit
def _rope_pair(xe, xo, cos, sin, FMA: tl.constexpr):
    nxo = xo * -1.0
    if FMA == 1:
        ye = tl.fma(xe, cos, nxo * sin)
        yo = tl.fma(xo, cos, xe * sin)
    elif FMA == 2:
        ye = tl.fma(nxo, sin, xe * cos)
        yo = tl.fma(xe, sin, xo * cos)
    else:
        ye = xe * cos + nxo * sin
        yo = xo * cos + xe * sin
    return ye, yo


@triton.jit
def _rope_quant_store(QN, QR, KN, KR, POS, CS, LOC, QO, KV,
                      sqn_t, sqn_h, sqr_t, sqr_h, skn_t, skr_t, sqo_t, sqo_h, skv, scs,
                      H: tl.constexpr, NOPE: tl.constexpr, HALF: tl.constexpr, FMA: tl.constexpr,
                      USE_PDL: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    if USE_PDL:
        t = _pdl_wait(t)
        t = _pdl_trigger(t)
    t64 = t.to(tl.int64)
    pos = tl.load(POS + t).to(tl.int64)
    p = tl.arange(0, HALF)
    cos = tl.load(CS + pos * scs + p).to(tl.float32)
    sin = tl.load(CS + pos * scs + HALF + p).to(tl.float32)
    n = tl.arange(0, NOPE)
    if h < H:
        out = QO + t64 * sqo_t + h * sqo_h
        tl.store(out + n, _e4m3(tl.load(QN + t64 * sqn_t + h * sqn_h + n).to(tl.float32)))
        rope = QR + t64 * sqr_t + h * sqr_h
        ye, yo = _rope_pair(tl.load(rope + 2 * p).to(tl.float32), tl.load(rope + 2 * p + 1).to(tl.float32),
                            cos, sin, FMA)
        tl.store(out + NOPE + 2 * p, _e4m3(ye))
        tl.store(out + NOPE + 2 * p + 1, _e4m3(yo))
    else:
        loc = tl.load(LOC + t).to(tl.int64)
        if loc != 0:
            dst = KV + loc * skv
            tl.store(dst + n, _e4m3(tl.load(KN + t64 * skn_t + n).to(tl.float32)))
            rope = KR + t64 * skr_t
            ye, yo = _rope_pair(tl.load(rope + 2 * p).to(tl.float32), tl.load(rope + 2 * p + 1).to(tl.float32),
                                cos, sin, FMA)
            tl.store(dst + NOPE + 2 * p, _e4m3(ye))
            tl.store(dst + NOPE + 2 * p + 1, _e4m3(yo))


@triton.jit
def _rope_quant_store_wide(QN, QR, KN, KR, POS, CS, LOC, QO, KV,
                           sqn_t, sqn_h, sqr_t, sqr_h, skn_t, skr_t, sqo_t, sqo_h, skv, scs,
                           H: tl.constexpr, HB: tl.constexpr, NOPE: tl.constexpr, HALF: tl.constexpr,
                           FMA: tl.constexpr, USE_PDL: tl.constexpr):
    t = tl.program_id(0)
    b = tl.program_id(1)
    if USE_PDL:
        t = _pdl_wait(t)
        t = _pdl_trigger(t)
    t64 = t.to(tl.int64)
    n = tl.arange(0, NOPE)
    r = tl.arange(0, 2 * HALF)
    p = tl.arange(0, HALF)
    if b * HB < H:
        h = (b * HB + tl.arange(0, HB)).to(tl.int64)
        nope = tl.load(QN + t64 * sqn_t + h[:, None] * sqn_h + n[None, :])
        x = tl.load(QR + t64 * sqr_t + h[:, None] * sqr_h + r[None, :]).to(tl.float32)
        pos = tl.load(POS + t).to(tl.int64)
        cos = tl.load(CS + pos * scs + p).to(tl.float32)
        sin = tl.load(CS + pos * scs + HALF + p).to(tl.float32)
        out = QO + t64 * sqo_t + h[:, None] * sqo_h
        tl.store(out + n[None, :], _e4m3(nope.to(tl.float32)))
        xe, xo = tl.split(tl.reshape(x, (HB, HALF, 2)))
        ye, yo = _rope_pair(xe, xo, cos[None, :], sin[None, :], FMA)
        tl.store(out + NOPE + r[None, :], _e4m3(tl.reshape(tl.join(ye, yo), (HB, 2 * HALF))))
    else:
        loc = tl.load(LOC + t).to(tl.int64)
        k_nope = tl.load(KN + t64 * skn_t + n)
        k_rope = tl.load(KR + t64 * skr_t + r).to(tl.float32)
        k_pos = tl.load(POS + t).to(tl.int64)
        k_cos = tl.load(CS + k_pos * scs + p).to(tl.float32)
        k_sin = tl.load(CS + k_pos * scs + HALF + p).to(tl.float32)
        dst = KV + loc * skv
        keep = loc != 0
        tl.store(dst + n, _e4m3(k_nope.to(tl.float32)), mask=keep)
        ke, ko = tl.split(tl.reshape(k_rope, (HALF, 2)))
        ke, ko = _rope_pair(ke, ko, k_cos, k_sin, FMA)
        tl.store(dst + NOPE + r, _e4m3(tl.reshape(tl.join(ke, ko), (2 * HALF,))), mask=keep)


def rope_quant_store(q_nope, q_rope, k_nope, k_rope, positions, cos_sin_cache, loc, kv_rows, fma=None, pdl=True,
                     wide=None):
    tokens, heads, nope = q_nope.shape
    rope = q_rope.shape[-1]
    q_out = torch.empty((tokens, heads, nope + rope), dtype=torch.float8_e4m3fn, device=q_nope.device)
    fma = FMA_FORM if fma is None else fma
    if tokens > FUSED_MAX_ROWS if wide is None else wide:
        _rope_quant_store_wide[(tokens, heads // WIDE_HB + 1)](
            q_nope, q_rope, k_nope, k_rope, positions, cos_sin_cache, loc, q_out, kv_rows,
            q_nope.stride(0), q_nope.stride(1), q_rope.stride(0), q_rope.stride(1), k_nope.stride(0),
            k_rope.stride(0), q_out.stride(0), q_out.stride(1), kv_rows.stride(0), cos_sin_cache.stride(0),
            H=heads, HB=WIDE_HB, NOPE=nope, HALF=rope // 2, FMA=fma, USE_PDL=pdl,
            num_warps=WIDE_WARPS, launch_pdl=pdl, enable_fp_fusion=False)
        return q_out
    _rope_quant_store[(tokens, heads + 1)](
        q_nope, q_rope, k_nope, k_rope, positions, cos_sin_cache, loc, q_out, kv_rows,
        q_nope.stride(0), q_nope.stride(1), q_rope.stride(0), q_rope.stride(1), k_nope.stride(0), k_rope.stride(0),
        q_out.stride(0), q_out.stride(1), kv_rows.stride(0), cos_sin_cache.stride(0),
        H=heads, NOPE=nope, HALF=rope // 2, FMA=fma, USE_PDL=pdl,
        num_warps=2, launch_pdl=pdl, enable_fp_fusion=False)
    return q_out


def _served(layer, q, k, forward_batch, save_kv_cache, kwargs, backend):
    from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )
    from sglang.srt.runtime_context import get_parallel

    q_rope, k_rope, cache = kwargs.get("q_rope"), kwargs.get("k_rope"), kwargs.get("cos_sin_cache")
    if (not forward_batch.forward_mode.is_target_verify() or not save_kv_cache or k is None
            or set(kwargs) - _KWARGS or kwargs.get("is_neox") is not False
            or kwargs.get("llama_4_scaling") is not None or q_rope is None or k_rope is None or cache is None
            or type(backend).__name__ != "DeepseekSparseAttnBackend" or backend.dsa_decode_impl != "trtllm"
            or backend.use_mha or backend.kv_cache_dtype != torch.float8_e4m3fn or not backend.use_fused_topk
            or backend.qk_rope_head_dim != q_rope.shape[-1] or layer.is_cross_attention
            or get_parallel().dcp_enabled or dsa_use_prefill_cp(forward_batch)
            or q.dim() != 3 or not 0 < q.shape[0] <= WIDE_MAX
            or (q.shape[0] > FUSED_MAX_ROWS and q.shape[1] % WIDE_HB)
            or q.dtype != torch.bfloat16 or q.stride(-1) != 1 or q_rope.stride(-1) != 1
            or k.stride(-1) != 1 or k_rope.stride(-1) != 1 or q_rope.shape[-1] % 2
            or forward_batch.out_cache_loc is None or forward_batch.out_cache_loc.shape[0] != q.shape[0]
            or get_tc_piecewise_forward_context() is not None):
        return False
    return backend._resolve_kpool_tail_backend(kwargs.get("topk_indices"), backend.dsa_decode_impl) == "trtllm"



def install(attn_mqa, join=None) -> None:
    stock = attn_mqa.forward

    def forward(q, k, v, forward_batch, save_kv_cache=True, **kwargs):
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        absorb = dqab.take(attn_mqa, q)
        if not _served(attn_mqa, q, k, forward_batch, save_kv_cache, kwargs, backend):
            _announce("skipped", f"mode {forward_batch.forward_mode} backend {type(backend).__name__}")
            if absorb is not None:
                dqab.materialize(absorb)
            if join is not None:
                join()
            return stock(q, k, v, forward_batch, save_kv_cache, **kwargs)
        _announce("served" if q.shape[0] <= FUSED_MAX_ROWS else "served-wide",
                  f"{q.shape[0]} rows x {q.shape[1]} heads, fma form {FMA_FORM}")
        return _verify(attn_mqa, q, k, forward_batch, kwargs, backend, join, absorb)

    attn_mqa.forward = forward


def _verify(layer, q, k, forward_batch, kwargs, backend, join=None, absorb=None):
    import flashinfer.decode
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.trtllm_mla_backend import grow_multi_ctas_kv_counter_buffer_if_needed

    topk_indices = kwargs.get("topk_indices")
    backend._check_kpool_tail_backend(topk_indices, "trtllm", "decode")
    metadata = backend.forward_metadata
    rope = backend.qk_rope_head_dim
    k_cache = backend.token_to_kv_pool.get_key_buffer(layer.layer_id)
    k_nope, k_rope = k.reshape(-1, layer.v_head_dim), kwargs["k_rope"].reshape(-1, rope)
    kv_rows = k_cache.view(k_cache.shape[0], -1)
    if absorb is not None and dqab.usable(kwargs["q_rope"], k_nope, k_rope, forward_batch.positions,
                                          kwargs["cos_sin_cache"], forward_batch.out_cache_loc):
        q_out = dqab.decode(layer, absorb, kwargs["q_rope"], k_nope, k_rope, forward_batch.positions,
                            kwargs["cos_sin_cache"], forward_batch.out_cache_loc, kv_rows)
    else:
        if absorb is not None:
            dqab.materialize(absorb)
        q_out = rope_quant_store(q, kwargs["q_rope"], k_nope, k_rope, forward_batch.positions,
                                 kwargs["cos_sin_cache"], forward_batch.out_cache_loc, kv_rows)
    if join is not None:
        join()
    kv_cache = k_cache.view(-1, backend.real_page_size, backend.kv_cache_dim).unsqueeze(1)
    q_all = q_out.view(-1, layer.tp_q_head_num, layer.head_dim)
    if topk_indices is not None:
        topk_indices = backend._pad_topk_indices(topk_indices, q_out.shape[0])
    page_table_1, sparse_mla_top_k = backend._pad_trtllm_sparse_page_table(
        backend._get_fused_topk_page_table(topk_indices))
    k_scale = layer.k_scale_float if getattr(layer, "k_scale_float", None) is not None else 1.0
    batch_size = page_table_1.shape[0]
    _, num_heads, head_dim = q_all.shape
    backend._multi_ctas_kv_counter_buffer = grow_multi_ctas_kv_counter_buffer_if_needed(
        backend._multi_ctas_kv_counter_buffer, torch.device(backend.device), backend.num_q_heads, batch_size)
    if (envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get() is None
            and (q.shape[0] <= FUSED_MAX_ROWS or DMLA_WIDE)):
        from glm53_baseline import dmla
        out = dmla.decode(q_all.view(batch_size, 1, num_heads, head_dim),
                          k_cache.view(-1, backend.kv_cache_dim), page_table_1,
                          metadata.dsa_cache_seqlens_int32, 1.0 * k_scale * layer.scaling)
        if out is not None:
            return out
    return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=q_all.view(batch_size, 1, num_heads, head_dim),
        kv_cache=kv_cache.view(-1, 1, backend.real_page_size, backend.kv_cache_dim),
        workspace_buffer=backend.workspace_buffer,
        qk_nope_head_dim=backend.qk_nope_head_dim,
        kv_lora_rank=backend.kv_lora_rank,
        qk_rope_head_dim=rope,
        block_tables=page_table_1.unsqueeze(1),
        seq_lens=metadata.dsa_cache_seqlens_int32,
        max_seq_len=metadata.max_seq_len_k,
        sparse_mla_top_k=sparse_mla_top_k,
        bmm1_scale=1.0 * k_scale * layer.scaling,
        backend="trtllm-gen",
        skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
        sparse_mla_top_k_lens=None,
        multi_ctas_kv_counter_buffer=backend._multi_ctas_kv_counter_buffer,
    )
