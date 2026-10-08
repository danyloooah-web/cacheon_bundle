
from __future__ import annotations

import copy
import sys
import types

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from glm53_baseline import lite_attn

_BM, _BN, _BK = 128, 256, 64
_STAGES, _WARPS = 4, 4

_announced: set = set()
_sm_counts: dict = {}
_side_streams: dict = {}


def _announce(tag: str, detail: str) -> None:
    if tag not in _announced:
        _announced.add(tag)
        print(f"CACHEON_QABSORB {tag} {detail}", file=sys.stderr, flush=True)


@triton.jit(do_not_specialize=["T"])
def _absorb_kernel(a_desc, w_desc, o_desc, T, H: tl.constexpr, K: tl.constexpr, KR: tl.constexpr,
                   N: tl.constexpr, NR: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr, NUM_SMS: tl.constexpr):
    tiles_n: tl.constexpr = N // BN
    k_tiles: tl.constexpr = K // BK
    per_head = tl.cdiv(T, BM) * tiles_n
    for tile in tl.range(tl.program_id(0), per_head * H, NUM_SMS, flatten=True, warp_specialize=True):
        h = tile // per_head
        r = tile % per_head
        m0 = (r // tiles_n) * BM
        n0 = (r % tiles_n) * BN
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for ki in range(k_tiles):
            a = a_desc.load([m0, h * KR + ki * BK])
            w = w_desc.load([h * N + n0, ki * BK])
            acc = tl.dot(a, w.T, acc)
        acc = tl.permute(tl.reshape(acc, (BM, 2, BN // 2)), (0, 2, 1))
        lo, hi = tl.split(acc)
        o_desc.store([m0, h * NR + n0], lo.to(tl.bfloat16).to(tl.float8e4nv))
        o_desc.store([m0, h * NR + n0 + BN // 2], hi.to(tl.bfloat16).to(tl.float8e4nv))


@triton.jit(do_not_specialize=["pos"])
def _rope_kernel(q_rope, k_rope, k_nope, q_out, k_rope_out, k_nope_out, cos_sin, pos,
                 q_rope_s0, q_rope_s1, k_rope_s0, k_nope_s0, cs_s0,
                 H: tl.constexpr, R: tl.constexpr, N: tl.constexpr, NR: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    p = tl.load(pos + t).to(tl.int64)
    half: tl.constexpr = R // 2
    j = tl.arange(0, half)
    cos = tl.load(cos_sin + p * cs_s0 + j)[None, :]
    sin = tl.load(cos_sin + p * cs_s0 + half + j)[None, :]
    heads = tl.arange(0, H)[:, None]
    cols = tl.arange(0, R)[None, :]
    x = tl.load(q_rope + t * q_rope_s0 + heads * q_rope_s1 + cols).to(tl.float32)
    even, odd = tl.split(tl.reshape(x, (H, half, 2)))
    q = tl.reshape(tl.join(tl.fma(even, cos, -(odd * sin)), tl.fma(odd, cos, even * sin)), (H, R))
    tl.store(q_out + t * (H * NR) + heads * NR + N + cols, q.to(tl.float8e4nv))
    kx = tl.load(k_rope + t * k_rope_s0 + cols).to(tl.float32)
    k_even, k_odd = tl.split(tl.reshape(kx, (1, half, 2)))
    k = tl.reshape(tl.join(tl.fma(k_even, cos, -(k_odd * sin)), tl.fma(k_odd, cos, k_even * sin)),
                   (1, R))
    tl.store(k_rope_out + t * R + cols, k.to(tl.float8e4nv))
    ncols = tl.arange(0, N)
    kn = tl.load(k_nope + t * k_nope_s0 + ncols).to(tl.float32)
    tl.store(k_nope_out + t * N + ncols, kn.to(tl.float8e4nv))


def _sms(device) -> int:
    index = torch.device(device).index
    if index not in _sm_counts:
        _sm_counts[index] = torch.cuda.get_device_properties(device).multi_processor_count
    return _sm_counts[index]


def fused_quantize(q_nope, w_t, q_rope, k_nope, k_rope, pos_ids, cos_sin_cache):
    T, H, K = q_nope.shape
    N, R = w_t.shape[1], q_rope.shape[2]
    inner = (q_nope.stride(2), q_rope.stride(2), k_nope.stride(-1), k_rope.stride(-1))
    if cos_sin_cache.dtype != torch.float32 or inner != (1, 1, 1, 1) or not w_t.is_contiguous():
        raise RuntimeError(f"born-fp8 q: cos_sin_cache {cos_sin_cache.dtype}, inner strides {inner}, "
                           f"w contiguous={w_t.is_contiguous()}")
    fp8 = torch.float8_e4m3fn
    q_out = q_rope.new_empty((T, H, N + R), dtype=fp8)
    k_nope_out = k_nope.new_empty(k_nope.shape, dtype=fp8)
    k_rope_out = k_rope.new_empty(k_rope.shape, dtype=fp8)
    if T == 0:
        return q_out, k_nope_out, k_rope_out
    kr = q_nope.stride(1)
    a_desc = TensorDescriptor(q_nope, [T, H * kr], [q_nope.stride(0), 1], [_BM, _BK])
    w_desc = TensorDescriptor(w_t, [H * N, K], [K, 1], [_BN, _BK])
    o_desc = TensorDescriptor(q_out, [T, H * (N + R)], [H * (N + R), 1], [_BM, _BN // 2])
    programs = min(_sms(q_nope.device), H * triton.cdiv(T, _BM) * (N // _BN))
    _absorb_kernel[(programs,)](a_desc, w_desc, o_desc, T, H=H, K=K, KR=kr, N=N, NR=N + R, BM=_BM,
                                BN=_BN, BK=_BK, NUM_SMS=programs, num_warps=_WARPS,
                                num_stages=_STAGES)
    _rope_kernel[(T,)](q_rope, k_rope, k_nope, q_out, k_rope_out, k_nope_out, cos_sin_cache, pos_ids,
                       q_rope.stride(0), q_rope.stride(1), k_rope.stride(0), k_nope.stride(0),
                       cos_sin_cache.stride(0), H=H, R=R, N=N, NR=N + R, num_warps=4)
    return q_out, k_nope_out, k_rope_out


def _static_reason(attn) -> str | None:
    w_kc = getattr(attn, "w_kc", None)
    if w_kc is None or w_kc.dtype != torch.bfloat16 or w_kc.dim() != 3:
        return "w_kc is not a bf16 [H, K, N] tensor"
    H, K, N = w_kc.shape
    if w_kc.stride() != (N * K, 1, K):
        return f"w_kc strides {w_kc.stride()} are not the [H, N, K] K-contiguous layout"
    if (H, K, N) != (attn.num_local_heads, attn.qk_nope_head_dim, attn.kv_lora_rank):
        return f"w_kc shape {(H, K, N)} does not match the attention's heads/dims"
    if K % _BK or N % _BN or attn.qk_rope_head_dim != 64:
        return f"dims K={K} N={N} R={attn.qk_rope_head_dim} are not tiled by this kernel"
    if getattr(attn, "use_deep_gemm_bmm", False):
        return "use_deep_gemm_bmm is set"
    if getattr(attn, "rotary_emb", None) is None or attn.rotary_emb.is_neox_style:
        return "rope is not interleaved (the only mode this kernel implements)"
    if getattr(attn, "_kimi_split_gguf_kv_b", False):
        return "gguf kv_b"
    return None


def _call_reason(attn, forward_batch, backend) -> str | None:
    mode = forward_batch.forward_mode
    if torch.cuda.is_current_stream_capturing():
        return "stream capture in progress"
    if not mode.is_extend() or mode.is_target_verify() or mode.is_draft_extend_v2():
        return f"forward_mode {mode!r} is not a prefill extend"
    if not attn._fuse_rope_for_trtllm_mla(forward_batch):
        return "rope is not deferred to the fp8 quantize"
    if getattr(backend, "dsa_prefill_impl", None) != "trtllm" or getattr(backend, "use_mha", True):
        return (f"backend prefill impl {getattr(backend, 'dsa_prefill_impl', None)!r} "
                f"use_mha={getattr(backend, 'use_mha', None)!r} is not trtllm MLA")
    if getattr(backend, "kv_cache_dtype", None) != torch.float8_e4m3fn:
        return "kv cache is not fp8"
    return None


class _BornPlan:

    def __init__(self, q_nope: torch.Tensor, sentinel: torch.Tensor) -> None:
        self.q_nope = q_nope
        self.q_nope_out_view = sentinel


def rebind_to_copy(replica) -> None:
    method = getattr(replica, "_forward_method", None)
    if isinstance(method, types.MethodType) and method.__self__ is not replica:
        replica._forward_method = types.MethodType(method.__func__, replica)
    method = getattr(replica, "_original_forward_method", None)
    if isinstance(method, types.MethodType) and method.__self__ is not replica:
        replica._original_forward_method = types.MethodType(method.__func__, replica)


_LOGITS_CAP = 2 << 30


class _CappedBudget(dict):

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, min(int(value), _LOGITS_CAP))


def _record(obj, stream) -> None:
    if isinstance(obj, torch.Tensor):
        obj.record_stream(stream)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            _record(item, stream)
    elif isinstance(obj, dict):
        for item in obj.values():
            _record(item, stream)


def install(attn) -> None:
    reason = _static_reason(attn)
    if reason is not None:
        _announce("static_stock", reason)
        return
    from sglang.srt.layers.attention import dsa_backend
    from sglang.srt.model_executor.forward_context import get_attn_backend

    import flashinfer.decode

    lite_attn.capture_stock()
    rebind_to_copy(attn)
    stock_can_fuse = attn._can_fuse_bmm_into_attention
    stock_plan = attn._make_mla_bmm_fusion_plan
    stock_core = attn.forward_absorb_core
    w_kc_t = attn.w_kc.transpose(1, 2)
    armed = []
    overlap = {}
    forks = attn._modules.get("indexer") is not None and not getattr(attn, "skip_topk", False)
    indexer = attn._modules.get("indexer")
    if indexer is not None:
        indexer = copy.copy(indexer)
        rebind_to_copy(indexer)
        indexer._mqa_logits_budget_bytes = _CappedBudget()
        from glm53_baseline import prefill_levers

        prefill_levers.install(indexer)
        attn._modules = dict(attn._modules)
        attn._modules["indexer"] = indexer

    stock_qb = attn.q_b_proj_forward

    def q_b_proj_forward(q):
        if not overlap.get("on"):
            return stock_qb(q)
        current = torch.cuda.current_stream()
        side = _side_streams.get(current.device.index)
        if side is None:
            side = _side_streams[current.device.index] = torch.cuda.Stream(device=current.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            out = stock_qb(q)
        q.record_stream(side)
        overlap["side"] = side
        _announce("qprep_overlap", "q_b and the born-fp8 q on the side stream beside the indexer")
        return out

    def join() -> None:
        side = overlap.pop("side", None)
        if side is not None:
            torch.cuda.current_stream().wait_stream(side)

    def can_fuse(forward_batch):
        armed.clear()
        overlap.clear()
        why = _call_reason(attn, forward_batch, get_attn_backend())
        if why is None:
            armed.append(True)
            overlap["on"] = forks
            return True
        _announce("call_stock", why)
        return stock_can_fuse(forward_batch)

    def make_plan(q, q_nope):
        if not armed:
            return stock_plan(q, q_nope)
        armed.clear()
        sentinel = q.new_full((1,), float("nan")).expand(q.shape[0], attn.num_local_heads,
                                                         attn.kv_lora_rank)
        return _BornPlan(q_nope, sentinel)

    def core(q_pe, k_pe, q_nope_out, k_nope, forward_batch, zero_allocator, positions, topk_indices,
             llama_4_scaling, fusion_plan=None, *rest):
        if not isinstance(fusion_plan, _BornPlan):
            return stock_core(q_pe, k_pe, q_nope_out, k_nope, forward_batch, zero_allocator, positions,
                              topk_indices, llama_4_scaling, fusion_plan, *rest)
        served = []

        def quantize(q, q_rope, k, k_rope, pos_ids, cos_sin_cache, is_neox, kv_lora_rank, rope_dim):
            if q is not fusion_plan.q_nope_out_view or served or is_neox:
                raise RuntimeError(f"born-fp8 q: unexpected quantize call (sentinel="
                                   f"{q is fusion_plan.q_nope_out_view}, repeat={bool(served)}, "
                                   f"is_neox={is_neox})")
            served.append(True)
            side = overlap.get("side")
            if side is None:
                return fused_quantize(fusion_plan.q_nope, w_kc_t, q_rope, k, k_rope, pos_ids, cos_sin_cache)
            with torch.cuda.stream(side):
                result = fused_quantize(fusion_plan.q_nope, w_kc_t, q_rope, k, k_rope, pos_ids, cos_sin_cache)
            _record((k, k_rope, pos_ids), side)
            join()
            _record(result, torch.cuda.current_stream())
            return result

        saved = dsa_backend.mla_quantize_and_rope_for_fp8
        saved_attention = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla
        dsa_backend.mla_quantize_and_rope_for_fp8 = quantize
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla = lite_attn.attention
        lite_attn.current_topk[:] = [topk_indices] if isinstance(topk_indices, torch.Tensor) else []
        from glm53_baseline import prefill_levers

        prefill_levers.lds_arm(forward_batch)
        try:
            out = stock_core(q_pe, k_pe, q_nope_out, k_nope, forward_batch, zero_allocator, positions,
                             topk_indices, llama_4_scaling, None, *rest)
        finally:
            dsa_backend.mla_quantize_and_rope_for_fp8 = saved
            flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla = saved_attention
            lite_attn.current_topk.clear()
            prefill_levers.lds_clear()
            join()
            overlap.clear()
        if not served:
            raise RuntimeError("born-fp8 q: the plan was armed but the DSA backend never quantized it")
        _announce("born", f"rows={q_pe.shape[0]} heads={attn.num_local_heads}")
        return out

    attn._can_fuse_bmm_into_attention = can_fuse
    attn._make_mla_bmm_fusion_plan = make_plan
    attn.forward_absorb_core = core
    attn.q_b_proj_forward = q_b_proj_forward
