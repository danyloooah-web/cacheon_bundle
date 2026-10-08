
import copy
from functools import lru_cache

import torch

from glm53_baseline.nvfp4_contract import prepare_args_from_layer
from glm53_baseline import (dpo, exchange, fp8_decode, fused_add_rmsnorm, fused_tail, handoff, levers,
                            moe_routed_experts, pad_rows, rope_kv, route_topk)
from glm53_baseline import fp8w
from glm53_baseline import qabsorb
from glm53_baseline import dsa_topk
from glm53_baseline import router_tri
from glm53_baseline import sx8
from glm53_baseline import dqab
from glm53_baseline import dfa
from glm53_baseline import dwuv
from glm53_baseline import rz
from glm53_baseline import prefill_levers

_PROLOGUE: dict = {}
_PENDING: dict = {}
_FILL_STREAMS: dict = {}
_MOE: dict = {}
DA_ROUTING = True

_HI_STREAMS: dict = {}
SHARED_TAIL_PRIORITY = True


def _hi_stream(device):
    stream = _HI_STREAMS.get(device.index)
    if stream is None:
        _, greatest = torch.cuda.Stream.priority_range()
        stream = _HI_STREAMS[device.index] = torch.cuda.Stream(device=device, priority=greatest)
    return stream


def _shared_split(replica, x, zero_allocator, alt, current):
    mlp = getattr(replica, "shared_experts", None)
    gate_up_proj = getattr(mlp, "gate_up_proj", None)
    weight = getattr(gate_up_proj, "weight", None)
    plain = (SHARED_TAIL_PRIORITY and mlp is not None and x.shape[0] > 0
             and getattr(replica, "num_fused_shared_experts", 1) == 0
             and not getattr(mlp, "_enable_nvfp4_gemm_swiglu_fusion", False)
             and getattr(mlp, "swiglu_limit", 0) is None
             and isinstance(weight, torch.Tensor) and weight.dtype != torch.uint8)
    if not plain:
        with torch.cuda.stream(alt):
            shared = replica._forward_shared_experts(x, zero_allocator)
        current.wait_stream(alt)
        return shared
    i8 = sx8.shared(mlp, x)
    i8_gate = i8 is not None and x.shape[0] <= sx8.GATE_UP_MAX_ROWS
    hi = _hi_stream(x.device)
    with torch.cuda.stream(alt):
        gate_up = i8.gate_up_act(x) if i8_gate else gate_up_proj(x)[0]
    hi.wait_stream(alt)
    with torch.cuda.stream(hi):
        gate_up.record_stream(hi)
        if i8 is not None:
            shared = i8.act_down(gate_up if i8_gate else mlp.act_fn(gate_up))
        else:
            act = mlp.act_fn(gate_up)
            shared, _ = mlp.down_proj(act)
    current.wait_stream(hi)
    shared.record_stream(current)
    return shared
MOE_WARM_EXPERTS = 20
MOE_WARM_MIN_ROWS = 17
MOE_WARM_MAX_ROWS = 32
WINDOW_B_DELAY_NS = 8000
WINDOW_B_DELAY_DQAB_NS = 4500
DPO_SELF_WARM = {"narrow": 0.40, "wide": 0.0}
ROUTER_WARM_MIN_LOCAL = 5
LATE_B_JOIN_MAX_ROWS = 8
WINDOW_B_TOGETHER = True
B_SEQUENTIAL_WITH_DQAB = True
FILL_BLOCKS_A_NARROW = 64
WINDOW_B_WITH_DWUV = False
_SAID: set = set()


def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        import sys
        print(f"CACHEON_WINDOWS {tag} {detail}", file=sys.stderr, flush=True)


def _dpo_band(rows: int) -> str:
    if dpo.I8 and rows <= dpo.I8_MAX_ROWS:
        return "fused_i8"
    if dpo.FUSED and dpo.FUSED_E4M3 and rows < dpo.FUSED_UMMA_MIN_ROWS:
        return "fused8"
    return "fused" if dpo.FUSED and rows <= dpo.FUSED_MAX_ROWS else "narrow" if rows <= 16 else "wide"
FILL_BLOCKS, FILL_CHUNK, FILL_DEPTH = 32, 65536, 3
FUSE_NEXT_NORM = levers.DECODE_NEXT_NORM
NEXT_NORM_MAX_ROWS = 128


@lru_cache(maxsize=1)
def _l2_fill():
    import l2_fill

    return l2_fill


def _storage_bytes(t: torch.Tensor) -> torch.Tensor:
    storage = t.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(storage, 0, (storage.nbytes(),))


def _fork_fill(weights, current, window: str, blocks: int = FILL_BLOCKS) -> None:
    device = current.device.index
    stream = _FILL_STREAMS.get((device, window))
    if stream is None:
        stream = _FILL_STREAMS[(device, window)] = torch.cuda.Stream(device=device)
    _join_pending(device, current, window)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for w in weights:
            _l2_fill().fill(w, blocks, FILL_CHUNK, FILL_DEPTH)
    _PENDING[(device, window)] = stream


def _warm_next(layer_id: int, current, local_rows: int) -> None:
    weights = None
    if 0 < local_rows <= dfa.MAX_ROWS:
        weights = _PROLOGUE.get((layer_id + 1, "dqab_dfa"))
    if weights is None and 0 < local_rows <= dqab.MAX_ROWS:
        weights = _PROLOGUE.get((layer_id + 1, "dqab"))
    if weights is None:
        weights = _PROLOGUE.get((layer_id + 1, 0 < local_rows <= fp8_decode.MAX_ROWS))
    if weights is not None:
        narrow = 0 < local_rows <= dqab.MAX_ROWS
        _fork_fill(weights, current, "A", FILL_BLOCKS_A_NARROW if narrow else FILL_BLOCKS)


def _decode_band(forward_batch) -> bool:
    mode = getattr(forward_batch, "forward_mode", None)
    return mode is not None and (mode.is_decode_or_idle() or mode.is_target_verify())


def _warm_experts(layer_id: int, current, rows: int) -> None:
    source = _MOE.get(layer_id)
    if source is not None and MOE_WARM_EXPERTS > 0 and MOE_WARM_MIN_ROWS <= rows <= MOE_WARM_MAX_ROWS:
        w13, w13_sf = source()
        _fork_fill((w13[:MOE_WARM_EXPERTS], w13_sf[:MOE_WARM_EXPERTS]), current, "C")


def _warm_attention_tail(weights, current, together: bool = False, delay_ns=None) -> None:
    device = current.device.index
    stream = _FILL_STREAMS.get((device, "B"))
    if stream is None:
        stream = _FILL_STREAMS[(device, "B")] = torch.cuda.Stream(device=device)
    _join_pending(device, current, "B")
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        _l2_fill().delay(WINDOW_B_DELAY_NS if delay_ns is None else delay_ns)
        if together and len(weights) >= 2:
            _l2_fill().fill2(weights[0], weights[1], FILL_BLOCKS, FILL_BLOCKS, FILL_CHUNK, FILL_DEPTH)
            weights = weights[2:]
        for w in weights:
            _l2_fill().fill(w, FILL_BLOCKS, FILL_CHUNK, FILL_DEPTH)
    _PENDING[(device, "B")] = stream


def _join_pending(device: int, current, window: str) -> None:
    stream = _PENDING.pop((device, window), None)
    if stream is not None:
        current.wait_stream(stream)


INDEXER_OVERLAP_MAX_ROWS = 32
_IDX_FORK: dict = {}
_IDX_JOIN: dict = {}
_IDX_STREAMS: dict = {}
_LOGITS_DONE: dict = {}


def _mark_indexer_fork(attn, rows: int) -> None:
    if (getattr(attn, "_cacheon_indexer_overlap", False) and 0 < rows <= INDEXER_OVERLAP_MAX_ROWS
            and torch.cuda.is_current_stream_capturing()):
        event = torch.cuda.Event()
        event.record()
        _IDX_FORK[id(attn)] = event


def _install_indexer_overlap(attn):
    indexer, mqa = attn.indexer, attn.attn_mqa
    stock_indexer = dsa_topk.scoped(indexer.forward)

    def indexer_forward(*args, **kwargs):
        event = _IDX_FORK.pop(id(attn), None)
        batch = kwargs.get("forward_batch")
        if event is None or batch is None or not batch.forward_mode.is_target_verify():
            return stock_indexer(*args, **kwargs)
        device = torch.cuda.current_device()
        stream = _IDX_STREAMS.get(device)
        if stream is None:
            _, greatest = torch.cuda.Stream.priority_range()
            stream = _IDX_STREAMS[device] = torch.cuda.Stream(device=device, priority=greatest)
        stream.wait_event(event)
        with torch.cuda.stream(stream):
            topk = stock_indexer(*args, **kwargs)
        _IDX_JOIN[id(mqa)] = (stream, topk)
        logits_done = _LOGITS_DONE.pop(id(indexer), None)
        if logits_done is not None:
            torch.cuda.current_stream().wait_event(logits_done)
        return topk

    def join():
        pending = _IDX_JOIN.pop(id(mqa), None)
        if pending is not None:
            torch.cuda.current_stream().wait_stream(pending[0])

    indexer.forward = indexer_forward
    attn._cacheon_indexer_overlap = True
    stock_mask = getattr(indexer, "_mask_init_and_local_tokens", None)
    if stock_mask is not None:
        def mask_then_mark(logits, *a, **k):
            out = stock_mask(logits, *a, **k)
            current = torch.cuda.current_stream()
            if torch.cuda.is_current_stream_capturing() and current == _IDX_STREAMS.get(current.device.index):
                event = torch.cuda.Event()
                event.record(current)
                _LOGITS_DONE[id(indexer)] = event
            return out

        indexer._mask_init_and_local_tokens = mask_then_mark
    return join


_EARLY_KW: dict = {}
_KW_STREAMS: dict = {}
_IDX_ALT: dict = {}


def _greatest_stream(table: dict, device) -> torch.cuda.Stream:
    index = torch.device(device).index
    stream = table.get(index)
    if stream is None:
        _, greatest = torch.cuda.Stream.priority_range()
        stream = table[index] = torch.cuda.Stream(device=index, priority=greatest)
    return stream


def _join_early_kw(current) -> None:
    for key in list(_EARLY_KW):
        current.wait_stream(_EARLY_KW.pop(key)[0])


def _install_indexer_early_kw(attn):
    indexer = attn._modules.get("indexer")
    lin = indexer._modules.get("wk_weights_proj") if indexer is not None else None
    if lin is None or getattr(indexer, "alt_stream", None) is None:
        return None
    qabsorb.rebind_to_copy(indexer)
    indexer.alt_stream = _greatest_stream(_IDX_ALT, lin.weight.device)
    stock = lin.forward
    key = id(indexer)

    def forward(x, *args, **kwargs):
        got = _EARLY_KW.pop(key, None)
        if got is not None:
            stream, x0, out = got
            torch.cuda.current_stream().wait_stream(stream)
            if (not args and not kwargs and isinstance(x, torch.Tensor) and x.data_ptr() == x0.data_ptr()
                    and x.shape == x0.shape and x.stride() == x0.stride() and x.dtype == x0.dtype):
                return out
        return stock(x, *args, **kwargs)

    lin.forward = forward

    def fork(x, forward_batch) -> None:
        rows = x.shape[0] if isinstance(x, torch.Tensor) and x.dim() == 2 else 0
        if (not 0 < rows <= INDEXER_OVERLAP_MAX_ROWS or not torch.cuda.is_current_stream_capturing()
                or not forward_batch.forward_mode.is_target_verify()):
            return
        current = torch.cuda.current_stream(x.device)
        stale = _EARLY_KW.pop(key, None)
        if stale is not None:
            current.wait_stream(stale[0])
        stream = _greatest_stream(_KW_STREAMS, x.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            out = stock(x)
        _EARLY_KW[key] = (stream, x, out)

    return fork


def prepare_norm(module):
    def normalize(x, residual=None, post_residual_addition=None, quant_linear=None):
        got = handoff.take(x, residual, module.weight, handoff.ROUNDED)
        if got is not None and post_residual_addition is None and quant_linear is None:
            normed, updated, exact = got
            if exact:
                return normed, updated
            out = torch.empty_like(updated)
            fused_add_rmsnorm.fused_add_rmsnorm(
                updated, None, module.weight, module.variance_epsilon, out, None)
            return out, updated
        if (x.dtype != torch.bfloat16 or x.numel() == 0
                or post_residual_addition is not None or quant_linear is not None
                or getattr(module, "fp32_residual", False)
                or module.variance_size_override is not None
                or module.cast_x_before_out_mul):
            return module.forward(x, residual, post_residual_addition, quant_linear)
        out = torch.empty_like(x)
        updated = torch.empty_like(residual) if residual is not None else None
        fused_add_rmsnorm.fused_add_rmsnorm(
            x, residual, module.weight, module.variance_epsilon, out, updated)
        return out if residual is None else (out, updated)
    return normalize


def _copy_with_norms(module, copies):
    from sglang.srt.layers.layernorm import RMSNorm

    if id(module) in copies:
        return copies[id(module)]
    replica = copy.copy(module)
    copies[id(module)] = replica
    replica._modules = {
        name: _copy_with_norms(child, copies) if child is not None else None
        for name, child in module._modules.items()
    }
    if isinstance(module, RMSNorm):
        replica.forward = prepare_norm(module)
    return replica


def _replace_experts(original, replica, tail, layer_id):
    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    from sglang.srt.layers.moe.topk import TopKOutputFormat
    from sglang.srt.layers.moe.utils import should_skip_post_experts_all_reduce

    prepared = None

    def prepare_once():
        nonlocal prepared
        if prepared is None:
            config = original.topk.topk_config
            prepared = moe_routed_experts.prepare(
                *prepare_args_from_layer(original.experts), config.top_k,
                config.routed_scaling_factor)
        return prepared

    def routed(x, topk_output, pre_quant_input=None):
        if (topk_output.format != TopKOutputFormat.BYPASSED
                or not 1 <= x.shape[0] <= 16384):
            return original.experts.forward_impl(x, topk_output, pre_quant_input)
        config = topk_output.topk_config
        out = torch.empty_like(x)
        moe_routed_experts.fused_routed_experts(
            x, topk_output.router_logits, config.correction_bias, prepare_once(), out)
        if original.experts.reduce_results and original.experts.moe_tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    replica.experts.forward = routed
    replica.experts.forward_impl = routed

    def gemm1_weights():
        moe = prepare_once()
        return moe["w13"], moe["w13_sf"]

    _MOE[layer_id] = gemm1_weights
    replica.experts.supports_deferred_finalize = False

    def decode_mlp(x, zero_allocator):
        current = torch.cuda.current_stream()
        alt = original.alt_stream
        alt.wait_stream(current)
        out = torch.empty_like(x)
        pre = tail.pop("fp4", None)
        zero = tail.pop("zero", None)
        logits = router_tri.slabs(x, replica.gate.weight) if x.shape[0] > moe_routed_experts.DA_MAX_ROWS else None
        rezero = logits is not None
        if logits is None and zero is not None and zero[0] is x:
            logits = rz.logits(replica.gate, x, zero[1])
        if logits is None:
            logits = replica.gate(x, zero_allocator)
        config = original.topk.topk_config
        quantized = pre[1:] if pre is not None and pre[0] is x else None
        da = (moe_routed_experts.decode_deferred_da(prepare_once(), x, logits, config.correction_bias,
                                                    config.routed_scaling_factor, out, quantized)
              if DA_ROUTING else None)
        if da is not None:
            gemm2, mapping, weights = da
        else:
            topk_ids, topk_weights = route_topk.route(logits, config.correction_bias, config.routed_scaling_factor,
                                                      rezero=rezero)
            gemm2, mapping, weights = moe_routed_experts.decode_deferred_routed(
                prepare_once(), x, topk_ids, topk_weights, out, quantized)
        _warm_next(layer_id, current, x.shape[0] // tail["world"])
        shared = _shared_split(replica, x, zero_allocator, alt, current)
        ring = tail["ring"](x.shape[0], x.shape[1]) if tail["scatters"]() else None
        residual = tail.pop("residual", None)
        norm = (handoff.next_norm(layer_id, tail["num_layers"])
                if (FUSE_NEXT_NORM or x.shape[0] <= NEXT_NORM_MAX_ROWS) and ring is not None and ring.norm_ok
                else None)
        if norm is not None and residual is not None:
            local = tail["local_buffer"]()
            if (residual.shape == local.shape and residual.dtype == torch.bfloat16
                    and residual.is_contiguous() and norm[0].is_contiguous()):
                normed = torch.empty_like(local)
                updated = torch.empty_like(local)
                fused_tail.finalize_shared_reduce_scatter_norm(
                    ring, gemm2, mapping, weights, shared, local, residual, norm[0], norm[1],
                    normed, updated, release=not norm[2])
                tail["scattered"] = True
                tail["handoff"] = (local, normed, updated, norm[0])
                return local
        if ring is not None:
            local = tail["local_buffer"]()
            fused_tail.finalize_shared_reduce_scatter(ring, gemm2, mapping, weights, shared, local)
            tail["scattered"] = True
            return local
        moe_routed_experts.decode_finalize_shared(gemm2, mapping, weights, shared, out)
        return out

    def decodes(x) -> bool:
        return (1 <= x.shape[0] <= moe_routed_experts.DECODE_MAX_TOKENS
                and original.alt_stream is not None
                and original.num_fused_shared_experts == 0
                and hasattr(original, "shared_experts")
                and not original.experts.reduce_results
                and getattr(original.experts, "use_flashinfer_trtllm_moe", False)
                and should_skip_post_experts_all_reduce(is_tp_path=True))

    def body(x, *args, **kwargs):
        if decodes(x):
            return decode_mlp(x, args[1] if len(args) > 1
                              else kwargs.get("gemm_output_zero_allocator"))
        if original._can_dual_stream_graph(x):
            return replica.forward_normal_dual_stream(x, *args[1:], **kwargs)
        return type(original).forward(replica, x, *args, **kwargs)

    def mlp(x, *args, **kwargs):
        out = body(x, *args, **kwargs)
        _join_pending(x.device.index, torch.cuda.current_stream(x.device), "C")
        _join_pending(x.device.index, torch.cuda.current_stream(x.device), "B")
        return out

    replica.forward = mlp


class _Deferred:

    def __init__(self, x):
        self.x = x


def prepare(module):
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers import communicator as comm
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if (parallel.tp_size != 4 or parallel.attn_dp_size != 4
            or parallel.attn_tp_size != 1 or module.config.hidden_size != 6144
            or parallel.enable_prefill_cp):
        raise ValueError("this baseline requires the commissioned GLM TP4/DP4 topology")
    replica = _copy_with_norms(module, {})
    prefill_levers.install(getattr(replica.self_attn, "indexer", None))
    if getattr(replica.self_attn, "attn_mqa", None) is not None:
        join = None
        if getattr(replica.self_attn, "indexer", None) is not None and replica.self_attn.skip_topk is False:
            join = _install_indexer_overlap(replica.self_attn)
        rope_kv.install(replica.self_attn.attn_mqa, join)
        dqab.install(replica.self_attn)
        dwuv.install(replica.self_attn)
    q_a_norm = getattr(replica.self_attn, "q_a_layernorm", None)
    fp8 = fp8_decode.install_q_b(replica.self_attn, q_a_norm.weight) if q_a_norm is not None else None
    fp8_q_b, inv_q_b = fp8 if fp8 is not None else (None, None)
    original_comm = module.layer_communicator
    replica.layer_communicator = copy.copy(original_comm)
    communicator = replica.layer_communicator
    communicator._context = copy.copy(original_comm._context)
    communicator.input_layernorm = replica.input_layernorm
    communicator.post_attention_layernorm = replica.post_attention_layernorm
    dfa.install(replica.self_attn)
    communicator.qkv_latent_func = replica.self_attn.prepare_qkv_latent
    early_kw = (_install_indexer_early_kw(replica.self_attn)
                if getattr(replica.self_attn, "_cacheon_indexer_overlap", False) else None)
    if early_kw is not None:
        latent = replica.self_attn.prepare_qkv_latent

        def qkv_latent_with_early_kw(hidden_states, forward_batch):
            early_kw(hidden_states, forward_batch)
            return latent(hidden_states, forward_batch)

        communicator.qkv_latent_func = qkv_latent_with_early_kw
    group = get_tp_group()
    projection = module.self_attn.o_proj
    norm = module.post_attention_layernorm
    prepared_projection = {}
    batch = None
    gather_fn = original_comm._communicate_with_all_reduce_and_layer_norm_fn
    gathers = (getattr(gather_fn, "func", gather_fn)
               is comm.CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual)

    def scatters(forward_batch) -> bool:
        return (communicator._communicate_summable_tensor_pair_fn
                is comm.CommunicateSummableTensorPairFn._scatter_hidden_states
                and communicator.allow_reduce_scatter
                and forward_batch is not None
                and forward_batch.dp_padding_mode.is_max_len()
                and not comm.should_use_dp_reduce_scatterv())

    handoff.register_norm(int(module.layer_id), module.input_layernorm.weight,
                          module.input_layernorm.variance_epsilon)
    tail = {
        "num_layers": int(module.config.num_hidden_layers),
        "world": group.world_size,
        "scatters": lambda: scatters(batch),
        "ring": lambda rows, hidden: fused_tail.state_for(
            rows, hidden, group.device_group, norm.weight.device),
        "local_buffer": lambda: comm.get_local_dp_buffer(group),
        "scattered": False,
    }
    moe_layer = isinstance(module.mlp, DeepseekV2MoE)
    if moe_layer:
        router_tri.prepare_buffer(module.mlp.gate.weight.device)
    if isinstance(module.mlp, DeepseekV2MoE):
        sx8.install(replica.mlp, int(module.layer_id))
        _replace_experts(module.mlp, replica.mlp, tail, int(module.layer_id))
    attn = module.self_attn
    w_vc = getattr(attn, "w_vc", None)
    tail_weights = {"w_uv": _storage_bytes(w_vc) if w_vc is not None and w_vc.is_cuda else None}
    gate = getattr(module.mlp, "gate", None) if moe_layer else None
    gate_w = getattr(gate, "weight", None)
    if (gate_w is not None and gate_w.is_cuda and gate_w.dtype == torch.bfloat16 and gate_w.is_contiguous()
            and gate_w.data_ptr() % 16 == 0):
        tail_weights["gate"] = gate_w
    width = projection.weight.shape[0] // group.world_size
    shard = projection.weight[group.rank_in_group * width:(group.rank_in_group + 1) * width]
    if dpo.I8 and shard.is_contiguous():
        tail_weights[("dpo_rest", "fused_i8")] = dpo.int8_copy(shard)[0]
    if shard.is_contiguous():
        for band, share in DPO_SELF_WARM.items():
            tail_weights[("dpo_rest", band)] = shard[int(shard.shape[0] * share):]
    if dpo.FUSED_E4M3 and projection.weight.dtype == torch.bfloat16 and projection.weight.is_contiguous():
        w8 = fp8w.quantized(projection.weight)[0]
        shard8 = w8[group.rank_in_group * width:(group.rank_in_group + 1) * width]
        if shard8.is_contiguous():
            tail_weights[("dpo_rest", "fused8")] = shard8
    q_a_norm = getattr(replica.self_attn, "q_a_layernorm", None)
    if q_a_norm is not None:
        normalize = q_a_norm.forward
        fused_fp8 = (fp8_q_b is not None and not getattr(q_a_norm, "fp32_residual", False)
                     and q_a_norm.variance_size_override is None and not q_a_norm.cast_x_before_out_mul)

        def push_ring(rows):
            if not (dwuv.PUSH and moe_layer and gathers and batch is not None and torch.cuda.is_current_stream_capturing()
                    and (batch.forward_mode.is_decode_or_idle() or batch.forward_mode.is_target_verify())
                    and batch.dp_padding_mode.is_max_len() and not batch.can_run_tbo
                    and 0 < rows <= min(dwuv.MAX_ROWS, dpo.I8_MAX_ROWS) and dpo.I8 and dpo.fuses_pad(rows)
                    and projection.weight.dtype == torch.bfloat16 and projection.bias is None
                    and dwuv.fragments(replica.self_attn) is not None):
                return None
            loc = batch.out_cache_loc
            if (loc is None or not loc.is_cuda or loc.dtype != torch.int64 or loc.dim() != 1 or loc.shape[0] != rows
                    or not loc.is_contiguous()):
                return None
            ring = dpo.fused_ring(group.device_group, projection.weight)
            return None if ring is None else (ring[0], ring[1], loc)

        def q_a_norm_then_warm(x, *args, **kwargs):
            rows = x.shape[0] if x.dim() == 2 else 0
            dwuv.offer_push(replica.self_attn, push_ring(rows))
            if (fused_fp8 and not args and not kwargs and 0 < rows <= fp8_decode.MAX_ROWS
                    and x.dtype == torch.bfloat16):
                out = torch.empty_like(x)
                x8 = torch.empty(out.shape, dtype=torch.float8_e4m3fn, device=x.device)
                fused_add_rmsnorm.fused_add_rmsnorm(x, None, q_a_norm.weight, q_a_norm.variance_epsilon, out, None,
                                                    out_fp8=x8, inv_fp8=inv_q_b)
                fp8_decode.stash(replica.self_attn, out, x8)
            else:
                out = normalize(x, *args, **kwargs)
            _mark_indexer_fork(replica.self_attn, rows)
            if 0 < rows <= 64 and tail_weights["w_uv"] is not None:
                weights = [tail_weights["w_uv"]]
                wuv_frag = dwuv.fragments(replica.self_attn) if rows <= dwuv.MAX_ROWS else None
                if wuv_frag is not None:
                    weights = [wuv_frag]
                if rows <= dpo._MAX_ROWS and ("dpo_rest", _dpo_band(rows)) in tail_weights:
                    weights.append(tail_weights[("dpo_rest", _dpo_band(rows))])
                if wuv_frag is not None and not WINDOW_B_WITH_DWUV and _dpo_band(rows) == "fused_i8":
                    weights = []
                    _say("b_none", f"{rows} local rows: no W_UV / o_proj fill beside the sparse MLA")
                if rows >= ROUTER_WARM_MIN_LOCAL and "gate" in tail_weights:
                    weights.append(tail_weights["gate"])
                served = rows <= dqab.MAX_ROWS and dqab.fragments(replica.self_attn) is not None
                sequential = B_SEQUENTIAL_WITH_DQAB and served
                if sequential:
                    _say("b_sequential", f"{rows} local rows, {len(weights)} tensors")
                if weights:
                    _warm_attention_tail(weights, torch.cuda.current_stream(x.device),
                                         together=WINDOW_B_TOGETHER and rows <= LATE_B_JOIN_MAX_ROWS and not sequential,
                                         delay_ns=WINDOW_B_DELAY_DQAB_NS if served else None)
            return out

        q_a_norm.forward = q_a_norm_then_warm
    fused_a = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
    head = ([fused_a.weight] if fused_a is not None
            else [attn.q_a_proj.weight, attn.kv_a_proj_with_mqa.weight])
    w_kc = getattr(attn, "w_kc", None)
    rest = ([_storage_bytes(w_kc)] if w_kc is not None and w_kc.is_cuda and w_kc.dtype == torch.bfloat16 else [])
    variants = {False: head + [attn.q_b_proj.weight] + rest,
                True: head + [fp8_q_b if fp8_q_b is not None else attn.q_b_proj.weight] + rest}
    if all(w.is_cuda and w.is_contiguous() and w.data_ptr() % 16 == 0 for ws in variants.values() for w in ws):
        for e4m3, weights in variants.items():
            _PROLOGUE[(int(module.layer_id), e4m3)] = tuple(weights)
        fragments = dqab.fragments(replica.self_attn)
        if rest and fragments is not None and fragments.data_ptr() % 16 == 0:
            a_frag = dfa.fragments(replica.self_attn)
            first = (a_frag,) if a_frag is not None and a_frag.data_ptr() % 16 == 0 and fused_a is not None else None
            _PROLOGUE[(int(module.layer_id), "dqab")] = tuple(variants[True][:-1]) + (fragments,)
            if first is not None:
                _PROLOGUE[(int(module.layer_id), "dqab_dfa")] = first + tuple(variants[True][1:-1]) + (fragments,)

    def project(x, *args, **kwargs):
        _join_pending(x.device.index, torch.cuda.current_stream(x.device), "A")
        if not (moe_layer and 0 < x.shape[0] <= LATE_B_JOIN_MAX_ROWS and _decode_band(batch)):
            _join_pending(x.device.index, torch.cuda.current_stream(x.device), "B")
        _join_pending(x.device.index, torch.cuda.current_stream(x.device), "C")
        _join_early_kw(torch.cuda.current_stream(x.device))
        if (gathers and (batch.forward_mode.is_decode_or_idle()
                         or batch.forward_mode.is_target_verify())
                and batch.dp_padding_mode.is_max_len() and not batch.can_run_tbo
                and 0 < x.shape[0] <= dpo._MAX_ROWS and x.dtype == torch.bfloat16
                and projection.weight.dtype == x.dtype
                and x.ndim == 2 and x.stride(1) == 1
                and projection.weight.shape[1] == x.shape[1]
                and projection.bias is None):
            return _Deferred(x), None
        return projection.forward(x, *args, **kwargs)

    replica.self_attn.o_proj.forward = project

    def prepare_mlp(hidden_states, residual, forward_batch, cache=None):
        if isinstance(hidden_states, _Deferred):
            x = hidden_states.x
            experts = getattr(module.mlp, "experts", None)
            method = getattr(experts, "quant_method", None)
            quantized = (method is not None and method.enable_flashinfer_trtllm_moe
                         and not method.quant_config.use_per_token_activation
                         and not module.mlp._can_dual_stream_graph(x))
            scale = (experts.w13_input_scale_quant if quantized
                     else torch.empty(0, device=x.device, dtype=torch.float32))
            if quantized not in prepared_projection:
                prepared_projection[quantized] = dpo.prepare(
                    projection.weight, norm.weight, norm.variance_epsilon, scale)
            rows, hidden = x.shape[0] * group.world_size, norm.weight.numel()
            out = torch.empty((rows, hidden), dtype=x.dtype, device=x.device)
            updated = torch.empty_like(residual)
            packed = torch.empty((rows, hidden // 2) if quantized else (0,),
                                 dtype=torch.uint8, device=x.device)
            scales = torch.empty((rows, hidden // 16) if quantized else (0,),
                                 dtype=torch.uint8, device=x.device)
            if moe_layer:
                fuse_pad = (dpo.fuses_pad(x.shape[0])
                            and pad_rows._matches(x, forward_batch.out_cache_loc)
                            and pad_rows._matches(residual, forward_batch.out_cache_loc))
                if not fuse_pad:
                    pad_rows.zero_padding(x, residual, forward_batch.out_cache_loc)
            else:
                fuse_pad = False
            pushed = dwuv.take_pushed(replica.self_attn, x)
            if pushed and not fuse_pad:
                raise RuntimeError("rows were sent to the peers for a scatter that does not zero its padding rows")
            zero = dpo.project_gather_norm(x, residual, prepared_projection[quantized],
                                          out, updated, packed, scales, group.device_group,
                                          loc=forward_batch.out_cache_loc if fuse_pad else None, xpushed=pushed,
                                          early_residual=pushed)
            if moe_layer and zero is not None:
                tail["zero"] = (out, zero)
            _warm_experts(int(module.layer_id), torch.cuda.current_stream(x.device), rows)
            tail["residual"] = updated
            if quantized:
                tail["fp4"] = (out, packed, scales.view(torch.float8_e4m3fn))
            return out, updated
        if (gathers and forward_batch.dp_padding_mode.is_max_len()
                and not comm.get_attn_tp_context().input_scattered):
            if hidden_states.shape[0]:
                if moe_layer and _decode_band(forward_batch):
                    pad_rows.zero_padding(hidden_states, residual, forward_batch.out_cache_loc)
                with comm.use_symmetric_memory(group, disabled=not comm.is_allocation_symmetric()):
                    hidden_states, residual = communicator.post_attention_layernorm(
                        hidden_states, residual)
            out = comm.get_global_dp_buffer(group)
            exchange.all_gather_into_tensor(hidden_states, out, group.device_group)
            if _decode_band(forward_batch):
                _warm_experts(int(module.layer_id), torch.cuda.current_stream(out.device), out.shape[0])
            tail["residual"] = residual
            return out, residual
        tail.pop("residual", None)
        return type(original_comm).prepare_mlp(
            communicator, hidden_states, residual, forward_batch, cache)

    def postprocess(hidden_states, residual, forward_batch):
        if tail["scattered"]:
            tail["scattered"] = False
            passed = tail.pop("handoff", None)
            if passed is not None and passed[0] is hidden_states:
                local, normed, updated, weight = passed
                handoff.publish(local, residual, normed, updated, weight, handoff.ROUNDED)
            return hidden_states, residual
        if scatters(forward_batch):
            out = comm.get_local_dp_buffer(group)
            exchange.reduce_scatter_tensor(hidden_states, out, group.device_group)
            return out, residual
        return type(original_comm).postprocess_layer(
            communicator, hidden_states, residual, forward_batch)

    communicator.prepare_mlp = prepare_mlp
    communicator.postprocess_layer = postprocess

    def layer(*args, **kwargs):
        nonlocal batch
        batch = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
        try:
            return type(module).forward(replica, *args, **kwargs)
        finally:
            batch = None

    return layer


def forward(prepared, *args, **kwargs):
    return prepared(*args, **kwargs)
