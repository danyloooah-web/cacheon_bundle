
from __future__ import annotations

import copy
import sys

import torch

from glm53_baseline.nvfp4_contract import prepare_args_from_layer
from glm53_baseline import layer as champion_layer
from glm53_baseline import (fp8w, fused_add_rmsnorm, handoff, levers, moe_routed_experts, prefill_gather,
                            prefill_tail, qabsorb)

_WORLD = 4
_HIDDEN, _EXPERTS, _TOP_K = 6144, 256, 8
_SF_BLOCK = 16
_ROUTED_SCALING = 2.5
_MIN_CAP, _MAX_CAP = 3072 + 1, 4096

FUSE_NEXT_NORM = levers.PREFILL_NEXT_NORM
SHARED_OVERLAP = levers.SHARED_OVERLAP
_SIDE_STREAMS: dict = {}

_announced: set = set()


def _announce(tag: str, detail: str) -> None:
    if tag not in _announced:
        _announced.add(tag)
        print(f"CACHEON_FP4_GATHER {tag} {detail}", file=sys.stderr, flush=True)


def prepare_norm(module):
    handoff.register_norm("final", module.weight, module.variance_epsilon)
    return champion_layer.prepare_norm(module)


class _HandoffNorm:

    def __init__(self, norm) -> None:
        self._norm = norm
        self.weight = norm.weight
        self.variance_epsilon = norm.variance_epsilon

    def __call__(self, x, residual=None, post_residual_addition=None):
        got = handoff.take(x, residual, self._norm.weight, handoff.FP32) if post_residual_addition is None else None
        if got is None:
            if post_residual_addition is None:
                return self._norm(x) if residual is None else self._norm(x, residual)
            return self._norm(x, residual, post_residual_addition)
        normed, updated, exact = got
        if exact:
            return normed, updated
        return self._norm(updated.clone()), updated


def _replicate_shared(shared, group) -> dict | None:
    gate_up, down = shared.gate_up_proj, shared.down_proj
    if gate_up.weight.dtype != torch.bfloat16 or down.weight.dtype != torch.bfloat16:
        return None
    if getattr(gate_up, "bias", None) is not None or getattr(down, "bias", None) is not None:
        return None
    world = torch.distributed.get_world_size(group)
    two_i_tp, hidden = gate_up.weight.shape
    if two_i_tp % 2 or hidden != _HIDDEN or down.weight.shape != (hidden, two_i_tp // 2):
        return None
    i_tp = two_i_tp // 2

    def _gather(local):
        parts = [torch.empty_like(local) for _ in range(world)]
        torch.distributed.all_gather(parts, local.contiguous(), group=group)
        return parts

    gu_parts = _gather(gate_up.weight.data)
    down_parts = _gather(down.weight.data)
    merged = torch.cat([torch.cat([p[:i_tp] for p in gu_parts], dim=0),
                        torch.cat([p[i_tp:] for p in gu_parts], dim=0)], dim=0)
    return {"merged": merged.contiguous(), "down": torch.cat(down_parts, dim=1).contiguous()}


def _shared_forward(full: dict, x: torch.Tensor) -> torch.Tensor:
    from sglang.srt.layers.activation import silu_and_mul

    fused = x @ full["merged"].T
    return silu_and_mul(fused) @ full["down"].T


def prepare(module):
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE
    from sglang.srt.runtime_context import get_forward

    plan = {"layer": champion_layer.prepare(module), "module": module, "mode": "stock"}
    dense = isinstance(module.mlp, DeepseekV2MLP)
    if not dense and not isinstance(module.mlp, DeepseekV2MoE):
        plan["why"] = "mlp is neither DeepseekV2MoE nor DeepseekV2MLP"
        return plan
    if getattr(module.layer_scatter_modes.mlp_mode, "name", "") != "FULL":
        plan["why"] = "mlp_mode != FULL"
        return plan
    communicator = module.layer_communicator
    if int(communicator._context.attn_tp_size) != 1:
        plan["why"] = "attn_tp_size != 1"
        return plan
    if int(communicator._context.attn_dp_size) != _WORLD:
        plan["why"] = "attn_dp_size != 4"
        return plan
    if dense:
        why = _dense_reason(module.mlp)
        if why is not None:
            plan["why"] = why
            return plan
    elif int(getattr(module.mlp, "num_fused_shared_experts", -1)) != 0:
        plan["why"] = "num_fused_shared_experts != 0"
        return plan
    elif not callable(getattr(module.mlp, "shared_experts", None)):
        plan["why"] = "shared_experts not callable"
        return plan
    group = get_tp_group()
    if torch.distributed.get_world_size(group.device_group) != _WORLD:
        plan["why"] = "tp group world != 4"
        return plan
    handoff.register_norm(int(module.layer_id), module.input_layernorm.weight,
                          module.input_layernorm.variance_epsilon)
    communicator = copy.copy(communicator)
    communicator.input_layernorm = _HandoffNorm(module.input_layernorm)
    attn = copy.copy(module.self_attn)
    fp8w.install_q_b(attn)
    qabsorb.install(attn)
    fp8w.install_o_proj(attn)
    plan.update({"communicator": communicator, "group": group,
                 "num_layers": int(module.config.num_hidden_layers),
                 "device_group": group.device_group,
                 "rank": torch.distributed.get_rank(group.device_group),
                 "get_forward": get_forward, "region": None, "mode": "dense" if dense else "fp4",
                 "layer_id": int(module.layer_id), "attn": attn})
    _fp4_layer_ids.add(int(module.layer_id))
    return plan


_build_cost = {"layers": 0, "moe": 0.0, "shared": 0.0}


def _region(prepared):
    if prepared["region"] is None:
        import time

        module = prepared["module"]
        t0 = time.perf_counter()
        moe = moe_routed_experts.prepare(
            *prepare_args_from_layer(module.mlp.experts), _TOP_K, _ROUTED_SCALING)
        t1 = time.perf_counter()
        full = _replicate_shared(module.mlp.shared_experts, prepared["device_group"])
        t2 = time.perf_counter()
        _build_cost["layers"] += 1
        _build_cost["moe"] += t1 - t0
        _build_cost["shared"] += t2 - t1
        print(f"CACHEON_FP4_PREPARE layer={_build_cost['layers']} "
              f"moe={t1 - t0:.3f}s shared={t2 - t1:.3f}s "
              f"cum_moe={_build_cost['moe']:.1f}s cum_shared={_build_cost['shared']:.1f}s "
              f"cum_total={_build_cost['moe'] + _build_cost['shared']:.1f}s",
              file=sys.stderr, flush=True)
        if full is None:
            raise RuntimeError("shared expert is not the bf16 unquantized layout this design needs")
        norm = module.post_attention_layernorm
        prepared["region"] = {"moe": moe, "shared": full, "norm_weight": norm.weight,
                              "norm_eps": norm.variance_epsilon,
                              "bias": module.mlp.gate.e_score_correction_bias,
                              "gate": module.mlp.gate}
    return prepared["region"]


def _dense_reason(mlp) -> str | None:
    gate_up, down = getattr(mlp, "gate_up_proj", None), getattr(mlp, "down_proj", None)
    if gate_up is None or down is None:
        return "dense mlp has no gate_up_proj/down_proj"
    if gate_up.weight.dtype != torch.bfloat16 or down.weight.dtype != torch.bfloat16:
        return "dense mlp is quantized"
    if getattr(gate_up, "bias", None) is not None or getattr(down, "bias", None) is not None:
        return "dense mlp has a bias"
    if getattr(mlp, "swiglu_limit", None) is not None:
        return "dense mlp clamps its SwiGLU; _shared_forward does not"
    two_i_tp, hidden = gate_up.weight.shape
    if two_i_tp % 2 or hidden != _HIDDEN or tuple(down.weight.shape) != (hidden, two_i_tp // 2):
        return f"dense mlp shapes {tuple(gate_up.weight.shape)} / {tuple(down.weight.shape)}"
    return None


def _dense_region(prepared):
    if prepared["region"] is None:
        import time

        module = prepared["module"]
        t0 = time.perf_counter()
        full = _replicate_shared(module.mlp, prepared["device_group"])
        if full is None:
            raise RuntimeError("dense mlp is not the bf16 TP layout _dense_reason accepted")
        norm = module.post_attention_layernorm
        prepared["region"] = {"full": full, "norm_weight": norm.weight,
                              "norm_eps": norm.variance_epsilon}
        print(f"CACHEON_FP4_PREPARE dense layer={prepared['layer_id']} "
              f"replicate={time.perf_counter() - t0:.3f}s", file=sys.stderr, flush=True)
    return prepared["region"]


def _dense_mlp(prepared, hidden_states, residual):
    region = _dense_region(prepared)
    if not hidden_states.shape[0]:
        return hidden_states, residual
    normed = torch.empty_like(hidden_states)
    updated = torch.empty_like(residual) if residual is not None else None
    fused_add_rmsnorm.fused_add_rmsnorm(
        hidden_states, residual, region["norm_weight"], region["norm_eps"], normed, updated)
    return _shared_forward(region["full"], normed), (updated if updated is not None else residual)


_fp4_layer_ids: set = set()


def _cap(prepared, forward_batch, rows):
    if torch.cuda.is_current_stream_capturing():
        _announce("capturing", "baseline layer(): stream capture in progress")
        return None
    if prepared["communicator"].should_fuse_mlp_allreduce_with_next_layer(forward_batch):
        _announce("fused_allreduce", "baseline layer(): next layer absorbs this all-reduce")
        return None
    sizes = getattr(forward_batch, "global_num_tokens_cpu", None)
    if sizes is None or len(sizes) != _WORLD:
        _announce("no_sizes", f"baseline layer(): global_num_tokens_cpu={sizes!r}")
        return None
    cap = int(max(sizes))
    total = sum(int(v) for v in sizes)
    if total <= _WORLD * 8:
        _announce("decode", f"baseline layer(): total {total} rows is a decode batch")
        return None
    if not _MIN_CAP <= cap <= _MAX_CAP:
        _announce("cap", f"baseline layer(): cap {cap} outside [{_MIN_CAP},{_MAX_CAP}]")
        return None
    if rows != int(sizes[prepared["rank"]]):
        raise RuntimeError(f"rank {prepared['rank']} holds {rows} rows but "
                           f"global_num_tokens_cpu says {int(sizes[prepared['rank']])}")
    if prepared["layer_id"] == min(_fp4_layer_ids):
        vote = torch.ones(1, dtype=torch.int32, device=torch.cuda.current_device())
        torch.distributed.all_reduce(vote, group=prepared["device_group"])
        torch._assert_async(vote == _WORLD, "fp4 region: not every rank entered it; the others "
                             "classified themselves stock and would deadlock")
    _announce("fp4", f"cap={cap} gathered={cap * _WORLD} sizes={[int(v) for v in sizes]}")
    return cap


def forward(prepared, *args, **kwargs):
    if not isinstance(prepared, dict):
        return prepared(*args, **kwargs)
    if prepared["mode"] not in ("fp4", "dense"):
        _announce("classified_stock", f"prepare said mode={prepared['mode']!r}: "
                                      f"{prepared.get('why', 'no reason recorded')}")
        return prepared["layer"](*args, **kwargs)
    names = ("positions", "hidden_states", "forward_batch", "residual", "zero_allocator",
             "gemm_output_zero_allocator", "llama_4_scaling", "prev_topk_indices",
             "captured_last_layer_outputs", "next_full_attention_layer_id")
    bound = dict(zip(names, args))
    bound.update(kwargs)
    forward_batch, hidden_states = bound.get("forward_batch"), bound.get("hidden_states")
    if forward_batch is None or hidden_states is None:
        return prepared["layer"](*args, **kwargs)
    cap = _cap(prepared, forward_batch, int(hidden_states.shape[0]))
    if cap is None:
        return prepared["layer"](*args, **kwargs)

    module, communicator = prepared["module"], prepared["communicator"]
    residual = bound.get("residual")
    from sglang.srt.layers.communicator import get_attn_tp_context
    from sglang.srt.layers.communicator_dsa_cp import maybe_prefetch_next_full_attention_kv

    hidden_states, residual = communicator.prepare_attn_and_capture_last_layer_outputs(
        hidden_states, residual, forward_batch,
        captured_last_layer_outputs=bound.get("captured_last_layer_outputs"),
        quant_format=module._resolve_gfx95_quant_format())
    attn = prepared["attn"]
    with attn.maybe_use_decode_attn_tp(forward_batch):
        hidden_states = attn(
            positions=bound["positions"], hidden_states=hidden_states,
            forward_batch=forward_batch, zero_allocator=bound["zero_allocator"],
            llama_4_scaling=bound.get("llama_4_scaling"),
            layer_scatter_modes=module.layer_scatter_modes,
            prev_topk_indices=bound.get("prev_topk_indices"))
    if isinstance(hidden_states, tuple):
        hidden_states, topk_indices = hidden_states
    else:
        topk_indices = None
    get_attn_tp_context().clear_attn_inputs()
    maybe_prefetch_next_full_attention_kv(forward_batch, bound.get("next_full_attention_layer_id"))
    if prepared["mode"] == "dense":
        hidden_states, residual = _dense_mlp(prepared, hidden_states, residual)
    else:
        hidden_states, residual = _fp4_mlp(prepared, hidden_states, residual, cap,
                                           forward_batch.global_num_tokens_cpu)
    return hidden_states, residual, topk_indices


_PACKED: dict = {}


def _packed_rows(cap, sizes, device):
    key = (int(cap), tuple(int(v) for v in sizes))
    hit = _PACKED.get(device.index)
    if hit is not None and hit[0] == key:
        return hit[1]
    dest = torch.cat([torch.arange(r * cap, r * cap + int(v), dtype=torch.int64, device=device)
                      for r, v in enumerate(sizes) if int(v)])
    _PACKED[device.index] = (key, dest)
    return dest


def _spread(mapping, weights, dest, total):
    rows, k = weights.shape
    idx = torch.full((total, k), -1, dtype=mapping.dtype, device=mapping.device)
    idx.index_copy_(0, dest, mapping[: rows * k].view(rows, k))
    w = torch.zeros((total, k), dtype=weights.dtype, device=weights.device)
    w.index_copy_(0, dest, weights)
    return idx.view(-1), w


def _fp4_mlp(prepared, hidden_states, residual, cap, sizes):
    region = _region(prepared)
    moe = region["moe"]
    rows, device = int(hidden_states.shape[0]), hidden_states.device
    total = cap * _WORLD
    dgroup = prepared["device_group"]
    gather = prefill_gather.state_for(dgroup, device, _MAX_CAP)
    tail = (prefill_tail.state_for(dgroup, device, _HIDDEN, _MAX_CAP, _TOP_K)
            if levers.PREFILL_PUSH_TAIL else None)
    overlap = SHARED_OVERLAP == "late" and tail is not None

    fp4 = scale = logits = None
    if rows:
        normed = torch.empty_like(hidden_states)
        updated = torch.empty_like(residual) if residual is not None else None
        fused_add_rmsnorm.fused_add_rmsnorm(
            hidden_states, residual, region["norm_weight"], region["norm_eps"], normed, updated)
        residual = updated if updated is not None else residual
        fp4, scale = moe_routed_experts._quantize(moe, normed)
        logits = region["gate"](normed).float().contiguous()
    else:
        normed = hidden_states
    gather.push(fp4, scale, logits, rows, cap)
    shared_local = _shared_forward(region["shared"], normed) if rows and not overlap else None
    g_fp4, g_scale, g_logits = gather.wait()

    real = sum(int(v) for v in sizes)
    packed = tail is not None and real < total
    if packed:
        dest = _packed_rows(cap, sizes, device)
        g_fp4 = torch.index_select(g_fp4, 0, dest)
        g_scale = torch.index_select(g_scale.view(torch.uint8), 0, dest).view(g_scale.dtype)
        g_logits = torch.index_select(g_logits, 0, dest)
        tactic = (list(moe["tactic"]) if real > moe_routed_experts._TACTIC_MIN_TOKENS else [-1, -1])
        ids, weights, width = moe["topk_ids"][:real], moe["topk_weights"][:real], real
    else:
        plan = moe["call_plans"].get(total)
        if plan is None:
            plan = (moe["topk_ids"][:total], moe["topk_weights"][:total], list(moe["tactic"]))
            moe["call_plans"][total] = plan
        ids, weights, tactic = plan
        width = total
    if tail is not None:
        deferred = moe.get("native_weights_deferred")
        if deferred is None:
            if moe["native_weights"][-1] is not True:
                raise RuntimeError("retained raw-op arguments do not end with do_finalize")
            deferred = moe["native_weights"][:-1] + (False,)
            moe["native_weights_deferred"] = deferred
        result = moe["rt"].native(
            moe["routing_mode"], g_logits, ids, weights, region["bias"],
            g_fp4, g_scale, *deferred,
            bool(moe["pdl"](width)), moe_routed_experts._ACT_SWIGLU,
            tail.dummy_out, tactic, True, None, [], [], False)
        gemm2 = torch.from_dlpack(result[0])
        mapping = torch.from_dlpack(result[2]).reshape(-1)
        if packed:
            mapping, weights = _spread(mapping, weights, dest, total)
        out = torch.empty((rows, _HIDDEN), dtype=torch.bfloat16, device=device)
        norm = (handoff.next_norm(prepared["layer_id"], prepared["num_layers"])
                if FUSE_NEXT_NORM and rows else None)
        fused = None
        if norm is not None and residual is not None and residual.is_contiguous():
            weight, eps, final = norm
            normed = torch.empty_like(out)
            updated = torch.empty_like(residual)
            fused = (residual, weight, eps, normed, updated, final)
        if overlap:
            current = torch.cuda.current_stream(device)
            moe_done = torch.cuda.Event()
            moe_done.record(current)
            tail.push(gemm2, mapping, weights, cap, partitioned=True, rows=sizes)
            if rows:
                side = _SIDE_STREAMS.get(device.index)
                if side is None:
                    side = _SIDE_STREAMS[device.index] = torch.cuda.Stream(device=device)
                side.wait_event(moe_done)
                with torch.cuda.stream(side):
                    shared_local = _shared_forward(region["shared"], normed)
                normed.record_stream(side)
                current.wait_stream(side)
                shared_local.record_stream(current)
            tail.reduce(shared_local, out, fused)
        else:
            tail.run(gemm2, mapping, weights, cap, shared_local, out, fused, rows=sizes)
        if fused is not None:
            handoff.publish(out, residual, fused[3], fused[4], fused[1],
                            handoff.ROUNDED if fused[5] else handoff.FP32)
        return out, residual

    partial = torch.empty((total, _HIDDEN), dtype=torch.bfloat16, device=device)
    moe["rt"].native(
        moe["routing_mode"], g_logits, ids, weights, region["bias"],
        g_fp4, g_scale, *moe["native_weights"],
        bool(moe["pdl"](total)), moe_routed_experts._ACT_SWIGLU,
        partial, tactic, True, None, [], [], False)

    out = torch.empty((cap, _HIDDEN), dtype=torch.bfloat16, device=device)
    torch.distributed.reduce_scatter_tensor(out, partial, group=dgroup)
    out = out[:rows] if rows != cap else out
    if rows:
        out += shared_local
    return out, residual
