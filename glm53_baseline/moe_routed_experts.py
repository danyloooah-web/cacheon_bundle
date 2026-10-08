
from __future__ import annotations

import threading
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

_SGLANG_VERSION, _FLASHINFER_VERSION = "0.5.20", "0.6.18"
_TAG, _GATE_UP = "nvfp4_layer", "gate_up"
_INTERLEAVED, _TRTLLM = (
    "up_gate_interleaved_64+sf_swizzled_128x4", "trtllm_fp4_shuffled")
_TOKENS, _EXPERTS, _HIDDEN, _INTERMEDIATE, _TOP_K = 16384, 256, 6144, 512, 8
_MIN_TOKENS = 1
_TACTIC_MIN_TOKENS = 12288
_ROUTED_SCALING, _SF_BLOCK = 2.5, 16
_ROUTING_DEEPSEEK_V3, _ACT_SWIGLU = 2, 3
_MIN_TACTIC_SPEEDUP, _AB_PAIRS, _AB_REPLAYS, _AB_ITERS = 1.05, 3, 5, 3
_W13_SHAPE = (_EXPERTS, 2 * _INTERMEDIATE, _HIDDEN // 2)
_W2_SHAPE = (_EXPERTS, _HIDDEN, _INTERMEDIATE // 2)
_W13_SF_SHAPE = (_EXPERTS, 2 * _INTERMEDIATE, _HIDDEN // _SF_BLOCK)
_W2_SF_SHAPE = (_EXPERTS, _HIDDEN, _INTERMEDIATE // _SF_BLOCK)

_tactic_lock = threading.Lock()
_tactics: dict[tuple, tuple[object, tuple[int, int], float]] = {}
_tactic_failures: dict[tuple, str] = {}


_RUNTIME_CACHE: list = []


def _runtime():
    if not _RUNTIME_CACHE:
        _RUNTIME_CACHE.append(_resolve_runtime())
    return _RUNTIME_CACHE[0]


def _resolve_runtime():
    import flashinfer
    import sglang
    if flashinfer.__version__ != _FLASHINFER_VERSION:
        raise RuntimeError(f"requires FlashInfer {_FLASHINFER_VERSION}, got {flashinfer.__version__}")
    if sglang.__version__ != _SGLANG_VERSION:
        raise RuntimeError(f"requires SGLang {_SGLANG_VERSION}, got {sglang.__version__}")

    from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
    from flashinfer.autotuner import AutoTuner, DynamicTensorSpec, TuningConfig
    from flashinfer.fused_moe.core import (ActivationType, Fp8QuantizationType,
        MoeRunnerInputs, RoutingInputMode, WeightLayout,
        deduce_trtllm_gen_tensor_dtype, get_trtllm_moe_sm100_module)
    import sglang.srt.layers.quantization
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import trtllm_moe_enable_pdl
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    module = get_trtllm_moe_sm100_module()
    if not hasattr(module, "MoERunner"):
        raise RuntimeError(f"FlashInfer {_FLASHINFER_VERSION} TRT-LLM module does not expose MoERunner")
    return SimpleNamespace(
        tuner=AutoTuner, spec=DynamicTensorSpec, config=TuningConfig,
        inputs=MoeRunnerInputs, routing=RoutingInputMode, layout=WeightLayout,
        activation=ActivationType, fp8=Fp8QuantizationType,
        deduce=deduce_trtllm_gen_tensor_dtype, runner=module.MoERunner,
        native=gen_trtllm_gen_fused_moe_sm100_module().build_and_load().trtllm_fp4_block_scale_moe,
        quantize=fp4_quantize, pdl=trtllm_moe_enable_pdl)


def _shape_dtype(name, tensor, shape, dtype, device=None):
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise ValueError(f"{name} must be contiguous {dtype} {shape}, got {tensor.dtype} {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} is on {tensor.device}, expected {device}")
    return tensor


def _input_scale(name, value, device):
    if not torch.is_tensor(value) or value.dtype != torch.float32 or value.device != device:
        raise ValueError(f"{name} must be a float32 tensor on the weight device")
    flat = value.reshape(-1)
    if flat.numel() == 1:
        return flat
    if flat.numel() != _EXPERTS or not bool(torch.all(flat == flat[0])):
        raise ValueError(f"{name} must be scalar or a uniform 256-vector")
    return flat[:1].contiguous()


def _view_tensors(view):
    layout = str(view.cacheon_w13_layout)
    w13 = _shape_dtype("w13_weight", view.w13_weight, _W13_SHAPE, torch.uint8)
    device = w13.device
    w2 = _shape_dtype("w2_weight", view.w2_weight, _W2_SHAPE, torch.uint8, device)
    w13_sf = _shape_dtype("w13_blockscale", view.w13_blockscale_swizzled,
                           _W13_SF_SHAPE, torch.float8_e4m3fn, device)
    w2_sf = _shape_dtype("w2_blockscale", view.w2_blockscale_swizzled,
                          _W2_SF_SHAPE, torch.float8_e4m3fn, device)
    if int(view.intermediate_size_per_partition) != _INTERMEDIATE:
        raise ValueError("GLM-5.3 TP4 requires intermediate_size_per_partition=512")
    if layout == _TRTLLM:
        return w13, w13_sf, w2, w2_sf
    if layout not in (_GATE_UP, _INTERLEAVED):
        raise ValueError(f"unsupported NVFP4 w13 layout {layout!r}")

    from cacheon_kernels import codec
    from sglang.srt.layers.quantization.utils import prepare_static_weights_for_trtllm_fp4_moe

    w13_sf = codec.unswizzle_blockscale(w13_sf, rows=2 * _INTERMEDIATE,
                                         cols=_HIDDEN // _SF_BLOCK)
    w2_sf = codec.unswizzle_blockscale(w2_sf, rows=_HIDDEN,
                                        cols=_INTERMEDIATE // _SF_BLOCK)
    if layout == _INTERLEAVED:
        w13 = codec.deinterleave_w13_halves(w13, group=64)
        w13_sf = codec.deinterleave_w13_halves(w13_sf, group=64)
    w13 = torch.cat((w13[:, _INTERMEDIATE:], w13[:, :_INTERMEDIATE]), dim=1)
    w13_sf = torch.cat((w13_sf[:, _INTERMEDIATE:], w13_sf[:, :_INTERMEDIATE]), dim=1)
    return prepare_static_weights_for_trtllm_fp4_moe(
        w13, w2, w13_sf, w2_sf, _HIDDEN, _INTERMEDIATE, _EXPERTS, is_gated=True)


def _quantize(prepared, x):
    fp4, scale = prepared["rt"].quantize(
        x, prepared["input_scale"], sf_vec_size=_SF_BLOCK,
        sf_use_ue8m0=False, is_sf_swizzled_layout=False)
    tokens = x.shape[0]
    return (fp4.reshape(tokens, _HIDDEN // 2),
            scale.view(torch.float8_e4m3fn).reshape(tokens, _HIDDEN // _SF_BLOCK))


def _runner_kwargs(prepared, bias, tokens):
    return {
        "routing_input_mode": prepared["routing_mode"], "num_experts": _EXPERTS,
        "routing_bias": bias, "gemm1_weights": prepared["w13"],
        "gemm1_weights_scale": prepared["w13_sf"], "gemm1_bias": None,
        "gemm1_alpha": None, "gemm1_beta": None, "gemm1_clamp_limit": None,
        "gemm2_weights": prepared["w2"], "gemm2_weights_scale": prepared["w2_sf"],
        "gemm2_bias": None, "output1_scale_scalar": prepared["g1_scale_c"],
        "output1_scale_gate_scalar": prepared["g1_alphas"],
        "output2_scale_scalar": prepared["g2_alphas"], "per_token_scale": None,
        "n_group": 1, "topk_group": 1, "local_expert_offset": 0,
        "routed_scaling_factor": _ROUTED_SCALING,
        "routing_method_type": _ROUTING_DEEPSEEK_V3, "do_finalize": True,
        "enable_pdl": bool(prepared["pdl"](tokens)), "activation_type": _ACT_SWIGLU,
        "num_fused_shared_experts": 0, "norm_topk_prob": True,
        "routing_replay_out": None,
    }


def _inputs(prepared, logits, bias, fp4, scale):
    device = fp4.device
    tokens = fp4.shape[0]
    values = prepared["rt"].inputs(
        output=torch.empty((tokens, _HIDDEN), dtype=torch.bfloat16, device=device),
        routing_logits=logits,
        topk_ids=torch.empty((tokens, _TOP_K), dtype=torch.int32, device=device),
        expert_weights=torch.empty((tokens, _TOP_K), dtype=torch.bfloat16, device=device),
        hidden_states=fp4, hidden_states_scale=scale, gemm1_lora_delta=None,
        per_token_scale=None)
    return values, _runner_kwargs(prepared, bias, tokens)


def _runner(prepared, fp4, scale):
    rt = prepared["rt"]
    return rt.runner(
        top_k=_TOP_K, num_local_experts=_EXPERTS,
        dtype_act=rt.deduce(fp4, scale),
        dtype_weights=rt.deduce(prepared["w13"], prepared["w13_sf"]),
        fp8_quantization_type=rt.fp8.NoneFp8, hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE, activation_type=rt.activation.Swiglu.value,
        use_shuffled_weight=True, weight_layout=rt.layout.MajorK,
        use_packed_weights=False, use_per_token_scaling=False,
        num_experts=_EXPERTS, num_fused_shared_experts=0)


def _exact_bucket(value: int) -> int:
    if not _MIN_TOKENS <= value <= _TOKENS:
        raise ValueError(f"tuner received off-domain token count {value}")
    return _TOKENS


def _tuning_config(prepared, runner, inputs):
    base = runner._make_tuning_config(
        inputs,
        tune_max_num_tokens=_TOKENS,
        use_cold_l2_cache=True,
        use_cuda_graph=True,
    )
    spec = base.dynamic_tensor_specs[0]
    exact = prepared["rt"].spec(
        input_idx=spec.input_idx, dim_idx=spec.dim_idx, gen_tuning_buckets=(_TOKENS,),
        map_to_tuning_buckets=_exact_bucket)
    return prepared["rt"].config(
        dynamic_tensor_specs=(exact,), constraint_specs=base.constraint_specs,
        tensor_initializers=base.tensor_initializers,
        value_aware_input_indices=base.value_aware_input_indices,
        profile_arena_input_indices=base.profile_arena_input_indices,
        use_cold_l2_cache=True, use_cuda_graph=True)


def _execute(runner, tactic, inputs, kwargs):
    runner.forward(inputs.to_list(), tactic=tactic, **kwargs)
    return inputs.output


def _capture(prepared, runner, tactic, x, logits, bias):
    fp4, scale = _quantize(prepared, x)
    inputs, kwargs = _inputs(prepared, logits, bias, fp4, scale)
    _execute(runner, tactic, inputs, kwargs)
    torch.cuda.synchronize(x.device)
    stream = torch.cuda.Stream(device=x.device)
    with torch.cuda.stream(stream):
        _execute(runner, tactic, inputs, kwargs)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(_AB_ITERS):
                fp4, scale = _quantize(prepared, x)
                inputs, kwargs = _inputs(prepared, logits, bias, fp4, scale)
                _execute(runner, tactic, inputs, kwargs)
    torch.cuda.synchronize(x.device)
    graph.replay()
    torch.cuda.synchronize(x.device)
    return graph, inputs


def _graph_ms(capture) -> float:
    graph = capture[0]
    best = float("inf")
    for _ in range(_AB_REPLAYS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        best = min(best, float(start.elapsed_time(end)) / _AB_ITERS)
    return best


def _ab_speedup(prepared, runner, tactic, x, logits, bias) -> float:
    tuned_graph = _capture(prepared, runner, list(tactic), x, logits, bias)
    stock_graph = _capture(prepared, runner, -1, x, logits, bias)
    ratios = []
    for pair in range(_AB_PAIRS):
        if pair % 2:
            stock_ms = _graph_ms(stock_graph)
            tuned_ms = _graph_ms(tuned_graph)
        else:
            tuned_ms = _graph_ms(tuned_graph)
            stock_ms = _graph_ms(stock_graph)
        ratios.append(stock_ms / tuned_ms)
    return min(ratios)


def _select_tactic(prepared):
    device = prepared["w13"].device
    key = (_FLASHINFER_VERSION, device.type, device.index,
           torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device),
           _TOKENS, _EXPERTS, _HIDDEN, _INTERMEDIATE, _TOP_K)
    with _tactic_lock:
        if key in _tactics:
            return _tactics[key]
        if key in _tactic_failures:
            raise RuntimeError(f"prior exact-shape tactic preparation failed: {_tactic_failures[key]}")
        try:
            generator = torch.Generator(device=device)
            generator.manual_seed(5316384)
            x = torch.randn((_TOKENS, _HIDDEN), dtype=torch.bfloat16, device=device,
                            generator=generator)
            logits = torch.randn((_TOKENS, _EXPERTS), dtype=torch.float32,
                                 device=device, generator=generator)
            bias = torch.randn((_EXPERTS,), dtype=torch.float32, device=device,
                               generator=generator)
            fp4, scale = _quantize(prepared, x)
            inputs, kwargs = _inputs(prepared, logits, bias, fp4, scale)
            runner = _runner(prepared, fp4, scale)
            tuner = prepared["rt"].tuner(warmup=3, repeat=7)
            tuner.is_tuning_mode = True
            try:
                _, tactic = tuner.choose_one(
                    "flashinfer::trtllm_fp4_block_scale_moe", [runner],
                    _tuning_config(prepared, runner, inputs), inputs.to_list(), **kwargs)
            finally:
                tuner.is_tuning_mode = False
            try:
                pair = tuple(int(item) for item in tactic)
            except TypeError:
                pair = ()
            if len(pair) != 2 or any(item < 0 for item in pair):
                raise RuntimeError(f"no fully explicit TRT-LLM tactic selected: {tactic!r} "
                                   f"({type(tactic).__name__})")
            tactic = pair
            tactic = (int(tactic[0]), int(tactic[1]))
            speedup = _ab_speedup(prepared, runner, tactic, x, logits, bias)
            if speedup < _MIN_TACTIC_SPEEDUP:
                raise RuntimeError(
                    f"explicit tactic {tactic} reached only {speedup:.4f}x at M={_TOKENS}; "
                    f"requires {_MIN_TACTIC_SPEEDUP:.2f}x")
            record = (runner, tactic, speedup)
            _tactics[key] = record
            return record
        except Exception as exc:
            _tactic_failures[key] = f"{type(exc).__name__}: {exc}"
            raise


def prepare(tag, view, topk, routed_scaling):
    if tag != _TAG:
        raise ValueError(f"requires {_TAG!r} weights, got {tag!r}")
    if int(topk) != _TOP_K or float(routed_scaling) != _ROUTED_SCALING:
        raise ValueError(f"requires topk={_TOP_K}, routed_scaling={_ROUTED_SCALING}, "
                         f"got {topk}, {routed_scaling}")
    runtime = _runtime()
    w13, w13_sf, w2, w2_sf = _view_tensors(view)
    device = w13.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (10, 3):
        raise ValueError("requires an sm103 CUDA device")
    prepared = {
        "rt": runtime, "w13": w13, "w13_sf": w13_sf, "w2": w2, "w2_sf": w2_sf,
        "g1_scale_c": _shape_dtype("g1_scale_c", view.g1_scale_c,
                                    (_EXPERTS,), torch.float32, device),
        "g1_alphas": _shape_dtype("g1_alphas", view.g1_alphas,
                                   (_EXPERTS,), torch.float32, device),
        "g2_alphas": _shape_dtype("g2_alphas", view.g2_alphas,
                                   (_EXPERTS,), torch.float32, device),
        "input_scale": _input_scale("w13_input_scale_quant",
                                    view.w13_input_scale_quant, device),
        "w2_input_scale": _input_scale("w2_input_scale_quant",
                                       view.w2_input_scale_quant, device),
        "routing_mode": runtime.routing.FromLogits,
        "pdl": runtime.pdl,
    }
    runner, tactic, speedup = _select_tactic(prepared)
    prepared["runner"] = runner
    prepared["tactic"] = tactic
    prepared["prepare_ab_speedup"] = speedup
    prepared["native_weights"] = (
        w13, w13_sf, None, None, None, None, None, w2, w2_sf, None,
        prepared["g1_scale_c"], prepared["g1_alphas"], prepared["g2_alphas"], None,
        _EXPERTS, _TOP_K, 0, 1, 1, _INTERMEDIATE, 0, _EXPERTS,
        _ROUTED_SCALING, _ROUTING_DEEPSEEK_V3, True)
    prepared["call_plans"] = {}
    prepared["topk_ids"] = torch.empty((_TOKENS, _TOP_K), dtype=torch.int32, device=device)
    prepared["topk_weights"] = torch.empty((_TOKENS, _TOP_K), dtype=torch.bfloat16, device=device)
    return prepared


_FIN_BLOCK = 1024
DECODE_MAX_TOKENS = 512


@triton.jit
def _finalize(GEMM2, IDX, W, OUT, hidden, stride, K: tl.constexpr, BLOCK: tl.constexpr):
    tok = tl.inline_asm_elementwise(
        "griddepcontrol.wait; mov.b32 $0, $1;", "=r,r", [tl.program_id(0)],
        dtype=tl.int32, is_pure=False, pack=1)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(K):
        idx = tl.load(IDX + tok * K + k)
        w = tl.load(W + tok * K + k).to(tl.float32)
        valid = idx >= 0
        row = tl.where(valid, idx, 0).to(tl.int64)
        v = tl.load(GEMM2 + row * stride + cols, mask=mask & valid, other=0.0).to(tl.float32)
        acc += w * v
    tl.store(OUT + tok.to(tl.int64) * hidden + cols, acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _finalize_shared(GEMM2, IDX, W, SHARED, OUT, hidden, stride, K: tl.constexpr, BLOCK: tl.constexpr):
    tok = tl.inline_asm_elementwise(
        "griddepcontrol.wait; mov.b32 $0, $1;", "=r,r", [tl.program_id(0)],
        dtype=tl.int32, is_pure=False, pack=1)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(K):
        idx = tl.load(IDX + tok * K + k)
        w = tl.load(W + tok * K + k).to(tl.float32)
        valid = idx >= 0
        row = tl.where(valid, idx, 0).to(tl.int64)
        v = tl.load(GEMM2 + row * stride + cols, mask=mask & valid, other=0.0).to(tl.float32)
        acc += w * v
    acc += tl.load(SHARED + tok.to(tl.int64) * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + tok.to(tl.int64) * hidden + cols, acc.to(tl.bfloat16), mask=mask)


def decode_finalize_shared(gemm2, mapping, weights, shared, out):
    tokens = int(out.shape[0])
    _shape_dtype("shared", shared, (tokens, _HIDDEN), torch.bfloat16, out.device)
    _finalize_shared[(tokens, _HIDDEN // _FIN_BLOCK)](
        gemm2, mapping, weights, shared, out, _HIDDEN, gemm2.stride(0),
        _TOP_K, _FIN_BLOCK, num_warps=4, launch_pdl=True)


def decode_deferred(prepared, x, router_logits, correction_bias, out, quantized=None):
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe

    tokens = int(x.shape[0])
    fp4, scale = quantized if quantized is not None else _quantize(prepared, x)
    gemm2, weights, mapping = trtllm_fp4_block_scale_moe(
        routing_logits=router_logits, routing_bias=correction_bias,
        hidden_states=fp4, hidden_states_scale=scale,
        gemm1_weights=prepared["w13"], gemm1_weights_scale=prepared["w13_sf"],
        gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=prepared["w2"], gemm2_weights_scale=prepared["w2_sf"], gemm2_bias=None,
        output1_scale_scalar=prepared["g1_scale_c"],
        output1_scale_gate_scalar=prepared["g1_alphas"],
        output2_scale_scalar=prepared["g2_alphas"],
        num_experts=_EXPERTS, top_k=_TOP_K, n_group=1, topk_group=1,
        intermediate_size=_INTERMEDIATE, local_expert_offset=0, local_num_experts=_EXPERTS,
        routed_scaling_factor=_ROUTED_SCALING, routing_method_type=_ROUTING_DEEPSEEK_V3,
        do_finalize=False, activation_type=_ACT_SWIGLU, per_token_scale=None,
        tune_max_num_tokens=1 << (tokens - 1).bit_length(),
        output=out, enable_pdl=bool(prepared["pdl"](tokens)))
    if gemm2.dtype != torch.bfloat16 or gemm2.dim() != 2 or gemm2.shape[1] < _HIDDEN:
        raise RuntimeError(f"unexpected deferred GEMM2 output {tuple(gemm2.shape)} {gemm2.dtype}")
    mapping = mapping.reshape(-1)
    if mapping.numel() < tokens * _TOP_K or mapping.dtype != torch.int32:
        raise RuntimeError(f"unexpected expanded->permuted map {tuple(mapping.shape)} {mapping.dtype}")
    return gemm2, mapping, weights.reshape(-1)


def decode_deferred_routed(prepared, x, topk_ids, topk_weights, out, quantized=None):
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe

    tokens = int(x.shape[0])
    fp4, scale = quantized if quantized is not None else _quantize(prepared, x)
    gemm2, weights, mapping = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(topk_ids, topk_weights), routing_bias=None,
        hidden_states=fp4, hidden_states_scale=scale,
        gemm1_weights=prepared["w13"], gemm1_weights_scale=prepared["w13_sf"],
        gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=prepared["w2"], gemm2_weights_scale=prepared["w2_sf"], gemm2_bias=None,
        output1_scale_scalar=prepared["g1_scale_c"],
        output1_scale_gate_scalar=prepared["g1_alphas"],
        output2_scale_scalar=prepared["g2_alphas"],
        num_experts=_EXPERTS, top_k=_TOP_K, n_group=1, topk_group=1,
        intermediate_size=_INTERMEDIATE, local_expert_offset=0, local_num_experts=_EXPERTS,
        routed_scaling_factor=None, routing_method_type=_ROUTING_DEEPSEEK_V3, do_finalize=False,
        enable_pdl=bool(prepared["pdl"](tokens)), activation_type=_ACT_SWIGLU, per_token_scale=None,
        output=out, tune_max_num_tokens=1 << (tokens - 1).bit_length())
    if gemm2.dtype != torch.bfloat16 or gemm2.dim() != 2 or gemm2.shape[1] < _HIDDEN:
        raise RuntimeError(f"unexpected deferred GEMM2 output {tuple(gemm2.shape)} {gemm2.dtype}")
    mapping = mapping.reshape(-1)
    if mapping.numel() < tokens * _TOP_K or mapping.dtype != torch.int32:
        raise RuntimeError(f"unexpected expanded->permuted map {tuple(mapping.shape)} {mapping.dtype}")
    return gemm2, mapping, weights.reshape(-1)


def _decode_dispatch(prepared, x, router_logits, correction_bias, out, tokens):
    gemm2, mapping, weights = decode_deferred(prepared, x, router_logits, correction_bias, out)
    _finalize[(tokens, _HIDDEN // _FIN_BLOCK)](
        gemm2, mapping, weights, out, _HIDDEN, gemm2.stride(0),
        _TOP_K, _FIN_BLOCK, num_warps=4, launch_pdl=True)


def _stock_dispatch(prepared, x, router_logits, correction_bias, out, tokens):
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe

    fp4, scale = _quantize(prepared, x)
    trtllm_fp4_block_scale_moe(
        routing_logits=router_logits, routing_bias=correction_bias,
        hidden_states=fp4, hidden_states_scale=scale,
        gemm1_weights=prepared["w13"], gemm1_weights_scale=prepared["w13_sf"],
        gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=prepared["w2"], gemm2_weights_scale=prepared["w2_sf"], gemm2_bias=None,
        output1_scale_scalar=prepared["g1_scale_c"],
        output1_scale_gate_scalar=prepared["g1_alphas"],
        output2_scale_scalar=prepared["g2_alphas"],
        num_experts=_EXPERTS, top_k=_TOP_K, n_group=1, topk_group=1,
        intermediate_size=_INTERMEDIATE, local_expert_offset=0, local_num_experts=_EXPERTS,
        routed_scaling_factor=_ROUTED_SCALING, routing_method_type=_ROUTING_DEEPSEEK_V3,
        do_finalize=True, activation_type=_ACT_SWIGLU, per_token_scale=None,
        tune_max_num_tokens=1 << (tokens - 1).bit_length(),
        output=out, enable_pdl=bool(prepared["pdl"](tokens)))


def fused_routed_experts(x, router_logits, correction_bias, prepared, out):
    device = prepared["w13"].device
    tokens = int(x.shape[0])
    if not _MIN_TOKENS <= tokens <= _TOKENS:
        raise ValueError(f"served rows {tokens} outside the declared {_MIN_TOKENS}..{_TOKENS} range")
    _shape_dtype("x", x, (tokens, _HIDDEN), torch.bfloat16, device)
    _shape_dtype("router_logits", router_logits, (tokens, _EXPERTS), torch.float32, device)
    _shape_dtype("correction_bias", correction_bias, (_EXPERTS,), torch.float32, device)
    _shape_dtype("out", out, (tokens, _HIDDEN), torch.bfloat16, device)
    if tokens <= DECODE_MAX_TOKENS:
        _decode_dispatch(prepared, x, router_logits, correction_bias, out, tokens)
        return
    if tokens <= _TACTIC_MIN_TOKENS:
        _stock_dispatch(prepared, x, router_logits, correction_bias, out, tokens)
        return
    plan = prepared["call_plans"].get(tokens)
    if plan is None:
        plan = (prepared["topk_ids"][:tokens], prepared["topk_weights"][:tokens],
                list(prepared["tactic"]))
        prepared["call_plans"][tokens] = plan
    ids, weights, tactic = plan
    fp4, scale = _quantize(prepared, x)
    prepared["rt"].native(
        prepared["routing_mode"], router_logits, ids, weights, correction_bias, fp4, scale,
        *prepared["native_weights"], bool(prepared["pdl"](tokens)), _ACT_SWIGLU,
        out, tactic, True, None, [], [], False)


DA_MAX_ROWS = 16
_DA_STATE: dict = {}
_R8M: list = []


def _route8m():
    if not _R8M:
        import route8s

        _R8M.append(route8s)
    return _R8M[0]


def _da_max_ctas(rows):
    filled = min(_EXPERTS, rows * _TOP_K)
    return filled + (rows * _TOP_K - filled) // 8


def _da_state(prepared, fp4, scale, out, rows, pdl):
    key = (fp4.device.index, rows, pdl)
    state = _DA_STATE.get(key)
    if state is not None:
        return state
    from flashinfer.autotuner import AutoTuner

    rt = prepared["rt"]
    device = fp4.device
    ids = torch.zeros((rows, _TOP_K), dtype=torch.int32, device=device)
    weights = torch.zeros((rows, _TOP_K), dtype=torch.float32, device=device)
    mc = _da_max_ctas(rows)
    zeros = lambda n: torch.zeros(n, dtype=torch.int32, device=device)
    meta = [zeros(1), zeros(rows * _TOP_K), zeros(mc * 8 + 1), weights, zeros(max(_EXPERTS * 2, 512)), zeros(_EXPERTS),
            zeros(mc), zeros(mc), zeros(1)]
    runner = rt.runner(
        top_k=_TOP_K, num_local_experts=_EXPERTS, dtype_act=rt.deduce(fp4, scale),
        dtype_weights=rt.deduce(prepared["w13"], prepared["w13_sf"]), fp8_quantization_type=rt.fp8.NoneFp8,
        hidden_size=_HIDDEN, intermediate_size=_INTERMEDIATE, activation_type=_ACT_SWIGLU,
        weight_layout=rt.layout.MajorK, use_shuffled_weight=True, use_per_token_scaling=False,
        num_experts=_EXPERTS, num_fused_shared_experts=0)
    inputs = rt.inputs(output=out, routing_logits=None, topk_ids=ids, expert_weights=weights, hidden_states=fp4,
                       hidden_states_scale=scale, gemm1_lora_delta=None, per_token_scale=None)
    tuning = runner._make_tuning_config(inputs, tune_max_num_tokens=1 << (rows - 1).bit_length(),
                                        routing_input_mode=rt.routing.UnpackedPrecomputed, use_cold_l2_cache=True,
                                        use_cuda_graph=True)
    kwargs = {
        "routing_input_mode": rt.routing.UnpackedPrecomputed, "num_experts": _EXPERTS, "routing_bias": None,
        "gemm1_weights": prepared["w13"], "gemm1_weights_scale": prepared["w13_sf"], "gemm1_bias": None,
        "gemm1_alpha": None, "gemm1_beta": None, "gemm1_clamp_limit": None, "gemm2_weights": prepared["w2"],
        "gemm2_weights_scale": prepared["w2_sf"], "gemm2_bias": None,
        "output1_scale_scalar": prepared["g1_scale_c"], "output1_scale_gate_scalar": prepared["g1_alphas"],
        "output2_scale_scalar": prepared["g2_alphas"], "per_token_scale": None, "n_group": 1, "topk_group": 1,
        "local_expert_offset": 0, "routed_scaling_factor": None, "routing_method_type": _ROUTING_DEEPSEEK_V3,
        "enable_pdl": pdl, "do_finalize": False, "activation_type": _ACT_SWIGLU, "num_fused_shared_experts": 0,
        "norm_topk_prob": True, "routing_replay_out": None,
    }
    _, tactic = AutoTuner.get().choose_one("flashinfer::trtllm_fp4_block_scale_moe", [runner], tuning,
                                           inputs.to_list(), **kwargs)
    config = [-1, -1] if tactic == -1 else [int(v) for v in tactic]
    state = {"ids": ids, "weights": weights, "meta": meta, "config": config,
             "ticket": torch.zeros(1, dtype=torch.int32, device=device), "body": None}
    state["body"] = list(_da_native(prepared, state, fp4, scale, out, pdl, prep=True))
    _DA_STATE[key] = state
    return state


def _da_native(prepared, state, fp4, scale, out, pdl, prep=False):
    return prepared["rt"].native(
        int(prepared["rt"].routing.UnpackedPrecomputed), None, state["ids"], state["weights"], None, fp4, scale,
        prepared["w13"], prepared["w13_sf"], None, None, None, None, None, prepared["w2"], prepared["w2_sf"], None,
        prepared["g1_scale_c"], prepared["g1_alphas"], prepared["g2_alphas"], None,
        _EXPERTS, _TOP_K, 0, 1, 1, _INTERMEDIATE, 0, _EXPERTS, None, _ROUTING_DEEPSEEK_V3, False, pdl, _ACT_SWIGLU,
        out, state["config"], True, None, list(state["meta"]), [] if prep else list(state["body"]), prep)


def decode_deferred_da(prepared, x, logits, bias, scale_factor, out, quantized=None):
    rows = int(x.shape[0])
    if (not 1 <= rows <= DA_MAX_ROWS or logits.dtype != torch.float32 or not logits.is_contiguous()
            or logits.dim() != 2 or logits.shape[1] != _EXPERTS or bias is None or bias.dtype != torch.float32
            or bias.numel() != _EXPERTS or not bias.is_contiguous()):
        return None
    fp4, scale = quantized if quantized is not None else _quantize(prepared, x)
    pdl = bool(prepared["pdl"](rows))
    state = _da_state(prepared, fp4, scale, out, rows, pdl)
    meta = state["meta"]
    from glm53_baseline import route_topk

    _route8m().route_meta(logits, bias, state["ids"], state["weights"], float(scale_factor), pdl, meta[0], meta[1],
                          meta[2], meta[5], meta[6], meta[7], meta[8], state["ticket"],
                          route_topk.PRUNE_MAX_COUNT, route_topk.PRUNE_BUDGET, route_topk.PRUNE_PAIR,
                          list(route_topk.PRUNE_POS_MULT), route_topk.PRUNE_RENORM)
    gemm2, weights, mapping = _da_native(prepared, state, fp4, scale, out, pdl)
    if gemm2.dtype != torch.bfloat16 or gemm2.dim() != 2 or gemm2.shape[1] < _HIDDEN:
        raise RuntimeError(f"unexpected DA GEMM2 output {tuple(gemm2.shape)} {gemm2.dtype}")
    return gemm2, mapping.reshape(-1), weights.reshape(-1)
