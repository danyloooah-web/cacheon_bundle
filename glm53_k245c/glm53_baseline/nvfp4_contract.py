
from __future__ import annotations

from types import SimpleNamespace

import torch

NVFP4_PREPARE_TAG = "nvfp4_layer"
NVFP4_GATE_UP_LAYOUT = "gate_up"
NVFP4_INTERLEAVED_LAYOUT = "up_gate_interleaved_64+sf_swizzled_128x4"
NVFP4_TRTLLM_LAYOUT = "trtllm_fp4_shuffled"


def _data(value):
    return getattr(value, "data", value)


def supports_layer(layer: object) -> bool:
    try:
        w13, w2 = _data(layer.w13_weight), _data(layer.w2_weight)
        w13_sf, w2_sf = _data(layer.w13_blockscale_swizzled), _data(layer.w2_blockscale_swizzled)
        rest = (_data(layer.g1_scale_c), _data(layer.g1_alphas), _data(layer.g2_alphas),
                _data(layer.w13_input_scale_quant), _data(layer.w2_input_scale_quant))
        return (
            w13.dtype == w2.dtype == torch.uint8
            and w13_sf.dtype == w2_sf.dtype == torch.float8_e4m3fn
            and all(torch.is_tensor(value) for value in (w13, w2, w13_sf, w2_sf, *rest))
            and int(_data(layer.intermediate_size_per_partition)) > 0
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _layer_view(layer: object, *, layout: str) -> SimpleNamespace:
    w13, w2 = _data(layer.w13_weight), _data(layer.w2_weight)
    w13_sf, w2_sf = _data(layer.w13_blockscale_swizzled), _data(layer.w2_blockscale_swizzled)
    experts = int(w13.shape[0])
    g1, g2 = _data(layer.g1_alphas), _data(layer.g2_alphas)
    g1_scale_c = _data(layer.g1_scale_c)
    a1, a2 = _data(layer.w13_input_scale_quant), _data(layer.w2_input_scale_quant)
    intermediate = int(_data(layer.intermediate_size_per_partition))
    top_k = int(_data(layer.top_k))
    tp_size = int(getattr(layer, "moe_tp_size", 1))
    fused_shared = int(getattr(layer, "num_fused_shared_experts", 0))
    runner_config = getattr(layer, "moe_runner_config", layer)
    return SimpleNamespace(
        w13_weight=w13, w2_weight=w2,
        w13_weight_scale=w13_sf, w2_weight_scale=w2_sf,
        w13_blockscale_swizzled=w13_sf, w2_blockscale_swizzled=w2_sf,
        g1_scale_c=g1_scale_c, g1_alphas=g1, g2_alphas=g2,
        w13_input_scale_quant=a1, w2_input_scale_quant=a2,
        fc1_input_dequant=a1.float().reciprocal(), fc1_dequant=g1,
        fc2_quant=a2, fc2_dequant=g2,
        intermediate_size_per_partition=intermediate,
        num_local_experts=experts, num_experts=experts,
        hidden_size=int(w13.shape[-1]) * 2,
        moe_ep_size=int(getattr(layer, "moe_ep_size", 1)),
        moe_ep_rank=int(getattr(layer, "moe_ep_rank", 0)),
        moe_tp_size=tp_size,
        reduce_results=bool(getattr(layer, "reduce_results", False)),
        num_fused_shared_experts=fused_shared,
        cacheon_group_size=16, cacheon_w13_layout=layout,
        moe_runner_config=SimpleNamespace(
            is_gated=True, num_experts=experts, top_k=top_k,
            hidden_size=int(w13.shape[-1]) * 2,
            intermediate_size_per_partition=intermediate,
            activation=str(getattr(runner_config, "activation", "swigluoai")),
            num_fused_shared_experts=fused_shared,
        ),
    )


def prepare_args_from_layer(layer: object) -> tuple[object, ...]:
    w13 = _data(layer.w13_weight)
    quantized = w13.dtype == torch.uint8 and (
        getattr(layer, "w13_weight_scale", None) is not None
        or getattr(layer, "g1_alphas", None) is not None
    )
    if not quantized:
        return w13, _data(layer.w2_weight)
    if not supports_layer(layer):
        raise ValueError("NVFP4 MoE layer is outside the canonical weight contract")
    if getattr(getattr(layer, "quant_method", None), "enable_flashinfer_trtllm_moe", False):
        layout = NVFP4_TRTLLM_LAYOUT
    elif getattr(layer, "w13_blockscale_mma", None) is not None:
        layout = NVFP4_INTERLEAVED_LAYOUT
    else:
        layout = NVFP4_GATE_UP_LAYOUT
    return NVFP4_PREPARE_TAG, _layer_view(layer, layout=layout)
