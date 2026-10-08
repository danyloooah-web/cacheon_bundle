
from __future__ import annotations

import copy
import sys

import torch
import triton
import triton.language as tl

_FP8_MAX = 448.0
_announced: set = set()
_quantized: dict = {}


def _announce(tag: str, detail: str) -> None:
    if tag not in _announced:
        _announced.add(tag)
        print(f"CACHEON_FP8W {tag} {detail}", file=sys.stderr, flush=True)


@triton.jit(do_not_specialize=["rows"])
def _rowwise_fp8(X, Y, S, rows, K: tl.constexpr):
    r = tl.program_id(0)
    offs = r.to(tl.int64) * K + tl.arange(0, K)
    x = tl.load(X + offs).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 448.0, 1e-12)
    inv = 1.0 / scale
    tl.store(Y + offs, (x * inv).to(tl.float8e4nv))
    tl.store(S + r, scale)


def quantized(weight: torch.Tensor):
    key = (weight.data_ptr(), tuple(weight.shape))
    got = _quantized.get(key)
    if got is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("fp8w: e4m3 weights are made in prepare, never under stream capture")
        if weight.dtype != torch.bfloat16 or weight.dim() != 2 or not weight.is_contiguous():
            raise ValueError(f"fp8w: needs a contiguous bf16 [N, K] weight, got {weight.dtype} {tuple(weight.shape)}")
        w = weight.float()
        scale = (w.abs().amax(dim=1) / _FP8_MAX).clamp_min(1e-12)
        got = _quantized[key] = ((w / scale[:, None]).to(torch.float8_e4m3fn), scale.contiguous())
        del w
    return got


def matmul(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, k = x.shape
    x8 = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=x.device)
    xs = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    _rowwise_fp8[(rows,)](x, x8, xs, rows, K=k, num_warps=8)
    return torch._scaled_mm(x8, w8.t(), scale_a=xs, scale_b=scale.view(1, -1), out_dtype=torch.bfloat16)


def _linear_reason(module, shape) -> str | None:
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    weight = getattr(module, "weight", None)
    if module is None or weight is None:
        return "no weight"
    if not isinstance(getattr(module, "quant_method", None), UnquantizedLinearMethod):
        return "quantized"
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != shape or not weight.is_contiguous():
        return f"weight {weight.dtype} {tuple(weight.shape)} is not the bf16 {shape} this serves"
    if getattr(module, "bias", None) is not None or int(getattr(module, "tp_size", 1)) != 1:
        return "bias or a reduction across attention TP"
    return None


def _private_linear(attn, name: str, shape, eligible) -> bool:
    stock_module = attn._modules.get(name)
    why = _linear_reason(stock_module, shape)
    if why is not None:
        _announce(f"static_stock_{name}", why)
        return False
    stock = stock_module.forward
    w8, scale = quantized(stock_module.weight)
    k = shape[1]

    def forward(input_, *args, **kwargs):
        if (args or kwargs or input_.dim() != 2 or input_.shape[1] != k or input_.dtype != torch.bfloat16
                or not input_.is_contiguous() or torch.cuda.is_current_stream_capturing() or not eligible(input_)):
            return stock(input_, *args, **kwargs)
        _announce(f"served_{name}", f"{name} in FP8 e4m3 (rowwise activation and per-channel weight scales)")
        return matmul(input_, w8, scale), None

    private = copy.copy(stock_module)
    private.forward = forward
    attn._modules = dict(attn._modules)
    attn._modules[name] = private
    return True


def install_o_proj(attn) -> None:
    _private_linear(attn, "o_proj", (6144, 16384), lambda x: x.shape[0] >= 128)


def install_q_b(attn) -> None:
    if _private_linear(attn, "q_b_proj", (16384, 2048), lambda x: x.shape[0] >= 128):
        attn._use_min_latency_q_b_gemm = False
