
from __future__ import annotations

import copy
import math
import sys

import torch
import triton
import triton.language as tl

MAX_ROWS = 32
_WARM_ROWS = (8, 16, 32)
_FP8_MAX = 448.0
_BLOCK = 4096
_PERMUTE: dict = {}
_announced: set = set()
_STASH: dict = {}


def _announce(tag: str, detail: str) -> None:
    if tag not in _announced:
        _announced.add(tag)
        print(f"CACHEON_FP8_DECODE {tag} {detail}", file=sys.stderr, flush=True)


@triton.jit
def _static_fp8(X, Y, inv, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32) * inv
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    tl.store(Y + offs, x.to(tl.float8e4nv), mask=mask)


def _prepared_weight(weight: torch.Tensor):
    from flashinfer.trtllm_low_latency_gemm import prepare_low_latency_gemm_weights

    w = weight.float()
    s_w = max(float(w.abs().amax()), 1e-12) / _FP8_MAX
    w8 = (w / s_w).to(torch.float8_e4m3fn)
    del w
    b = prepare_low_latency_gemm_weights(w8.view(torch.uint8), _PERMUTE).view(torch.float8_e4m3fn)
    del w8
    return b, s_w


def install_q_b(attn, gamma: torch.Tensor):
    from flashinfer.trtllm_low_latency_gemm import get_trtllm_low_latency_gemm_module
    from glm53_baseline import fp8w

    name = "q_b_proj"
    stock_module = attn._modules.get(name)
    why = fp8w._linear_reason(stock_module, (16384, 2048))
    if why is not None:
        _announce(f"static_stock_{name}", why)
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("fp8_decode: the e4m3 weights and the warm-up run in prepare, never under capture")
    weight = stock_module.weight
    n, k = weight.shape
    device = weight.device
    b, s_w = _prepared_weight(weight)
    s_a = math.sqrt(k) * max(float(gamma.float().abs().amax()), 1e-12) / _FP8_MAX
    scale = torch.tensor([s_a * s_w], dtype=torch.float32, device=device)
    inv_a = 1.0 / s_a
    runner = get_trtllm_low_latency_gemm_module().gemm_runner()
    for rows in _WARM_ROWS:
        x8 = torch.zeros((rows, k), dtype=torch.float8_e4m3fn, device=device)
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=device)
        runner(inputs=[x8, b, scale, out], tactic=-1)
    torch.cuda.synchronize(device)
    stock = stock_module.forward
    key = id(attn)

    def forward(input_, *args, **kwargs):
        rows = input_.shape[0] if input_.dim() == 2 else 0
        if (args or kwargs or not 1 <= rows <= MAX_ROWS or input_.shape[1] != k or input_.dtype != torch.bfloat16
                or not input_.is_contiguous()):
            return stock(input_, *args, **kwargs)
        got = _STASH.pop(key, None)
        if got is not None and got[0] is input_:
            x8 = got[1]
        else:
            x8 = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=input_.device)
            _static_fp8[(triton.cdiv(rows * k, _BLOCK),)](input_, x8, inv_a, rows * k, BLOCK=_BLOCK, num_warps=4)
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=input_.device)
        runner(inputs=[x8, b, scale, out], tactic=-1)
        return out, None

    private = copy.copy(stock_module)
    private.forward = forward
    attn._modules = dict(attn._modules)
    attn._modules[name] = private
    attn._use_min_latency_q_b_gemm = False
    _announce(f"served_{name}", f"rows <= {MAX_ROWS} through the e4m3 low-latency GEMM "
                                f"(per-tensor s_w {s_w:.4g}, static s_a {s_a:.4g})")
    return b, inv_a


def stash(attn, normed: torch.Tensor, x8: torch.Tensor) -> None:
    _STASH[id(attn)] = (normed, x8)
