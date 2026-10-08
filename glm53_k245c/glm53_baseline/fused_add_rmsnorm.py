from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _pdl_wait(row):
    return tl.inline_asm_elementwise(
        "griddepcontrol.wait; mov.b32 $0, $1;", "=r,r", [row],
        dtype=tl.int32, is_pure=False, pack=1,
    )


@triton.jit
def _pdl_trigger(row):
    return tl.inline_asm_elementwise(
        "griddepcontrol.launch_dependents; mov.b32 $0, $1;", "=r,r", [row],
        dtype=tl.int32, is_pure=False, pack=1,
    )


@triton.jit
def _norm(
    X, R, W, Y, S, Y8, INV8,
    SX: tl.constexpr, SR: tl.constexpr, SY: tl.constexpr, SS: tl.constexpr,
    CX: tl.constexpr, CR: tl.constexpr, CW: tl.constexpr,
    CY: tl.constexpr, CS: tl.constexpr,
    H: tl.constexpr, EPS: tl.constexpr, RES: tl.constexpr,
    BLOCK: tl.constexpr, USE_PDL: tl.constexpr, FP8: tl.constexpr,
):
    row = tl.program_id(0)
    if USE_PDL:
        row = _pdl_wait(row)
    cols = tl.arange(0, BLOCK)
    cols = tl.max_contiguous(tl.multiple_of(cols, 8), BLOCK)
    mask = cols < H
    value = tl.load(X + row * SX + cols * CX, mask, other=0).to(tl.float32)
    if RES:
        residual = tl.load(R + row * SR + cols * CR, mask, other=0).to(tl.float32)
        rounded = (value + residual).to(X.dtype.element_ty)
        tl.store(S + row * SS + cols * CS, rounded, mask)
        value = rounded.to(tl.float32)
    variance = tl.sum(value * value, axis=0) * (1.0 / H)
    inv = tl.rsqrt(variance + EPS)
    weight = tl.load(W + cols * CW, mask, other=0).to(tl.float32)
    y = (value * inv) * weight
    tl.store(Y + row * SY + cols * CY, y, mask)
    if FP8:
        yb = y.to(Y.dtype.element_ty).to(tl.float32) * INV8
        tl.store(Y8 + row * H + cols, tl.minimum(tl.maximum(yb, -448.0), 448.0).to(tl.float8e4nv), mask)
    if USE_PDL:
        _pdl_trigger(row)


@functools.lru_cache(maxsize=8)
def _multiprocessors(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _launch_shape(rows: int, hidden: int, sms: int) -> tuple[int, int, bool]:
    block = triton.next_power_of_2(hidden)
    if hidden < 1024:
        return block, 1, True
    if rows < sms:
        return block, min(32, block // 256), True
    if rows < 2 * sms:
        return block, min(16, block // 256), False
    if rows < 6 * sms:
        return block, min(8, block // 256), False
    return block, min(4, block // 256), False


def fused_add_rmsnorm(x, residual, weight, eps, out_norm, out_residual, out_fp8=None, inv_fp8=0.0):
    hidden = x.shape[-1]
    x2 = x.view(-1, hidden)
    y2 = out_norm.view(-1, hidden)
    has_residual = residual is not None
    r2 = residual.view(-1, hidden) if has_residual else x2
    s2 = out_residual.view(-1, hidden) if has_residual else y2
    block, warps, decode = _launch_shape(
        x2.shape[0], hidden, _multiprocessors(x2.device.index))
    y8 = out_fp8.view(-1, hidden) if out_fp8 is not None else y2
    _norm[(x2.shape[0],)](
        x2, r2, weight, y2, s2, y8, float(inv_fp8),
        x2.stride(0), r2.stride(0), y2.stride(0), s2.stride(0),
        x2.stride(1), r2.stride(1), weight.stride(0), y2.stride(1), s2.stride(1),
        hidden, float(eps), has_residual, block, decode, out_fp8 is not None,
        num_warps=warps, launch_pdl=decode,
        enable_fp_fusion=decode,
    )
