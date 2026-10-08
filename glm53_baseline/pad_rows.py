
import sys

import torch
import triton
import triton.language as tl

from glm53_baseline.fused_add_rmsnorm import _pdl_trigger, _pdl_wait

_BLOCK = 2048
_announced: set = set()


def _announce(tag: str, detail: str) -> None:
    if tag not in _announced:
        _announced.add(tag)
        print(f"CACHEON_PAD_ROWS {tag} {detail}", file=sys.stderr, flush=True)


@triton.jit
def _zero_rows(X, R, LOC, SX, SR, HX: tl.constexpr, HR: tl.constexpr, BLOCK: tl.constexpr,
               USE_PDL: tl.constexpr):
    row = tl.program_id(0)
    if USE_PDL:
        row = _pdl_wait(row)
        row = _pdl_trigger(row)
    if tl.load(LOC + row) == 0:
        for start in tl.static_range(0, HX, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            tl.store(X + row * SX + cols, tl.zeros([BLOCK], X.dtype.element_ty), mask=cols < HX)
        for start in tl.static_range(0, HR, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            tl.store(R + row * SR + cols, tl.zeros([BLOCK], R.dtype.element_ty), mask=cols < HR)


def _matches(rows: torch.Tensor, cache_loc) -> bool:
    return (cache_loc is not None and cache_loc.dim() == 1 and cache_loc.shape[0] == rows.shape[0]
            and rows.dim() == 2 and rows.stride(1) == 1 and rows.dtype == torch.bfloat16)


def zero_padding(x: torch.Tensor, residual: torch.Tensor, cache_loc) -> None:
    if not (_matches(x, cache_loc) and _matches(residual, cache_loc)) or x.shape[0] == 0:
        _announce("zero_skipped", f"rows {tuple(x.shape)} residual {tuple(residual.shape)} cache_loc "
                                  f"{None if cache_loc is None else tuple(cache_loc.shape)}")
        return
    _announce("zero_served", f"{x.shape[0]} local rows, x width {x.shape[1]}")
    _zero_rows[(x.shape[0],)](
        x, residual, cache_loc, x.stride(0), residual.stride(0), x.shape[1], residual.shape[1], _BLOCK, True,
        num_warps=4, launch_pdl=True)
