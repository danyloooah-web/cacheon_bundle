
from __future__ import annotations

import torch

ROUNDED = "rounded"
FP32 = "fp32"

_NORMS: dict = {}
_PENDING: dict = {}


def register_norm(key, weight: torch.Tensor, eps: float) -> None:
    _NORMS[key] = (weight, float(eps))


def next_norm(layer_id: int, num_layers: int):
    final = layer_id == num_layers - 1
    got = _NORMS.get("final" if final else layer_id + 1)
    return None if got is None else (got[0], got[1], final)


def publish(hidden: torch.Tensor, residual: torch.Tensor, normed: torch.Tensor,
            next_residual: torch.Tensor, weight: torch.Tensor, convention: str) -> None:
    _PENDING[hidden.device.index] = (hidden, residual, normed, next_residual, weight, convention)


def take(x, residual, weight: torch.Tensor, convention: str):
    if not torch.is_tensor(x) or not torch.is_tensor(residual):
        return None
    entry = _PENDING.get(x.device.index)
    if entry is None or entry[0] is not x or entry[1] is not residual:
        return None
    del _PENDING[x.device.index]
    _, _, normed, next_residual, published_weight, published_convention = entry
    exact = published_weight is weight and published_convention == convention
    return normed, next_residual, exact
