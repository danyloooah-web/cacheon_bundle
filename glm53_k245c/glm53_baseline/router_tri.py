
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
from triton.tools.tensor_descriptor import TensorDescriptor

ENABLED = True
MIN_ROWS = 17
MAX_ROWS = 256
SPLITS = 24
BK = 64
STAGES = 3
WARPS = 4
_E = tl.constexpr(256)
_H = 6144
_descriptors: dict = {}


@triton.jit
def _router_splitk(x_desc, w_desc, out_ptr, M, K_PER: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr, STAGES: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k0 = pid_k * K_PER
    acc = tl.zeros([BN, BM], dtype=tl.float32)
    gdc_wait()
    gdc_launch_dependents()
    for kk in tl.range(0, K_PER, BK, num_stages=STAGES):
        w = w_desc.load([pid_n * BN, k0 + kk])
        x = x_desc.load([0, k0 + kk])
        acc = tl.dot(w, x.T, acc)
    re = pid_n * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    tl.atomic_add(out_ptr + rm[None, :] * _E + re[:, None], acc, mask=rm[None, :] < M, sem="relaxed")


def _descriptor(t, shape, block):
    key = (t.data_ptr(), tuple(shape), tuple(block))
    d = _descriptors.get(key)
    if d is None:
        d = _descriptors[key] = TensorDescriptor(t, list(shape), [shape[1], 1], list(block))
    return d


_SUMS: dict = {}


def _sum_buffer(device):
    b = _SUMS.get(device)
    if b is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("router_tri: the logits buffer must exist before capture")
        b = _SUMS[device] = torch.zeros((MAX_ROWS, 256), dtype=torch.float32, device=device)
    return b


def prepare_buffer(device):
    if ENABLED:
        _sum_buffer(device)


def slabs(x: torch.Tensor, weight: torch.Tensor):
    if (not ENABLED or x.dim() != 2 or not MIN_ROWS <= x.shape[0] <= MAX_ROWS or x.shape[1] != _H
            or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16
            or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (256, _H) or not weight.is_contiguous()):
        return None
    m = x.shape[0]
    bm = triton.next_power_of_2(m)
    out = _sum_buffer(x.device)[:m]
    xd = _descriptor(x, (m, _H), (bm, BK))
    wd = _descriptor(weight, (256, _H), (128, BK))
    _router_splitk[(2, SPLITS)](xd, wd, out, m, K_PER=_H // SPLITS, BM=bm, BN=128, BK=BK, STAGES=STAGES,
                                num_warps=WARPS, launch_pdl=True)
    return out
