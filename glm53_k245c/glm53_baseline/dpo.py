
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
from triton.tools.tensor_descriptor import TensorDescriptor

_WORLD = 4
_MCOL = _WORLD * 64
_MAX_ROWS = _MCOL // _WORLD
FUSED_MAX_ROWS = 6
FUSED = True
FUSED_UMMA_MIN_ROWS = 5

I8 = True
I8_MAX_ROWS = 8
_I8_PERM = (0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15)
_I8_COPIES: dict = {}

_NO_LOC: dict = {}


def _no_loc(device):
    t = _NO_LOC.get(device)
    if t is None:
        t = _NO_LOC[device] = torch.empty(0, dtype=torch.int64, device=device)
    return t


def int8_copy(shard: torch.Tensor):
    key = (shard.data_ptr(), tuple(shard.shape))
    copy = _I8_COPIES.get(key)
    if copy is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("int8 o_proj shard must be built before capture")
        copy = _I8_COPIES[key] = quantize_shard(shard.contiguous())
    return copy


def quantize_shard(shard: torch.Tensor):
    n, k = shard.shape
    w = shard.float().view(n, k // 128, 128)
    s = (w.abs().amax(-1) / 127.0).to(torch.float16)
    sf = s.float()
    q = torch.round(w / torch.where(sf > 0, sf, torch.ones_like(sf)).unsqueeze(-1)).clamp_(-127, 127)
    q = q.to(torch.int8).view(n, k)
    perm = torch.tensor(_I8_PERM, device=shard.device)
    qv = q.view(n // 8, 8, k // 16, 16)[..., perm].reshape(n // 8, 8, k // 64, 4, 4, 4)
    qv = qv.permute(0, 2, 1, 4, 3, 5).contiguous()
    return qv.view(torch.uint8).reshape(-1), s.contiguous()
FUSED_E4M3 = True
WIDE_SELF_FILL = False
_empty: dict = {}
_workspaces = {}
_descriptors = {}


@lru_cache(maxsize=1)
def _native():
    import dpo_native

    return dpo_native


@triton.jit
def _shard_gemm(x_desc, w_desc, ws_ptr, M, N, k_per_split,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    k0 = pid_k * k_per_split
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    gdc_wait()
    gdc_launch_dependents()
    for kk in range(0, k_per_split, BK):
        xt = x_desc.load([pid_m * BM, k0 + kk])
        w = w_desc.load([pid_n * BN, k0 + kk])
        acc = tl.dot(xt, w.T, acc)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    tl.atomic_add(ws_ptr + rm[:, None] * N + rn[None, :], acc,
                  mask=(rm[:, None] < M) & (rn[None, :] < N), sem="relaxed")


_CFG_VERIFY = (128, 64, 6, 8, 6)
_CFG_WIDE = (128, 64, 12, 8, 6)
_CFG_NARROW = (64, 128, 12, 4, 4)
_CFG_MID = (128, 128, 12, 4, 4)


def _gemm_plan(m, k):
    bn, bk, s, warps, stages = (_CFG_VERIFY if m > 128 else _CFG_WIDE if m > 64 else _CFG_MID if m > 32
                                else _CFG_NARROW)
    bm = 128 if m > 64 else 64 if m > 32 else 32
    kps = triton.cdiv(triton.cdiv(k, s), bk) * bk
    return bm, bn, bk, kps, triton.cdiv(k, kps), warps, stages, triton.cdiv(m, bm)


def _descriptor(t, shape, strides, block):
    key = (t.data_ptr(), tuple(shape), tuple(strides), tuple(block))
    d = _descriptors.get(key)
    if d is None:
        d = _descriptors[key] = TensorDescriptor(t, list(shape), list(strides), list(block))
    return d


def _shard_gemm_launch(gx, shard, acc):
    m, k = gx.shape
    n = shard.shape[0]
    bm, bn, bk, kps, s, warps, stages, mt = _gemm_plan(m, k)
    xd = _descriptor(gx, (m, k), (gx.stride(0), 1), (bm, bk))
    wd = _descriptor(shard, (n, k), (shard.stride(0), 1), (bn, bk))
    _shard_gemm[(mt, n // bn, s)](xd, wd, acc, m, n, kps,
                              BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages, launch_pdl=True)


@dataclass
class Prepared:

    weight: torch.Tensor
    gamma: torch.Tensor
    epsilon: float
    quant_scale: torch.Tensor
    w8: torch.Tensor | None = None
    s8: torch.Tensor | None = None


def prepare(weight, gamma, epsilon, quant_scale):
    if weight.dtype != torch.bfloat16 or weight.ndim != 2 or weight.stride(1) != 1:
        raise ValueError("projection requires a row-major BF16 weight matrix")
    if gamma.shape != weight.shape[:1] or quant_scale.numel() not in (0, 1):
        raise ValueError("normalization or quantization parameter shape differs")
    if weight.shape[0] % (_WORLD * 8) or weight.shape[1] % 8:
        raise ValueError("hidden width must split into 16-byte aligned column shards")
    _native()
    w8 = s8 = None
    if FUSED_E4M3:
        from glm53_baseline import fp8w

        w8, s8 = fp8w.quantized(weight)
    return Prepared(weight, gamma.contiguous(), float(epsilon), quant_scale, w8, s8)


class Workspace:

    def __init__(self, group, device, width, hidden):
        import torch.distributed._symmetric_memory as symmetric

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("projection workspace must be created before capture")
        native = _native()
        self.rings, self.handles, self.maps = [], [], []
        for size in (native.input_ring_bytes(width, hidden), native.column_ring_bytes(hidden)):
            ring = symmetric.empty(size, dtype=torch.uint8, device=device)
            ring.view(torch.int32).fill_(-2147483648)
            ring[: native.header_bytes()].zero_()
            torch.cuda.current_stream(device).synchronize()
            handle = symmetric.rendezvous(ring, group)
            if handle is None or handle.world_size != _WORLD:
                raise RuntimeError("projection requires four mapped peers")
            pointers = [handle.get_buffer(p, (size,), torch.uint8).data_ptr() for p in range(_WORLD)]
            self.rings.append(ring)
            self.handles.append(handle)
            self.maps.append(torch.tensor(pointers + [0], dtype=torch.int64, device=device))
        size = native.fused_ring_bytes()
        ring = symmetric.empty(size, dtype=torch.uint8, device=device)
        ring.view(torch.int32).fill_(-2147483648)
        ring[: native.header_bytes()].zero_()
        torch.cuda.current_stream(device).synchronize()
        handle = symmetric.rendezvous(ring, group)
        if handle is None or handle.world_size != _WORLD:
            raise RuntimeError("projection requires four mapped peers")
        pointers = [handle.get_buffer(p, (size,), torch.uint8).data_ptr() for p in range(_WORLD)]
        self.fused_ring, self.fused_handle = ring, handle
        self.fused_maps = torch.tensor(pointers + [0], dtype=torch.int64, device=device)
        self.nofill = torch.empty(64, dtype=torch.bfloat16, device=device)
        self.gx = torch.empty((_MCOL, width), dtype=torch.bfloat16, device=device)
        self.gr = torch.empty((_MCOL, hidden // _WORLD), dtype=torch.bfloat16, device=device)
        self.acc = torch.zeros((_MCOL, hidden // _WORLD), dtype=torch.float32, device=device)
        self.zero = torch.zeros(_MCOL, dtype=torch.uint8, device=device)


def _workspace(group, weight):
    key = (group, weight.device, tuple(weight.shape))
    ws = _workspaces.get(key)
    if ws is None:
        ws = _workspaces[key] = Workspace(group, weight.device, weight.shape[1], weight.shape[0])
    return ws


def fused_ring(group, weight):
    ws = _workspaces.get((group, weight.device, tuple(weight.shape)))
    if ws is None or dist.get_world_size(group) != _WORLD:
        return None
    return ws.fused_maps, dist.get_rank(group)


def project_gather_norm(x, residual, prepared, out, local_residual, packed, scales, group, loc=None, xpushed=False,
                        early_residual=False):
    if dist.get_world_size(group) != _WORLD or not 0 < x.shape[0] <= _MAX_ROWS:
        raise ValueError(f"projection requires TP4 and 1..{_MAX_ROWS} padded local rows")
    rank = dist.get_rank(group)
    weight, gamma = prepared.weight, prepared.gamma
    hidden, width = weight.shape
    if x.shape[1] != width or residual.shape[1] != hidden:
        raise ValueError("projection geometry is outside the prepared envelope")
    ws = _workspace(group, weight)
    native = _native()
    rows = x.shape[0]
    shard_rows = hidden // _WORLD
    shard = weight[rank * shard_rows:(rank + 1) * shard_rows]
    gx, gr = ws.gx[:_WORLD * rows], ws.gr[:_WORLD * rows]
    fused = FUSED and rows <= FUSED_MAX_ROWS
    if I8 and rows <= I8_MAX_ROWS:
        copy = int8_copy(shard)
        native.scatter_gemm_i8(x, residual, gr, copy[0], copy[1], ws.acc, ws.fused_maps, rank, True,
                               loc if loc is not None else _no_loc(x.device),
                               (1 if xpushed else 0) | (2 if early_residual else 0))
        fused = True
    elif xpushed or early_residual:
        raise ValueError("rows already sent to the peers belong to the INT8 fused scatter only")
    elif fused:
        umma = rows >= FUSED_UMMA_MIN_ROWS
        if prepared.w8 is not None and not umma:
            shard8 = prepared.w8[rank * shard_rows:(rank + 1) * shard_rows]
            scale8 = prepared.s8[rank * shard_rows:(rank + 1) * shard_rows]
        else:
            empty = _empty.get(x.device)
            if empty is None:
                empty = _empty[x.device] = (torch.empty(0, dtype=torch.uint8, device=x.device),
                                            torch.empty(0, dtype=torch.float32, device=x.device))
            shard8, scale8 = empty
        native.scatter_gemm(x, residual, gr, shard, ws.acc, ws.fused_maps, rank, True, umma, shard8, scale8)
    else:
        fill_src = shard if (rows <= 16 or WIDE_SELF_FILL) else ws.nofill
        native.scatter(x, residual, gx, gr, ws.maps[0], rank, fill_src, rows > 16, True, rows > 16,
                       loc if loc is not None else _no_loc(x.device))
        _shard_gemm_launch(gx, shard, ws.acc)
    native.merge(ws.acc, gr, gamma, out, local_residual, ws.maps[1], rank, prepared.epsilon,
                 packed, scales, prepared.quant_scale, True, fused, ws.zero)
    return ws.zero if _WORLD * rows <= 16 else None


def fuses_pad(rows: int) -> bool:
    return (I8 and rows <= I8_MAX_ROWS) or not (FUSED and rows <= FUSED_MAX_ROWS)
