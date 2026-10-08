
from __future__ import annotations

import sys

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from glm53_baseline import levers

try:
    import torch.distributed._symmetric_memory as _symm
except Exception:
    _symm = None

_WORLD = 4
_BLOCK = 2048
_WARPS = 8
_PROGS_PER_SM = 4
PART_TOK, PART_WARPS, PART_SMS = 4, 32, levers.PART_SMS
USE_NATIVE_PUSH = levers.NATIVE_PUSH


def _native_push():
    try:
        import prefill_push
    except Exception:
        return None
    return prefill_push


@triton.jit(do_not_specialize=["rank", "cap", "tiles", "parity", "cnt_off"])
def _push_kernel(GEMM2, IDX, W, PEERS, rank, cap, tiles, parity, g2_stride, cnt_off,
                 K: tl.constexpr, HIDDEN: tl.constexpr, BLOCK: tl.constexpr, CAP_MAX: tl.constexpr,
                 NPROG: tl.constexpr, THREADS: tl.constexpr, FENCE: tl.constexpr):
    NCB: tl.constexpr = HIDDEN // BLOCK
    SLOT: tl.constexpr = CAP_MAX * HIDDEN
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    for g in range(pid, tiles, NPROG):
        cb = g % NCB
        q = g // NCB
        owner = (rank + 1 + q % 4) % 4
        j = q // 4
        t = owner * cap + j
        c = cb * BLOCK + cols
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(K):
            idx = tl.load(IDX + t * K + k)
            valid = idx >= 0
            w = tl.where(valid, tl.load(W + t * K + k).to(tl.float32), 0.0)
            row = tl.where(valid, idx, 0).to(tl.int64)
            v = tl.load(GEMM2 + row * g2_stride + c, mask=valid, other=0.0,
                        eviction_policy="evict_first")
            acc += w * v.to(tl.float32)
        base = tl.multiple_of(tl.load(PEERS + owner).to(tl.pointer_type(tl.bfloat16)), 16)
        dst = base + (parity * (4 * SLOT) + rank * SLOT + tl.cast(j, tl.int64) * HIDDEN) + c
        tl.store(dst, acc.to(tl.bfloat16))
    if FENCE:
        lane = tl.arange(0, THREADS)
        tl.inline_asm_elementwise("fence.acq_rel.sys;\nmov.u32 $0, 0;", "=r,r", [lane],
                                  dtype=tl.int32, is_pure=False, pack=1)
    tl.debug_barrier()
    for o in tl.static_range(4):
        counter = (tl.load(PEERS + o) + cnt_off).to(tl.pointer_type(tl.int64))
        tl.atomic_add(counter, 1, sem="release", scope="sys")


@triton.jit(do_not_specialize=["rank", "cap", "tiles", "parity", "cnt_off"])
def _push_rows_kernel(GEMM2, IDX, W, PEERS, rank, cap, tiles, parity, g2_stride, cnt_off,
                      K: tl.constexpr, HIDDEN: tl.constexpr, BLOCK: tl.constexpr, CAP_MAX: tl.constexpr,
                      TOK: tl.constexpr, NPROG: tl.constexpr, THREADS: tl.constexpr,
                      FENCE: tl.constexpr):
    NCB: tl.constexpr = HIDDEN // BLOCK
    SLOT: tl.constexpr = CAP_MAX * HIDDEN
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    toks = tl.arange(0, TOK)
    for g in range(pid, tiles, NPROG):
        cb = g % NCB
        q = g // NCB
        owner = (rank + 1 + q % 4) % 4
        j = (q // 4) * TOK + toks
        live = j < cap
        t = owner * cap + j
        c = cb * BLOCK + cols
        acc = tl.zeros((TOK, BLOCK), dtype=tl.float32)
        for k in tl.static_range(K):
            idx = tl.load(IDX + t * K + k, mask=live, other=-1)
            valid = idx >= 0
            w = tl.where(valid, tl.load(W + t * K + k, mask=live, other=0.0).to(tl.float32), 0.0)
            row = tl.where(valid, idx, 0).to(tl.int64)
            v = tl.load(GEMM2 + row[:, None] * g2_stride + c[None, :], mask=valid[:, None],
                        other=0.0, eviction_policy="evict_first")
            acc += w[:, None] * v.to(tl.float32)
        base = tl.multiple_of(tl.load(PEERS + owner).to(tl.pointer_type(tl.bfloat16)), 16)
        dst = (base + (parity * (4 * SLOT) + rank * SLOT)
               + tl.cast(j, tl.int64)[:, None] * HIDDEN + c[None, :])
        tl.store(dst, acc.to(tl.bfloat16), mask=live[:, None])
    if FENCE:
        lane = tl.arange(0, THREADS)
        tl.inline_asm_elementwise("fence.acq_rel.sys;\nmov.u32 $0, 0;", "=r,r", [lane],
                                  dtype=tl.int32, is_pure=False, pack=1)
    tl.debug_barrier()
    for o in tl.static_range(4):
        counter = (tl.load(PEERS + o) + cnt_off).to(tl.pointer_type(tl.int64))
        tl.atomic_add(counter, 1, sem="release", scope="sys")


@triton.jit(do_not_specialize=["target", "rows", "nprog", "has_shared"])
def _reduce_kernel(RECV, SHARED, OUT, CNT, target, rows, nprog, has_shared,
                   HIDDEN: tl.constexpr, BLOCK: tl.constexpr, CAP_MAX: tl.constexpr):
    NCB: tl.constexpr = HIDDEN // BLOCK
    SLOT: tl.constexpr = CAP_MAX * HIDDEN
    seen = tl.atomic_add(CNT, 0, sem="acquire", scope="sys")
    while seen < target:
        seen = tl.atomic_add(CNT, 0, sem="acquire", scope="sys")
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    for g in range(pid, rows * NCB, nprog):
        off = tl.cast(g // NCB, tl.int64) * HIDDEN + (g % NCB) * BLOCK + cols
        s = tl.load(RECV + off, cache_modifier=".cg").to(tl.float32)
        s += tl.load(RECV + SLOT + off, cache_modifier=".cg").to(tl.float32)
        s += tl.load(RECV + 2 * SLOT + off, cache_modifier=".cg").to(tl.float32)
        s += tl.load(RECV + 3 * SLOT + off, cache_modifier=".cg").to(tl.float32)
        if has_shared:
            s += tl.load(SHARED + off).to(tl.float32)
        tl.store(OUT + off, s.to(tl.bfloat16))


@triton.jit(do_not_specialize=["target", "rows", "nprog", "has_shared"])
def _reduce_norm_kernel(RECV, SHARED, RES, GAMMA, OUT, NORMED, RESOUT, CNT, target, rows, nprog,
                        has_shared, eps,
                        HIDDEN: tl.constexpr, BLOCK: tl.constexpr, CAP_MAX: tl.constexpr,
                        ROUNDED: tl.constexpr):
    tl.static_assert(HIDDEN == 3 * BLOCK, "one row is three column blocks")
    SLOT: tl.constexpr = CAP_MAX * HIDDEN
    seen = tl.atomic_add(CNT, 0, sem="acquire", scope="sys")
    while seen < target:
        seen = tl.atomic_add(CNT, 0, sem="acquire", scope="sys")
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    w0 = tl.load(GAMMA + cols).to(tl.float32)
    w1 = tl.load(GAMMA + BLOCK + cols).to(tl.float32)
    w2 = tl.load(GAMMA + 2 * BLOCK + cols).to(tl.float32)
    for r in range(pid, rows, nprog):
        base = tl.cast(r, tl.int64) * HIDDEN
        o0 = base + cols
        o1 = base + BLOCK + cols
        o2 = base + 2 * BLOCK + cols
        s0 = tl.load(RECV + o0, cache_modifier=".cg").to(tl.float32)
        s1 = tl.load(RECV + o1, cache_modifier=".cg").to(tl.float32)
        s2 = tl.load(RECV + o2, cache_modifier=".cg").to(tl.float32)
        for src in tl.static_range(1, 4):
            s0 += tl.load(RECV + src * SLOT + o0, cache_modifier=".cg").to(tl.float32)
            s1 += tl.load(RECV + src * SLOT + o1, cache_modifier=".cg").to(tl.float32)
            s2 += tl.load(RECV + src * SLOT + o2, cache_modifier=".cg").to(tl.float32)
        if has_shared:
            s0 += tl.load(SHARED + o0).to(tl.float32)
            s1 += tl.load(SHARED + o1).to(tl.float32)
            s2 += tl.load(SHARED + o2).to(tl.float32)
        h0 = s0.to(tl.bfloat16).to(tl.float32) + tl.load(RES + o0).to(tl.float32)
        h1 = s1.to(tl.bfloat16).to(tl.float32) + tl.load(RES + o1).to(tl.float32)
        h2 = s2.to(tl.bfloat16).to(tl.float32) + tl.load(RES + o2).to(tl.float32)
        tl.store(OUT + o0, s0.to(tl.bfloat16))
        tl.store(OUT + o1, s1.to(tl.bfloat16))
        tl.store(OUT + o2, s2.to(tl.bfloat16))
        b0 = h0.to(tl.bfloat16)
        b1 = h1.to(tl.bfloat16)
        b2 = h2.to(tl.bfloat16)
        tl.store(RESOUT + o0, b0)
        tl.store(RESOUT + o1, b1)
        tl.store(RESOUT + o2, b2)
        if ROUNDED:
            h0 = b0.to(tl.float32)
            h1 = b1.to(tl.float32)
            h2 = b2.to(tl.float32)
        sq = tl.sum(h0 * h0, axis=0) + tl.sum(h1 * h1, axis=0) + tl.sum(h2 * h2, axis=0)
        inv = tl.rsqrt(sq * (1.0 / HIDDEN) + eps)
        tl.store(NORMED + o0, ((h0 * inv) * w0).to(tl.bfloat16))
        tl.store(NORMED + o1, ((h1 * inv) * w1).to(tl.bfloat16))
        tl.store(NORMED + o2, ((h2 * inv) * w2).to(tl.bfloat16))


class PrefillTail:

    def __init__(self, group, device, hidden: int, cap_max: int, top_k: int) -> None:
        if _symm is None:
            raise RuntimeError("torch.distributed._symmetric_memory is unavailable")
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if self.world != _WORLD:
            raise ValueError(f"the prefill tail is world {_WORLD} only, got {self.world}")
        if hidden % _BLOCK:
            raise ValueError(f"hidden {hidden} is not a multiple of {_BLOCK}")
        self.hidden, self.cap_max, self.top_k = int(hidden), int(cap_max), int(top_k)
        self.slot_elems = self.cap_max * self.hidden
        self.data_bytes = 2 * _WORLD * self.slot_elems * 2
        self.cnt_bytes = self.data_bytes
        total = self.data_bytes + 256
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.nprog = sms * _PROGS_PER_SM
        grids = torch.tensor([self.nprog, -self.nprog], dtype=torch.int64, device=device)
        dist.all_reduce(grids, op=dist.ReduceOp.MAX, group=group)
        if int(grids[0]) != -int(grids[1]):
            raise RuntimeError(f"push grids differ across ranks: {int(-grids[1])}..{int(grids[0])}")
        buf = _symm.empty(total, dtype=torch.uint8, device=device)
        buf[self.data_bytes:].zero_()
        torch.cuda.current_stream(device).synchronize()
        handle = _symm.rendezvous(buf, group)
        if handle is None or handle.world_size != self.world:
            raise RuntimeError("symmetric memory rendezvous did not cover the group")
        peers = [handle.get_buffer(peer, (total,), torch.uint8).data_ptr() for peer in range(self.world)]
        if any(p % 256 for p in peers):
            raise RuntimeError("symmetric buffers are not 256-byte aligned")
        self.buf, self.handle = buf, handle
        self.peer_list = [int(p) for p in peers]
        self.peers = torch.tensor(peers, dtype=torch.int64, device=device)
        self.recv = buf[:self.data_bytes].view(torch.bfloat16)
        self.counters = buf[self.data_bytes:self.data_bytes + 256].view(torch.int64)
        self.dummy_out = torch.empty((1, self.hidden), dtype=torch.bfloat16, device=device)
        self.calls = 0
        self.expected = [0, 0]
        self._pending = None
        self.part_sms = min(PART_SMS, sms)
        self.sms = sms
        native = _native_push() if USE_NATIVE_PUSH else None
        can = torch.tensor([1 if native is not None else 0], dtype=torch.int32, device=device)
        dist.all_reduce(can, op=dist.ReduceOp.MIN, group=group)
        self.native = native if int(can.item()) == 1 else None
        dist.barrier(group=group)

    def push(self, gemm2, mapping, weights, cap: int, partitioned: bool = False, rows=None) -> None:
        if gemm2.dtype != torch.bfloat16 or gemm2.dim() != 2 or gemm2.shape[1] != self.hidden \
                or gemm2.stride(1) != 1:
            raise RuntimeError(f"unexpected deferred GEMM2 output {tuple(gemm2.shape)} {gemm2.dtype}")
        total = cap * _WORLD
        if cap > self.cap_max or mapping.dtype != torch.int32 or mapping.numel() < total * self.top_k:
            raise RuntimeError(f"unexpected map {tuple(mapping.shape)} {mapping.dtype} for cap {cap}")
        if self._pending is not None:
            raise RuntimeError("prefill tail: push issued twice without a reduce")
        parity = self.calls & 1
        self.calls += 1
        ncb = self.hidden // _BLOCK
        if self.native is not None:
            grid = self.part_sms if partitioned else self.sms
            owners = [cap] * _WORLD if rows is None else [int(v) for v in rows]
            self.native.push(gemm2, mapping, weights, self.peer_list, self.rank, cap, parity,
                             self.cnt_bytes + 128 * parity, self.cap_max, grid, owners)
        elif partitioned:
            grid = self.part_sms
            _push_rows_kernel[(grid,)](
                gemm2, mapping, weights, self.peers, self.rank, cap,
                _WORLD * triton.cdiv(cap, PART_TOK) * ncb, parity, gemm2.stride(0),
                self.cnt_bytes + 128 * parity,
                K=self.top_k, HIDDEN=self.hidden, BLOCK=_BLOCK, CAP_MAX=self.cap_max, TOK=PART_TOK,
                NPROG=grid, THREADS=32 * PART_WARPS, FENCE=True, num_warps=PART_WARPS)
        else:
            grid = self.nprog
            _push_kernel[(grid,)](
                gemm2, mapping, weights, self.peers, self.rank, cap, total * ncb,
                parity, gemm2.stride(0), self.cnt_bytes + 128 * parity,
                K=self.top_k, HIDDEN=self.hidden, BLOCK=_BLOCK, CAP_MAX=self.cap_max,
                NPROG=grid, THREADS=32 * _WARPS, FENCE=True, num_warps=_WARPS)
        self.expected[parity] += _WORLD * grid
        self._pending = (parity, self.expected[parity])

    def reduce(self, shared, out, norm=None) -> None:
        if self._pending is None:
            raise RuntimeError("prefill tail: reduce without a push")
        parity, target = self._pending
        self._pending = None
        rows = int(out.shape[0])
        recv = self.recv[parity * _WORLD * self.slot_elems:]
        grid = max(1, min(self.nprog, rows * (self.hidden // _BLOCK)))
        if norm is not None and rows > 0:
            residual, gamma, eps, normed, residual_out, rounded = norm
            _reduce_norm_kernel[(min(self.nprog, rows),)](
                recv, shared if shared is not None else out, residual, gamma, out, normed, residual_out,
                self.counters[16 * parity:], target, rows, min(self.nprog, rows),
                int(shared is not None), float(eps),
                HIDDEN=self.hidden, BLOCK=_BLOCK, CAP_MAX=self.cap_max, ROUNDED=bool(rounded),
                num_warps=_WARPS)
            return
        _reduce_kernel[(grid,)](
            recv, shared if shared is not None else out, out, self.counters[16 * parity:], target, rows,
            grid, int(shared is not None and rows > 0),
            HIDDEN=self.hidden, BLOCK=_BLOCK, CAP_MAX=self.cap_max, num_warps=_WARPS)

    def run(self, gemm2, mapping, weights, cap: int, shared, out, norm=None,
            partitioned: bool = False, rows=None) -> None:
        self.push(gemm2, mapping, weights, cap, partitioned, rows)
        self.reduce(shared, out, norm)


_STATE: dict = {}
_WARNED: set = set()


def _warn_once(message: str) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        print(f"CACHEON_PREFILL_TAIL {message}", file=sys.stderr, flush=True)


def state_for(group, device, hidden: int, cap_max: int, top_k: int):
    key = (id(group), device.index, int(hidden), int(cap_max), int(top_k))
    if key in _STATE:
        return _STATE[key]
    if _symm is None or not dist.is_initialized() or torch.cuda.is_current_stream_capturing():
        return None
    state, why = None, ""
    try:
        state = PrefillTail(group, device, hidden, cap_max, top_k)
    except Exception as exc:
        why = f"{type(exc).__name__}: {exc}"
    ok = torch.tensor([1 if state is not None else 0], dtype=torch.int32, device=device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    if int(ok.item()) != 1:
        _warn_once(f"push tail off on every rank (this rank: {why or 'built'}); u2's tail serves")
        state = None
    else:
        _warn_once(f"push tail on: grid {state.nprog}, {state.data_bytes >> 20} MiB of receive slots")
    _STATE[key] = state
    return state
