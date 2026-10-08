
from __future__ import annotations

import sys

import torch
import torch.distributed as dist
import triton
import triton.language as tl

try:
    import torch.distributed._symmetric_memory as _symm
except Exception:
    _symm = None

_WORLD = 4
_FP4_WORDS, _SCALE_WORDS, _LOGIT_WORDS = 3072 // 4, 384 // 4, 256
_BLOCK = 2048
_WARPS = 8


@triton.jit
def _copy_to_all(SRC, PEERS, n, dst_off, pid, NPROG: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for start in range(pid * BLOCK, n, NPROG * BLOCK):
        j = start + offs
        inside = j < n
        v = tl.load(SRC + j, mask=inside, other=0)
        for d in tl.static_range(4):
            base = tl.multiple_of(tl.load(PEERS + d).to(tl.pointer_type(tl.int32)), 16)
            tl.store(base + dst_off + j, v, mask=inside)


@triton.jit(do_not_specialize=["rank", "cap", "rows", "parity", "cnt_off"])
def _push_kernel(FP4, SCALE, LOGITS, PEERS, rank, cap, rows, parity, cnt_off,
                 CAP_MAX: tl.constexpr, NPROG: tl.constexpr, BLOCK: tl.constexpr, THREADS: tl.constexpr):
    WF: tl.constexpr = 768
    WS: tl.constexpr = 96
    WL: tl.constexpr = 256
    SLOT_F: tl.constexpr = 4 * CAP_MAX * WF
    SLOT_S: tl.constexpr = 4 * CAP_MAX * WS
    SLOT: tl.constexpr = SLOT_F + SLOT_S + 4 * CAP_MAX * WL
    pid = tl.program_id(0)
    base = tl.cast(parity, tl.int64) * SLOT
    _copy_to_all(FP4, PEERS, rows * WF, base + tl.cast(rank * cap, tl.int64) * WF, pid, NPROG, BLOCK)
    _copy_to_all(SCALE, PEERS, rows * WS, base + SLOT_F + tl.cast(rank * cap, tl.int64) * WS, pid, NPROG, BLOCK)
    _copy_to_all(LOGITS, PEERS, rows * WL, base + SLOT_F + SLOT_S + tl.cast(rank * cap, tl.int64) * WL,
                 pid, NPROG, BLOCK)
    lane = tl.arange(0, THREADS)
    tl.inline_asm_elementwise("fence.acq_rel.sys;\nmov.u32 $0, 0;", "=r,r", [lane],
                              dtype=tl.int32, is_pure=False, pack=1)
    tl.debug_barrier()
    for o in tl.static_range(4):
        counter = (tl.load(PEERS + o) + cnt_off).to(tl.pointer_type(tl.int64))
        tl.atomic_add(counter, 1, sem="release", scope="sys")


@triton.jit(do_not_specialize=["target"])
def _wait_kernel(CNT, target):
    seen = tl.atomic_add(CNT, 0, sem="acquire", scope="sys")
    while seen < target:
        seen = tl.atomic_add(CNT, 0, sem="acquire", scope="sys")


class PrefillGather:

    def __init__(self, group, device, cap_max: int) -> None:
        if _symm is None:
            raise RuntimeError("torch.distributed._symmetric_memory is unavailable")
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        if self.world != _WORLD:
            raise ValueError(f"the prefill gather is world {_WORLD} only, got {self.world}")
        self.cap_max = int(cap_max)
        rows = _WORLD * self.cap_max
        self.fp4_bytes, self.scale_bytes = rows * _FP4_WORDS * 4, rows * _SCALE_WORDS * 4
        self.slot_bytes = self.fp4_bytes + self.scale_bytes + rows * _LOGIT_WORDS * 4
        self.cnt_bytes = 2 * self.slot_bytes
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.nprog = sms
        grids = torch.tensor([self.nprog, -self.nprog], dtype=torch.int64, device=device)
        dist.all_reduce(grids, op=dist.ReduceOp.MAX, group=group)
        if int(grids[0]) != -int(grids[1]):
            raise RuntimeError(f"gather grids differ across ranks: {int(-grids[1])}..{int(grids[0])}")
        buf = _symm.empty(self.cnt_bytes + 256, dtype=torch.uint8, device=device)
        buf[self.cnt_bytes:].zero_()
        torch.cuda.current_stream(device).synchronize()
        handle = _symm.rendezvous(buf, group)
        if handle is None or handle.world_size != self.world:
            raise RuntimeError("symmetric memory rendezvous did not cover the group")
        peers = [handle.get_buffer(peer, (self.cnt_bytes + 256,), torch.uint8).data_ptr()
                 for peer in range(self.world)]
        if any(p % 256 for p in peers):
            raise RuntimeError("symmetric buffers are not 256-byte aligned")
        self.buf, self.handle = buf, handle
        self.peers = torch.tensor(peers, dtype=torch.int64, device=device)
        self.counters = buf[self.cnt_bytes:self.cnt_bytes + 256].view(torch.int64)
        self.idle = torch.zeros(4, dtype=torch.int32, device=device)
        self.calls = 0
        self.expected = [0, 0]
        self._pending = None
        dist.barrier(group=group)

    def push(self, fp4, scale, logits, rows: int, cap: int) -> None:
        if not 0 <= rows <= cap <= self.cap_max:
            raise RuntimeError(f"prefill gather: rows {rows} cap {cap} outside [0, {self.cap_max}]")
        if self._pending is not None:
            raise RuntimeError("prefill gather: push issued twice without a wait")
        if rows:
            if (tuple(fp4.shape), tuple(scale.shape), tuple(logits.shape)) != ((rows, 3072), (rows, 384), (rows, 256)) \
                    or logits.dtype != torch.float32 or not (fp4.is_contiguous() and scale.is_contiguous()
                                                             and logits.is_contiguous()):
                raise RuntimeError(f"prefill gather: unexpected sources {tuple(fp4.shape)} {tuple(scale.shape)} "
                                   f"{tuple(logits.shape)} {logits.dtype}")
            srcs = (fp4.view(torch.int32), scale.view(torch.uint8).view(torch.int32), logits.view(torch.int32))
        else:
            srcs = (self.idle, self.idle, self.idle)
        parity = self.calls & 1
        self.calls += 1
        _push_kernel[(self.nprog,)](
            *srcs, self.peers, self.rank, cap, rows, parity, self.cnt_bytes + 128 * parity,
            CAP_MAX=self.cap_max, NPROG=self.nprog, BLOCK=_BLOCK, THREADS=32 * _WARPS, num_warps=_WARPS)
        self.expected[parity] += _WORLD * self.nprog
        self._pending = (parity, self.expected[parity], cap)

    def wait(self):
        if self._pending is None:
            raise RuntimeError("prefill gather: wait without a push")
        parity, target, cap = self._pending
        self._pending = None
        _wait_kernel[(1,)](self.counters[16 * parity:], target)
        total = _WORLD * cap
        base = parity * self.slot_bytes
        fp4 = self.buf[base:base + total * 3072].view(total, 3072)
        base += self.fp4_bytes
        scale = self.buf[base:base + total * 384].view(torch.float8_e4m3fn).view(total, 384)
        base += self.scale_bytes
        logits = self.buf[base:base + total * 1024].view(torch.float32).view(total, 256)
        return fp4, scale, logits


_STATE: dict = {}


def state_for(group, device, cap_max: int) -> PrefillGather:
    key = (id(group), device.index, int(cap_max))
    state = _STATE.get(key)
    if state is not None:
        return state
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prefill gather: first built under stream capture")
    why = ""
    try:
        state = PrefillGather(group, device, cap_max)
    except Exception as exc:
        why = f"{type(exc).__name__}: {exc}"
    ok = torch.tensor([1 if state is not None else 0], dtype=torch.int32, device=device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    if int(ok.item()) != 1:
        raise RuntimeError(f"prefill gather could not be built on every rank (this rank: {why or 'built'})")
    print(f"CACHEON_PREFILL_GATHER on: grid {state.nprog}, {2 * state.slot_bytes >> 20} MiB of receive slots",
          file=sys.stderr, flush=True)
    _STATE[key] = state
    return state
