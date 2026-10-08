
from __future__ import annotations

import sys

import torch
import torch.distributed as dist

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

try:
    import torch.distributed._symmetric_memory as _symm

    _HAS_SYMM = True
except Exception:
    _HAS_SYMM = False

try:
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

    _HAS_GDC = True
except Exception:
    _HAS_GDC = False

USE_PDL = True

_NUM_SLOTS = 3
_MAX_WORLD = 8
_CTAS_PER_PEER = 12
_MAX_RESIDENT_CTAS_PER_SM = 2
_CTA_BYTES = 16384
_MAX_CPP = 64
FAST_PATH_MAX_TOKENS = 512
_BLOCK_WORDS = 1024
NUM_WARPS = 8
USE_LANE_KERNEL = True
LANE_WARPS = 4
_LANE_MAX_GRID = 65536
_ALIGN = 256
_SENTINEL = -2147483648

_ELEM_F32 = 0
_ELEM_BF16 = 1
_ELEM_F16 = 2

_STATES: dict = {}
_GROUP_REFS: dict = {}


if _HAS_TRITON:

    @triton.jit
    def _ld_v4(addr):

        return tl.inline_asm_elementwise(
            "ld.global.v4.b32 {$0,$1,$2,$3}, [$4];", "=r,=r,=r,=r,l", [addr],
            dtype=(tl.int32, tl.int32, tl.int32, tl.int32), is_pure=False, pack=1,
        )

    @triton.jit
    def _st_v4_volatile(addr, w0, w1, w2, w3):

        return tl.inline_asm_elementwise(
            "st.volatile.global.v4.b32 [$1], {$2,$3,$4,$5}; mov.u32 $0, 0;", "=r,l,r,r,r,r",
            [addr, w0, w1, w2, w3], dtype=tl.int32, is_pure=False, pack=1,
        )

    @triton.jit
    def _poll3_v4(a0, a1, a2, sent):

        return tl.inline_asm_elementwise(
            "{\n"
            " .reg .u32 w<12>;\n .reg .pred p, q;\n"
            "KING_POLL:\n"
            " ld.volatile.global.v4.b32 {w0,w1,w2,w3}, [$12];\n"
            " ld.volatile.global.v4.b32 {w4,w5,w6,w7}, [$13];\n"
            " ld.volatile.global.v4.b32 {w8,w9,w10,w11}, [$14];\n"
            " setp.eq.u32 p, w0, $15;\n"
            " setp.eq.u32 q, w1, $15; or.pred p, p, q;\n setp.eq.u32 q, w2, $15; or.pred p, p, q;\n"
            " setp.eq.u32 q, w3, $15; or.pred p, p, q;\n setp.eq.u32 q, w4, $15; or.pred p, p, q;\n"
            " setp.eq.u32 q, w5, $15; or.pred p, p, q;\n setp.eq.u32 q, w6, $15; or.pred p, p, q;\n"
            " setp.eq.u32 q, w7, $15; or.pred p, p, q;\n setp.eq.u32 q, w8, $15; or.pred p, p, q;\n"
            " setp.eq.u32 q, w9, $15; or.pred p, p, q;\n setp.eq.u32 q, w10, $15; or.pred p, p, q;\n"
            " setp.eq.u32 q, w11, $15; or.pred p, p, q;\n"
            " @p bra KING_POLL;\n"
            " mov.u32 $0, w0; mov.u32 $1, w1; mov.u32 $2, w2; mov.u32 $3, w3;\n"
            " mov.u32 $4, w4; mov.u32 $5, w5; mov.u32 $6, w6; mov.u32 $7, w7;\n"
            " mov.u32 $8, w8; mov.u32 $9, w9; mov.u32 $10, w10; mov.u32 $11, w11;\n"
            "}",
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,l,l,l,r", [a0, a1, a2, sent],
            dtype=(tl.int32, tl.int32, tl.int32, tl.int32, tl.int32, tl.int32,
                   tl.int32, tl.int32, tl.int32, tl.int32, tl.int32, tl.int32),
            is_pure=False, pack=1,
        )

    @triton.jit
    def _sum4_words(a, b, c, d, ELEM: tl.constexpr):

        lo = tl.zeros_like(a).to(tl.float32)
        hi = tl.zeros_like(a).to(tl.float32)
        lo, hi = _unpack_sum(lo, hi, a, ELEM)
        lo, hi = _unpack_sum(lo, hi, b, ELEM)
        lo, hi = _unpack_sum(lo, hi, c, ELEM)
        lo, hi = _unpack_sum(lo, hi, d, ELEM)
        return _pack(lo, hi, ELEM)

    @triton.jit
    def _lane_kernel(
        x_ptr, out_ptr, peers_ptr, phase_ptr, rank, chunk_words, slot_bytes, scratch_addr, trash_addr,
        REDUCE: tl.constexpr, ELEM: tl.constexpr, SENT: tl.constexpr, BLOCK_VEC: tl.constexpr, USE_PDL: tl.constexpr,
    ):
        if USE_PDL:
            gdc_wait()
            gdc_launch_dependents()
        pid = tl.program_id(0)
        phase = tl.load(phase_ptr + pid)
        slot = phase % 3
        n_vec = chunk_words // 4
        v = pid * BLOCK_VEC + tl.arange(0, BLOCK_VEC)
        live = v < n_vec
        woff = v.to(tl.int64) * 16
        chunk_bytes = chunk_words * 4
        r1 = (rank + 1) % 4
        r2 = (rank + 2) % 4
        r3 = (rank + 3) % 4
        my_base = tl.load(peers_ptr + rank)
        b1 = tl.load(peers_ptr + r1)
        b2 = tl.load(peers_ptr + r2)
        b3 = tl.load(peers_ptr + r3)
        xb = x_ptr.to(tl.int64)
        ob = out_ptr.to(tl.int64)
        region = slot * slot_bytes + rank * chunk_bytes

        if REDUCE == 0:
            src = tl.where(live, xb + woff, scratch_addr)
            m0, m1, m2, m3 = _ld_v4(src)
            s0 = _sanitize(m0, SENT)
            s1 = _sanitize(m1, SENT)
            s2 = _sanitize(m2, SENT)
            s3 = _sanitize(m3, SENT)
            _st_v4_volatile(tl.where(live, b1 + region + woff, trash_addr), s0, s1, s2, s3)
            _st_v4_volatile(tl.where(live, b2 + region + woff, trash_addr), s0, s1, s2, s3)
            _st_v4_volatile(tl.where(live, b3 + region + woff, trash_addr), s0, s1, s2, s3)
        else:
            m0, m1, m2, m3 = _ld_v4(tl.where(live, xb + rank * chunk_bytes + woff, scratch_addr))
            for k in tl.static_range(1, 4):
                pk = (rank + k) % 4
                bk = tl.load(peers_ptr + pk)
                q0, q1, q2, q3 = _ld_v4(tl.where(live, xb + pk * chunk_bytes + woff, scratch_addr))
                _st_v4_volatile(tl.where(live, bk + region + woff, trash_addr),
                                _sanitize(q0, SENT), _sanitize(q1, SENT), _sanitize(q2, SENT), _sanitize(q3, SENT))

        inbox = my_base + slot * slot_bytes + woff
        a1 = tl.where(live, inbox + r1 * chunk_bytes, scratch_addr)
        a2 = tl.where(live, inbox + r2 * chunk_bytes, scratch_addr)
        a3 = tl.where(live, inbox + r3 * chunk_bytes, scratch_addr)
        w = _poll3_v4(a1, a2, a3, SENT)

        if REDUCE == 0:
            _st_v4_volatile(tl.where(live, ob + rank * chunk_bytes + woff, trash_addr), m0, m1, m2, m3)
            _st_v4_volatile(tl.where(live, ob + r1 * chunk_bytes + woff, trash_addr), w[0], w[1], w[2], w[3])
            _st_v4_volatile(tl.where(live, ob + r2 * chunk_bytes + woff, trash_addr), w[4], w[5], w[6], w[7])
            _st_v4_volatile(tl.where(live, ob + r3 * chunk_bytes + woff, trash_addr), w[8], w[9], w[10], w[11])
        else:
            o0 = _sum4_words(m0, w[0], w[4], w[8], ELEM)
            o1 = _sum4_words(m1, w[1], w[5], w[9], ELEM)
            o2 = _sum4_words(m2, w[2], w[6], w[10], ELEM)
            o3 = _sum4_words(m3, w[3], w[7], w[11], ELEM)
            _st_v4_volatile(tl.where(live, ob + woff, trash_addr), o0, o1, o2, o3)

        z = tl.full([BLOCK_VEC], SENT, tl.int32)
        _st_v4_volatile(a1, z, z, z, z)
        _st_v4_volatile(a2, z, z, z, z)
        _st_v4_volatile(a3, z, z, z, z)
        tl.debug_barrier()
        tl.store(phase_ptr + pid, phase + 1)

    @triton.jit
    def _sanitize(word, SENT: tl.constexpr):
        return tl.where(word == SENT, 0, word)

    @triton.jit
    def _wait_words(src_ptr, idx, m, SENT: tl.constexpr):
        v = tl.load(src_ptr + idx, mask=m, other=0, volatile=True)
        ready = tl.min(tl.where(m, (v != SENT).to(tl.int32), 1), axis=0)
        while ready == 0:
            v = tl.load(src_ptr + idx, mask=m, other=0, volatile=True)
            ready = tl.min(tl.where(m, (v != SENT).to(tl.int32), 1), axis=0)
        return v

    @triton.jit
    def _unpack_sum(acc_lo, acc_hi, word, ELEM: tl.constexpr):
        if ELEM == 0:
            acc_lo += word.to(tl.float32, bitcast=True)
        else:
            lo = (word & 0xFFFF).to(tl.int16)
            hi = (word >> 16).to(tl.int16)
            if ELEM == 1:
                acc_lo += lo.to(tl.bfloat16, bitcast=True).to(tl.float32)
                acc_hi += hi.to(tl.bfloat16, bitcast=True).to(tl.float32)
            else:
                acc_lo += lo.to(tl.float16, bitcast=True).to(tl.float32)
                acc_hi += hi.to(tl.float16, bitcast=True).to(tl.float32)
        return acc_lo, acc_hi

    @triton.jit
    def _pack(acc_lo, acc_hi, ELEM: tl.constexpr):
        if ELEM == 0:
            return acc_lo.to(tl.int32, bitcast=True)
        if ELEM == 1:
            lo = acc_lo.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
            hi = acc_hi.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        else:
            lo = acc_lo.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
            hi = acc_hi.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        return (hi << 16) | lo

    @triton.jit
    def _lamport_kernel(
        x_ptr,
        out_ptr,
        peers_ptr,
        epoch_ptr,
        rank,
        world,
        chunk_words,
        slot_bytes,
        REDUCE: tl.constexpr,
        ELEM: tl.constexpr,
        CPP: tl.constexpr,
        BLOCK: tl.constexpr,
        SENT: tl.constexpr,
        USE_PDL: tl.constexpr,
    ):
        if USE_PDL:
            gdc_wait()
            gdc_launch_dependents()
        pid = tl.program_id(0)
        peer = pid // CPP
        part = pid % CPP
        n_cta = world * CPP
        epoch = tl.load(epoch_ptr)
        slot = epoch % 3
        chunk_bytes = chunk_words * 4
        my_base = tl.load(peers_ptr + rank)

        peer_base = tl.load(peers_ptr + peer)
        dst = (peer_base + slot * slot_bytes + rank * chunk_bytes).to(tl.pointer_type(tl.int32))
        if REDUCE == 0:
            src_off = 0
        else:
            src_off = peer * chunk_words
        per_cta = (chunk_words + CPP - 1) // CPP
        w0 = part * per_cta
        w1 = tl.minimum(w0 + per_cta, chunk_words)
        for off in range(w0, w1, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < w1
            d = tl.load(x_ptr + src_off + idx, mask=m, other=0)
            tl.store(dst + idx, _sanitize(d, SENT), mask=m)


        if REDUCE == 0:
            src = (my_base + slot * slot_bytes + peer * chunk_bytes).to(tl.pointer_type(tl.int32))
            for off in range(w0, w1, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                m = idx < w1
                d = _wait_words(src, idx, m, SENT)
                tl.store(out_ptr + peer * chunk_words + idx, d, mask=m)
                tl.store(src + idx, tl.full([BLOCK], SENT, tl.int32), mask=m)
        else:
            per_r = (chunk_words + n_cta - 1) // n_cta
            s0 = pid * per_r
            s1 = tl.minimum(s0 + per_r, chunk_words)
            for off in range(s0, s1, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                m = idx < s1
                acc_lo = tl.zeros([BLOCK], dtype=tl.float32)
                acc_hi = tl.zeros([BLOCK], dtype=tl.float32)
                for q in range(0, world):
                    src = (my_base + slot * slot_bytes + q * chunk_bytes).to(tl.pointer_type(tl.int32))
                    d = _wait_words(src, idx, m, SENT)
                    acc_lo, acc_hi = _unpack_sum(acc_lo, acc_hi, d, ELEM)
                    tl.store(src + idx, tl.full([BLOCK], SENT, tl.int32), mask=m)
                tl.store(out_ptr + idx, _sanitize(_pack(acc_lo, acc_hi, ELEM), SENT), mask=m)

        tl.debug_barrier()
        done = tl.atomic_add(epoch_ptr + 1, 1, sem="acq_rel", scope="gpu")
        if done == n_cta - 1:
            tl.store(epoch_ptr + 1, 0)
            tl.store(epoch_ptr, epoch + 1)


_WARNED: set = set()


def _warn_once(message: str) -> None:

    try:
        if message in _WARNED:
            return
        _WARNED.add(message)
        print(f"[king-glm53-exchange] {message}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _arm_slots(words: "torch.Tensor") -> None:

    try:
        import arm_slots as _native
    except Exception:
        words.fill_(_SENTINEL)
        return
    try:
        armed = int(_native.arm_slots(words, _SENTINEL))
    except Exception as exc:
        _warn_once(f"native arm_slots failed, using the framework fill: {type(exc).__name__}: {exc}")
        words.fill_(_SENTINEL)
        return
    if armed != words.numel():
        raise RuntimeError(f"native arm_slots armed {armed} of {words.numel()} words")


def _cpp_for(chunk_bytes: int) -> int:
    return max(_CTAS_PER_PEER, min(_MAX_CPP, -(-chunk_bytes // _CTA_BYTES)))


def _elem_kind(dtype: torch.dtype):
    return {torch.float32: _ELEM_F32, torch.bfloat16: _ELEM_BF16, torch.float16: _ELEM_F16}.get(dtype)


class _LaneExchange:

    def __init__(self, x: torch.Tensor, group, kind: str) -> None:
        self.kind = kind
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        rows_in, hidden = x.shape
        self.elem = _elem_kind(x.dtype)
        words_per_row = hidden * x.element_size() // 4
        chunk_rows = rows_in if kind == "ag" else rows_in // self.world
        self.chunk_words = chunk_rows * words_per_row
        self.slot_bytes = (self.world * self.chunk_words * 4 + _ALIGN - 1) // _ALIGN * _ALIGN
        total = _NUM_SLOTS * self.slot_bytes
        buf = _symm.empty(total + 64, dtype=torch.uint8, device=x.device)
        words = buf.view(torch.int32)
        _arm_slots(words)
        words[total // 4: total // 4 + 4] = 0
        torch.cuda.current_stream(x.device).synchronize()
        handle = _symm.rendezvous(buf, group)
        if handle is None or handle.world_size != self.world:
            raise RuntimeError("symmetric memory rendezvous did not cover the group")
        peers = [handle.get_buffer(peer, (buf.numel(),), torch.uint8).data_ptr() for peer in range(self.world)]
        self.buf = buf
        self.handle = handle
        self.peers = torch.tensor(peers, dtype=torch.int64, device=x.device)
        my = peers[self.rank]
        self.scratch_addr = my + total
        self.trash_addr = my + total + 16
        self.block_vec = 32 * LANE_WARPS
        self.grid = -(-(self.chunk_words // 4) // self.block_vec)
        if self.grid > _LANE_MAX_GRID:
            raise RuntimeError("lane kernel grid exceeds the phase array")
        self.phase = torch.zeros(_LANE_MAX_GRID, dtype=torch.int32, device=x.device)
        self.pdl = bool(USE_PDL and _HAS_GDC)
        dist.barrier(group=group)

    def _launch(self, x, out, pdl: bool):
        kwargs = {"num_warps": LANE_WARPS}
        if pdl:
            kwargs["launch_pdl"] = True
        _lane_kernel[(self.grid,)](
            x.view(torch.int32), out.view(torch.int32), self.peers, self.phase, self.rank,
            self.chunk_words, self.slot_bytes, self.scratch_addr, self.trash_addr,
            REDUCE=0 if self.kind == "ag" else 1, ELEM=self.elem, SENT=_SENTINEL,
            BLOCK_VEC=self.block_vec, USE_PDL=pdl, **kwargs,
        )

    def run(self, x, out):
        if self.pdl:
            try:
                self._launch(x, out, True)
                return
            except Exception:
                self.pdl = False
        self._launch(x, out, False)


class _Exchange:
    def __init__(self, x: torch.Tensor, group, kind: str) -> None:
        self.kind = kind
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        rows_in, hidden = x.shape
        self.elem = _elem_kind(x.dtype)
        words_per_row = hidden * x.element_size() // 4
        chunk_rows = rows_in if kind == "ag" else rows_in // self.world
        self.chunk_words = chunk_rows * words_per_row
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        self.cpp = max(1, min(_cpp_for(self.chunk_words * 4), (_MAX_RESIDENT_CTAS_PER_SM * sms) // self.world))
        self.slot_bytes = (self.world * self.chunk_words * 4 + _ALIGN - 1) // _ALIGN * _ALIGN
        total = _NUM_SLOTS * self.slot_bytes
        buf = _symm.empty(total, dtype=torch.uint8, device=x.device)
        _arm_slots(buf.view(torch.int32))
        torch.cuda.current_stream(x.device).synchronize()
        handle = _symm.rendezvous(buf, group)
        if handle is None or handle.world_size != self.world:
            raise RuntimeError("symmetric memory rendezvous did not cover the group")
        peers = [handle.get_buffer(peer, (total,), torch.uint8).data_ptr() for peer in range(self.world)]
        self.buf = buf
        self.handle = handle
        self.peers = torch.tensor(peers, dtype=torch.int64, device=x.device)
        self.epoch = torch.zeros(2, dtype=torch.int32, device=x.device)
        self.pdl = bool(USE_PDL and _HAS_GDC)
        dist.barrier(group=group)

    def _launch(self, x, out, pdl: bool):
        kwargs = {"num_warps": NUM_WARPS}
        if pdl:
            kwargs["launch_pdl"] = True
        _lamport_kernel[(self.world * self.cpp,)](
            x.view(torch.int32), out.view(torch.int32), self.peers, self.epoch, self.rank, self.world,
            self.chunk_words, self.slot_bytes, REDUCE=0 if self.kind == "ag" else 1, ELEM=self.elem,
            CPP=self.cpp, BLOCK=_BLOCK_WORDS, SENT=_SENTINEL, USE_PDL=pdl, **kwargs,
        )

    def run(self, x, out):
        if self.pdl:
            try:
                self._launch(x, out, True)
                return
            except Exception:
                self.pdl = False
        self._launch(x, out, False)


def _stock(kind, x, out, group) -> None:
    if kind == "ag":
        if x.is_cuda:
            dist.all_gather_into_tensor(out, x, group=group)
        else:
            world = dist.get_world_size(group)
            parts = [torch.empty_like(x) for _ in range(world)]
            dist.all_gather(parts, x.contiguous(), group=group)
            out.copy_(torch.cat(parts, dim=0))
    else:
        if x.is_cuda:
            dist.reduce_scatter_tensor(out, x, group=group)
        else:
            world = dist.get_world_size(group)
            rank = dist.get_rank(group)
            summed = x.float().clone()
            dist.all_reduce(summed, op=dist.ReduceOp.SUM, group=group)
            out.copy_(summed.chunk(world, dim=0)[rank].to(out.dtype))


def _resolve_group(group):
    return group if group is not None else dist.group.WORLD


def _state(x, out, group, kind):
    if not (
        _HAS_TRITON and _HAS_SYMM and x.is_cuda and out.is_cuda and x.dim() == 2 and x.is_contiguous()
        and out.is_contiguous() and x.dtype == out.dtype and _elem_kind(x.dtype) is not None
        and (x.shape[1] * x.element_size()) % 4 == 0 and dist.is_initialized()
    ):
        return None
    try:
        world = dist.get_world_size(group)
    except Exception:
        return None
    if world < 2 or world > _MAX_WORLD:
        return None
    chunk_rows = int(x.shape[0]) if kind == "ag" else int(x.shape[0]) // world
    if chunk_rows > FAST_PATH_MAX_TOKENS:
        return None
    key = (id(group), kind, int(x.shape[0]), int(x.shape[1]), x.dtype, x.device.index)
    if key in _STATES:
        return _STATES[key]
    if torch.cuda.is_current_stream_capturing():
        _warn_once(f"shape first seen during graph capture, so this graph replays the stock "
                   f"collective: {kind} rows={key[2]} hidden={key[3]}")
        return None
    try:
        chunk_words = chunk_rows * (int(x.shape[1]) * x.element_size() // 4)
        if USE_LANE_KERNEL and world == 4 and chunk_words % 4 == 0:
            state = _LaneExchange(x, group, kind)
        else:
            state = _Exchange(x, group, kind)
    except Exception as exc:
        _warn_once(f"exchange fast path unavailable for {key[1]} rows={key[2]}: {type(exc).__name__}: {exc}")
        state = None
    _GROUP_REFS[id(group)] = group
    _STATES[key] = state
    return state


def all_gather_into_tensor(x, out, group=None):
    group = _resolve_group(group)
    state = _state(x, out, group, "ag")
    if state is None:
        _stock("ag", x, out, group)
        return None
    state.run(x, out)
    return None


def reduce_scatter_tensor(x, out, group=None):
    group = _resolve_group(group)
    state = _state(x, out, group, "rs")
    if state is None:
        _stock("rs", x, out, group)
        return None
    state.run(x, out)
    return None
