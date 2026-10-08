
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

try:
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

    _HAS_GDC = True
except Exception:
    _HAS_GDC = False

_SENTINEL = -2147483648
_NUM_SLOTS = 3
_ALIGN = 256
_LANE_MAX_GRID = 65536
_BLOCK_VEC = 32
USE_PDL = True

_FIN_BLOCK = 1024
_TOP_K = 8
FAST_PATH_MAX_TOKENS = 512

NATIVE = True
NATIVE_MAX_TOKENS = 96
NATIVE_CLUSTER = ((64, 6), (96, 4))
NATIVE_TRIGGER = 2
_NATIVE: list = []


def _native():
    if not _NATIVE:
        import ftail_native

        _NATIVE.append(ftail_native)
    return _NATIVE[0]




@triton.jit
def _finalize_shared_kernel(GEMM2, IDX, W, SHARED, OUT, hidden, stride,
                            K: tl.constexpr, BLOCK: tl.constexpr):
    tok = tl.inline_asm_elementwise(
        "griddepcontrol.wait; mov.b32 $0, $1;", "=r,r", [tl.program_id(0)],
        dtype=tl.int32, is_pure=False, pack=1)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(K):
        idx = tl.load(IDX + tok * K + k)
        w = tl.load(W + tok * K + k).to(tl.float32)
        valid = idx >= 0
        row = tl.where(valid, idx, 0).to(tl.int64)
        v = tl.load(GEMM2 + row * stride + cols, mask=mask & valid, other=0.0).to(tl.float32)
        acc += w * v
    acc += tl.load(SHARED + tok.to(tl.int64) * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + tok.to(tl.int64) * hidden + cols, acc.to(tl.bfloat16), mask=mask)


def finalize_shared(gemm2, mapping, weights, shared, out, top_k: int = _TOP_K,
                    block: int = _FIN_BLOCK, pdl: bool = True) -> None:
    tokens, hidden = out.shape
    kwargs = {"num_warps": 4}
    if pdl:
        kwargs["launch_pdl"] = True
    _finalize_shared_kernel[(tokens, triton.cdiv(hidden, block))](
        gemm2, mapping, weights, shared, out, hidden, gemm2.stride(0), top_k, block, **kwargs)




@triton.jit
def _ld_v4(addr):
    return tl.inline_asm_elementwise(
        "ld.global.v4.b32 {$0,$1,$2,$3}, [$4];", "=r,=r,=r,=r,l", [addr],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32), is_pure=False, pack=1)


@triton.jit
def _st_v4_volatile(addr, w0, w1, w2, w3):
    return tl.inline_asm_elementwise(
        "st.volatile.global.v4.b32 [$1], {$2,$3,$4,$5}; mov.u32 $0, 0;", "=r,l,r,r,r,r",
        [addr, w0, w1, w2, w3], dtype=tl.int32, is_pure=False, pack=1)


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
        dtype=(tl.int32,) * 12, is_pure=False, pack=1)


@triton.jit
def _sanitize(word, SENT: tl.constexpr):
    return tl.where(word == SENT, 0, word)


@triton.jit
def _lo(word):
    return (word & 0xFFFF).to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)


@triton.jit
def _hi(word):
    return (word >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)


@triton.jit
def _pack(lo, hi):
    a = lo.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    b = hi.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    return (b << 16) | a


@triton.jit
def _sum4(a, b, c, d):
    return _pack(_lo(a) + _lo(b) + _lo(c) + _lo(d), _hi(a) + _hi(b) + _hi(c) + _hi(d))


@triton.jit
def _fma8(l0, h0, l1, h1, l2, h2, l3, h3, wk, d0, d1, d2, d3):
    return (tl.fma(wk, _lo(d0), l0), tl.fma(wk, _hi(d0), h0), tl.fma(wk, _lo(d1), l1), tl.fma(wk, _hi(d1), h1),
            tl.fma(wk, _lo(d2), l2), tl.fma(wk, _hi(d2), h2), tl.fma(wk, _lo(d3), l3), tl.fma(wk, _hi(d3), h3))


@triton.jit
def _finalize_chunk(IDX, W, g2, shared_base, t, g2_stride_bytes, hidden_bytes, K: tl.constexpr,
                    BLOCK_VEC: tl.constexpr):
    tl.static_assert(K == 8, "the split tail is written for GLM-5.3's top-8")
    s0, s1, s2, s3 = _ld_v4(shared_base + t.to(tl.int64) * hidden_bytes)
    b = t * K
    i0 = tl.load(IDX + b + 0)
    i1 = tl.load(IDX + b + 1)
    i2 = tl.load(IDX + b + 2)
    i3 = tl.load(IDX + b + 3)
    i4 = tl.load(IDX + b + 4)
    i5 = tl.load(IDX + b + 5)
    i6 = tl.load(IDX + b + 6)
    i7 = tl.load(IDX + b + 7)
    w0 = tl.where(i0 >= 0, tl.load(W + b + 0).to(tl.float32), 0.0)
    w1 = tl.where(i1 >= 0, tl.load(W + b + 1).to(tl.float32), 0.0)
    w2 = tl.where(i2 >= 0, tl.load(W + b + 2).to(tl.float32), 0.0)
    w3 = tl.where(i3 >= 0, tl.load(W + b + 3).to(tl.float32), 0.0)
    w4 = tl.where(i4 >= 0, tl.load(W + b + 4).to(tl.float32), 0.0)
    w5 = tl.where(i5 >= 0, tl.load(W + b + 5).to(tl.float32), 0.0)
    w6 = tl.where(i6 >= 0, tl.load(W + b + 6).to(tl.float32), 0.0)
    w7 = tl.where(i7 >= 0, tl.load(W + b + 7).to(tl.float32), 0.0)
    r = (_ld_v4(g2 + tl.where(i0 >= 0, i0, 0).to(tl.int64) * g2_stride_bytes)
         + _ld_v4(g2 + tl.where(i1 >= 0, i1, 0).to(tl.int64) * g2_stride_bytes)
         + _ld_v4(g2 + tl.where(i2 >= 0, i2, 0).to(tl.int64) * g2_stride_bytes)
         + _ld_v4(g2 + tl.where(i3 >= 0, i3, 0).to(tl.int64) * g2_stride_bytes)
         + _ld_v4(g2 + tl.where(i4 >= 0, i4, 0).to(tl.int64) * g2_stride_bytes)
         + _ld_v4(g2 + tl.where(i5 >= 0, i5, 0).to(tl.int64) * g2_stride_bytes)
         + _ld_v4(g2 + tl.where(i6 >= 0, i6, 0).to(tl.int64) * g2_stride_bytes)
         + _ld_v4(g2 + tl.where(i7 >= 0, i7, 0).to(tl.int64) * g2_stride_bytes))
    l0 = tl.zeros([BLOCK_VEC], dtype=tl.float32)
    h0, l1, h1, l2, h2, l3, h3 = l0, l0, l0, l0, l0, l0, l0
    l0, h0, l1, h1, l2, h2, l3, h3 = _fma8(l0, h0, l1, h1, l2, h2, l3, h3, w0, r[0], r[1], r[2], r[3])
    l0, h0, l1, h1, l2, h2, l3, h3 = _fma8(l0, h0, l1, h1, l2, h2, l3, h3, w1, r[4], r[5], r[6], r[7])
    l0, h0, l1, h1, l2, h2, l3, h3 = _fma8(l0, h0, l1, h1, l2, h2, l3, h3, w2, r[8], r[9], r[10], r[11])
    l0, h0, l1, h1, l2, h2, l3, h3 = _fma8(l0, h0, l1, h1, l2, h2, l3, h3, w3, r[12], r[13], r[14], r[15])
    l0, h0, l1, h1, l2, h2, l3, h3 = _fma8(l0, h0, l1, h1, l2, h2, l3, h3, w4, r[16], r[17], r[18], r[19])
    l0, h0, l1, h1, l2, h2, l3, h3 = _fma8(l0, h0, l1, h1, l2, h2, l3, h3, w5, r[20], r[21], r[22], r[23])
    l0, h0, l1, h1, l2, h2, l3, h3 = _fma8(l0, h0, l1, h1, l2, h2, l3, h3, w6, r[24], r[25], r[26], r[27])
    l0, h0, l1, h1, l2, h2, l3, h3 = _fma8(l0, h0, l1, h1, l2, h2, l3, h3, w7, r[28], r[29], r[30], r[31])
    return (_pack(l0 + _lo(s0), h0 + _hi(s0)), _pack(l1 + _lo(s1), h1 + _hi(s1)),
            _pack(l2 + _lo(s2), h2 + _hi(s2)), _pack(l3 + _lo(s3), h3 + _hi(s3)))


@triton.jit
def _tail_kernel(
    GEMM2, IDX, W, SHARED, out_ptr, peers_ptr, phase_ptr, rank, chunk_rows, chunk_words,
    slot_bytes, g2_stride_bytes, hidden_bytes,
    K: tl.constexpr, VEC_PER_ROW: tl.constexpr, BLOCK_VEC: tl.constexpr, SENT: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        gdc_wait()
    pid = tl.program_id(0)
    phase = tl.load(phase_ptr + pid)
    slot = phase % 3
    v = pid * BLOCK_VEC + tl.arange(0, BLOCK_VEC)
    woff = v.to(tl.int64) * 16
    chunk_bytes = chunk_words * 4
    colb = (v % VEC_PER_ROW).to(tl.int64) * 16
    vrow = v // VEC_PER_ROW
    region = slot * slot_bytes + rank * chunk_bytes
    g2 = GEMM2.to(tl.int64) + colb
    shared_base = SHARED.to(tl.int64) + colb
    m0 = tl.zeros([BLOCK_VEC], dtype=tl.int32)
    m1, m2, m3 = m0, m0, m0

    for p in tl.static_range(4):
        t = p * chunk_rows + vrow
        l0 = tl.zeros([BLOCK_VEC], dtype=tl.float32)
        h0 = tl.zeros([BLOCK_VEC], dtype=tl.float32)
        l1, h1, l2, h2, l3, h3 = l0, l0, l0, l0, l0, l0
        for k in tl.static_range(K):
            idx = tl.load(IDX + t * K + k)
            valid = idx >= 0
            wk = tl.where(valid, tl.load(W + t * K + k).to(tl.float32), 0.0)
            row = tl.where(valid, idx, 0).to(tl.int64)
            d0, d1, d2, d3 = _ld_v4(g2 + row * g2_stride_bytes)
            l0 += wk * _lo(d0)
            h0 += wk * _hi(d0)
            l1 += wk * _lo(d1)
            h1 += wk * _hi(d1)
            l2 += wk * _lo(d2)
            h2 += wk * _hi(d2)
            l3 += wk * _lo(d3)
            h3 += wk * _hi(d3)
        s0, s1, s2, s3 = _ld_v4(shared_base + t.to(tl.int64) * hidden_bytes)
        q0 = _pack(l0 + _lo(s0), h0 + _hi(s0))
        q1 = _pack(l1 + _lo(s1), h1 + _hi(s1))
        q2 = _pack(l2 + _lo(s2), h2 + _hi(s2))
        q3 = _pack(l3 + _lo(s3), h3 + _hi(s3))
        m0 = tl.where(p == rank, q0, m0)
        m1 = tl.where(p == rank, q1, m1)
        m2 = tl.where(p == rank, q2, m2)
        m3 = tl.where(p == rank, q3, m3)
        if p != rank:
            _st_v4_volatile(tl.load(peers_ptr + p) + region + woff, _sanitize(q0, SENT),
                            _sanitize(q1, SENT), _sanitize(q2, SENT), _sanitize(q3, SENT))

    inbox = tl.load(peers_ptr + rank) + slot * slot_bytes + woff
    a1 = inbox + ((rank + 1) % 4) * chunk_bytes
    a2 = inbox + ((rank + 2) % 4) * chunk_bytes
    a3 = inbox + ((rank + 3) % 4) * chunk_bytes
    w = _poll3_v4(a1, a2, a3, SENT)

    _st_v4_volatile(out_ptr.to(tl.int64) + woff,
                    _sum4(m0, w[0], w[4], w[8]), _sum4(m1, w[1], w[5], w[9]),
                    _sum4(m2, w[2], w[6], w[10]), _sum4(m3, w[3], w[7], w[11]))

    z = tl.full([BLOCK_VEC], SENT, tl.int32)
    _st_v4_volatile(a1, z, z, z, z)
    _st_v4_volatile(a2, z, z, z, z)
    _st_v4_volatile(a3, z, z, z, z)
    tl.store(phase_ptr + pid, phase + 1)


@triton.jit
def _tail_split_kernel(
    GEMM2, IDX, W, SHARED, out_ptr, peers_ptr, phase_ptr, rank, nvb, chunk_rows, chunk_words,
    slot_bytes, g2_stride_bytes, hidden_bytes,
    K: tl.constexpr, VEC_PER_ROW: tl.constexpr, BLOCK_VEC: tl.constexpr, SENT: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        gdc_wait()
    pid = tl.program_id(0)
    j = pid // nvb
    vb = pid - j * nvb
    phase = tl.load(phase_ptr + pid)
    slot = phase % 3
    v = vb * BLOCK_VEC + tl.arange(0, BLOCK_VEC)
    woff = v.to(tl.int64) * 16
    chunk_bytes = chunk_words * 4
    colb = (v % VEC_PER_ROW).to(tl.int64) * 16
    vrow = v // VEC_PER_ROW
    q0, q1, q2, q3 = _finalize_chunk(IDX, W, GEMM2.to(tl.int64) + colb, SHARED.to(tl.int64) + colb,
                                     ((rank + 1 + j) % 4) * chunk_rows + vrow, g2_stride_bytes, hidden_bytes,
                                     K, BLOCK_VEC)
    if j < 3:
        region = slot * slot_bytes + rank * chunk_bytes
        _st_v4_volatile(tl.load(peers_ptr + (rank + 1 + j) % 4) + region + woff, _sanitize(q0, SENT),
                        _sanitize(q1, SENT), _sanitize(q2, SENT), _sanitize(q3, SENT))
    else:
        inbox = tl.load(peers_ptr + rank) + slot * slot_bytes + woff
        a1 = inbox + ((rank + 1) % 4) * chunk_bytes
        a2 = inbox + ((rank + 2) % 4) * chunk_bytes
        a3 = inbox + ((rank + 3) % 4) * chunk_bytes
        w = _poll3_v4(a1, a2, a3, SENT)
        _st_v4_volatile(out_ptr.to(tl.int64) + woff,
                        _sum4(q0, w[0], w[4], w[8]), _sum4(q1, w[1], w[5], w[9]),
                        _sum4(q2, w[2], w[6], w[10]), _sum4(q3, w[3], w[7], w[11]))
        z = tl.full([BLOCK_VEC], SENT, tl.int32)
        _st_v4_volatile(a1, z, z, z, z)
        _st_v4_volatile(a2, z, z, z, z)
        _st_v4_volatile(a3, z, z, z, z)
    tl.store(phase_ptr + pid, phase + 1)


@triton.jit
def _st_v4(addr, w0, w1, w2, w3):
    return tl.inline_asm_elementwise(
        "st.global.v4.b32 [$1], {$2,$3,$4,$5}; mov.u32 $0, 0;", "=r,l,r,r,r,r",
        [addr, w0, w1, w2, w3], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _tail_norm_kernel(
    GEMM2, IDX, W, SHARED, RES, GAMMA, out_ptr, normed_ptr, resout_ptr, peers_ptr, phase_ptr, nphase_ptr,
    rowcnt_ptr, part_ptr, rank, chunk_rows, chunk_words, slot_bytes, g2_stride_bytes, hidden_bytes,
    eps,
    K: tl.constexpr, VEC_PER_ROW: tl.constexpr, BLOCK_VEC: tl.constexpr, SENT: tl.constexpr,
    USE_PDL: tl.constexpr, SEGS: tl.constexpr, SEGP2: tl.constexpr, HIDDEN: tl.constexpr,
):
    if USE_PDL:
        gdc_wait()
    pid = tl.program_id(0)
    phase = tl.load(phase_ptr + pid)
    slot = phase % 3
    v = pid * BLOCK_VEC + tl.arange(0, BLOCK_VEC)
    woff = v.to(tl.int64) * 16
    chunk_bytes = chunk_words * 4
    colb = (v % VEC_PER_ROW).to(tl.int64) * 16
    vrow = v // VEC_PER_ROW
    region = slot * slot_bytes + rank * chunk_bytes
    g2 = GEMM2.to(tl.int64) + colb
    shared_base = SHARED.to(tl.int64) + colb
    m0 = tl.zeros([BLOCK_VEC], dtype=tl.int32)
    m1, m2, m3 = m0, m0, m0

    for p in tl.static_range(4):
        t = p * chunk_rows + vrow
        l0 = tl.zeros([BLOCK_VEC], dtype=tl.float32)
        h0 = tl.zeros([BLOCK_VEC], dtype=tl.float32)
        l1, h1, l2, h2, l3, h3 = l0, l0, l0, l0, l0, l0
        for k in tl.static_range(K):
            idx = tl.load(IDX + t * K + k)
            valid = idx >= 0
            wk = tl.where(valid, tl.load(W + t * K + k).to(tl.float32), 0.0)
            row = tl.where(valid, idx, 0).to(tl.int64)
            d0, d1, d2, d3 = _ld_v4(g2 + row * g2_stride_bytes)
            l0 += wk * _lo(d0)
            h0 += wk * _hi(d0)
            l1 += wk * _lo(d1)
            h1 += wk * _hi(d1)
            l2 += wk * _lo(d2)
            h2 += wk * _hi(d2)
            l3 += wk * _lo(d3)
            h3 += wk * _hi(d3)
        s0, s1, s2, s3 = _ld_v4(shared_base + t.to(tl.int64) * hidden_bytes)
        q0 = _pack(l0 + _lo(s0), h0 + _hi(s0))
        q1 = _pack(l1 + _lo(s1), h1 + _hi(s1))
        q2 = _pack(l2 + _lo(s2), h2 + _hi(s2))
        q3 = _pack(l3 + _lo(s3), h3 + _hi(s3))
        m0 = tl.where(p == rank, q0, m0)
        m1 = tl.where(p == rank, q1, m1)
        m2 = tl.where(p == rank, q2, m2)
        m3 = tl.where(p == rank, q3, m3)
        if p != rank:
            _st_v4_volatile(tl.load(peers_ptr + p) + region + woff, _sanitize(q0, SENT),
                            _sanitize(q1, SENT), _sanitize(q2, SENT), _sanitize(q3, SENT))

    inbox = tl.load(peers_ptr + rank) + slot * slot_bytes + woff
    a1 = inbox + ((rank + 1) % 4) * chunk_bytes
    a2 = inbox + ((rank + 2) % 4) * chunk_bytes
    a3 = inbox + ((rank + 3) % 4) * chunk_bytes
    w = _poll3_v4(a1, a2, a3, SENT)

    t0 = _sum4(m0, w[0], w[4], w[8])
    t1 = _sum4(m1, w[1], w[5], w[9])
    t2 = _sum4(m2, w[2], w[6], w[10])
    t3 = _sum4(m3, w[3], w[7], w[11])
    _st_v4(out_ptr.to(tl.int64) + woff, t0, t1, t2, t3)

    z = tl.full([BLOCK_VEC], SENT, tl.int32)
    _st_v4_volatile(a1, z, z, z, z)
    _st_v4_volatile(a2, z, z, z, z)
    _st_v4_volatile(a3, z, z, z, z)
    tl.store(phase_ptr + pid, phase + 1)

    r0, r1, r2, r3 = _ld_v4(RES.to(tl.int64) + woff)
    n0 = _pack(_lo(t0) + _lo(r0), _hi(t0) + _hi(r0))
    n1 = _pack(_lo(t1) + _lo(r1), _hi(t1) + _hi(r1))
    n2 = _pack(_lo(t2) + _lo(r2), _hi(t2) + _hi(r2))
    n3 = _pack(_lo(t3) + _lo(r3), _hi(t3) + _hi(r3))
    _st_v4(resout_ptr.to(tl.int64) + woff, n0, n1, n2, n3)
    x0 = _lo(n0)
    x1 = _hi(n0)
    x2 = _lo(n1)
    x3 = _hi(n1)
    x4 = _lo(n2)
    x5 = _hi(n2)
    x6 = _lo(n3)
    x7 = _hi(n3)
    ps = x0 * x0
    ps += x1 * x1
    ps += x2 * x2
    ps += x3 * x3
    ps += x4 * x4
    ps += x5 * x5
    ps += x6 * x6
    ps += x7 * x7
    part = tl.sum(ps, axis=0)

    nrow = pid // SEGS
    seg = pid % SEGS
    nph = tl.load(nphase_ptr + pid)
    tl.store(part_ptr + nrow * SEGS + seg, part)
    tl.atomic_add(rowcnt_ptr + nrow, 1, sem="release", scope="gpu")
    target = (nph + 1) * SEGS
    seen = tl.atomic_add(rowcnt_ptr + nrow, 0, sem="acquire", scope="gpu")
    while seen < target:
        seen = tl.atomic_add(rowcnt_ptr + nrow, 0, sem="acquire", scope="gpu")
    segs = tl.arange(0, SEGP2)
    parts = tl.load(part_ptr + nrow * SEGS + segs, mask=segs < SEGS, other=0.0,
                    cache_modifier=".cg")
    variance = tl.sum(parts, axis=0) * (1.0 / HIDDEN)
    inv = tl.rsqrt(variance + eps)
    g0, g1, g2w, g3 = _ld_v4(GAMMA.to(tl.int64) + colb)
    o0 = _pack((x0 * inv) * _lo(g0), (x1 * inv) * _hi(g0))
    o1 = _pack((x2 * inv) * _lo(g1), (x3 * inv) * _hi(g1))
    o2 = _pack((x4 * inv) * _lo(g2w), (x5 * inv) * _hi(g2w))
    o3 = _pack((x6 * inv) * _lo(g3), (x7 * inv) * _hi(g3))
    _st_v4(normed_ptr.to(tl.int64) + woff, o0, o1, o2, o3)
    tl.store(nphase_ptr + pid, nph + 1)


@triton.jit
def _tail_norm_split_kernel(
    GEMM2, IDX, W, SHARED, RES, GAMMA, out_ptr, normed_ptr, resout_ptr, peers_ptr, phase_ptr, nphase_ptr,
    rowcnt_ptr, part_ptr, rank, nvb, chunk_rows, chunk_words, slot_bytes, g2_stride_bytes, hidden_bytes,
    eps,
    K: tl.constexpr, VEC_PER_ROW: tl.constexpr, BLOCK_VEC: tl.constexpr, SENT: tl.constexpr,
    USE_PDL: tl.constexpr, SEGS: tl.constexpr, SEGP2: tl.constexpr, HIDDEN: tl.constexpr,
):
    if USE_PDL:
        gdc_wait()
    pid = tl.program_id(0)
    j = pid // nvb
    vb = pid - j * nvb
    phase = tl.load(phase_ptr + pid)
    slot = phase % 3
    v = vb * BLOCK_VEC + tl.arange(0, BLOCK_VEC)
    woff = v.to(tl.int64) * 16
    chunk_bytes = chunk_words * 4
    colb = (v % VEC_PER_ROW).to(tl.int64) * 16
    vrow = v // VEC_PER_ROW
    q0, q1, q2, q3 = _finalize_chunk(IDX, W, GEMM2.to(tl.int64) + colb, SHARED.to(tl.int64) + colb,
                                     ((rank + 1 + j) % 4) * chunk_rows + vrow, g2_stride_bytes, hidden_bytes,
                                     K, BLOCK_VEC)
    if j < 3:
        region = slot * slot_bytes + rank * chunk_bytes
        _st_v4_volatile(tl.load(peers_ptr + (rank + 1 + j) % 4) + region + woff, _sanitize(q0, SENT),
                        _sanitize(q1, SENT), _sanitize(q2, SENT), _sanitize(q3, SENT))
        tl.store(phase_ptr + pid, phase + 1)
    else:
        r0, r1, r2, r3 = _ld_v4(RES.to(tl.int64) + woff)
        g0, g1, g2w, g3 = _ld_v4(GAMMA.to(tl.int64) + colb)
        inbox = tl.load(peers_ptr + rank) + slot * slot_bytes + woff
        a1 = inbox + ((rank + 1) % 4) * chunk_bytes
        a2 = inbox + ((rank + 2) % 4) * chunk_bytes
        a3 = inbox + ((rank + 3) % 4) * chunk_bytes
        w = _poll3_v4(a1, a2, a3, SENT)

        t0 = _sum4(q0, w[0], w[4], w[8])
        t1 = _sum4(q1, w[1], w[5], w[9])
        t2 = _sum4(q2, w[2], w[6], w[10])
        t3 = _sum4(q3, w[3], w[7], w[11])
        _st_v4(out_ptr.to(tl.int64) + woff, t0, t1, t2, t3)

        z = tl.full([BLOCK_VEC], SENT, tl.int32)
        _st_v4_volatile(a1, z, z, z, z)
        _st_v4_volatile(a2, z, z, z, z)
        _st_v4_volatile(a3, z, z, z, z)
        tl.store(phase_ptr + pid, phase + 1)

        n0 = _pack(_lo(t0) + _lo(r0), _hi(t0) + _hi(r0))
        n1 = _pack(_lo(t1) + _lo(r1), _hi(t1) + _hi(r1))
        n2 = _pack(_lo(t2) + _lo(r2), _hi(t2) + _hi(r2))
        n3 = _pack(_lo(t3) + _lo(r3), _hi(t3) + _hi(r3))
        _st_v4(resout_ptr.to(tl.int64) + woff, n0, n1, n2, n3)
        x0 = _lo(n0)
        x1 = _hi(n0)
        x2 = _lo(n1)
        x3 = _hi(n1)
        x4 = _lo(n2)
        x5 = _hi(n2)
        x6 = _lo(n3)
        x7 = _hi(n3)
        ps = x0 * x0
        ps += x1 * x1
        ps += x2 * x2
        ps += x3 * x3
        ps += x4 * x4
        ps += x5 * x5
        ps += x6 * x6
        ps += x7 * x7
        part = tl.sum(ps, axis=0)

        nrow = vb // SEGS
        seg = vb % SEGS
        nph = tl.load(nphase_ptr + vb)
        tl.store(part_ptr + nrow * SEGS + seg, part)
        tl.atomic_add(rowcnt_ptr + nrow, 1, sem="release", scope="gpu")
        target = (nph + 1) * SEGS
        seen = tl.atomic_add(rowcnt_ptr + nrow, 0, sem="acquire", scope="gpu")
        while seen < target:
            seen = tl.atomic_add(rowcnt_ptr + nrow, 0, sem="acquire", scope="gpu")
        segs = tl.arange(0, SEGP2)
        parts = tl.load(part_ptr + nrow * SEGS + segs, mask=segs < SEGS, other=0.0,
                        cache_modifier=".cg")
        variance = tl.sum(parts, axis=0) * (1.0 / HIDDEN)
        inv = tl.rsqrt(variance + eps)
        o0 = _pack((x0 * inv) * _lo(g0), (x1 * inv) * _hi(g0))
        o1 = _pack((x2 * inv) * _lo(g1), (x3 * inv) * _hi(g1))
        o2 = _pack((x4 * inv) * _lo(g2w), (x5 * inv) * _hi(g2w))
        o3 = _pack((x6 * inv) * _lo(g3), (x7 * inv) * _hi(g3))
        _st_v4(normed_ptr.to(tl.int64) + woff, o0, o1, o2, o3)
        tl.store(nphase_ptr + vb, nph + 1)


class FusedTailState:

    def __init__(self, tokens: int, hidden: int, group, device, dtype=torch.bfloat16,
                 top_k: int = _TOP_K, block_vec: int = _BLOCK_VEC) -> None:
        if _symm is None:
            raise RuntimeError("torch.distributed._symmetric_memory is unavailable")
        if dtype != torch.bfloat16:
            raise ValueError("the fused tail is bf16 only")
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if self.world != 4:
            raise ValueError("the fused tail kernel is world 4 only")
        if tokens % self.world:
            raise ValueError(f"{tokens} gathered rows do not divide over {self.world} ranks")
        self.top_k = int(top_k)
        self.hidden = int(hidden)
        self.chunk_rows = tokens // self.world
        self.vec_per_row = self.hidden // 8
        if self.hidden % 8:
            raise ValueError("hidden must be a multiple of 8 bf16 values (one 16-byte vector)")
        self.chunk_words = self.chunk_rows * self.hidden // 2
        n_vec = self.chunk_words // 4
        self.block_vec = int(block_vec)
        if n_vec % self.block_vec or self.block_vec % 32:
            raise ValueError(f"{n_vec} vectors per chunk is not a multiple of {self.block_vec}")
        self.grid = n_vec // self.block_vec
        if self.grid > _LANE_MAX_GRID:
            raise RuntimeError("fused tail grid exceeds the phase array")
        self.slot_bytes = (self.world * self.chunk_words * 4 + _ALIGN - 1) // _ALIGN * _ALIGN
        total = _NUM_SLOTS * self.slot_bytes
        buf = _symm.empty(total, dtype=torch.uint8, device=device)
        buf.view(torch.int32).fill_(_SENTINEL)
        torch.cuda.current_stream(device).synchronize()
        handle = _symm.rendezvous(buf, group)
        if handle is None or handle.world_size != self.world:
            raise RuntimeError("symmetric memory rendezvous did not cover the group")
        peers = [handle.get_buffer(peer, (buf.numel(),), torch.uint8).data_ptr()
                 for peer in range(self.world)]
        self.buf, self.handle = buf, handle
        self.peers = torch.tensor(peers, dtype=torch.int64, device=device)
        self.phase = torch.zeros(_LANE_MAX_GRID, dtype=torch.int32, device=device)
        self.pdl = bool(USE_PDL and _HAS_GDC)
        self.segs = self.vec_per_row // self.block_vec
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.split = 4 * self.grid <= 16 * sms and 4 * self.grid <= _LANE_MAX_GRID
        self.norm_ok = (self.vec_per_row % self.block_vec == 0 and self.grid == self.chunk_rows * self.segs
                        and self.grid <= 16 * sms and self.segs <= 32)
        self.nphase = torch.zeros(self.grid, dtype=torch.int32, device=device)
        self.rowcnt = torch.zeros(self.chunk_rows, dtype=torch.int32, device=device)
        self.parts = torch.zeros(self.chunk_rows * self.segs, dtype=torch.float32, device=device)
        self.native = None
        if NATIVE and tokens <= NATIVE_MAX_TOKENS and self.hidden == 6144 and self.top_k == 8:
            self.native = _native()
            self.cluster = next(c for cap, c in NATIVE_CLUSTER if tokens <= cap)
            self.peer_list = [int(p) for p in peers]
            self.nat_phase = torch.zeros(tokens * self.cluster, dtype=torch.int32, device=device)
        dist.barrier(group=group)
        if self.native is not None:
            _warn_once(f"native tail: tokens={tokens} clusters={tokens} x {self.cluster} CTAs trigger={NATIVE_TRIGGER}")
        elif self.split:
            _warn_once(f"split tail: tokens={tokens} programs={4 * self.grid}")

    def _launch(self, gemm2, mapping, weights, shared, out, pdl: bool) -> None:
        kwargs = {"num_warps": self.block_vec // 32}
        if pdl:
            kwargs["launch_pdl"] = True
        if self.split:
            _tail_split_kernel[(4 * self.grid,)](
                gemm2, mapping, weights, shared, out.view(torch.int32), self.peers, self.phase,
                self.rank, self.grid, self.chunk_rows, self.chunk_words, self.slot_bytes,
                gemm2.stride(0) * gemm2.element_size(), self.hidden * shared.element_size(),
                K=self.top_k, VEC_PER_ROW=self.vec_per_row, BLOCK_VEC=self.block_vec,
                SENT=_SENTINEL, USE_PDL=pdl, **kwargs)
            return
        _tail_kernel[(self.grid,)](
            gemm2, mapping, weights, shared, out.view(torch.int32), self.peers, self.phase,
            self.rank, self.chunk_rows, self.chunk_words, self.slot_bytes,
            gemm2.stride(0) * gemm2.element_size(), self.hidden * shared.element_size(),
            K=self.top_k, VEC_PER_ROW=self.vec_per_row, BLOCK_VEC=self.block_vec,
            SENT=_SENTINEL, USE_PDL=pdl, **kwargs)

    def run(self, gemm2, mapping, weights, shared, out) -> None:
        if self.native is not None:
            self.native.run(gemm2, mapping, weights, shared, out, out, out, out, out, self.nat_phase, self.peer_list,
                            self.rank, self.slot_bytes, 0.0, False, self.pdl, 0, self.cluster)
            return
        if self.pdl:
            try:
                self._launch(gemm2, mapping, weights, shared, out, True)
                return
            except Exception:
                self.pdl = False
        self._launch(gemm2, mapping, weights, shared, out, False)


def finalize_shared_reduce_scatter_norm(state: FusedTailState, gemm2, mapping, weights, shared, out,
                                        residual, gamma, eps: float, normed, residual_out, release: bool = False) -> None:
    if state.native is not None:
        state.native.run(gemm2, mapping, weights, shared, out, residual, gamma, normed, residual_out, state.nat_phase,
                         state.peer_list, state.rank, state.slot_bytes, float(eps), True, state.pdl,
                         NATIVE_TRIGGER if release else 0, state.cluster)
        return

    def launch(pdl: bool) -> None:
        kwargs = {"num_warps": state.block_vec // 32}
        if pdl:
            kwargs["launch_pdl"] = True
        if state.split:
            _tail_norm_split_kernel[(4 * state.grid,)](
                gemm2, mapping, weights, shared, residual, gamma, out.view(torch.int32), normed.view(torch.int32),
                residual_out.view(torch.int32), state.peers, state.phase, state.nphase, state.rowcnt,
                state.parts, state.rank, state.grid, state.chunk_rows, state.chunk_words, state.slot_bytes,
                gemm2.stride(0) * gemm2.element_size(), state.hidden * shared.element_size(), float(eps),
                K=state.top_k, VEC_PER_ROW=state.vec_per_row, BLOCK_VEC=state.block_vec, SENT=_SENTINEL,
                USE_PDL=pdl, SEGS=state.segs, SEGP2=32, HIDDEN=state.hidden, **kwargs)
            return
        _tail_norm_kernel[(state.grid,)](
            gemm2, mapping, weights, shared, residual, gamma, out.view(torch.int32), normed.view(torch.int32),
            residual_out.view(torch.int32), state.peers, state.phase, state.nphase, state.rowcnt,
            state.parts, state.rank, state.chunk_rows, state.chunk_words, state.slot_bytes,
            gemm2.stride(0) * gemm2.element_size(), state.hidden * shared.element_size(), float(eps),
            K=state.top_k, VEC_PER_ROW=state.vec_per_row, BLOCK_VEC=state.block_vec, SENT=_SENTINEL,
            USE_PDL=pdl, SEGS=state.segs, SEGP2=32, HIDDEN=state.hidden, **kwargs)

    if state.pdl:
        try:
            launch(True)
            return
        except Exception:
            state.pdl = False
    launch(False)


def finalize_shared_reduce_scatter(state: FusedTailState, gemm2, mapping, weights, shared,
                                   out) -> None:
    state.run(gemm2, mapping, weights, shared, out)


_STATES: dict = {}
_GROUP_REFS: dict = {}
_WARNED: set = set()


def _warn_once(message: str) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        print(f"CACHEON_FUSED_TAIL {message}", file=sys.stderr, flush=True)


def state_for(tokens: int, hidden: int, group, device):
    if _symm is None or not dist.is_initialized() or tokens > FAST_PATH_MAX_TOKENS:
        return None
    key = (id(group), int(tokens), int(hidden), device.index)
    if key in _STATES:
        return _STATES[key]
    if torch.cuda.is_current_stream_capturing():
        _warn_once(f"shape first seen during graph capture, so this graph replays the unfused "
                   f"tail: tokens={tokens} hidden={hidden}")
        return None
    try:
        state = FusedTailState(int(tokens), int(hidden), group, device)
    except Exception as exc:
        _warn_once(f"fused tail unavailable for tokens={tokens}: {type(exc).__name__}: {exc}")
        state = None
    _GROUP_REFS[id(group)] = group
    _STATES[key] = state
    return state
