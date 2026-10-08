#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdint>

namespace dsatk {

constexpr int NT = 512;
constexpr int NW = NT / 32;
constexpr int NB = 4096;
constexpr int TIEMAX = 2048;
constexpr int KMAX = 2048;
constexpr int RB = 2048;

struct Params {
  const float* scores;
  long long sstride;
  const int* lens;
  const int* pt;
  long long ptstride;
  int* out;
  int64_t* stamps;
  int topk;
  int pb;
  int floor10;
  int pt_cap;
  int dcap;
};

__device__ __forceinline__ uint32_t okey32(float x) {
  const uint32_t b = __float_as_uint(x);
  return b ^ ((static_cast<uint32_t>(static_cast<int32_t>(b) >> 31)) | 0x80000000u);
}
__device__ __forceinline__ float key2f(uint32_t k) {
  return __uint_as_float((k & 0x80000000u) ? (k ^ 0x80000000u) : ~k);
}
__device__ __forceinline__ uint32_t bin12(float x) {   
  const uint32_t b = __half_as_ushort(__float2half_rn(x));
  return ((b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u)) >> 4;
}
__device__ __forceinline__ uint32_t rawbin(float x) { return static_cast<uint32_t>(__half_as_ushort(__float2half_rn(x))) >> 4; }
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ float okey16_to_finite(uint32_t okey) {
  const uint32_t ob = okey & 0xFFFFu;
  const uint16_t hb = (ob & 0x8000u) ? static_cast<uint16_t>(ob ^ 0x8000u) : static_cast<uint16_t>(~ob);
  return __half2float(__ushort_as_half(hb));
}
__device__ __noinline__ uint32_t exact_lbkey_slow(uint32_t b) {
  if (b == 0) return 0;
  if (b >= 4096u) return 0xFFFFFFFFu;
  uint32_t lo = 0, hi = 0xFFFFFFFFu;
  while (hi - lo > 1) {
    const uint32_t mid = lo + (hi - lo) / 2;
    if (bin12(key2f(mid)) >= b) hi = mid; else lo = mid;
  }
  return hi;
}
__device__ __forceinline__ uint32_t exact_lbkey(uint32_t b) {
  const uint32_t key = b << 4;
  if (key - 0x0401u <= 0xFBFFu - 0x0401u && b < 4096u)
    return okey32(0.5f * (okey16_to_finite(key) + okey16_to_finite(key - 1))) + (b < 2048u ? 1u : 0u);
  return exact_lbkey_slow(b);
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t r) {
  uint32_t o;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r));
  return o;
}
__device__ __forceinline__ uint2 ld_cl_v2(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared::cluster.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void st_async_v4(uint32_t a, uint4 v, uint32_t mbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.u32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(a),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(mbar)
               : "memory");
}
__device__ __forceinline__ void mbar_init(uint32_t m, uint32_t cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(m), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_expect_local(uint32_t m, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(m), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t m) {   
  uint32_t done = 0;
  do {
    asm volatile(
        "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
        : "=r"(done)
        : "r"(m)
        : "memory");
  } while (!done);
}
__device__ __forceinline__ void cl_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ long long gtimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint32_t warp_suffix_excl(uint32_t v, int lane) {   
  uint32_t suf = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t t = __shfl_down_sync(0xFFFFFFFFu, suf, o);
    if (lane + o < 32) suf += t;
  }
  return suf - v;
}

struct Smem {
  uint4* tie;       
  uint32_t* hist;   
  uint32_t* sup;    
  uint32_t* l2;     
  uint2* cnt;       
  uint32_t* misc;   
  uint64_t* mbar;   
  int* pts;         
  float* data;      
};
__host__ __device__ constexpr int al16(int x) { return (x + 15) / 16 * 16; }
template <int C>
__host__ __device__ constexpr int smem_fixed() {
  return TIEMAX * 16 + NB * 4 + C * 64 * 4 + 64 * 4 + al16(C * 8) + 64 * 4 + 4 * 8;
}
template <int C>
__device__ __forceinline__ Smem carve(uint8_t* b, int pt_cap) {
  Smem s;
  int o = 0;
  s.tie = reinterpret_cast<uint4*>(b + o); o += TIEMAX * 16;
  s.hist = reinterpret_cast<uint32_t*>(b + o); o += NB * 4;
  s.sup = reinterpret_cast<uint32_t*>(b + o); o += C * 64 * 4;
  s.l2 = reinterpret_cast<uint32_t*>(b + o); o += 64 * 4;
  s.cnt = reinterpret_cast<uint2*>(b + o); o += al16(C * 8);
  s.misc = reinterpret_cast<uint32_t*>(b + o); o += 64 * 4;
  s.mbar = reinterpret_cast<uint64_t*>(b + o); o += 4 * 8;
  s.pts = reinterpret_cast<int*>(b + o); o += al16(pt_cap * 4);
  s.data = reinterpret_cast<float*>(b + o);
  return s;
}

#define STAMP(i)                                                                                         \
  do {                                                                                                   \
    if constexpr (STAMPS) {                                                                              \
      if (tid == 0) p.stamps[((long long)row * C + rank) * 32 + (i)] = (i) == 0 ? gtimer() : clock64(); \
    }                                                                                                    \
  } while (0)

__device__ __noinline__ void trivial_rows(const Params& p, int row, uint32_t n) {
  const uint32_t K = p.topk, pb = p.pb, pmask = (1u << pb) - 1u;
  int* out = p.out + static_cast<long long>(row) * K;
  const int* ptr = p.pt + static_cast<long long>(row) * p.ptstride;
  pdl_wait();
  for (uint32_t t = threadIdx.x; t < K; t += NT)
    out[t] = t < n ? static_cast<int>((static_cast<uint32_t>(ptr[t >> pb]) << pb) | (t & pmask)) : -1;
}

__device__ __noinline__ void hist_global(const float* __restrict__ xs, uint32_t nsm, uint32_t nv, uint32_t m,
                                        uint32_t* hist) {
  const uint32_t nfull = m >> 2;
  for (uint32_t i = nsm + threadIdx.x; i < nfull; i += 2 * NT) {
    const float4 v = *reinterpret_cast<const float4*>(xs + 4 * i);
    float4 w = make_float4(0.f, 0.f, 0.f, 0.f);
    const bool two = i + NT < nfull;
    if (two) w = *reinterpret_cast<const float4*>(xs + 4 * (i + NT));
    const uint32_t pa = h2u(__floats2half2_rn(v.x, v.y)), pc = h2u(__floats2half2_rn(v.z, v.w));
    atomicAdd(&hist[(pa >> 4) & 0xFFFu], 1u);
    atomicAdd(&hist[pa >> 20], 1u);
    atomicAdd(&hist[(pc >> 4) & 0xFFFu], 1u);
    atomicAdd(&hist[pc >> 20], 1u);
    if (two) {
      const uint32_t qa = h2u(__floats2half2_rn(w.x, w.y)), qc = h2u(__floats2half2_rn(w.z, w.w));
      atomicAdd(&hist[(qa >> 4) & 0xFFFu], 1u);
      atomicAdd(&hist[qa >> 20], 1u);
      atomicAdd(&hist[(qc >> 4) & 0xFFFu], 1u);
      atomicAdd(&hist[qc >> 20], 1u);
    }
  }
  if (threadIdx.x < (m & 3u)) atomicAdd(&hist[rawbin(xs[4 * nfull + threadIdx.x])], 1u);
}

struct EmitCtx {
  int* out;
  const int* pts;
  float* data;
  uint32_t* lcnt;   
  uint32_t goff, eoff, cs, page0, pb, kE, kG, tie0, mbt;
};
__device__ __forceinline__ uint32_t slot_of(const EmitCtx& c, uint32_t el) {
  const uint32_t idx = c.cs + el;
  return (static_cast<uint32_t>(c.pts[(idx >> c.pb) - c.page0]) << c.pb) | (idx & ((1u << c.pb) - 1u));
}
__device__ __noinline__ void classify_global(const EmitCtx& c, const float* __restrict__ xs, uint32_t nsm,
                                            uint32_t nv, uint32_t m, uint32_t ecut) {
  auto one = [&](float x, uint32_t el) {
    const uint32_t kx = okey32(x);
    if (kx >= c.kG) {
      c.out[c.goff + (atomicAdd(c.lcnt, 1u) & 0xFFFFu)] = static_cast<int>(slot_of(c, el));
    } else if (kx >= c.kE && el < ecut) {
      const uint32_t ep = c.eoff + (atomicAdd(c.lcnt, 0x10000u) >> 16);
      if (ep < TIEMAX) st_async_v4(c.tie0 + ep * 16, make_uint4(__float_as_uint(x), c.cs + el, slot_of(c, el), 0u), c.mbt);
    }
  };
  const uint32_t nfull = m >> 2;
  for (uint32_t i = nsm + threadIdx.x; i < nfull; i += NT) {
    const float4 v = *reinterpret_cast<const float4*>(xs + 4 * i);
    if (okey32(fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w))) >= c.kE || isnan(v.x + v.y + v.z + v.w)) {
      one(v.x, 4 * i); one(v.y, 4 * i + 1); one(v.z, 4 * i + 2); one(v.w, 4 * i + 3);
    }
  }
  if (threadIdx.x < (m & 3u)) one(xs[4 * nfull + threadIdx.x], 4 * nfull + threadIdx.x);
}
__device__ __noinline__ uint32_t find_cut(const EmitCtx& ec, const float4* d4, const float* __restrict__ xs,
                                         uint32_t nsm, uint32_t nv, uint32_t m, uint32_t need, uint32_t* misc) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t vpw = (nv + NW - 1) / NW;
  const uint32_t wv0 = min(warp * vpw, nv), wv1 = min(wv0 + vpw, nv);
  auto ecount4 = [&](uint32_t i) -> uint32_t {   
    if (i >= nv) return 0u;
    uint32_t kk[4];
    uint32_t lim = 4;
    if (i < nsm) {
      const float4 v = d4[i];
      kk[0] = okey32(v.x); kk[1] = okey32(v.y); kk[2] = okey32(v.z); kk[3] = okey32(v.w);
    } else {
      lim = min(4u, m - 4 * i);
      for (uint32_t j = 0; j < 4; ++j) kk[j] = j < lim ? okey32(xs[4 * i + j]) : 0u;
    }
    uint32_t b = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) b |= (static_cast<uint32_t>(j) < lim && kk[j] >= ec.kE && kk[j] < ec.kG) ? (1u << j) : 0u;
    return b;
  };
  uint32_t wc = 0;
#pragma unroll 1
  for (uint32_t k = wv0; k < wv1; k += 32) wc += __popc(k + lane < wv1 ? ecount4(k + lane) : 0u);
  wc = __reduce_add_sync(0xFFFFFFFFu, wc);
  if (lane == 0) misc[warp] = wc;
  __syncthreads();
  uint32_t pre = 0;
#pragma unroll
  for (int w = 0; w < NW; ++w) pre += w < warp ? misc[w] : 0u;
  if (pre < need && pre + wc >= need) {   
    uint32_t run = pre;
#pragma unroll 1
    for (uint32_t k = wv0; k < wv1; k += 32) {
      const uint32_t b = k + lane < wv1 ? ecount4(k + lane) : 0u;
      const uint32_t c = __popc(b);
      uint32_t inc = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t t = __shfl_up_sync(0xFFFFFFFFu, inc, o);
        if (lane >= o) inc += t;
      }
      const uint32_t ex = run + inc - c;
      if (ex < need && ex + c >= need) {   
        uint32_t bb = b;
        for (uint32_t r = 1; r < need - ex; ++r) bb &= bb - 1;
        misc[46] = 4 * (k + lane) + (__ffs(bb) - 1) + 1;
      }
      run += __shfl_sync(0xFFFFFFFFu, inc, 31);
      if (run >= need) break;
    }
  }
  __syncthreads();
  return misc[46];
}
__device__ __noinline__ uint2 masks_key(const EmitCtx& ec, const float4* d4, uint32_t k0, uint32_t nsm, uint32_t ecut) {
  uint32_t cm = 0, gm = 0;
  for (uint32_t u = 0; u < 8; ++u) {
    const uint32_t i = threadIdx.x + (k0 + u) * NT;
    if (i >= nsm) break;
    const float4 v = d4[i];
    const uint32_t a = okey32(v.x), b = okey32(v.y), c2 = okey32(v.z), d2 = okey32(v.w);
    const uint32_t e0 = 4 * i;
    cm |= ((a >= ec.kG || (a >= ec.kE && e0 < ecut) ? 1u : 0u) | (b >= ec.kG || (b >= ec.kE && e0 + 1 < ecut) ? 2u : 0u) |
           (c2 >= ec.kG || (c2 >= ec.kE && e0 + 2 < ecut) ? 4u : 0u) | (d2 >= ec.kG || (d2 >= ec.kE && e0 + 3 < ecut) ? 8u : 0u)) << (4 * u);
    gm |= ((a >= ec.kG ? 1u : 0u) | (b >= ec.kG ? 2u : 0u) | (c2 >= ec.kG ? 4u : 0u) | (d2 >= ec.kG ? 8u : 0u)) << (4 * u);
  }
  return make_uint2(cm, gm);
}
__device__ __noinline__ uint32_t cut_mask(uint32_t cm, uint32_t gm, uint32_t k0, uint32_t nsm, uint32_t ecut) {
  for (uint32_t u = 0; u < 8; ++u) {
    const uint32_t i = threadIdx.x + (k0 + u) * NT;
    if (i >= nsm) break;
    for (uint32_t j = 0; j < 4; ++j)
      if (4 * i + j >= ecut) cm &= ~((1u << (4 * u + j)) & ~gm);
  }
  return cm;
}

__device__ __forceinline__ uint32_t sel_okey(uint32_t bits) {
  return bits ^ ((static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31)) | 0x80000000u);
}
__device__ __forceinline__ void warp_rank_select(const uint4* cand, uint32_t nc, uint32_t R, uint32_t base,
                                                 int* out, bool keys_raw) {
  const int lane = threadIdx.x & 31;
  const bool ok = lane < static_cast<int>(nc);
  const uint4 me = ok ? cand[lane] : make_uint4(0u, 0xFFFFFFFFu, 0u, 0u);
  const uint32_t mk = keys_raw ? sel_okey(me.x) : me.x;
  uint32_t rk = 0;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    const uint32_t kj = __shfl_sync(0xFFFFFFFFu, mk, j), ij = __shfl_sync(0xFFFFFFFFu, me.y, j);
    rk += (j < static_cast<int>(nc) && (kj > mk || (kj == mk && ij < me.y))) ? 1u : 0u;
  }
  if (ok && rk < R) out[base + rk] = static_cast<int>(me.z);
}
template <int NT_, int RB_>
__device__ __forceinline__ void leader_select2(const uint4* tie, uint32_t n, uint32_t R, uint32_t base, int* out,
                                               uint32_t* rh, uint32_t* misc, uint32_t kmin, int kbits) {
  constexpr int NW_ = NT_ / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (n <= 32) {
    if (warp == 0) warp_rank_select(tie, n, R, base, out, true);
    return;
  }
  if (n <= 64) {   
    const uint32_t e = tid >> 3, sub = tid & 7;
    uint32_t rk = 0, mk = 0;
    uint4 me = make_uint4(0, 0, 0, 0);
    if (e < n) {
      me = tie[e];
      mk = sel_okey(me.x);
#pragma unroll 4
      for (uint32_t q = sub; q < n; q += 8) {
        const uint2 o = *reinterpret_cast<const uint2*>(&tie[q]);
        const uint32_t ok = sel_okey(o.x);
        rk += (ok > mk || (ok == mk && o.y < me.y)) ? 1u : 0u;
      }
    }
    rk += __shfl_xor_sync(0xFFFFFFFFu, rk, 1);
    rk += __shfl_xor_sync(0xFFFFFFFFu, rk, 2);
    rk += __shfl_xor_sync(0xFFFFFFFFu, rk, 4);
    if (e < n && sub == 0 && rk < R) out[base + rk] = static_cast<int>(me.z);
    return;
  }
  uint32_t d[4], st = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t q = tid + i * NT_;
    const bool ok = q < n;
    d[i] = ok ? sel_okey(tie[q].x) - kmin : 0u;
    st |= ok ? (1u << i) : 0u;
  }
  uint32_t Rrem = R, nact = n;
  int hi = kbits;
  uint32_t* h0 = rh;
  uint32_t* h1 = rh + RB_;
  bool idxpass = false;
#pragma unroll 1
  while (Rrem > 0 && nact > 32) {
    if (hi == 0) {   
      if (idxpass) break;
      idxpass = true;
#pragma unroll
      for (int i = 0; i < 4; ++i) d[i] = 0x7FFFFFFFu - tie[tid + i * NT_].y;
      hi = 31;
    }
    const int lo = hi > 11 ? hi - 11 : 0;
    const uint32_t dmask = (1u << (hi - lo)) - 1u;
#pragma unroll
    for (int i = 0; i < 4; ++i) atomicAdd((st >> i) & 1u ? &h0[(d[i] >> lo) & dmask] : &misc[61], 1u);
    reinterpret_cast<uint4*>(h1)[tid] = make_uint4(0, 0, 0, 0);
    __syncthreads();
    const uint4 c = reinterpret_cast<const uint4*>(h0)[tid];
    const uint32_t s4 = c.x + c.y + c.z + c.w;
    uint32_t suf = s4;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t t = __shfl_down_sync(0xFFFFFFFFu, suf, o);
      if (lane + o < 32) suf += t;
    }
    if (lane == 0) misc[warp] = suf;
    __syncthreads();
    uint32_t above = suf - s4;
#pragma unroll
    for (int w = 0; w < NW_; ++w) above += w > warp ? misc[w] : 0u;
    const uint32_t cb[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
    for (int b = 3; b >= 0; --b) {
      if (above < Rrem && above + cb[b] >= Rrem) {
        misc[32] = tid * 4 + b;
        misc[33] = above;
        misc[34] = cb[b];
      }
      above += cb[b];
    }
    __syncthreads();
    const uint32_t D = misc[32], aboveD = misc[33], cntD = misc[34];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t dg = (d[i] >> lo) & dmask;
      const bool act = (st >> i) & 1u;
      st |= (act && dg > D) ? (1u << (4 + i)) : 0u;
      st &= (act && dg != D) ? ~(1u << i) : 0xFFFFFFFFu;
    }
    Rrem -= aboveD;
    nact = cntD;
    hi = lo;
    uint32_t* t = h0; h0 = h1; h1 = t;
    if (nact == Rrem) { st = (st & 0xF0u) | ((st & 0xFu) << 4); Rrem = 0; }
  }
  if (Rrem == 0) st &= 0xF0u;
  const uint32_t sel = __popc(st >> 4);
  uint32_t inc = sel;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t t = __shfl_up_sync(0xFFFFFFFFu, inc, o);
    if (lane >= o) inc += t;
  }
  uint32_t wb = 0;
  if (lane == 31 && inc) wb = atomicAdd(&misc[63], inc);
  wb = __shfl_sync(0xFFFFFFFFu, wb, 31) + inc - sel;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    if ((st >> (4 + i)) & 1u) out[base + wb++] = static_cast<int>(tie[tid + i * NT_].z);
  if (Rrem == 0) return;
  uint4* cand = reinterpret_cast<uint4*>(h1);
  if (tid == 0) misc[62] = 0;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; ++i)
    if ((st >> i) & 1u) {
      const uint4 e = tie[tid + i * NT_];
      const uint32_t q = atomicAdd(&misc[62], 1u);
      if (q < 32) cand[q] = make_uint4(sel_okey(e.x), e.y, e.z, 0u);
    }
  __syncthreads();
  if (warp == 0) warp_rank_select(cand, min(misc[62], 32u), Rrem, base + misc[63], out, false);
}

template <int C, bool STAMPS>
__global__ void __launch_bounds__(NT, 2) dsatk_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int row = blockIdx.y;
  const uint32_t rank = blockIdx.x;
  const uint32_t n = static_cast<uint32_t>(p.lens[row]);
  STAMP(0);
  if constexpr (STAMPS) { if (tid == 0) p.stamps[((long long)row * C + rank) * 32 + 15] = clock64(); }
  if (n <= static_cast<uint32_t>(p.topk)) {
    if (rank == 0) trivial_rows(p, row, n);
    return;
  }
  const Smem s = carve<C>(smem_raw, p.pt_cap);
  const uint32_t K = p.topk, pb = p.pb;
  const uint32_t chunk = ((n + C - 1) / C + 3) & ~3u;
  const uint32_t cs = min(rank * chunk, n);
  const uint32_t ce = min(cs + chunk, n);
  const uint32_t m = ce - cs;
  const uint32_t nv = (m + 3) >> 2;
  const uint32_t nsm = min(m >> 2, static_cast<uint32_t>(p.dcap) >> 2);   
  const float* __restrict__ xs = p.scores + static_cast<long long>(row) * p.sstride + cs;
  const int* __restrict__ ptr = p.pt + static_cast<long long>(row) * p.ptstride;
  const uint32_t page0 = cs >> pb;
  const uint32_t npages = m ? ((ce - 1) >> pb) - page0 + 1 : 0;
  const uint32_t mb_sup = smem_u32(&s.mbar[0]), mb_tie = smem_u32(&s.mbar[1]), mb_dat = smem_u32(&s.mbar[2]);

  if (tid == 0) {
    mbar_init(mb_sup, 1);
    mbar_init(mb_tie, 1);
    mbar_init(mb_dat, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  reinterpret_cast<uint4*>(s.hist)[tid] = make_uint4(0, 0, 0, 0);
  reinterpret_cast<uint4*>(s.hist)[tid + NT] = make_uint4(0, 0, 0, 0);
  cl_arrive_relaxed();   
  __syncthreads();
  if (tid == 0) mbar_expect_local(mb_sup, C * 256);
  pdl_wait();
  const int pt0 = tid < static_cast<int>(npages) ? ptr[page0 + tid] : 0;   
  STAMP(1);
  if (tid == 0 && nsm) {
    mbar_expect_local(mb_dat, nsm * 16);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     smem_u32(s.data)),
                 "l"(xs), "r"(nsm * 16), "r"(mb_dat)
                 : "memory");
  }
  const bool tail_reg = (nsm == (m >> 2)) && (m & 3u);
  float tv0 = 0.f, tv1 = 0.f, tv2 = 0.f;
  if (tid == 0 && tail_reg) {
    const uint32_t t0 = 4 * nsm, tn = m & 3u;
    tv0 = xs[t0];
    if (tn > 1) tv1 = xs[t0 + 1];
    if (tn > 2) tv2 = xs[t0 + 2];
  }
  pdl_trigger();
  if (tid < static_cast<int>(npages)) s.pts[tid] = pt0;
  for (uint32_t i = tid + NT; i < npages; i += NT) s.pts[i] = ptr[page0 + i];
  if (nsm) mbar_wait(mb_dat);
  STAMP(2);

  const float4* d4 = reinterpret_cast<const float4*>(s.data);
#pragma unroll 1
  for (uint32_t i = tid; i < nsm; i += NT) {
    const float4 v = d4[i];
    const uint32_t pa = h2u(__floats2half2_rn(v.x, v.y)), pc = h2u(__floats2half2_rn(v.z, v.w));
    atomicAdd(&s.hist[(pa >> 4) & 0xFFFu], 1u);
    atomicAdd(&s.hist[pa >> 20], 1u);
    atomicAdd(&s.hist[(pc >> 4) & 0xFFFu], 1u);
    atomicAdd(&s.hist[pc >> 20], 1u);
  }
  if (tid == 0 && tail_reg) {
    const uint32_t tn = m & 3u;
    atomicAdd(&s.hist[rawbin(tv0)], 1u);
    if (tn > 1) atomicAdd(&s.hist[rawbin(tv1)], 1u);
    if (tn > 2) atomicAdd(&s.hist[rawbin(tv2)], 1u);
  }
  if (nsm < nv && !tail_reg) hist_global(xs, nsm, nv, m, s.hist);
  __syncthreads();
  STAMP(3);

  {
    const uint4 a = reinterpret_cast<const uint4*>(s.hist)[tid * 2];
    const uint4 b = reinterpret_cast<const uint4*>(s.hist)[tid * 2 + 1];
    uint32_t sum = a.x + a.y + a.z + a.w + b.x + b.y + b.z + b.w;
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 1);
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 2);
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 4);
    const uint32_t q0 = __shfl_sync(0xFFFFFFFFu, sum, 0), q1 = __shfl_sync(0xFFFFFFFFu, sum, 8),
                   q2 = __shfl_sync(0xFFFFFFFFu, sum, 16), q3 = __shfl_sync(0xFFFFFFFFu, sum, 24);
    const uint32_t so = warp < 8 ? 32 + 4 * warp : 60 - 4 * warp;
    const uint4 o4 = warp < 8 ? make_uint4(q0, q1, q2, q3) : make_uint4(q3, q2, q1, q0);
    cl_wait();   
    if (lane < C) st_async_v4(mapa(smem_u32(&s.sup[rank * 64 + so]), lane), o4, mapa(mb_sup, lane));
  }
  if (warp == 0) {
    mbar_wait(mb_sup);
    STAMP(4);
    uint32_t a = 0, b = 0;
#pragma unroll
    for (int q = 0; q < C; ++q) {
      const uint2 t = *reinterpret_cast<const uint2*>(&s.sup[q * 64 + 2 * lane]);
      a += t.x;
      b += t.y;
    }
    const uint32_t ab_hi = warp_suffix_excl(a + b, lane);
    uint32_t S = 0xFFFFFFFFu, abS = 0;
    if (ab_hi < K && ab_hi + b >= K) { S = 2 * lane + 1; abS = ab_hi; }
    else if (ab_hi + b < K && ab_hi + b + a >= K) { S = 2 * lane; abS = ab_hi + b; }
    const uint32_t bal = __ballot_sync(0xFFFFFFFFu, S != 0xFFFFFFFFu);
    const int src = bal ? __ffs(bal) - 1 : 0;
    S = __shfl_sync(0xFFFFFFFFu, S, src);
    abS = __shfl_sync(0xFFFFFFFFu, abS, src);
    *reinterpret_cast<uint2*>(&s.l2[2 * lane]) = make_uint2(0u, 0u);
    if (lane == 0) { s.misc[53] = S; s.misc[47] = abS; }
  }
  __syncthreads();
  const uint32_t S = s.misc[53];
  uint2 pb2 = make_uint2(0u, 0u);
  if (warp < C) {
    const bool sneg = S < 32;
    const uint32_t addr = smem_u32(&s.hist[sneg ? 4094 - 64 * S - 2 * lane : 64 * (S - 32) + 2 * lane]);
    const uint2 r = ld_cl_v2(mapa(addr, warp));
    pb2 = sneg ? make_uint2(r.y, r.x) : r;
    atomicAdd(&s.l2[2 * lane], pb2.x);
    atomicAdd(&s.l2[2 * lane + 1], pb2.y);
  }
  __syncthreads();
  STAMP(10);
  const bool b10 = static_cast<int>(n) > p.floor10;   
  if (warp == 0) {
    const uint32_t abS = s.misc[47];
    const uint2 cc = *reinterpret_cast<const uint2*>(&s.l2[2 * lane]);
    const uint32_t c0 = cc.x, c1 = cc.y;
    const uint32_t ah = abS + warp_suffix_excl(c0 + c1, lane);   
    uint32_t T12 = 0xFFFFFFFFu, a12 = 0, h12 = 0;
    if (ah < K && ah + c1 >= K) { T12 = S * 64 + 2 * lane + 1; a12 = ah; h12 = c1; }
    else if (ah + c1 < K && ah + c1 + c0 >= K) { T12 = S * 64 + 2 * lane; a12 = ah + c1; h12 = c0; }
    const uint32_t bt = __ballot_sync(0xFFFFFFFFu, T12 != 0xFFFFFFFFu);
    const int sl = bt ? __ffs(bt) - 1 : 0;
    T12 = __shfl_sync(0xFFFFFFFFu, T12, sl);
    a12 = __shfl_sync(0xFFFFFFFFu, a12, sl);
    h12 = __shfl_sync(0xFFFFFFFFu, h12, sl);
    const uint32_t T10 = T12 >> 2;
    const int L0 = static_cast<int>(2 * (T10 - 16 * S));
    const uint32_t c2 = c0 + c1;
    const uint32_t h10 = __shfl_sync(0xFFFFFFFFu, c2, L0) + __shfl_sync(0xFFFFFFFFu, c2, L0 + 1);
    const uint32_t a10 = __shfl_sync(0xFFFFFFFFu, ah, L0 + 1);
    const bool ovf = (b10 ? h10 : h12) > TIEMAX;
    uint32_t tlo = T12, thi = T12, abv = a12, hin = h12;
    if (ovf && b10) { tlo = 4 * T10; thi = 4 * T10 + 3; abv = a10; hin = h10; }
    const uint32_t kx = exact_lbkey(lane == 0 ? tlo : thi + 1);   
    const uint32_t kG_ = __shfl_sync(0xFFFFFFFFu, kx, 1);
    if (lane == 0) {
      s.misc[48] = tlo;
      s.misc[49] = thi;
      s.misc[50] = abv;
      s.misc[51] = hin;
      s.misc[52] = ovf;
      s.misc[54] = kx;
      s.misc[55] = kG_;
      auto hbits = [](uint32_t b) -> uint32_t {
        const uint32_t k = (b << 4) & 0xFFFFu;
        return (k & 0x8000u) ? (k ^ 0x8000u) : (~k & 0xFFFFu);
      };
      const uint32_t hE = hbits(tlo), hG = hbits(thi + 1);
      auto hok = [](uint32_t b, uint32_t h) -> bool {
        return b >= 1u && b <= 4095u && h != 0x0000u && (h & 0x7C00u) != 0x7C00u;   
      };
      s.misc[57] = hE | (hG << 16);
      s.misc[58] = (hok(tlo, hE) && hok(thi + 1, hG)) ? 1u : 0u;
      s.misc[56] = 0;   
    }
  }
  __syncthreads();
  cl_arrive_relaxed();   
  STAMP(5);
  const uint32_t tlo = s.misc[48], thi = s.misc[49], Gtot = s.misc[50], Etot = s.misc[51];
  const bool ovf = s.misc[52] != 0;
  if (warp < C) {
    const uint32_t blo = tlo - S * 64, bhi = thi - S * 64;
    const uint32_t b0 = 2 * lane, b1 = 2 * lane + 1;
    const uint2 su = *reinterpret_cast<const uint2*>(&s.sup[warp * 64 + 2 * lane]);
    uint32_t above = (b0 > S ? su.x : 0u) + (b1 > S ? su.y : 0u) + (b0 > bhi ? pb2.x : 0u) + (b1 > bhi ? pb2.y : 0u);
    uint32_t inside = ((b0 >= blo && b0 <= bhi) ? pb2.x : 0u) + ((b1 >= blo && b1 <= bhi) ? pb2.y : 0u);
    above = __reduce_add_sync(0xFFFFFFFFu, above);
    inside = __reduce_add_sync(0xFFFFFFFFu, inside);
    if (lane == 0) s.cnt[warp] = make_uint2(above, inside);
  }
  if (rank == 0 && tid == 0) mbar_expect_local(mb_tie, min(Etot, static_cast<uint32_t>(TIEMAX)) * 16);
  __syncthreads();
  uint32_t goff = 0, eoff = 0;
#pragma unroll
  for (int q = 0; q < C; ++q) {
    const uint2 t = s.cnt[q];
    if (q < static_cast<int>(rank)) { goff += t.x; eoff += t.y; }
  }
  const uint32_t myE = s.cnt[rank].y;
  STAMP(6);

  int* __restrict__ out = p.out + static_cast<long long>(row) * K;
  EmitCtx ec;
  ec.out = out; ec.pts = s.pts; ec.data = s.data; ec.lcnt = &s.misc[56];
  ec.goff = goff; ec.eoff = eoff; ec.cs = cs; ec.page0 = page0; ec.pb = pb;
  ec.kE = s.misc[54]; ec.kG = s.misc[55];
  ec.tie0 = mapa(smem_u32(s.tie), 0); ec.mbt = mapa(mb_tie, 0);
  uint32_t ecut = m;
  if (ovf && eoff < TIEMAX && eoff + myE > TIEMAX) {
    ecut = find_cut(ec, d4, xs, nsm, nv, m, TIEMAX - eoff, s.misc);
  } else if (ovf && eoff >= TIEMAX) {
    ecut = 0;
  }
    const bool h2fast = s.misc[58] != 0;
    const uint32_t hEG = s.misc[57];
    const __half2 hE2 = __half2half2(__ushort_as_half(static_cast<unsigned short>(hEG & 0xFFFFu)));
    const __half2 hG2 = __half2half2(__ushort_as_half(static_cast<unsigned short>(hEG >> 16)));
#pragma unroll 1
    for (uint32_t k0 = 0; k0 * NT < nsm; k0 += 8) {
      uint32_t gm = 0, cm = 0;
#pragma unroll 1
      if (h2fast) {   
        for (uint32_t u = 0; u < 8; ++u) {
          const uint32_t i = tid + (k0 + u) * NT;
          if (i >= nsm) break;
          const float4 v = d4[i];
          const __half2 a2 = __floats2half2_rn(v.x, v.y), b2 = __floats2half2_rn(v.z, v.w);
          const uint32_t pe = __byte_perm(__hge2_mask(a2, hE2), __hge2_mask(b2, hE2), 0x6420);
          const uint32_t pg = __byte_perm(__hge2_mask(a2, hG2), __hge2_mask(b2, hG2), 0x6420);
          const uint32_t eb = ((pe & 0x08040201u) * 0x01010101u) >> 24;
          const uint32_t gb = ((pg & 0x08040201u) * 0x01010101u) >> 24;
          cm |= eb << (4 * u);
          gm |= gb << (4 * u);
        }
        if (ecut < m) cm = cut_mask(cm, gm, k0, nsm, ecut);
      } else {
        const uint2 mk = masks_key(ec, d4, k0, nsm, ecut);
        cm = mk.x;
        gm = mk.y;
      }
      const uint32_t c = __popc(gm) | (__popc(cm & ~gm) << 16);
      uint32_t inc = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t t = __shfl_up_sync(0xFFFFFFFFu, inc, o);
        if (lane >= o) inc += t;
      }
      uint32_t base = 0;
      if (lane == 31 && inc) base = atomicAdd(ec.lcnt, inc);
      base = __shfl_sync(0xFFFFFFFFu, base, 31) + inc - c;
      uint32_t gp = goff + (base & 0xFFFFu), ep = eoff + (base >> 16);
#pragma unroll 1
      while (cm) {
        const uint32_t bt = __ffs(cm) - 1;
        cm &= cm - 1;
        const uint32_t el = 4 * (tid + (k0 + (bt >> 2)) * NT) + (bt & 3);
        const uint32_t sl = slot_of(ec, el);
        if ((gm >> bt) & 1u) {
          out[gp++] = static_cast<int>(sl);
        } else {
          if (ep < TIEMAX) st_async_v4(ec.tie0 + ep * 16, make_uint4(__float_as_uint(s.data[el]), cs + el, sl, 0u), ec.mbt);
          ++ep;
        }
      }
    }
    if (tid == 0 && tail_reg) {
      const uint32_t tn = m & 3u;
      const float tvs[3] = {tv0, tv1, tv2};
      for (uint32_t j = 0; j < tn; ++j) {
        const uint32_t el = 4 * nsm + j, kx = okey32(tvs[j]);
        if (kx >= ec.kG) {
          out[goff + (atomicAdd(ec.lcnt, 1u) & 0xFFFFu)] = static_cast<int>(slot_of(ec, el));
        } else if (kx >= ec.kE && el < ecut) {
          const uint32_t ep = eoff + (atomicAdd(ec.lcnt, 0x10000u) >> 16);
          if (ep < TIEMAX) st_async_v4(ec.tie0 + ep * 16, make_uint4(__float_as_uint(tvs[j]), cs + el, slot_of(ec, el), 0u), ec.mbt);
        }
      }
    }
    if (nsm < nv && !tail_reg) classify_global(ec, xs, nsm, nv, m, ecut);

  STAMP(7);
  cl_wait();   
  if (rank != 0) return;
  if (tid == 0) s.misc[63] = 0;
  reinterpret_cast<uint4*>(s.data)[tid] = make_uint4(0, 0, 0, 0);   
  mbar_wait(mb_tie);
  __syncthreads();
  STAMP(8);
  const uint32_t R = K - Gtot;   
  const uint32_t ne = min(Etot, static_cast<uint32_t>(TIEMAX));
  if (ne <= R) {
    for (uint32_t i = tid; i < ne; i += NT) out[Gtot + i] = static_cast<int>(s.tie[i].z);
  } else {
    const uint32_t kE = s.misc[54], kG = s.misc[55];
    const int kbits = 32 - __clz(kG - kE);
    leader_select2<NT, RB>(s.tie, ne, R, Gtot, out, reinterpret_cast<uint32_t*>(s.data), s.misc, kE, kbits);
  }
  STAMP(9);
}

template <int C, bool STAMPS>
void prepare_kernel() {
  static bool done = false;
  if (done) return;
  auto kern = dsatk_kernel<C, STAMPS>;
  int dev = 0, optin = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
  if (C > 8) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  done = true;
}

template <int C, bool STAMPS>
void launch(const Params& p, int rows, int dyn, cudaStream_t stream, bool pdl) {
  auto kern = dsatk_kernel<C, STAMPS>;
  prepare_kernel<C, STAMPS>();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(C, rows, 1);
  cfg.blockDim = dim3(NT, 1, 1);
  cfg.dynamicSmemBytes = dyn;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = C;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  const cudaError_t e = cudaLaunchKernelEx(&cfg, kern, p);
  TORCH_CHECK(e == cudaSuccess, "dsa_topk launch failed: ", cudaGetErrorString(e));
}

}   

void topk(torch::Tensor scores, torch::Tensor lens, torch::Tensor page_table, torch::Tensor out, int64_t page_size,
          int64_t cluster, int64_t dcap, int64_t floor10, torch::Tensor stamps, bool pdl) {
  TORCH_CHECK(scores.dtype() == torch::kFloat32 && scores.stride(1) == 1 && scores.stride(0) % 4 == 0);
  TORCH_CHECK(lens.dtype() == torch::kInt32 && page_table.dtype() == torch::kInt32 && out.dtype() == torch::kInt32);
  TORCH_CHECK(page_table.stride(1) == 1 && out.is_contiguous());
  TORCH_CHECK((reinterpret_cast<uintptr_t>(scores.data_ptr()) & 15) == 0);
  const int B = static_cast<int>(scores.size(0));
  const long long L = scores.size(1);
  const int K = static_cast<int>(out.size(1));
  TORCH_CHECK(K > 0 && K <= dsatk::KMAX && out.size(0) == B && lens.size(0) == B);
  TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0);
  const int pb = __builtin_ctzll(page_size);
  if (floor10 < 0) {
    const long long fl = B <= 15 ? 32768 : 65536;
    TORCH_CHECK(B <= 30, "dsa_topk covers the stock small-batch regime (rows <= 30)");
    floor10 = (L > fl) ? fl : 0x7FFFFFFF;
  }
  const int C = static_cast<int>(cluster);
  TORCH_CHECK(dcap >= 4096 && dcap % 4 == 0);
  const long long chunk = ((L + C - 1) / C + 3) & ~3LL;
  const int pt_cap = static_cast<int>((chunk >> pb) + 2);
  dsatk::Params p;
  p.scores = scores.data_ptr<float>();
  p.sstride = scores.stride(0);
  p.lens = lens.data_ptr<int>();
  p.pt = page_table.data_ptr<int>();
  p.ptstride = page_table.stride(0);
  p.out = out.data_ptr<int>();
  p.stamps = (stamps.defined() && stamps.numel() > 0) ? stamps.data_ptr<int64_t>() : nullptr;
  p.topk = K;
  p.pb = pb;
  p.floor10 = static_cast<int>(floor10);
  p.pt_cap = pt_cap;
  p.dcap = static_cast<int>(dcap);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const bool S = p.stamps != nullptr;
#define DSATK_CASE(CC)                                                                                    \
  if (C == CC) {                                                                                        \
    const int dyn = dsatk::smem_fixed<CC>() + dsatk::al16(pt_cap * 4) + static_cast<int>(dcap) * 4;         \
    if (S) dsatk::launch<CC, true>(p, B, dyn, st, pdl); else dsatk::launch<CC, false>(p, B, dyn, st, pdl);   \
    return;                                                                                             \
  }
  DSATK_CASE(16) DSATK_CASE(12) DSATK_CASE(9) DSATK_CASE(8) DSATK_CASE(6) DSATK_CASE(5) DSATK_CASE(4)
  TORCH_CHECK(false, "unsupported cluster size");
}

void topk_auto(torch::Tensor scores, torch::Tensor lens, torch::Tensor page_table, torch::Tensor out,
               int64_t page_size, bool pdl) {
  const int64_t B = scores.size(0);
  int64_t C = 16, dcap = 16384;
  if (B <= 4) { C = 16; dcap = 16384; }
  else if (B <= 12) { C = 9; dcap = 28672; }
  else if (B <= 22) { C = 6; dcap = 43776; }
  else if (B <= 26) { C = 5; dcap = 40960; }
  else { C = 4; dcap = 40960; }
  topk(scores, lens, page_table, out, page_size, C, dcap, -1, torch::Tensor(), pdl);
}

void prepare() {
  dsatk::prepare_kernel<16, false>();
  dsatk::prepare_kernel<9, false>();
  dsatk::prepare_kernel<6, false>();
  dsatk::prepare_kernel<5, false>();
  dsatk::prepare_kernel<4, false>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk", &topk, "explicit config: (scores, lens, page_table, out, page_size, cluster, dcap, floor10, stamps, pdl)");
  m.def("topk_auto", &topk_auto, "dispatch by rows: (scores, lens, page_table, out, page_size, pdl)");
  m.def("prepare", &prepare, "set kernel attributes (call before graph capture)");
}
