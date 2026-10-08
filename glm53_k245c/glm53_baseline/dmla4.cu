#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace dmla4 {

constexpr int H = 64, DQK = 576, DV = 512, TOPK = 2048, TK = 128;
constexpr int NCB = 5;                     
constexpr int KBLK = TK * 128;             
constexpr int KTILE = NCB * KBLK;          
constexpr int QBLK = H * 128;              
constexpr int QBYTES = NCB * QBLK;         
constexpr int PTILE = TK * H;              
constexpr int MAXNT = 2;
constexpr int NSM = 8;                     
constexpr int THREADS = (NSM + 2) * 32;
constexpr int WMMA = NSM, WQ = NSM + 1;
constexpr float PSCALE = 448.f, LN2 = 0.6931471805599453f;
constexpr int MAXT = 64;

constexpr int OFF_K = 0;
constexpr int OFF_Q = OFF_K + MAXNT * KTILE;
constexpr int OFF_P = OFF_Q + QBYTES;
constexpr int OFF_RED = OFF_P + MAXNT * PTILE;             
constexpr int OFF_IDX = OFF_RED + (4 * H + 2 * H) * 4;     
constexpr int OFF_BAR = OFF_IDX + MAXNT * TK * 4;          
constexpr int OFF_TMEM = OFF_BAR + 24 * 8;
constexpr int SMEM = OFF_TMEM + 16 + 1024;                 

constexpr uint32_t IDESC_QK = (1u << 4) | ((64u >> 3) << 17) | ((128u >> 4) << 24);    
constexpr uint32_t IDESC_PV = IDESC_QK | (1u << 15) | (1u << 16);                      

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint64_t sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo, uint32_t layout) {
  return uint64_t((addr >> 4) & 0x3FFF) | (uint64_t((lbo >> 4) & 0x3FFF) << 16) |
         (uint64_t((sbo >> 4) & 0x3FFF) << 32) | (1ull << 46) | (uint64_t(layout) << 61);
}
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}"
               :: "r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_gather4(uint32_t dst, const CUtensorMap* tm, int col, int4 rows, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1 "
               "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
               :: "r"(dst), "l"(tm), "r"(col), "r"(rows.x), "r"(rows.y), "r"(rows.z), "r"(rows.w), "r"(bar) : "memory");
}
__device__ __forceinline__ void umma(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}"
               :: "r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(acc) : "memory");
}
__device__ __forceinline__ void umma_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, float (&v)[32]) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
        "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
        "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
        "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ float ex2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ void sm_sync() { asm volatile("bar.sync 1, %0;" :: "n"(NSM * 32) : "memory"); }

template <bool MAX>
__device__ __forceinline__ void warp_reduce32(float (&v)[32], int lane) {
#pragma unroll
  for (int o = 16, n = 32; o >= 1; o >>= 1, n >>= 1) {
    const bool up = (lane & o) != 0;
#pragma unroll
    for (int i = 0; i < n / 2; ++i) {
      const float send = up ? v[i] : v[i + n / 2];
      const float keep = up ? v[i + n / 2] : v[i];
      const float recv = __shfl_xor_sync(0xffffffffu, send, o);
      v[i] = MAX ? fmaxf(keep, recv) : keep + recv;
    }
  }
}

template <int NT, int FUSE>
__global__ void __launch_bounds__(THREADS, 1) dmla4_kernel(
    const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_kv,
    const int32_t* __restrict__ pt, int64_t pt_stride, const int32_t* __restrict__ lens, int keys, float scale_log2,
    __nv_bfloat16* __restrict__ parts, float* __restrict__ lse_out, int pdl, unsigned long long* __restrict__ stamps,
    __nv_bfloat16* __restrict__ out) {
  extern __shared__ uint8_t smem_raw[];
#ifdef DMLA_STAMPS
  const bool stamp_cta = stamps != nullptr && blockIdx.x == 0 && blockIdx.y == 0;
#define ST(n) do { if (stamp_cta) { unsigned long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t)); stamps[n] = _t; } } while (0)
#else
#define ST(n) do { } while (0)
#endif
  uint8_t* smem = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);    
  float* s_red = reinterpret_cast<float*>(smem + OFF_RED);         
  float* s_m = s_red + 4 * H;                                      
  float* s_inv = s_m + H;                                          
  int32_t* s_idx = reinterpret_cast<int32_t*>(smem + OFF_IDX);     
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + OFF_BAR);
  uint32_t* s_tmem = reinterpret_cast<uint32_t*>(smem + OFF_TMEM);
  const uint32_t bar_q = smem_u32(&bars[0]), bar_idx = smem_u32(&bars[19]);
#define BAR_K(j, c) smem_u32(&bars[1 + 5 * (j) + (c)])
#define BAR_S(j) smem_u32(&bars[11 + (j)])
#define BAR_P(j) smem_u32(&bars[13 + (j)])
#define BAR_O(cb) smem_u32(&bars[15 + (cb)])
  const uint32_t k_base = smem_u32(smem + OFF_K), q_base = smem_u32(smem + OFF_Q), p_base = smem_u32(smem + OFF_P);

  const int split = blockIdx.x, t = blockIdx.y, splits = gridDim.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) ST(0);
  if (tid == 0) {
    mbar_init(bar_q, 1);
    for (int j = 0; j < NT; ++j) {
      for (int c = 0; c < NCB; ++c) mbar_init(BAR_K(j, c), 1);
      mbar_init(BAR_S(j), 1);
      mbar_init(BAR_P(j), NSM * 32);
    }
    for (int cb = 0; cb < DV / 128; ++cb) mbar_init(BAR_O(cb), 1);
    mbar_init(bar_idx, NSM * 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == WMMA) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(smem_u32(s_tmem)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *s_tmem;                  
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (tid == 0) ST(1);

  const int key0 = split * keys;
  if (warp == WQ) {
    if (lane == 0) {
      mbar_expect_tx(bar_q, QBYTES);
#pragma unroll
      for (int c = 0; c < NCB; ++c) tma_load_2d(q_base + c * QBLK, &tm_q, c * 128, t * H, bar_q);
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int c = 0; c < NCB; ++c) mbar_expect_tx(BAR_K(j, c), KBLK);
    }
  } else if (warp == WMMA) {
    if (lane == 0) {
      mbar_wait(bar_q, 0);
#pragma unroll
      for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int kk = 0; kk < DQK / 32; ++kk) {             
          const int c = kk >> 2, off = (kk & 3) * 32;
          if ((kk & 3) == 0) {
            mbar_wait(BAR_K(j, c), 0);
            tc_fence_after();
            if (j == 0 && c == 0) ST(2);
          }
          umma(tmem + 256 + 64 * j, sdesc(k_base + j * KTILE + c * KBLK + off, 16, 1024, 2),
               sdesc(q_base + c * QBLK + off, 16, 1024, 2), IDESC_QK, kk > 0);
        }
        umma_commit(BAR_S(j));                             
        if (j == 0) ST(14);
      }
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        mbar_wait(BAR_P(j), 0);                            
        tc_fence_after();
        if (j == 0) ST(5);
#pragma unroll
        for (int cb = 0; cb < DV / 128; ++cb) {
#pragma unroll
          for (int ks = 0; ks < TK / 32; ++ks)
            umma(tmem + 64 * cb, sdesc(k_base + j * KTILE + cb * KBLK + ks * 32 * 128, KBLK, 1024, 2),
                 sdesc(p_base + j * PTILE + ks * 32 * 64, PTILE, 512, 4), IDESC_PV, (j | ks) != 0);
          if (j == NT - 1) umma_commit(BAR_O(cb));         
        }
      }
    }
    __syncwarp();
  } else {
    if (lane < 4) {
      const int nvalid = min(__ldg(lens + t), TOPK);
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int r0 = TK * j + 16 * warp + 4 * lane;    
        int v[4];
        if (r0 < keys) {
          const int4 s4 = __ldg(reinterpret_cast<const int4*>(pt + size_t(t) * pt_stride + key0 + r0));
          v[0] = s4.x; v[1] = s4.y; v[2] = s4.z; v[3] = s4.w;
        } else {
          v[0] = v[1] = v[2] = v[3] = -1;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (r0 + i >= keys || key0 + r0 + i >= nvalid || v[i] < 0) v[i] = -1;
          s_idx[r0 + i] = v[i];
        }
        const int4 rows = make_int4(v[0], v[1], v[2], v[3]);
#pragma unroll
        for (int c = 0; c < NCB; ++c)
          tma_gather4(k_base + j * KTILE + c * KBLK + (r0 - TK * j) * 128, &tm_kv, c * 128, rows, BAR_K(j, c));
      }
      mbar_arrive(bar_idx);                       
    }
    if (pdl && tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    const int q4 = warp & 3, hh = warp >> 2;
    const uint32_t lane_base = uint32_t(32 * q4) << 16;
    const float c = scale_log2;
    mbar_wait(bar_idx, 0);
    float m[32], v[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) m[i] = -INFINITY;
    bool valid[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {                        
      mbar_wait(BAR_S(j), 0);
      tc_fence_after();
      if (tid == 0 && j == 0) ST(3);
      valid[j] = s_idx[TK * j + 32 * q4 + lane] >= 0;
      tmem_ld32(tmem + lane_base + 256 + 64 * j + 32 * hh, v);
      if (valid[j]) {
#pragma unroll
        for (int i = 0; i < 32; ++i) m[i] = fmaxf(m[i], v[i]);
      }
    }
    if (tid == 0) ST(11);
    warp_reduce32<true>(m, lane);
    if (tid == 0) ST(12);
    s_red[q4 * H + 32 * hh + lane] = m[0];
    sm_sync();
    if (q4 == 0) {
      const int h = 32 * hh + lane;
      s_m[h] = fmaxf(fmaxf(s_red[h], s_red[H + h]), fmaxf(s_red[2 * H + h], s_red[3 * H + h])) * c;
    }
    sm_sync();
#pragma unroll
    for (int i = 0; i < 32; ++i) m[i] = s_m[32 * hh + i];
    if (tid == 0) ST(13);
    constexpr float LOG2_PSCALE = 8.807354922057604f;
    float mq[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) mq[i] = m[i] - LOG2_PSCALE;
    float l[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) l[i] = 0.f;
    const int key = 32 * q4 + lane;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      tmem_ld32(tmem + lane_base + 256 + 64 * j + 32 * hh, v);
      uint8_t* prow = smem + OFF_P + j * PTILE + key * 64;
      const bool on = valid[j] && m[0] != -INFINITY;
#pragma unroll
      for (int half = 0; half < 2; ++half) {              
        uint32_t w4[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          uint32_t packed = 0;
#pragma unroll
          for (int pr = 0; pr < 2; ++pr) {
            const int i = 16 * half + 4 * e + 2 * pr;
            const float p0 = on ? ex2(fmaf(v[i], c, -mq[i])) : 0.f;
            const float p1 = on ? ex2(fmaf(v[i + 1], c, -mq[i + 1])) : 0.f;
            l[i] += p0;                                      
            l[i + 1] += p1;
            const uint16_t q2 = static_cast<uint16_t>(
                __nv_cvt_float2_to_fp8x2(make_float2(p0, p1), __NV_SATFINITE, __NV_E4M3));
            packed |= uint32_t(q2) << (16 * pr);
          }
          w4[e] = packed;
        }
        const int ch = 2 * hh + half;                      
        *reinterpret_cast<uint4*>(prow + 16 * (ch ^ ((key >> 1) & 3))) = make_uint4(w4[0], w4[1], w4[2], w4[3]);
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      mbar_arrive(BAR_P(j));                              
    }
    if (tid == 0) ST(4);
    warp_reduce32<false>(l, lane);
    s_red[q4 * H + 32 * hh + lane] = l[0];                 
    sm_sync();
    if (q4 == 0) {
      const int h = 32 * hh + lane;
      const float L = s_red[h] + s_red[H + h] + s_red[2 * H + h] + s_red[3 * H + h];
      s_inv[h] = L > 0.f ? 1.f / L : 0.f;
      lse_out[(size_t(t) * splits + split) * H + h] = L > 0.f ? (s_m[h] * LN2 + logf(L * (1.f / PSCALE))) : INFINITY;
    }
    sm_sync();
    float inv[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) inv[i] = s_inv[32 * hh + i];
    if (tid == 0) ST(6);
    uint4* pbase = reinterpret_cast<uint4*>(parts + (size_t(t) * splits + split) * DV * H);
#pragma unroll
    for (int cb = 0; cb < DV / 128; ++cb) {
      mbar_wait(BAR_O(cb), 0);
      tc_fence_after();
      tmem_ld32(tmem + lane_base + 64 * cb + 32 * hh, v);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint32_t w[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int k = 8 * i + 2 * e;
          const __nv_bfloat162 b = __floats2bfloat162_rn(v[k] * inv[k], v[k + 1] * inv[k + 1]);
          w[e] = *reinterpret_cast<const uint32_t*>(&b);
        }
        pbase[((cb * 2 + hh) * 4 + i) * 128 + 32 * q4 + lane] = make_uint4(w[0], w[1], w[2], w[3]);
      }
    }
    if (tid == 0) ST(7);
  }
  tc_fence_before();
  __syncthreads();
  if (warp == WMMA) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem) : "memory");
    if (lane == 0) ST(8);
  }
  if constexpr (FUSE > 0) {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp >= NSM) return;
    if (tid == 0) ST(9);
    constexpr int COLS = DV / FUSE;                  
    constexpr int ITEMS = COLS * 8;                  
    constexpr int PER = ITEMS / (NSM * 32);
    float* s_w = reinterpret_cast<float*>(smem + OFF_P);    
    const uint4* pb = reinterpret_cast<const uint4*>(parts) + size_t(t) * FUSE * (DV * H / 8);
    uint4 pv[PER][FUSE];
#pragma unroll
    for (int u = 0; u < PER; ++u) {
      const int item = tid + u * NSM * 32;
      const int cl = item % COLS, combo = item / COLS;
      const int col = split * COLS + cl;
      const int idx = ((col / 128) * 8 + combo) * 128 + (col % 128);
#pragma unroll
      for (int s2 = 0; s2 < FUSE; ++s2) pv[u][s2] = *(pb + size_t(s2) * (DV * H / 8) + idx);
    }
    if (tid < H) {
      float l[FUSE];
      float mx = -INFINITY;
#pragma unroll
      for (int s2 = 0; s2 < FUSE; ++s2) {
        l[s2] = lse_out[(size_t(t) * FUSE + s2) * H + tid];
        if (l[s2] != INFINITY) mx = fmaxf(mx, l[s2]);
      }
      float den = 0.f;
#pragma unroll
      for (int s2 = 0; s2 < FUSE; ++s2) {
        l[s2] = (l[s2] == INFINITY || mx == -INFINITY) ? 0.f : __expf(l[s2] - mx);
        den += l[s2];
      }
      const float iv = den > 0.f ? 1.f / den : 0.f;
#pragma unroll
      for (int s2 = 0; s2 < FUSE; ++s2) s_w[s2 * H + tid] = l[s2] * iv;
    }
    sm_sync();
#pragma unroll
    for (int u = 0; u < PER; ++u) {
      const int item = tid + u * NSM * 32;
      const int cl = item % COLS, combo = item / COLS;
      const int col = split * COLS + cl;
      const int h0 = 8 * combo;                                    
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int s2 = 0; s2 < FUSE; ++s2) {
        const uint32_t* vv = reinterpret_cast<const uint32_t*>(&pv[u][s2]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vv[e]));
          acc[2 * e] += s_w[s2 * H + h0 + 2 * e] * f.x;
          acc[2 * e + 1] += s_w[s2 * H + h0 + 2 * e + 1] * f.y;
        }
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) out[(size_t(t) * H + h0 + e) * DV + col] = __float2bfloat16_rn(acc[e]);
    }
    if (tid == 0) ST(10);
  }
}

template <int SK>
__global__ void __launch_bounds__(256) dmla4_merge(const __nv_bfloat16* __restrict__ parts,
                                                   const float* __restrict__ lse, __nv_bfloat16* __restrict__ out,
                                                   int pdl) {
  __shared__ float s_w[SK][32];
  __shared__ __align__(16) __nv_bfloat16 s_o[32][128 + 8];
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
  const int cb = blockIdx.x, hh = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
  if (tid < 32) {
    const int h = 32 * hh + tid;
    float l[SK];
    float m = -INFINITY;
#pragma unroll
    for (int s = 0; s < SK; ++s) {
      l[s] = __ldg(lse + (size_t(t) * SK + s) * H + h);
      if (l[s] != INFINITY) m = fmaxf(m, l[s]);
    }
    float den = 0.f;
#pragma unroll
    for (int s = 0; s < SK; ++s) {
      l[s] = (l[s] == INFINITY || m == -INFINITY) ? 0.f : __expf(l[s] - m);
      den += l[s];
    }
    const float iv = den > 0.f ? 1.f / den : 0.f;
#pragma unroll
    for (int s = 0; s < SK; ++s) s_w[s][tid] = l[s] * iv;
  }
  __syncthreads();
  const int j = tid & 127, ih = tid >> 7;              
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.f;
  constexpr int B = 8;                                 
  const uint4* base = reinterpret_cast<const uint4*>(parts) + size_t(t) * SK * (DV * H / 8);
#pragma unroll
  for (int s0 = 0; s0 < SK; s0 += B) {
    uint4 v[B][2];
#pragma unroll
    for (int b = 0; b < B; ++b)
#pragma unroll
      for (int u = 0; u < 2; ++u)
        v[b][u] = __ldg(base + size_t(s0 + b) * (DV * H / 8) + ((cb * 2 + hh) * 4 + 2 * ih + u) * 128 + j);
#pragma unroll
    for (int b = 0; b < B; ++b)
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const uint32_t* vv = reinterpret_cast<const uint32_t*>(&v[b][u]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vv[e]));
          const int k = 8 * u + 2 * e;                 
          acc[k] += s_w[s0 + b][16 * ih + k] * f.x;
          acc[k + 1] += s_w[s0 + b][16 * ih + k + 1] * f.y;
        }
      }
  }
#pragma unroll
  for (int k = 0; k < 16; ++k) s_o[16 * ih + k][j] = __float2bfloat16_rn(acc[k]);
  __syncthreads();
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int h = tid >> 3, c0 = 16 * (tid & 7);
  const uint4* src = reinterpret_cast<const uint4*>(&s_o[h][c0]);
  uint4* dst = reinterpret_cast<uint4*>(out + (size_t(t) * H + 32 * hh + h) * DV + 128 * cb + c0);
  dst[0] = src[0];
  dst[1] = src[1];
}

static PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess &&
                q == cudaDriverEntryPointSuccess && p, "dmla4: cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

static CUtensorMap make_map(const void* base, uint64_t rows, uint32_t box_rows) {
  CUtensorMap m;
  const uint64_t size[2] = {uint64_t(DQK), rows};
  const uint64_t stride[1] = {uint64_t(DQK)};
  const uint32_t box[2] = {128u, box_rows};
  const uint32_t es[2] = {1u, 1u};
  const CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), size, stride, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "dmla4: cuTensorMapEncodeTiled failed ", int(r));
  return m;
}

static void launch_ex(const void* fn, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, bool pdl, void** args,
                      const char* what) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t e = cudaLaunchKernelExC(&cfg, fn, args);
  TORCH_CHECK(e == cudaSuccess, what, " launch failed: ", cudaGetErrorString(e));
}

template <int NT, int FUSE>
static void launch_attn(const CUtensorMap& mq, const CUtensorMap& mkv, const int32_t* pt, int64_t pst,
                        const int32_t* lens, int keys, float sl2, __nv_bfloat16* parts, float* lse, int pdl, int splits,
                        int T, cudaStream_t stream, unsigned long long* stamps, __nv_bfloat16* out) {
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(dmla4_kernel<NT, FUSE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)
                == cudaSuccess, "dmla4: smem attribute");
    if (FUSE > 8)
      TORCH_CHECK(cudaFuncSetAttribute(dmla4_kernel<NT, FUSE>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1)
                  == cudaSuccess, "dmla4: non-portable cluster size");
    attr = true;
  }
  void* args[] = {const_cast<CUtensorMap*>(&mq), const_cast<CUtensorMap*>(&mkv), &pt, &pst, &lens, &keys, &sl2,
                  &parts, &lse, &pdl, &stamps, &out};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(splits, T);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = stream;
  cudaLaunchAttribute at[2];
  int na = 0;
  if (FUSE > 0) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = FUSE; at[na].val.clusterDim.y = 1; at[na].val.clusterDim.z = 1;
    ++na;
  }
  if (pdl) {
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  const cudaError_t e = cudaLaunchKernelExC(&cfg, reinterpret_cast<const void*>(dmla4_kernel<NT, FUSE>), args);
  TORCH_CHECK(e == cudaSuccess, "dmla4 launch failed: ", cudaGetErrorString(e));
}

template <int NT, int FUSE>
static int max_clusters_t() {
  cudaFuncSetAttribute(dmla4_kernel<NT, FUSE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  if (FUSE > 8) cudaFuncSetAttribute(dmla4_kernel<NT, FUSE>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(FUSE, 1);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = SMEM;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = FUSE; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  int n = -1;
  TORCH_CHECK(cudaOccupancyMaxActiveClusters(&n, reinterpret_cast<const void*>(dmla4_kernel<NT, FUSE>), &cfg)
              == cudaSuccess, "dmla4: occupancy query");
  return n;
}
int64_t max_clusters(int64_t splits) {
  return splits == 16 ? max_clusters_t<1, 16>() : max_clusters_t<2, 8>();
}

template <int SK>
static void launch_merge(const __nv_bfloat16* parts, const float* lse, __nv_bfloat16* out, int T, int pdl,
                         cudaStream_t stream) {
  void* args[] = {&parts, &lse, &out, &pdl};
  launch_ex(reinterpret_cast<const void*>(dmla4_merge<SK>), dim3(DV / 128, 2, T), dim3(256), 0, stream, pdl, args,
            "dmla4 merge");
}

void decode(torch::Tensor q, torch::Tensor kv, torch::Tensor pt, torch::Tensor lens, double scale, int64_t splits,
            torch::Tensor parts, torch::Tensor lse, torch::Tensor out, bool pdl, bool merge,
            c10::optional<torch::Tensor> dbg, bool fuse) {
  unsigned long long* stamps = dbg.has_value() ? reinterpret_cast<unsigned long long*>(dbg->data_ptr()) : nullptr;
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kFloat8_e4m3fn && q.dim() == 3 && q.size(1) == H
              && q.size(2) == DQK && q.is_contiguous(), "dmla4: q must be contiguous e4m3 [T, 64, 576]");
  TORCH_CHECK(kv.scalar_type() == at::kFloat8_e4m3fn && kv.size(-1) == DQK && kv.is_contiguous()
              && reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0, "dmla4: kv must be contiguous e4m3 [S, 576]");
  const int T = q.size(0);
  TORCH_CHECK(T >= 1 && T <= MAXT, "dmla4: tokens ", T);
  TORCH_CHECK(splits == 8 || splits == 16 || splits == 32, "dmla4: splits must be 8, 16 or 32");
  TORCH_CHECK(pt.scalar_type() == at::kInt && pt.dim() == 2 && pt.size(0) == T && pt.size(1) >= TOPK
              && pt.stride(1) == 1 && pt.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(pt.data_ptr()) % 16 == 0,
              "dmla4: page table must be 16-byte aligned int32 [T, >= 2048]");
  TORCH_CHECK(lens.scalar_type() == at::kInt && lens.numel() == T && lens.is_contiguous(), "dmla4: lens");
  TORCH_CHECK(parts.scalar_type() == at::kBFloat16 && parts.is_contiguous() && parts.numel() == int64_t(T) * splits * H * DV,
              "dmla4: parts must be bf16 [T, splits, 512, 64]");
  TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.is_contiguous() && lse.numel() == int64_t(T) * splits * H,
              "dmla4: lse must be fp32 [T, splits, 64]");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == int64_t(T) * H * DV,
              "dmla4: out must be contiguous bf16 [T, 64, 512]");
  c10::cuda::CUDAGuard guard(q.device());
  auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
  const CUtensorMap mq = make_map(q.data_ptr(), uint64_t(T) * H, H);
  const CUtensorMap mkv = make_map(kv.data_ptr(), uint64_t(kv.numel() / DQK), 1);
  const int keys = TOPK / int(splits);
  const float sl2 = float(scale) * 1.4426950408889634f;
  __nv_bfloat16* pp = reinterpret_cast<__nv_bfloat16*>(parts.data_ptr());
  float* lsp = lse.data_ptr<float>();
  const int pd = pdl ? 1 : 0;
  const int32_t* ptp = pt.data_ptr<int32_t>();
  const int32_t* lp = lens.data_ptr<int32_t>();
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if (fuse) {
    TORCH_CHECK(splits == 8 || splits == 16, "dmla4: the fused merge needs splits 8 or 16");
    if (splits == 8) launch_attn<2, 8>(mq, mkv, ptp, pt.stride(0), lp, keys, sl2, pp, lsp, pd, 8, T, stream, stamps, op);
    else launch_attn<1, 16>(mq, mkv, ptp, pt.stride(0), lp, keys, sl2, pp, lsp, pd, 16, T, stream, stamps, op);
    return;
  }
  if (keys > TK) launch_attn<2, 0>(mq, mkv, ptp, pt.stride(0), lp, keys, sl2, pp, lsp, pd, int(splits), T, stream,
                                   stamps, op);
  else launch_attn<1, 0>(mq, mkv, ptp, pt.stride(0), lp, keys, sl2, pp, lsp, pd, int(splits), T, stream, stamps, op);
  if (!merge) return;
  if (splits == 8) launch_merge<8>(pp, lsp, op, T, pd, stream);
  else if (splits == 16) launch_merge<16>(pp, lsp, op, T, pd, stream);
  else launch_merge<32>(pp, lsp, op, T, pd, stream);
}

}   

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &dmla4::decode);
  m.def("max_clusters", &dmla4::max_clusters);
}
