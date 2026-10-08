#pragma once
#include <cuda_fp16.h>    

namespace dpo {

namespace fg {
constexpr int RM = 16;                  
constexpr int NB = 192;                 
constexpr int NBLK = 8;                 
constexpr int KS = 16;                  
constexpr int KSPL = 1024;              
constexpr int BK = 64;                  
constexpr int STEPS = KSPL / BK;        
constexpr int BK8 = 128;
constexpr int STEPS8 = KSPL / BK8;      
constexpr int THREADS = 256;            
constexpr int GRID = NBLK * KS;         
constexpr int WIDTH = 16384, SH = 1536, PACKED = WIDTH + SH;
constexpr int XCH = KSPL / 8;           
constexpr int PCH = XCH / NBLK;         
constexpr int RCH = SH / 8;             
constexpr int WSTAGE = NB * BK * 2;     
constexpr int POLLB = 12;               
constexpr int PREFETCH = 7;
constexpr int TMEM_COLS = 256;          
constexpr int PROW = NB + 4;            
template <int MT, bool UMMA> struct Tile {
  static constexpr int ROWS = UMMA ? 64 : MT * 16;
  static constexpr int STAGES = ROWS == 16 ? 8 : ROWS == 32 ? 6 : ROWS == 48 ? 5 : 4;
  static constexpr int PRE = PREFETCH < STAGES - 1 ? PREFETCH : STAGES - 1;
  static constexpr size_t SMEM = size_t(ROWS) * KSPL * 2 + size_t(STAGES) * WSTAGE;
};
constexpr uint32_t IDESC = (1u << 4) | (1u << 7) | (1u << 10) | (uint32_t(NB >> 3) << 17) | (uint32_t(64 >> 4) << 24);
}   

__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void cp_wait_n(int n) {
  switch (n) {
    case 0: cp_wait<0>(); break;
    case 1: cp_wait<1>(); break;
    case 2: cp_wait<2>(); break;
    case 3: cp_wait<3>(); break;
    case 4: cp_wait<4>(); break;
    case 5: cp_wait<5>(); break;
    case 6: cp_wait<6>(); break;
    default: cp_wait<7>(); break;
  }
}
__device__ __forceinline__ void ldm_x4(uint32_t addr, uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
}
__device__ __forceinline__ void ldm_x2(uint32_t addr, uint32_t& b0, uint32_t& b1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16816_f16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t ld_e4m3x2_f16x2(uint32_t addr) {
  uint32_t out;
  asm volatile("{\n.reg .b16 t;\nld.shared.b16 t, [%1];\ncvt.rn.f16x2.e4m3x2 %0, t;\n}" : "=r"(out) : "r"(addr));
  return out;
}
__device__ __forceinline__ void st_shared(uint32_t addr, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void red_v2(float* p, float a, float b) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(p), "f"(a), "f"(b) : "memory");
}
__device__ __forceinline__ void red_v4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
template <int ROWS>
__device__ __forceinline__ uint32_t x_addr(uint32_t base, int r, int c) {
  return base + uint32_t(c >> 3) * (ROWS * 128) + uint32_t(r) * 128 + uint32_t((c & 7) ^ (r & 7)) * 16;
}
__device__ __forceinline__ uint32_t w_addr(uint32_t base, int n, int c) {
  return base + uint32_t(n) * (fg::BK * 2) + uint32_t(c ^ (n & 7)) * 16;
}
__device__ __forceinline__ void reduce_partial(uint32_t tile, float* __restrict__ acc, int gathered, int n0) {
  constexpr int Q = fg::NB / 4;
  for (int q = threadIdx.x; q < gathered * Q; q += fg::THREADS) {
    const int r = q / Q, c = (q % Q) * 4;
    float a, b, cc, d;
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(a), "=f"(b), "=f"(cc), "=f"(d)
                 : "r"(tile + uint32_t(r * fg::PROW + c) * 4) : "memory");
    red_v4(acc + size_t(r) * fg::SH + n0 + c, a, b, cc, d);
  }
}
__device__ __forceinline__ size_t fg_off(int slot, int src, int row, int col) {
  return HDR + ((size_t(slot * W + src) * fg::RM + row) * fg::PACKED + col) * sizeof(bf16);
}

__device__ __forceinline__ uint64_t umma_desc(uint32_t saddr) {
  return uint64_t((saddr & 0x3FFFFu) >> 4) | (uint64_t(1) << 16) | (uint64_t(1024 >> 4) << 32) | (uint64_t(1) << 46)
         | (uint64_t(2) << 61);
}
__device__ __forceinline__ void umma_f16(uint32_t tmem_d, uint64_t a, uint64_t b, uint32_t accumulate) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
               :: "r"(tmem_d), "l"(a), "l"(b), "r"(fg::IDESC), "r"(accumulate) : "memory");
}
__device__ __forceinline__ void umma_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void tc_before_sync() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_after_sync() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, float (&v)[16]) {
  uint32_t r[16];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = __uint_as_float(r[i]);
}

template <int MT, bool UMMA, bool W8 = false>
__global__ void __launch_bounds__(fg::THREADS, 1)
scatter_gemm_kernel(const bf16* __restrict__ x, const bf16* __restrict__ res, bf16* __restrict__ gr,
                    const bf16* __restrict__ shard, float* __restrict__ acc, const int64_t* __restrict__ maps,
                    int rank, int rows, int64_t x_stride, int64_t r_stride, int pdl,
                    const uint8_t* __restrict__ shard8, const float* __restrict__ scale8) {
  static_assert(!W8 || (MT == 1 && !UMMA), "the e4m3 shard serves the one-tile mma.sync path only");
  using T = fg::Tile<MT, UMMA>;
  constexpr int ROWS = T::ROWS, STAGES = T::STAGES, PRE = T::PRE;
  extern __shared__ __align__(1024) char fg_smem[];
  const uint32_t sx = smem_u32(fg_smem);
  const uint32_t sw = sx + uint32_t(ROWS) * fg::KSPL * 2;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nblk = blockIdx.x % fg::NBLK, ks = blockIdx.x / fg::NBLK;
  const int k0 = ks * fg::KSPL, n0 = nblk * fg::NB;
  const bf16* wbase = shard + size_t(n0) * fg::WIDTH + k0;
  auto load_stage = [&](int step) {
    const uint32_t dst = sw + uint32_t(step % STAGES) * fg::WSTAGE;
#pragma unroll
    for (int i = 0; i < (fg::NB * 8) / fg::THREADS; ++i) {
      const int q = tid + i * fg::THREADS;
      cp16(w_addr(dst, q >> 3, q & 7), wbase + size_t(q >> 3) * fg::WIDTH + step * fg::BK + (q & 7) * 8);
    }
  };
  const uint8_t* wbase8 = W8 ? shard8 + size_t(n0) * fg::WIDTH + k0 : nullptr;
  auto load_stage8 = [&](int step) {    
    const uint32_t dst = sw + uint32_t(step % STAGES) * fg::WSTAGE;
#pragma unroll
    for (int i = 0; i < (fg::NB * 8) / fg::THREADS; ++i) {
      const int q = tid + i * fg::THREADS;
      cp16(w_addr(dst, q >> 3, q & 7), wbase8 + size_t(q >> 3) * fg::WIDTH + step * fg::BK8 + (q & 7) * 16);
    }
  };
  __shared__ uint64_t epoch_s;
  char* own = ring_of(maps, rank);
  uint64_t* counter = reinterpret_cast<uint64_t*>(own) + blockIdx.x;
  uint64_t prev = 0;
  if (tid == 0) prev = *counter;
  if constexpr (W8) {
    static_assert(STAGES == fg::STEPS8, "the e4m3 K-slice must fill the stages exactly");
#pragma unroll
    for (int st = 0; st < STAGES; ++st) { load_stage8(st); cp_commit(); }
  } else {
#pragma unroll
    for (int st = 0; st < PRE; ++st) { load_stage(st); cp_commit(); }
  }
  if (!W8 && tid < fg::NB) {
    constexpr uint32_t REST = uint32_t(fg::KSPL - PRE * fg::BK) * 2;
    if (REST)
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                   :: "l"(wbase + size_t(tid) * fg::WIDTH + PRE * fg::BK), "r"(REST) : "memory");
  }
  if (tid == 0) { epoch_s = prev + 1; *counter = prev + 1; }
  __syncthreads();
  const uint64_t epoch = epoch_s;
  const int slot = int(epoch % SLOTS);
  if (pdl) pdl_wait();
  {
    const uint4 empty = make_uint4(EMPTY, EMPTY, EMPTY, EMPTY);
    const int before = int((epoch + SLOTS - 1) % SLOTS);
    for (int q = tid; q < (W - 1) * fg::RM * fg::PCH; q += fg::THREADS) {
      const int t = q / fg::PCH, d = 1 + t / fg::RM, r = t % fg::RM;
      st_sys(own + fg_off(before, (rank + d) & 3, r, k0 + (nblk * fg::PCH + q % fg::PCH) * 8), empty);
    }
  }
  const int gathered = W * rows;
  const int g = blockIdx.x * fg::THREADS + tid;

  if constexpr (W8) {
    for (int q = tid; q < rows * fg::XCH; q += fg::THREADS)
      st_shared(x_addr<ROWS>(sx, rank * rows + q / fg::XCH, q % fg::XCH),
                ld_cg(x + (q / fg::XCH) * x_stride + k0 + (q % fg::XCH) * 8));
  } else {
    for (int q = tid; q < rows * fg::XCH; q += fg::THREADS)
      cp16(x_addr<ROWS>(sx, rank * rows + q / fg::XCH, q % fg::XCH), x + (q / fg::XCH) * x_stride + k0 + (q % fg::XCH) * 8);
    cp_commit();
  }
  if (tid < rows * fg::PCH) {
    const int r = tid / fg::PCH, c = nblk * fg::PCH + tid % fg::PCH;
    const uint4 v = scrub(ld_cg(x + r * x_stride + k0 + c * 8));
    const size_t off = fg_off(slot, rank, r, k0 + c * 8);
#pragma unroll
    for (int d = 1; d < W; ++d) st_sys(ring_of(maps, (rank + d) & 3) + off, v);
  }
  if (g < W * rows * fg::RCH) {
    const int p = g / (rows * fg::RCH), rem = g % (rows * fg::RCH);
    const int r = rem / fg::RCH, c = rem % fg::RCH;
    const uint4 v = scrub(ld_cg(res + r * r_stride + p * fg::SH + c * 8));
    if (p == rank) *reinterpret_cast<uint4*>(gr + (size_t(rank) * rows + r) * fg::SH + c * 8) = v;
    else st_sys(ring_of(maps, p) + fg_off(slot, rank, r, fg::WIDTH + c * 8), v);
  }
  {
    const uint4 empty = make_uint4(EMPTY, EMPTY, EMPTY, EMPTY);
    const bool has_res = g < (W - 1) * rows * fg::RCH;
    char* ra = own;
    bf16* rdst = gr;
    uint4 rv = empty;
    if (has_res) {
      const int t = g / fg::RCH, c = g % fg::RCH;
      const int d = 1 + t / rows, r = t % rows, src = (rank + d) & 3;
      ra = own + fg_off(slot, src, r, fg::WIDTH + c * 8);
      rdst = gr + (size_t(src) * rows + r) * fg::SH + c * 8;
      rv = ld_sys(ra);
    }
    const int c = tid % fg::XCH;
    constexpr int PAIRS = fg::THREADS / fg::XCH;                      
    for (int t0 = tid / fg::XCH; t0 < (W - 1) * rows; t0 += PAIRS * fg::POLLB) {
      const char* pa[fg::POLLB];
      uint32_t ps[fg::POLLB];
      uint4 pv[fg::POLLB];
#pragma unroll
      for (int j = 0; j < fg::POLLB; ++j) {
        const int t = t0 + j * PAIRS;                                  
        if (t < (W - 1) * rows) {
          const int d = 1 + t / rows, r = t % rows, src = (rank + d) & 3;
          pa[j] = own + fg_off(slot, src, r, k0 + c * 8);
          ps[j] = x_addr<ROWS>(sx, src * rows + r, c);
          pv[j] = ld_sys(pa[j]);
        }
      }
      bool pending;
      do {
        pending = false;
#pragma unroll
        for (int j = 0; j < fg::POLLB; ++j)
          if (t0 + j * PAIRS < (W - 1) * rows && !full(pv[j])) { pv[j] = ld_sys(pa[j]); pending = true; }
      } while (pending);
#pragma unroll
      for (int j = 0; j < fg::POLLB; ++j)
        if (t0 + j * PAIRS < (W - 1) * rows) st_shared(ps[j], pv[j]);
    }
    if (has_res) {
      while (!full(rv)) rv = ld_sys(ra);
      *reinterpret_cast<uint4*>(rdst) = rv;
      st_sys(ra, empty);
    }
  }
  if (pdl) pdl_trigger();
  if constexpr (!W8) {
#pragma unroll
    for (int st = PRE; st < STAGES - 1; ++st) { load_stage(st); cp_commit(); }
    cp_wait<STAGES - 1 - PRE>();
  }

  if constexpr (UMMA) {
    __shared__ __align__(8) uint64_t bars[STAGES + 1];
    __shared__ uint32_t tmem_base;
    if (tid == 0) {
      for (int i = 0; i < STAGES + 1; ++i) mbar_init(smem_u32(&bars[i]));
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");    
    }
    if (warp == 0) {
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(smem_u32(&tmem_base)), "r"(fg::TMEM_COLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    proxy_fence();
    tc_before_sync();
    __syncthreads();
    tc_after_sync();
    const uint32_t tmem = tmem_base;
    for (int step = 0; step < fg::STEPS; ++step) {
      cp_wait<STAGES - 2>();
      proxy_fence();
      __syncthreads();
      if (tid == 0) {
        const uint32_t a0 = sx + uint32_t(step) * (ROWS * 128);
        const uint32_t b0 = sw + uint32_t(step % STAGES) * fg::WSTAGE;
#pragma unroll
        for (int kk = 0; kk < fg::BK / 16; ++kk)
          umma_f16(tmem, umma_desc(a0 + kk * 32), umma_desc(b0 + kk * 32), (step | kk) != 0);
        umma_commit(smem_u32(&bars[step % STAGES]));
      }
      const int next = step + STAGES - 1;
      if (next < fg::STEPS) {
        if (step) mbar_wait(smem_u32(&bars[next % STAGES]), uint32_t((step - 1) / STAGES) & 1u);
        load_stage(next);
      }
      cp_commit();
    }
    if (tid == 0) umma_commit(smem_u32(&bars[STAGES]));
    mbar_wait(smem_u32(&bars[STAGES]), 0);
    tc_after_sync();
    const int quad = warp & 3, half = warp >> 2;
    const int row = 16 * quad + lane;
#pragma unroll
    for (int ch = 0; ch < fg::NB / 2 / 16; ++ch) {
      float v[16];
      tmem_ld16(tmem + (uint32_t(32 * quad) << 16) + uint32_t(half * (fg::NB / 2) + ch * 16), v);
      if (lane < 16 && row < gathered) {
        const uint32_t dst = sw + uint32_t(row * fg::PROW + half * (fg::NB / 2) + ch * 16) * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dst + 16 * j), "f"(v[4 * j]), "f"(v[4 * j + 1]),
                       "f"(v[4 * j + 2]), "f"(v[4 * j + 3]) : "memory");
      }
    }
    tc_before_sync();
    __syncthreads();
    if (warp == 0) {
      tc_after_sync();
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem), "r"(fg::TMEM_COLS) : "memory");
    }
    reduce_partial(sw, acc, gathered, n0);
  } else {
    __syncthreads();
    float accum[MT][3][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int t = 0; t < 3; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) accum[m][t][e] = 0.f;
    if constexpr (W8) {
      for (int q = tid; q < ROWS * fg::KSPL / 8; q += fg::THREADS) {
        const uint32_t addr = sx + uint32_t(q) * 16;
        uint32_t w[4];
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]) : "r"(addr));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[j]));
          const __half2 h = __float22half2_rn(f);
          w[j] = *reinterpret_cast<const uint32_t*>(&h);
        }
        st_shared(addr, make_uint4(w[0], w[1], w[2], w[3]));
      }
#pragma unroll 1
      for (int step = 0; step < fg::STEPS8; ++step) {
        cp_wait_n(fg::STEPS8 - 1 - step);    
        __syncthreads();                     
        const uint32_t wst = sw + uint32_t(step) * fg::WSTAGE;
#pragma unroll
        for (int kk = 0; kk < fg::BK8 / 16; ++kk) {
          uint32_t b[3][2];
#pragma unroll
          for (int t = 0; t < 3; ++t) {
            const uint32_t at = w_addr(wst, warp * 24 + t * 8 + (lane >> 2), kk) + (lane & 3) * 2;
            b[t][0] = ld_e4m3x2_f16x2(at);
            b[t][1] = ld_e4m3x2_f16x2(at + 8);
          }
#pragma unroll
          for (int m = 0; m < MT; ++m) {
            uint32_t a0, a1, a2, a3;
            ldm_x4(x_addr<ROWS>(sx, m * 16 + (lane & 15), step * 16 + kk * 2 + (lane >> 4)), a0, a1, a2, a3);
#pragma unroll
            for (int t = 0; t < 3; ++t) mma16816_f16(accum[m][t], a0, a1, a2, a3, b[t][0], b[t][1]);
          }
        }
      }
#pragma unroll
      for (int t = 0; t < 3; ++t) {
        const int col = warp * 24 + t * 8 + (lane & 3) * 2;
        const float s0 = scale8[n0 + col], s1 = scale8[n0 + col + 1];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          accum[m][t][0] *= s0; accum[m][t][1] *= s1; accum[m][t][2] *= s0; accum[m][t][3] *= s1;
        }
      }
    } else {
      for (int step = 0; step < fg::STEPS; ++step) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        const uint32_t wst = sw + uint32_t(step % STAGES) * fg::WSTAGE;
#pragma unroll
        for (int kk = 0; kk < fg::BK / 16; ++kk) {
          uint32_t b[3][2];
#pragma unroll
          for (int t = 0; t < 3; ++t)
            ldm_x2(w_addr(wst, warp * 24 + t * 8 + (lane & 7), kk * 2 + ((lane >> 3) & 1)), b[t][0], b[t][1]);
#pragma unroll
          for (int m = 0; m < MT; ++m) {
            uint32_t a0, a1, a2, a3;
            ldm_x4(x_addr<ROWS>(sx, m * 16 + (lane & 15), step * 8 + kk * 2 + (lane >> 4)), a0, a1, a2, a3);
#pragma unroll
            for (int t = 0; t < 3; ++t) mma16816(accum[m][t], a0, a1, a2, a3, b[t][0], b[t][1]);
          }
        }
        if (step + STAGES - 1 < fg::STEPS) load_stage(step + STAGES - 1);
        cp_commit();
      }
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int t = 0; t < 3; ++t) {
        const int row = m * 16 + (lane >> 2);
        const int col = warp * 24 + t * 8 + (lane & 3) * 2;
        if (row < gathered)
          asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(sw + uint32_t(row * fg::PROW + col) * 4),
                       "f"(accum[m][t][0]), "f"(accum[m][t][1]) : "memory");
        if (row + 8 < gathered)
          asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(sw + uint32_t((row + 8) * fg::PROW + col) * 4),
                       "f"(accum[m][t][2]), "f"(accum[m][t][3]) : "memory");
      }
    __syncthreads();
    reduce_partial(sw, acc, gathered, n0);
  }
}


int64_t fused_ring_bytes() { return HDR + int64_t(SLOTS) * W * fg::RM * fg::PACKED * 2; }

void scatter_gemm(torch::Tensor x, torch::Tensor res, torch::Tensor gr, torch::Tensor shard, torch::Tensor acc,
                  torch::Tensor maps, int64_t rank, bool pdl, bool umma, torch::Tensor shard8, torch::Tensor scale8) {
  check_mat(x); check_mat(res); check_mat(gr); check_maps(maps, rank, x);
  const int rows = x.size(0);
  TORCH_CHECK(rows > 0 && rows <= fg::RM && x.size(1) == fg::WIDTH && res.size(0) == rows && res.size(1) == W * fg::SH,
              "fused projection geometry");
  TORCH_CHECK(gr.is_contiguous() && gr.size(0) == W * rows && gr.size(1) == fg::SH, "gathered residual geometry");
  TORCH_CHECK(shard.is_cuda() && shard.is_contiguous() && shard.scalar_type() == at::kBFloat16
              && shard.size(0) == fg::SH && shard.size(1) == fg::WIDTH
              && reinterpret_cast<uintptr_t>(shard.data_ptr()) % 16 == 0, "shard");
  TORCH_CHECK(acc.is_cuda() && acc.is_contiguous() && acc.scalar_type() == at::kFloat && acc.size(1) == fg::SH
              && acc.size(0) >= W * rows, "accumulator");
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const bf16* xp = static_cast<const bf16*>(x.data_ptr()); const bf16* rp = static_cast<const bf16*>(res.data_ptr());
  bf16* grp = static_cast<bf16*>(gr.data_ptr()); const bf16* sp = static_cast<const bf16*>(shard.data_ptr());
  float* ap = acc.data_ptr<float>(); const int64_t* mp = maps.data_ptr<int64_t>();
  int rk = int(rank), rw = rows, pd = pdl ? 1 : 0;
  int64_t xs = x.stride(0), rs = res.stride(0);
  const bool w8 = shard8.numel() > 0;
  if (w8)
    TORCH_CHECK(shard8.is_cuda() && shard8.is_contiguous() && shard8.element_size() == 1 && shard8.size(0) == fg::SH
                && shard8.size(1) == fg::WIDTH && reinterpret_cast<uintptr_t>(shard8.data_ptr()) % 16 == 0
                && scale8.is_cuda() && scale8.is_contiguous() && scale8.scalar_type() == at::kFloat
                && scale8.numel() == fg::SH, "e4m3 shard");
  const uint8_t* s8p = w8 ? static_cast<const uint8_t*>(shard8.data_ptr()) : nullptr;
  const float* c8p = w8 ? scale8.data_ptr<float>() : nullptr;
  void* args[] = {&xp, &rp, &grp, &sp, &ap, &mp, &rk, &rw, &xs, &rs, &pd, &s8p, &c8p};
#define FG_LAUNCH(MT_, U_, W8_)                                                                                \
  {                                                                                                            \
    using T = fg::Tile<MT_, U_>;                                                                               \
    const cudaError_t a = cudaFuncSetAttribute(reinterpret_cast<const void*>(scatter_gemm_kernel<MT_, U_, W8_>),\
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, int(T::SMEM));     \
    TORCH_CHECK(a == cudaSuccess, "fused projection shared memory: ", cudaGetErrorString(a));                   \
    launch_pdl(pdl, 1, stream, dim3(fg::GRID), dim3(fg::THREADS), scatter_gemm_kernel<MT_, U_, W8_>, args,     \
               T::SMEM);                                                                                       \
  }
  const int mt = (W * rows + 15) / 16;
  if (umma) FG_LAUNCH(4, true, false)
  else if (mt == 1 && w8) FG_LAUNCH(1, false, true)
  else if (mt == 1) FG_LAUNCH(1, false, false)
  else if (mt == 2) FG_LAUNCH(2, false, false)
  else if (mt == 3) FG_LAUNCH(3, false, false)
  else FG_LAUNCH(4, false, false)
#undef FG_LAUNCH
}

}   
