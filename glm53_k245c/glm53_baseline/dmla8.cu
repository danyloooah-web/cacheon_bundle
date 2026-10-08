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

namespace dmla8 {

constexpr int H = 64, DQK = 576, DV = 512, TOPK = 2048, TK = 128, SPLITS = TOPK / TK;
constexpr int NCB = 5;
constexpr int KBLK = TK * 128;
constexpr int KTILE = NCB * KBLK;          
constexpr int QBLK = H * 128;
constexpr int QBYTES = NCB * QBLK;         
constexpr int PTILE = TK * H;              
constexpr int NSM = 8;                     
constexpr int WCOR = 8;                    
constexpr int WMMA = 12, WQ = 13;          
constexpr int WPROD = 14;                  
#ifndef DMLA8_NPROD
#define DMLA8_NPROD 6
#endif
constexpr int NPROD = DMLA8_NPROD;         
constexpr int NGROUPS = 32;
constexpr int NWARPS = WPROD + NPROD;
constexpr int THREADS = NWARPS * 32;
constexpr float PSCALE = 448.f, LN2 = 0.6931471805599453f, LOG2_PSCALE = 8.807354922057604f;
constexpr int MAXT = 64;
constexpr int MAXPER = 8;                  
constexpr int DSLICE = DV * H * 2;         
constexpr int OFF_LSE_ST = 36864;          
constexpr unsigned FL_PERIOD = 256;        

constexpr float REF_DROP_LOG2 = 24.f;       
constexpr int OFF_K = 0;
constexpr int OFF_Q = OFF_K + 2 * KTILE;
constexpr int OFF_P = OFF_Q + QBYTES;
constexpr int OFF_RED = OFF_P + 2 * PTILE;                 
constexpr int OFF_IDX = OFF_RED + (4 * H + 4 * H) * 4;     
constexpr int OFF_BAR = OFF_IDX + 2 * TK * 4;
constexpr int OFF_TMEM = OFF_BAR + 48 * 8;
constexpr int SMEM = OFF_TMEM + 16 + 1024;
static_assert(SMEM <= 232448, "smem");

constexpr uint32_t TM_S = 256;

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
#ifdef DMLA8_HANGDBG
#ifndef DBGCTA
#define DBGCTA 3
#endif
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  long long n = 0;
  while (true) {
    uint32_t ok;
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    if (ok) return;
    if (++n == (1ll << 22)) {
      printf("HANG cta %d warp %d lane %d bar_off %u parity %u\n", int(blockIdx.x), int(threadIdx.x >> 5),
             int(threadIdx.x & 31), bar & 0x3ff, parity);
    }
  }
}
#else
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}"
               :: "r"(bar), "r"(parity) : "memory");
}
#endif
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
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
__device__ __forceinline__ void tma_prefetch_gather4(const CUtensorMap* tm, int col, int4 rows) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4 [%0, {%1, %2, %3, %4, %5}];"
               :: "l"(tm), "r"(col), "r"(rows.x), "r"(rows.y), "r"(rows.z), "r"(rows.w) : "memory");
}
__device__ __forceinline__ void umma(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}"
               :: "r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(acc) : "memory");
}
__device__ __forceinline__ void umma_bs(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc,
                                        uint32_t sfa, uint32_t sfb) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n}"
               :: "r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(acc), "r"(sfa), "r"(sfb) : "memory");
}
__device__ __forceinline__ void utccp_32x128b(uint32_t tmem_dst, uint64_t sdesc_src) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(tmem_dst), "l"(sdesc_src) : "memory");
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
__device__ __forceinline__ void tmem_ld32_nowait(uint32_t taddr, float (&v)[32]) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
        "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
        "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
        "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
}
__device__ __forceinline__ void tmem_st32(uint32_t taddr, const float (&v)[32]) {
  const uint32_t* r = reinterpret_cast<const uint32_t*>(v);
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,"
      "%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
      :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
        "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]),
        "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]),
        "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31])
      : "memory");
}
__device__ __forceinline__ void tmem_ld_16x256b_x4(uint32_t taddr, float (&v)[16]) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tmem_st_16x256b_x4(uint32_t taddr, const float (&v)[16]) {
  const uint32_t* r = reinterpret_cast<const uint32_t*>(v);
  asm volatile("tcgen05.st.sync.aligned.16x256b.x4.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                  "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
               : "memory");
}
__device__ __forceinline__ float ex2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ uint32_t mapa_u32(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
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

__host__ __device__ __forceinline__ int first_cta(int t, int per) { return (SPLITS * t) / per; }
__host__ __device__ __forceinline__ int n_parts(int t, int per) {
  return (SPLITS * t + SPLITS - 1) / per - (SPLITS * t) / per + 1;
}
__host__ __device__ __forceinline__ int max_parts(int per) { return (SPLITS + per - 1) / per + 1; }

#define ST6T(n) do { if (stamps != nullptr && blockIdx.x == 0 && (n) < 64) { unsigned long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t)); stamps[n] = _t; } } while (0)
#define ST6(n) do { if (stamps != nullptr && blockIdx.x == 0 && threadIdx.x == 0 && (n) < 64) { unsigned long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t)); stamps[n] = _t; } } while (0)

template <int ABL, int G>
__global__ void __launch_bounds__(THREADS, 1) dmla8_kernel(
    const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_kv,
    const int32_t* __restrict__ pt, int64_t pt_stride, const int32_t* __restrict__ lens, int n_tiles, int per,
    float scale_log2, __nv_bfloat16* __restrict__ parts, float* __restrict__ lse_out, int pdl,
    unsigned long long* __restrict__ stamps, __nv_bfloat16* __restrict__ out, const uint8_t* __restrict__ kvbase,
    unsigned int* __restrict__ counter) {
  constexpr int GA = G < 0 ? -G : G;               
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);    
  float* s_red = reinterpret_cast<float*>(smem + OFF_RED);
  float* s_m = s_red + 4 * H;                       
  float* s_alpha = s_m + H;                         
  float* s_inv = s_alpha + H;                       
  float* s_alpha1 = s_inv + H;                      
  int32_t* s_idx = reinterpret_cast<int32_t*>(smem + OFF_IDX);
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + OFF_BAR);
  uint32_t* s_tmem = reinterpret_cast<uint32_t*>(smem + OFF_TMEM);
#define B7_K(b, c) smem_u32(&bars[5 * (b) + (c)])
#define B7_Q smem_u32(&bars[10])
#define B7_S(b) smem_u32(&bars[11 + (b)])
#define B7_P(b) smem_u32(&bars[13 + (b)])
#define B7_OD(b) smem_u32(&bars[15 + (b)])               
#define B7_OFREE smem_u32(&bars[19])
#define B7_KFREE(b, c) smem_u32(&bars[32 + 5 * (b) + (c)])    
#define B7_IDX(b) smem_u32(&bars[22 + (b)])
#define B8_ALPHA(b) smem_u32(&bars[24 + (b)])
#define B8_CORR(b) smem_u32(&bars[30 + (b)])           
#define B8_SFREE(b) smem_u32(&bars[27 + (b)])            
#define B8_RECV smem_u32(&bars[29])                      
  const uint32_t k_base = smem_u32(smem + OFF_K), q_base = smem_u32(smem + OFF_Q), p_base = smem_u32(smem + OFF_P);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int first = int(blockIdx.x) * per;
  const int my = max(0, min(n_tiles, first + per) - first);
  const int t0 = first / SPLITS;
  auto seg_first = [&](int k) { return k == 0 || ((first + k) % SPLITS) == 0; };
  auto seg_last = [&](int k) { return k == my - 1 || ((first + k) % SPLITS) == SPLITS - 1; };

  if (tid == 0) {
    ST6(0);
    for (int b = 0; b < 2; ++b) {
      for (int c = 0; c < NCB; ++c) mbar_init(B7_K(b, c), NGROUPS);
      mbar_init(B7_S(b), 1);
      mbar_init(B7_P(b), NSM * 32);
      for (int c = 0; c < NCB; ++c) mbar_init(B7_KFREE(b, c), 1);
      mbar_init(B7_IDX(b), NGROUPS);
    }
    mbar_init(B7_Q, 1);
    mbar_init(B7_OD(0), 1);
    mbar_init(B7_OD(1), 1);
    mbar_init(B7_OFREE, NSM * 32);
    mbar_init(B8_ALPHA(0), 1);
    mbar_init(B8_ALPHA(1), 1);
    mbar_init(B8_CORR(0), 4 * 32);
    mbar_init(B8_CORR(1), 4 * 32);
    mbar_init(B8_SFREE(0), NSM * 32);
    mbar_init(B8_SFREE(1), NSM * 32);
    mbar_init(B8_RECV, 1);
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
  unsigned int fl_base = 0;
  if constexpr (G < 0) {
    if (tid == 0) {
      unsigned int v;
      asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(counter + first / SPLITS) : "memory");
      fl_base = v & ~unsigned(FL_PERIOD - 1);
    }
  }

  if (warp == WMMA) {
    if (lane == 0) {
      for (int k = 0; k < my; ++k) {
        const int b = k & 1;
        if (seg_first(k)) {
          if (k > 0) mbar_wait(B7_S((k - 1) & 1), ((k - 1) >> 1) & 1);
          const int t = (first + k) / SPLITS;
          mbar_expect_tx(B7_Q, QBYTES);
#pragma unroll
          for (int c = 0; c < NCB; ++c) tma_load_2d(q_base + c * QBLK, &tm_q, c * 128, t * H, B7_Q);
          mbar_wait(B7_Q, (t - t0) & 1);
        }
        if (k >= 2) mbar_wait(B8_SFREE(b), ((k - 2) >> 1) & 1);
#pragma unroll
        for (int j = 0; j < DQK / 32; ++j) {
          const int kk = (j + 16) % 18;                     
          const int c = kk >> 2, off = (kk & 3) * 32;
          if (j == 0 || (kk & 3) == 0) {
            mbar_wait(B7_K(b, c), (k >> 1) & 1);
            tc_fence_after();
            if (c == 0 && k < 8) ST6T(48 + k);
          }
          umma(tmem + TM_S + 64 * b, sdesc(k_base + b * KTILE + c * KBLK + off, 16, 1024, 2),
               sdesc(q_base + c * QBLK + off, 16, 1024, 2), IDESC_QK, j > 0);
        }
        umma_commit(B7_S(b));
        umma_commit(B7_KFREE(b, 4));                        
        if (k < 4) ST6T(56 + k);
      }
    }
    __syncwarp();
  } else if (warp == WQ) {
    if (lane == 0) {
      for (int k = 0; k < my; ++k) {
        const int b = k & 1, si = (first + k) / SPLITS - t0;
        const bool sf = seg_first(k);
        mbar_wait(B7_P(b), (k >> 1) & 1);
#ifdef DMLA8_HANGDBG
        if (blockIdx.x == DBGCTA) printf("pv cta5 k %d saw P\n", k);
#endif
        if (k < 2) ST6T(60 + k);
        mbar_wait(B8_CORR(b), (k >> 1) & 1);                 
        if (sf && si > 0) mbar_wait(B7_OFREE, (si - 1) & 1);
        tc_fence_after();
#pragma unroll
        for (int cb = 0; cb < DV / 128; ++cb) {
#pragma unroll
          for (int ks = 0; ks < TK / 32; ++ks)
            umma(tmem + 64 * cb, sdesc(k_base + b * KTILE + cb * KBLK + ks * 32 * 128, KBLK, 1024, 2),
                 sdesc(p_base + b * PTILE + ks * 32 * 64, PTILE, 512, 4), IDESC_PV, !(sf && ks == 0));
          umma_commit(B7_KFREE(b, cb));                     
        }
        umma_commit(B7_OD(b));
        if (k < 4) ST6T(36 + k);
#ifdef DMLA8_DIAG
        if (k < 4) { mbar_wait(B7_OD(b), (k >> 1) & 1); ST6T(44 + k); }
#endif
      }
    }
    __syncwarp();
  } else if (warp >= WCOR && warp < WCOR + 4) {
    const int qq = warp - WCOR;
    const uint32_t lane_base = uint32_t(32 * qq) << 16;
    for (int k = 0; k < my; ++k) {
      mbar_wait(B8_ALPHA(k & 1), (k >> 1) & 1);
      if (!seg_first(k) && ABL != 3) {
        mbar_wait(B7_OD((k - 1) & 1), ((k - 1) >> 1) & 1);
        tc_fence_after();
        if (qq == 0 && lane == 0) ST6T(32 + k);
#pragma unroll
        for (int cb = 0; cb < DV / 128; ++cb) {
          float v0[32], v1[32];
          tmem_ld32_nowait(tmem + lane_base + 64 * cb, v0);
          tmem_ld32_nowait(tmem + lane_base + 64 * cb + 32, v1);
          tmem_wait_ld();
#pragma unroll
          for (int i = 0; i < 32; i += 4) {
            const float* al_k = (k & 1) ? s_alpha1 : s_alpha;
            const float4 a0 = *reinterpret_cast<const float4*>(&al_k[i]);
            const float4 a1 = *reinterpret_cast<const float4*>(&al_k[32 + i]);
            v0[i] *= a0.x; v0[i + 1] *= a0.y; v0[i + 2] *= a0.z; v0[i + 3] *= a0.w;
            v1[i] *= a1.x; v1[i + 1] *= a1.y; v1[i + 2] *= a1.z; v1[i + 3] *= a1.w;
          }
          tmem_st32(tmem + lane_base + 64 * cb, v0);
          tmem_st32(tmem + lane_base + 64 * cb + 32, v1);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        tc_fence_before();
      }
      mbar_arrive(B8_CORR(k & 1));
#ifdef DMLA8_HANGDBG
      if (blockIdx.x == DBGCTA && lane == 0) printf("corr cta5 warp %d k %d arrived\n", warp, k);
#endif
      if (qq == 0 && lane == 0 && k < 4) ST6T(52 + k);
    }
  } else if (warp >= WPROD) {
    const int p = warp - WPROD;
    if (p + NPROD * lane < NGROUPS) {
      const int r0 = 4 * (p + NPROD * lane);
      int4 raw[MAXPER];
#pragma unroll
      for (int k = 0; k < MAXPER; ++k) {
        const int it = first + k;
        raw[k] = k < my ? __ldg(reinterpret_cast<const int4*>(pt + size_t(it / SPLITS) * pt_stride + (it % SPLITS) * TK + r0))
                        : make_int4(-1, -1, -1, -1);
      }
      const int nv0 = min(__ldg(lens + t0), TOPK);
      const int nv1 = (first + my - 1) / SPLITS > t0 ? min(__ldg(lens + t0 + 1), TOPK) : 0;
#pragma unroll
      for (int k = 0; k < MAXPER; ++k) {
        if (k >= my) break;
        if (k == 1) continue;                             
        const int b = k & 1, it = first + k, t = it / SPLITS, s = it % SPLITS;
        const int nvalid = t == t0 ? nv0 : nv1;
        int v[4] = {raw[k].x, raw[k].y, raw[k].z, raw[k].w};
#pragma unroll
        for (int e = 0; e < 4; ++e)
          if (s * TK + r0 + e >= nvalid || v[e] < 0) v[e] = -1;
        const int4 rows = make_int4(v[0], v[1], v[2], v[3]);
        const uint32_t ph = ((k - 2) >> 1) & 1;
        if (k >= 2) mbar_wait(B7_KFREE(b, 4), ph);
        mbar_expect_tx(B7_K(b, 4), 4 * 128);
        tma_gather4(k_base + b * KTILE + 4 * KBLK + r0 * 128, &tm_kv, 4 * 128, rows, B7_K(b, 4));
        if (k >= 2) mbar_wait(B7_KFREE(b, 0), ph);           
        if (p == 0 && lane == 0 && k < 8) ST6T(40 + k);
#pragma unroll
        for (int e = 0; e < 4; ++e) s_idx[b * TK + r0 + e] = v[e];
        mbar_arrive(B7_IDX(b));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          if (k >= 2 && c > 0) mbar_wait(B7_KFREE(b, c), ph);
          mbar_expect_tx(B7_K(b, c), 4 * 128);
          tma_gather4(k_base + b * KTILE + c * KBLK + r0 * 128, &tm_kv, c * 128, rows, B7_K(b, c));
        }
      }
    }
  } else {
    const int q4 = warp & 3, hh = warp >> 2;
    const uint32_t lane_base = uint32_t(32 * q4) << 16;
    const float c = scale_log2;
    const int tq = lane & 3, tk = lane >> 2;
    const int hsel = tk & 7;                                    
    const int head_rel = 8 * (hsel >> 1) + 2 * tq + (hsel & 1);
    if (my > 1) {
      if (lane < 4) {
        const int r0 = 4 * (4 * warp + lane), it = first + 1, t = it / SPLITS, s = it % SPLITS;
        const int4 raw = __ldg(reinterpret_cast<const int4*>(pt + size_t(t) * pt_stride + s * TK + r0));
        const int nvalid = min(__ldg(lens + t), TOPK);
        int v[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int e = 0; e < 4; ++e)
          if (s * TK + r0 + e >= nvalid || v[e] < 0) v[e] = -1;
#pragma unroll
        for (int e = 0; e < 4; ++e) s_idx[TK + r0 + e] = v[e];
        mbar_arrive(B7_IDX(1));
        const int4 rows = make_int4(v[0], v[1], v[2], v[3]);
#pragma unroll
        for (int c = 0; c < NCB; ++c) {
          mbar_expect_tx(B7_K(1, c), 4 * 128);
          tma_gather4(k_base + KTILE + c * KBLK + r0 * 128, &tm_kv, c * 128, rows, B7_K(1, c));
        }
      }
#if DMLA8_PREFETCH
      for (int x = tid; x < (my - 2) * TK; x += NSM * 32) {
        const int it = first + 2 + x / TK, row = x % TK;
        const int slot = __ldg(pt + size_t(it / SPLITS) * pt_stride + (it % SPLITS) * TK + row);
        if (slot >= 0) {
          const char* a = reinterpret_cast<const char*>(kvbase) + size_t(slot) * DQK;
#pragma unroll
          for (int c = 0; c < 5; ++c) asm volatile("prefetch.global.L2 [%0];" :: "l"(a + 128 * c));
        }
      }
#endif
    }
    float l8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) l8[i] = 0.f;
    for (int k = 0; k < my; ++k) {
      const int b = k & 1, it = first + k, t = it / SPLITS;
      const bool sfirst = seg_first(k), slast = seg_last(k);
#ifdef DMLA8_HANGDBG
      if (blockIdx.x == DBGCTA && lane == 0) printf("sm cta5 warp %d k %d start\n", warp, k);
#endif
      mbar_wait(B7_IDX(b), (k >> 1) & 1);
      mbar_wait(B7_S(b), (k >> 1) & 1);
      tc_fence_after();
      ST6(1 + 8 * k);
      bool valid[4];
#pragma unroll
      for (int kx = 0; kx < 4; ++kx) valid[kx] = s_idx[b * TK + 32 * q4 + tk + 8 * kx] >= 0;
      float a0[16], a1[16];
      tmem_ld_16x256b_x4(tmem + lane_base + TM_S + 64 * b + 32 * hh, a0);
      tmem_ld_16x256b_x4(tmem + lane_base + (16u << 16) + TM_S + 64 * b + 32 * hh, a1);
      tmem_wait_ld();
      tc_fence_before();
      mbar_arrive(B8_SFREE(b));
#define SV(kx, h) ((kx) < 2 ? a0[4 * ((h) >> 1) + 2 * (kx) + ((h) & 1)] : a1[4 * ((h) >> 1) + 2 * ((kx) - 2) + ((h) & 1)])
      {
        float mx[8];
#pragma unroll
        for (int h = 0; h < 8; ++h) {
          float x = -INFINITY;
#pragma unroll
          for (int kx = 0; kx < 4; ++kx) x = valid[kx] ? fmaxf(x, SV(kx, h)) : x;
          mx[h] = x;
        }
#pragma unroll
        for (int o = 16, n = 8; o >= 4; o >>= 1, n >>= 1) {
          const bool up = (lane & o) != 0;
#pragma unroll
          for (int i = 0; i < n / 2; ++i) {
            const float send = up ? mx[i] : mx[i + n / 2];
            const float keep = up ? mx[i + n / 2] : mx[i];
            mx[i] = fmaxf(keep, __shfl_xor_sync(0xffffffffu, send, o));
          }
        }
        s_red[q4 * H + 32 * hh + head_rel] = mx[0];
      }
      sm_sync();
      if (q4 == 0) {
        const int h = 32 * hh + lane;
        const float mt = fmaxf(fmaxf(s_red[h], s_red[H + h]), fmaxf(s_red[2 * H + h], s_red[3 * H + h])) * c;
        const float mp = sfirst ? -INFINITY : s_m[h];
        const float mr = (mt == -INFINITY || mp == -INFINITY) ? mt : fmaxf(mt, mp - REF_DROP_LOG2);
        s_m[h] = mt == -INFINITY ? mp : mr;
        (b ? s_alpha1 : s_alpha)[h] = (mp == -INFINITY || mt == -INFINITY) ? 1.f : ex2(mp - mr);
      }
      sm_sync();
      if (tid == 0) mbar_arrive(B8_ALPHA(b));                 
      ST6(2 + 8 * k);
      float mq[8], al[8];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const float2 m2 = *reinterpret_cast<const float2*>(&s_m[32 * hh + 8 * r + 2 * tq]);
        const float2 a2 = *reinterpret_cast<const float2*>(&(b ? s_alpha1 : s_alpha)[32 * hh + 8 * r + 2 * tq]);
        mq[2 * r] = m2.x - LOG2_PSCALE; mq[2 * r + 1] = m2.y - LOG2_PSCALE;
        al[2 * r] = a2.x; al[2 * r + 1] = a2.y;
      }
#pragma unroll
      for (int h = 0; h < 8; ++h) l8[h] *= al[h];
      uint8_t* pt_s = smem + OFF_P + b * PTILE;
#pragma unroll
      for (int kx = 0; kx < 4; ++kx) {
        const int key = 32 * q4 + tk + 8 * kx;
        const int sw = (key >> 1) & 3;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          float p0, p1;
          if constexpr (ABL == 1) {
            p0 = SV(kx, 2 * r); p1 = SV(kx, 2 * r + 1);
          } else {
            p0 = valid[kx] ? ex2(fmaf(SV(kx, 2 * r), c, -mq[2 * r])) : 0.f;
            p1 = valid[kx] ? ex2(fmaf(SV(kx, 2 * r + 1), c, -mq[2 * r + 1])) : 0.f;
          }
          l8[2 * r] += p0;
          l8[2 * r + 1] += p1;
          const uint16_t q2 = static_cast<uint16_t>(
              __nv_cvt_float2_to_fp8x2(make_float2(p0, p1), __NV_SATFINITE, __NV_E4M3));
          const int hb = 32 * hh + 8 * r + 2 * tq;               
          *reinterpret_cast<uint16_t*>(pt_s + key * 64 + 16 * ((hb >> 4) ^ sw) + (hb & 15)) = q2;
        }
      }
#undef SV
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      ST6(3 + 8 * k);
      mbar_arrive(B7_P(b));
#ifdef DMLA8_HANGDBG
      if (blockIdx.x == DBGCTA && lane == 0) printf("sm cta5 warp %d k %d arrived P\n", warp, k);
#endif
      ST6(5 + 8 * k);
      if (!slast) continue;
#pragma unroll
      for (int o = 16, n = 8; o >= 4; o >>= 1, n >>= 1) {
        const bool up = (lane & o) != 0;
#pragma unroll
        for (int i = 0; i < n / 2; ++i) {
          const float send = up ? l8[i] : l8[i + n / 2];
          const float keep = up ? l8[i + n / 2] : l8[i];
          l8[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
      }
      s_red[q4 * H + 32 * hh + head_rel] = l8[0];
      sm_sync();
      const int slot = int(blockIdx.x) - first_cta(t, per);
      const size_t prow_i = size_t(t) * (G != 0 ? GA : max_parts(per)) + slot;
      if (q4 == 0) {
        const int h = 32 * hh + lane;
        const float L = s_red[h] + s_red[H + h] + s_red[2 * H + h] + s_red[3 * H + h];
        s_inv[h] = L > 0.f ? 1.f / L : 0.f;
        const float lse = L > 0.f ? (s_m[h] * LN2 + logf(L * (1.f / PSCALE))) : INFINITY;
        lse_out[prow_i * H + h] = lse;
      }
      sm_sync();
#ifdef DMLA8_HANGDBG
      if (blockIdx.x == DBGCTA && lane == 0) printf("sm cta5 warp %d k %d sums done\n", warp, k);
#endif
      ST6(6 + 8 * k);
      uint4* pbase = reinterpret_cast<uint4*>(parts + prow_i * DV * H);
      mbar_wait(B7_OD(b), (k >> 1) & 1);
      tc_fence_after();
      ST6(7 + 8 * k);
#pragma unroll
      for (int cb = 0; cb < DV / 128; ++cb) {
        float v[32];
        tmem_ld32(tmem + lane_base + 64 * cb + 32 * hh, v);
        if constexpr (ABL != 2) {
#pragma unroll
          for (int x = 0; x < 4; ++x) {
            const float4 i0 = *reinterpret_cast<const float4*>(&s_inv[32 * hh + 8 * x]);
            const float4 i1 = *reinterpret_cast<const float4*>(&s_inv[32 * hh + 8 * x + 4]);
            const float inv[8] = {i0.x, i0.y, i0.z, i0.w, i1.x, i1.y, i1.z, i1.w};
            uint32_t w[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int y = 8 * x + 2 * e;
              const __nv_bfloat162 bb = __floats2bfloat162_rn(v[y] * inv[2 * e], v[y + 1] * inv[2 * e + 1]);
              w[e] = *reinterpret_cast<const uint32_t*>(&bb);
            }
            pbase[((cb * 2 + hh) * 4 + x) * 128 + 32 * q4 + lane] = make_uint4(w[0], w[1], w[2], w[3]);
          }
        }
      }
      tc_fence_before();
      mbar_arrive(B7_OFREE);
#ifdef DMLA8_HANGDBG
      if (blockIdx.x == DBGCTA && lane == 0) printf("sm cta5 warp %d k %d OFREE\n", warp, k);
#endif
      ST6(8 + 8 * k);
#pragma unroll
      for (int i = 0; i < 8; ++i) l8[i] = 0.f;
    }
    if (pdl && tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
  tc_fence_before();
  __syncthreads();
  if (warp == WMMA) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem) : "memory");
  }
  if constexpr (G > 0) {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  }
  if constexpr (G != 0) {
    if (warp >= NSM) return;
    const int t = int(blockIdx.x) / GA, rank = int(blockIdx.x) % GA;
    if (G < 0) {
    __threadfence();
    sm_sync();
    if (tid == 0) {
      atomicAdd(counter + t, rank == 0 ? 1u + FL_PERIOD - GA : 1u);
      while (true) {
        unsigned int v;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(counter + t) : "memory");
        if (v - fl_base >= FL_PERIOD) break;
        __nanosleep(32);
      }
    }
    sm_sync();
    }
    if (tid == 0) ST6(62);
    constexpr int COLS = DV / GA;
    constexpr int ITEMS = COLS * 8;
    constexpr int PER_T = ITEMS / (NSM * 32);
    float* s_w = reinterpret_cast<float*>(smem + OFF_P);                                   
    __nv_bfloat16* s_o = reinterpret_cast<__nv_bfloat16*>(smem + OFF_Q);                  
    const uint4* pb = reinterpret_cast<const uint4*>(parts) + size_t(t) * GA * (DV * H / 8);
    uint4 pv[PER_T][GA];
#pragma unroll
    for (int u = 0; u < PER_T; ++u) {
      const int item = tid + u * NSM * 32;
      const int cl = item % COLS, combo = item / COLS;
      const int col = rank * COLS + cl;
      const int idx = ((col >> 7) * 8 + combo) * 128 + (col & 127);
#pragma unroll
      for (int g = 0; g < GA; ++g) pv[u][g] = __ldcg(pb + size_t(g) * (DV * H / 8) + idx);
    }
    if (tid < H) {
      float l[GA];
      float mx = -INFINITY;
#pragma unroll
      for (int g = 0; g < GA; ++g) {
        l[g] = __ldcg(lse_out + (size_t(t) * GA + g) * H + tid);
        if (l[g] != INFINITY) mx = fmaxf(mx, l[g]);
      }
      float den = 0.f;
#pragma unroll
      for (int g = 0; g < GA; ++g) {
        l[g] = (l[g] == INFINITY || mx == -INFINITY) ? 0.f : __expf(l[g] - mx);
        den += l[g];
      }
      const float iv = den > 0.f ? 1.f / den : 0.f;
#pragma unroll
      for (int g = 0; g < GA; ++g) s_w[g * H + tid] = l[g] * iv;
    }
    sm_sync();
#pragma unroll
    for (int u = 0; u < PER_T; ++u) {
      const int item = tid + u * NSM * 32;
      const int cl = item % COLS, combo = item / COLS;
      const int h0 = 8 * combo;
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int g = 0; g < GA; ++g) {
        const uint32_t* vv = reinterpret_cast<const uint32_t*>(&pv[u][g]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vv[e]));
          acc[2 * e] += s_w[g * H + h0 + 2 * e] * f.x;
          acc[2 * e + 1] += s_w[g * H + h0 + 2 * e + 1] * f.y;
        }
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) s_o[(h0 + e) * (COLS + 8) + cl] = __float2bfloat16_rn(acc[e]);
    }
    sm_sync();
    constexpr int V8 = COLS / 8;
#pragma unroll
    for (int i = tid; i < H * V8; i += NSM * 32) {
      const int h = i / V8, c8 = i % V8;
      *reinterpret_cast<uint4*>(out + (size_t(t) * H + h) * DV + rank * COLS + 8 * c8) =
          *reinterpret_cast<const uint4*>(s_o + h * (COLS + 8) + 8 * c8);
    }
    if (tid == 0) ST6(63);
  }
}

template <int MAXN>
__global__ void __launch_bounds__(256) dmla8_merge(const __nv_bfloat16* __restrict__ parts,
                                                   const float* __restrict__ lse, __nv_bfloat16* __restrict__ out,
                                                   int per, int pdl) {
  __shared__ float s_w[MAXN][32];
  __shared__ __align__(16) __nv_bfloat16 s_o[32][128 + 8];
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
  const int cb = blockIdx.x, hh = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
  const int np = max_parts(per), n = n_parts(t, per);
  if (tid < 32) {
    const int h = 32 * hh + tid;
    float l[MAXN];
    float m = -INFINITY;
#pragma unroll
    for (int s = 0; s < MAXN; ++s) {
      l[s] = s < n ? __ldg(lse + (size_t(t) * np + s) * H + h) : INFINITY;
      if (l[s] != INFINITY) m = fmaxf(m, l[s]);
    }
    float den = 0.f;
#pragma unroll
    for (int s = 0; s < MAXN; ++s) {
      l[s] = (l[s] == INFINITY || m == -INFINITY) ? 0.f : __expf(l[s] - m);
      den += l[s];
    }
    const float iv = den > 0.f ? 1.f / den : 0.f;
#pragma unroll
    for (int s = 0; s < MAXN; ++s) s_w[s][tid] = l[s] * iv;
  }
  __syncthreads();
  const int j = tid & 127, ih = tid >> 7;
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.f;
  const uint4* base = reinterpret_cast<const uint4*>(parts) + size_t(t) * np * (DV * H / 8);
  uint4 v[MAXN][2];
#pragma unroll
  for (int s = 0; s < MAXN; ++s)
#pragma unroll
    for (int u = 0; u < 2; ++u)
      v[s][u] = s < n ? __ldg(base + size_t(s) * (DV * H / 8) + ((cb * 2 + hh) * 4 + 2 * ih + u) * 128 + j)
                      : make_uint4(0, 0, 0, 0);
#pragma unroll
  for (int s = 0; s < MAXN; ++s)
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const uint32_t* vv = reinterpret_cast<const uint32_t*>(&v[s][u]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vv[e]));
        const int k = 8 * u + 2 * e;
        acc[k] += s_w[s][16 * ih + k] * f.x;
        acc[k + 1] += s_w[s][16 * ih + k + 1] * f.y;
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
                q == cudaDriverEntryPointSuccess && p, "dmla8: cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

static CUtensorMap make_map(const void* base, uint64_t rows, uint32_t box_rows, CUtensorMapL2promotion promo) {
  CUtensorMap m;
  const uint64_t size[2] = {uint64_t(DQK), rows};
  const uint64_t stride[1] = {uint64_t(DQK)};
  const uint32_t box[2] = {128u, box_rows};
  const uint32_t es[2] = {1u, 1u};
  const CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), size, stride, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, promo,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "dmla8: cuTensorMapEncodeTiled failed ", int(r));
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

int64_t parts_rows(int64_t T, int64_t per) { return T * max_parts(int(per)); }

template <int ABL>
static const void* kernel_ptr_g(int g) {
  switch (g) {
    case 0: return reinterpret_cast<const void*>(dmla8_kernel<ABL, 0>);
    case 2: return reinterpret_cast<const void*>(dmla8_kernel<ABL, 2>);
    case 4: return reinterpret_cast<const void*>(dmla8_kernel<ABL, 4>);
    case 8: return reinterpret_cast<const void*>(dmla8_kernel<ABL, 8>);
    case -4: return reinterpret_cast<const void*>(dmla8_kernel<ABL, -4>);
    case -8: return reinterpret_cast<const void*>(dmla8_kernel<ABL, -8>);
    default: return reinterpret_cast<const void*>(dmla8_kernel<ABL, 16>);
  }
}
template <int ABL>
static const void* kernel_ptr_a(int g) {       
  return g == 0 ? reinterpret_cast<const void*>(dmla8_kernel<ABL, 0>) : reinterpret_cast<const void*>(dmla8_kernel<ABL, 4>);
}
static const void* kernel_ptr(int abl, int g) {
#ifdef DMLA8_ABL
  return abl == 1 ? kernel_ptr_a<1>(g) : abl == 2 ? kernel_ptr_a<2>(g) : abl == 3 ? kernel_ptr_a<3>(g) : kernel_ptr_g<0>(g);
#else
  TORCH_CHECK(abl == 0, "dmla8: ablations need -DDMLA8_ABL");
  return kernel_ptr_g<0>(g);
#endif
}

int64_t max_clusters(int64_t g) {
  const void* fn = kernel_ptr(0, int(g));
  cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  if (g > 8) cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(int(g), 1);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = SMEM;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = unsigned(g); at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  int n = -1;
  const cudaError_t e = cudaOccupancyMaxActiveClusters(&n, fn, &cfg);
  TORCH_CHECK(e == cudaSuccess, "occupancy query: ", cudaGetErrorString(e));
  return n;
}

void decode8(torch::Tensor q, torch::Tensor kv, torch::Tensor pt, torch::Tensor lens, double scale, int64_t per,
             torch::Tensor parts, torch::Tensor lse, torch::Tensor out, bool pdl, bool merge,
             c10::optional<torch::Tensor> dbg, int64_t abl, int64_t fuse, c10::optional<torch::Tensor> counter) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kFloat8_e4m3fn && q.dim() == 3 && q.size(1) == H
              && q.size(2) == DQK && q.is_contiguous(), "dmla8: q must be contiguous e4m3 [T, 64, 576]");
  TORCH_CHECK(kv.scalar_type() == at::kFloat8_e4m3fn && kv.size(-1) == DQK && kv.is_contiguous()
              && reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0, "dmla8: kv must be contiguous e4m3 [S, 576]");
  const int T = q.size(0);
  TORCH_CHECK(T >= 1 && T <= MAXT, "dmla8: tokens ", T);
  TORCH_CHECK(per >= 1 && per <= MAXPER, "dmla8: per must be 1..8");
  TORCH_CHECK(pt.scalar_type() == at::kInt && pt.dim() == 2 && pt.size(0) == T && pt.size(1) >= TOPK
              && pt.stride(1) == 1 && pt.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(pt.data_ptr()) % 16 == 0,
              "dmla8: page table must be 16-byte aligned int32 [T, >= 2048]");
  TORCH_CHECK(lens.scalar_type() == at::kInt && lens.numel() == T && lens.is_contiguous(), "dmla8: lens");
  const int64_t rows = parts_rows(T, per);
  TORCH_CHECK(parts.scalar_type() == at::kBFloat16 && parts.is_contiguous() && parts.numel() >= rows * H * DV,
              "dmla8: parts must be bf16 [T * max_parts(per), 32768]");
  TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.is_contiguous() && lse.numel() >= rows * H, "dmla8: lse");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == int64_t(T) * H * DV,
              "dmla8: out must be contiguous bf16 [T, 64, 512]");
  c10::cuda::CUDAGuard guard(q.device());
  auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
  const int p = int(per);
  const int GA = fuse ? SPLITS / p : 0;
  TORCH_CHECK(!fuse || (GA * p == SPLITS && (GA == 2 || GA == 4 || GA == 8 || GA == 16)), "dmla8: fuse needs per in 1, 2, 4, 8");
#ifndef DMLA8_FLAG
  TORCH_CHECK(fuse != 2, "dmla8: fuse = 2 (flag merge) is experimental and disabled");
#endif
  TORCH_CHECK(fuse != 2 || ((GA == 4 || GA == 8) && counter.has_value() && counter->numel() >= T
              && counter->scalar_type() == at::kInt), "dmla8: flag merge needs per 2 or 4 and an int32 counter[>= T]");
  const int G = fuse == 2 ? -GA : GA;
  unsigned int* ctr = counter.has_value() ? reinterpret_cast<unsigned int*>(counter->data_ptr()) : nullptr;
  const CUtensorMap mq = make_map(q.data_ptr(), uint64_t(T) * H, H, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
  const CUtensorMap mkv = make_map(kv.data_ptr(), uint64_t(kv.numel() / DQK), 1, CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  int n_tiles = T * SPLITS, pp_ = p;
  const int grid = (n_tiles + p - 1) / p;
  float sl2 = float(scale) * 1.4426950408889634f;
  const int32_t* ptp = pt.data_ptr<int32_t>();
  int64_t pst = pt.stride(0);
  const int32_t* lp = lens.data_ptr<int32_t>();
  __nv_bfloat16* pp = reinterpret_cast<__nv_bfloat16*>(parts.data_ptr());
  float* lsp = lse.data_ptr<float>();
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const uint8_t* kvb = reinterpret_cast<const uint8_t*>(kv.data_ptr());
  int pd = pdl ? 1 : 0;
  unsigned long long* stp = dbg.has_value() ? reinterpret_cast<unsigned long long*>(dbg->data_ptr()) : nullptr;
  void* args[] = {const_cast<CUtensorMap*>(&mq), const_cast<CUtensorMap*>(&mkv), &ptp, &pst, &lp, &n_tiles, &pp_, &sl2,
                  &pp, &lsp, &pd, &stp, &op, &kvb, &ctr};
  const void* fn = kernel_ptr(int(abl), G);
  static bool attr[4][7] = {};
  const int gi = G == 0 ? 0 : G == 2 ? 1 : G == 4 ? 2 : G == 8 ? 3 : G == 16 ? 4 : G == -4 ? 5 : 6;
  if (!attr[abl & 3][gi]) {
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess, "dmla8: smem");
    if (G > 8)
      TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) == cudaSuccess, "dmla8: cluster");
    attr[abl & 3][gi] = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = stream;
  cudaLaunchAttribute at[2];
  int na = 0;
  if (G > 0) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = unsigned(G); at[na].val.clusterDim.y = 1; at[na].val.clusterDim.z = 1;
    ++na;
  }
  if (pdl) {
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  const cudaError_t e = cudaLaunchKernelExC(&cfg, fn, args);
  TORCH_CHECK(e == cudaSuccess, "dmla8 launch failed: ", cudaGetErrorString(e));
  if (!merge || G > 0) return;
  void* margs[] = {&pp, &lsp, &op, &pp_, &pd};
  const int mp = max_parts(p);
  const void* mfn = mp <= 4 ? reinterpret_cast<const void*>(dmla8_merge<4>)
                  : mp <= 8 ? reinterpret_cast<const void*>(dmla8_merge<8>)
                            : reinterpret_cast<const void*>(dmla8_merge<17>);
  launch_ex(mfn, dim3(DV / 128, 2, T), dim3(256), 0, stream, pdl, margs, "dmla8 merge");
}

}   

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode8", &dmla8::decode8);
  m.def("parts_rows", &dmla8::parts_rows);
  m.def("max_clusters", &dmla8::max_clusters);
}
