#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace sx8 {

using bf16 = __nv_bfloat16;

struct Args {
  const bf16* x;            
  int64_t xb, xm;
  const uint8_t* w;         
  const __half* s;          
  bf16* out;                
  int64_t ob, om;
  int M, pdl;
};

__device__ __forceinline__ uint4 ldg_nc(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ldg_nc32(const void* p) {
  uint32_t v;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t word_of(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}
__device__ __forceinline__ void i8_frag(uint32_t u, uint32_t& b0, uint32_t& b1) {
  b0 = __byte_perm(u, 0x64646464u, 0x5140);
  b1 = __byte_perm(u, 0x64646464u, 0x7362);
  asm("sub.f16x2 %0, %0, %1;" : "+r"(b0) : "r"(0x64806480u));
  asm("sub.f16x2 %0, %0, %1;" : "+r"(b1) : "r"(0x64806480u));
}
__device__ __forceinline__ uint32_t bf2h(uint32_t v) {
  uint32_t h;
  asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;" : "=r"(h) : "f"(__uint_as_float(v & 0xFFFF0000u)), "f"(__uint_as_float(v << 16)));
  return h;
}
__device__ __forceinline__ void mma_f16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t phase) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}"
               :: "r"(bar), "r"(phase) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ float silu(float v) { return v / (1.f + __expf(-v)); }

__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

template <int NT, int KGT, int WP, int KG, int WK, int WN, int DEPTH, int PRE, int SG, int KC, int TT, int EPI,
          bool TMA, int NS, bool ATOM>
struct Cfg {
  static constexpr int WT = 2 * WP;
  static constexpr int THREADS = 32 * WK * WN;
  static constexpr int ROWS = 8 * TT;
  static constexpr int HC = TT <= 2 ? 2 : 1;         
  static constexpr int KCTA = WK * KG;
  static constexpr int XS = KG * 128 + 64;
  static constexpr int ACC = 4 * TT;                 
  static constexpr int RBYTES = (WK - 1) * WN * WP * 32 * ACC * 4;
  static constexpr int CBYTES = ATOM ? 0 : (KC - 1) * WN * WP * 32 * ACC * 4;
  static constexpr int WBYTES = TMA ? WK * WN * WT * KG * 512 : 0;
  static constexpr int SGK = KG / NS;
  static constexpr int FIXED = RBYTES + CBYTES + WBYTES + (TMA ? WK * WN * NS * 8 : 0);   
  static constexpr int NGW = KG / SG;
  static int smem(int m) { return WK * m * XS + FIXED; }           
  static_assert(KCTA * KC == KGT && KG % SG == 0 && DEPTH <= KG && PRE <= DEPTH && NT % (WN * WT) == 0, "geometry");
  static_assert(WK <= 15 && (!TMA || KG % NS == 0), "barriers");
  static_assert(EPI == 0 || WP == 1, "the SiLU * up epilogue reads one gate / up pair a warp");
};

template <int NT, int KGT, int WP, int KG, int WK, int WN, int DEPTH, int PRE, int SG, int KC, int TT, int EPI,
          bool TMA, int NS, bool ATOM>
__global__ void __launch_bounds__(32 * WK * WN)
i8_kernel(const Args a) {
  using C = Cfg<NT, KGT, WP, KG, WK, WN, DEPTH, PRE, SG, KC, TT, EPI, TMA, NS, ATOM>;
  constexpr int WT = C::WT;
  extern __shared__ __align__(128) char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wn = warp % WN, wk = warp / WN;
  const int g = lane >> 2, q = lane & 3;
  const int b = blockIdx.y;
  const int kr = ATOM ? int(blockIdx.x % KC) : KC > 1 ? int(cluster_rank()) : 0;    
  const int tile0 = (blockIdx.x / KC) * (WN * WT) + wn * WT;        
  const int j0 = kr * C::KCTA + wk * KG;
  if (KC > 1 && !ATOM) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  const uint8_t* wl[WT];
#pragma unroll
  for (int t = 0; t < WT; ++t)
    wl[t] = a.w + ((size_t(b) * NT + tile0 + t) * KGT + j0) * 512 + lane * 16;
  uint32_t sc[WP][C::NGW];                                          
#pragma unroll
  for (int p = 0; p < WP; ++p)
#pragma unroll
    for (int i = 0; i < C::NGW; ++i)
      sc[p][i] = ldg_nc32(a.s + ((size_t(b) * (NT / 2) + tile0 / 2 + p) * (KGT / SG) + j0 / SG + i) * 16 + 2 * g);
  uint4 ring[TMA ? 1 : DEPTH][WT];
  const int xbytes = WK * a.M * C::XS;                              
  const uint32_t wsm = smem_u32(smem) + xbytes + C::RBYTES + C::CBYTES + warp * (WT * KG * 512);
  const uint32_t bars = smem_u32(smem) + xbytes + C::RBYTES + C::CBYTES + C::WBYTES + warp * NS * 8;
  if (TMA) {
    if (lane == 0) {
#pragma unroll
      for (int s = 0; s < NS; ++s) mbar_init(bars + 8 * s, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll
      for (int s = 0; s < NS; ++s) {
        mbar_expect_tx(bars + 8 * s, uint32_t(WT * C::SGK * 512));
#pragma unroll
        for (int t = 0; t < WT; ++t)
          bulk_g2s(wsm + (t * KG + s * C::SGK) * 512, wl[t] - lane * 16 + s * C::SGK * 512, uint32_t(C::SGK * 512),
                   bars + 8 * s);
      }
    }
  } else {
#pragma unroll
    for (int d = 0; d < PRE; ++d)
#pragma unroll
      for (int t = 0; t < WT; ++t) ring[d][t] = ldg_nc(wl[t] + d * 512);
  }
  if (a.pdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
  const uint32_t sx = smem_u32(smem) + wk * a.M * C::XS;
  {
    constexpr int CH = KG * 8;
    const bf16* xb = a.x + b * a.xb + j0 * 64;
    for (int i = wn * 32 + lane; i < a.M * CH; i += WN * 32) {
      const int r = i / CH, c = i % CH;
      cp16(sx + r * C::XS + c * 16, xb + r * a.xm + c * 8);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    if (!TMA) {
#pragma unroll
      for (int d = PRE; d < DEPTH; ++d)
#pragma unroll
        for (int t = 0; t < WT; ++t) ring[d][t] = ldg_nc(wl[t] + d * 512);
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    for (int i = wn * 32 + lane; i < a.M * CH; i += WN * 32) {
      const uint32_t at = sx + (i / CH) * C::XS + (i % CH) * 16;
      uint4 v = lds128(at);
      v = make_uint4(bf2h(v.x), bf2h(v.y), bf2h(v.z), bf2h(v.w));
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(at), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                   : "memory");
    }
  }
  if (WN > 1) asm volatile("bar.sync %0, %1;" :: "r"(1 + wk), "r"(WN * 32) : "memory");
  else __syncwarp();
  float acc[WP][C::ACC], part[WP][C::HC][C::ACC];                  
#pragma unroll
  for (int p = 0; p < WP; ++p)
#pragma unroll
    for (int e = 0; e < C::ACC; ++e) acc[p][e] = 0.f;
#pragma unroll
  for (int j = 0; j < KG; ++j) {
    if (j % SG == 0) {
#pragma unroll
      for (int p = 0; p < WP; ++p)
#pragma unroll
        for (int h = 0; h < C::HC; ++h)
#pragma unroll
          for (int e = 0; e < C::ACC; ++e) part[p][h][e] = 0.f;
    }
    uint4 cur[WT];
    if (TMA) {
      if (j % C::SGK == 0) mbar_wait(bars + 8 * (j / C::SGK), 0);
#pragma unroll
      for (int t = 0; t < WT; ++t) cur[t] = lds128(wsm + (t * KG + j) * 512 + lane * 16);
    } else {
#pragma unroll
      for (int t = 0; t < WT; ++t) cur[t] = ring[j % DEPTH][t];
      if (j + DEPTH < KG) {
#pragma unroll
        for (int t = 0; t < WT; ++t) ring[j % DEPTH][t] = ldg_nc(wl[t] + (j + DEPTH) * 512);
      }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t fr[WP][2][4];
#pragma unroll
      for (int p = 0; p < WP; ++p)
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          i8_frag(word_of(cur[2 * p], 2 * h + u), fr[p][u][0], fr[p][u][2]);
          i8_frag(word_of(cur[2 * p + 1], 2 * h + u), fr[p][u][1], fr[p][u][3]);
        }
#pragma unroll
      for (int tt = 0; tt < TT; ++tt) {
        const uint4 xa = lds128(sx + min(g + 8 * tt, a.M - 1) * C::XS + 16 * q + (j * 64 + 32 * h) * 2);
#pragma unroll
        for (int p = 0; p < WP; ++p) {
#pragma unroll
          for (int u = 0; u < 2; ++u)                               
            mma_f16(*reinterpret_cast<float(*)[4]>(&part[p][h % C::HC][4 * tt]), fr[p][u][0], fr[p][u][1], fr[p][u][2],
                    fr[p][u][3], u ? xa.z : xa.x, u ? xa.w : xa.y);
        }
      }
    }
    if (j % SG == SG - 1) {
#pragma unroll
      for (int p = 0; p < WP; ++p) {
        const float2 s2 = __half22float2(*reinterpret_cast<const __half2*>(&sc[p][j / SG]));
#pragma unroll
        for (int e = 0; e < C::ACC; ++e)
          acc[p][e] = fmaf(C::HC == 2 ? part[p][0][e] + part[p][C::HC - 1][e] : part[p][0][e], (e & 2) ? s2.y : s2.x,
                           acc[p][e]);
      }
    }
  }
  if (WK > 1) {
    float* red = reinterpret_cast<float*>(smem + xbytes);
    if (wk > 0) {
#pragma unroll
      for (int p = 0; p < WP; ++p)
#pragma unroll
        for (int e = 0; e < C::ACC; e += 4)
          *reinterpret_cast<float4*>(&red[((((wk - 1) * WN + wn) * WP + p) * 32 + lane) * C::ACC + e]) =
              make_float4(acc[p][e], acc[p][e + 1], acc[p][e + 2], acc[p][e + 3]);
    }
    __syncthreads();
    if (wk == 0) {
#pragma unroll
      for (int w = 0; w < WK - 1; ++w)
#pragma unroll
        for (int p = 0; p < WP; ++p)
#pragma unroll
          for (int e = 0; e < C::ACC; e += 4) {
            const float4 v = *reinterpret_cast<const float4*>(&red[(((w * WN + wn) * WP + p) * 32 + lane) * C::ACC + e]);
            acc[p][e] += v.x; acc[p][e + 1] += v.y; acc[p][e + 2] += v.z; acc[p][e + 3] += v.w;
          }
    }
  }
  if (KC > 1 && !ATOM) {
    float* cred = reinterpret_cast<float*>(smem + xbytes + C::RBYTES);
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    if (kr > 0 && wk == 0) {
#pragma unroll
      for (int p = 0; p < WP; ++p)
#pragma unroll
        for (int e = 0; e < C::ACC; e += 4) {
          const uint32_t local = smem_u32(&cred[((((kr - 1) * WN + wn) * WP + p) * 32 + lane) * C::ACC + e]);
          uint32_t remote;
          asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(remote) : "r"(local));
          asm volatile("st.shared::cluster.v4.f32 [%0], {%1,%2,%3,%4};"
                       :: "r"(remote), "f"(acc[p][e]), "f"(acc[p][e + 1]), "f"(acc[p][e + 2]), "f"(acc[p][e + 3])
                       : "memory");
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (kr > 0) return;
    if (wk == 0) {
#pragma unroll
      for (int r = 0; r < KC - 1; ++r)
#pragma unroll
        for (int p = 0; p < WP; ++p)
#pragma unroll
          for (int e = 0; e < C::ACC; e += 4) {
            const float4 v = *reinterpret_cast<const float4*>(&cred[(((r * WN + wn) * WP + p) * 32 + lane) * C::ACC + e]);
            acc[p][e] += v.x; acc[p][e + 1] += v.y; acc[p][e + 2] += v.z; acc[p][e + 3] += v.w;
          }
    }
  }
  if (wk > 0) return;
  if (ATOM) {
    float* sums = reinterpret_cast<float*>(a.out);
#pragma unroll
    for (int p = 0; p < WP; ++p) {
#pragma unroll
      for (int e = 0; e < C::ACC; ++e) {
        const int tok = 2 * q + (e & 1) + 8 * (e >> 2);
        if (tok >= a.M) continue;
        const int ch = (tile0 + 2 * p) * 8 + ((e & 2) ? 8 : 0) + g;
        asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" :: "l"(sums + size_t(tok) * a.om + ch), "f"(acc[p][e])
                     : "memory");
      }
    }
    return;
  }
  bf16* ob = a.out + b * a.ob;
#pragma unroll
  for (int p = 0; p < WP; ++p) {
#pragma unroll
    for (int e = 0; e < C::ACC; ++e) {
      const int tok = 2 * q + (e & 1) + 8 * (e >> 2);               
      if (tok >= a.M || ((e & 2) && EPI == 1)) continue;
      if (EPI == 0) {
        const int ch = (tile0 + 2 * p) * 8 + ((e & 2) ? 8 : 0) + g;
        ob[tok * a.om + ch] = __float2bfloat16_rn(acc[p][e]);
      } else {
        const float gt = __bfloat162float(__float2bfloat16_rn(acc[p][e]));
        const float up = __bfloat162float(__float2bfloat16_rn(acc[p][e + 2]));
        ob[tok * a.om + (tile0 / 2) * 8 + g] = __float2bfloat16_rn(silu(gt) * up);
      }
    }
  }
}

template <typename K>
static void launch(K kernel, int smem, dim3 grid, int threads, int cluster, bool pdl, const Args& args) {
  if (smem > 48 * 1024) {
    const cudaError_t e = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    TORCH_CHECK(e == cudaSuccess, "i8dec shared memory: ", cudaGetErrorString(e));
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(threads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute attr[2];
  int n = 0;
  if (pdl) {
    attr[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  if (cluster > 1) {
    attr[n].id = cudaLaunchAttributeClusterDimension;
    attr[n].val.clusterDim.x = cluster;
    attr[n].val.clusterDim.y = 1;
    attr[n].val.clusterDim.z = 1;
    ++n;
  }
  cfg.attrs = attr;
  cfg.numAttrs = n;
  Args copy = args;
  void* kargs[] = {&copy};
  const cudaError_t e = cudaLaunchKernelExC(&cfg, reinterpret_cast<const void*>(kernel), kargs);
  TORCH_CHECK(e == cudaSuccess, "i8dec launch: ", cudaGetErrorString(e));
}

template <int NT, int KGT, int WP, int KG, int WK, int WN, int DEPTH, int PRE, int SG, int KC, int TT, int EPI,
          bool TMA, int NS, bool ATOM>
static void run_rows(const Args& args, int batch, bool pdl, int min_smem) {
  using C = Cfg<NT, KGT, WP, KG, WK, WN, DEPTH, PRE, SG, KC, TT, EPI, TMA, NS, ATOM>;
  launch(i8_kernel<NT, KGT, WP, KG, WK, WN, DEPTH, PRE, SG, KC, TT, EPI, TMA, NS, ATOM>,
         std::max(C::smem(args.M), min_smem), dim3(NT / (WN * C::WT) * KC, batch), C::THREADS, ATOM ? 1 : KC, pdl, args);
}

template <int NT, int KGT, int WP, int KG, int WK, int WN, int DEPTH, int PRE, int SG, int KC, int EPI, bool ATOM>
static void run(const Args& args, bool pdl, int min_smem = 0) {
  TORCH_CHECK(args.M >= 1 && args.M <= 128, "sx8: 1..128 rows");
#define SX8_TT(n) run_rows<NT, KGT, WP, KG, WK, WN, DEPTH, PRE, SG, KC, n, EPI, false, 1, ATOM>(args, 1, pdl, min_smem)
  if (args.M <= 8) SX8_TT(1);
  else if (args.M <= 16) SX8_TT(2);
  else if (args.M <= 32) SX8_TT(4);
  else if (args.M <= 48) SX8_TT(6);
  else if (args.M <= 64) SX8_TT(8);
  else if (args.M <= 96) SX8_TT(12);
  else SX8_TT(16);
#undef SX8_TT
}

static Args make_args(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s,
                      const torch::Tensor& out, int64_t batch, int64_t k, int64_t n, int64_t sg, int64_t nout,
                      bool pdl) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 3 && x.stride(2) == 1
              && x.size(0) == batch && x.size(2) == k, "i8dec: x must be bf16 [batch, M, K] with a unit K stride");
  TORCH_CHECK(x.stride(0) % 8 == 0 && x.stride(1) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "i8dec: x rows must be 16-byte aligned");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.dim() == 3 && out.stride(2) == 1
              && out.size(0) == batch && out.size(1) == x.size(1) && out.size(2) == nout
              && out.stride(0) % 2 == 0 && out.stride(1) % 2 == 0
              && reinterpret_cast<uintptr_t>(out.data_ptr()) % 4 == 0, "i8dec: out must be bf16 [batch, M, N]");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kByte && w.is_contiguous() && w.numel() == batch * n * k
              && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "i8dec: weight bytes");
  TORCH_CHECK(s.is_cuda() && s.scalar_type() == at::kHalf && s.is_contiguous() && s.numel() == batch * n * k / (64 * sg)
              && reinterpret_cast<uintptr_t>(s.data_ptr()) % 4 == 0, "i8dec: scales");
  TORCH_CHECK(x.device() == w.device() && x.device() == out.device() && x.device() == s.device(), "i8dec: devices");
  Args a;
  a.x = static_cast<const bf16*>(x.data_ptr());
  a.xb = x.stride(0);
  a.xm = x.stride(1);
  a.w = w.data_ptr<uint8_t>();
  a.s = reinterpret_cast<const __half*>(s.data_ptr());
  a.out = static_cast<bf16*>(out.data_ptr());
  a.ob = out.stride(0);
  a.om = out.stride(1);
  a.M = int(x.size(1));
  a.pdl = pdl ? 1 : 0;
  return a;
}

void gate_up_sums(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor sums, bool pdl) {
  TORCH_CHECK(sums.is_cuda() && sums.scalar_type() == at::kFloat && sums.dim() == 2 && sums.size(1) == 1024
              && sums.size(0) >= x.size(1) && sums.is_contiguous(), "sx8: sums fp32 [>= M, 1024]");
  torch::Tensor fake = torch::empty({1, x.size(1), 1024}, x.options());       
  Args a = make_args(x, w, s, fake, 1, 6144, 1024, 2, 1024, pdl);
  a.out = reinterpret_cast<bf16*>(sums.data_ptr<float>());
  a.om = 1024;
  c10::cuda::CUDAGuard guard(x.device());
  run<128, 96, 1, 12, 1, 8, 6, 2, 2, 8, 0, true>(a, pdl);
}

__global__ void act_rezero_kernel(float* __restrict__ sums, bf16* __restrict__ act, int M) {
  const int m = blockIdx.x, c = threadIdx.x;                    
  if (m >= M) return;
  float* row = sums + size_t(m) * 1024 + 16 * (c >> 3) + (c & 7);
  const float gt = __bfloat162float(__float2bfloat16_rn(row[0]));
  const float up = __bfloat162float(__float2bfloat16_rn(row[8]));
  row[0] = 0.f;
  row[8] = 0.f;
  act[size_t(m) * 512 + c] = __float2bfloat16_rn(silu(gt) * up);
}

void act_rezero(torch::Tensor sums, torch::Tensor act) {
  TORCH_CHECK(sums.is_cuda() && sums.scalar_type() == at::kFloat && sums.dim() == 2 && sums.size(1) == 1024
              && sums.is_contiguous(), "sx8: sums fp32 [>= M, 1024]");
  TORCH_CHECK(act.is_cuda() && act.scalar_type() == at::kBFloat16 && act.dim() == 2 && act.size(1) == 512
              && act.is_contiguous() && act.size(0) <= sums.size(0), "sx8: act bf16 [M, 512]");
  c10::cuda::CUDAGuard guard(sums.device());
  act_rezero_kernel<<<static_cast<unsigned>(act.size(0)), 512, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
      sums.data_ptr<float>(), static_cast<bf16*>(act.data_ptr()), int(act.size(0)));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "sx8: act launch");
}

void down(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor out, bool pdl) {
  const Args a = make_args(x, w, s, out, 1, 512, 6144, 2, 6144, pdl);
  c10::cuda::CUDAGuard guard(x.device());
  run<768, 8, 1, 8, 1, 4, 8, 8, 2, 1, 0, false>(a, pdl);
}

}   

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gate_up_sums", &sx8::gate_up_sums);
  m.def("act_rezero", &sx8::act_rezero);
  m.def("down", &sx8::down);
}
