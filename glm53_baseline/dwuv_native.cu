#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace dwuv {

constexpr int H = 64, K = 512, N = 256;
constexpr int KS = K / 16;                     
constexpr int TILES = H * N / 16;              
constexpr int TPC = 13;                        
constexpr int CTAS = (TILES + TPC - 1) / TPC;  
constexpr uint32_t TBYTES = KS * 512;          
constexpr int MAX_T = 8;
constexpr int PITCH = K * 2 + 16;              
constexpr uint32_t XBYTES = 2 * MAX_T * PITCH;
constexpr size_t SMEM = size_t(TPC) * TBYTES + XBYTES;
constexpr int THREADS = TPC * 32;
constexpr int RING_W = 4, RING_SLOTS = 3, RING_RM = 16, RING_NBLK = 8, RING_KSPL = 1024;
constexpr size_t RING_HDR = 1024 * 8, RING_PACKED = 16384 + 1536;
constexpr uint32_t RING_EMPTY = 0x80000000u;
static_assert(H * N == 16384, "x is the scatter's 16384-wide row");

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect(uint32_t bar, uint32_t n) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t phase) {
  asm volatile("{ .reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W; }"
               :: "r"(bar), "r"(phase) : "memory");
}
__device__ __forceinline__ void bulk_copy(uint32_t dst, const void* src, uint32_t n, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory");
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ uint4 lds_v4(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ uint32_t lds_u32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint16_t bf16_bits(float v) {
  const __nv_bfloat16 b = __float2bfloat16_rn(v);
  return *reinterpret_cast<const uint16_t*>(&b);
}

__device__ __forceinline__ void st_sys(void* p, uint4 v) {
  asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint4 scrub(uint4 v) {
  return make_uint4(v.x == RING_EMPTY ? 0u : v.x, v.y == RING_EMPTY ? 0u : v.y, v.z == RING_EMPTY ? 0u : v.z,
                    v.w == RING_EMPTY ? 0u : v.w);
}

__global__ void __launch_bounds__(THREADS, 1)
dwuv_kernel(const __nv_bfloat16* __restrict__ o, const uint8_t* __restrict__ frag, __nv_bfloat16* __restrict__ out,
            long long so_t, long long sout_t, int T, int pdl, int early,
            const int64_t* __restrict__ maps, int rank, const int64_t* __restrict__ loc, int push) {
  extern __shared__ __align__(1024) char smem[];
  __shared__ uint64_t wbar[TPC];
  __shared__ uint64_t epoch_s[2];
  const uint32_t sw = smem_u32(smem), sx = sw + uint32_t(TPC) * TBYTES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int tile0 = blockIdx.x * TPC, tile = tile0 + warp;
  const int nt = min(TPC, TILES - tile0);                           
  const bool live = warp < nt;
  const int h0 = tile0 >> 4, nh = ((tile0 + nt - 1) >> 4) - h0 + 1;  
  const uint32_t bar = smem_u32(&wbar[warp]);
  if (lane == 0) {
    mbar_init(bar);
    if (live) {
      mbar_arrive_expect(bar, TBYTES);
      bulk_copy(sw + uint32_t(warp) * TBYTES, frag + size_t(tile) * TBYTES, TBYTES, bar);
    }
  }
  const int ks0 = (tile0 * 16) / RING_KSPL;
  if (push) {
    if (tid == 0) {
      const uint64_t* counter = reinterpret_cast<const uint64_t*>(maps[rank]);
      epoch_s[0] = counter[ks0 * RING_NBLK] + 1;
      epoch_s[1] = counter[min(ks0 + 1, 16384 / RING_KSPL - 1) * RING_NBLK] + 1;
    }
    __syncthreads();
  }
  if (pdl) {
    if (early) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  for (int q = tid; q < nh * T * (K / 8); q += THREADS) {
    const int hh = q / (T * (K / 8)), r = (q / (K / 8)) % T, c = q % (K / 8);
    cp16(sx + uint32_t(hh * MAX_T + r) * PITCH + c * 16, o + r * so_t + size_t(h0 + hh) * K + c * 8);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  if (live) {
    mbar_wait(bar, 0);
    const uint32_t wa = sw + uint32_t(warp) * TBYTES + lane * 16;
    const uint32_t xb = sx + uint32_t(((tile >> 4) - h0) * MAX_T + (lane >> 2)) * PITCH + (lane & 3) * 4;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      const uint4 a = lds_v4(wa + ks * 512);
      const uint32_t b0 = lds_u32(xb + ks * 32), b1 = lds_u32(xb + ks * 32 + 16);
      mma_bf16(c, a, b0, b1);
    }
  }
  __syncthreads();
  if (live) {
    const int g = lane >> 2, q = lane & 3;
    const uint32_t s0 = sx + uint32_t((2 * q) * TPC * 16 + warp * 16 + g) * 2;
    const uint32_t s1 = s0 + uint32_t(TPC * 16) * 2;
    asm volatile("st.shared.u16 [%0], %1;" :: "r"(s0), "h"(bf16_bits(c[0])) : "memory");
    asm volatile("st.shared.u16 [%0], %1;" :: "r"(s1), "h"(bf16_bits(c[1])) : "memory");
    asm volatile("st.shared.u16 [%0], %1;" :: "r"(s0 + 16), "h"(bf16_bits(c[2])) : "memory");
    asm volatile("st.shared.u16 [%0], %1;" :: "r"(s1 + 16), "h"(bf16_bits(c[3])) : "memory");
  }
  __syncthreads();
  for (int q = tid; q < T * nt * 2; q += THREADS) {
    const int t = q / (nt * 2), ch = q % (nt * 2);
    const uint4 v = lds_v4(sx + uint32_t(t * TPC * 16) * 2 + ch * 16);
    *reinterpret_cast<uint4*>(out + t * sout_t + size_t(tile0) * 16 + ch * 8) = v;
    if (push) {
      const int col = tile0 * 16 + ch * 8;                          
      const int slot = int(epoch_s[col / RING_KSPL - ks0] % RING_SLOTS);
      const uint4 w = (loc != nullptr && loc[t] == 0) ? make_uint4(0, 0, 0, 0) : scrub(v);
      const size_t off = RING_HDR + ((size_t(slot * RING_W + rank) * RING_RM + t) * RING_PACKED + col) * 2;
#pragma unroll
      for (int d = 1; d < RING_W; ++d) st_sys(reinterpret_cast<char*>(maps[(rank + d) & 3]) + off, w);
    }
  }
}

void run(torch::Tensor o, torch::Tensor frag, torch::Tensor out, bool pdl, bool early, torch::Tensor maps, int64_t rank,
         torch::Tensor loc) {
  TORCH_CHECK(o.is_cuda() && o.scalar_type() == at::kBFloat16 && o.dim() == 3 && o.size(1) == H && o.size(2) == K
              && o.stride(2) == 1 && o.stride(1) == K && o.size(0) >= 1 && o.size(0) <= MAX_T
              && reinterpret_cast<uintptr_t>(o.data_ptr()) % 16 == 0 && o.stride(0) % 8 == 0,
              "dwuv: o must be bf16 [T <= 8, 64, 512], heads contiguous, 16-byte aligned rows");
  TORCH_CHECK(frag.is_cuda() && frag.is_contiguous() && frag.numel() * frag.element_size() == int64_t(TILES) * TBYTES
              && reinterpret_cast<uintptr_t>(frag.data_ptr()) % 16 == 0, "dwuv: fragment copy");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.dim() == 3 && out.size(0) == o.size(0)
              && out.size(1) == H && out.size(2) == N && out.stride(2) == 1 && out.stride(1) == N
              && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0 && out.stride(0) % 8 == 0,
              "dwuv: out must be bf16 [T, 64, 256], heads contiguous, 16-byte aligned rows");
  c10::cuda::CUDAGuard guard(o.device());
  auto stream = c10::cuda::getCurrentCUDAStream(o.get_device()).stream();
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(dwuv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM)) == cudaSuccess,
                "dwuv: shared memory attribute");
    attr = true;
  }
  const __nv_bfloat16* op = reinterpret_cast<const __nv_bfloat16*>(o.data_ptr());
  const uint8_t* fp = reinterpret_cast<const uint8_t*>(frag.data_ptr());
  __nv_bfloat16* outp = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  long long so = o.stride(0), sout = out.stride(0);
  int T = int(o.size(0)), pd = pdl ? 1 : 0, ea = early ? 1 : 0;
  const bool push = maps.numel() > 0;
  if (push)
    TORCH_CHECK(maps.is_cuda() && maps.scalar_type() == at::kLong && maps.numel() >= 4 && rank >= 0 && rank < 4
                && (loc.numel() == 0 || (loc.is_cuda() && loc.scalar_type() == at::kLong && loc.dim() == 1
                                         && loc.is_contiguous() && loc.size(0) == o.size(0))),
                "dwuv: the early push needs the scatter's ring pointers, a rank and the rows' cache slots");
  const int64_t* mp = push ? maps.data_ptr<int64_t>() : nullptr;
  const int64_t* lp = push && loc.numel() > 0 ? loc.data_ptr<int64_t>() : nullptr;
  int rk = int(rank), pu = push ? 1 : 0;
  void* args[] = {&op, &fp, &outp, &so, &sout, &T, &pd, &ea, &mp, &rk, &lp, &pu};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(CTAS); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = SMEM; cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t e = cudaLaunchKernelExC(&cfg, reinterpret_cast<const void*>(dwuv_kernel), args);
  TORCH_CHECK(e == cudaSuccess, "dwuv launch failed: ", cudaGetErrorString(e));
}

}   

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &dwuv::run); }
