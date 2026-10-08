#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <vector>

namespace pp {

constexpr int H = 6144;               
constexpr int BC = 2048;              
constexpr int NCB = H / BC;           
constexpr int K = 8;                  
constexpr int CW = 8;                 
constexpr int NT = (CW + 1) * 32;     
constexpr int SEG = BC * 2;           
constexpr int STAGE = K * SEG;        
constexpr int STAGES = 6;
constexpr size_t SMEM = size_t(STAGES) * STAGE + STAGES * K * sizeof(float) + STAGES * sizeof(uint32_t)
                        + 2 * STAGES * sizeof(uint64_t);

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile("{\n"
               " .reg .pred P;\n"
               "PP_WAIT:\n"
               " mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
               " @!P bra PP_WAIT;\n"
               "}\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}

__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar,
                                         uint64_t policy) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1], %2, [%3], %4;"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy) : "memory");
}

__device__ __forceinline__ void st_global_v4(uint64_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
               :: "l"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const uint32_t a = __bfloat16_as_ushort(__float2bfloat16_rn(lo));
  const uint32_t b = __bfloat16_as_ushort(__float2bfloat16_rn(hi));
  return a | (b << 16);
}

__device__ __forceinline__ int owner_rows(int4 rows_of, int owner) {
  return owner == 0 ? rows_of.x : owner == 1 ? rows_of.y : owner == 2 ? rows_of.z : rows_of.w;
}

__device__ __forceinline__ void decode(int g, int rank, int& owner, int& j, int& cb) {
  cb = g % NCB;
  const int q = g / NCB;
  owner = (rank + 1 + (q & 3)) & 3;
  j = q >> 2;
}

__global__ void __launch_bounds__(NT, 1)
push_kernel(const unsigned short* __restrict__ gemm2, const int* __restrict__ map,
            const unsigned short* __restrict__ wts, uint64_t b0, uint64_t b1, uint64_t b2, uint64_t b3,
            int rank, int cap, int tiles, int parity, uint64_t slot_elems, uint64_t cnt_off,
            int4 rows_of) {
  extern __shared__ __align__(1024) char smem[];
  char* data = smem;
  float* wsm = reinterpret_cast<float*>(smem + size_t(STAGES) * STAGE);
  uint32_t* vm = reinterpret_cast<uint32_t*>(wsm + STAGES * K);
  uint64_t* full = reinterpret_cast<uint64_t*>(vm + STAGES);
  uint64_t* empty = full + STAGES;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], CW);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == CW) {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    int idx_n = -1;
    unsigned short w_n = 0;
    if (static_cast<int>(blockIdx.x) < tiles && lane < K) {
      int owner, j, cb;
      decode(blockIdx.x, rank, owner, j, cb);
      const size_t t = size_t(owner) * cap + j;
      const bool live = j < owner_rows(rows_of, owner);
      idx_n = live ? __ldg(map + t * K + lane) : -1;
      w_n = live ? __ldg(wts + t * K + lane) : 0;
    }
    int i = 0;
    for (int g = blockIdx.x; g < tiles; g += gridDim.x, ++i) {
      const int idx = idx_n;
      const unsigned short wb = w_n;
      const int gn = g + static_cast<int>(gridDim.x);
      if (gn < tiles && lane < K) {
        int owner, j, cb;
        decode(gn, rank, owner, j, cb);
        const size_t t = size_t(owner) * cap + j;
        const bool live = j < owner_rows(rows_of, owner);
        idx_n = live ? __ldg(map + t * K + lane) : -1;
        w_n = live ? __ldg(wts + t * K + lane) : 0;
      }
      const int s = i % STAGES;
      const uint32_t round = static_cast<uint32_t>(i / STAGES);
      if (i >= STAGES) mbar_wait(&empty[s], (round - 1) & 1u);
      int owner, j, cb;
      decode(g, rank, owner, j, cb);
      const bool valid = lane < K && idx >= 0;
      if (lane < K) wsm[s * K + lane] = valid ? __uint_as_float(uint32_t(wb) << 16) : 0.f;
      const uint32_t vmask = __ballot_sync(0xffffffffu, valid);
      if (lane == 0) vm[s] = vmask;
      __syncwarp();
      if (lane == 0) mbar_arrive_expect_tx(&full[s], static_cast<uint32_t>(__popc(vmask)) * SEG);
      __syncwarp();
      if (valid)
        bulk_g2s(data + size_t(s) * STAGE + size_t(lane) * SEG,
                 gemm2 + size_t(idx) * H + size_t(cb) * BC, SEG, &full[s], policy);
    }
  } else {
    const int tc = threadIdx.x;
    int i = 0;
    for (int g = blockIdx.x; g < tiles; g += gridDim.x, ++i) {
      const int s = i % STAGES;
      mbar_wait(&full[s], static_cast<uint32_t>(i / STAGES) & 1u);
      int owner, j, cb;
      decode(g, rank, owner, j, cb);
      const uint32_t vmask = vm[s];
      float acc[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = 0.f;
      const char* stg = data + size_t(s) * STAGE + size_t(tc) * 16;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        if (vmask & (1u << k)) {
          const float w = wsm[s * K + k];
          const uint4 d = *reinterpret_cast<const uint4*>(stg + size_t(k) * SEG);
          acc[0] = fmaf(w, bf16_lo(d.x), acc[0]);
          acc[1] = fmaf(w, bf16_hi(d.x), acc[1]);
          acc[2] = fmaf(w, bf16_lo(d.y), acc[2]);
          acc[3] = fmaf(w, bf16_hi(d.y), acc[3]);
          acc[4] = fmaf(w, bf16_lo(d.z), acc[4]);
          acc[5] = fmaf(w, bf16_hi(d.z), acc[5]);
          acc[6] = fmaf(w, bf16_lo(d.w), acc[6]);
          acc[7] = fmaf(w, bf16_hi(d.w), acc[7]);
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
      if (j < owner_rows(rows_of, owner)) {
        const uint64_t base = owner == 0 ? b0 : owner == 1 ? b1 : owner == 2 ? b2 : b3;
        const uint64_t elem = uint64_t(parity * 4 + rank) * slot_elems + uint64_t(j) * H
                              + uint64_t(cb) * BC + uint64_t(tc) * 8;
        st_global_v4(base + 2 * elem, pack_bf16(acc[0], acc[1]), pack_bf16(acc[2], acc[3]),
                     pack_bf16(acc[4], acc[5]), pack_bf16(acc[6], acc[7]));
      }
    }
    asm volatile("fence.acq_rel.sys;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint64_t one = 1;
    asm volatile("red.release.sys.global.add.u64 [%0], %1;" :: "l"(b0 + cnt_off), "l"(one) : "memory");
    asm volatile("red.release.sys.global.add.u64 [%0], %1;" :: "l"(b1 + cnt_off), "l"(one) : "memory");
    asm volatile("red.release.sys.global.add.u64 [%0], %1;" :: "l"(b2 + cnt_off), "l"(one) : "memory");
    asm volatile("red.release.sys.global.add.u64 [%0], %1;" :: "l"(b3 + cnt_off), "l"(one) : "memory");
  }
}

}   

int64_t push(torch::Tensor gemm2, torch::Tensor map, torch::Tensor weights, std::vector<int64_t> peers,
             int64_t rank, int64_t cap, int64_t parity, int64_t cnt_off, int64_t cap_max, int64_t grid,
             std::vector<int64_t> rows) {
  TORCH_CHECK(gemm2.is_cuda() && gemm2.scalar_type() == at::kBFloat16 && gemm2.dim() == 2
              && gemm2.size(1) == pp::H && gemm2.stride(1) == 1 && gemm2.stride(0) == pp::H,
              "prefill_push: gemm2 must be contiguous bf16 [rows, 6144]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(gemm2.data_ptr()) % 16 == 0, "prefill_push: gemm2 alignment");
  TORCH_CHECK(map.is_cuda() && map.scalar_type() == at::kInt && map.is_contiguous()
              && map.numel() >= 4 * cap * pp::K, "prefill_push: map must be contiguous int32 [>= 32 cap]");
  TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == at::kBFloat16 && weights.is_contiguous()
              && weights.numel() >= 4 * cap * pp::K, "prefill_push: weights must be contiguous bf16 [>= 4 cap, 8]");
  TORCH_CHECK(peers.size() == 4, "prefill_push: four peer bases");
  TORCH_CHECK(rows.size() == 4, "prefill_push: four owner row counts");
  for (int64_t r : rows) TORCH_CHECK(r >= 0 && r <= cap, "prefill_push: owner rows must be in [0, cap]");
  for (int64_t p : peers) TORCH_CHECK(p % 256 == 0, "prefill_push: peer base alignment");
  TORCH_CHECK(rank >= 0 && rank < 4 && cap > 0 && cap <= cap_max && (parity == 0 || parity == 1)
              && cnt_off % 8 == 0 && grid > 0, "prefill_push: arguments");
  const int dev = gemm2.get_device();
  static int configured[64] = {0};
  if (!configured[dev]) {
    TORCH_CHECK(cudaFuncSetAttribute(pp::push_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(pp::SMEM)) == cudaSuccess,
                "prefill_push: shared-memory attribute");
    configured[dev] = 1;
  }
  const int tiles = static_cast<int>(4 * cap * pp::NCB);
  auto stream = c10::cuda::getCurrentCUDAStream(dev).stream();
  pp::push_kernel<<<static_cast<unsigned>(grid), pp::NT, pp::SMEM, stream>>>(
      reinterpret_cast<const unsigned short*>(gemm2.data_ptr()), map.data_ptr<int>(),
      reinterpret_cast<const unsigned short*>(weights.data_ptr()),
      static_cast<uint64_t>(peers[0]), static_cast<uint64_t>(peers[1]),
      static_cast<uint64_t>(peers[2]), static_cast<uint64_t>(peers[3]),
      static_cast<int>(rank), static_cast<int>(cap), tiles, static_cast<int>(parity),
      static_cast<uint64_t>(cap_max) * pp::H, static_cast<uint64_t>(cnt_off),
      make_int4(static_cast<int>(rows[0]), static_cast<int>(rows[1]), static_cast<int>(rows[2]),
                static_cast<int>(rows[3])));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "prefill_push: launch failed");
  return tiles;
}

int64_t smem_bytes() { return static_cast<int64_t>(pp::SMEM); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("push", &push, "Prefill MoE push: TMA-gather finalize into the owners' symmetric slots");
  m.def("smem_bytes", &smem_bytes, "Shared memory one push CTA takes");
}
