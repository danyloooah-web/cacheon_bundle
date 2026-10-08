#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

namespace {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void fill_ring(char* smem, const char* __restrict__ p, uint64_t bytes, uint32_t chunk,
                                          int depth, uint32_t bidx, uint32_t nblk) {
  uint64_t* bar = reinterpret_cast<uint64_t*>(smem + size_t(depth) * chunk);
  for (int k = 0; k < depth; ++k)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(&bar[k])) : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  const uint64_t stride = uint64_t(nblk) * chunk;
  uint64_t cur = uint64_t(bidx) * chunk;
  uint64_t nxt = cur + uint64_t(depth) * stride;
  for (int k = 0; k < depth; ++k) {
    const uint64_t b = cur + uint64_t(k) * stride;
    if (b < bytes) {
      const uint32_t n = static_cast<uint32_t>((bytes - b) < uint64_t(chunk) ? (bytes - b) : uint64_t(chunk));
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                   :: "r"(smem_u32(&bar[k])), "r"(n) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(smem_u32(smem + size_t(k) * chunk)), "l"(p + b), "r"(n), "r"(smem_u32(&bar[k]))
                   : "memory");
    }
  }
  uint32_t phase = 0;
  int slot = 0;
  while (cur < bytes) {
    asm volatile("{ .reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W; }"
                 :: "r"(smem_u32(&bar[slot])), "r"(phase) : "memory");
    if (nxt < bytes) {
      const uint32_t n = static_cast<uint32_t>((bytes - nxt) < uint64_t(chunk) ? (bytes - nxt) : uint64_t(chunk));
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                   :: "r"(smem_u32(&bar[slot])), "r"(n) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(smem_u32(smem + size_t(slot) * chunk)), "l"(p + nxt), "r"(n),
                      "r"(smem_u32(&bar[slot])) : "memory");
    }
    cur += stride;
    nxt += stride;
    if (++slot == depth) { slot = 0; phase ^= 1u; }
  }
}

__global__ void l2_fill_kernel(const char* __restrict__ p, uint64_t bytes, uint32_t chunk, int depth) {
  extern __shared__ __align__(1024) char smem[];
  if (threadIdx.x != 0) return;
  fill_ring(smem, p, bytes, chunk, depth, blockIdx.x, gridDim.x);
}

__global__ void l2_fill2_kernel(const char* __restrict__ p0, uint64_t bytes0, const char* __restrict__ p1,
                                uint64_t bytes1, int split, uint32_t chunk, int depth) {
  extern __shared__ __align__(1024) char smem[];
  if (threadIdx.x != 0) return;
  if (int(blockIdx.x) < split)
    fill_ring(smem, p0, bytes0, chunk, depth, blockIdx.x, uint32_t(split));
  else
    fill_ring(smem, p1, bytes1, chunk, depth, blockIdx.x - uint32_t(split), gridDim.x - uint32_t(split));
}

__global__ void delay_kernel(uint64_t ns) {
  uint64_t t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}

}   

void delay(int64_t ns) {
  TORCH_CHECK(ns >= 0, "l2_fill.delay: negative delay");
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  delay_kernel<<<1, 32, 0, stream>>>(static_cast<uint64_t>(ns));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "l2_fill.delay: kernel launch failed");
}

int64_t fill(torch::Tensor t, int64_t blocks, int64_t chunk, int64_t depth) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "l2_fill: needs a contiguous CUDA tensor");
  TORCH_CHECK(blocks > 0 && chunk > 0 && chunk % 16 == 0 && depth > 0, "l2_fill: bad launch shape");
  const uint64_t bytes = uint64_t(t.numel()) * uint64_t(t.element_size());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 && bytes % 16 == 0,
              "l2_fill: TMA bulk copies need 16-byte alignment and size");
  const size_t smem = size_t(depth) * size_t(chunk) + size_t(depth) * sizeof(uint64_t);
  TORCH_CHECK(smem <= 227 * 1024, "l2_fill: ring exceeds shared memory");
  static int configured_device = -1;
  if (configured_device != t.device().index()) {
    TORCH_CHECK(cudaFuncSetAttribute(l2_fill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     227 * 1024) == cudaSuccess, "l2_fill: smem attribute");
    configured_device = t.device().index();
  }
  auto stream = c10::cuda::getCurrentCUDAStream(t.device().index()).stream();
  l2_fill_kernel<<<static_cast<unsigned>(blocks), 32, smem, stream>>>(
      reinterpret_cast<const char*>(t.data_ptr()), bytes, static_cast<uint32_t>(chunk),
      static_cast<int>(depth));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "l2_fill: kernel launch failed");
  return static_cast<int64_t>(bytes);
}

int64_t fill2(torch::Tensor t0, torch::Tensor t1, int64_t blocks0, int64_t blocks1, int64_t chunk, int64_t depth) {
  TORCH_CHECK(t0.is_cuda() && t0.is_contiguous() && t1.is_cuda() && t1.is_contiguous()
              && t0.device() == t1.device(), "l2_fill2: needs two contiguous CUDA tensors on one device");
  TORCH_CHECK(blocks0 > 0 && blocks1 > 0 && chunk > 0 && chunk % 16 == 0 && depth > 0, "l2_fill2: bad launch shape");
  const uint64_t bytes0 = uint64_t(t0.numel()) * uint64_t(t0.element_size());
  const uint64_t bytes1 = uint64_t(t1.numel()) * uint64_t(t1.element_size());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t0.data_ptr()) % 16 == 0 && bytes0 % 16 == 0
              && reinterpret_cast<uintptr_t>(t1.data_ptr()) % 16 == 0 && bytes1 % 16 == 0,
              "l2_fill2: TMA bulk copies need 16-byte alignment and size");
  const size_t smem = size_t(depth) * size_t(chunk) + size_t(depth) * sizeof(uint64_t);
  TORCH_CHECK(smem <= 227 * 1024, "l2_fill2: ring exceeds shared memory");
  static int configured_device = -1;
  if (configured_device != t0.device().index()) {
    TORCH_CHECK(cudaFuncSetAttribute(l2_fill2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     227 * 1024) == cudaSuccess, "l2_fill2: smem attribute");
    configured_device = t0.device().index();
  }
  auto stream = c10::cuda::getCurrentCUDAStream(t0.device().index()).stream();
  l2_fill2_kernel<<<static_cast<unsigned>(blocks0 + blocks1), 32, smem, stream>>>(
      reinterpret_cast<const char*>(t0.data_ptr()), bytes0, reinterpret_cast<const char*>(t1.data_ptr()), bytes1,
      static_cast<int>(blocks0), static_cast<uint32_t>(chunk), static_cast<int>(depth));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "l2_fill2: kernel launch failed");
  return static_cast<int64_t>(bytes0 + bytes1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fill", &fill, "Warm a contiguous CUDA tensor into L2 with discarded TMA bulk copies");
  m.def("fill2", &fill2, "Warm two contiguous CUDA tensors into L2 in one launch");
  m.def("delay", &delay, "Hold the current stream for ns nanoseconds without touching memory");
}
