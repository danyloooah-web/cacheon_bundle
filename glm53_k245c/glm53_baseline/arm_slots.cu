#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void arm_slots_kernel(int* __restrict__ words, long long n, int sentinel) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (; i < n; i += stride) words[i] = sentinel;
}

int64_t arm_slots(torch::Tensor slots, int64_t sentinel) {
  TORCH_CHECK(slots.is_cuda(), "arm_slots: slots must be a CUDA tensor");
  TORCH_CHECK(slots.scalar_type() == torch::kInt32, "arm_slots: slots must be int32");
  TORCH_CHECK(slots.is_contiguous(), "arm_slots: slots must be contiguous");
  const long long n = static_cast<long long>(slots.numel());
  if (n == 0) return 0;
  auto stream = c10::cuda::getCurrentCUDAStream(slots.device().index()).stream();
  const int threads = 256;
  long long blocks = (n + threads - 1) / threads;
  if (blocks > 4096) blocks = 4096;
  arm_slots_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      slots.data_ptr<int32_t>(), n, static_cast<int>(sentinel));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "arm_slots: kernel launch failed");
  return n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("arm_slots", &arm_slots, "Poison a rank's symmetric slot ring with the Lamport sentinel");
}
