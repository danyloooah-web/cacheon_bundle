
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "litedsa_attention_sm100.cuh"

namespace {

constexpr int kTopK = 2048;
constexpr int kCap = 2 * kTopK;           
constexpr int kHash = 4 * kTopK;          
constexpr int kThreads = 256;
constexpr int kPerThread = kTopK / kThreads;

__device__ __forceinline__ unsigned slot_of(int id) {
  return (static_cast<unsigned>(id) * 2654435761u) >> 19;   
}

__device__ __forceinline__ void insert(int* set, int id) {
  unsigned s = slot_of(id);
  while (true) {
    const int old = atomicCAS(&set[s], -1, id);
    if (old == -1 || old == id) return;
    s = (s + 1) & (kHash - 1);
  }
}

__device__ __forceinline__ bool contains(const int* set, int id) {
  unsigned s = slot_of(id);
  while (true) {
    const int v = set[s];
    if (v == id) return true;
    if (v == -1) return false;
    s = (s + 1) & (kHash - 1);
  }
}

__global__ void __launch_bounds__(kThreads) pair_union_kernel(
    const int* __restrict__ tables, long ld, const int* __restrict__ lens,
    int* __restrict__ u, int* __restrict__ counts, uint32_t* __restrict__ memb) {
  extern __shared__ int dyn_smem[];
  int* set_a = dyn_smem;                
  int* set_b = dyn_smem + kHash;        
  __shared__ uint8_t bytes_a[kTopK / 8];    
  __shared__ uint8_t bytes_b[kTopK / 8];
  __shared__ int warp_sums[kThreads / 32 + 1];
  const int p = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int la = min(max(lens[2 * p], 0), kTopK);
  const int lb = min(max(lens[2 * p + 1], 0), kTopK);
  const int4* a4 = reinterpret_cast<const int4*>(tables + (2L * p) * ld) + 2 * tid;
  const int4* b4 = reinterpret_cast<const int4*>(tables + (2L * p + 1) * ld) + 2 * tid;
  const int4 a0 = a4[0], a1 = a4[1], b0 = b4[0], b1 = b4[1];
  int ra[kPerThread] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
  int rb[kPerThread] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
  const int e0 = tid * kPerThread;
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    if (e0 + j >= la) ra[j] = -1;               
    if (e0 + j >= lb) rb[j] = -1;
  }
  for (int i = tid; i < kHash; i += kThreads) { set_a[i] = -1; set_b[i] = -1; }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    if (ra[j] >= 0) insert(set_a, ra[j]);
    if (rb[j] >= 0) insert(set_b, rb[j]);
  }
  __syncthreads();

  unsigned extra = 0, mask_a = 0, mask_b = 0;
  int mine = 0;
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    if (rb[j] >= 0 && !contains(set_a, rb[j])) { extra |= 1u << j; ++mine; }
    if (ra[j] >= 0) { mask_a |= 1u << j; if (contains(set_b, ra[j])) mask_b |= 1u << j; }
  }
  bytes_a[tid] = static_cast<uint8_t>(mask_a);
  bytes_b[tid] = static_cast<uint8_t>(mask_b);
  int incl = mine;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, off);
    if (lane >= off) incl += v;
  }
  if (lane == 31) warp_sums[warp + 1] = incl;
  if (tid == 0) warp_sums[0] = 0;
  __syncthreads();
  if (tid == 0)
    for (int w = 1; w <= kThreads / 32; ++w) warp_sums[w] += warp_sums[w - 1];
  __syncthreads();
  const int total = la + warp_sums[kThreads / 32];
  int* up = u + static_cast<long>(p) * kCap;
  if (e0 + kPerThread <= la) {
    int4* dst = reinterpret_cast<int4*>(up + e0);
    dst[0] = make_int4(ra[0], ra[1], ra[2], ra[3]);
    dst[1] = make_int4(ra[4], ra[5], ra[6], ra[7]);
  } else {
#pragma unroll
    for (int j = 0; j < kPerThread; ++j)
      if (e0 + j < la) up[e0 + j] = ra[j];
  }
  int pos = la + warp_sums[warp] + incl - mine;
#pragma unroll
  for (int j = 0; j < kPerThread; ++j)
    if (extra >> j & 1u) up[pos++] = rb[j];
  if (tid == 0) counts[p] = total;

  uint32_t* ma = memb + static_cast<long>(p) * 2 * (kCap / 32);
  uint32_t* mb = ma + kCap / 32;
  const uint32_t* wa = reinterpret_cast<const uint32_t*>(bytes_a);
  const uint32_t* wb = reinterpret_cast<const uint32_t*>(bytes_b);
  for (int w = tid; w < kCap / 32; w += kThreads) {
    const int lo = w * 32;
    uint32_t extras = 0;
    const int from = max(la, lo), to = min(total, lo + 32);
    if (to > from) {
      const int n = to - from;
      extras = (n == 32 ? 0xffffffffu : ((1u << n) - 1u)) << (from - lo);
    }
    const bool in_list = w < kTopK / 32;
    ma[w] = in_list ? wa[w] : 0u;
    mb[w] = (in_list ? wb[w] : 0u) | extras;
  }
}

void check_cuda(const torch::Tensor& t, const char* name, at::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has dtype ", t.scalar_type());
}

}   

void pair_union(torch::Tensor tables, torch::Tensor lens, torch::Tensor u, torch::Tensor counts,
                torch::Tensor memb) {
  check_cuda(tables, "tables", at::kInt);
  check_cuda(lens, "lens", at::kInt);
  check_cuda(u, "u", at::kInt);
  check_cuda(counts, "counts", at::kInt);
  check_cuda(memb, "memb", at::kInt);
  TORCH_CHECK(tables.dim() == 2 && tables.stride(1) == 1 && tables.size(1) >= kTopK, "tables must be [2P, >=2048]");
  TORCH_CHECK(tables.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(tables.data_ptr()) % 16 == 0,
              "tables rows must be 16-byte aligned");
  const long rows = tables.size(0);
  TORCH_CHECK(rows % 2 == 0 && rows >= 2, "tables needs an even number of rows");
  const long pairs = rows / 2;
  TORCH_CHECK(lens.numel() == rows && lens.is_contiguous(), "lens must be [2P]");
  TORCH_CHECK(u.is_contiguous() && u.numel() == pairs * kCap, "u must be [P, 4096]");
  TORCH_CHECK(counts.is_contiguous() && counts.numel() == pairs, "counts must be [P]");
  TORCH_CHECK(memb.is_contiguous() && memb.numel() == pairs * 2 * (kCap / 32), "memb must be [P, 2, 128]");
  const c10::cuda::CUDAGuard guard(tables.device());
  constexpr int kDynSmem = 2 * kHash * static_cast<int>(sizeof(int));
  static bool attr_set = false;
  if (!attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(pair_union_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynSmem));
    attr_set = true;
  }
  pair_union_kernel<<<static_cast<unsigned>(pairs), kThreads, kDynSmem, at::cuda::getCurrentCUDAStream()>>>(
      tables.data_ptr<int>(), tables.stride(0), lens.data_ptr<int>(), u.data_ptr<int>(),
      counts.data_ptr<int>(), reinterpret_cast<uint32_t*>(memb.data_ptr<int>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void masked_mla(torch::Tensor q, torch::Tensor kv, torch::Tensor u, double sm_scale, double out_scale,
                torch::Tensor counts, torch::Tensor memb, torch::Tensor out, torch::Tensor max_logits,
                torch::Tensor lse) {
  check_cuda(q, "q", at::kFloat8_e4m3fn);
  check_cuda(kv, "kv", at::kFloat8_e4m3fn);
  check_cuda(u, "u", at::kInt);
  check_cuda(counts, "counts", at::kInt);
  check_cuda(memb, "memb", at::kInt);
  check_cuda(out, "out", at::kBFloat16);
  check_cuda(max_logits, "max_logits", at::kFloat);
  check_cuda(lse, "lse", at::kFloat);
  TORCH_CHECK(q.dim() == 3 && q.size(1) == 128 && q.size(2) == 576 && q.stride(2) == 1, "q must be [P, 128, 576]");
  TORCH_CHECK(kv.dim() == 3 && kv.size(1) == 1 && kv.size(2) == 576 && kv.stride(2) == 1, "kv must be [S, 1, 576]");
  const long pairs = q.size(0);
  TORCH_CHECK(u.is_contiguous() && u.numel() == pairs * kCap, "u must be [P, 4096]");
  TORCH_CHECK(counts.is_contiguous() && counts.numel() == pairs, "counts must be [P]");
  TORCH_CHECK(memb.is_contiguous() && memb.numel() == pairs * 2 * (kCap / 32), "memb must be [P, 2, 128]");
  TORCH_CHECK(out.is_contiguous() && out.numel() == pairs * 128 * 512, "out must be [P, 128, 512]");
  TORCH_CHECK(max_logits.is_contiguous() && lse.is_contiguous() && max_logits.numel() == pairs * 128 &&
              lse.numel() == pairs * 128, "max_logits / lse must be [P, 128]");
  const c10::cuda::CUDAGuard guard(q.device());
  int sms = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, q.get_device()));
  constexpr float kLog2e = 1.4426950408889634f;
  SparseAttnFwdParams params = {
      static_cast<int>(pairs),
      static_cast<int>(kv.size(0)),
      128,
      1,
      576,
      512,
      kCap,
      static_cast<float>(sm_scale),
      static_cast<float>(sm_scale) * kLog2e,
      reinterpret_cast<cutlass::bfloat16_t*>(q.data_ptr()),
      reinterpret_cast<cutlass::bfloat16_t*>(kv.data_ptr()),
      u.data_ptr<int>(),
      nullptr,
      counts.data_ptr<int>(),
      static_cast<int>(q.stride(0)),
      static_cast<int>(q.stride(1)),
      static_cast<int>(kv.stride(0)),
      static_cast<int>(kv.stride(1)),
      kCap,
      kCap,
      reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()),
      max_logits.data_ptr<float>(),
      lse.data_ptr<float>(),
      sms,
      at::cuda::getCurrentCUDAStream()};
  params.membership = nullptr;
  params.h_per_q = 64;
  params.out_scale = static_cast<float>(out_scale);
  params.topk_length_per_q = nullptr;
  params.q_group_div = 1;
  params.membership_qm = reinterpret_cast<const uint32_t*>(memb.data_ptr<int>());
  sm100::fwd::head128_fp8::run_fwd_phase1_kernel<576>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pair_union", &pair_union, "union + membership of adjacent token pairs' top-k slots");
  m.def("masked_mla", &masked_mla, "LiteDSA FP8 masked sparse MLA over token-pair unions");
}
