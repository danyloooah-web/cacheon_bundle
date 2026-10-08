#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <vector>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace route8 {

constexpr int E = 256, K = 8, PER = E / 32;

__device__ __forceinline__ void route_row(const float* __restrict__ logits, const float (&bv)[PER], int row, int lane,
                                          float scale, int& out_id, float& out_w) {
  float sc[PER], key[PER];
  bool nz = false;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const float x = logits[size_t(row) * E + lane + 32 * i];
    nz |= x != 0.f;
    sc[i] = 1.f / (1.f + __expf(-x));
    key[i] = sc[i] + bv[i];
  }
  nz = __any_sync(0xffffffffu, nz);
  out_id = -1;
  out_w = 0.f;
  if (!nz) return;
  int my_id = -1;
  float my_score = 0.f, total = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    float best = key[0];
    int bi = 0;
#pragma unroll
    for (int i = 1; i < PER; ++i)
      if (key[i] > best) { best = key[i]; bi = i; }
    int bidx = lane + 32 * bi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bidx, o);
      if (ob > best || (ob == best && oi < bidx)) { best = ob; bidx = oi; }
    }
    const int owner = bidx & 31, slot = bidx >> 5;
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (lane == owner && i == slot) { s = sc[i]; key[i] = -INFINITY; }
    s = __shfl_sync(0xffffffffu, s, owner);
    total += s;
    if (lane == k) { my_id = bidx; my_score = s; }
  }
  if (lane < K) {
    out_id = my_id;
    out_w = my_score / total * scale;
  }
}

__device__ __forceinline__ uint32_t ord_key(float k) {
  uint32_t b = __float_as_uint(k);
  if (b == 0x80000000u) b = 0u;
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ float pos_budget(const float4 budget, int row) {
  const int p = row & 3;
  return p == 0 ? budget.x : p == 1 ? budget.y : p == 2 ? budget.z : budget.w;
}

__device__ __forceinline__ void route_row_fast(const float* __restrict__ logits, const float (&bv)[PER], int row,
                                               int lane, float scale, int& out_id, float& out_w, int nslab = 1,
                                               size_t slab = 0, bool rezero = false) {
  float sc[PER];
  uint32_t ok[PER];
  int ix[PER];
  bool nz = false;
  float xs[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) xs[i] = __ldcg(logits + size_t(row) * E + lane + 32 * i);
  constexpr int SB = 8;
  for (int s0 = 1; s0 < nslab; s0 += SB) {
    float v[SB][PER];
#pragma unroll
    for (int u = 0; u < SB; ++u)
#pragma unroll
      for (int i = 0; i < PER; ++i)
        v[u][i] = s0 + u < nslab ? __ldcg(logits + size_t(s0 + u) * slab + size_t(row) * E + lane + 32 * i) : 0.f;
#pragma unroll
    for (int u = 0; u < SB; ++u)
      if (s0 + u < nslab) {
#pragma unroll
        for (int i = 0; i < PER; ++i) xs[i] += v[u][i];
      }
  }
  if (rezero) {
#pragma unroll
    for (int i = 0; i < PER; ++i) __stcg(const_cast<float*>(logits) + size_t(row) * E + lane + 32 * i, 0.f);
  }
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const float x = xs[i];
    nz |= x != 0.f;
    sc[i] = 1.f / (1.f + __expf(-x));
    ok[i] = ord_key(sc[i] + bv[i]);
    ix[i] = lane + 32 * i;
  }
  nz = __any_sync(0xffffffffu, nz);
  out_id = -1;
  out_w = 0.f;
  if (!nz) return;
#define CSW(a, b)                                                                          \
  if (ok[b] > ok[a] || (ok[b] == ok[a] && ix[b] < ix[a])) {                                \
    const uint32_t tk = ok[a]; ok[a] = ok[b]; ok[b] = tk;                                  \
    const int ti = ix[a]; ix[a] = ix[b]; ix[b] = ti;                                       \
    const float ts = sc[a]; sc[a] = sc[b]; sc[b] = ts;                                     \
  }
  CSW(0, 1) CSW(2, 3) CSW(4, 5) CSW(6, 7)
  CSW(0, 2) CSW(1, 3) CSW(4, 6) CSW(5, 7)
  CSW(1, 2) CSW(5, 6)
  CSW(0, 4) CSW(1, 5) CSW(2, 6) CSW(3, 7)
  CSW(2, 4) CSW(3, 5)
  CSW(1, 2) CSW(3, 4) CSW(5, 6)
#undef CSW
  float my_score = 0.f, total = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const uint32_t h = ok[0];
    const uint32_t m = __reduce_max_sync(0xffffffffu, h);
    const uint32_t tie = __ballot_sync(0xffffffffu, h == m);
    int win;
    if ((tie & (tie - 1u)) == 0u) {
      win = __ffs(tie) - 1;
    } else {
      win = int(__reduce_min_sync(0xffffffffu, h == m ? uint32_t(ix[0]) : 0xffffffffu) & 31u);
    }
    const int widx = __shfl_sync(0xffffffffu, ix[0], win);
    const float s = __shfl_sync(0xffffffffu, sc[0], win);
    total += s;
    if (lane == k) { out_id = widx; my_score = s; }
    if (lane == win) {
#pragma unroll
      for (int j = 0; j < PER - 1; ++j) { ok[j] = ok[j + 1]; ix[j] = ix[j + 1]; sc[j] = sc[j + 1]; }
      ok[PER - 1] = 0u;
      ix[PER - 1] = 0x7fffffff;
    }
  }
  if (lane < K) out_w = my_score / total * scale;
}

__global__ void __launch_bounds__(128) route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                    int32_t* __restrict__ ids, float* __restrict__ w, float scale,
                                                    int rows, int pdl) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  float bv[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) bv[i] = bias[lane + 32 * i];       
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (row >= rows) return;
  int id;
  float wt;
  route_row(logits, bv, row, lane, scale, id, wt);
  if (lane < K) {
    ids[size_t(row) * K + lane] = id;
    w[size_t(row) * K + lane] = wt;
  }
}

__device__ __forceinline__ float tree8(const float (&v)[K]) {
  return __fadd_rn(__fadd_rn(__fadd_rn(v[0], v[1]), __fadd_rn(v[2], v[3])),
                   __fadd_rn(__fadd_rn(v[4], v[5]), __fadd_rn(v[6], v[7])));
}

__device__ __forceinline__ void apply_drops(int32_t* ids, float* w, int r, const float (&wt)[K], uint32_t drop,
                                            int renorm) {
  if (!renorm) {
#pragma unroll
    for (int k = 0; k < K; ++k)
      if (drop & (1u << k)) { ids[r * K + k] = -1; w[r * K + k] = 0.f; }
    return;
  }
  if (!drop) return;
  float kept[K];
#pragma unroll
  for (int k = 0; k < K; ++k) kept[k] = drop & (1u << k) ? 0.f : wt[k];
  const float ks = tree8(kept), f = ks > 0.f ? __fdiv_rn(tree8(wt), ks) : 1.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    if (drop & (1u << k)) ids[r * K + k] = -1;
    w[r * K + k] = drop & (1u << k) ? 0.f : __fmul_rn(wt[k], f);
  }
}

__global__ void __launch_bounds__(128) route_prune_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                          int32_t* ids, float* w, float scale, int rows, int pdl,
                                                          int* counts, int max_count, float4 budgets, int pair,
                                                          int renorm, int nslab, long long slab, int rezero) {
  __shared__ int last;
  __shared__ int hist[E], vcnt[E];
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  float bv[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) bv[i] = bias[lane + 32 * i];
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (row < rows) {
    int id;
    float wt;
    route_row_fast(logits, bv, row, lane, scale, id, wt, nslab, size_t(slab), rezero != 0);
    if (lane < K) {
      ids[size_t(row) * K + lane] = id;
      w[size_t(row) * K + lane] = wt;
    }
  }
  __syncthreads();
  if (gridDim.x > 1) {
    if (threadIdx.x == 0) {
      int old;
      asm volatile("atom.add.acq_rel.gpu.s32 %0, [%1], 1;" : "=r"(old) : "l"(counts) : "memory");
      last = old == int(gridDim.x) - 1;
      if (last) *counts = 0;                                    
    }
    __syncthreads();
    if (!last) return;
  }
  for (int i = threadIdx.x; i < E; i += blockDim.x) { hist[i] = 0; vcnt[i] = 0; }
  const int r0 = threadIdx.x;                                   
  int id[K];
  float wt[K];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    id[k] = r0 < rows ? __ldcg(&ids[r0 * K + k]) : -1;
    wt[k] = r0 < rows ? __ldcg(&w[r0 * K + k]) : 0.f;
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < K; ++k)
    if (id[k] >= 0) atomicAdd(&hist[id[k]], 1);
  for (int r = r0 + blockDim.x; r < rows; r += blockDim.x)
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int e = __ldcg(&ids[r * K + k]);
      if (e >= 0) atomicAdd(&hist[e], 1);
    }
  __syncthreads();
  constexpr int VROWS = 256;
  __shared__ int s_vote[VROWS];
  __shared__ float s_wt[VROWS * K];
  __shared__ uint32_t s_drop[VROWS];
  const bool defer = renorm && pair && rows <= VROWS;
  for (int r = r0; r < rows; r += blockDim.x) {
    if (r != r0)
#pragma unroll
      for (int k = 0; k < K; ++k) {
        id[k] = __ldcg(&ids[r * K + k]);
        wt[k] = __ldcg(&w[r * K + k]);
      }
    bool el[K];
#pragma unroll
    for (int k = 0; k < K; ++k) el[k] = id[k] >= 0 && hist[id[k]] <= max_count;
    const float budget = pos_budget(budgets, r);
    float used = 0.f;
    uint32_t drop = 0u;
#pragma unroll 1
    for (int it = 0; it < K; ++it) {
      int b = -1;
      float bw = INFINITY;
#pragma unroll
      for (int k = 0; k < K; ++k)
        if (el[k] && wt[k] < bw) { bw = wt[k]; b = k; }
      if (b < 0 || used + bw > budget) break;
      used += bw;
#pragma unroll
      for (int k = 0; k < K; ++k)
        if (k == b) el[k] = false;
      drop |= 1u << b;
    }
    if (defer) {
      s_drop[r] = drop;
#pragma unroll
      for (int k = 0; k < K; ++k) s_wt[r * K + k] = wt[k];
    } else {
      apply_drops(ids, w, r, wt, drop, renorm);
    }
    if (pair && r < VROWS) {
      int v = -1;
      float vw = INFINITY;
#pragma unroll
      for (int k = 0; k < K; ++k)
        if (id[k] >= 0 && hist[id[k]] == 2 && used + wt[k] <= budget && wt[k] < vw) { vw = wt[k]; v = id[k] * K + k; }
      s_vote[r] = v;                                            
      if (v >= 0) atomicAdd(&vcnt[v / K], 1);
    }
  }
  if (!pair || rows > VROWS) return;
  __syncthreads();
  for (int r = threadIdx.x; r < rows; r += blockDim.x) {
    const int v = s_vote[r];
    const bool both = v >= 0 && vcnt[v / K] == 2;
    if (!defer) {
      if (both) { ids[r * K + v % K] = -1; w[r * K + v % K] = 0.f; }
      continue;
    }
    float wt[K];
#pragma unroll
    for (int k = 0; k < K; ++k) wt[k] = s_wt[r * K + k];
    apply_drops(ids, w, r, wt, s_drop[r] | (both ? 1u << (v % K) : 0u), 1);
  }
}

static float4 pos_budgets(double budget, double scale, const std::vector<double>& pos_mult) {
  TORCH_CHECK(pos_mult.size() == 4, "pos_mult: one multiplier per verify position (4)");
  for (const double m : pos_mult) TORCH_CHECK(std::isfinite(m) && m >= 0.0, "pos_mult: finite and >= 0");
  return make_float4(float(budget * pos_mult[0] * scale), float(budget * pos_mult[1] * scale),
                     float(budget * pos_mult[2] * scale), float(budget * pos_mult[3] * scale));
}

void route_prune(torch::Tensor logits, torch::Tensor bias, torch::Tensor ids, torch::Tensor w, double scale, bool pdl,
                 torch::Tensor counts, int64_t max_count, double budget, bool pair, std::vector<double> pos_mult,
                 bool renorm, bool rezero) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && (logits.dim() == 2 || logits.dim() == 3)
              && logits.size(-1) == E && logits.is_contiguous(), "logits");
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kFloat && bias.numel() == E && bias.is_contiguous(), "bias");
  const int rows = logits.size(-2);
  int ns = logits.dim() == 3 ? int(logits.size(0)) : 1;
  long long sl = (long long)rows * E;
  TORCH_CHECK(ids.scalar_type() == at::kInt && ids.numel() == int64_t(rows) * K && ids.is_contiguous(), "ids");
  TORCH_CHECK(w.scalar_type() == at::kFloat && w.numel() == int64_t(rows) * K && w.is_contiguous(), "w");
  TORCH_CHECK(counts.scalar_type() == at::kInt && counts.numel() >= 1 && counts.is_contiguous(), "counts");
  if (rows == 0) return;
  c10::cuda::CUDAGuard guard(logits.device());
  auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  const float* lp = logits.data_ptr<float>(); const float* bp = bias.data_ptr<float>();
  int32_t* ip = ids.data_ptr<int32_t>(); float* wp = w.data_ptr<float>();
  int* cp = counts.data_ptr<int>();
  float sc = float(scale);
  float4 ta = pos_budgets(budget, scale, pos_mult);
  int r = rows, pd = pdl ? 1 : 0, mc = int(max_count), pr = pair ? 1 : 0, rn = renorm ? 1 : 0;
  cudaLaunchConfig_t cfg = {};
  cfg.dynamicSmemBytes = 0; cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  int na = 0;
  if (pdl) {
    attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[na++].val.programmaticStreamSerializationAllowed = 1;
  }
  cfg.gridDim = dim3((rows + 3) / 4); cfg.blockDim = dim3(128); cfg.attrs = attr; cfg.numAttrs = na;
  TORCH_CHECK(!rezero || logits.dim() == 2, "rezero takes the [rows, E] atomic-sum buffer");
  int rz = rezero ? 1 : 0;
  void* args[] = {&lp, &bp, &ip, &wp, &sc, &r, &pd, &cp, &mc, &ta, &pr, &rn, &ns, &sl, &rz};
  const cudaError_t err = cudaLaunchKernelExC(&cfg, reinterpret_cast<const void*>(route_prune_kernel), args);
  TORCH_CHECK(err == cudaSuccess, "route8 prune launch failed: ", cudaGetErrorString(err));
}

void route(torch::Tensor logits, torch::Tensor bias, torch::Tensor ids, torch::Tensor w, double scale, bool pdl) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.dim() == 2 && logits.size(1) == E
              && logits.is_contiguous(), "logits");
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kFloat && bias.numel() == E && bias.is_contiguous(), "bias");
  const int rows = logits.size(0);
  TORCH_CHECK(ids.scalar_type() == at::kInt && ids.numel() == int64_t(rows) * K && ids.is_contiguous(), "ids");
  TORCH_CHECK(w.scalar_type() == at::kFloat && w.numel() == int64_t(rows) * K && w.is_contiguous(), "w");
  c10::cuda::CUDAGuard guard(logits.device());
  auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  const float* lp = logits.data_ptr<float>(); const float* bp = bias.data_ptr<float>();
  int32_t* ip = ids.data_ptr<int32_t>(); float* wp = w.data_ptr<float>();
  float sc = float(scale);
  int r = rows, pd = pdl ? 1 : 0;
  void* args[] = {&lp, &bp, &ip, &wp, &sc, &r, &pd};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((rows + 3) / 4); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = 0; cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr; cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t err = cudaLaunchKernelExC(&cfg, reinterpret_cast<const void*>(route_kernel), args);
  TORCH_CHECK(err == cudaSuccess, "route8 launch failed: ", cudaGetErrorString(err));
}

}   

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("route", &route8::route);
  m.def("route_prune", &route8::route_prune, py::arg("logits"), py::arg("bias"), py::arg("ids"), py::arg("w"),
        py::arg("scale"), py::arg("pdl"), py::arg("counts"), py::arg("max_count"), py::arg("budget"), py::arg("pair"),
        py::arg("pos_mult") = std::vector<double>{1.0, 1.0, 1.0, 1.0}, py::arg("renorm") = false,
        py::arg("rezero") = false);
}
