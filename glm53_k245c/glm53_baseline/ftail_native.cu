#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace ftail {

constexpr int HID = 6144;
constexpr int VPR = HID / 8;           
constexpr int WARPS = VPR / 32;        
constexpr int TOPK = 8;
constexpr uint32_t SENT = 0x80000000u;

struct Args {
  const char* g2;             
  const int* map;             
  const void* w;              
  const char* shared;         
  const char* res;            
  const char* gamma;          
  char* out;                  
  char* normed;
  char* resout;
  char* peer[4];              
  int* phase;                 
  long long g2_stride;        
  long long slot_bytes, chunk_bytes;
  float eps;
  int rank, chunk_rows, trigger, wbf16;
};

__device__ __forceinline__ uint4 ld16(const char* p) {
  uint4 v;
  asm volatile("ld.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ld16_volatile(const char* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void st16(char* p, const uint4 v) {
  asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void st16_volatile(char* p, const uint4 v) {
  asm volatile("st.volatile.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ float lo(uint32_t word) { return __uint_as_float(word << 16); }
__device__ __forceinline__ float hi(uint32_t word) { return __uint_as_float(word & 0xffff0000u); }
__device__ __forceinline__ uint32_t pack(float l, float h) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(l)))
         | (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(h))) << 16);
}
__device__ __forceinline__ uint32_t sanitize(uint32_t word) { return word == SENT ? 0u : word; }
__device__ __forceinline__ bool poisoned(const uint4 v) {
  return (v.x == SENT) | (v.y == SENT) | (v.z == SENT) | (v.w == SENT);
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <bool NORM, int CL>
__global__ void __launch_bounds__(VPR / CL) tail_kernel(const Args a) {
  __shared__ float parts[32];
  const int cta = blockIdx.x, cl = cta / CL, part_id = cta - cl * CL;
  const int lane = part_id * (VPR / CL) + threadIdx.x;          
  const int j = cl / a.chunk_rows, row = cl - j * a.chunk_rows;
  const int dest = (a.rank + 1 + j) & 3;                       
  const int t = dest * a.chunk_rows + row;
  const long long col = static_cast<long long>(lane) * 16;
  const long long voff = (static_cast<long long>(row) * VPR + lane) * 16;

  const int phase = a.phase[cta];
  int m[TOPK];
  float wk[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    m[k] = a.map[t * TOPK + k];
    wk[k] = a.wbf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(a.w)[t * TOPK + k])
                    : static_cast<const float*>(a.w)[t * TOPK + k];
  }
  uint4 r = {}, g = {};
  if (NORM && j == 3) {
    r = ld16(a.res + voff);
    g = ld16(a.gamma + col);
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (a.trigger == 1) asm volatile("griddepcontrol.launch_dependents;");

  const uint4 s = ld16(a.shared + static_cast<long long>(t) * (HID * 2) + col);
  uint4 d[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    const bool valid = m[k] >= 0;
    d[k] = ld16(a.g2 + static_cast<long long>(valid ? m[k] : 0) * a.g2_stride + col);
    wk[k] = valid ? wk[k] : 0.0f;
  }
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    acc[0] = fmaf(wk[k], lo(d[k].x), acc[0]);
    acc[1] = fmaf(wk[k], hi(d[k].x), acc[1]);
    acc[2] = fmaf(wk[k], lo(d[k].y), acc[2]);
    acc[3] = fmaf(wk[k], hi(d[k].y), acc[3]);
    acc[4] = fmaf(wk[k], lo(d[k].z), acc[4]);
    acc[5] = fmaf(wk[k], hi(d[k].z), acc[5]);
    acc[6] = fmaf(wk[k], lo(d[k].w), acc[6]);
    acc[7] = fmaf(wk[k], hi(d[k].w), acc[7]);
  }
  uint4 q;
  q.x = pack(acc[0] + lo(s.x), acc[1] + hi(s.x));
  q.y = pack(acc[2] + lo(s.y), acc[3] + hi(s.y));
  q.z = pack(acc[4] + lo(s.z), acc[5] + hi(s.z));
  q.w = pack(acc[6] + lo(s.w), acc[7] + hi(s.w));

  const long long slot = static_cast<long long>(phase % 3) * a.slot_bytes;
  if (j < 3) {
    uint4 p;
    p.x = sanitize(q.x); p.y = sanitize(q.y); p.z = sanitize(q.z); p.w = sanitize(q.w);
    st16_volatile(a.peer[dest] + slot + a.rank * a.chunk_bytes + voff, p);    
    if (threadIdx.x == 0) a.phase[cta] = phase + 1;
    return;
  }

  char* inbox = a.peer[a.rank] + slot + voff;
  char* a1 = inbox + ((a.rank + 1) & 3) * a.chunk_bytes;
  char* a2 = inbox + ((a.rank + 2) & 3) * a.chunk_bytes;
  char* a3 = inbox + ((a.rank + 3) & 3) * a.chunk_bytes;
  uint4 p1, p2, p3;
  do {
    p1 = ld16_volatile(a1);
    p2 = ld16_volatile(a2);
    p3 = ld16_volatile(a3);
  } while (poisoned(p1) | poisoned(p2) | poisoned(p3));
  uint4 o;
  o.x = pack(((lo(q.x) + lo(p1.x)) + lo(p2.x)) + lo(p3.x), ((hi(q.x) + hi(p1.x)) + hi(p2.x)) + hi(p3.x));
  o.y = pack(((lo(q.y) + lo(p1.y)) + lo(p2.y)) + lo(p3.y), ((hi(q.y) + hi(p1.y)) + hi(p2.y)) + hi(p3.y));
  o.z = pack(((lo(q.z) + lo(p1.z)) + lo(p2.z)) + lo(p3.z), ((hi(q.z) + hi(p1.z)) + hi(p2.z)) + hi(p3.z));
  o.w = pack(((lo(q.w) + lo(p1.w)) + lo(p2.w)) + lo(p3.w), ((hi(q.w) + hi(p1.w)) + hi(p2.w)) + hi(p3.w));
  if (NORM) st16(a.out + voff, o);
  else st16_volatile(a.out + voff, o);
  if (a.trigger == 2) asm volatile("griddepcontrol.launch_dependents;");
  const uint4 z = {SENT, SENT, SENT, SENT};
  st16_volatile(a1, z);
  st16_volatile(a2, z);
  st16_volatile(a3, z);
  if (threadIdx.x == 0) a.phase[cta] = phase + 1;
  if (!NORM) return;

  uint4 n;
  n.x = pack(lo(o.x) + lo(r.x), hi(o.x) + hi(r.x));
  n.y = pack(lo(o.y) + lo(r.y), hi(o.y) + hi(r.y));
  n.z = pack(lo(o.z) + lo(r.z), hi(o.z) + hi(r.z));
  n.w = pack(lo(o.w) + lo(r.w), hi(o.w) + hi(r.w));
  st16(a.resout + voff, n);
  const float x0 = lo(n.x), x1 = hi(n.x), x2 = lo(n.y), x3 = hi(n.y), x4 = lo(n.z), x5 = hi(n.z), x6 = lo(n.w),
              x7 = hi(n.w);
  float ps = x0 * x0;
  ps = fmaf(x1, x1, ps);
  ps = fmaf(x2, x2, ps);
  ps = fmaf(x3, x3, ps);
  ps = fmaf(x4, x4, ps);
  ps = fmaf(x5, x5, ps);
  ps = fmaf(x6, x6, ps);
  ps = fmaf(x7, x7, ps);
  const float part = warp_sum(ps);
  const int wl = threadIdx.x & 31;
  if (CL == 1) {
    if (wl == 0) parts[lane >> 5] = part;
    __syncthreads();
  } else {
    auto cluster = cooperative_groups::this_cluster();
    if (wl < CL) *cluster.map_shared_rank(&parts[lane >> 5], wl) = part;
    cluster.sync();
  }
  const float total = warp_sum(wl < WARPS ? parts[wl] : 0.0f);
  const float inv = rsqrtf(total * static_cast<float>(1.0 / HID) + a.eps);
  uint4 y;
  y.x = pack((x0 * inv) * lo(g.x), (x1 * inv) * hi(g.x));
  y.y = pack((x2 * inv) * lo(g.y), (x3 * inv) * hi(g.y));
  y.z = pack((x4 * inv) * lo(g.z), (x5 * inv) * hi(g.z));
  y.w = pack((x6 * inv) * lo(g.w), (x7 * inv) * hi(g.w));
  st16(a.normed + voff, y);
}

static void check_rows(const at::Tensor& x, int64_t rows, const char* what) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(0) == rows && x.size(1) == HID
              && x.is_contiguous() && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "ftail: ", what);
}

void run(at::Tensor gemm2, at::Tensor map, at::Tensor weights, at::Tensor shared, at::Tensor out, at::Tensor residual,
         at::Tensor gamma, at::Tensor normed, at::Tensor resout, at::Tensor phase, std::vector<int64_t> peers,
         int64_t rank, int64_t slot_bytes, double eps, bool norm, bool pdl, int64_t trigger, int64_t cluster) {
  const int64_t T = shared.size(0), rows = T / 4;
  TORCH_CHECK(T >= 4 && T % 4 == 0 && T <= 96, "ftail: gathered rows");
  check_rows(shared, T, "shared");
  check_rows(out, rows, "out");
  TORCH_CHECK(gemm2.is_cuda() && gemm2.scalar_type() == at::kBFloat16 && gemm2.dim() == 2 && gemm2.size(1) >= HID
              && gemm2.stride(1) == 1 && gemm2.stride(0) % 8 == 0
              && reinterpret_cast<uintptr_t>(gemm2.data_ptr()) % 16 == 0, "ftail: gemm2");
  TORCH_CHECK(map.scalar_type() == at::kInt && map.is_contiguous() && map.numel() >= T * TOPK, "ftail: map");
  TORCH_CHECK((weights.scalar_type() == at::kFloat || weights.scalar_type() == at::kBFloat16) && weights.is_contiguous()
              && weights.numel() >= T * TOPK, "ftail: weights");
  TORCH_CHECK(phase.scalar_type() == at::kInt && phase.is_contiguous() && phase.numel() >= T * cluster, "ftail: phase");
  TORCH_CHECK(peers.size() == 4 && rank >= 0 && rank < 4 && slot_bytes >= 4 * rows * HID * 2, "ftail: ring");
  Args a = {};
  if (norm) {
    check_rows(residual, rows, "residual");
    check_rows(normed, rows, "normed");
    check_rows(resout, rows, "resout");
    TORCH_CHECK(gamma.is_cuda() && gamma.scalar_type() == at::kBFloat16 && gamma.numel() == HID && gamma.is_contiguous()
                && reinterpret_cast<uintptr_t>(gamma.data_ptr()) % 16 == 0, "ftail: gamma");
    a.res = reinterpret_cast<const char*>(residual.data_ptr());
    a.gamma = reinterpret_cast<const char*>(gamma.data_ptr());
    a.normed = reinterpret_cast<char*>(normed.data_ptr());
    a.resout = reinterpret_cast<char*>(resout.data_ptr());
  }
  a.g2 = reinterpret_cast<const char*>(gemm2.data_ptr());
  a.map = reinterpret_cast<const int*>(map.data_ptr());
  a.w = weights.data_ptr();
  a.wbf16 = weights.scalar_type() == at::kBFloat16;
  a.shared = reinterpret_cast<const char*>(shared.data_ptr());
  a.out = reinterpret_cast<char*>(out.data_ptr());
  for (int p = 0; p < 4; ++p) a.peer[p] = reinterpret_cast<char*>(static_cast<uintptr_t>(peers[p]));
  a.phase = reinterpret_cast<int*>(phase.data_ptr());
  a.g2_stride = gemm2.stride(0) * 2;
  a.slot_bytes = slot_bytes;
  a.chunk_bytes = rows * HID * 2;
  a.eps = static_cast<float>(eps);
  a.rank = static_cast<int>(rank);
  a.chunk_rows = static_cast<int>(rows);
  a.trigger = static_cast<int>(trigger);

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(T * cluster));
  cfg.blockDim = dim3(static_cast<unsigned>(VPR / cluster));
  cfg.stream = c10::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute attr[2];
  int n = 0;
  if (cluster > 1) {
    attr[n].id = cudaLaunchAttributeClusterDimension;
    attr[n].val.clusterDim.x = static_cast<unsigned>(cluster);
    attr[n].val.clusterDim.y = 1;
    attr[n].val.clusterDim.z = 1;
    ++n;
  }
  if (pdl) {
    attr[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  cfg.attrs = attr;
  cfg.numAttrs = n;
  void* kargs[] = {&a};
  const void* kernel = nullptr;
#define FTAIL_PICK(CL)                                                                              \
  if (cluster == CL)                                                                                \
    kernel = norm ? reinterpret_cast<const void*>(tail_kernel<true, CL>) : reinterpret_cast<const void*>(tail_kernel<false, CL>);
  FTAIL_PICK(1) FTAIL_PICK(2) FTAIL_PICK(3) FTAIL_PICK(4) FTAIL_PICK(6) FTAIL_PICK(8)
#undef FTAIL_PICK
  TORCH_CHECK(kernel != nullptr, "ftail: cluster size");
  const cudaError_t e = cudaLaunchKernelExC(&cfg, kernel, kargs);
  TORCH_CHECK(e == cudaSuccess, "ftail launch: ", cudaGetErrorString(e));
}

}   

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &ftail::run); }
