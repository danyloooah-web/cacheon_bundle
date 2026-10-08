#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace route8s {

constexpr int E = 256, K = 8, PER = E / 32, TILE = 8, MAXR = 16;

struct Meta {
  int32_t* padded;     
  int32_t* map;        
  int32_t* tok;        
  int32_t* counts;     
  int32_t* batch;      
  int32_t* limit;      
  int32_t* nonexit;    
};

__device__ __forceinline__ uint32_t ord_key(float k) {
  uint32_t b = __float_as_uint(k);
  if (b == 0x80000000u) b = 0u;
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ float pos_budget(const float4 budget, int row) {
  const int p = row & 3;
  return p == 0 ? budget.x : p == 1 ? budget.y : p == 2 ? budget.z : budget.w;
}

__device__ __forceinline__ void prune_slot(int slot, int e, float wt, int cnt, int max_count, float budget, int pair,
                                           float4* sk, bool& drop, bool& vote) {
  float* const keys = reinterpret_cast<float*>(sk);
  const bool fin = wt < INFINITY;                                   
  const float key = e >= 0 && cnt <= max_count && fin ? wt : INFINITY;
  float kj[K];
#pragma unroll
  for (int j = 0; j < K; ++j) kj[j] = __shfl_sync(0xffffffffu, key, j, K);
  int rank = 0;
#pragma unroll
  for (int j = 0; j < K; ++j) rank += kj[j] < key || (kj[j] == key && j < slot);
  keys[rank] = key;
  __syncwarp();
  const float4 c = sk[0], d = sk[1];
  const float kt[K] = {c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w};
  float run[K];
  run[0] = 0.f + kt[0];
#pragma unroll
  for (int t = 1; t < K; ++t) run[t] = run[t - 1] + kt[t];
  uint32_t fits = 0u;
#pragma unroll
  for (int t = 0; t < K; ++t) fits |= uint32_t(kt[t] < INFINITY && run[t] <= budget) << t;
  const int nd = __ffs(~fits) - 1;                                   
  float used = 0.f;
#pragma unroll
  for (int t = 0; t < K; ++t) used = t < nd ? run[t] : used;
  drop = rank < nd;
  const float k2 = pair && e >= 0 && cnt == 2 && fin && used + wt <= budget ? wt : INFINITY;
  float vj[K];
#pragma unroll
  for (int j = 0; j < K; ++j) vj[j] = __shfl_sync(0xffffffffu, k2, j, K);
  bool beaten = false;
#pragma unroll
  for (int j = 0; j < K; ++j) beaten |= vj[j] < k2 || (vj[j] == k2 && j < slot);
  vote = k2 < INFINITY && !beaten;
}

__device__ __forceinline__ void route_row_fast(const float* __restrict__ logits, const float (&bv)[PER], int row,
                                               int lane, float scale, int& out_id, float& out_w) {
  float sc[PER];
  uint32_t ok[PER];
  int ix[PER];
  bool nz = false;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const float x = __ldcg(logits + size_t(row) * E + lane + 32 * i);
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

__global__ void __launch_bounds__(MAXR * 32, 1) route_meta_kernel(
    const float* __restrict__ logits, const float* __restrict__ bias, int32_t* __restrict__ ids,
    float* __restrict__ w, float scale, int rows, int pdl, Meta meta, int32_t* __restrict__ ticket, int max_count,
    float4 budgets, int pair, int renorm) {
  __shared__ int s_all[MAXR * K];
  __shared__ float s_w[MAXR * K];
  __shared__ float4 s_key[MAXR * 2];
  __shared__ uint32_t mask[E], dropm[E];
  __shared__ int s_off[E], s_wtot[4], vcnt[E];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float bv[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) bv[i] = bias[lane + 32 * i];       
  const bool prune = fmaxf(fmaxf(budgets.x, budgets.y), fmaxf(budgets.z, budgets.w)) > 0.f;
  const float budget = pos_budget(budgets, tid / K);               
  for (int i = tid; i < E; i += MAXR * 32) {
    mask[i] = 0u;
    dropm[i] = 0u;
    vcnt[i] = 0;
  }
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __syncthreads();                                                  
  int my_id = -1;
  float my_w = 0.f;
  if (warp < rows) {
    route_row_fast(logits, bv, warp, lane, scale, my_id, my_w);
    if (lane < K) {
      s_all[warp * K + lane] = my_id;
      s_w[warp * K + lane] = my_w;
      if (my_id >= 0) atomicOr(&mask[my_id], 1u << warp);
    }
  }
  __syncthreads();
  if (prune) {
    if (tid < MAXR * K && warp * 32 < rows * K) {
      const bool real = tid < rows * K;
      const int e = real ? s_all[tid] : -1;
      const float wt = real ? s_w[tid] : 0.f;
      const int cnt = e >= 0 ? __popc(mask[e]) : 0;
      bool drop, vote;
      prune_slot(tid % K, e, wt, cnt, max_count, budget, pair, s_key + (tid / K) * 2, drop, vote);
      if (drop) atomicOr(&dropm[e], 1u << (tid / K));
      if (vote) atomicAdd(&vcnt[e], 1);
    }
    __syncthreads();
  }
  const bool rn = prune && renorm;
  const int e0 = 2 * tid, e1 = 2 * tid + 1;
  int c0 = 0, c1 = 0, n0 = 0, n1 = 0, incl = 0;
  if (tid < 128) {
    uint32_t m0 = mask[e0], m1 = mask[e1];
    if (prune) {
      m0 = vcnt[e0] == 2 ? 0u : m0 & ~dropm[e0];
      m1 = vcnt[e1] == 2 ? 0u : m1 & ~dropm[e1];
      mask[e0] = m0;
      mask[e1] = m1;
    }
    c0 = __popc(m0);
    c1 = __popc(m1);
    n0 = (c0 + TILE - 1) / TILE;
    n1 = (c1 + TILE - 1) / TILE;
    incl = n0 + n1;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += v;
    }
    if (lane == 31) s_wtot[warp] = incl;
  }
  __syncthreads();
  if (tid < 128) {
    int before = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) before += i < warp ? s_wtot[i] : 0;
    const int off0 = before + incl - n0 - n1, off1 = off0 + n0;
    s_off[e0] = off0;
    s_off[e1] = off1;
    meta.counts[e0] = c0;
    meta.counts[e1] = c1;
    for (int j = 0; j < n0; ++j) {
      meta.batch[off0 + j] = e0;
      meta.limit[off0 + j] = min((off0 + j + 1) * TILE, off0 * TILE + c0);
    }
    for (int j = 0; j < n1; ++j) {
      meta.batch[off1 + j] = e1;
      meta.limit[off1 + j] = min((off1 + j + 1) * TILE, off1 * TILE + c1);
    }
    if (tid == 127) {
      meta.padded[0] = (off1 + n1) * TILE;
      meta.nonexit[0] = off1 + n1;
    }
  }
  __syncthreads();
  for (int s = tid; s < rows * K; s += MAXR * 32) {
    const int e = s_all[s], t = s / K;
    if (e < 0 || !(mask[e] & (1u << t))) {
      meta.map[s] = -1;
      ids[s] = -1;
      if (!rn) w[s] = 0.f;
    } else {
      const int p = s_off[e] * TILE + __popc(mask[e] & ((1u << t) - 1u));
      meta.map[s] = p;
      meta.tok[p] = t;
      ids[s] = e;
      if (!rn) w[s] = s_w[s];
    }
  }
  if (rn && warp >= 4 && (warp - 4) * 32 < rows * K) {
    const int s = tid - 128, t = s / K;
    const bool real = s < rows * K;
    const int e = real ? s_all[s] : -1;
    const bool kept = e >= 0 && !((dropm[e] >> t) & 1u) && vcnt[e] != 2;
    const float wo = real ? s_w[s] : 0.f;
    float tot = wo, ks = kept ? wo : 0.f;
#pragma unroll
    for (int o = 1; o < K; o <<= 1) {
      tot = __fadd_rn(tot, __shfl_xor_sync(0xffffffffu, tot, o));
      ks = __fadd_rn(ks, __shfl_xor_sync(0xffffffffu, ks, o));
    }
    const bool row_drop = (__ballot_sync(0xffffffffu, e >= 0 && !kept) >> (lane & ~(K - 1))) & ((1u << K) - 1u);
    if (real) w[s] = !kept ? 0.f : row_drop && ks > 0.f ? __fmul_rn(wo, __fdiv_rn(tot, ks)) : wo;
  }
}

void route_meta(torch::Tensor logits, torch::Tensor bias, torch::Tensor ids, torch::Tensor w, double scale, bool pdl,
                torch::Tensor padded, torch::Tensor map, torch::Tensor tok, torch::Tensor counts, torch::Tensor batch,
                torch::Tensor limit, torch::Tensor nonexit, torch::Tensor ticket, int64_t max_count, double budget, bool pair,
                std::vector<double> pos_mult, bool renorm) {
  TORCH_CHECK(pos_mult.size() == 4, "pos_mult: one multiplier per verify position (4)");
  for (const double m : pos_mult) TORCH_CHECK(std::isfinite(m) && m >= 0.0, "pos_mult: finite and >= 0");
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.dim() == 2 && logits.size(1) == E
              && logits.is_contiguous(), "logits");
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kFloat && bias.numel() == E && bias.is_contiguous(), "bias");
  const int rows = logits.size(0);
  TORCH_CHECK(rows >= 1 && rows <= MAXR, "route_meta: rows ", rows, " outside 1..", MAXR);
  TORCH_CHECK(ids.scalar_type() == at::kInt && ids.numel() == int64_t(rows) * K && ids.is_contiguous(), "ids");
  TORCH_CHECK(w.scalar_type() == at::kFloat && w.numel() == int64_t(rows) * K && w.is_contiguous(), "w");
  const int64_t filled = std::min<int64_t>(E, int64_t(rows) * K);
  const int64_t max_ctas = filled + (int64_t(rows) * K - filled) / TILE;
  for (auto* t : {&padded, &map, &tok, &counts, &batch, &limit, &nonexit, &ticket})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kInt && t->is_contiguous(), "meta tensors must be int32");
  TORCH_CHECK(map.numel() >= int64_t(rows) * K && tok.numel() >= max_ctas * TILE + 1 && counts.numel() >= E
              && batch.numel() >= max_ctas && limit.numel() >= max_ctas && padded.numel() >= 1
              && nonexit.numel() >= 1 && ticket.numel() >= 1, "meta tensor sizes");
  c10::cuda::CUDAGuard guard(logits.device());
  auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  const float* lp = logits.data_ptr<float>(); const float* bp = bias.data_ptr<float>();
  int32_t* ip = ids.data_ptr<int32_t>(); float* wp = w.data_ptr<float>();
  Meta m{padded.data_ptr<int32_t>(), map.data_ptr<int32_t>(), tok.data_ptr<int32_t>(), counts.data_ptr<int32_t>(),
         batch.data_ptr<int32_t>(), limit.data_ptr<int32_t>(), nonexit.data_ptr<int32_t>()};
  int32_t* tp = ticket.data_ptr<int32_t>();
  float sc = float(scale);
  float4 ta = make_float4(float(budget * pos_mult[0] * scale), float(budget * pos_mult[1] * scale),
                          float(budget * pos_mult[2] * scale), float(budget * pos_mult[3] * scale));
  int r = rows, pd = pdl ? 1 : 0, mc = int(max_count), pr = pair ? 1 : 0, rn = renorm ? 1 : 0;
  void* args[] = {&lp, &bp, &ip, &wp, &sc, &r, &pd, &m, &tp, &mc, &ta, &pr, &rn};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1); cfg.blockDim = dim3(MAXR * 32);
  cfg.dynamicSmemBytes = 0; cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr; cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t err = cudaLaunchKernelExC(&cfg, reinterpret_cast<const void*>(route_meta_kernel), args);
  TORCH_CHECK(err == cudaSuccess, "route8s launch failed: ", cudaGetErrorString(err));
}

}   

namespace rz {

constexpr int N = 256, K = 6144, NS = 2, MAXM = 16;
constexpr int THREADS = K / 16;                
constexpr int WARPS = THREADS / 32;            
constexpr int BLOCKS = N / NS;                 

struct Vec32 { uint4 lo, hi; };                
__device__ __forceinline__ Vec32 ld32(const void* p) {
  Vec32 v;
  v.lo = *reinterpret_cast<const uint4*>(p);
  v.hi = *reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(p) + 16);
  return v;
}
__device__ __forceinline__ float fma_bf16(uint32_t a, uint32_t b, float acc) {
  float r;
  asm("fma.rn.f32.bf16 %0, %1, %2, %3;" : "=f"(r) : "h"(static_cast<uint16_t>(a)), "h"(static_cast<uint16_t>(b)), "f"(acc));
  return r;
}
__device__ __forceinline__ float dot16(const Vec32& a, const Vec32& b, float acc) {
  const uint32_t aw[8] = {a.lo.x, a.lo.y, a.lo.z, a.lo.w, a.hi.x, a.hi.y, a.hi.z, a.hi.w};
  const uint32_t bw[8] = {b.lo.x, b.lo.y, b.lo.z, b.lo.w, b.hi.x, b.hi.y, b.hi.z, b.hi.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc = fma_bf16(aw[i] & 0xffffu, bw[i] & 0xffffu, acc);
    acc = fma_bf16(aw[i] >> 16, bw[i] >> 16, acc);
  }
  return acc;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int mask = 16; mask >= 1; mask >>= 1) v = v + __shfl_xor_sync(0xffffffffu, v, mask, 32);
  return v;
}

template <int M, bool FLAGGED>
__global__ void __launch_bounds__(THREADS, 1)
rz_kernel(const __nv_bfloat16* __restrict__ x, long long stride_x, const __nv_bfloat16* __restrict__ w,
          float* __restrict__ out, const uint8_t* __restrict__ zero, int pdl) {
  const int bx = blockIdx.x, tx = threadIdx.x, warp = tx >> 5;
  const __nv_bfloat16* wt = w + size_t(bx) * (NS * K);
  Vec32 wv[NS];
#pragma unroll
  for (int n = 0; n < NS; ++n) wv[n] = ld32(wt + size_t(n) * K + tx * 16);
  bool hint = false;                                                   
  if constexpr (FLAGGED) {
    const uint4 hf = *reinterpret_cast<const uint4*>(zero);
    const uint32_t hw[4] = {hf.x, hf.y, hf.z, hf.w};
#pragma unroll
    for (int g = 0; g < (M + 3) / 4; ++g) {
      const uint32_t need = 4 * g + 4 <= M ? 0x01010101u : 0x01010101u >> (8 * (4 * g + 4 - M));
      hint |= (hw[g] & need) == need;
    }
  }
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
  constexpr int G = (M + 3) / 4;                                       
  bool live[G];
#pragma unroll
  for (int g = 0; g < G; ++g) live[g] = true;
  if constexpr (FLAGGED) {
    if (hint) {
      const uint4 zf = *reinterpret_cast<const uint4*>(zero);           
      const uint32_t zw[4] = {zf.x, zf.y, zf.z, zf.w};
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const uint32_t need = 4 * g + 4 <= M ? 0x01010101u : 0x01010101u >> (8 * (4 * g + 4 - M));
        live[g] = (zw[g] & need) != need;
      }
    }
  }
  uint32_t mask = 0;
#pragma unroll
  for (int g = 0; g < G; ++g) mask |= live[g] ? 1u << g : 0u;
  __shared__ float s_acc[WARPS][M * NS];
  Vec32 xv[M];
  if (mask == (1u << G) - 1u) {
#pragma unroll
    for (int m = 0; m < M; ++m) xv[m] = ld32(x + m * stride_x + tx * 16);
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int n = 0; n < NS; ++n) s_acc[warp][m * NS + n] = warp_sum(dot16(xv[m], wv[n], 0.0f));
  } else {
#pragma unroll
    for (int g = 0; g < G; ++g)
      if (mask & (1u << g)) {
#pragma unroll
        for (int m = 4 * g; m < 4 * g + 4 && m < M; ++m) xv[m] = ld32(x + m * stride_x + tx * 16);
      }
#pragma unroll
    for (int g = 0; g < G; ++g) {
      if (mask & (1u << g)) {
#pragma unroll
        for (int m = 4 * g; m < 4 * g + 4 && m < M; ++m)
#pragma unroll
          for (int n = 0; n < NS; ++n) s_acc[warp][m * NS + n] = warp_sum(dot16(xv[m], wv[n], 0.0f));
      } else {
#pragma unroll
        for (int m = 4 * g; m < 4 * g + 4 && m < M; ++m)
#pragma unroll
          for (int n = 0; n < NS; ++n) s_acc[warp][m * NS + n] = 0.0f;
      }
    }
  }
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __syncthreads();
  if (tx < M * NS) {
    float acc[WARPS];
#pragma unroll
    for (int i = 0; i < WARPS; ++i) acc[i] = s_acc[i][tx];
#pragma unroll
    for (int i = 1; i < WARPS; ++i) acc[0] += acc[i];
    out[size_t(tx / NS) * N + bx * NS + tx % NS] = acc[0];
  }
}

template <bool FLAGGED, std::size_t... I>
static constexpr auto make_table(std::index_sequence<I...>) {
  using Fn = void (*)(const __nv_bfloat16*, long long, const __nv_bfloat16*, float*, const uint8_t*, int);
  return std::array<Fn, MAXM + 1>{nullptr, rz_kernel<int(I) + 1, FLAGGED>...};
}
static constexpr auto kPlain = make_table<false>(std::make_index_sequence<MAXM>{});
static constexpr auto kFlagged = make_table<true>(std::make_index_sequence<MAXM>{});

void run(torch::Tensor x, torch::Tensor w, torch::Tensor zero, torch::Tensor out, bool pdl, bool flagged) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == K && x.stride(1) == 1
              && x.size(0) >= 1 && x.size(0) <= MAXM && x.stride(0) % 16 == 0
              && reinterpret_cast<uintptr_t>(x.data_ptr()) % 32 == 0, "rz: x must be bf16 [M <= 16, 6144], 32-byte aligned rows");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.dim() == 2 && w.size(0) == N
              && w.size(1) == K && reinterpret_cast<uintptr_t>(w.data_ptr()) % 32 == 0, "rz: gate weight bf16 [256, 6144]");
  TORCH_CHECK(zero.is_cuda() && zero.scalar_type() == at::kByte && zero.is_contiguous() && zero.numel() >= MAXM
              && reinterpret_cast<uintptr_t>(zero.data_ptr()) % 16 == 0, "rz: zero-row flags uint8 [>= 16]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.is_contiguous() && out.dim() == 2
              && out.size(0) == x.size(0) && out.size(1) == N, "rz: out fp32 [M, 256]");
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  float* op = out.data_ptr<float>(); const uint8_t* zp = zero.data_ptr<uint8_t>();
  long long sx = x.stride(0); int M = int(x.size(0)), pd = pdl ? 1 : 0;
  void* args[] = {&xp, &sx, &wp, &op, &zp, &pd};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(BLOCKS); cfg.blockDim = dim3(THREADS); cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
  const void* fn = reinterpret_cast<const void*>(flagged ? kFlagged[M] : kPlain[M]);
  TORCH_CHECK(cudaLaunchKernelExC(&cfg, fn, args) == cudaSuccess, "rz launch failed");
}

}   

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rz_run", &rz::run);
  namespace py = pybind11;
  m.def("route_meta", &route8s::route_meta, py::arg("logits"), py::arg("bias"), py::arg("ids"), py::arg("w"),
        py::arg("scale"), py::arg("pdl"), py::arg("padded"), py::arg("map"), py::arg("tok"), py::arg("counts"),
        py::arg("batch"), py::arg("limit"), py::arg("nonexit"), py::arg("ticket"), py::arg("max_count"),
        py::arg("budget"), py::arg("pair"), py::arg("pos_mult") = std::vector<double>{1.0, 1.0, 1.0, 1.0},
        py::arg("renorm") = false);
}
