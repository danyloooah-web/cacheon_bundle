#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace dqab {

constexpr int H_K = 192;         
constexpr int KS = H_K / 16;     
constexpr int NOPE = 512;        
constexpr int ROPE = 64;
constexpr int ROW = NOPE + ROPE;
constexpr int MAX_T = 8;

struct Args {
  const __nv_bfloat16* qn;       
  const __nv_bfloat16* qr;       
  const uint4* w;                
  const __nv_bfloat16* kn;       
  const __nv_bfloat16* kr;       
  const long long* pos;          
  const float* cs;               
  const long long* loc;          
  uint8_t* qo;                   
  uint8_t* kv;                   
  long long sqn_t, sqn_h, sqr_t, sqr_h, skn_t, skr_t, sqo_t, sqo_h, skv, scs;    
  int T, H;
};

__device__ __forceinline__ void mma_bf16(float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
  uint32_t out;
  asm volatile("{\n.reg .b16 t;\ncvt.rn.satfinite.e4m3x2.f32 t, %2, %1;\ncvt.u32.u16 %0, t;\n}" : "=r"(out) : "f"(lo), "f"(hi));
  return out;
}
__device__ __forceinline__ uint8_t e4m3(float v) { return static_cast<uint8_t>(e4m3x2(v, 0.f) & 0xff); }
__device__ __forceinline__ float bf16_round(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ void rope_pair(float xe, float xo, float c, float s, float& ye, float& yo) {
  ye = __fmaf_rn(xe, c, __fmul_rn(__fmul_rn(xo, -1.f), s));
  yo = __fmaf_rn(xo, c, __fmul_rn(xe, s));
}

template <int NP>
__global__ void __launch_bounds__(128) dqab_kernel(const Args a) {
  constexpr int TILES = NOPE / 16 / NP / 4;     
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int T = a.T;
  const bool kv_cta = static_cast<int>(blockIdx.x) >= a.H * NP;
  const int kt = static_cast<int>(blockIdx.x) - a.H * NP;       
  const int h = blockIdx.x / NP, part = blockIdx.x % NP;

  uint4 af[TILES][KS];
  if (!kv_cta) {
    const uint4* w = a.w + ((static_cast<long long>(h) * (NOPE / 16) + (part * 4 + warp) * TILES) * KS) * 32 + lane;
#pragma unroll
    for (int m = 0; m < TILES; ++m)
#pragma unroll
      for (int s = 0; s < KS; ++s) af[m][s] = w[(m * KS + s) * 32];
  }

  float rc[2], rs[2];
  int rt[2], rp[2];
  bool rv[2];
  long long loc = 0;
  if (kv_cta) loc = a.loc[kt];
  for (int j = 0; j < 2; ++j) {
    const int i = tid + 128 * j;
    rt[j] = kv_cta ? kt : i >> 5; rp[j] = i & 31;
    rv[j] = kv_cta ? (j == 0 && tid < 32) : (part == 0 && rt[j] < T);
    if (rv[j]) {
      const long long p = a.pos[rt[j]];
      rc[j] = a.cs[p * a.scs + rp[j]];
      rs[j] = a.cs[p * a.scs + 32 + rp[j]];
    }
  }

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  if (kv_cta) {
    if (loc == 0) return;
    const __nv_bfloat16* src = a.kn + kt * a.skn_t + tid * 4;
    const uint32_t lo = e4m3x2(__bfloat162float(src[0]), __bfloat162float(src[1]));
    const uint32_t hi = e4m3x2(__bfloat162float(src[2]), __bfloat162float(src[3]));
    *reinterpret_cast<uint32_t*>(a.kv + loc * a.skv + tid * 4) = lo | (hi << 16);
    if (rv[0]) {
      const __nv_bfloat16* r = a.kr + kt * a.skr_t + 2 * rp[0];
      float ye, yo;
      rope_pair(__bfloat162float(r[0]), __bfloat162float(r[1]), rc[0], rs[0], ye, yo);
      *reinterpret_cast<uint16_t*>(a.kv + loc * a.skv + NOPE + 2 * rp[0]) = static_cast<uint16_t>(e4m3x2(ye, yo));
    }
    return;
  }

  const int tok = lane >> 2, c = lane & 3;
  uint32_t b[KS][2];
  if (tok < T) {
    const uint32_t* q = reinterpret_cast<const uint32_t*>(a.qn + tok * a.sqn_t + h * a.sqn_h);
#pragma unroll
    for (int s = 0; s < KS; ++s) {
      b[s][0] = q[8 * s + c];
      b[s][1] = q[8 * s + 4 + c];
    }
  } else {
#pragma unroll
    for (int s = 0; s < KS; ++s) b[s][0] = b[s][1] = 0u;
  }

  const int g = lane >> 2;                      
  const int t0 = 2 * c;                         
  uint8_t* out0 = a.qo + t0 * a.sqo_t + h * a.sqo_h + ((part * 4 + warp) * TILES) * 16;
  uint8_t* out1 = out0 + a.sqo_t;
  float acc[TILES][4];
#pragma unroll
  for (int m = 0; m < TILES; ++m) acc[m][0] = acc[m][1] = acc[m][2] = acc[m][3] = 0.f;
#pragma unroll
  for (int s = 0; s < KS; ++s) {
#pragma unroll
    for (int m = 0; m < TILES; ++m) {
      const uint32_t fr[4] = {af[m][s].x, af[m][s].y, af[m][s].z, af[m][s].w};
      mma_bf16(acc[m], fr, b[s]);
    }
  }
#pragma unroll
  for (int m = 0; m < TILES; ++m) {
    const uint32_t lo = e4m3x2(bf16_round(acc[m][0]), bf16_round(acc[m][2]));    
    const uint32_t hi = e4m3x2(bf16_round(acc[m][1]), bf16_round(acc[m][3]));    
    if (t0 < T) {
      out0[m * 16 + g] = static_cast<uint8_t>(lo);
      out0[m * 16 + g + 8] = static_cast<uint8_t>(lo >> 8);
    }
    if (t0 + 1 < T) {
      out1[m * 16 + g] = static_cast<uint8_t>(hi);
      out1[m * 16 + g + 8] = static_cast<uint8_t>(hi >> 8);
    }
  }

  for (int j = 0; j < 2; ++j) {
    if (!rv[j]) continue;
    const __nv_bfloat16* src = a.qr + rt[j] * a.sqr_t + h * a.sqr_h + 2 * rp[j];
    float ye, yo;
    rope_pair(__bfloat162float(src[0]), __bfloat162float(src[1]), rc[j], rs[j], ye, yo);
    *reinterpret_cast<uint16_t*>(a.qo + rt[j] * a.sqo_t + h * a.sqo_h + NOPE + 2 * rp[j]) =
        static_cast<uint16_t>(e4m3x2(ye, yo));
  }
}

template <int NP>
static void launch(const Args& args, bool pdl) {
  auto kernel = dqab_kernel<NP>;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(args.H * NP + args.T);
  cfg.blockDim = dim3(128);
  cfg.stream = c10::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  Args copy = args;
  void* kargs[] = {&copy};
  const cudaError_t e = cudaLaunchKernelExC(&cfg, reinterpret_cast<const void*>(kernel), kargs);
  TORCH_CHECK(e == cudaSuccess, "dqab launch: ", cudaGetErrorString(e));
}

void decode(at::Tensor q_nope, at::Tensor q_rope, at::Tensor w, at::Tensor k_nope, at::Tensor k_rope, at::Tensor pos,
            at::Tensor cos_sin, at::Tensor loc, at::Tensor q_out, at::Tensor kv, int64_t parts, bool pdl) {
  const int64_t T = q_nope.size(0), H = q_nope.size(1);
  TORCH_CHECK(q_nope.is_cuda() && q_nope.dim() == 3 && q_nope.scalar_type() == at::kBFloat16 && q_nope.size(2) == H_K
              && q_nope.stride(2) == 1 && T >= 1 && T <= MAX_T, "dqab: q_nope");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q_nope.data_ptr()) % 4 == 0 && q_nope.stride(0) % 2 == 0
              && q_nope.stride(1) % 2 == 0, "dqab: q_nope alignment");
  TORCH_CHECK(q_rope.dim() == 3 && q_rope.scalar_type() == at::kBFloat16 && q_rope.size(0) == T && q_rope.size(1) == H
              && q_rope.size(2) == ROPE && q_rope.stride(2) == 1, "dqab: q_rope");
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.numel() == H * NOPE * H_K
              && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "dqab: w (fragment-major copy)");
  TORCH_CHECK(k_nope.dim() == 2 && k_nope.scalar_type() == at::kBFloat16 && k_nope.size(0) == T && k_nope.size(1) == NOPE
              && k_nope.stride(1) == 1, "dqab: k_nope");
  TORCH_CHECK(k_rope.dim() == 2 && k_rope.scalar_type() == at::kBFloat16 && k_rope.size(0) == T && k_rope.size(1) == ROPE
              && k_rope.stride(1) == 1, "dqab: k_rope");
  TORCH_CHECK(pos.scalar_type() == at::kLong && pos.dim() == 1 && pos.size(0) == T && pos.is_contiguous(), "dqab: pos");
  TORCH_CHECK(loc.scalar_type() == at::kLong && loc.dim() == 1 && loc.size(0) == T && loc.is_contiguous(), "dqab: loc");
  TORCH_CHECK(cos_sin.scalar_type() == at::kFloat && cos_sin.dim() == 2 && cos_sin.size(1) >= ROPE
              && cos_sin.stride(1) == 1, "dqab: cos_sin");
  TORCH_CHECK(q_out.dim() == 3 && q_out.element_size() == 1 && q_out.size(0) == T && q_out.size(1) == H
              && q_out.size(2) == ROW && q_out.stride(2) == 1 && q_out.stride(0) % 16 == 0 && q_out.stride(1) % 16 == 0
              && reinterpret_cast<uintptr_t>(q_out.data_ptr()) % 16 == 0, "dqab: q_out");
  TORCH_CHECK(kv.dim() == 2 && kv.element_size() == 1 && kv.size(1) >= ROW && kv.stride(1) == 1
              && kv.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(kv.data_ptr()) % 4 == 0, "dqab: kv");
  TORCH_CHECK(parts == 2 || parts == 4, "dqab: parts");
  Args a;
  a.qn = reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr());
  a.qr = reinterpret_cast<const __nv_bfloat16*>(q_rope.data_ptr());
  a.w = reinterpret_cast<const uint4*>(w.data_ptr());
  a.kn = reinterpret_cast<const __nv_bfloat16*>(k_nope.data_ptr());
  a.kr = reinterpret_cast<const __nv_bfloat16*>(k_rope.data_ptr());
  a.pos = reinterpret_cast<const long long*>(pos.data_ptr());
  a.cs = reinterpret_cast<const float*>(cos_sin.data_ptr());
  a.loc = reinterpret_cast<const long long*>(loc.data_ptr());
  a.qo = reinterpret_cast<uint8_t*>(q_out.data_ptr());
  a.kv = reinterpret_cast<uint8_t*>(kv.data_ptr());
  a.sqn_t = q_nope.stride(0); a.sqn_h = q_nope.stride(1);
  a.sqr_t = q_rope.stride(0); a.sqr_h = q_rope.stride(1);
  a.skn_t = k_nope.stride(0); a.skr_t = k_rope.stride(0);
  a.sqo_t = q_out.stride(0); a.sqo_h = q_out.stride(1);
  a.skv = kv.stride(0); a.scs = cos_sin.stride(0);
  a.T = static_cast<int>(T); a.H = static_cast<int>(H);
  if (parts == 2) launch<2>(a, pdl);
  else launch<4>(a, pdl);
}

}   

namespace dfa {
using bf16 = __nv_bfloat16;
constexpr int HIN = 6144, HOUT = 2624;
constexpr int QUART = HIN / 4;              
constexpr int STEPS = QUART / 16;           
constexpr int CTAS = HOUT / 32;             
#ifndef DFA_CS4
#define DFA_CS4 16
#endif
#ifndef DFA_CS8
#define DFA_CS8 12
#endif
constexpr int XROW = HIN * 2 + 16;          
constexpr int PART = 4 * 2 * 16 * 8 * 4;    
template <int TMAX> struct Cfg {
  static constexpr int CS = TMAX == 4 ? DFA_CS4 : DFA_CS8;
  static constexpr int CHUNK = CS * 1024;    
  static constexpr int NCH = STEPS / CS;     
  static constexpr int DEPTH = (int(232448) - PART - TMAX * XROW) / (4 * CHUNK) < NCH ? (int(232448) - PART - TMAX * XROW) / (4 * CHUNK) : NCH;
  static constexpr size_t RING = size_t(4) * DEPTH * CHUNK;
  static constexpr size_t XB = size_t(TMAX) * XROW;
  static constexpr size_t SMEM = RING + XB + PART;
  static_assert(STEPS % CS == 0 && DEPTH >= 1, "a quarter is a whole number of chunks");
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint32_t bar, uint32_t n) {
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
__device__ __forceinline__ uint4 lds_v4(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ void ldm_x2(uint32_t addr, uint32_t& b0, uint32_t& b1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int TMAX>
__global__ void __launch_bounds__(128, 1)
fused_a_kernel(const char* __restrict__ x, long long x_stride, const char* __restrict__ f2, bf16* __restrict__ out,
               int T, int trigger, int pdl) {
  using C = Cfg<TMAX>;
  constexpr int DEPTH = C::DEPTH, CS = C::CS, CHUNK = C::CHUNK, NCH = C::NCH;
  extern __shared__ __align__(1024) char smem[];
  __shared__ uint64_t bar[4][16];
  __shared__ uint64_t xbar[4];
  const int tid = threadIdx.x, lane = tid & 31, q = tid >> 5;       
  const uint32_t ring = smem_u32(smem) + uint32_t(q) * DEPTH * CHUNK;
  const uint32_t xs = smem_u32(smem) + uint32_t(C::RING);
  float* part = reinterpret_cast<float*>(smem + C::RING + C::XB);
  const char* src = f2 + (size_t(blockIdx.x) * 4 + q) * (size_t(NCH) * CHUNK);
  if (lane == 0) {
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) mbar_init(smem_u32(&bar[q][d]));
    mbar_init(smem_u32(&xbar[q]));
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (lane == 0) {
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) {
      mbar_expect(smem_u32(&bar[q][d]), CHUNK);
      bulk_copy(ring + d * CHUNK, src + size_t(d) * CHUNK, CHUNK, smem_u32(&bar[q][d]));
    }
  }
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (trigger == 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (lane == 0) {
    mbar_expect(smem_u32(&xbar[q]), uint32_t(T) * QUART * 2);
    for (int r = 0; r < T; ++r)
      bulk_copy(xs + r * XROW + q * QUART * 2, x + r * x_stride + size_t(q) * QUART * 2, QUART * 2, smem_u32(&xbar[q]));
  }
  mbar_wait(smem_u32(&xbar[q]), 0);
  float acc[2][4];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[m][e] = 0.f;
  const int brow = (lane & 7) < T ? (lane & 7) : 0;
  const uint32_t xa = xs + brow * XROW + q * QUART * 2 + ((lane >> 3) & 1) * 16;
  for (int i = 0; i < NCH; ++i) {
    const int d = i % DEPTH;
    mbar_wait(smem_u32(&bar[q][d]), uint32_t(i / DEPTH) & 1u);
    const uint32_t base = ring + d * CHUNK + lane * 16;
#pragma unroll
    for (int s = 0; s < CS; ++s) {
      uint32_t b0, b1;
      ldm_x2(xa + (i * CS + s) * 32, b0, b1);
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const uint4 a = lds_v4(base + s * 1024 + m * 512);
        mma_bf16(acc[m], a.x, a.y, a.z, a.w, b0, b1);
      }
    }
    if (lane == 0 && i + DEPTH < NCH) {
      mbar_expect(smem_u32(&bar[q][d]), CHUNK);
      bulk_copy(ring + d * CHUNK, src + size_t(i + DEPTH) * CHUNK, CHUNK, smem_u32(&bar[q][d]));
    }
  }
  {
    const int g = lane >> 2, r = lane & 3;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      float* p = part + (q * 2 + m) * 128;
      p[g * 8 + 2 * r] = acc[m][0];
      p[g * 8 + 2 * r + 1] = acc[m][1];
      p[(g + 8) * 8 + 2 * r] = acc[m][2];
      p[(g + 8) * 8 + 2 * r + 1] = acc[m][3];
    }
  }
  __syncthreads();
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int ch = tid & 31, tok = (tid >> 5) + 4 * h;
    if (tok < T) {
      const int o = ((ch >> 4) * 16 + (ch & 15)) * 8 + tok;
      float s = __fadd_rn(part[o], part[256 + o]);
      s = __fadd_rn(s, part[512 + o]);
      s = __fadd_rn(s, part[768 + o]);
      s = __fadd_rn(0.f, s);
      out[size_t(tok) * HOUT + blockIdx.x * 32 + ch] = __float2bfloat16(s);
    }
  }
  if (trigger == 2) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

void run(torch::Tensor x, torch::Tensor f2, torch::Tensor out, int64_t trigger, bool pdl) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.size(1) == HIN && x.stride(1) == 1
              && x.size(0) >= 1 && x.size(0) <= 8 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0
              && (x.stride(0) * 2) % 16 == 0, "dfa: rows must be bf16 [1..8][6144], 16-byte aligned");
  TORCH_CHECK(f2.is_cuda() && f2.is_contiguous() && f2.numel() * f2.element_size() == int64_t(HOUT) * HIN * 2
              && reinterpret_cast<uintptr_t>(f2.data_ptr()) % 16 == 0, "dfa: fragment-major weight");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.dim() == 2
              && out.size(0) == x.size(0) && out.size(1) == HOUT, "dfa: output must be bf16 [rows][2624]");
  c10::cuda::CUDAGuard guard(x.device());
  const char* xp = static_cast<const char*>(x.data_ptr()); const char* fp = static_cast<const char*>(f2.data_ptr());
  bf16* op = static_cast<bf16*>(out.data_ptr());
  long long xst = x.stride(0) * 2;
  int T = int(x.size(0)), tg = int(trigger), pd = pdl ? 1 : 0;
  void* args[] = {&xp, &xst, &fp, &op, &T, &tg, &pd};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(CTAS); cfg.blockDim = dim3(128);
  cfg.stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr; cfg.numAttrs = pdl ? 1 : 0;
#define DFA_LAUNCH(TM)                                                                                              \
  {                                                                                                                 \
    cfg.dynamicSmemBytes = Cfg<TM>::SMEM;                                                                           \
    TORCH_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(fused_a_kernel<TM>),                             \
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, int(Cfg<TM>::SMEM)) == cudaSuccess, \
                "dfa: shared memory");                                                                              \
    const cudaError_t e = cudaLaunchKernelExC(&cfg, reinterpret_cast<const void*>(fused_a_kernel<TM>), args);       \
    TORCH_CHECK(e == cudaSuccess, "dfa launch: ", cudaGetErrorString(e));                                           \
  }
  if (T <= 4) DFA_LAUNCH(4) else DFA_LAUNCH(8)
#undef DFA_LAUNCH
}
int64_t depth(int64_t tmax) { return tmax <= 4 ? Cfg<4>::DEPTH : Cfg<8>::DEPTH; }
}   

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dfa_run", &dfa::run);
  m.def("dfa_depth", &dfa::depth); m.def("decode", &dqab::decode); }
