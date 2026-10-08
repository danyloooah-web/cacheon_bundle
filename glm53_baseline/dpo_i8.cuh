#pragma once
#include <cuda_fp16.h>

namespace dpo {
namespace fq {
constexpr int TILES = fg::NB / 8;             
constexpr int WT = 3;                         
constexpr int KB16 = fg::KSPL / 16;           
constexpr int JG = KB16 / 4;                  
constexpr int ROWJ = fg::WIDTH / 64;          
constexpr int SGRP = fg::KSPL / 128;          
constexpr int NGRP = fg::WIDTH / 128;         
constexpr int DEPTH = 4;                      
template <int MT> struct Tile {
  static constexpr int ROWS = MT * 16;
  static constexpr size_t XBYTES = size_t(ROWS) * fg::KSPL * 2;
  static constexpr size_t SMEM = XBYTES + size_t(ROWS) * fg::PROW * 4;
};
}   

__device__ __forceinline__ uint4 ldg_nc(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void i8_frag(uint32_t w, uint32_t& b0, uint32_t& b1) {
  const uint32_t u = w ^ 0x80808080u;
  b0 = __byte_perm(u, 0x64646464u, 0x5140);
  b1 = __byte_perm(u, 0x64646464u, 0x7362);
  asm("sub.f16x2 %0, %0, %1;" : "+r"(b0) : "r"(0x64806480u));
  asm("sub.f16x2 %0, %0, %1;" : "+r"(b1) : "r"(0x64806480u));
}
__device__ __forceinline__ void mma_f16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 f16_of(uint4 v) {
  uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[i]));
    const __half2 h = __float22half2_rn(f);
    w[i] = *reinterpret_cast<const uint32_t*>(&h);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}
__device__ __forceinline__ uint32_t word_of(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}
__device__ __forceinline__ float scale_of(const uint4& v, int g) {
  const uint32_t w = word_of(v, g >> 1);
  const __half h = __ushort_as_half(static_cast<unsigned short>((g & 1) ? (w >> 16) : (w & 0xFFFFu)));
  return __half2float(h);
}

template <int MT>
__global__ void __launch_bounds__(fg::THREADS, 1)
scatter_gemm_i8_kernel(const bf16* __restrict__ x, const bf16* __restrict__ res, bf16* __restrict__ gr,
                       const uint8_t* __restrict__ wq, const __half* __restrict__ sc, float* __restrict__ acc,
                       const int64_t* __restrict__ maps, int rank, int rows, int64_t x_stride, int64_t r_stride,
                       int pdl, const int64_t* __restrict__ loc, int early) {
  using T = fq::Tile<MT>;
  constexpr int ROWS = T::ROWS;
  extern __shared__ __align__(1024) char fq_smem[];
  const uint32_t sx = smem_u32(fq_smem);
  const uint32_t stage = sx + uint32_t(T::XBYTES);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nblk = blockIdx.x % fg::NBLK, ks = blockIdx.x / fg::NBLK;
  const int k0 = ks * fg::KSPL, n0 = nblk * fg::NB;
  const int tile0 = nblk * fq::TILES + warp * fq::WT;             
  if (tid < fq::TILES)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                 :: "l"(wq + (size_t(nblk * fq::TILES + tid) * fq::ROWJ + ks * fq::JG) * 512),
                    "r"(uint32_t(fq::JG * 512)) : "memory");
  const uint8_t* wl[fq::WT];
#pragma unroll
  for (int t = 0; t < fq::WT; ++t) wl[t] = wq + (size_t(tile0 + t) * fq::ROWJ + ks * fq::JG) * 512 + lane * 16;
  uint4 ring[fq::DEPTH][fq::WT];
#pragma unroll
  for (int d = 0; d < fq::DEPTH; ++d)
#pragma unroll
    for (int t = 0; t < fq::WT; ++t) ring[d][t] = ldg_nc(wl[t] + d * 512);
  uint4 s_lo[fq::WT], s_hi[fq::WT];                                 
#pragma unroll
  for (int t = 0; t < fq::WT; ++t) {
    const int ch = (tile0 + t) * 8 + 2 * (lane & 3);
    s_lo[t] = ldg_nc(sc + size_t(ch) * fq::NGRP + ks * fq::SGRP);
    s_hi[t] = ldg_nc(sc + size_t(ch + 1) * fq::NGRP + ks * fq::SGRP);
  }
  __shared__ uint64_t epoch_s;
  char* own = ring_of(maps, rank);
  uint64_t* counter = reinterpret_cast<uint64_t*>(own) + blockIdx.x;
  uint64_t prev = 0;
  if (tid == 0) prev = *counter;
  if (tid == 0) { epoch_s = prev + 1; *counter = prev + 1; }
  __syncthreads();
  const uint64_t epoch = epoch_s;
  const int slot = int(epoch % SLOTS);
  auto land_rows = [&]() {
      const int c = tid % fg::XCH;
      constexpr int PAIRS = fg::THREADS / fg::XCH;
      for (int t0 = tid / fg::XCH; t0 < (W - 1) * rows; t0 += PAIRS * fg::POLLB) {
        const char* pa[fg::POLLB];
        uint32_t ps[fg::POLLB];
        uint4 pv[fg::POLLB];
#pragma unroll
        for (int j = 0; j < fg::POLLB; ++j) {
          const int t = t0 + j * PAIRS;
          if (t < (W - 1) * rows) {
            const int d = 1 + t / rows, r = t % rows, src = (rank + d) & 3;
            pa[j] = own + fg_off(slot, src, r, k0 + c * 8);
            ps[j] = x_addr<ROWS>(sx, src * rows + r, c);
            pv[j] = ld_sys(pa[j]);
          }
        }
        bool pending;
        do {
          pending = false;
#pragma unroll
          for (int j = 0; j < fg::POLLB; ++j)
            if (t0 + j * PAIRS < (W - 1) * rows && !full(pv[j])) { pv[j] = ld_sys(pa[j]); pending = true; }
        } while (pending);
#pragma unroll
        for (int j = 0; j < fg::POLLB; ++j)
          if (t0 + j * PAIRS < (W - 1) * rows) st_shared(ps[j], f16_of(pv[j]));
      }
  };
  const int g = blockIdx.x * fg::THREADS + tid;
  auto push_res = [&]() {
    if (g < W * rows * fg::RCH) {
      const int p = g / (rows * fg::RCH), rem = g % (rows * fg::RCH);
      const int r = rem / fg::RCH, c = rem % fg::RCH;
      const bool pad = loc != nullptr && loc[r] == 0;
      const uint4 v = pad ? make_uint4(0, 0, 0, 0) : scrub(ld_cg(res + r * r_stride + p * fg::SH + c * 8));
      if (p == rank) *reinterpret_cast<uint4*>(gr + (size_t(rank) * rows + r) * fg::SH + c * 8) = v;
      else st_sys(ring_of(maps, p) + fg_off(slot, rank, r, fg::WIDTH + c * 8), v);
    }
  };
  const bool xpushed = (early & 1) != 0, early_res = (early & 2) != 0;
  if (early_res) push_res();
  {
    const uint4 empty = make_uint4(EMPTY, EMPTY, EMPTY, EMPTY);
    const int before = int((epoch + SLOTS - 1) % SLOTS);
    for (int q = tid; q < (W - 1) * fg::RM * fg::PCH; q += fg::THREADS) {
      const int t = q / fg::PCH, d = 1 + t / fg::RM, r = t % fg::RM;
      st_sys(own + fg_off(before, (rank + d) & 3, r, k0 + (nblk * fg::PCH + q % fg::PCH) * 8), empty);
    }
  }
  if (pdl) pdl_wait();
  const int gathered = W * rows;
  for (int q = tid; q < rows * fg::XCH; q += fg::THREADS) {
    const int r = q / fg::XCH;
    const bool pad = loc != nullptr && loc[r] == 0;
    st_shared(x_addr<ROWS>(sx, rank * rows + r, q % fg::XCH),
              pad ? make_uint4(0, 0, 0, 0) : f16_of(ld_cg(x + r * x_stride + k0 + (q % fg::XCH) * 8)));
  }
  if (!xpushed && tid < rows * fg::PCH) {
    const int r = tid / fg::PCH, c = nblk * fg::PCH + tid % fg::PCH;
    const bool pad = loc != nullptr && loc[r] == 0;
    const uint4 v = pad ? make_uint4(0, 0, 0, 0) : scrub(ld_cg(x + r * x_stride + k0 + c * 8));
    const size_t off = fg_off(slot, rank, r, k0 + c * 8);
#pragma unroll
    for (int d = 1; d < W; ++d) st_sys(ring_of(maps, (rank + d) & 3) + off, v);
  }
  if (!early_res) push_res();
  const bool has_res = g < (W - 1) * rows * fg::RCH;
  char* ra = own;
  bf16* rdst = gr;
  uint4 rv = make_uint4(EMPTY, EMPTY, EMPTY, EMPTY);
  {
    if (has_res) {
      const int t = g / fg::RCH, c = g % fg::RCH;
      const int d = 1 + t / rows, r = t % rows, src = (rank + d) & 3;
      ra = own + fg_off(slot, src, r, fg::WIDTH + c * 8);
      rdst = gr + (size_t(src) * rows + r) * fg::SH + c * 8;
      rv = ld_sys(ra);
    }
    land_rows();
    if (has_res && early_res) {                                     
      while (!full(rv)) rv = ld_sys(ra);
      *reinterpret_cast<uint4*>(rdst) = rv;
      st_sys(ra, make_uint4(EMPTY, EMPTY, EMPTY, EMPTY));
    }
  }
  if (pdl) pdl_trigger();
  __syncthreads();
  float accum[MT][fq::WT][4], part[MT][fq::WT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int t = 0; t < fq::WT; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) accum[m][t][e] = 0.f;
#pragma unroll
  for (int J = 0; J < fq::JG; ++J) {
    if ((J & 1) == 0) {
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int t = 0; t < fq::WT; ++t)
#pragma unroll
          for (int e = 0; e < 4; ++e) part[m][t][e] = 0.f;
    }
    uint4 cur[fq::WT];
#pragma unroll
    for (int t = 0; t < fq::WT; ++t) cur[t] = ring[J % fq::DEPTH][t];
    if (J + fq::DEPTH < fq::JG) {
#pragma unroll
      for (int t = 0; t < fq::WT; ++t) ring[J % fq::DEPTH][t] = ldg_nc(wl[t] + (J + fq::DEPTH) * 512);
    }
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = J * 4 + jj;
      uint32_t a[MT][4];
#pragma unroll
      for (int m = 0; m < MT; ++m)
        ldm_x4(x_addr<ROWS>(sx, m * 16 + (lane & 15), j * 2 + (lane >> 4)), a[m][0], a[m][1], a[m][2], a[m][3]);
#pragma unroll
      for (int t = 0; t < fq::WT; ++t) {
        uint32_t b0, b1;
        i8_frag(word_of(cur[t], jj), b0, b1);
#pragma unroll
        for (int m = 0; m < MT; ++m) mma_f16(part[m][t], a[m][0], a[m][1], a[m][2], a[m][3], b0, b1);
      }
    }
    if (J & 1) {
      const int grp = J >> 1;
#pragma unroll
      for (int t = 0; t < fq::WT; ++t) {
        const float s0 = scale_of(s_lo[t], grp), s1 = scale_of(s_hi[t], grp);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          accum[m][t][0] = fmaf(part[m][t][0], s0, accum[m][t][0]);
          accum[m][t][1] = fmaf(part[m][t][1], s1, accum[m][t][1]);
          accum[m][t][2] = fmaf(part[m][t][2], s0, accum[m][t][2]);
          accum[m][t][3] = fmaf(part[m][t][3], s1, accum[m][t][3]);
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int t = 0; t < fq::WT; ++t) {
      const int row = m * 16 + (lane >> 2);
      const int col = warp * fq::WT * 8 + t * 8 + (lane & 3) * 2;
      if (row < gathered)
        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(stage + uint32_t(row * fg::PROW + col) * 4),
                     "f"(accum[m][t][0]), "f"(accum[m][t][1]) : "memory");
      if (row + 8 < gathered)
        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(stage + uint32_t((row + 8) * fg::PROW + col) * 4),
                     "f"(accum[m][t][2]), "f"(accum[m][t][3]) : "memory");
    }
  __syncthreads();
  reduce_partial(stage, acc, gathered, n0);
  if (has_res && !early_res) {
    while (!full(rv)) rv = ld_sys(ra);
    *reinterpret_cast<uint4*>(rdst) = rv;
    st_sys(ra, make_uint4(EMPTY, EMPTY, EMPTY, EMPTY));
  }
}

void scatter_gemm_i8(torch::Tensor x, torch::Tensor res, torch::Tensor gr, torch::Tensor wq, torch::Tensor sc,
                     torch::Tensor acc, torch::Tensor maps, int64_t rank, bool pdl, torch::Tensor loc, int64_t early) {
  check_mat(x); check_mat(res); check_mat(gr); check_maps(maps, rank, x);
  const int rows = x.size(0);
  TORCH_CHECK(rows > 0 && rows <= 8 && x.size(1) == fg::WIDTH && res.size(0) == rows && res.size(1) == W * fg::SH,
              "int8 fused projection geometry");
  TORCH_CHECK(gr.is_contiguous() && gr.size(0) == W * rows && gr.size(1) == fg::SH, "gathered residual geometry");
  TORCH_CHECK(wq.is_cuda() && wq.is_contiguous() && wq.scalar_type() == at::kByte
              && wq.numel() == int64_t(fg::SH) * fg::WIDTH && reinterpret_cast<uintptr_t>(wq.data_ptr()) % 128 == 0,
              "int8 shard");
  TORCH_CHECK(sc.is_cuda() && sc.is_contiguous() && sc.scalar_type() == at::kHalf
              && sc.numel() == int64_t(fg::SH) * fq::NGRP && reinterpret_cast<uintptr_t>(sc.data_ptr()) % 16 == 0,
              "int8 shard scales");
  TORCH_CHECK(acc.is_cuda() && acc.is_contiguous() && acc.scalar_type() == at::kFloat && acc.size(1) == fg::SH
              && acc.size(0) >= W * rows, "accumulator");
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const bf16* xp = static_cast<const bf16*>(x.data_ptr()); const bf16* rp = static_cast<const bf16*>(res.data_ptr());
  bf16* grp = static_cast<bf16*>(gr.data_ptr());
  const uint8_t* wp = wq.data_ptr<uint8_t>(); const __half* sp = reinterpret_cast<const __half*>(sc.data_ptr());
  float* ap = acc.data_ptr<float>(); const int64_t* mp = maps.data_ptr<int64_t>();
  int rk = int(rank), rw = rows, pd = pdl ? 1 : 0;
  int64_t xs = x.stride(0), rs = res.stride(0);
  const bool has_loc = loc.numel() > 0;
  if (has_loc)
    TORCH_CHECK(loc.is_cuda() && loc.scalar_type() == at::kLong && loc.dim() == 1 && loc.is_contiguous()
                && loc.size(0) == rows, "cache slots must be int64 [rows]");
  const int64_t* lp = has_loc ? loc.data_ptr<int64_t>() : nullptr;
  int ea = int(early);
  void* args[] = {&xp, &rp, &grp, &wp, &sp, &ap, &mp, &rk, &rw, &xs, &rs, &pd, &lp, &ea};
#define FQ_LAUNCH(MT_)                                                                                         \
  {                                                                                                            \
    using T = fq::Tile<MT_>;                                                                                   \
    const cudaError_t a = cudaFuncSetAttribute(reinterpret_cast<const void*>(scatter_gemm_i8_kernel<MT_>),     \
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, int(T::SMEM));     \
    TORCH_CHECK(a == cudaSuccess, "int8 fused projection shared memory: ", cudaGetErrorString(a));             \
    launch_pdl(pdl, 1, stream, dim3(fg::GRID), dim3(fg::THREADS), scatter_gemm_i8_kernel<MT_>, args, T::SMEM); \
  }
  if (W * rows <= 16) FQ_LAUNCH(1)
  else FQ_LAUNCH(2)
#undef FQ_LAUNCH
}

}   
