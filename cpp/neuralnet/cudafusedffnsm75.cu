// Hand-written SM75 (Turing) dual-GEMM + SwiGLU implementation of CudaFusedFFNSm75 (see
// cudafusedffnsm75.h). Computes out = SiLU(A @ W1) * (A @ Wgate) in a single kernel so the two
// GEMM intermediates never touch global memory. This is the sm_75 counterpart of the
// CUTLASS-based CudaFusedFFN (cudafusedffn.cu), whose dual-GEMM pipeline is built on sm_80's
// cp.async and cannot serve Turing GPUs.
//
// Design notes, for T4-class GPUs (sm_75: tensor cores and ldmatrix, no cp.async, 64KB smem/SM):
//   - Threadblock tile 128 tokens x 64 out-channels x 32 k per pipeline step, 256 threads
//     (8 warps). Each warp computes a 32x32 sub-tile held as 2 m16-tiles x 4 n8-tiles of
//     mma.sync.aligned.m16n8k8.f16 accumulators. The f16-accumulate variant is 2x the f32
//     variant's throughput on Turing and matches the precision class of both code paths it
//     can replace: the unfused cublasHgemm path and the CUTLASS fused kernel (sm_80+).
//   - All fragments come from shared memory via non-transposed ldmatrix.x4. Weights are
//     stored out-major ([n][k] row-major, the same packed layout the CUTLASS path uploads),
//     and a non-transposed ldmatrix over such a tile yields exactly the col-major B fragment
//     mma.row.col expects, so no transpose pass is needed.
//   - No cp.async: a two-level software pipeline. Each k-step's global loads (uint4) are
//     issued into per-thread register staging a full k-step ahead and stored to the
//     double-buffered shared tiles only after the mma batch reading the other buffer has
//     retired; one __syncthreads per k-step. The k-loop is manually unrolled in even/odd
//     pairs so the staging arrays are only ever indexed with compile-time constants and stay
//     in registers.
//   - Shared memory: 2 x (128x40 + 64x40 + 64x40) halfs = 40KB, under the 48KB static limit,
//     so no opt-in attribute is needed. Row stride 40 = 32 + 8 halfs keeps every row
//     16-byte aligned and the ldmatrix loads bank-conflict free: their lane addresses
//     always cover 8 consecutive rows of one k-segment, and 40 halfs = 20 banks means 8
//     such rows sweep all 32 banks exactly once. The uint4 stores use a different mapping
//     (2 rows x 4 k-segments per coalescing phase) and are two-way conflicted at worst -
//     chosen deliberately over a conflict-free but 2x-less-coalesced global-load mapping,
//     and not a bottleneck (total shared traffic stays well under the mma issue time).
//   - Epilogue: SiLU is evaluated in FP32 with the same expression as siluf in
//     cudaandrocmhelpers.inc and the product with the gate is rounded to half once, which is
//     bit-identical to the unfused sequence (cublasHgemm x2 then customCudaSwiGLU).

#include "../neuralnet/cudafusedffnsm75.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

constexpr int BM = 128;     // tokens (M) per block
constexpr int BN = 64;      // out channels (N) per block
constexpr int BK = 32;      // k-extent per pipeline step
constexpr int ST = BK + 8;  // padded shared tile row stride in halfs (see design notes)
constexpr int NTHREADS = 256;

static_assert(BM * BK / 8 == 2 * NTHREADS, "A tile must map exactly two uint4 chunks per thread");
static_assert(BN * BK / 8 == NTHREADS, "B tile must map exactly one uint4 chunk per thread");
static_assert(BM % 16 == 0 && BN % 8 == 0 && BK % 8 == 0, "mma fragment tiling");
static_assert(BK == 32, "pipeline staging sizes assume 32 halfs of k per step");

// D += A x B, all FP16 operands with FP16 accumulation. d is a register pair (two packed
// halfs), a a register pair, b a single register, following the m16n8k8 f16-variant fragment
// layouts: a = rows (gr, gr+8) x cols (q4*2, q4*2+1) of the m16xk8 tile, b = rows (k pairs)
// x col gr of the k8xn8 tile, d laid out like a over m16xn8.
__device__ __forceinline__ void mmaF16Acc1688(uint32_t& d0, uint32_t& d1, uint32_t a0, uint32_t a1, uint32_t b) {
#if __CUDA_ARCH__ >= 750
  asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
    "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
    : "+r"(d0), "+r"(d1)
    : "r"(a0), "r"(a1), "r"(b));
#else
  (void)d0; (void)d1; (void)a0; (void)a1; (void)b;
#endif
}

__device__ __forceinline__ void ldmatrixX4(
  uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const half* rowPtr
) {
#if __CUDA_ARCH__ >= 750
  uint32_t addr = (uint32_t)__cvta_generic_to_shared(rowPtr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
#else
  (void)r0; (void)r1; (void)r2; (void)r3; (void)rowPtr;
#endif
}

// SiLU in FP32, the same expression as siluf in cudaandrocmhelpers.inc. Keep the two in sync
// so the fused epilogue stays bit-identical to the unfused customCudaSwiGLU path.
__device__ __forceinline__ float sm75Siluf(float x) {
  return x / (1.0f + expf(-x));
}

__global__ __launch_bounds__(NTHREADS)
void fusedFfnSm75Kernel(
  const half* __restrict__ A, const half* __restrict__ w1, const half* __restrict__ wGate,
  half* __restrict__ out, int M, int N, int K
) {
#if __CUDA_ARCH__ >= 750
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  // Warp sub-tile within the block tile: rows [warpM*32, warpM*32+32) x cols
  // [warpN*32, warpN*32+32), held as 2 m16-tiles x 4 n8-tiles of mma accumulators.
  const int warpM = warp >> 1;
  const int warpN = warp & 1;
  const int gr = lane >> 2;       // fragment row within an 8-row group (+8 via the second reg)
  const int qc = (lane & 3) * 2;  // fragment column pair within an 8-col tile

  const int mBlock = blockIdx.x * BM;
  const int nBlock = blockIdx.y * BN;
  const int numKSteps = K / BK;

  // Double-buffered k-step tiles: buffer (ks & 1) holds the tile the mma batch at k-step ks
  // reads.
  __shared__ __align__(16) half aTiles[2][BM * ST];
  __shared__ __align__(16) half b1Tiles[2][BN * ST];
  __shared__ __align__(16) half bgTiles[2][BN * ST];

  // Per-thread chunk geometry of one k-step tile: a tile row spans BK/8 = 4 uint4 chunks.
  // Thread t takes A chunks t and t + NTHREADS (rows 0..63 and 64..127) and chunk t of each B
  // tile, so each warp's 32 lanes sweep contiguous 16-byte source segments (fully coalesced
  // global loads); the matching stores are two-way bank-conflicted at worst with this
  // stride-40 layout, a deliberate trade for the coalescing (see the design notes above).
  const int aRowLo = tid >> 2;
  const int aRowHi = aRowLo + BM / 2;
  const int aKSeg = (tid & 3) * 8;
  const int bRow = tid >> 2;
  const int bKSeg = (tid & 3) * 8;

  // Register staging, two ping-pong sets so one k-step's global load is always in flight while
  // the previous k-step's mma batch runs. Set s holds k-step-parity-s tiles; every access uses
  // a literal 0/1 index (the k-loop below is unrolled in even/odd pairs), keeping these in
  // registers. The lambdas take the set by reference so the call sites pass st[0]/st[1]
  // literally.
  uint4 st[2][4];

  // Issue k-step ks's global loads into stage set stg. A rows past M zero-fill: their
  // accumulators are never stored, and zero-fill avoids both out-of-bounds reads and garbage
  // NaNs entering the mma pipeline. Weight rows always exist (supportsShape guarantees
  // N % 64 == 0, K % 32 == 0).
  auto loadTile = [&](int ks, uint4 (&stg)[4]) {
    const size_t koff = (size_t)ks * BK;
    uint4 v = make_uint4(0, 0, 0, 0);
    int row = mBlock + aRowLo;
    if(row < M)
      v = *reinterpret_cast<const uint4*>(A + (size_t)row * K + koff + aKSeg);
    stg[0] = v;
    v = make_uint4(0, 0, 0, 0);
    row = mBlock + aRowHi;
    if(row < M)
      v = *reinterpret_cast<const uint4*>(A + (size_t)row * K + koff + aKSeg);
    stg[1] = v;
    stg[2] = *reinterpret_cast<const uint4*>(w1 + (size_t)(nBlock + bRow) * K + koff + bKSeg);
    stg[3] = *reinterpret_cast<const uint4*>(wGate + (size_t)(nBlock + bRow) * K + koff + bKSeg);
  };
  auto storeTile = [&](int buf, uint4 (&stg)[4]) {
    *reinterpret_cast<uint4*>(&aTiles[buf][aRowLo * ST + aKSeg]) = stg[0];
    *reinterpret_cast<uint4*>(&aTiles[buf][aRowHi * ST + aKSeg]) = stg[1];
    *reinterpret_cast<uint4*>(&b1Tiles[buf][bRow * ST + bKSeg]) = stg[2];
    *reinterpret_cast<uint4*>(&bgTiles[buf][bRow * ST + bKSeg]) = stg[3];
  };

  // Accumulators: [m-tile][n8-tile][2 regs], x2 GEMMs (linear1, gate). In the m16n8k8 f16
  // D fragment, per lane: regs[0] = (row gr, cols qc..qc+1), regs[1] = (row gr+8, same cols),
  // cols relative to the n8 tile's left edge.
  uint32_t acc1[2][4][2];
  uint32_t accg[2][4][2];
  #pragma unroll
  for(int mt = 0; mt < 2; mt++) {
    #pragma unroll
    for(int nt = 0; nt < 4; nt++) {
      acc1[mt][nt][0] = 0u; acc1[mt][nt][1] = 0u;
      accg[mt][nt][0] = 0u; accg[mt][nt][1] = 0u;
    }
  }

  // One k-step of mma work over the double-buffered tile in smem.
  auto computeTile = [&](int buf) {
    const half* aT = aTiles[buf];
    const half* b1T = b1Tiles[buf];
    const half* bgT = bgTiles[buf];

    // A fragments: per m-tile, two ldmatrix.x4 (each m16 x k16) cover the k32 step. For an
    // x4 at (m0, k0), lane group sub = lane >> 3 supplies the addresses of sub-tile sub's
    // rows: sub-tile rows are m0 + (lane & 7) + 8*(sub & 1) and cols k0 + 8*(sub >> 1), so
    // the four results are the m16n8k8 A operands for k-slices k0 and k0+8 ({r0,r1} and
    // {r2,r3}) in fragment layout.
    uint32_t aFrag[2][4][2];
    #pragma unroll
    for(int mt = 0; mt < 2; mt++) {
      const int m0 = warpM * 32 + mt * 16;
      #pragma unroll
      for(int kh = 0; kh < 2; kh++) {
        const int sub = lane >> 3;
        const half* rp = aT + (m0 + (lane & 7) + 8 * (sub & 1)) * ST + kh * 16 + 8 * (sub >> 1);
        uint32_t r0, r1, r2, r3;
        ldmatrixX4(r0, r1, r2, r3, rp);
        aFrag[mt][kh * 2][0] = r0;
        aFrag[mt][kh * 2][1] = r1;
        aFrag[mt][kh * 2 + 1][0] = r2;
        aFrag[mt][kh * 2 + 1][1] = r3;
      }
    }

    // B fragments and mmas, one 16-col pair of the warp's 32-col window at a time to bound
    // live registers. The weights sit in smem as [n][k] rows; a non-transposed ldmatrix.x4 at
    // (n0, k0) with sub-tile rows n0 + (lane & 7) + 8*(sub >> 1) and cols k0 + 8*(sub & 1)
    // yields exactly the col-major B fragments: {r0,r1} = k-slices k0,k0+8 of n-tile n0,
    // {r2,r3} = the same for n-tile n0+8.
    #pragma unroll
    for(int np = 0; np < 2; np++) {
      const int n0 = warpN * 32 + np * 16;
      uint32_t b1Frag[2][4];
      uint32_t bgFrag[2][4];
      #pragma unroll
      for(int kh = 0; kh < 2; kh++) {
        const int sub = lane >> 3;
        const half* rp1 = b1T + (n0 + (lane & 7) + 8 * (sub >> 1)) * ST + kh * 16 + 8 * (sub & 1);
        uint32_t r0, r1, r2, r3;
        ldmatrixX4(r0, r1, r2, r3, rp1);
        b1Frag[0][kh * 2] = r0;
        b1Frag[0][kh * 2 + 1] = r1;
        b1Frag[1][kh * 2] = r2;
        b1Frag[1][kh * 2 + 1] = r3;
        const half* rpg = bgT + (n0 + (lane & 7) + 8 * (sub >> 1)) * ST + kh * 16 + 8 * (sub & 1);
        ldmatrixX4(r0, r1, r2, r3, rpg);
        bgFrag[0][kh * 2] = r0;
        bgFrag[0][kh * 2 + 1] = r1;
        bgFrag[1][kh * 2] = r2;
        bgFrag[1][kh * 2 + 1] = r3;
      }
      #pragma unroll
      for(int s = 0; s < 4; s++) {
        #pragma unroll
        for(int mt = 0; mt < 2; mt++) {
          mmaF16Acc1688(acc1[mt][np * 2][0], acc1[mt][np * 2][1],
                        aFrag[mt][s][0], aFrag[mt][s][1], b1Frag[0][s]);
          mmaF16Acc1688(acc1[mt][np * 2 + 1][0], acc1[mt][np * 2 + 1][1],
                        aFrag[mt][s][0], aFrag[mt][s][1], b1Frag[1][s]);
          mmaF16Acc1688(accg[mt][np * 2][0], accg[mt][np * 2][1],
                        aFrag[mt][s][0], aFrag[mt][s][1], bgFrag[0][s]);
          mmaF16Acc1688(accg[mt][np * 2 + 1][0], accg[mt][np * 2 + 1][1],
                        aFrag[mt][s][0], aFrag[mt][s][1], bgFrag[1][s]);
        }
      }
    }
  };

  // Prologue: k-step 0 staged through registers into buffer 0, then k-step 1's global loads
  // issued (they complete behind the barrier and the first mma batch). The barrier publishes
  // buffer 0.
  loadTile(0, st[0]);
  storeTile(0, st[0]);
  if(numKSteps > 1)
    loadTile(1, st[1]);
  __syncthreads();

  // Main pipeline loop, unrolled in even/odd k-step pairs so the staging sets are only ever
  // addressed with literal indices. Invariant entering the pair at ks (even): tile ks is in
  // buffer 0, tile ks+1 is in register set st[1], and st[0] is free. Each store targets the
  // buffer whose readers all finished before the preceding barrier (tile ks+1 overwrote
  // tile ks-1's buffer; tile ks+2 overwrites tile ks's).
  int ks = 0;
  for(; ks + 2 <= numKSteps; ks += 2) {
    if(ks + 2 < numKSteps)
      loadTile(ks + 2, st[0]);  // in flight across this whole pair
    computeTile(0);             // consumes tile ks
    storeTile(1, st[1]);        // publishes tile ks+1
    __syncthreads();
    if(ks + 3 < numKSteps)
      loadTile(ks + 3, st[1]);
    computeTile(1);             // consumes tile ks+1
    if(ks + 2 < numKSteps)
      storeTile(0, st[0]);      // publishes tile ks+2 for the next pair
    __syncthreads();
  }
  if(ks < numKSteps) {
    // Trailing odd k-step: its tile (even parity) is already in buffer 0.
    computeTile(0);
  }

  // Epilogue: out = SiLU(acc1) * accg, with SiLU and the product in FP32 and one round to
  // half - bit-identical to the unfused customCudaSwiGLU math. Rows past M are never written.
  #pragma unroll
  for(int mt = 0; mt < 2; mt++) {
    const int rowLo = mBlock + warpM * 32 + mt * 16 + gr;
    const int rowHi = rowLo + 8;
    #pragma unroll
    for(int nt = 0; nt < 4; nt++) {
      const size_t col = (size_t)nBlock + warpN * 32 + nt * 8 + qc;
      __half2 a1 = *reinterpret_cast<__half2*>(&acc1[mt][nt][0]);
      __half2 ag = *reinterpret_cast<__half2*>(&accg[mt][nt][0]);
      __half2 res = __floats2half2_rn(
        sm75Siluf(__half2float(__low2half(a1))) * __half2float(__low2half(ag)),
        sm75Siluf(__half2float(__high2half(a1))) * __half2float(__high2half(ag)));
      if(rowLo < M)
        *reinterpret_cast<__half2*>(out + (size_t)rowLo * N + col) = res;
      a1 = *reinterpret_cast<__half2*>(&acc1[mt][nt][1]);
      ag = *reinterpret_cast<__half2*>(&accg[mt][nt][1]);
      res = __floats2half2_rn(
        sm75Siluf(__half2float(__low2half(a1))) * __half2float(__low2half(ag)),
        sm75Siluf(__half2float(__high2half(a1))) * __half2float(__high2half(ag)));
      if(rowHi < M)
        *reinterpret_cast<__half2*>(out + (size_t)rowHi * N + col) = res;
    }
  }
#else
  // Pre-sm_75 builds (possible with older CUDA toolkits) compile an empty stub; the host-side
  // compute-capability gate in supportedOnCurrentDevice() keeps it from ever launching.
  (void)A; (void)w1; (void)wGate; (void)out; (void)M; (void)N; (void)K;
#endif  // __CUDA_ARCH__ >= 750
}

}  // namespace

namespace CudaFusedFFNSm75 {

bool supportedOnCurrentDevice() {
  // Claim only sm_75: the kernel is a Turing-tuned mma.sync f16 pipeline. sm_80+ takes the
  // CUTLASS kernel, and older GPUs have no tensor cores.
  {
    int device = 0;
    int major = 0;
    int minor = 0;
    if(cudaGetDevice(&device) != cudaSuccess)
      return false;
    if(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
      return false;
    if(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess)
      return false;
    if(major != 7 || minor != 5)
      return false;
  }
  // Run a real tiny dual GEMM rather than a trivial probe, mirroring CudaFusedFFN on the
  // sm_80+ side: with all-zero inputs the epilogue writes SiLU(0) * 0 == 0, so pre-filling the
  // output nonzero and checking it became zero verifies the kernel body truly executed (a
  // launch of an empty JIT stub would still report success). K = 64 exercises the pipelined
  // main loop rather than just the prologue, and M = 16 exercises the row predication.
  constexpr int M = 16;
  constexpr int N = 64;
  constexpr int K = 64;
  constexpr size_t numIn = (size_t)M * K + 2 * (size_t)N * K;
  constexpr size_t numOut = (size_t)M * N;
  half* buf = nullptr;
  if(cudaMalloc(&buf, (numIn + numOut) * sizeof(half)) != cudaSuccess)
    return false;
  half* A = buf;
  half* w1 = A + (size_t)M * K;
  half* wGate = w1 + (size_t)N * K;
  half* out = wGate + (size_t)N * K;

  bool ok = cudaMemset(buf, 0, numIn * sizeof(half)) == cudaSuccess;
  ok = ok && cudaMemset(out, 0xFF, numOut * sizeof(half)) == cudaSuccess;
  if(ok) {
    dim3 grid((M + BM - 1) / BM, N / BN);
    fusedFfnSm75Kernel<<<grid, NTHREADS>>>(A, w1, wGate, out, M, N, K);
    ok = cudaGetLastError() == cudaSuccess;
  }
  half hostOut[numOut];
  ok = ok && cudaMemcpy(hostOut, out, numOut * sizeof(half), cudaMemcpyDeviceToHost) == cudaSuccess;
  if(ok) {
    for(size_t i = 0; i < numOut; i++) {
      if(__half2float(hostOut[i]) != 0.0f) {
        ok = false;
        break;
      }
    }
  }
  // Clear any recoverable (non-sticky) launch error so a failed probe cannot leak error state
  // into later, unrelated CUDA calls.
  (void)cudaGetLastError();
  (void)cudaFree(buf);
  return ok;
}

bool supportsShape(int N, int K) {
  if(N <= 0 || K <= 0)
    return false;
  // Tile coverage is exact along N (multiples of 64) and K (multiples of 32); M is predicated
  // per row. Those divisibilities also make every uint4 global load 16-byte aligned given the
  // backend's 16-byte-aligned buffers (row strides K and N are multiples of 32 halfs).
  return N % BN == 0 && K % BK == 0;
}

void runSwiGLU(
  const half* A, const half* w1, const half* wGate, half* out,
  int M, int N, int K, cudaStream_t stream
) {
  // supportsShape was checked at model load, so enforce only the alignment it cannot: 16-byte
  // aligned operand pointers (guaranteed by the backend's cudaMalloc'd buffers).
  if((uintptr_t(A) | uintptr_t(w1) | uintptr_t(wGate) | uintptr_t(out)) & 15)
    throw std::runtime_error("CudaFusedFFNSm75::runSwiGLU: operand pointers must be 16-byte aligned");
  dim3 grid((M + BM - 1) / BM, N / BN);
  fusedFfnSm75Kernel<<<grid, NTHREADS, 0, stream>>>(A, w1, wGate, out, M, N, K);
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    // The caller checked shape support and device support at model load, so this is a genuine
    // failure (a CUDA error, possibly a sticky async one from an earlier kernel), never a
    // condition to silently fall back on.
    throw std::runtime_error(
      std::string("SM75 fused FFN kernel failed to launch (or a prior CUDA error was pending): ") +
      cudaGetErrorString(err) +
      ", M=" + std::to_string(M) + " N=" + std::to_string(N) + " K=" + std::to_string(K));
  }
}

}  // namespace CudaFusedFFNSm75
