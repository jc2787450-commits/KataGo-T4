// Hand-written SM75 (Turing) fused transformer FFN for the CUDA backend: one kernel computing
// SiLU(A @ W1) * (A @ Wgate) without writing the two intermediate GEMM outputs to global
// memory. This is the sm_75 counterpart of the CUTLASS-based CudaFusedFFN (cudafusedffn.h /
// cudafusedffn.cu), whose dual-GEMM pipeline is built on sm_80's cp.async and cannot run on
// Turing. The kernel uses mma.sync.aligned.m16n8k8 with FP16 accumulation (the same precision
// class as the unfused cublasHgemm path and the CUTLASS kernel) and a register-staged software
// pipeline in place of cp.async. It has no CUTLASS dependency and is compiled with every CUDA
// backend build (see cudafusedffnsm75.cu), so call sites need only the KATAGO_GPU_CUDA guard.

#ifndef NEURALNET_CUDAFUSEDFFNSM75_H_
#define NEURALNET_CUDAFUSEDFFNSM75_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace CudaFusedFFNSm75 {
  // Whether the fused FFN kernel actually works on the current device: claims only sm_75, and
  // runs a tiny real dual GEMM to verify it executed, guarding against the same hazard as
  // flashAttentionMmaSupportedOnCurrentDevice in cudaflashmma.cuh (a pre-sm_75 PTX JIT
  // compiling to an empty stub). Synchronous and slightly costly, so call once per handle
  // creation.
  bool supportedOnCurrentDevice();

  // Whether the kernel supports an FFN with weight matrices [N, K] (activations are [M, K]
  // with M varying per forward and not affecting support). Depends only on shape and
  // alignment, so callers may decide at model load time whether the fused path will be used
  // and commit to weight layouts accordingly. The layout contract matches CudaFusedFFN
  // exactly: A is [M, K] row-major (NHWC tokens), w1/wGate are packed out-major ([N, K]
  // row-major), out is [M, N] row-major, all 16-byte aligned.
  bool supportsShape(int N, int K);

  // out = SiLU(A @ W1) * (A @ Wgate), FP16 io and FP16 accumulation (matching the unfused
  // cublasHgemm path). The epilogue evaluates SiLU and the product in FP32, bit-identical to
  // the unfused customCudaSwiGLU kernel. The caller must have verified
  // supportedOnCurrentDevice() once and supportsShape(N, K) beforehand, so any failure here
  // is a genuine error and throws std::runtime_error.
  void runSwiGLU(
    const half* A, const half* w1, const half* wGate, half* out,
    int M, int N, int K, cudaStream_t stream
  );
}

#endif  // NEURALNET_CUDAFUSEDFFNSM75_H_
