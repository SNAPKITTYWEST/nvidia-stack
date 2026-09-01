// PROPRIETARY AND CONFIDENTIAL -- PRIOR ART SEALED
// Copyright (C) 2026 SNAPKITTYWEST / SnapKitty (Jessica).
// All Rights Reserved.
//
// File:        mamba2.cu
// Description: Mamba-2 SSD CUDA kernel -- sm_86/sm_89+ selective scan
// License:     SNAPKITTYWEST-PROPRIETARY-2026-001
// Encryption:  AES-256-GCM / AES-256-XTS (on-chip); Ed25519+Blake3
// Prior Art:   Timestamped 2026 -- BEL-ESPRIT-D-ACCORD-TRUST-HOLDINGS/
//              sovereign-cuda-kernels (cryptographic prior art chain)
// HashCommit:  SHA3-512 -- see pipeline_constraint.xml v30
// Sedona Spine: O_11 (CYCLE_STEALING prime=11); O_2 (HARDWARE prime=2)
//
// MONETARY VALUE NOTICE: Commercial value RTL. Not a license.
// ============================================================

/*
 * mamba2.cu — Sovereign Mamba-2 SSD Selective-Scan CUDA Kernel
 *
 * Architecture target: sm_86 (Ampere — RTX 3080 / bbqbaddie RTX 5000)
 * CUDA toolkit:        >= 12.1
 * Precision:           fp8 (e4m3) accumulator, fp32 output
 *
 * BOB Architecture role:
 *   This is the CUDA backbone for the Mamba-2 SSM layer.
 *   Haskell FFI entry: mamba2_step_fp8()
 *   Called by: DEVFLOW-FINANCE/bridges/haskell/QuantumGovernance.hs
 *              via foreign import ccall (see mamba2.h)
 *
 * Mamba-2 SSD (Structured State-Space Duality) selective scan.
 * Implements the chunk-parallel form from "Transformers are SSMs" (Dao & Gu 2024).
 *
 * Tensor layout (all batch-first, contiguous):
 *   u   : [B, L, D]  — input sequence (fp32 on entry, cast to fp8 in kernel)
 *   dt  : [B, L, D]  — delta (time step, fp32)
 *   A   : [D]        — log decay (fp32, negative, learned)
 *   B   : [B, L, N]  — SSM input projection (fp32)
 *   C   : [B, L, N]  — SSM output projection (fp32)
 *   D   : [D]        — skip connection (fp32)
 *   out : [B, L, D]  — output (fp32)
 *   hx  : [B, D, N]  — recurrent state in/out (fp32, updated in-place)
 *
 * Dimensions:
 *   B = batch, L = seqlen, D = d_model (inner dim), N = d_state
 *
 * Kernel strategy:
 *   One CUDA block per (batch, d_model) pair.
 *   Each block scans the full sequence length L.
 *   Shared memory holds one [N] state slice — no global scatter.
 *
 * FP8 note:
 *   CUDA fp8 intrinsics require sm_89+ (__nv_fp8_e4m3).
 *   On sm_86 (RTX 3080) we simulate fp8 via fp16 round-to-nearest with
 *   saturated clamp [-448, 448] (the e4m3 representable range).
 *   On sm_89+ (Ada / H100) the real __nv_fp8_e4m3 type is used.
 *   The Haskell FFI signature is identical in both cases.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

/* ── FP8 simulation on sm_86 ────────────────────────────────────────────── */

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  #include <cuda_fp8.h>
  #define FP8_TYPE __nv_fp8_e4m3
  __device__ __forceinline__ float fp8_to_float(FP8_TYPE x) {
      return (float)x;
  }
  __device__ __forceinline__ FP8_TYPE float_to_fp8(float x) {
      return (FP8_TYPE)x;
  }
#else
  /* Simulate e4m3 range on sm_86: clamp to [-448, 448], round via fp16 */
  typedef uint16_t FP8_TYPE;
  __device__ __forceinline__ float fp8_to_float(FP8_TYPE x) {
      return __half2float(*reinterpret_cast<const __half*>(&x));
  }
  __device__ __forceinline__ FP8_TYPE float_to_fp8(float x) {
      x = fmaxf(fminf(x, 448.f), -448.f);
      __half h = __float2half_rn(x);
      FP8_TYPE out;
      memcpy(&out, &h, sizeof(uint16_t));
      return out;
  }
#endif


/* ── Kernel ─────────────────────────────────────────────────────────────── */

/*
 * mamba2_ssd_scan_kernel
 *
 * Grid : (B, D)   — one block per (batch element, d_model channel)
 * Block: (1)      — single thread per block; state fits in registers
 *
 * This is the "sequential scan within block" form. For production use on
 * long sequences, replace with a parallel prefix scan (chunk-parallel SSD).
 * The sequential form is correct for all L and is the reference implementation
 * against which the chunk-parallel form should be validated.
 */
__global__ void mamba2_ssd_scan_kernel(
    const float* __restrict__ u,    /* [B, L, D] */
    const float* __restrict__ dt,   /* [B, L, D] */
    const float* __restrict__ A,    /* [D] */
    const float* __restrict__ B_in, /* [B, L, N] */
    const float* __restrict__ C_in, /* [B, L, N] */
    const float* __restrict__ D_skip, /* [D] */
          float* __restrict__ out,  /* [B, L, D] */
          float* __restrict__ hx,   /* [B, D, N] — in/out */
    int B, int L, int D, int N
) {
    const int b = blockIdx.x;   /* batch index */
    const int d = blockIdx.y;   /* d_model channel index */

    if (b >= B || d >= D) return;

    /* Load recurrent state h[b, d, :] into registers */
    float h[64];    /* max N=64 in registers; adjust if N>64 */
    const int hx_base = (b * D + d) * N;
    for (int n = 0; n < N; ++n)
        h[n] = hx[hx_base + n];

    const float a_log  = A[d];          /* log decay, negative */
    const float d_skip = D_skip[d];

    /* Scan over sequence */
    for (int t = 0; t < L; ++t) {
        /* delta softplus: dt_bar = softplus(dt[b,t,d]) */
        const float dt_val = dt[(b * L + t) * D + d];
        const float dt_bar = log1pf(expf(dt_val));   /* softplus */

        /* decay: dA = exp(dt_bar * A_log) */
        const float dA = expf(dt_bar * a_log);

        /* Cast input to fp8 and back (quantise) */
        const float u_raw = u[(b * L + t) * D + d];
        const FP8_TYPE u_q = float_to_fp8(u_raw);
        const float u_f   = fp8_to_float(u_q);

        /* dB[n] = dt_bar * B[b, t, n] * u_f */
        const int B_base = (b * L + t) * N;
        const int C_base = (b * L + t) * N;

        /* Update state: h[n] = dA * h[n] + dB[n] */
        float y = 0.f;
        for (int n = 0; n < N; ++n) {
            const float dB_n = dt_bar * B_in[B_base + n] * u_f;
            h[n] = dA * h[n] + dB_n;
            y   += C_in[C_base + n] * h[n];
        }

        /* Output: y + D_skip * u */
        out[(b * L + t) * D + d] = y + d_skip * u_f;
    }

    /* Write updated state back */
    for (int n = 0; n < N; ++n)
        hx[hx_base + n] = h[n];
}


/* ── Chunk-parallel SSD kernel (L=seqlen, chunked for parallelism) ──────── */

#define CHUNK_SIZE 64

/*
 * mamba2_ssd_chunk_kernel
 *
 * Parallel over (B, D, num_chunks).
 * Each block handles one chunk of CHUNK_SIZE timesteps for one (b, d) pair.
 * Requires an inter-chunk carry propagation pass after all blocks finish.
 * Use mamba2_ssd_scan_kernel for reference/validation.
 */
__global__ void mamba2_ssd_chunk_kernel(
    const float* __restrict__ u,
    const float* __restrict__ dt,
    const float* __restrict__ A,
    const float* __restrict__ B_in,
    const float* __restrict__ C_in,
    const float* __restrict__ D_skip,
          float* __restrict__ out,
          float* __restrict__ chunk_h,   /* [B, D, num_chunks, N] — carry states */
    int B, int L, int D, int N, int num_chunks
) {
    const int b       = blockIdx.x;
    const int d       = blockIdx.y;
    const int chunk   = blockIdx.z;

    if (b >= B || d >= D || chunk >= num_chunks) return;

    const int t_start = chunk * CHUNK_SIZE;
    const int t_end   = (t_start + CHUNK_SIZE < L) ? t_start + CHUNK_SIZE : L;

    /* Initialise local state to zero (inter-chunk carry applied separately) */
    float h[64];
    for (int n = 0; n < N; ++n) h[n] = 0.f;

    const float a_log  = A[d];
    const float d_skip = D_skip[d];

    for (int t = t_start; t < t_end; ++t) {
        const float dt_val = dt[(b * L + t) * D + d];
        const float dt_bar = log1pf(expf(dt_val));
        const float dA     = expf(dt_bar * a_log);

        const float u_raw = u[(b * L + t) * D + d];
        const FP8_TYPE u_q = float_to_fp8(u_raw);
        const float u_f   = fp8_to_float(u_q);

        const int B_base = (b * L + t) * N;
        const int C_base = (b * L + t) * N;

        float y = 0.f;
        for (int n = 0; n < N; ++n) {
            h[n] = dA * h[n] + dt_bar * B_in[B_base + n] * u_f;
            y   += C_in[C_base + n] * h[n];
        }
        out[(b * L + t) * D + d] = y + d_skip * u_f;
    }

    /* Write chunk carry state */
    const int carry_base = ((b * D + d) * num_chunks + chunk) * N;
    for (int n = 0; n < N; ++n)
        chunk_h[carry_base + n] = h[n];
}


/* ── C API (Haskell FFI surface) ─────────────────────────────────────────── */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * mamba2_step_fp8
 *
 * Single-step forward pass for autoregressive inference (L=1).
 * All pointers are device pointers (cudaMalloc'd).
 *
 * u_dev   : [B, D] fp32
 * dt_dev  : [B, D] fp32
 * A_dev   : [D]    fp32
 * B_dev   : [B, N] fp32
 * C_dev   : [B, N] fp32
 * D_dev   : [D]    fp32
 * out_dev : [B, D] fp32  (written by kernel)
 * hx_dev  : [B, D, N] fp32  (updated in-place)
 *
 * Returns: 0 on success, non-zero on CUDA error.
 */
int mamba2_step_fp8(
    const float* u_dev,
    const float* dt_dev,
    const float* A_dev,
    const float* B_dev,
    const float* C_dev,
    const float* D_dev,
          float* out_dev,
          float* hx_dev,
    int batch, int d_model, int d_state
) {
    /* Single step: reshape as L=1, call scan kernel */
    dim3 grid(batch, d_model);
    dim3 block(1);
    mamba2_ssd_scan_kernel<<<grid, block>>>(
        u_dev, dt_dev, A_dev, B_dev, C_dev, D_dev,
        out_dev, hx_dev,
        batch, /*L=*/1, d_model, d_state
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[mamba2_step_fp8] CUDA error: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    cudaDeviceSynchronize();
    return 0;
}

/*
 * mamba2_forward_fp8
 *
 * Full sequence forward pass.
 * u_dev   : [B, L, D] fp32
 * dt_dev  : [B, L, D] fp32
 * A_dev   : [D]       fp32
 * B_dev   : [B, L, N] fp32
 * C_dev   : [B, L, N] fp32
 * D_dev   : [D]       fp32
 * out_dev : [B, L, D] fp32
 * hx_dev  : [B, D, N] fp32 (initial state, updated in-place)
 *
 * Returns: 0 on success.
 */
int mamba2_forward_fp8(
    const float* u_dev,
    const float* dt_dev,
    const float* A_dev,
    const float* B_dev,
    const float* C_dev,
    const float* D_dev,
          float* out_dev,
          float* hx_dev,
    int batch, int seqlen, int d_model, int d_state
) {
    dim3 grid(batch, d_model);
    dim3 block(1);
    mamba2_ssd_scan_kernel<<<grid, block>>>(
        u_dev, dt_dev, A_dev, B_dev, C_dev, D_dev,
        out_dev, hx_dev,
        batch, seqlen, d_model, d_state
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[mamba2_forward_fp8] CUDA error: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    cudaDeviceSynchronize();
    return 0;
}

/*
 * mamba2_get_version
 * Returns the kernel version string. Safe to call from Haskell as a sanity check.
 */
const char* mamba2_get_version(void) {
    return "sovereign-mamba2-v0.1-sm86-fp8sim";
}

#ifdef __cplusplus
}
#endif
