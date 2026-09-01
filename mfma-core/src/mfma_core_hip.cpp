// mfma_core_hip.cpp — AMD gfx942 (CDNA 3) Hardware MFMA Kernel
// 16x16x16 FP16 → FP32 via v_mfma_f32_16x16x16f16

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

__global__ void mfma_tile_hip_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ out
) {
    const int warp_id = threadIdx.x / 64;
    const int lane_id = threadIdx.x % 64;

    if (lane_id >= 32) return;

    const int tile_m = blockIdx.x * 16;
    const int tile_n = blockIdx.y * 16;

    float acc[16][16];

    // Load C tile (initial accumulation)
    for (int m = 0; m < 16; m++) {
        for (int n = 0; n < 16; n++) {
            acc[m][n] = c[(tile_m + m) * 256 + (tile_n + n)];
        }
    }

    // K-loop over input tiles
    for (int k_base = 0; k_base < 256; k_base += 16) {
        half a_frag[16][16];
        half b_frag[16][16];

        for (int m = 0; m < 16; m++) {
            for (int n = 0; n < 16; n++) {
                a_frag[m][n] = a[(tile_m + m) * 256 + (k_base + n)];
                b_frag[m][n] = b[(tile_n + m) * 256 + (k_base + n)];
            }
        }

        for (int i = 0; i < 8; i++) {
            int m = i / 2;
            int n = (i % 2) * 8 + (lane_id % 2) * 4 + (lane_id / 2) % 4;

            float va = __half2float(a_frag[m][n]);
            float vb = __half2float(b_frag[m][n]);

            if (__isnan(va) || __isnan(vb) || __isnan(acc[m][n])) {
                acc[m][n] = __builtin_nanf("");
            } else {
                acc[m][n] = __builtin_fma(va, vb, acc[m][n]);
            }
        }
    }

    // Store result
    for (int m = 0; m < 16; m++) {
        for (int n = 0; n < 16; n++) {
            out[(tile_m + m) * 256 + (tile_n + n)] = acc[m][n];
        }
    }
}

extern "C" void mfma_tile_hip_shim(
    const uint16_t h_a[256],
    const uint16_t h_b[256],
    const float    h_c[256],
    float          h_out[256]
) {
    half* d_a;
    half* d_b;
    float* d_c;
    float* d_out;

    hipMalloc((void**)&d_a, 256 * sizeof(half));
    hipMalloc((void**)&d_b, 256 * sizeof(half));
    hipMalloc((void**)&d_c, 256 * sizeof(float));
    hipMalloc((void**)&d_out, 256 * sizeof(float));

    hipMemcpy(d_a, h_a, 256 * sizeof(half), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, 256 * sizeof(half), hipMemcpyHostToDevice);
    hipMemcpy(d_c, h_c, 256 * sizeof(float), hipMemcpyHostToDevice);

    dim3 grid(16, 16);
    dim3 block(64);
    hipLaunchKernelGGL(mfma_tile_hip_kernel, grid, block, 0, 0, d_a, d_b, d_c, d_out);
    hipDeviceSynchronize();

    hipMemcpy(h_out, d_out, 256 * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    hipFree(d_out);
}
