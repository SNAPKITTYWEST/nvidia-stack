// mfma_core.cu — NVIDIA RTX 3080 (Ampere SM_86) Tensor Core Kernel
// 16x16x16 FP16 → FP32 via WMMA mma.sync

#include <mma.h>
#include <cuda_fp16.h>
#include <cstdint>

using namespace nvcuda;

__global__ void wmma_mfma_tile_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ out
) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);
    wmma::load_matrix_sync(c_frag, c, 16, wmma::mem_row_major);

    wmma::mma_sync(acc_frag, a_frag, b_frag, c_frag);

    wmma::store_matrix_sync(out, acc_frag, 16, wmma::mem_row_major);
}

extern "C" void mfma_tile_cuda_shim(
    const uint16_t h_a[256],
    const uint16_t h_b[256],
    const float    h_c[256],
    float          h_out[256]
) {
    half* d_a;
    half* d_b;
    float* d_c;
    float* d_out;

    cudaMalloc((void**)&d_a, 256 * sizeof(half));
    cudaMalloc((void**)&d_b, 256 * sizeof(half));
    cudaMalloc((void**)&d_c, 256 * sizeof(float));
    cudaMalloc((void**)&d_out, 256 * sizeof(float));

    cudaMemcpy(d_a, h_a, 256 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 256 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, 256 * sizeof(float), cudaMemcpyHostToDevice);

    wmma_mfma_tile_kernel<<<1, 32>>>(d_a, d_b, d_c, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, 256 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_out);
}
