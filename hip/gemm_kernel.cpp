#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using half_t = _Float16;

template <int BlockThreads>
__global__ void gemm16x16_mfma(
    const half_t* __restrict__ A,
    const half_t* __restrict__ B,
    const float* __restrict__ C,
    float* __restrict__ D,
    int M,
    int N,
    int K)
{
    using namespace rocwmma;

    constexpr int WM = 16;
    constexpr int WN = 16;
    constexpr int WK = 16;
    constexpr int warpSize = hipWarpSize; // 64 for AMD
    constexpr int WavesPerBlock = BlockThreads / warpSize;

    static_assert(BlockThreads % warpSize == 0);

    const int tid = threadIdx.x;
    const int wave = tid / warpSize; // which wave in the block

    const int tileM = (blockIdx.y * WavesPerBlock + wave) * WM;
    const int tileN = blockIdx.x * WN;

    extern __shared__ unsigned char smemRaw[];

    // Allocate shared memory for A and B tiles for all waves in the block
    auto* ldsA = reinterpret_cast<half_t*>(smemRaw);
    auto* ldsB = ldsA + WavesPerBlock * WM * WK;

    // Pointers to the current wave's A and B tiles in shared memory
    half_t* waveA = ldsA + wave * WM * WK;
    half_t* waveB = ldsB + wave * WK * WN;

    // Accumulator fragment for this wave (initialized to zero)
    fragment<accumulator, WM, WN, WK, float> acc;
    fill_fragment(acc, 0.0f);

    // Load C tile for this wave from global memory (if in bounds)
    if (tileM < M && tileN < N) {
        // The C tile is at [tileM:tileM+WM, tileN:tileN+WN]
        // Leading dimension is N (the number of columns in the matrix)
        load_matrix_sync(acc, C + tileM * N + tileN, N, mem_row_major);
    }
    // If out of bounds, we leave the accumulator as zero (which is correct for out-of-bounds output)

    // Loop over K in steps of WK (16)
    for (int kBase = 0; kBase < K; kBase += WK) {
        // Load A tile for this wave: [tileM:tileM+WM, kBase:kBase+WK]
        for (int idx = tid; idx < WavesPerBlock * WM * WK; idx += BlockThreads) {
            const int ownerWave = idx / (WM * WK);
            const int local = idx % (WM * WK);
            const int row = local / WK;
            const int col = local % WK;

            const int globalM = (blockIdx.y * WavesPerBlock + ownerWave) * WM + row;
            const int globalK = kBase + col;

            // Check bounds for A
            half_t val = half_t(0);
            if (globalM < M && globalK < K) {
                val = A[globalM * K + globalK];
            }
            ldsA[idx] = val;
        }

        // Load B tile for this wave: [kBase:kBase+WK, tileN:tileN+WN]
        for (int idx = tid; idx < WavesPerBlock * WK * WN; idx += BlockThreads) {
            const int ownerWave = idx / (WK * WN);
            const int local = idx % (WK * WN);
            const int row = local / WN;
            const int col = local % WN;

            const int globalK = kBase + row;
            const int globalN = blockIdx.x * WN + col;

            half_t val = half_t(0);
            if (globalK < K && globalN < N) {
                val = B[globalK * N + globalN];
            }
            ldsB[idx] = val;
        }

        // Make sure all waves have finished loading their A and B tiles
        __syncthreads();

        // Declare fragments for A and B for this wave
        fragment<matrix_a, WM, WN, WK, half_t, row_major> a;
        fragment<matrix_b, WM, WN, WK, half_t, col_major> b;

        // Load the A and B tiles from shared memory into fragments
        load_matrix_sync(a, waveA, WK); // lda = WK (number of columns in the A tile)
        load_matrix_sync(b, waveB, WN); // ldb = WN (number of columns in the B tile)

        // Perform the MFMA: acc = acc + a * b
        mfma_sync(acc, a, b, acc);

        // Make sure all waves have finished the MFMA before we overwrite the shared memory in the next iteration
        __syncthreads();
    }

    // Store the accumulator tile to global memory (if in bounds)
    if (tileM < M && tileN < N) {
        store_matrix_sync(D + tileM * N + tileN, acc, N, mem_row_major);
    }
    // If out of bounds, we do nothing (the output is not written, which is correct)
}

// Host Launch Example
// To launch the kernel for matrices of size M x K and K x N, producing M x N output:
//
// constexpr int BlockThreads = 256; // Must be multiple of 64 (warp size)
// dim3 block(BlockThreads);
// dim3 grid(
//     (N + 15) / 16, // number of 16x16 tiles in N dimension
//     (M + 15) / 16 / (BlockThreads / 64) // number of 16x16 tiles in M dimension per wave, divided by waves per block
// );
// gemm16x16_mfma<BlockThreads><<<grid, block>>>(d_A, d_B, d_C, d_D, M, N, K);
// hipDeviceSynchronize();
//
// Compilation: hipcc -std=c++17 -offload-arch=gfx942 gemm_kernel.cpp -o gemm -lrocwmma