#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

// Host test harness
void run_test(int test_case) {
    const int M = 16, N = 16, K = 16;
    const size_t A_size = M * K;
    const size_t B_size = K * N;
    const size_t C_size = M * N;
    const size_t D_size = M * N;

    half_t *h_A = (half_t*)malloc(A_size * sizeof(half_t));
    half_t *h_B = (half_t*)malloc(B_size * sizeof(half_t));
    float *h_C = (float*)malloc(C_size * sizeof(float));
    float *h_D = (float*)malloc(D_size * sizeof(float));
    float *h_D_ref = (float*)malloc(D_size * sizeof(float));

    // Initialize to zero
    memset(h_A, 0, A_size * sizeof(half_t));
    memset(h_B, 0, B_size * sizeof(half_t));
    memset(h_C, 0, C_size * sizeof(float));

    // Set values based on test case
    half_t inf = __float2half(INFINITY);
    half_t neg_inf = __float2half(-INFINITY);
    half_t nan = __float2half(NAN); // quiet NaN
    float nanf = NAN;

    switch (test_case) {
        case 0: // Normal
            for (size_t i = 0; i < A_size; i++) h_A[i] = __float2half(1.0f);
            for (size_t i = 0; i < B_size; i++) h_B[i] = __float2half(1.0f);
            break;
        case 1: // NaN in A at [0,0]
            h_A[0] = nan;
            break;
        case 2: // NaN in B at [0,0]
            h_B[0] = nan;
            break;
        case 3: // NaN in C at [0,0]
            h_C[0] = nanf;
            break;
        case 4: // 0 * Inf: A[0,0]=0, B[0,0]=Inf
            // h_A[0] is already 0
            h_B[0] = inf;
            break;
        case 5: // Inf * 0: A[0,0]=Inf, B[0,0]=0
            h_A[0] = inf;
            break;
        case 6: // +Inf + -Inf
            h_A[0] = __float2half(1.0f); // A[0,0]
            h_B[0] = inf; // B[0,0]
            h_A[1] = __float2half(1.0f); // A[0,1] (since K=16, A[0,1] is at index 1)
            h_B[16] = neg_inf; // B[1,0] (B is [K][N], so B[1,0] is at index 1*N+0 = 16)
            break;
        default:
            printf("Invalid test case %d\n", test_case);
            free(h_A); free(h_B); free(h_C); free(h_D); free(h_D_ref);
            return;
    }

    // Allocate device memory
    half_t *d_A, *d_B;
    float *d_C, *d_D;
    hipMalloc(&d_A, A_size * sizeof(half_t));
    hipMalloc(&d_B, B_size * sizeof(half_t));
    hipMalloc(&d_C, C_size * sizeof(float));
    hipMalloc(&d_D, D_size * sizeof(float));
    hipMemcpy(d_A, h_A, A_size * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, B_size * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C, C_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_D, 0, D_size * sizeof(float)); // initialize D to zero

    // Launch kernel
    constexpr int BlockThreads = 256; // must be multiple of 64
    const int warpSize = hipWarpSize;
    const int WavesPerBlock = BlockThreads / warpSize;
    dim3 block(BlockThreads);
    dim3 grid(
        (N + 15) / 16, // grid.x: ceil(N / 16.0)
        (M + 16 * WavesPerBlock - 1) / (16 * WavesPerBlock) // grid.y: ceil(M / (16.0 * WavesPerBlock))
    );

    gemm16x16_mfma<BlockThreads><<<grid, block>>>(d_A, d_B, d_C, d_D, M, N, K);
    hipDeviceSynchronize();

    // Copy D back to host
    hipMemcpy(h_D, d_D, D_size * sizeof(float), hipMemcpyDeviceToHost);

    // Compute reference on host
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = h_C[m * N + n]; // C is float*
            for (int k = 0; k < K; k++) {
                half_t a = h_A[m * K + k];
                half_t b = h_B[k * N + n];
                float product = __half2float(__hmul(a, b));
                acc += product;
            }
            h_D_ref[m * N + n] = acc;
        }
    }

    // Compare
    bool passed = true;
    for (size_t i = 0; i < D_size; i++) {
        float ref = h_D_ref[i];
        float res = h_D[i];
        if (std::isnan(ref)) {
            if (!std::isnan(res)) {
                printf("Error at %zu: expected NaN, got %f\n", i, res);
                passed = false;
            }
        } else {
            if (std::isnan(res)) {
                printf("Error at %zu: expected %f, got NaN\n", i, ref);
                passed = false;
            } else {
                float diff = fabsf(ref - res);
                if (diff > 1e-5f) {
                    printf("Error at %zu: expected %f, got %f (diff=%f)\n", i, ref, res, diff);
                    passed = false;
                }
            }
        }
    }

    if (passed) {
        printf("Test case %d passed.\n", test_case);
    } else {
        printf("Test case %d failed.\n", test_case);
    }

    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_D); free(h_D_ref);
    hipFree(d_A); hipFree(d_B); hipFree(d_C); hipFree(d_D);
}

int main() {
    // Run all test cases
    for (int test_case = 0; test_case <= 6; test_case++) {
        run_test(test_case);
    }
    return 0;
}