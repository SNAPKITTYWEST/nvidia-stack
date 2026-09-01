// ============================================================
// fsl_mamba_step.cpp — Basic SSM state transition kernel
// ============================================================
// Implements: s_{t+1} = A * s_t + B * u_t
// Fixed A, B matrices (non-selective version).
//
// YAML parameters:
//   d_state = 16 (n)
//   d_model = 512 (m)
//
// This is a hand-rolled C implementation targeting the lowest
// publicly inspectable layer (C/C++ callable kernel).

#include <cstddef>
#include <cstring>
#include <cmath>

// ============================================================
// Basic Mamba Step: s_{t+1} = A * s_t + B * u_t
// ============================================================

extern "C" void fsl_mamba_step(
    const float* state,        // [n] current SSM state
    const float* input,        // [m] convolved input
    const float* matrix_a,     // [n*n] state matrix A (row-major)
    const float* matrix_b,     // [n*m] input matrix B (row-major)
    float* next_state,         // [n] next SSM state (output)
    float* output,             // [m] intermediate output (zeroed)
    size_t n,                  // d_state = 16
    size_t m                   // d_model = 512
) {
    // Accumulator for A*s + B*u
    float acc[n];
    std::memset(acc, 0, n * sizeof(float));

    // Compute v1 = A * state
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            acc[i] += matrix_a[i * n + j] * state[j];
        }
    }

    // Compute v2 = B * input and accumulate into acc
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            acc[i] += matrix_b[i * m + j] * input[j];
        }
    }

    // Store next_state = A*s + B*u
    std::memcpy(next_state, acc, n * sizeof(float));

    // Zero output (per FSTK semantics)
    std::memset(output, 0, m * sizeof(float));
}

// ============================================================
// Vector operations for testing
// ============================================================

extern "C" void fsl_vec_add(
    const float* a,
    const float* b,
    float* result,
    size_t n
) {
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}

extern "C" void fsl_vec_scale(
    const float* a,
    float scalar,
    float* result,
    size_t n
) {
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] * scalar;
    }
}

extern "C" float fsl_vec_norm(
    const float* a,
    size_t n
) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * a[i];
    }
    return std::sqrt(sum);
}

// ============================================================
// Matrix-vector multiply (for testing)
// ============================================================

extern "C" void fsl_matvec(
    const float* matrix,    // [n*m] row-major
    const float* vec,       // [m]
    float* result,          // [n]
    size_t n,
    size_t m
) {
    for (size_t i = 0; i < n; ++i) {
        result[i] = 0.0f;
        for (size_t j = 0; j < m; ++j) {
            result[i] += matrix[i * m + j] * vec[j];
        }
    }
}
