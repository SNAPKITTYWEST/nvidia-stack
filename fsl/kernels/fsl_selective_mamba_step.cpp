// ============================================================
// fsl_selective_mamba_step.cpp — Selective SSM (Mamba-2)
// ============================================================
// Implements the selective state-space model step:
//   1. Depthwise convolution: z_t = Conv_{d_c}(x_t; W)
//   2. Split and SiLU gating: u_t = z1 ⊙ silu(z2)
//   3. SSM update: s_{t+1} = A * s_t + B * u_t
//   4. Zero output: y_t = 0_m
//
// YAML parameters:
//   d_state = 16 (n)
//   d_model = 512 (m)
//   d_conv = 4 (d_c)
//
// A is diagonal: A = diag(-exp(A_log))
// B is fixed (full n×m matrix)

#include <cstddef>
#include <cstring>
#include <cmath>

// ============================================================
// SiLU activation: silu(x) = x * sigmoid(x)
// ============================================================

static inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// ============================================================
// Depthwise 1D convolution (causal padding)
// ============================================================

static void depthwise_conv1d(
    const float* input,     // [m] input signal
    const float* W_conv,    // [m * d_c] convolution weights
    float* output,          // [m] output signal
    size_t m,               // model dimension
    size_t d_c              // convolution width
) {
    for (size_t i = 0; i < m; ++i) {
        float sum = 0.0f;
        for (size_t k = 0; k < d_c; ++k) {
            // Causal padding: pad with zeros on the left
            size_t idx = i + k - (d_c / 2);
            float x_val = (idx < m) ? input[idx] : 0.0f;
            sum += W_conv[i * d_c + k] * x_val;
        }
        output[i] = sum;
    }
}

// ============================================================
// Selective Mamba Step
// ============================================================

extern "C" void fsl_selective_mamba_step(
    const float* state,        // [n] current SSM state
    const float* input,        // [m] raw token (pre-convolution)
    const float* A_log,        // [n] log-space diagonal matrix
    const float* B_full,       // [n*m] full input matrix (row-major)
    const float* W_conv,       // [m*d_c] depthwise conv kernel
    float* next_state,         // [n] next SSM state (output)
    float* output,             // [m] intermediate output (zeroed)
    size_t n,                  // d_state = 16
    size_t m,                  // d_model = 512
    size_t d_c                 // d_conv = 4
) {
    // Temporary buffers (stack-allocated for small sizes)
    float z[m];            // Conv output
    float z1[m / 2];       // First half
    float z2[m / 2];       // Second half
    float u[m];            // Selective input
    float As[n];           // A * state
    float Bu[n];           // B * u

    // Step 1: Depthwise convolution
    depthwise_conv1d(input, W_conv, z, m, d_c);

    // Step 2: Split and apply SiLU gating (selectivity)
    std::memcpy(z1, z, (m / 2) * sizeof(float));
    std::memcpy(z2, z + (m / 2), (m / 2) * sizeof(float));

    // u = z1 ⊙ silu(z2)
    for (size_t i = 0; i < m / 2; ++i) {
        u[i] = z1[i] * silu(z2[i]);
    }
    // Zero-pad u to full size m
    std::memset(u + (m / 2), 0, (m / 2) * sizeof(float));

    // Step 3: Compute A * state (A = diag(-exp(A_log)))
    for (size_t i = 0; i < n; ++i) {
        As[i] = -std::exp(A_log[i]) * state[i];
    }

    // Step 4: Compute B * u (B_full is n×m, row-major)
    std::memset(Bu, 0, n * sizeof(float));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            Bu[i] += B_full[i * m + j] * u[j];
        }
    }

    // Step 5: State update: s_{t+1} = A*s + B*u
    for (size_t i = 0; i < n; ++i) {
        next_state[i] = As[i] + Bu[i];
    }

    // Step 6: Zero output (per FSTK semantics)
    std::memset(output, 0, m * sizeof(float));
}

// ============================================================
// Output projection: y_t = C * s_t + D * u_t
// ============================================================

extern "C" void fsl_output_projection(
    const float* state,        // [n] SSM state
    const float* input,        // [m] convolved input
    const float* matrix_c,     // [m*n] output matrix C (row-major)
    const float* matrix_d,     // [m*m] skip matrix D (row-major)
    float* output,             // [m] output token
    size_t n,                  // d_state = 16
    size_t m                   // d_model = 512
) {
    // y = C * s
    for (size_t i = 0; i < m; ++i) {
        output[i] = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            output[i] += matrix_c[i * n + j] * state[j];
        }
    }

    // y += D * u
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            output[i] += matrix_d[i * m + j] * input[j];
        }
    }
}

// ============================================================
// FSM transition: evaluate condition and update state
// ============================================================

extern "C" int fsl_fsm_transition(
    int from_state,            // current FSM state (integer ID)
    int to_state,              // target FSM state (integer ID)
    int condition              // boolean condition flag
) {
    // If condition is true, transition to to_state
    // Otherwise, stay in from_state
    return condition ? to_state : from_state;
}

// ============================================================
// Scan complete check: ||s||_2 < epsilon
// ============================================================

extern "C" int fsl_scan_complete(
    const float* state,        // [n] SSM state
    size_t n,                  // state dimension
    float epsilon              // convergence threshold
) {
    float norm = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        norm += state[i] * state[i];
    }
    norm = std::sqrt(norm);
    return (norm < epsilon) ? 1 : 0;
}
