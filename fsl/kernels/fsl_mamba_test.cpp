// ============================================================
// fsl_mamba_test.cpp — Tests for FSL Mamba step kernels
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

// ============================================================
// External declarations
// ============================================================

extern "C" void fsl_mamba_step(
    const float* state,
    const float* input,
    const float* matrix_a,
    const float* matrix_b,
    float* next_state,
    float* output,
    size_t n,
    size_t m
);

extern "C" void fsl_selective_mamba_step(
    const float* state,
    const float* input,
    const float* A_log,
    const float* B_full,
    const float* W_conv,
    float* next_state,
    float* output,
    size_t n,
    size_t m,
    size_t d_c
);

extern "C" void fsl_output_projection(
    const float* state,
    const float* input,
    const float* matrix_c,
    const float* matrix_d,
    float* output,
    size_t n,
    size_t m
);

extern "C" int fsl_fsm_transition(
    int from_state,
    int to_state,
    int condition
);

extern "C" int fsl_scan_complete(
    const float* state,
    size_t n,
    float epsilon
);

// ============================================================
// Test helpers
// ============================================================

static const float EPSILON = 1e-6f;

static bool approx_equal(float a, float b, float eps = EPSILON) {
    return std::fabs(a - b) < eps;
}

static bool vec_equal(const float* a, const float* b, size_t n, float eps = EPSILON) {
    for (size_t i = 0; i < n; ++i) {
        if (!approx_equal(a[i], b[i], eps)) return false;
    }
    return true;
}

static bool vec_zero(const float* a, size_t n, float eps = EPSILON) {
    for (size_t i = 0; i < n; ++i) {
        if (!approx_equal(a[i], 0.0f, eps)) return false;
    }
    return true;
}

// ============================================================
// Test 1: Basic Mamba step with identity A, zero B
// ============================================================

static int test_basic_identity() {
    printf("Test 1: Basic Mamba step (A=I, B=0)...\n");

    const size_t n = 16;
    const size_t m = 512;

    float state[n];
    float input[m];
    float matrix_a[n * n];
    float matrix_b[n * m];
    float next_state[n];
    float output[m];

    // Initialize state
    for (size_t i = 0; i < n; ++i) state[i] = (float)i;

    // Zero input
    std::memset(input, 0, m * sizeof(float));

    // A = identity
    std::memset(matrix_a, 0, n * n * sizeof(float));
    for (size_t i = 0; i < n; ++i) matrix_a[i * n + i] = 1.0f;

    // B = zero
    std::memset(matrix_b, 0, n * m * sizeof(float));

    // Run kernel
    fsl_mamba_step(state, input, matrix_a, matrix_b, next_state, output, n, m);

    // Verify: next_state == state (A=I, B=0)
    bool state_ok = vec_equal(next_state, state, n);
    bool output_ok = vec_zero(output, m);

    printf("  State: %s\n", state_ok ? "PASS" : "FAIL");
    printf("  Output: %s\n", output_ok ? "PASS" : "FAIL");

    return (state_ok && output_ok) ? 0 : 1;
}

// ============================================================
// Test 2: Basic Mamba step with zero state, non-zero input
// ============================================================

static int test_basic_input_response() {
    printf("Test 2: Basic Mamba step (A=0, B=I)...\n");

    const size_t n = 16;
    const size_t m = 512;

    float state[n];
    float input[m];
    float matrix_a[n * n];
    float matrix_b[n * m];
    float next_state[n];
    float output[m];

    // Zero state
    std::memset(state, 0, n * sizeof(float));

    // Input: first element = 1
    std::memset(input, 0, m * sizeof(float));
    input[0] = 1.0f;

    // A = zero
    std::memset(matrix_a, 0, n * n * sizeof(float));

    // B = [I_n | 0] (first n columns of identity)
    std::memset(matrix_b, 0, n * m * sizeof(float));
    for (size_t i = 0; i < n; ++i) {
        matrix_b[i * m + i] = 1.0f;
    }

    // Run kernel
    fsl_mamba_step(state, input, matrix_a, matrix_b, next_state, output, n, m);

    // Verify: next_state[0] = 1, others = 0
    bool state_ok = true;
    for (size_t i = 0; i < n; ++i) {
        float expected = (i == 0) ? 1.0f : 0.0f;
        if (!approx_equal(next_state[i], expected)) {
            state_ok = false;
            break;
        }
    }
    bool output_ok = vec_zero(output, m);

    printf("  State: %s\n", state_ok ? "PASS" : "FAIL");
    printf("  Output: %s\n", output_ok ? "PASS" : "FAIL");

    return (state_ok && output_ok) ? 0 : 1;
}

// ============================================================
// Test 3: Selective Mamba step with zero A_log, zero B, zero W
// ============================================================

static int test_selective_zero_params() {
    printf("Test 3: Selective Mamba step (A=0, B=0, W=0)...\n");

    const size_t n = 16;
    const size_t m = 512;
    const size_t d_c = 4;

    float state[n];
    float input[m];
    float A_log[n];
    float B_full[n * m];
    float W_conv[m * d_c];
    float next_state[n];
    float output[m];

    // State = [1, 2, ..., n]
    for (size_t i = 0; i < n; ++i) state[i] = (float)(i + 1);

    // Input = [1, 0, ..., 0]
    std::memset(input, 0, m * sizeof(float));
    input[0] = 1.0f;

    // A_log = 0 → A = diag(-exp(0)) = diag(-1)
    std::memset(A_log, 0, n * sizeof(float));

    // B = 0
    std::memset(B_full, 0, n * m * sizeof(float));

    // W_conv = 0
    std::memset(W_conv, 0, m * d_c * sizeof(float));

    // Run kernel
    fsl_selective_mamba_step(state, input, A_log, B_full, W_conv,
                             next_state, output, n, m, d_c);

    // Verify: next_state = -state (A = -I, B*u = 0)
    bool state_ok = true;
    for (size_t i = 0; i < n; ++i) {
        if (!approx_equal(next_state[i], -state[i])) {
            state_ok = false;
            printf("  next_state[%zu] = %f, expected %f\n", i, next_state[i], -state[i]);
            break;
        }
    }
    bool output_ok = vec_zero(output, m);

    printf("  State: %s\n", state_ok ? "PASS" : "FAIL");
    printf("  Output: %s\n", output_ok ? "PASS" : "FAIL");

    return (state_ok && output_ok) ? 0 : 1;
}

// ============================================================
// Test 4: FSM transition
// ============================================================

static int test_fsm_transition() {
    printf("Test 4: FSM transition...\n");

    // State 0 → State 1 if condition true
    int result1 = fsl_fsm_transition(0, 1, 1);
    int result2 = fsl_fsm_transition(0, 1, 0);

    bool ok1 = (result1 == 1);  // Condition true → transition
    bool ok2 = (result2 == 0);  // Condition false → stay

    printf("  Transition (true): %s\n", ok1 ? "PASS" : "FAIL");
    printf("  Transition (false): %s\n", ok2 ? "PASS" : "FAIL");

    return (ok1 && ok2) ? 0 : 1;
}

// ============================================================
// Test 5: Scan complete check
// ============================================================

static int test_scan_complete() {
    printf("Test 5: Scan complete check...\n");

    const size_t n = 16;

    // State = small values → converged
    float state_converged[n];
    for (size_t i = 0; i < n; ++i) state_converged[i] = 1e-8f;
    int result1 = fsl_scan_complete(state_converged, n, 1e-6f);

    // State = large values → not converged
    float state_large[n];
    for (size_t i = 0; i < n; ++i) state_large[i] = 1.0f;
    int result2 = fsl_scan_complete(state_large, n, 1e-6f);

    bool ok1 = (result1 == 1);  // Converged
    bool ok2 = (result2 == 0);  // Not converged

    printf("  Converged: %s\n", ok1 ? "PASS" : "FAIL");
    printf("  Not converged: %s\n", ok2 ? "PASS" : "FAIL");

    return (ok1 && ok2) ? 0 : 1;
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("=== FSL Mamba Step Kernel Tests ===\n\n");

    int failures = 0;
    failures += test_basic_identity();
    failures += test_basic_input_response();
    failures += test_selective_zero_params();
    failures += test_fsm_transition();
    failures += test_scan_complete();

    printf("\n=== Results: %d failures ===\n", failures);
    return failures;
}
