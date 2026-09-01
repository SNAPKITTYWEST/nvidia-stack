// ============================================================
// FSLOps.td — Operation definitions for the FSL dialect
// ============================================================
// Covers: MambaStep, SelectiveMambaStep, OutputProjection, FSMTransition.
// Hybrid continuous-discrete semantics.

#ifndef FSL_OPS
#define FSL_OPS

include "FSLDialect.td"
include "FSLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// ============================================================
// MambaStepOp — Basic SSM state transition
// ============================================================

def FSL_MambaStepOp : FSL_Op<"mamba_step", [
    NoMemoryEffect
  ]> {
  let summary = "Linear SSM state transition (fixed A, B)";
  let description = [{
    Executes one step of the state-space model recurrence:
      s_{t+1} = A * s_t + B * u_t

    This is the non-selective version where A and B are fixed
    matrices provided as explicit operands. The output is
    zeroed (output_projection is a separate op).

    Parameters from YAML:
      n = d_state = 16 (state dimension)
      m = d_model = 512 (model dimension)
  }];

  let arguments = (ins
    FSL_StateVectorType:$state,           // s_t ∈ R^n
    FSL_TokenVectorType:$input,           // u_t ∈ R^m (convolved)
    AnyType:$matrix_a,                    // A ∈ R^{n×n}
    AnyType:$matrix_b                     // B ∈ R^{n×m}
  );
  let results = (outs
    FSL_StateVectorType:$next_state,      // s_{t+1} ∈ R^n
    FSL_TokenVectorType:$output           // y_t = 0_m (placeholder)
  );

  let assemblyFormat = [{
    $state `,` $input `,` $matrix_a `,` $matrix_b
    attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

// ============================================================
// SelectiveMambaStepOp — Selective SSM (Mamba-2)
// ============================================================

def FSL_SelectiveMambaStepOp : FSL_Op<"selective_mamba_step", [
    NoMemoryEffect
  ]> {
  let summary = "Selective SSM state transition (Mamba-2 architecture)";
  let description = [{
    Executes one step of the selective state-space model:
      s_{t+1} = A * s_t + B * u_t

    where u_t is computed from the raw input via:
      1. Depthwise convolution: z_t = Conv_{d_c}(x_t; W)
      2. Split: z1 = z_t[:, :m/2], z2 = z_t[:, m/2:]
      3. SiLU gating: u_t = z1 ⊙ silu(z2)

    A is diagonal: A = diag(-exp(A_log))
    B is fixed (provided as full n×m matrix or low-rank factors)

    This implements the Mamba-2 selectivity mechanism where
    input-dependence flows through u_t, not through A/B.

    Parameters from YAML:
      n = d_state = 16
      m = d_model = 512
      d_c = d_conv = 4
  }];

  let arguments = (ins
    FSL_StateVectorType:$state,           // s_t ∈ R^n
    FSL_TokenVectorType:$input,           // x_t ∈ R^m (raw token)
    FSL_SSMMatricesType:$params           // A_log, B, W_conv, V, U
  );
  let results = (outs
    FSL_StateVectorType:$next_state,      // s_{t+1} ∈ R^n
    FSL_TokenVectorType:$output           // y_t = 0_m (placeholder)
  );

  let assemblyFormat = [{
    $state `,` $input `,` $params
    attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

// ============================================================
// OutputProjectionOp — Emit output from SSM state
// ============================================================

def FSL_OutputProjectionOp : FSL_Op<"output_projection", [
    NoMemoryEffect
  ]> {
  let summary = "Project SSM state to output token";
  let description = [{
    Projects the SSM state to an output token:
      y_t = C * s_t + D * u_t

    In Mamba-2, C and D are fixed matrices. This op is
    executed in the S1_EMIT state (per YAML FSM).

    Note: This op is separate from mamba_step to enable
    hybrid FSM semantics where emission is gated by
    discrete state transitions.
  }];

  let arguments = (ins
    FSL_StateVectorType:$state,           // s_t ∈ R^n
    FSL_TokenVectorType:$input,           // u_t ∈ R^m (optional)
    AnyType:$matrix_c,                    // C ∈ R^{m×n}
    AnyType:$matrix_d                     // D ∈ R^{m×m}
  );
  let results = (outs
    FSL_TokenVectorType:$output           // y_t ∈ R^m
  );

  let assemblyFormat = [{
    $state `,` $input `,` $matrix_c `,` $matrix_d
    attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

// ============================================================
// FSMTransitionOp — Discrete state transition
// ============================================================

def FSL_FSMTransitionOp : FSL_Op<"transition", [
    NoMemoryEffect
  ]> {
  let summary = "Discrete FSM state transition (gated by condition)";
  let description = [{
    Evaluates a transition condition and updates the FSM state.

    The condition is a boolean flag derived from the SSM state:
      condition(s) = ||s||_2 > theta  (threshold)
      condition(s) = scan_complete      (external signal)

    If the condition is true, the FSM transitions from
    from_state to to_state. Otherwise, it stays in from_state.

    This enables hybrid continuous-discrete semantics:
      - Continuous: SSM state evolves via mamba_step
      - Discrete: FSM state gates which actions are executed
  }];

  let arguments = (ins
    FSL_FSMStateType:$from_state,
    FSL_FSMStateType:$to_state,
    IntegerAttr<I1>:$condition            // boolean flag
  );
  let results = (outs
    FSL_FSMStateType:$new_state           // updated FSM state
  );

  let assemblyFormat = [{
    $from_state `->` $to_state `if` $condition
    attr-dict `:` type($new_state)
  }];
}

// ============================================================
// ScanCompleteOp — Generate scan_complete flag
// ============================================================

def FSL_ScanCompleteOp : FSL_Op<"scan_complete", [
    Pure
  ]> {
  let summary = "Check if SSM scan is complete";
  let description = [{
    Evaluates whether the SSM scan is complete based on
    the state vector. Returns a boolean flag.

    Common conditions:
      - ||s_t||_2 < epsilon (state converged)
      - t >= T_max (maximum timesteps reached)
      - External trigger (e.g., end-of-sequence token)
  }];

  let arguments = (ins
    FSL_StateVectorType:$state
  );
  let results = (outs
    I1:$is_complete
  );

  let assemblyFormat = [{
    $state attr-dict `:` type($is_complete)
  }];
}

#endif // FSL_OPS
