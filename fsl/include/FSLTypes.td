// ============================================================
// FSLTypes.td — Type definitions for the FSL dialect
// ============================================================
// Hybrid continuous-discrete types for Finite State Logic.

#ifndef FSL_TYPES
#define FSL_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

// ============================================================
// StateVector Type — Continuous SSM state
// ============================================================

def FSL_StateVectorType : TypeDef<"FSL", "StateVector"> {
  let mnemonic = "statevector";
  let parameters = (ins
    "int64_t":$dim     // state dimension n
  );
  let summary = "SSM state vector (continuous evolution)";
  let description = [{
    Represents the continuous state of a state-space model.

    The dimension is fixed at construction (from YAML d_state).
    State vectors evolve via the Mamba step recurrence:
      s_{t+1} = A * s_t + B * u_t

    State vectors are consumed by MambaStepOp and cannot be
    cloned or reused without explicit copy.
  }];
}

// ============================================================
// TokenVector Type — Discrete input token
// ============================================================

def FSL_TokenVectorType : TypeDef<"FSL", "TokenVector"> {
  let mnemonic = "tokenvector";
  let parameters = (ins
    "int64_t":$dim     // model dimension m
  );
  let summary = "Input token vector (discrete input)";
  let description = [{
    Represents a discrete input token to the Mamba layer.

    The dimension is fixed at construction (from YAML d_model).
    Tokens are processed through depthwise convolution and
    selectivity gating before entering the SSM.
  }];
}

// ============================================================
// SSMMatrices Type — Pre-allocated SSM parameter storage
// ============================================================

def FSL_SSMMatricesType : TypeDef<"FSL", "SSMMatrices"> {
  let mnemonic = "ssmmatrices";
  let parameters = (ins
    "int64_t":$state_dim,    // n = d_state
    "int64_t":$model_dim,    // m = d_model
    "int64_t":$conv_width,   // d_c = d_conv
    "int64_t":$rank          // r (low-rank basis)
  );
  let summary = "Pre-allocated SSM parameter storage";
  let description = [{
    Stores the learnable parameters for the selective Mamba step:
      - A_log: [n] log-space diagonal matrix
      - B: [n x m] input matrix (or low-rank factors)
      - W_conv: [m x d_c] depthwise conv kernel
      - V, U: [m x m] gating projections (for selectivity)

    This type bundles all parameters to enable efficient
    memory management and hardware-specific layout optimization.
  }];
}

// ============================================================
// FSMState Type — Discrete FSM state identifier
// ============================================================

def FSL_FSMStateType : TypeDef<"FSL", "FSMState"> {
  let mnemonic = "fsmstate";
  let parameters = (ins
    "StringAttr":$label   // e.g. "S0_IDLE", "S1_EMIT"
  );
  let summary = "Finite state machine state identifier";
  let description = [{
    Identifies a discrete state in the FSL finite state machine.

    FSM states trigger different actions (e.g., mamba_step,
    output_projection) and transitions are guarded by conditions
    on the continuous SSM state.

    Example FSM from YAML:
      S0_IDLE: scan_complete → S1_EMIT
      S1_EMIT: output_complete → S0_IDLE
  }];
}

// ============================================================
// FSMTransition Type — Discrete state transition
// ============================================================

def FSL_FSMTransitionType : TypeDef<"FSL", "FSMTransition"> {
  let mnemonic = "fsmtransition";
  let parameters = (ins
    "FSL_FSMStateType":$from_state,
    "FSL_FSMStateType":$to_state,
    "StringAttr":$condition   // e.g. "scan_complete"
  );
  let summary = "Finite state machine transition";
  let description = [{
    Defines a transition from one FSM state to another,
    guarded by a boolean condition on the SSM state.

    The condition is evaluated as a function of the SSM state:
      condition(s) = ||s||_2 > theta  (threshold-based)
      condition(s) = classifier(s)    (learned)
  }];
}

#endif // FSL_TYPES
