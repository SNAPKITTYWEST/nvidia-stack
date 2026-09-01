// ============================================================
// QuantumTypes.td — Type definitions for the #q quantum dialect
// ============================================================
// Linear-type quantum resources with no-cloning enforcement.
// Designed as a strict refinement of CUDA-Q Quake types.

#ifndef QUANTUM_TYPES
#define QUANTUM_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

// ============================================================
// Qubit Type — Linear resource (no copy, no discard)
// ============================================================

def Quantum_QubitType : TypeDef<"Quantum", "Qubit", [
    DeclareTypeInterfaceMethods<MemRefResourceTypeInterface>
  ]> {
  let mnemonic = "qubit";
  let summary = "A linear qubit resource (no-cloning enforced)";
  let description = [{
    Represents a single qubit under linear type discipline.

    The verifier rejects any use that would:
    - Duplicate an SSA value holding a qubit (use-def chain has >1 user)
    - Drop a qubit without measurement or explicit deallocation
    - Reuse a qubit after measurement without a fresh allocation

    This is stricter than CUDA-Q Quake, which uses memory semantics
    (!quake.ref) without enforcement at the type level.
  }];

  // Assembly format: !quantum.qubit
  let assemblyFormat = "`qubit`";
}

// ============================================================
// Qureg Type — Fixed or dynamically-sized register
// ============================================================

def Quantum_QuregType : TypeDef<"Quantum", "Qureg"> {
  let mnemonic = "qureg";
  let parameters = (ins
    "std::optional<int64_t>":$size  // none = dynamic
  );
  let assemblyFormat = "`<` (`?`:$size^):($size) `>`";
  let summary = "A register of qubits (contiguous allocation)";
  let description = [{
    Represents a contiguous register of qubits.

    If the size is known at compile time, the verifier can check
    that indexing operations stay within bounds.  A dynamic-size
    register (!quantum.qureg<?>) defers the check to runtime.

    Qureg values are consumed by entangle/measure ops; slicing
    produces sub-regions or individual qubits via extract/ref.
  }];
}

// ============================================================
// PauliOperator Type — Exact algebraic angles
// ============================================================

def Quantum_PauliOperatorType : TypeDef<"Quantum", "PauliOperator"> {
  let mnemonic = "pauli";
  let parameters = (ins
    "StringAttr":$label,   // "X","Y","Z","R",...
    "Attribute":$angle     // rational or symbolic θ
  );
  let assemblyFormat = "`<` $label (`,` $angle^)? `>`";
  let summary = "Non-commutative Pauli / phase operator";
  let description = [{
    Represents a Pauli operator with an exact algebraic angle.

    The label selects the axis:
      "X" → σ_x  (bit flip)
      "Y" → σ_y  (bit + phase flip)
      "Z" → σ_z  (phase flip)
      "R" → R(θ) = exp(-iθ/2 · σ_z)  (rotation)

    The angle is stored as a rational or symbolic attribute,
    not a floating-point approximation.  This enables exact
    algebraic simplification (e.g. R(π) = Z, R(2π) = I).

    Use cases:
      - Exact compilation of Clifford+T circuits
      - Symbolic parameter optimization (variational algorithms)
      - Noise-aware compilation where angle precision matters
  }];
}

// ============================================================
// MeasurementResult Type — Classical bit
// ============================================================

def Quantum_MeasurementResult : TypeDef<"Quantum", "MeasurementResult"> {
  let mnemonic = "mresult";
  let summary = "Classical measurement result (i1 with metadata)";
  let description = [{
    Wraps a single classical bit (i1) with optional metadata
    (register name, measurement basis, timestamp).

    Distinguished from plain i1 to prevent accidental mixing
    of classical control flow bits with quantum measurement outcomes.
  }];
}

#endif // QUANTUM_TYPES
