// ============================================================
// QuantumOps.td — Operation definitions for the #q quantum dialect
// ============================================================
// Covers: alloc, unitary, entangle, measure, reset, concat, extract.
// Linear-type discipline enforced via traits + verifier.

#ifndef QUANTUM_OPS
#define QUANTUM_OPS

include "QuantumDialect.td"
include "QuantumTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// ============================================================
// Traits
// ============================================================

// Enforce no-cloning: every !quantum.qubit SSA value must have
// exactly one use (consumed by unitary, entangle, or measure).
def Quantum_NoCloning : NativeOpTrait<"NoCloning"> {
  let cppNamespace = "::mlir::quantum";
}

// ============================================================
// Allocation Operations
// ============================================================

def Quantum_AllocOp : Quantum_Op<"alloc", [
    MemoryEffects<[MemAlloc]>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    Quantum_NoCloning
  ]> {
  let summary = "Allocate a clean qubit or qureg";
  let description = [{
    Allocates a fresh qubit or register in the |0⟩ state.

    The result type determines the allocation:
      !quantum.qubit   → single qubit
      !quantum.qureg<N> → register of N qubits
      !quantum.qureg<?> → dynamic-size register

    The allocated resource must be consumed by a unitary, entangle,
    or measure operation before the function returns.  The verifier
    rejects dangling allocations (no-cloning trait).
  }];

  let arguments = (ins Optional<I64>:$size);
  let results = (outs AnyTypeOf<[Quantum_QubitType, Quantum_QuregType]>:$result);
  let assemblyFormat = "($size^)? attr-dict `:` type($result)";

  let hasVerifier = 1;
}

def Quantum_AllocWithStateOp : Quantum_Op<"alloc_with_state", [
    MemoryEffects<[MemAlloc]>,
    Quantum_NoCloning
  ]> {
  let summary = "Allocate qubits with a specific initial state";
  let description = [{
    Allocates a qubit or register initialized to a user-specified
    state vector.  The state must be normalized.

    This mirrors the CUDA-Q RAII allocation with initialisation:
      qubit q = cudaq::qrt::qubit_alloca(initialState);

    The verifier checks that the state length matches the allocation
    size (2^N for N qubits).
  }];

  let arguments = (ins
    AnyTypeOf<[Quantum_QubitType, Quantum_QuregType]>:$qubits,
    Attribute:$state           // DenseComplexFPElementsAttr
  );
  let results = (outs AnyTypeOf<[Quantum_QubitType, Quantum_QuregType]>:$result);
  let assemblyFormat = [{
    $qubits `with` $state attr-dict `:` type($result)
  }];
}

// ============================================================
// Unitary Operations
// ============================================================

def Quantum_UnitaryOp : Quantum_Op<"unitary", [
    NoMemoryEffect,
    Quantum_NoCloning
  ]> {
  let summary = "Parametrised multi-axis rotation (exact algebraic angles)";
  let description = [{
    Applies a parametrised unitary to one or more qubits.

    The angles are stored as exact algebraic values (rational or
    symbolic), not floating-point approximations.  This enables:
      - Exact Clifford+T synthesis
      - Symbolic gradient computation for variational algorithms
      - Noise-aware compilation with precision guarantees

    The axis parameter selects the rotation axis:
      "X" → R_x(θ) = exp(-iθ/2 · σ_x)
      "Y" → R_y(θ) = exp(-iθ/2 · σ_y)
      "Z" → R_z(θ) = exp(-iθ/2 · σ_z)
      "arbitrary" → arbitrary single-qubit unitary

    Examples:
      quantum.unitary %q [0.5] axis "Y"    // H gate (θ=π/2)
      quantum.unitary %q [0.25]             // T gate (θ=π/4)
      quantum.unitary %q [0.125, 0.5, 0.0] // U3 gate
  }];

  let arguments = (ins
    Variadic<Quantum_QubitType>:$qubits,
    ArrayAttr:$angles,                // e.g. [89/2462, ...]
    OptionalAttr<StrAttr>:$axis       // "X","Y","Z","arbitrary"
  );
  let results = (outs Variadic<Quantum_QubitType>:$results);  // linear consumption
  let assemblyFormat = [{
    $qubits `(` $angles `)` (`axis` $axis^)?
    attr-dict `:` functional-type($qubits, $results)
  }];

  let hasVerifier = 1;  // enforce angle domain, no-cloning
}

// ============================================================
// Entangle Operations (controlled gates)
// ============================================================

def Quantum_EntangleOp : Quantum_Op<"entangle", [
    NoMemoryEffect,
    Quantum_NoCloning
  ]> {
  let summary = "Multi-qubit controlled operation (CNOT, Toffoli, CPhase, ...)";
  let description = [{
    Controlled operation acting on control and target qubits.

    This is the universal controlled gate.  The base gate is
    determined by the number of targets and optional phases:
      1 target, no phases  → CNOT (X) or controlled-U
      1 target, phase π    → CZ (Z)
      2 targets            → Toffoli (CCX) or Fredkin (CSWAP)

    The adjoint flag negates all angles for parameterised gates
    and reverses the gate sequence for non-parameterised gates.

    Linear-type discipline: all input qubits are consumed and
    replaced by output qubits in the same positions.
  }];

  let arguments = (ins
    Variadic<Quantum_QubitType>:$controls,
    Variadic<Quantum_QubitType>:$targets,
    OptionalAttr<ArrayAttr>:$phases,  // for controlled-phase
    UnitAttr:$is_adj
  );
  let results = (outs
    Variadic<Quantum_QubitType>:$out_controls,
    Variadic<Quantum_QubitType>:$out_targets
  );
  let assemblyFormat = [{
    (`adj` $is_adj^)? `[` $controls `]` $targets
    (`phases` $phases^)?
    attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

// ============================================================
// Measurement Operations
// ============================================================

def Quantum_MeasureOp : Quantum_Op<"measure", [
    MemoryEffects<[MemRead, MemWrite]>,
    Quantum_NoCloning
  ]> {
  let summary = "Collapse amplitude vector into classical bits";
  let description = [{
    Measures the specified qubits in the computational (Z) basis.

    Returns:
      - A classical bit (i1) for each measured qubit
      - The post-measurement qubit state (consumed, cannot be reused
        without a fresh allocation)

    The optional registerName attaches metadata for classical
    control flow (e.g. "c" for the full register, "q0" for a
    single qubit).

    This mirrors the CUDA-Q QuakeToLLVM measurement pattern:
      %r = call %Result* @__quantum__qis__mz(%Qubit* %q)
      %bit = trunc %r to i1
  }];

  let arguments = (ins
    Variadic<Quantum_QubitType>:$qubits,
    OptionalAttr<StrAttr>:$registerName
  );
  let results = (outs
    Variadic<I1>:$bits,                    // classical results
    Variadic<Quantum_QubitType>:$collapsed  // post-measurement state
  );
  let assemblyFormat = [{
    $qubits (`->` $registerName^)?
    attr-dict `:` functional-type($qubits, results)
  }];

  let hasVerifier = 1;
}

// ============================================================
// Register Operations
// ============================================================

def Quantum_ConcatOp : Quantum_Op<"concat", [
    NoMemoryEffect,
    Quantum_NoCloning
  ]> {
  let summary = "Concatenate two quregs into one";
  let arguments = (ins
    Quantum_QuregType:$left,
    Quantum_QuregType:$right
  );
  let results = (outs Quantum_QuregType:$result);
  let assemblyFormat = [{
    $left `,` $right attr-dict `:` type($result)
  }];
}

def Quantum_ExtractRefOp : Quantum_Op<"extract_ref", [
    NoMemoryEffect,
    Quantum_NoCloning
  ]> {
  let summary = "Extract a single qubit from a qureg by index";
  let arguments = (ins
    Quantum_QuregType:$source,
    IntegerAttr<I64>:$index
  );
  let results = (outs Quantum_QubitType:$result);
  let assemblyFormat = [{
    $source `[` $index `]` attr-dict `:` type($result)
  }];

  let hasVerifier = 1;  // bounds check
}

def Quantum_SubveqOp : Quantum_Op<"subveq", [
    NoMemoryEffect,
    Quantum_NoCloning
  ]> {
  let summary = "Extract a contiguous sub-register";
  let arguments = (ins
    Quantum_QuregType:$source,
    IntegerAttr<I64>:$low,
    IntegerAttr<I64>:$high
  );
  let results = (outs Quantum_QuregType:$result);
  let assemblyFormat = [{
    $source `[` $low `:` $high `]` attr-dict `:` type($result)
  }];

  let hasVerifier = 1;  // bounds check, low < high
}

def Quantum_VeqSizeOp : Quantum_Op<"veq_size", [
    Pure
  ]> {
  let summary = "Return the size of a qureg";
  let arguments = (ins Quantum_QuregType:$source);
  let results = (outs I64:$size);
  let assemblyFormat = [{
    $source attr-dict `:` type($size)
  }];
}

// ============================================================
// Reset Operation
// ============================================================

def Quantum_ResetOp : Quantum_Op<"reset", [
    MemoryEffects<[MemWrite]>,
    Quantum_NoCloning
  ]> {
  let summary = "Reset qubit to |0⟩ without measurement";
  let arguments = (ins Quantum_QubitType:$target);
  let results = (outs Quantum_QubitType:$result);
  let assemblyFormat = [{
    $target attr-dict `:` type($result)
  }];
}

// ============================================================
// Exp Pauli (exponentiation of Pauli string)
// ============================================================

def Quantum_ExpPauliOp : Quantum_Op<"exp_pauli", [
    NoMemoryEffect,
    Quantum_NoCloning
  ]> {
  let summary = "Exponentiation of a Pauli string: exp(-iθ/2 · P)";
  let description = [{
    Applies exp(-iθ/2 · P) where P is a tensor product of Pauli
    operators (X, Y, Z, I) on the specified qubits.

    This is the native gate for:
      - QAOA cost Hamiltonian evolution
      - Variational quantum eigensolver (VQE) ansatz
      - Suzuki-Trotter decomposition of molecular Hamiltonians

    The pauli string is encoded as a dense integer array:
      0 = I, 1 = X, 2 = Y, 3 = Z

    Example:
      // e^{-iθ/2 · X⊗Z} on q0, q1
      quantum.exp_pauli %q0, %q1 [1, 3] for θ = 0.5
  }];

  let arguments = (ins
    Variadic<Quantum_QubitType>:$qubits,
    DenseI32ArrayAttr:$pauli,   // Pauli string encoding
    AnyAttr:$theta              // angle (rational or float)
  );
  let results = (outs Variadic<Quantum_QubitType>:$results);
  let assemblyFormat = [{
    $qubits `(` $pauli `)` `for` $theta
    attr-dict `:` functional-type($qubits, $results)
  }];

  let hasVerifier = 1;  // pauli length == qubit count
}

#endif // QUANTUM_OPS
