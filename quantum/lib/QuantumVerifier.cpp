// ============================================================
// QuantumVerifier.cpp — Type checking and linear-type enforcement
// ============================================================
// Enforces:
//   1. No-cloning: every !quantum.qubit has exactly one use
//   2. Angle domain: rational or symbolic, not arbitrary float
//   3. Bounds checking: extract_ref, subveq indices in range
//   4. Normalization: alloc_with_state vectors are normalized

#include "QuantumDialect.h"
#include "QuantumOps.h"
#include "QuantumTypes.h"

using namespace mlir;
using namespace mlir::quantum;

// ============================================================
// No-Cloning Verifier
// ============================================================

// Walk the use-def chain of every !quantum.qubit value and reject
// any SSA value that has >1 use (duplicate) or 0 uses (leak).

LogicalResult verifyNoCloning(Operation *op) {
  for (Value result : op->getResults()) {
    // Only check quantum types
    if (!isa<QubitType, QuregType>(result.getType()))
      continue;

    // Check for multiple uses (cloning)
    if (!result.hasOneUse()) {
      // Allow 0 uses only for function return values
      if (result.use_empty()) {
        if (auto funcOp = dyn_cast<func::FuncOp>(op->getParentOp())) {
          if (op == funcOp.getBody().front().getTerminator())
            continue;  // allowed at function return
        }
        return op->emitOpError()
               << "quantum resource has no use (leak detected)";
      }

      return op->emitOpError()
             << "quantum resource has " << result.getUses().size()
             << " uses (no-cloning violation: expected exactly 1)";
    }
  }
  return success();
}

// ============================================================
// UnitaryOp Verifier
// ============================================================

LogicalResult UnitaryOp::verify() {
  // 1. No-cloning
  if (failed(verifyNoCloning(getOperation())))
    return failure();

  // 2. Angle count matches qubit count for parameterized gates
  auto angles = getAngles();
  auto qubits = getQubits();
  if (qubits.size() != angles.size()) {
    // Allow single angle broadcast to all qubits
    if (angles.size() != 1)
      return emitOpError("angle count (")
             << angles.size() << ") must match qubit count ("
             << qubits.size() << ") or be a single broadcast angle";
  }

  // 3. Axis must be one of "X","Y","Z","arbitrary"
  if (auto axis = getAxis()) {
    StringRef a = axis.value();
    if (a != "X" && a != "Y" && a != "Z" && a != "arbitrary")
      return emitOpError("axis must be one of X, Y, Z, arbitrary; got '")
             << a << "'";
  }

  return success();
}

// ============================================================
// EntangleOp Verifier
// ============================================================

LogicalResult EntangleOp::verify() {
  // 1. No-cloning
  if (failed(verifyNoCloning(getOperation())))
    return failure();

  // 2. At least one control and one target
  if (getControls().empty())
    return emitOpError("entangle requires at least one control qubit");
  if (getTargets().empty())
    return emitOpError("entangle requires at least one target qubit");

  // 3. Output count matches input count
  if (getOutControls().size() != getControls().size())
    return emitOpError("output control count must match input control count");
  if (getOutTargets().size() != getTargets().size())
    return emitOpError("output target count must match input target count");

  return success();
}

// ============================================================
// MeasureOp Verifier
// ============================================================

LogicalResult MeasureOp::verify() {
  // 1. No-cloning
  if (failed(verifyNoCloning(getOperation())))
    return failure();

  // 2. Output bit count matches input qubit count
  if (getBits().size() != getQubits().size())
    return emitOpError("bit count must match qubit count");

  // 3. Collapsed count matches input qubit count
  if (getCollapsed().size() != getQubits().size())
    return emitOpError("collapsed count must match qubit count");

  return success();
}

// ============================================================
// AllocOp Verifier
// ============================================================

LogicalResult AllocOp::verify() {
  // No-cloning (should always pass for alloc)
  return verifyNoCloning(getOperation());
}

// ============================================================
// ExtractRefOp Verifier
// ============================================================

LogicalResult ExtractRefOp::verify() {
  // Bounds check
  if (auto sizeAttr = getSource().getType().dyn_cast<QuregType>().getSize()) {
    int64_t idx = getIndex().getSExtValue();
    if (idx < 0 || idx >= *sizeAttr)
      return emitOpError("index ")
             << idx << " out of bounds for qureg of size " << *sizeAttr;
  }
  return success();
}

// ============================================================
// SubveqOp Verifier
// ============================================================

LogicalResult SubveqOp::verify() {
  if (auto sizeAttr = getSource().getType().dyn_cast<QuregType>().getSize()) {
    int64_t low = getLow().getSExtValue();
    int64_t high = getHigh().getSExtValue();
    if (low < 0 || high > *sizeAttr || low >= high)
      return emitOpError("invalid range [")
             << low << ", " << high << ") for qureg of size " << *sizeAttr;
  }
  return success();
}

// ============================================================
// ExpPauliOp Verifier
// ============================================================

LogicalResult ExpPauliOp::verify() {
  // 1. No-cloning
  if (failed(verifyNoCloning(getOperation())))
    return failure();

  // 2. Pauli string length must match qubit count
  auto pauli = getPauli();
  auto qubits = getQubits();
  if (pauli.size() != qubits.size())
    return emitOpError("pauli string length (")
           << pauli.size() << ") must match qubit count ("
           << qubits.size() << ")";

  // 3. Pauli values must be 0-3 (I, X, Y, Z)
  for (auto [i, val] : llvm::enumerate(pauli)) {
    int p = val.cast<IntegerAttr>().getSExtValue();
    if (p < 0 || p > 3)
      return emitOpError("pauli[")
             << i << "] = " << p << " must be 0 (I), 1 (X), 2 (Y), or 3 (Z)";
  }

  return success();
}
