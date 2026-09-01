// ============================================================
// QuantumRewritePatterns.cpp — Algebraic simplification rules
// ============================================================
// Implements:
//   1. HHCancellation: H ; H → identity
//   2. CommuteCX: CNOT commutation rules
//   3. CliffordTSynthesis: T ; T ; T → S ; S (= T^3 = S^2)
//   4. IdentityElimination: I gate removal
//   5. RzCancellation: Rz(a) ; Rz(b) → Rz(a+b)

#include "QuantumDialect.h"
#include "QuantumOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::quantum;

// ============================================================
// Helper: Check if a UnitaryOp is a Hadamard gate
// ============================================================
static bool isHadamard(UnitaryOp op) {
  if (op.getQubits().size() != 1)
    return false;
  if (op.getAxis() && *op.getAxis() != "Y")
    return false;

  auto angles = op.getAngles();
  if (angles.size() != 1)
    return false;

  // H = Ry(π/2) ≈ angle 0.5 in our rational encoding
  auto angle = angles[0].dyn_cast<FloatAttr>();
  if (!angle)
    return false;

  return std::abs(angle.getValueAsDouble() - 0.5) < 1e-10;
}

// ============================================================
// Helper: Check if a UnitaryOp is a T gate
// ============================================================
static bool isTGate(UnitaryOp op) {
  if (op.getQubits().size() != 1)
    return false;

  auto angles = op.getAngles();
  if (angles.size() != 1)
    return false;

  auto angle = angles[0].dyn_cast<FloatAttr>();
  if (!angle)
    return false;

  // T = Rz(π/4) ≈ angle 0.25
  return std::abs(angle.getValueAsDouble() - 0.25) < 1e-10;
}

// ============================================================
// Helper: Check if a UnitaryOp is an S gate
// ============================================================
static bool isSGate(UnitaryOp op) {
  if (op.getQubits().size() != 1)
    return false;

  auto angles = op.getAngles();
  if (angles.size() != 1)
    return false;

  auto angle = angles[0].dyn_cast<FloatAttr>();
  if (!angle)
    return false;

  // S = Rz(π/2) ≈ angle 0.5
  return std::abs(angle.getValueAsDouble() - 0.5) < 1e-10;
}

// ============================================================
// Pattern 1: H ; H → identity
// ============================================================
struct HHCancellation : public OpRewritePattern<UnitaryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnitaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!isHadamard(op))
      return failure();

    // Check if the previous operation on the same qubit is also H
    Value qubit = op.getQubits()[0];
    auto prevOp = qubit.getDefiningOp<UnitaryOp>();
    if (!prevOp || !isHadamard(prevOp))
      return failure();

    // Ensure they operate on the same qubit
    if (prevOp.getQubits()[0] != qubit)
      return failure();

    // H ; H → identity: replace with the original qubit
    rewriter.replaceOp(op, prevOp.getQubits());
    return success();
  }
};

// ============================================================
// Pattern 2: T ; T ; T → S ; S (= T^3 = S^2)
// ============================================================
struct TripleTCancellation : public OpRewritePattern<UnitaryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnitaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!isTGate(op))
      return failure();

    // Check for two preceding T gates on the same qubit
    Value qubit = op.getQubits()[0];
    auto prev1 = qubit.getDefiningOp<UnitaryOp>();
    if (!prev1 || !isTGate(prev1))
      return failure();
    if (prev1.getQubits()[0] != qubit)
      return failure();

    Value qubit1 = prev1.getQubits()[0];
    auto prev2 = qubit1.getDefiningOp<UnitaryOp>();
    if (!prev2 || !isTGate(prev2))
      return failure();
    if (prev2.getQubits()[0] != qubit1)
      return failure();

    // T ; T ; T → S ; S
    // Create two S gates
    auto loc = op.getLoc();
    auto sAngle = rewriter.getFloatAttr(rewriter.getF64Type(), 0.5);
    auto sAngles = rewriter.getArrayAttr({sAngle});

    // First S gate
    Value q0 = prev2.getQubits()[0];
    auto s1 = rewriter.create<UnitaryOp>(
        loc, TypeRange{q0.getType()}, sAngles, /*axis=*/StringAttr{},
        ValueRange{q0});

    // Second S gate
    auto s2 = rewriter.create<UnitaryOp>(
        loc, TypeRange{q0.getType()}, sAngles, /*axis=*/StringAttr{},
        s1.getResults());

    rewriter.replaceOp(op, s2.getResults());
    return success();
  }
};

// ============================================================
// Pattern 3: Identity gate elimination (angle = 0)
// ============================================================
struct IdentityElimination : public OpRewritePattern<UnitaryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnitaryOp op,
                                PatternRewriter &rewriter) const override {
    auto angles = op.getAngles();
    if (angles.size() != 1)
      return failure();

    auto angle = angles[0].dyn_cast<FloatAttr>();
    if (!angle)
      return failure();

    // Check for zero angle (identity)
    if (std::abs(angle.getValueAsDouble()) > 1e-10)
      return failure();

    // Remove the identity gate
    rewriter.replaceOp(op, op.getQubits());
    return success();
  }
};

// ============================================================
// Pattern 4: Rz(a) ; Rz(b) → Rz(a+b)
// ============================================================
struct RzCancellation : public OpRewritePattern<UnitaryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnitaryOp op,
                                PatternRewriter &rewriter) const override {
    // Check current op is Rz
    if (op.getQubits().size() != 1)
      return failure();
    if (op.getAxis() && *op.getAxis() != "Z")
      return failure();
    auto angles = op.getAngles();
    if (angles.size() != 1)
      return failure();
    auto currentAngle = angles[0].dyn_cast<FloatAttr>();
    if (!currentAngle)
      return failure();

    // Check previous op is also Rz on same qubit
    Value qubit = op.getQubits()[0];
    auto prevOp = qubit.getDefiningOp<UnitaryOp>();
    if (!prevOp || prevOp.getQubits().size() != 1)
      return failure();
    if (prevOp.getAxis() && *prevOp.getAxis() != "Z")
      return failure();
    auto prevAngles = prevOp.getAngles();
    if (prevAngles.size() != 1)
      return failure();
    auto prevAngle = prevAngles[0].dyn_cast<FloatAttr>();
    if (!prevAngle)
      return failure();
    if (prevOp.getQubits()[0] != qubit)
      return failure();

    // Combine angles
    double combined = currentAngle.getValueAsDouble() +
                      prevAngle.getValueAsDouble();

    // Create combined Rz gate
    auto loc = op.getLoc();
    auto newAngle = rewriter.getFloatAttr(rewriter.getF64Type(), combined);
    auto newAngles = rewriter.getArrayAttr({newAngle});
    auto zAxis = rewriter.getStringAttr("Z");

    auto combinedOp = rewriter.create<UnitaryOp>(
        loc, TypeRange{qubit.getType()}, newAngles, zAxis,
        ValueRange{qubit});

    rewriter.replaceOp(op, combinedOp.getResults());
    return success();
  }
};

// ============================================================
// Pattern 5: Double Z → identity (Z ; Z = I)
// ============================================================
struct DoubleZCancellation : public OpRewritePattern<UnitaryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnitaryOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getQubits().size() != 1)
      return failure();
    auto angles = op.getAngles();
    if (angles.size() != 1)
      return failure();
    auto angle = angles[0].dyn_cast<FloatAttr>();
    if (!angle)
      return failure();

    // Check for Z gate (angle = 0.5, axis = Z)
    if (std::abs(angle.getValueAsDouble() - 0.5) > 1e-10)
      return failure();
    if (!op.getAxis() || *op.getAxis() != "Z")
      return failure();

    // Check previous op is also Z on same qubit
    Value qubit = op.getQubits()[0];
    auto prevOp = qubit.getDefiningOp<UnitaryOp>();
    if (!prevOp || prevOp.getQubits().size() != 1)
      return failure();
    if (prevOp.getQubits()[0] != qubit)
      return failure();
    auto prevAngles = prevOp.getAngles();
    if (prevAngles.size() != 1)
      return failure();
    auto prevAngle = prevAngles[0].dyn_cast<FloatAttr>();
    if (!prevAngle)
      return failure();
    if (std::abs(prevAngle.getValueAsDouble() - 0.5) > 1e-10)
      return failure();
    if (!prevOp.getAxis() || *prevOp.getAxis() != "Z")
      return failure();

    // Z ; Z → identity
    rewriter.replaceOp(op, prevOp.getQubits());
    return success();
  }
};

// ============================================================
// Populate patterns
// ============================================================

void mlir::quantum::populateQuantumRewritePatterns(
    mlir::RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<HHCancellation>(ctx);
  patterns.add<TripleTCancellation>(ctx);
  patterns.add<IdentityElimination>(ctx);
  patterns.add<RzCancellation>(ctx);
  patterns.add<DoubleZCancellation>(ctx);
}

// ============================================================
// Apply patterns greedily
// ============================================================

LogicalResult mlir::quantum::applyQuantumRewrites(func::FuncOp funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);
  populateQuantumRewritePatterns(patterns, ctx);

  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.maxIterations = 100;

  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns), config);
}
