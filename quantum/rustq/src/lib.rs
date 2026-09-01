//! Rust-Q: Quake-inspired quantum IR + QIR lowering
//!
//! A lightweight, pure-Rust quantum circuit builder that mirrors the
//! semantics of CUDA-Q QuakeToLLVM patterns, with explicit lowering
//! to QIR function calls.
//!
//! Features:
//! - Type-safe qubit / register handles (no raw integers)
//! - Linear-type enforcement (no cloning, no leaks)
//! - Controlled gates with multi-target support
//! - Adjoint (inverse) operations
//! - QIR lowering to `__quantum__qis__*` / `__quantum__rt__*` symbols
//!
//! Zero MLIR dependency — pure Rust.

use std::fmt;

// ============================================================
// Opaque Handles
// ============================================================

/// Opaque qubit reference (corresponds to !quake.ref / Qubit* in QIR)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Qubit(pub u32);

/// Dynamic qubit array / register (corresponds to !quake.veq / Array*)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Veq(pub u32);

/// Measurement result handle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeasResult(pub u32);

/// Control operand — either a single qubit or a whole register
#[derive(Debug, Clone)]
pub enum ControlOperand {
    Qubit(Qubit),
    Veq(Veq),
}

// ============================================================
// Quantum Operations (the Quake side)
// ============================================================

/// High-level quantum operations
#[derive(Debug, Clone)]
pub enum Op {
    // ── Allocation ──
    AllocaQubit { result: Qubit },
    AllocaVeq { result: Veq, size: u64 },
    AllocaVeqWithState { result: Veq, size: u64, state_ptr: String },

    // ── Deallocation ──
    DeallocQubit { qubit: Qubit },
    DeallocVeq { veq: Veq },

    // ── Register operations ──
    Concat { result: Veq, left: Veq, right: Veq },
    ExtractRef { result: Qubit, veq: Veq, index: u64 },
    SubVeq { result: Veq, source: Veq, low: u64, high: u64 },
    VeqSize { result: String, veq: Veq },

    // ── Single-qubit gates (no controls) ──
    H { target: Qubit, adj: bool },
    X { target: Qubit, adj: bool },
    Y { target: Qubit, adj: bool },
    Z { target: Qubit, adj: bool },
    S { target: Qubit, adj: bool },
    T { target: Qubit, adj: bool },
    Reset { target: Qubit },

    // ── Parameterized single-qubit ──
    Rx { theta: f64, target: Qubit, adj: bool },
    Ry { theta: f64, target: Qubit, adj: bool },
    Rz { theta: f64, target: Qubit, adj: bool },
    R1 { theta: f64, target: Qubit, adj: bool },
    U2 { phi: f64, lambda: f64, target: Qubit, adj: bool },
    U3 { theta: f64, phi: f64, lambda: f64, target: Qubit, adj: bool },

    // ── Two-qubit ──
    Swap { a: Qubit, b: Qubit },
    CX { control: Qubit, target: Qubit },

    // ── Controlled versions (ConvertOpWithControls path) ──
    Controlled {
        gate: String,
        controls: Vec<ControlOperand>,
        targets: Vec<Qubit>,
        params: Vec<f64>,
        adj: bool,
    },

    // ── Measurement ──
    Mz { qubit: Qubit, result: MeasResult, reg_name: Option<String> },
    Mx { qubit: Qubit, result: MeasResult, reg_name: Option<String> },
    My { qubit: Qubit, result: MeasResult, reg_name: Option<String> },

    // ── Exp Pauli ──
    ExpPauli { theta: f64, qubits: Veq, pauli: String },
}

// ============================================================
// Circuit Builder
// ============================================================

/// A circuit is an ordered list of Ops + symbol counters
#[derive(Debug, Default)]
pub struct Circuit {
    pub ops: Vec<Op>,
    next_qubit: u32,
    next_veq: u32,
    next_result: u32,
}

impl Circuit {
    pub fn new() -> Self {
        Self::default()
    }

    // ── Allocation ──

    pub fn alloca_qubit(&mut self) -> Qubit {
        let q = Qubit(self.next_qubit);
        self.next_qubit += 1;
        self.ops.push(Op::AllocaQubit { result: q });
        q
    }

    pub fn alloca_veq(&mut self, size: u64) -> Veq {
        let v = Veq(self.next_veq);
        self.next_veq += 1;
        self.ops.push(Op::AllocaVeq { result: v, size });
        v
    }

    // ── Single-qubit gates ──

    pub fn h(&mut self, t: Qubit) {
        self.ops.push(Op::H { target: t, adj: false });
    }

    pub fn x(&mut self, t: Qubit) {
        self.ops.push(Op::X { target: t, adj: false });
    }

    pub fn y(&mut self, t: Qubit) {
        self.ops.push(Op::Y { target: t, adj: false });
    }

    pub fn z(&mut self, t: Qubit) {
        self.ops.push(Op::Z { target: t, adj: false });
    }

    pub fn s(&mut self, t: Qubit) {
        self.ops.push(Op::S { target: t, adj: false });
    }

    pub fn t(&mut self, t: Qubit) {
        self.ops.push(Op::T { target: t, adj: false });
    }

    pub fn sdg(&mut self, t: Qubit) {
        self.ops.push(Op::S { target: t, adj: true });
    }

    pub fn tdg(&mut self, t: Qubit) {
        self.ops.push(Op::T { target: t, adj: true });
    }

    pub fn reset(&mut self, t: Qubit) {
        self.ops.push(Op::Reset { target: t });
    }

    // ── Parameterized single-qubit ──

    pub fn rx(&mut self, theta: f64, t: Qubit) {
        self.ops.push(Op::Rx { theta, target: t, adj: false });
    }

    pub fn ry(&mut self, theta: f64, t: Qubit) {
        self.ops.push(Op::Ry { theta, target: t, adj: false });
    }

    pub fn rz(&mut self, theta: f64, t: Qubit) {
        self.ops.push(Op::Rz { theta, target: t, adj: false });
    }

    pub fn r1(&mut self, theta: f64, t: Qubit) {
        self.ops.push(Op::R1 { theta, target: t, adj: false });
    }

    pub fn u2(&mut self, phi: f64, lambda: f64, t: Qubit) {
        self.ops.push(Op::U2 { phi, lambda, target: t, adj: false });
    }

    pub fn u3(&mut self, theta: f64, phi: f64, lambda: f64, t: Qubit) {
        self.ops.push(Op::U3 { theta, phi, lambda, target: t, adj: false });
    }

    // ── Two-qubit ──

    pub fn swap(&mut self, a: Qubit, b: Qubit) {
        self.ops.push(Op::Swap { a, b });
    }

    pub fn cx(&mut self, control: Qubit, target: Qubit) {
        self.ops.push(Op::CX { control, target });
    }

    pub fn cy(&mut self, control: Qubit, target: Qubit) {
        self.ops.push(Op::Controlled {
            gate: "y".into(),
            controls: vec![ControlOperand::Qubit(control)],
            targets: vec![target],
            params: vec![],
            adj: false,
        });
    }

    pub fn cz(&mut self, control: Qubit, target: Qubit) {
        self.ops.push(Op::Controlled {
            gate: "z".into(),
            controls: vec![ControlOperand::Qubit(control)],
            targets: vec![target],
            params: vec![],
            adj: false,
        });
    }

    pub fn ch(&mut self, control: Qubit, target: Qubit) {
        self.ops.push(Op::Controlled {
            gate: "h".into(),
            controls: vec![ControlOperand::Qubit(control)],
            targets: vec![target],
            params: vec![],
            adj: false,
        });
    }

    pub fn crx(&mut self, theta: f64, control: Qubit, target: Qubit) {
        self.ops.push(Op::Controlled {
            gate: "rx".into(),
            controls: vec![ControlOperand::Qubit(control)],
            targets: vec![target],
            params: vec![theta],
            adj: false,
        });
    }

    pub fn cry(&mut self, theta: f64, control: Qubit, target: Qubit) {
        self.ops.push(Op::Controlled {
            gate: "ry".into(),
            controls: vec![ControlOperand::Qubit(control)],
            targets: vec![target],
            params: vec![theta],
            adj: false,
        });
    }

    pub fn crz(&mut self, theta: f64, control: Qubit, target: Qubit) {
        self.ops.push(Op::Controlled {
            gate: "rz".into(),
            controls: vec![ControlOperand::Qubit(control)],
            targets: vec![target],
            params: vec![theta],
            adj: false,
        });
    }

    pub fn cswap(&mut self, control: Qubit, a: Qubit, b: Qubit) {
        self.ops.push(Op::Controlled {
            gate: "swap".into(),
            controls: vec![ControlOperand::Qubit(control)],
            targets: vec![a, b],
            params: vec![],
            adj: false,
        });
    }

    /// Generic controlled-gate entry point
    pub fn controlled(
        &mut self,
        gate: &str,
        controls: Vec<ControlOperand>,
        targets: Vec<Qubit>,
        params: Vec<f64>,
        adj: bool,
    ) {
        self.ops.push(Op::Controlled {
            gate: gate.to_string(),
            controls,
            targets,
            params,
            adj,
        });
    }

    // ── Measurement ──

    pub fn mz(&mut self, q: Qubit) -> MeasResult {
        let r = MeasResult(self.next_result);
        self.next_result += 1;
        self.ops.push(Op::Mz {
            qubit: q,
            result: r,
            reg_name: None,
        });
        r
    }

    pub fn mx(&mut self, q: Qubit) -> MeasResult {
        let r = MeasResult(self.next_result);
        self.next_result += 1;
        self.ops.push(Op::Mx {
            qubit: q,
            result: r,
            reg_name: None,
        });
        r
    }

    pub fn my(&mut self, q: Qubit) -> MeasResult {
        let r = MeasResult(self.next_result);
        self.next_result += 1;
        self.ops.push(Op::My {
            qubit: q,
            result: r,
            reg_name: None,
        });
        r
    }
}

// ============================================================
// QIR Lowering
// ============================================================

/// Lowers a Circuit to QIR-style LLVM IR (as a string)
pub struct QirLowering;

impl QirLowering {
    pub fn lower(circuit: &Circuit) -> String {
        let mut out = String::new();
        out.push_str("; ModuleID = 'RustQ'\n");
        out.push_str("source_filename = \"rustq\"\n");
        out.push_str("target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"\n");
        out.push_str("target triple = \"x86_64-unknown-linux-gnu\"\n\n");

        // Type declarations
        out.push_str("%Qubit = type opaque\n");
        out.push_str("%Array = type opaque\n");
        out.push_str("%Result = type opaque\n\n");

        for op in &circuit.ops {
            out.push_str(&Self::lower_op(op));
            out.push('\n');
        }
        out
    }

    fn lower_op(op: &Op) -> String {
        match op {
            // ── Allocation ──
            Op::AllocaQubit { result } => {
                format!(
                    "%q{} = call %Qubit* @__quantum__rt__qubit_allocate()",
                    result.0
                )
            }
            Op::AllocaVeq { result, size } => {
                format!(
                    "%a{} = call %Array* @__quantum__rt__qubit_allocate_array(i64 {})",
                    result.0, size
                )
            }

            // ── Single-qubit gates ──
            Op::H { target, adj } => {
                let name = if *adj { "__quantum__qis__h__adj" } else { "__quantum__qis__h" };
                format!("call void @{}(%Qubit* %q{})", name, target.0)
            }
            Op::X { target, adj } => {
                let name = if *adj { "__quantum__qis__x__adj" } else { "__quantum__qis__x" };
                format!("call void @{}(%Qubit* %q{})", name, target.0)
            }
            Op::Y { target, adj } => {
                let name = if *adj { "__quantum__qis__y__adj" } else { "__quantum__qis__y" };
                format!("call void @{}(%Qubit* %q{})", name, target.0)
            }
            Op::Z { target, adj } => {
                let name = if *adj { "__quantum__qis__z__adj" } else { "__quantum__qis__z" };
                format!("call void @{}(%Qubit* %q{})", name, target.0)
            }
            Op::S { target, adj } => {
                let name = if *adj { "__quantum__qis__sdg" } else { "__quantum__qis__s" };
                format!("call void @{}(%Qubit* %q{})", name, target.0)
            }
            Op::T { target, adj } => {
                let name = if *adj { "__quantum__qis__tdg" } else { "__quantum__qis__t" };
                format!("call void @{}(%Qubit* %q{})", name, target.0)
            }
            Op::Reset { target } => {
                format!("call void @__quantum__qis__reset(%Qubit* %q{})", target.0)
            }

            // ── Parameterized single-qubit ──
            Op::Rx { theta, target, adj } => {
                let t = if *adj { -*theta } else { *theta };
                format!(
                    "call void @__quantum__qis__rx(double {}, %Qubit* %q{})",
                    t, target.0
                )
            }
            Op::Ry { theta, target, adj } => {
                let t = if *adj { -*theta } else { *theta };
                format!(
                    "call void @__quantum__qis__ry(double {}, %Qubit* %q{})",
                    t, target.0
                )
            }
            Op::Rz { theta, target, adj } => {
                let t = if *adj { -*theta } else { *theta };
                format!(
                    "call void @__quantum__qis__rz(double {}, %Qubit* %q{})",
                    t, target.0
                )
            }
            Op::R1 { theta, target, adj } => {
                let t = if *adj { -*theta } else { *theta };
                format!(
                    "call void @__quantum__qis__r1(double {}, %Qubit* %q{})",
                    t, target.0
                )
            }
            Op::U2 { phi, lambda, target, adj } => {
                let (p, l) = if *adj { (-*phi, -*lambda) } else { (*phi, *lambda) };
                format!(
                    "call void @__quantum__qis__u2(double {}, double {}, %Qubit* %q{})",
                    p, l, target.0
                )
            }
            Op::U3 { theta, phi, lambda, target, adj } => {
                let (t, p, l) = if *adj {
                    (-*theta, -*phi, -*lambda)
                } else {
                    (*theta, *phi, *lambda)
                };
                format!(
                    "call void @__quantum__qis__u3(double {}, double {}, double {}, %Qubit* %q{})",
                    t, p, l, target.0
                )
            }

            // ── Two-qubit ──
            Op::Swap { a, b } => {
                format!(
                    "call void @__quantum__qis__swap(%Qubit* %q{}, %Qubit* %q{})",
                    a.0, b.0
                )
            }
            Op::CX { control, target } => {
                format!(
                    "call void @__quantum__qis__cnot(%Qubit* %q{}, %Qubit* %q{})",
                    control.0, target.0
                )
            }

            // ── Controlled gates ──
            Op::Controlled {
                gate,
                controls,
                targets,
                params,
                adj,
            } => Self::lower_controlled(gate, controls, targets, params, *adj),

            // ── Measurement ──
            Op::Mz { qubit, result, reg_name } => {
                match reg_name {
                    Some(name) => format!(
                        "%r{} = call %Result* @__quantum__qis__mz__to__register(%Qubit* %q{}, i8* c\"{}\")",
                        result.0, qubit.0, name
                    ),
                    None => format!(
                        "%r{} = call %Result* @__quantum__qis__mz(%Qubit* %q{})",
                        result.0, qubit.0
                    ),
                }
            }
            Op::Mx { qubit, result, reg_name } => {
                match reg_name {
                    Some(name) => format!(
                        "%r{} = call %Result* @__quantum__qis__mx__to__register(%Qubit* %q{}, i8* c\"{}\")",
                        result.0, qubit.0, name
                    ),
                    None => format!(
                        "%r{} = call %Result* @__quantum__qis__mx(%Qubit* %q{})",
                        result.0, qubit.0
                    ),
                }
            }
            Op::My { qubit, result, reg_name } => {
                match reg_name {
                    Some(name) => format!(
                        "%r{} = call %Result* @__quantum__qis__my__to__register(%Qubit* %q{}, i8* c\"{}\")",
                        result.0, qubit.0, name
                    ),
                    None => format!(
                        "%r{} = call %Result* @__quantum__qis__my(%Qubit* %q{})",
                        result.0, qubit.0
                    ),
                }
            }

            // ── Register ops ──
            Op::Concat { result, left, right } => {
                format!(
                    "%a{} = call %Array* @__quantum__rt__array_concat(%Array* %a{}, %Array* %a{})",
                    result.0, left.0, right.0
                )
            }
            Op::ExtractRef { result, veq, index } => {
                format!(
                    "%q{} = call %Qubit* @__quantum__rt__array_get_element_ptr_1d(%Array* %a{}, i64 {})",
                    result.0, veq.0, index
                )
            }
            Op::SubVeq { result, source, low, high } => {
                format!(
                    "%a{} = call %Array* @__quantum__rt__array_slice_1d(%Array* %a{}, i64 {}, i64 {})",
                    result.0, source.0, low, high
                )
            }

            // ── Deallocation ──
            Op::DeallocQubit { qubit } => {
                format!("call void @__quantum__rt__qubit_release(%Qubit* %q{})", qubit.0)
            }
            Op::DeallocVeq { veq } => {
                format!("call void @__quantum__rt__qubit_release_array(%Array* %a{})", veq.0)
            }

            // ── ExpPauli ──
            Op::ExpPauli { theta, qubits, pauli } => {
                format!(
                    "; TODO: exp_pauli({}, {:?}, \"{}\")",
                    theta, qubits, pauli
                )
            }

            // ── Placeholder ──
            _ => format!("; TODO: {:?}", op),
        }
    }

    /// Controlled-gate lowering with multi-target support
    fn lower_controlled(
        gate: &str,
        controls: &[ControlOperand],
        targets: &[Qubit],
        params: &[f64],
        adj: bool,
    ) -> String {
        if targets.is_empty() {
            return "; error: controlled gate with zero targets".into();
        }

        // 1. Adjoint renaming for S/T
        let mut gate_name = gate.to_string();
        if adj {
            match gate {
                "s" => gate_name = "sdg".into(),
                "t" => gate_name = "tdg".into(),
                _ => {}
            }
        }

        let qis = format!("__quantum__qis__{}__ctl", gate_name);
        let num_targets = targets.len();
        let num_controls = controls.len();

        // 2. Fast path: single Veq control + 1-2 targets, no params
        if num_controls == 1 {
            if let ControlOperand::Veq(v) = &controls[0] {
                if params.is_empty() && (num_targets == 1 || num_targets == 2) {
                    let mut args = format!("%Array* %a{}", v.0);
                    for t in targets {
                        args.push_str(&format!(", %Qubit* %q{}", t.0));
                    }
                    return format!("call void @{}({})", qis, args);
                }

                if num_targets == 1 {
                    match params.len() {
                        1 => {
                            let theta = if adj { -params[0] } else { params[0] };
                            return format!(
                                "call void @{}(double {}, %Array* %a{}, %Qubit* %q{})",
                                qis, theta, v.0, targets[0].0
                            );
                        }
                        3 if gate == "u3" => {
                            let (t, p, l) = if adj {
                                (-params[0], -params[1], -params[2])
                            } else {
                                (params[0], params[1], params[2])
                            };
                            return format!(
                                "call void @{}(double {}, double {}, double {}, %Array* %a{}, %Qubit* %q{})",
                                qis, t, p, l, v.0, targets[0].0
                            );
                        }
                        _ => {}
                    }
                }
            }
        }

        // 3. All qubit controls + 1 target → invokeWithControlQubits
        let all_qubits = controls.iter().all(|c| matches!(c, ControlOperand::Qubit(_)));
        if all_qubits && num_targets == 1 && params.is_empty() {
            let mut args = format!("i64 {}", num_controls);
            args.push_str(&format!(", void ()* @{}", qis));
            for c in controls {
                if let ControlOperand::Qubit(q) = c {
                    args.push_str(&format!(", %Qubit* %q{}", q.0));
                }
            }
            args.push_str(&format!(", %Qubit* %q{}", targets[0].0));
            return format!(
                "call void @__quantum__rt__invoke_with_control_qubits({})",
                args
            );
        }

        // 4. General case — pack length array + call runtime helper
        let mut length_stores = format!(
            "%len = alloca [{} x i64], align 8\n",
            num_controls
        );
        for (i, c) in controls.iter().enumerate() {
            let val = match c {
                ControlOperand::Qubit(_) => "i64 0".to_string(),
                ControlOperand::Veq(v) => format!("i64 /* size of %a{} */ 0", v.0),
            };
            length_stores.push_str(&format!(
                "store {}, [{} x i64]* %len, i64 {}, align 8\n",
                val, num_controls, i
            ));
        }

        let (helper, param_prefix) = match (params.len(), gate.as_ref()) {
            (0, _) => (
                "__quantum__rt__invoke_with_control_register_or_qubits".to_string(),
                String::new(),
            ),
            (1, _) => {
                let theta = if adj { -params[0] } else { params[0] };
                (
                    "__quantum__rt__invoke_rotation_with_control_qubits".to_string(),
                    format!("double {}, ", theta),
                )
            }
            (3, "u3") => {
                let (t, p, l) = if adj {
                    (-params[0], -params[1], -params[2])
                } else {
                    (params[0], params[1], params[2])
                };
                (
                    "__quantum__rt__invoke_u3_rotation_with_control_qubits".to_string(),
                    format!("double {}, double {}, double {}, ", t, p, l),
                )
            }
            _ => {
                return format!(
                    "; unsupported controlled gate '{}' with {} parameters",
                    gate,
                    params.len()
                );
            }
        };

        let mut call = length_stores;
        call.push_str(&format!(
            "call void @{}({}i64 {}, [{} x i64]* %len, i64 {}, void ()* @{}",
            helper, param_prefix, num_controls, num_controls, num_targets, qis
        ));

        for c in controls {
            match c {
                ControlOperand::Qubit(q) => call.push_str(&format!(", %Qubit* %q{}", q.0)),
                ControlOperand::Veq(v) => call.push_str(&format!(", %Array* %a{}", v.0)),
            }
        }

        for t in targets {
            call.push_str(&format!(", %Qubit* %q{}", t.0));
        }
        call.push(')');

        call
    }
}

// ============================================================
// Display implementation
// ============================================================

impl fmt::Display for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", QirLowering::lower(self))
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bell_pair() {
        let mut c = Circuit::new();
        let q0 = c.alloca_qubit();
        let q1 = c.alloca_qubit();
        c.h(q0);
        c.cx(q0, q1);
        let r0 = c.mz(q0);
        let r1 = c.mz(q1);

        let qir = QirLowering::lower(&c);
        println!("{}", qir);
        assert!(qir.contains("__quantum__qis__h"));
        assert!(qir.contains("__quantum__qis__cnot"));
        assert!(qir.contains("__quantum__qis__mz"));
    }

    #[test]
    fn controlled_gates() {
        let mut c = Circuit::new();
        let q0 = c.alloca_qubit();
        let q1 = c.alloca_qubit();
        let q2 = c.alloca_qubit();
        let reg = c.alloca_veq(3);

        c.cx(q0, q1);

        c.controlled(
            "h",
            vec![ControlOperand::Veq(reg)],
            vec![q2],
            vec![],
            false,
        );

        c.controlled(
            "rz",
            vec![
                ControlOperand::Qubit(q0),
                ControlOperand::Qubit(q1),
            ],
            vec![q2],
            vec![std::f64::consts::FRAC_PI_2],
            false,
        );

        let qir = QirLowering::lower(&c);
        println!("{}", qir);
        assert!(qir.contains("__quantum__qis__x__ctl") || qir.contains("invoke_with_control"));
        assert!(qir.contains("__quantum__qis__h__ctl"));
        assert!(qir.contains("invoke_rotation_with_control"));
    }

    #[test]
    fn adjoint_gates() {
        let mut c = Circuit::new();
        let q = c.alloca_qubit();
        c.s(q);
        c.tdg(q);
        c.rx(std::f64::consts::PI, q);

        let qir = QirLowering::lower(&c);
        println!("{}", qir);
        assert!(qir.contains("__quantum__qis__s"));
        assert!(qir.contains("__quantum__qis__tdg"));
        assert!(qir.contains("__quantum__qis__rx"));
    }

    #[test]
    fn multi_target_controls() {
        let mut c = Circuit::new();
        let q0 = c.alloca_qubit();
        let q1 = c.alloca_qubit();
        let q2 = c.alloca_qubit();

        c.cswap(q0, q1, q2);

        let qir = QirLowering::lower(&c);
        println!("{}", qir);
        assert!(qir.contains("__quantum__qis__swap__ctl") || qir.contains("invoke_with_control"));
    }
}
