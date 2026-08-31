from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from fragment_map import FragmentElement


@dataclass(frozen=True)
class MfmaShape:
    target: str = "gfx942"
    opcode: str = "v_mfma_f32_16x16x16f16"
    m: int = 16
    n: int = 16
    k: int = 16
    wave_size: int = 64

    @property
    def expected_elements(self) -> Dict[str, int]:
        return {
            "A": self.m * self.k,
            "B": self.k * self.n,
            "C": self.m * self.n,
            "D": self.m * self.n,
        }

    @property
    def operand_bounds(self) -> Dict[str, Tuple[int, int]]:
        return {
            "A": (self.m, self.k),
            "B": (self.k, self.n),
            "C": (self.m, self.n),
            "D": (self.m, self.n),
        }


@dataclass(frozen=True)
class ValidationIssue:
    severity: str # "error" or "warning"
    code: str
    message: str
    operand: Optional[str] = None
    lane: Optional[int] = None
    vgpr: Optional[int] = None
    coordinate: Optional[Tuple[int, int]] = None


class FragmentMapValidationError(ValueError):
    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        errors = [x for x in issues if x.severity == "error"]

        lines = [
            f"MFMA fragment-map validation failed with {len(errors)} error(s)"
        ]
        for issue in errors[:32]:
            where = []
            if issue.operand is not None:
                where.append(f"operand={issue.operand}")
            if issue.lane is not None:
                where.append(f"lane={issue.lane}")
            if issue.vgpr is not None:
                where.append(f"vgpr={issue.vgpr}")
            if issue.coordinate is not None:
                where.append(f"coord={issue.coordinate}")

            suffix = f" ({', '.join(where)})" if where else ""
            lines.append(f"[{issue.code}] {issue.message}{suffix}")

        if len(errors) > 32:
            lines.append(f"... {len(errors) - 32} additional error(s) omitted")

        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class FragmentMapReport:
    shape: MfmaShape
    issues: Tuple[ValidationIssue, ...]
    element_counts: Mapping[str, int]
    unique_coordinate_counts: Mapping[str, int]
    per_lane_element_counts: Mapping[str, Mapping[int, int]]
    per_lane_vgpr_counts: Mapping[str, Mapping[int, int]]

    @property
    def errors(self) -> Tuple[ValidationIssue, ...]:
        return tuple(x for x in self.issues if x.severity == "error")

    @property
    def warnings(self) -> Tuple[ValidationIssue, ...]:
        return tuple(x for x in self.issues if x.severity == "warning")

    @property
    def valid(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise FragmentMapValidationError(self.issues)


def validate_fragment_map(
    fragments: Mapping[str, Sequence[FragmentElement]],
    *,
    shape: MfmaShape = MfmaShape(),
    strict_register_layout: bool = True,
    require_all_lanes_for_ab: bool = True,
    require_all_lanes_for_cd: bool = True,
    require_c_d_same_layout: bool = True,
) -> FragmentMapReport:
    """
    Validate an imported gfx942 v_mfma_f32_16x16x16f16 fragment map.

    The validator establishes structural facts:

      * A has exactly M*K unique coordinates in [0,M) x [0,K).
      * B has exactly K*N unique coordinates in [0,K) x [0,N).
      * C and D each have exactly M*N unique coordinates in [0,M) x [0,N).
      * All elements identify the correct operand and a lane in [0,wave_size).
      * A/B are packed FP16: each logical element has packed_half in {0,1}.
      * C/D are FP32: packed_half is None.
      * A/B each use exactly 4 FP16 elements per lane for a 16x16x16 tile.
      * C/D each use exactly 4 FP32 elements per lane for a 16x16 output tile.
      * Each lane's A/B halves form valid packed dwords:
            (lane, vgpr) -> exactly one low and one high half.
      * No lane maps two distinct C/D elements to the same accumulator VGPR.
      * C and D use the same lane/VGPR/coordinate ownership map.

    It does NOT claim that a given lane/VGPR/coordinate formula is the
    hardware's canonical MFMA formula. Compare that stronger claim against
    an ISA-calculator export before treating the map as opcode-authoritative.
    """
    issues: List[ValidationIssue] = []

    required_operands = ("A", "B", "C", "D")
    expected_elements = shape.expected_elements
    bounds = shape.operand_bounds

    normalized: Dict[str, List[FragmentElement]] = {}

    # ------------------------------------------------------------------
    # 1. Schema and element-level checks.
    # ------------------------------------------------------------------
    for operand in required_operands:
        if operand not in fragments:
            issues.append(ValidationIssue(
                severity="error",
                code="missing-operand",
                message=f"Fragment map is missing required operand {operand}",
                operand=operand,
            ))
            normalized[operand] = []
            continue

        elems = list(fragments[operand])
        normalized[operand] = elems

        if len(elems) != expected_elements[operand]:
            issues.append(ValidationIssue(
                severity="error",
                code="wrong-element-count",
                message=(
                    f"Expected {expected_elements[operand]} logical elements, "
                    f"found {len(elems)}"
                ),
                operand=operand,
            ))

        row_limit, col_limit = bounds[operand]

        for e in elems:
            if e.operand != operand:
                issues.append(ValidationIssue(
                    severity="error",
                    code="wrong-operand-tag",
                    message=(
                        f"Element appears in {operand} list but has "
                        f"operand tag {e.operand!r}"
                    ),
                    operand=operand,
                    lane=e.lane,
                    vgpr=e.vgpr,
                    coordinate=(e.logical_row, e.logical_col),
                ))

            if not (0 <= e.lane < shape.wave_size):
                issues.append(ValidationIssue(
                    severity="error",
                    code="lane-out-of-range",
                    message=f"Lane must be in [0, {shape.wave_size})",
                    operand=operand,
                    lane=e.lane,
                    vgpr=e.vgpr,
                    coordinate=(e.logical_row, e.logical_col),
                ))

            if e.vgpr < 0:
                issues.append(ValidationIssue(
                    severity="error",
                    code="negative-vgpr",
                    message="VGPR index must be non-negative",
                    operand=operand,
                    lane=e.lane,
                    vgpr=e.vgpr,
                    coordinate=(e.logical_row, e.logical_col),
                ))

            if not (0 <= e.logical_row < row_limit):
                issues.append(ValidationIssue(
                    severity="error",
                    code="row-out-of-range",
                    message=f"Row must be in [0, {row_limit})",
                    operand=operand,
                    lane=e.lane,
                    vgpr=e.vgpr,
                    coordinate=(e.logical_row, e.logical_col),
                ))

            if not (0 <= e.logical_col < col_limit):
                issues.append(ValidationIssue(
                    severity="error",
                    code="column-out-of-range",
                    message=f"Column must be in [0, {col_limit})",
                    operand=operand,
                    lane=e.lane,
                    vgpr=e.vgpr,
                    coordinate=(e.logical_row, e.logical_col),
                ))

            if operand in ("A", "B"):
                if e.packed_half not in (0, 1):
                    issues.append(ValidationIssue(
                        severity="error",
                        code="invalid-fp16-half",
                        message=(
                            "A/B entries must identify packed_half=0 (low) "
                            "or packed_half=1 (high)"
                        ),
                        operand=operand,
                        lane=e.lane,
                        vgpr=e.vgpr,
                        coordinate=(e.logical_row, e.logical_col),
                    ))
            else:
                if e.packed_half is not None:
                    issues.append(ValidationIssue(
                        severity="error",
                        code="invalid-fp32-packing",
                        message=(
                            "C/D entries are FP32 accumulator values and "
                            "must use packed_half=None"
                        ),
                        operand=operand,
                        lane=e.lane,
                        vgpr=e.vgpr,
                        coordinate=(e.logical_row, e.logical_col),
                    ))

    unexpected = sorted(set(fragments) - set(required_operands))
    for operand in unexpected:
        issues.append(ValidationIssue(
            severity="warning",
            code="unexpected-operand",
            message=f"Ignoring unexpected fragment-map operand {operand!r}",
            operand=operand,
        ))

    # ------------------------------------------------------------------
    # 2. Coordinate bijectivity: every logical matrix element must appear
    # exactly once across the wave.
    # ------------------------------------------------------------------
    unique_coordinate_counts: Dict[str, int] = {}

    for operand in required_operands:
        elems = normalized[operand]
        coord_to_entries: Dict[Tuple[int, int], List[FragmentElement]] = defaultdict(list)

        for e in elems:
            coord_to_entries[(e.logical_row, e.logical_col)].append(e)

        unique_coordinate_counts[operand] = len(coord_to_entries)

        row_limit, col_limit = bounds[operand]
        expected_coords = {
            (row, col)
            for row in range(row_limit)
            for col in range(col_limit)
        }

        actual_coords = set(coord_to_entries)
        missing = sorted(expected_coords - actual_coords)
        extra = sorted(actual_coords - expected_coords)

        if missing:
            issues.append(ValidationIssue(
                severity="error",
                code="missing-logical-coordinates",
                message=(
                    f"Map omits {len(missing)} logical coordinate(s); "
                    f"first few: {missing[:8]}"
                ),
                operand=operand,
            ))

        if extra:
            issues.append(ValidationIssue(
                severity="error",
                code="extra-logical-coordinates",
                message=(
                    f"Map contains {len(extra)} out-of-domain coordinate(s); "
                    f"first few: {extra[:8]}"
                ),
                operand=operand,
            ))

        for coordinate, entries in coord_to_entries.items():
            if len(entries) > 1:
                owners = [(e.lane, e.vgpr, e.packed_half) for e in entries]
                issues.append(ValidationIssue(
                    severity="error",
                    code="duplicate-logical-coordinate",
                    message=(
                        f"Logical matrix element has {len(entries)} owners: "
                        f"{owners}"
                    ),
                    operand=operand,
                    coordinate=coordinate,
                ))

    # ------------------------------------------------------------------
    # 3. Per-lane occupancy.
    #
    # For this exact tile:
    # A: 16*16 / 64 = 4 FP16 values per lane
    # B: 16*16 / 64 = 4 FP16 values per lane
    # C: 16*16 / 64 = 4 FP32 values per lane
    # D: 16*16 / 64 = 4 FP32 values per lane
    # ------------------------------------------------------------------
    expected_per_lane = {"A": 4, "B": 4, "C": 4, "D": 4}
    per_lane_element_counts: Dict[str, Dict[int, int]] = {}
    per_lane_vgpr_counts: Dict[str, Dict[int, int]] = {}

    for operand in required_operands:
        elems = normalized[operand]
        counts = Counter(e.lane for e in elems)
        per_lane_element_counts[operand] = {
            lane: counts.get(lane, 0)
            for lane in range(shape.wave_size)
        }

        distinct_vgprs_by_lane: Dict[int, set[int]] = defaultdict(set)
        for e in elems:
            if 0 <= e.lane < shape.wave_size:
                distinct_vgprs_by_lane[e.lane].add(e.vgpr)

        per_lane_vgpr_counts[operand] = {
            lane: len(distinct_vgprs_by_lane.get(lane, set()))
            for lane in range(shape.wave_size)
        }

        require_all = (
            operand in ("A", "B") and require_all_lanes_for_ab
        ) or (
            operand in ("C", "D") and require_all_lanes_for_cd
        )

        for lane in range(shape.wave_size):
            actual = counts.get(lane, 0)

            if require_all and actual != expected_per_lane[operand]:
                issues.append(ValidationIssue(
                    severity="error",
                    code="wrong-per-lane-element-count",
                    message=(
                        f"Expected {expected_per_lane[operand]} elements in "
                        f"this lane, found {actual}"
                    ),
                    operand=operand,
                    lane=lane,
                ))
            elif not require_all and actual not in (0, expected_per_lane[operand]):
                issues.append(ValidationIssue(
                    severity="error",
                    code="partial-lane-fragment",
                    message=(
                        f"Lane owns {actual} values; expected either 0 or "
                        f"{expected_per_lane[operand]}"
                    ),
                    operand=operand,
                    lane=lane,
                ))

    # ------------------------------------------------------------------
    # 4. Packed FP16 register validity for A and B.
    #
    # Every input VGPR dword represented in this model must contain exactly
    # a low and high FP16 value for the same lane. With 4 values/lane this
    # gives exactly two distinct source VGPR dwords per lane.
    # ------------------------------------------------------------------
    for operand in ("A", "B"):
        by_lane_vgpr: Dict[Tuple[int, int], List[FragmentElement]] = defaultdict(list)

        for e in normalized[operand]:
            if 0 <= e.lane < shape.wave_size:
                by_lane_vgpr[(e.lane, e.vgpr)].append(e)

        for lane in range(shape.wave_size):
            lane_regs = [
                vgpr
                for (entry_lane, vgpr) in by_lane_vgpr
                if entry_lane == lane
            ]

            if strict_register_layout and len(lane_regs) != 2:
                issues.append(ValidationIssue(
                    severity="error",
                    code="wrong-input-vgpr-count",
                    message=(
                        "Expected exactly 2 packed-FP16 source VGPR dwords "
                        "for this lane"
                    ),
                    operand=operand,
                    lane=lane,
                ))

        for (lane, vgpr), entries in by_lane_vgpr.items():
            half_counts = Counter(e.packed_half for e in entries)

            if len(entries) != 2:
                issues.append(ValidationIssue(
                    severity="error",
                    code="wrong-packed-vgpr-arity",
                    message=(
                        f"Packed FP16 source VGPR must own exactly 2 logical "
                        f"halves, found {len(entries)}"
                    ),
                    operand=operand,
                    lane=lane,
                    vgpr=vgpr,
                ))
                continue

            if half_counts.get(0, 0) != 1 or half_counts.get(1, 0) != 1:
                issues.append(ValidationIssue(
                    severity="error",
                    code="invalid-packed-half-pair",
                    message=(
                        "Packed FP16 source VGPR must contain exactly one "
                        "low half and one high half"
                    ),
                    operand=operand,
                    lane=lane,
                    vgpr=vgpr,
                ))

    # ------------------------------------------------------------------
    # 5. FP32 accumulator register validity for C and D.
    #
    # A lane owns 4 output values. In the conventional model, they occupy
    # four distinct accumulator-register positions. No two distinct
    # coordinates may alias one (lane, vgpr) location.
    # ------------------------------------------------------------------
    for operand in ("C", "D"):
        by_lane_vgpr: Dict[Tuple[int, int], List[FragmentElement]] = defaultdict(list)

        for e in normalized[operand]:
            if 0 <= e.lane < shape.wave_size:
                by_lane_vgpr[(e.lane, e.vgpr)].append(e)

        for lane in range(shape.wave_size):
            regs = {
                e.vgpr
                for e in normalized[operand]
                if e.lane == lane
            }

            if strict_register_layout and len(regs) != 4:
                issues.append(ValidationIssue(
                    severity="error",
                    code="wrong-accumulator-vgpr-count",
                    message=(
                        "Expected exactly 4 distinct FP32 accumulator VGPRs "
                        "for this lane"
                    ),
                    operand=operand,
                    lane=lane,
                ))

        for (lane, vgpr), entries in by_lane_vgpr.items():
            if len(entries) != 1:
                coords = [(e.logical_row, e.logical_col) for e in entries]
                issues.append(ValidationIssue(
                    severity="error",
                    code="accumulator-vgpr-alias",
                    message=(
                        f"One accumulator VGPR aliases {len(entries)} "
                        f"distinct FP32 values: {coords}"
                    ),
                    operand=operand,
                    lane=lane,
                    vgpr=vgpr,
                ))

    # ------------------------------------------------------------------
    # 6. C/D correspondence.
    #
    # An MFMA updates C into D with identical fragment ownership. The values
    # differ, but (lane, vgpr) -> (logical row, logical column) should match.
    # ------------------------------------------------------------------
    if require_c_d_same_layout:
        def accumulator_ownership(
            entries: Iterable[FragmentElement],
        ) -> Dict[Tuple[int, int], Tuple[int, int]]:
            result: Dict[Tuple[int, int], Tuple[int, int]] = {}

            for e in entries:
                key = (e.lane, e.vgpr)
                value = (e.logical_row, e.logical_col)

                if key not in result:
                    result[key] = value

            return result

        c_layout = accumulator_ownership(normalized["C"])
        d_layout = accumulator_ownership(normalized["D"])

        if c_layout != d_layout:
            c_keys = set(c_layout)
            d_keys = set(d_layout)

            missing_in_d = sorted(c_keys - d_keys)
            extra_in_d = sorted(d_keys - c_keys)
            changed = sorted(
                key for key in (c_keys & d_keys)
                if c_layout[key] != d_layout[key]
            )

            issues.append(ValidationIssue(
                severity="error",
                code="c-d-layout-mismatch",
                message=(
                    "C and D must have identical accumulator ownership; "
                    f"missing-in-D={missing_in_d[:8]}, "
                    f"extra-in-D={extra_in_d[:8]}, "
                    f"changed={[(key, c_layout[key], d_layout[key]) for key in changed[:8]]}"
                ),
            ))

    # ------------------------------------------------------------------
    # 7. Soft checks: source VGPR numbering may be local tuple offsets
    # rather than absolute hardware VGPR IDs. Emit warnings only.
    # ------------------------------------------------------------------
    for operand in ("A", "B", "C", "D"):
        used = sorted({e.vgpr for e in normalized[operand]})

        if not used:
            continue

        contiguous = used == list(range(used[0], used[-1] + 1))
        if not contiguous:
            issues.append(ValidationIssue(
                severity="warning",
                code="noncontiguous-vgpr-numbering",
                message=(
                    f"{operand} uses non-contiguous VGPR indices {used}; "
                    "this can be valid for an absolute register allocation, "
                    "but is unexpected for a compact local fragment tuple"
                ),
                operand=operand,
            ))

    report = FragmentMapReport(
        shape=shape,
        issues=tuple(issues),
        element_counts={
            operand: len(normalized[operand])
            for operand in required_operands
        },
        unique_coordinate_counts=unique_coordinate_counts,
        per_lane_element_counts=per_lane_element_counts,
        per_lane_vgpr_counts=per_lane_vgpr_counts,
    )

    report.raise_if_invalid()
    return report


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    from fragment_map import generate_v_mfma_f32_16x16x16f16_fragments

    frags = generate_v_mfma_f32_16x16x16f16_fragments()

    report = validate_fragment_map(
        frags,
        shape=MfmaShape(
            target="gfx942",
            opcode="v_mfma_f32_16x16x16f16",
            m=16,
            n=16,
            k=16,
            wave_size=64,
        ),
    )

    print("Fragment map is structurally valid.")
    print(f"Element counts: {report.element_counts}")
    print(f"Unique coordinate counts: {report.unique_coordinate_counts}")