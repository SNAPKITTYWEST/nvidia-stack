from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional

# -----------------------------
# Data structures for read plan
# -----------------------------
@dataclass(frozen=True)
class FragmentElement:
    operand: str # "A", "B", "C", or "D"
    lane: int
    vgpr: int # VGPR index (0-based)
    packed_half: Optional[int] # 0=low, 1=high if two FP16 packed in VGPR, else None
    logical_row: int # row within the MFMA tile (0..15)
    logical_col: int # column within the MFMA tile (0..15)

@dataclass(frozen=True)
class ReadOp:
    lane: int
    operand: str # "A" or "B"
    address: int # LDS byte address for the b64 read (must be 4-byte aligned)
    width_bytes: int = 64 # width of the load in bits (64 for b64)

# -----------------------------
# Opcode-accurate fragment map for v_mfma_f32_16x16x16f16
# -----------------------------
def mfma_16x16x16_f16_a_coords(lane: int) -> List[Tuple[int, int, int, int]]:
    if not 0 <= lane < 64:
        raise ValueError("lane must be in 0..63")
    m = lane >> 2
    k0 = (lane & 0x3) << 2
    return [
        (m, k0 + 0, 0, 0), # (row, col, source-vgpr, half)
        (m, k0 + 1, 0, 1),
        (m, k0 + 2, 1, 0),
        (m, k0 + 3, 1, 1),
    ]

def mfma_16x16x16_f16_b_coords(lane: int) -> List[Tuple[int, int, int, int]]:
    if not 0 <= lane < 64:
        raise ValueError("lane must be in 0..63")
    k0 = (lane >> 4) << 2
    n = lane & 0xF
    return [
        (k0 + 0, n, 0, 0),
        (k0 + 1, n, 0, 1),
        (k0 + 2, n, 1, 0),
        (k0 + 3, n, 1, 1),
    ]

def mfma_16x16x16_f16_cd_coords(lane: int) -> List[Tuple[int, int, int]]:
    if not 0 <= lane < 64:
        raise ValueError("lane must be in 0..63")
    n = lane & 0xF
    m0 = lane >> 4
    return [
        (m0 + 0, n, 0),
        (m0 + 4, n, 1),
        (m0 + 8, n, 2),
        (m0 + 12, n, 3),
    ]

def generate_v_mfma_f32_16x16x16f16_fragments() -> Dict[str, List[FragmentElement]]:
    fragments: Dict[str, List[FragmentElement]] = {"A": [], "B": [], "C": [], "D": []}
    for lane in range(64):
        for m, k, reg, half in mfma_16x16x16_f16_a_coords(lane):
            fragments["A"].append(FragmentElement(
                operand="A", lane=lane, vgpr=reg, packed_half=half,
                logical_row=m, logical_col=k
            ))
        for k, n, reg, half in mfma_16x16x16_f16_b_coords(lane):
            fragments["B"].append(FragmentElement(
                operand="B", lane=lane, vgpr=reg, packed_half=half,
                logical_row=k, logical_col=n
            ))
        for m, n, reg in mfma_16x16x16_f16_cd_coords(lane):
            fragments["C"].append(FragmentElement(
                operand="C", lane=lane, vgpr=reg, packed_half=None,
                logical_row=m, logical_col=n
            ))
            fragments["D"].append(FragmentElement(
                operand="D", lane=lane, vgpr=reg, packed_half=None,
                logical_row=m, logical_col=n
            ))
    return fragments

# -----------------------------
# Validate the fragment map
# -----------------------------
def validate_fragment_map(
    fragments: Dict[str, List[FragmentElement]],
    m: int = 16,
    n: int = 16,
    k: int = 16,
) -> None:
    expected = {
        "A": m * k,
        "B": k * n,
        "C": m * n,
        "D": m * n,
    }
    for operand, count in expected.items():
        actual = len(fragments[operand])
        if actual != count:
            raise ValueError(
                f"{operand}: expected {count} logical elements, got {actual}"
            )
        coords = {
            (x.logical_row, x.logical_col)
            for x in fragments[operand]
        }
        if len(coords) != count:
            raise ValueError(
                f"{operand}: logical-coordinate map is not bijective; "
                f"{len(coords)} unique coordinates for {count} elements"
            )

# -----------------------------
# Build a ReadPlan from fragment elements (for b64 loads)
# -----------------------------
def build_read_plan_b64(
    elements: List[FragmentElement],
    operand: str,
    opcode: str = "v_mfma_f32_16x16x16f16"
) -> List[ReadOp]:
    """
    Assumes each lane's four FP16 elements are to be loaded with one ds_read_b64.
    The four elements must be stored in LDS as two consecutive 32-bit words:
        word0: [elem0, elem1] at address A
        word1: [elem2, elem3] at address A+4
    and the address A must be 4-byte aligned.
    We compute the address per lane from the logical coordinates and a layout function
    that will be provided later (here we just return a placeholder; the address will be
    filled in by the layout function).
    """
    # Group by lane
    lane_to_elements: Dict[int, List[FragmentElement]] = {}
    for elem in elements:
        lane_to_elements.setdefault(elem.lane, []).append(elem)
    
    reads: List[ReadOp] = []
    for lane in range(64):
        elems = lane_to_elements[lane]
        if len(elems) != 4:
            raise ValueError(f"Lane {lane} has {len(elems)} elements, expected 4")
        # Sort by logical coordinate to ensure consistent ordering
        elems.sort(key=lambda e: (e.logical_row, e.logical_col))
        # We will not compute the address here; we leave it as 0 and will fill it later
        reads.append(ReadOp(
            lane=lane,
            operand=operand,
            address=0, # placeholder
            width_bytes=64
        ))
    return reads

# -----------------------------
# LDS address functions for A and B (to be used with layout)
# -----------------------------
def address_A(
    lane: int,
    row_stride_fp16: int, # in FP16 elements, must be even
) -> int:
    """
    Compute LDS byte address for the b64 read of operand A for a given lane.
    Assumes row-major storage with row stride = row_stride_fp16 (FP16 elements).
    Address = 2 * [ m * row_stride_fp16 + k_start ]
    where m = lane >> 2, k_start = (lane & 0x3) << 2
    """
    m = lane >> 2
    k_start = (lane & 0x3) << 2
    index = m * row_stride_fp16 + k_start
    return 2 * index # byte address

def address_B(
    lane: int,
    col_stride_fp16: int, # in FP16 elements, must be even (column stride in column-major)
) -> int:
    """
    Compute LDS byte address for the b64 read of operand B for a given lane.
    Assumes column-major storage with column stride = col_stride_fp16 (FP16 elements).
    Address = 2 * [ n * col_stride_fp16 + k_start ]
    where k_start = (lane >> 4) << 2, n = lane & 0xF
    """
    k_start = (lane >> 4) << 2
    n = lane & 0xF
    index = n * col_stride_fp16 + k_start
    return 2 * index # byte address

# -----------------------------
# Conflict detection for b64 reads (two 32-bit words)
# -----------------------------
DS_READ_B128_GROUPS = [
    list(range(0, 4)) + list(range(20, 24)), # G0
    list(range(4, 8)) + list(range(16, 20)), # G1
    list(range(8, 12)) + list(range(28, 32)), # G2
    list(range(12, 16)) + list(range(24, 28)), # G3
    list(range(32, 36)) + list(range(52, 56)), # G4
    list(range(36, 40)) + list(range(48, 52)), # G5
    list(range(40, 44)) + list(range(60, 64)), # G6
    list(range(44, 48)) + list(range(56, 60)), # G7
]

def conflict_report_b64(
    read_ops: List[ReadOp],
    address_of: Callable[[int], int] # function(lane) -> address
) -> List[dict]:
    conflicts = []
    for gid, group in enumerate(DS_READ_B128_GROUPS):
        for q in range(2): # dword phase within b64 (q=0,1)
            bank_to_entries: Dict[int, List[Tuple[int, int]]] = {}
            for lane in group:
                addr = address_of(lane)
                if addr % 4 != 0:
                    conflicts.append({
                        "kind": "misalignment",
                        "group": gid,
                        "q": q,
                        "lane": lane,
                        "base_addr": addr,
                    })
                    continue
                word_addr = (addr // 4) + q
                bank = word_addr % 32
                bank_to_entries.setdefault(bank, []).append((lane, word_addr))
            for bank, entries in bank_to_entries.items():
                distinct = {wd for _, wd in entries}
                if len(distinct) > 1:
                    conflicts.append({
                        "kind": "bank-conflict",
                        "group": gid,
                        "q": q,
                        "bank": bank,
                        "accesses": entries,
                        "way": len(distinct),
                    })
    return conflicts

def has_conflict_b64(read_ops: List[ReadOp], address_of: Callable[[int], int]) -> bool:
    return bool(conflict_report_b64(read_ops, address_of))

# -----------------------------
# Layout search for A and B (padding only)
# -----------------------------
def find_layout_padding(
    address_func: Callable[[int, int], int], # func(lane, stride) -> address
    max_padding: int = 32
) -> Optional[Dict]:
    """
    Tries padding (making the stride even) to eliminate b64 bank conflicts.
    Returns the first layout (dict) that yields zero conflicts and 4-byte alignment.
    """
    for P in range(max_padding + 1):
        stride = 16 + P # logical dimension in FP16 elements
        if stride % 2 != 0: # must be even to ensure 4-byte alignment
            continue
        # Create address function for this stride
        def addr_fn(lane_id: int) -> int:
            return address_func(lane_id, stride)
        # Build read plan (we don't have the fragment elements here, but we know there are 64 lanes)
        # We'll create a dummy read plan with 64 lanes, each with a ReadOp (address to be filled by addr_fn)
        reads = [ReadOp(lane=i, operand="dummy", address=0, width_bytes=64) for i in range(64)]
        # Now fill in the address
        reads_with_addr = [
            ReadOp(
                lane=read.lane,
                operand=read.operand,
                address=addr_fn(read.lane),
                width_bytes=read.width_bytes
            )
            for read in reads
        ]
        if not has_conflict_b64(reads_with_addr, addr_fn):
            return {
                "kind": "padded",
                "pad_words": P,
                "stride_fp16": stride,
                "conflicts": []
            }
    return None

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Generate and validate the fragment map
    frags = generate_v_mfma_f32_16x16x16f16_fragments()
    validate_fragment_map(frags)
    print("Fragment map validation passed.")
    
    # Build read plans (we only need the lane count for now)
    plan_a = build_read_plan_b64(frags["A"], operand="A")
    plan_b = build_read_plan_b64(frags["B"], operand="B")
    
    print("\n=== Operand A (row-major) ===")
    layout_a = find_layout_padding(address_A, max_padding=32)
    if layout_a:
        print(f"Layout: {layout_a['kind']}")
        print(f" Padding: {layout_a['pad_words']} FP16 elements")
        print(f" Row stride: {layout_a['stride_fp16']} FP16 elements")
        print(f" = {layout_a['stride_fp16'] * 2} bytes")
    else:
        print("No conflict-free padding found for A")

    print("\n=== Operand B (column-major) ===")
    layout_b = find_layout_padding(address_B, max_padding=32)
    if layout_b:
        print(f"Layout: {layout_b['kind']}")
        print(f" Padding: {layout_b['pad_words']} FP16 elements")
        print(f" Column stride: {layout_b['stride_fp16']} FP16 elements")
        print(f" = {layout_b['stride_fp16'] * 2} bytes")
    else:
        print("No conflict-free padding found for B")

    # Emit a machine-readable certificate (JSON-like) for the chosen layout
    if layout_a and layout_b:
        cert = {
            "target": "gfx942",
            "opcode": "v_mfma_f32_16x16x16f16",
            "wavefront_size": 64,
            "mfma_tile": { "M": 16, "N": 16, "K": 16 },
            "operand_A": {
                "fragment_map_sha256": "TODO",
                "lds_layout": {
                    "kind": layout_a["kind"],
                    "row_stride_fp16": layout_a["stride_fp16"],
                    "pad_words": layout_a["pad_words"],
                },
                "load": "ds_read_b64",
                "conflicts": layout_a["conflicts"]
            },
            "operand_B": {
                "fragment_map_sha256": "TODO",
                "lds_layout": {
                    "kind": layout_b["kind"],
                    "col_stride_fp16": layout_b["stride_fp16"],
                    "pad_words": layout_b["pad_words"],
                },
                "load": "ds_read_b64",
                "conflicts": layout_b["conflicts"]
            }
        }
        import json
        print("\n=== Layout Certificate ===")
        print(json.dumps(cert, indent=2))