def calculate_ds_read_b128_padding(
    logical_row_words: int,
    lane_to_fragment_map: callable,
    max_padding: int = 16
) -> int:
    """
    Calculate minimal LDS padding (in 32-bit bank words) to eliminate ds_read_b128 conflicts
    for gfx942 (CDNA 3) hardware.
    
    Args:
        logical_row_words: Logical row width in 32-bit words (W = ceil(K*2/4) for FP16)
        lane_to_fragment_map: Function(lane_id) -> (row, col) in logical LDS coordinates
                             where col is in FP16 elements (not bank words)
        max_padding: Maximum padding to search (bank words)
    
    Returns:
        Minimal padding P (bank words) that yields conflict-free ds_read_b128
        Returns -1 if no solution found within max_padding
    
    Hardware constraints (gfx942):
        - 32 LDS banks, 4 bytes/bank
        - ds_read_b128 groups: 8 specific non-contiguous 8-lane groups
        - Each lane reads 4 consecutive 32-bit words (q=0,1,2,3)
        - 16-byte alignment required for ds_read_b128 source address
    """
    # gfx942 ds_read_b128 lane groups (from AMD documentation)
    DS_READ_B128_GROUPS = [
        list(range(0, 4)) + list(range(20, 24)), # G0: 0-3 + 20-23
        list(range(4, 8)) + list(range(16, 20)), # G1: 4-7 + 16-19
        list(range(8, 12)) + list(range(28, 32)), # G2: 8-11 + 28-31
        list(range(12, 16)) + list(range(24, 28)), # G3: 12-15 + 24-27
        list(range(32, 36)) + list(range(52, 56)), # G4: 32-35 + 52-55
        list(range(36, 40)) + list(range(48, 52)), # G5: 36-39 + 48-51
        list(range(40, 44)) + list(range(60, 64)), # G6: 40-43 + 60-63
        list(range(44, 48)) + list(range(56, 60)) # G7: 44-47 + 56-59
    ]
    
    def lds_address(lane_id: int, stride_words: int) -> int:
        """
        Calculate LDS byte address for a lane's ds_read_b128 source.
        Assumes lane_to_fragment_map returns (row, col) in logical FP16 elements.
        """
        row, col_fp16 = lane_to_fragment_map(lane_id)
        # Convert FP16 column to bank-word column (2 FP16 = 1 bank word)
        col_bank_word = col_fp16 // 2
        # Physical address in bytes: 4 * (row * stride_words + col_bank_word)
        return 4 * (row * stride_words + col_bank_word)
    
    def is_16byte_aligned(address: int) -> bool:
        """Check if address is 16-byte aligned (required for ds_read_b128)"""
        return address % 16 == 0
    
    def has_conflict(stride_words: int) -> bool:
        """Check if given stride causes any ds_read_b128 bank conflict"""
        for group in DS_READ_B128_GROUPS:
            for q in range(4): # q = 0,1,2,3 for the 4 dwords in b128
                bank_to_address = {} # Maps bank -> first address seen at this bank/q
                for lane in group:
                    addr = lds_address(lane, stride_words)
                    if not is_16byte_aligned(addr):
                        return True # Alignment violation
                    bank_word = addr // 4 # Convert byte address to bank-word index
                    bank = (bank_word + q) % 32 # Bank for this dword phase
                    if bank in bank_to_address:
                        # Conflict: different addresses mapping to same bank in same phase
                        if bank_to_address[bank] != addr + 4 * q:
                            return True
                    else:
                        bank_to_address[bank] = addr
        return False
    
    # Search for minimal padding
    for P in range(max_padding + 1):
        stride_words = logical_row_words + P
        if not has_conflict(stride_words):
            return P
    return -1 # No solution found

# EXAMPLE USAGE FOR gfx942 v_mfma_f32_16x16x16f16:
if __name__ == "__main__":
    # Lane-to-fragment map for A operand in v_mfma_f32_16x16x16f16
    # (From previous fragment: 8 FP16 elements as [2 rows × 4 columns])
    def a_fragment_map(lane_id: int) -> tuple[int, int]:
        m_in_tile = 2 * (lane_id // 32) + (lane_id % 2) # Row start [0,14] step 2
        k_in_tile = 4 * (lane_id % 16) # Column start [0,60] step 4
        # For ds_read_b128, we read 8 consecutive FP16 elements (4 bank words)
        # Starting at (m_in_tile, k_in_tile)
        return (m_in_tile, k_in_tile) # Returns logical (row, col) in FP16 elements
    
    # For FP16 row with 64 elements (typical MFMA K dimension)
    logical_row_words = 64 * 2 // 4 # 32 bank words
    
    padding = calculate_ds_read_b128_padding(
        logical_row_words=logical_row_words,
        lane_to_fragment_map=a_fragment_map,
        max_padding=16
    )
    
    if padding >= 0:
        print(f"Minimal padding: {padding} bank words")
        print(f" = {padding * 4} bytes")
        print(f" = {padding * 2} FP16 elements")
        print(f"Physical row stride: {logical_row_words + padding} bank words")
    else:
        print("No conflict-free padding found within search range")
        
    # To verify, plug padding into your kernel's LDS layout:
    # .align 256
    # .lgs A_tile: .skip ((64 + padding*2) * 16 * 2) ; 64 rows, (64+2P) cols, FP16