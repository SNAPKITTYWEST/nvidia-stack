; gfx942 MFMA GEMM Kernel Fragment: Direct LDS Staging Path
; Assumes: 16x16x16 MFMA, FP16 input, FP32 acc
; LDS allocation: A tile (0.5KB), B tile (0.5KB) ping-pong buffers

; s0-s3: A/B buffer descriptors (global mem)
; s4: K-loop counter
; s5: LDS base offset for A current tile
; s6: LDS base offset for B current tile
; s7: LDS base offset for A next tile (s5 + 0x200)
; s8: LDS base offset for B next tile (s6 + 0x200)
; v0-v3: Accumulator registers (c0-c3)
; v4-v7: A fragment registers
; v8-v11: B fragment registers

; ===== PROLOGUE: Load initial tiles into LDS[0] =====
buffer_load_lds v[0:1], s[0:3], 0 offen offset:0 lds:0 ; Load A tile (coalesced)
buffer_load_lds v[2:3], s[0:3], 0 offen offset:0 lds:0 ; Load B tile (coalesced)
s_waitcnt vmcnt(0) ; Wait for this wave's global loads
s_barrier ; Workgroup sync: all waves populated LDS[0]

; ===== MAIN K-LOOP =====
.L_loop:
    ; Prefetch NEXT tile into LDS[1] (overlap with current MFMA)
    buffer_load_lds v[0:1], s[0:3], 0 offen offset:0 lds:1 ; A next
    buffer_load_lds v[2:3], s[0:3], 0 offen offset:0 lds:1 ; B next

    ; Consume CURRENT tile from LDS[0] -> VGPR fragments
    ; (Example: 16x16 tile -> 4 lanes * 4 fragments each for MFMA)
    ds_read_b32 v4, s5 offset:0 ; Lane 0: A frag0
    ds_read_b32 v5, s5 offset:4 ; Lane 0: A frag1
    ds_read_b32 v6, s5 offset:8 ; Lane 0: A frag2
    ds_read_b32 v7, s5 offset:12 ; Lane 0: A frag3
    ds_read_b32 v8, s6 offset:0 ; Lane 0: B frag0
    ds_read_b32 v9, s6 offset:4 ; Lane 0: B frag1
    ds_read_b32 v10, s6 offset:8 ; Lane 0: B frag2
    ds_read_b32 v11, s6 offset:12 ; Lane 0: B frag3
    ; ... (other lanes implicitly handled by ds_read addressing)

    s_waitcnt lgkmcnt(0) ; Wait for LDS reads to complete

    ; MFMA operation on VGPR-resident fragments
    v_mfma_f32_16x16x16f16 v[0:3], v4, v5, v[0:3], 0, 0, 0 ; C += A*B
    v_mfma_f32_16x16x16f16 v[0:3], v6, v7, v[0:3], 0, 0, 0
    v_mfma_f32_16x16x16f16 v[0:3], v8, v9, v[0:3], 0, 0, 0
    v_mfma_f32_16x16x16f16 v[0:3], v10, v11, v[0:3], 0, 0, 0

    ; Prepare for buffer swap: wait for next tile prefetch to finish
    s_waitcnt vmcnt(0) ; Ensure global->LDS[1] done
    s_barrier ; All waves agree: LDS[1] ready

    ; Swap ping-pong buffers (advance K pointers implicitly via s4)
    s_add s5, s5, 0x400 ; A current = A next
    s_add s6, s6, 0x400 ; B current = B next
    s_sub s7, s7, 0x400 ; A next = A current (for next iter)
    s_sub s8, s8, 0x400 ; B next = B current

    s_sub s4, s4, 1 ; Decrement K tile counter
    s_cbranch scc1 .L_loop ; Loop if more K tiles

; ===== EPILOGUE: Store C (not shown per focus on staging) =====
; v[0:3] holds final accumulators -> global store via vector_store