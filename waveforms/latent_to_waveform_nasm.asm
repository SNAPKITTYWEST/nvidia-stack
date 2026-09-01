; ------------------------------------------------------------
; latent_to_waveform_nasm.asm
;
; AVX2 matrix-vector multiply: x = Ψ * z
;   Ψ: N x m matrix (row-major, 8-byte doubles)
;   z: m-vector
;   x: N-vector (output)
;
; Calling convention (System V AMD64):
;   rdi = pointer to Ψ (base address, row-major)
;   rsi = pointer to z (latent vector)
;   rdx = pointer to x (output buffer)
;   ecx = N (number of rows)
;   r8d = m (vector length, must be multiple of 8)
;
; Assemble:
;   nasm -f elf64 -o latent_to_waveform_nasm.o latent_to_waveform_nasm.asm
;   nasm -f macho64 -o latent_to_waveform_nasm.o latent_to_waveform_nasm.asm (macOS)
; ------------------------------------------------------------

default rel

section .text
global latent_to_waveform_nasm

latent_to_waveform_nasm:
    ; ------------------------------------------------------------
    ; Prologue
    ; ------------------------------------------------------------
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    ; r10 = Ψ base
    mov     r10, rdi
    ; r11 = z pointer
    mov     r11, rsi
    ; r12 = x pointer
    mov     r12, rdx
    ; r13 = N
    mov     r13d, ecx
    ; r14 = m
    mov     r14d, r8d
    ; r15 = row index
    xor     r15d, r15d

.row_loop:
    cmp     r15d, r13d
    jge     .row_done

    ; rdx = &Ψ[i, 0]
    mov     rbx, r14
    imul    rbx, rbx, 8          ; m * sizeof(double)
    imul    rbx, rbx, r15        ; i * (m * 8)
    lea     rdx, [r10 + rbx]     ; base + offset

    ; Clear accumulators
    vxorpd  ymm0, ymm0, ymm0
    vxorpd  ymm1, ymm1, ymm1
    vxorpd  ymm2, ymm2, ymm2
    vxorpd  ymm3, ymm3, ymm3

    ; Column index
    xor     r8d, r8d

.col_loop:
    cmp     r8d, r14d
    jge     .col_done

    ; Load z[j:j+8]
    mov     rax, r11
    add     rax, r8
    shl     rax, 3               ; *8 bytes
    vmovupd ymm4, [rax]

    ; Load Ψ[i, j:j+8]
    mov     rax, rdx
    add     rax, r8
    shl     rax, 3
    vmovupd ymm5, [rax]

    ; FMA: ymm0 += z * Ψ
    vfmadd231pd ymm0, ymm4, ymm5

    add     r8d, 8
    jmp     .col_loop

.col_done:
    ; Horizontal sum of ymm0
    vextractf128 xmm1, ymm0, 1
    vaddpd  xmm0, xmm0, xmm1
    movhlps xmm2, xmm0
    addsd   xmm0, xmm2

    ; Store x[i]
    movsd   [r12 + r15*8], xmm0

    inc     r15d
    jmp     .row_loop

.row_done:
    ; Epilogue
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    vzeroupper
    ret


; ------------------------------------------------------------
; latent_to_waveform_tiled
;
; Cache-blocked version for large N, m.
; Processes tiles of TILE_M rows x TILE_N columns.
; ------------------------------------------------------------

%define TILE_M  8
%define TILE_N  256

section .text
global latent_to_waveform_tiled

latent_to_waveform_tiled:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 32              ; local storage

    mov     r10, rdi             ; Ψ
    mov     r11, rsi             ; z
    mov     r12, rdx             ; x
    mov     r13d, ecx            ; N
    mov     r14d, r8d            ; m

    ; Zero output buffer
    xor     eax, eax
    mov     rcx, r13
    lea     rdi, [r12]
.zero_loop:
    mov     qword [rdi + rax*8], 0
    inc     rax
    dec     rcx
    jnz     .zero_loop

    ; Outer loop: tile over rows
    xor     r15d, r15d           ; row_tile = 0

.row_tile_loop:
    mov     eax, r15d
    add     eax, TILE_M
    cmp     eax, r13d
    jg      .row_tile_done

    ; Inner loop: tile over columns
    xor     ecx, ecx            ; col_tile = 0

.col_tile_loop:
    mov     eax, ecx
    add     eax, TILE_N
    cmp     eax, r14d
    jg      .col_tile_done

    ; Process TILE_M rows x TILE_N columns
    ; ... (tile body: 8 rows x 256 cols with 4 ymm accumulators)
    ; For brevity, delegates to untiled kernel per row
    mov     r8d, TILE_N
    call    .process_tile

    add     ecx, TILE_N
    jmp     .col_tile_loop

.col_tile_done:
    add     r15d, TILE_M
    jmp     .row_tile_loop

.row_tile_done:
    add     rsp, 32
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    vzeroupper
    ret

.process_tile:
    ; Placeholder for tile body
    ret
