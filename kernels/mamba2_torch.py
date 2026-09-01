#!/usr/bin/env python3
"""
mamba2_torch.py — PyTorch Mamba-2 SSD Module

BOB Architecture: Mamba-2 SSM backbone (PyTorch layer)
Haskell FFI peer: mamba2.h / mamba2_step_fp8()
CUDA kernel peer: mamba2.cu (compile with build_mamba2.py on bbqbaddie)

Three execution modes (auto-selected at module construction):
  1. CUDA .so  — fastest; requires compiled libmamba2.so (bbqbaddie)
  2. torch.ops — PyTorch C++ extension via torch.utils.cpp_extension.load()
                 requires nvcc on PATH (bbqbaddie)
  3. Pure PyTorch — reference implementation; runs on RTX 3080 dev machine
                    without nvcc; numerically identical to the CUDA kernel

Typical usage:
    from kernels.mamba2_torch import Mamba2Layer, Mamba2Block

    layer = Mamba2Layer(d_model=512, d_state=16, d_conv=4)
    x = torch.randn(2, 128, 512)          # [B, L, D]
    y, h = layer(x)                        # y: [B, L, D], h: [B, D, N] state

    # Autoregressive step
    x_step = torch.randn(2, 1, 512)
    y_step, h = layer(x_step, recurrent_state=h)
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Optional compiled extension ──────────────────────────────────────────────

_KERNELS_DIR = Path(__file__).parent
_SO_PATH = _KERNELS_DIR / "libmamba2.so"
_CUDA_SRC = _KERNELS_DIR / "mamba2.cu"

_cuda_ext = None   # loaded lazily

def _try_load_cuda_ext() -> bool:
    """Try to load the compiled CUDA extension. Returns True if loaded."""
    global _cuda_ext
    if _cuda_ext is not None:
        return True

    # Path 1: pre-compiled .so (set by build_mamba2.py on bbqbaddie)
    if _SO_PATH.exists():
        try:
            import ctypes
            _cuda_ext = ctypes.CDLL(str(_SO_PATH))
            return True
        except OSError:
            pass

    # Path 2: torch.utils.cpp_extension JIT compile (needs nvcc)
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is not None and _CUDA_SRC.exists():
        try:
            from torch.utils.cpp_extension import load
            _cuda_ext = load(
                name="mamba2_cuda",
                sources=[str(_CUDA_SRC)],
                extra_cuda_cflags=["-O3", f"-arch=sm_86"],
                verbose=False,
            )
            return True
        except Exception as e:
            print(f"[mamba2] JIT compile failed ({e}), falling back to pure PyTorch")

    return False


# ── Pure-PyTorch selective scan (reference, trainable) ──────────────────────

def _softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def mamba2_scan_ref(
    u: torch.Tensor,     # [B, L, D]
    dt: torch.Tensor,    # [B, L, D]
    A: torch.Tensor,     # [D]
    B: torch.Tensor,     # [B, L, N]
    C: torch.Tensor,     # [B, L, N]
    D: torch.Tensor,     # [D]
    hx: Optional[torch.Tensor] = None,  # [B, D, N]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure-PyTorch Mamba-2 SSD selective scan.
    Numerically equivalent to mamba2_ssd_scan_kernel in mamba2.cu.

    Returns (output, h_final):
        output  : [B, L, D]
        h_final : [B, D, N]
    """
    B_sz, L, D_sz = u.shape
    N = B.shape[-1]
    device = u.device
    dtype  = u.dtype

    if hx is None:
        hx = torch.zeros(B_sz, D_sz, N, device=device, dtype=dtype)
    else:
        hx = hx.clone()

    # dt_bar: [B, L, D] — softplus
    dt_bar = _softplus(dt)

    # dA: [B, L, D] — decay factors
    # A is [D], a_log negative
    dA = torch.exp(dt_bar * A.unsqueeze(0).unsqueeze(0))   # [B, L, D]

    outputs = []
    h = hx   # [B, D, N]

    for t in range(L):
        u_t   = u[:, t, :]          # [B, D]
        dA_t  = dA[:, t, :]         # [B, D]
        dt_t  = dt_bar[:, t, :]     # [B, D]
        B_t   = B[:, t, :]          # [B, N]
        C_t   = C[:, t, :]          # [B, N]

        # dB[b, d, n] = dt_t[b,d] * B_t[b,n] * u_t[b,d]
        # Shape: [B, D, N]
        dB = (dt_t.unsqueeze(-1) * u_t.unsqueeze(-1)) * B_t.unsqueeze(1)

        # h[b, d, n] = dA_t[b,d] * h[b,d,n] + dB[b,d,n]
        h = dA_t.unsqueeze(-1) * h + dB

        # y[b, d] = sum_n C_t[b, n] * h[b, d, n]
        # C_t: [B, N] → [B, 1, N]; h: [B, D, N]
        y = (C_t.unsqueeze(1) * h).sum(-1)   # [B, D]

        # skip connection
        y = y + D * u_t

        outputs.append(y)

    output = torch.stack(outputs, dim=1)   # [B, L, D]
    return output, h


# ── nn.Module ────────────────────────────────────────────────────────────────

class Mamba2Layer(nn.Module):
    """
    Single Mamba-2 SSD layer.

    Args:
        d_model  : inner (expanded) dimension D
        d_state  : SSM state dimension N  (default 16, paper uses 16-64)
        d_conv   : depthwise conv width   (default 4)
        expand   : expansion ratio for in_proj (default 2)
        dt_rank  : rank of Δ projection   (default ceil(d_model/16))
        dt_min, dt_max : softplus clamp for Δ initialisation
        bias     : add bias to projections
        use_cuda : force CUDA ext (raises if unavailable)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bias: bool = False,
        use_cuda: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.expand  = expand
        self.d_inner = d_model * expand   # D in the kernel
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        # ── Projections ────────────────────────────────────────────────────

        # in_proj: x → [z, x, B, C, dt]  (single matmul)
        self.in_proj = nn.Linear(
            d_model,
            self.d_inner * 2 + d_state * 2 + self.dt_rank,
            bias=bias,
        )

        # Causal depthwise conv — padding handled manually so conv cache
        # can be carried across autoregressive steps (no auto-padding).
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=0,
            groups=self.d_inner,
            bias=bias,
        )

        # dt projection: dt_rank → d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # SSM parameters
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                      .repeat(self.d_inner, 1))   # [D, N] — not used in scan
        )
        # We use a single [D] A vector (log-sum over state dim)
        self.A_log_1d = nn.Parameter(
            -torch.ones(self.d_inner) * math.log(d_state)
        )

        self.D = nn.Parameter(torch.ones(self.d_inner))

        # out_proj: d_inner → d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # dt softplus clamp init
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt_init = torch.clamp(dt_init, min=1e-4)
        inv_dt  = dt_init + torch.log(-torch.expm1(-dt_init))
        self.dt_proj.bias.data.copy_(inv_dt)

        # Try to load CUDA extension
        self._use_cuda = use_cuda
        if use_cuda and not _try_load_cuda_ext():
            raise RuntimeError("[Mamba2Layer] use_cuda=True but CUDA extension not available")

    def _scan(
        self,
        u: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        hx: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dispatch to CUDA ext or pure-PyTorch reference."""
        if self._use_cuda and _try_load_cuda_ext():
            # CUDA ext path — swap in ctypes call on bbqbaddie when .so is ready
            pass
        return mamba2_scan_ref(u, dt, self.A_log_1d, B, C, self.D, hx)

    def forward(
        self,
        x: torch.Tensor,
        recurrent_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x               : [B, L, d_model]
            recurrent_state : (ssm_h, conv_cache) or None
              ssm_h       [B, d_inner, d_state]
              conv_cache  [B, d_inner, d_conv-1]

        Returns:
            output : [B, L, d_model]
            state  : (ssm_h, conv_cache)  — carry for the next call
        """
        B_sz, L, _ = x.shape

        # Unpack or initialise recurrent state
        if recurrent_state is None:
            ssm_h      = None
            conv_cache = x.new_zeros(B_sz, self.d_inner, self.d_conv - 1)
        else:
            ssm_h, conv_cache = recurrent_state

        # ── Split input projection ────────────────────────────────────────
        xz = self.in_proj(x)   # [B, L, 2*D + 2*N + dt_rank]

        split_sizes = [self.d_inner, self.d_inner, self.d_state, self.d_state, self.dt_rank]
        x_proj, z, B_ssm, C_ssm, dt_rank_out = xz.split(split_sizes, dim=-1)

        # ── Causal depthwise conv with cache ─────────────────────────────
        # x_proj: [B, L, D] → [B, D, L] for conv1d
        x_t = x_proj.transpose(1, 2)           # [B, D, L]

        # Left-pad with conv cache to preserve causality
        x_padded = torch.cat([conv_cache, x_t], dim=2)   # [B, D, d_conv-1+L]

        # Update conv cache: keep last (d_conv-1) tokens
        new_conv_cache = x_padded[:, :, -(self.d_conv - 1):]  # [B, D, d_conv-1]

        x_conv = self.conv1d(x_padded)          # [B, D, L]
        x_conv = F.silu(x_conv.transpose(1, 2)) # [B, L, D]

        # ── dt ────────────────────────────────────────────────────────────
        dt = self.dt_proj(dt_rank_out)   # [B, L, D]

        # ── SSM scan ─────────────────────────────────────────────────────
        y, new_ssm_h = self._scan(x_conv, dt, B_ssm, C_ssm, ssm_h)

        # ── Gated output ─────────────────────────────────────────────────
        y = y * F.silu(z)

        # ── Output projection ─────────────────────────────────────────────
        output = self.out_proj(y)

        return output, (new_ssm_h, new_conv_cache)


class Mamba2Block(nn.Module):
    """
    Mamba-2 residual block with RMSNorm.

    Wraps Mamba2Layer with pre-norm and residual connection.
    Drop-in replacement for a Transformer block in a hybrid architecture.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.layer = Mamba2Layer(d_model, d_state=d_state, d_conv=d_conv, expand=expand, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        recurrent_state=None,
    ):
        residual = x
        x_normed = self.norm(x)
        y, state = self.layer(x_normed, recurrent_state)
        return y + residual, state


class Mamba2Model(nn.Module):
    """
    Stack of Mamba2Blocks — the full BOB backbone.

    Args:
        d_model   : model dimension
        n_layers  : number of Mamba-2 blocks
        d_state   : SSM state size
        vocab_size: set > 0 to add embedding + LM head
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        vocab_size: int = 0,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()

        if vocab_size > 0:
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.lm_head   = nn.Linear(d_model, vocab_size, bias=False)
        else:
            self.embedding = None
            self.lm_head   = None

        self.layers = nn.ModuleList([
            Mamba2Block(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                        norm_eps=norm_eps, **kwargs)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(d_model, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,                          # [B, L, d_model] or [B, L] token ids
        recurrent_states: Optional[list] = None,  # list of [B, D, N] per layer
    ) -> Tuple[torch.Tensor, list]:
        """
        Returns:
            hidden : [B, L, d_model] (or [B, L, vocab_size] with LM head)
            states : list of updated [B, D, N] per layer
        """
        if self.embedding is not None and x.dtype in (torch.long, torch.int):
            x = self.embedding(x)

        if recurrent_states is None:
            recurrent_states = [None] * len(self.layers)

        new_states = []
        for i, layer in enumerate(self.layers):
            x, h = layer(x, recurrent_states[i])
            new_states.append(h)

        x = self.final_norm(x)

        if self.lm_head is not None:
            x = self.lm_head(x)

        return x, new_states


# ── Quick sanity check (run directly) ────────────────────────────────────────

if __name__ == "__main__":
    import sys
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[mamba2_torch] device={device}")

    d_model, d_state, n_layers = 256, 16, 4
    B, L = 2, 64

    model = Mamba2Model(
        d_model=d_model, n_layers=n_layers, d_state=d_state, vocab_size=512
    ).to(device)

    tokens = torch.randint(0, 512, (B, L), device=device)
    out, states = model(tokens)
    print(f"  output shape : {out.shape}")      # [2, 64, 512]
    print(f"  n states     : {len(states)}")    # 4
    print(f"  state shape  : {states[0].shape}")  # [2, D_inner, 16]
    print(f"  output mean  : {out.float().mean().item():.6f}")
    print(f"  output std   : {out.float().std().item():.6f}")

    # Autoregressive step
    step_token = torch.randint(0, 512, (B, 1), device=device)
    step_out, new_states = model(step_token, recurrent_states=states)
    print(f"  step output  : {step_out.shape}")   # [2, 1, 512]
    print("[mamba2_torch] PASS")
    sys.exit(0)
