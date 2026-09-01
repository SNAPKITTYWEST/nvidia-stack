#!/usr/bin/env python3
"""
lw_lgm.py — Latent-to-Waveform Linear Geometric Map (Reference Implementation)

Maps a latent vector z ∈ ℝ^d to an analog waveform x(t) ∈ C^0(ℝ)
using a linear expansion in a fixed dictionary of geometrically
transformed atoms (affine group acting on a mother waveform).

Usage:
    python lw_lgm.py

Output:
    First 10 samples of the generated waveform.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def mother_gaussian(t: np.ndarray, sigma0: float) -> np.ndarray:
    """Normalized Gaussian mother waveform: φ(t) = (1/(2πσ₀²)^{1/4}) · exp(-t²/(2σ₀²))"""
    norm = 1.0 / (2.0 * np.pi * sigma0**2) ** 0.25
    return norm * np.exp(-0.5 * t**2 / (sigma0**2))


def build_dictionary(
    sigma0: float,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    m: int,
    t_start: float,
    t_end: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the dictionary matrix Ψ ∈ ℝ^{N×m} from an affine group action.

    Returns:
        psi: Dictionary matrix of shape (N, m)
        t: Time axis of length N
    """
    t = np.arange(t_start, t_end, dt)
    n = len(t)
    psi = np.zeros((n, m))

    log_a_min = np.log(a_min)
    log_a_max = np.log(a_max)
    log_a_step = (log_a_max - log_a_min) / (m // 2)

    for i in range(m):
        # Logarithmic dilation grid
        if i < m // 2:
            a = np.exp(log_a_min + i * log_a_step)
        else:
            a = -np.exp(log_a_min + (m - 1 - i) * log_a_step)

        # Uniform translation
        b = b_min + i * (b_max - b_min) / (m - 1)

        # Precompute 1/√|a|
        scale = 1.0 / np.sqrt(np.abs(a))

        # Fill column i
        arg = (t - b) / a
        phi_val = mother_gaussian(arg, sigma0)
        psi[:, i] = scale * phi_val

    return psi, t


def latent_to_waveform(
    z: np.ndarray,
    W: np.ndarray,
    psi: np.ndarray,
) -> np.ndarray:
    """
    Map a latent vector z to waveform samples x = Ψ(Wz).

    Args:
        z: Latent vector of length d
        W: Fixed matrix of shape (m, d), or identity if d == m
        psi: Dictionary matrix of shape (N, m)

    Returns:
        x: Output waveform samples of length N
    """
    c = W @ z if W.shape[1] == z.shape[0] else z
    return psi @ c


def test_linearity():
    """Verify L(αz₁ + βz₂) = αL(z₁) + βL(z₂)"""
    sigma0 = 1.0
    a_min, a_max = 0.5, 2.0
    b_min, b_max = -5.0, 5.0
    m = 32
    t_start, t_end, dt = -10.0, 10.0, 0.1

    psi, _ = build_dictionary(sigma0, a_min, a_max, b_min, b_max, m, t_start, t_end, dt)
    W = np.eye(m)

    z1 = np.random.uniform(-1.0, 1.0, m)
    z2 = np.random.uniform(-1.0, 1.0, m)

    alpha, beta = 2.5, -1.3

    lhs = latent_to_waveform(alpha * z1 + beta * z2, W, psi)
    rhs = alpha * latent_to_waveform(z1, W, psi) + beta * latent_to_waveform(z2, W, psi)

    diff = np.sum(np.abs(lhs - rhs))
    assert diff < 1e-10, f"Linearity test failed: diff = {diff}"
    print(f"✓ Linearity test passed (diff = {diff:.2e})")


def test_energy_bounds():
    """Verify energy ratio is bounded"""
    sigma0 = 1.0
    a_min, a_max = 0.5, 2.0
    b_min, b_max = -5.0, 5.0
    m = 64
    t_start, t_end, dt = -10.0, 10.0, 0.01

    psi, _ = build_dictionary(sigma0, a_min, a_max, b_min, b_max, m, t_start, t_end, dt)
    W = np.eye(m)

    z = np.random.uniform(-1.0, 1.0, m)
    x = latent_to_waveform(z, W, psi)

    energy_x = np.sum(x**2) * dt
    energy_z = np.sum(z**2)

    ratio = energy_x / energy_z
    assert 0 < ratio < np.inf, f"Energy ratio invalid: {ratio}"
    print(f"✓ Energy bounds test passed (ratio = {ratio:.4f})")


if __name__ == "__main__":
    print("LW-LGM: Latent-to-Waveform Linear Geometric Map (Python Reference)")
    print("=" * 70)

    # Parameters
    sigma0 = 1.0
    a_min, a_max = 0.5, 2.0
    b_min, b_max = -5.0, 5.0
    m = 64
    t_start, t_end, dt = -10.0, 10.0, 0.01

    print(f"Parameters:")
    print(f"  σ₀ = {sigma0}")
    print(f"  a ∈ [{a_min}, {a_max}]")
    print(f"  b ∈ [{b_min}, {b_max}]")
    print(f"  m = {m} atoms")
    print(f"  t ∈ [{t_start}, {t_end}] dt={dt}")
    print()

    # Build dictionary
    psi, t = build_dictionary(sigma0, a_min, a_max, b_min, b_max, m, t_start, t_end, dt)
    print(f"Dictionary Ψ: {psi.shape}")

    # Identity mapping
    W = np.eye(m)

    # Random latent vector
    z = np.random.uniform(-1.0, 1.0, m)
    print(f"Latent z: {z.shape}")

    # Generate waveform
    x = latent_to_waveform(z, W, psi)
    print(f"Output x: {x.shape}")
    print(f"x[0:10] = {x[:10]}")

    energy = np.sum(x**2) * dt
    print(f"Signal energy: {energy:.6f}")
    print()

    # Run tests
    test_linearity()
    test_energy_bounds()
    print()
    print("All tests passed!")
