"""Concentrated Differential Privacy (zCDP) accountant.

References:
    Bun & Steinke (2016). Concentrated Differential Privacy:
    Simplifications, Extensions, and Lower Bounds. arXiv:1605.02065.
"""

from __future__ import annotations

import math


def _zcdp_single_step(noise_multiplier: float, sampling_rate: float) -> float:
    """Compute ρ for a single step of subsampled Gaussian mechanism.

    For the Gaussian mechanism with sensitivity Δ=1 and noise N(0, σ²):
        ρ = 1 / (2σ²)    [Bun & Steinke 2016, Proposition 1.6]

    For subsampled mechanism with rate q, a simple bound is:
        ρ_sub ≤ q² · ρ    (privacy amplification by subsampling for zCDP)

    A tighter bound exists (Balle et al. 2018) but this suffices for comparison.
    """
    if noise_multiplier == 0:
        return float("inf")
    rho = 1.0 / (2.0 * noise_multiplier**2)
    # Subsampling amplification (simple bound)
    return sampling_rate**2 * rho


def _zcdp_to_eps_delta(rho: float, delta: float) -> float:
    """Convert ρ-zCDP to (ε, δ)-DP.

    ε = ρ + 2√(ρ · ln(1/δ))    [Bun & Steinke 2016, Proposition 1.3]
    """
    if rho <= 0:
        return 0.0
    if delta <= 0:
        return float("inf")
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def zcdp_accountant(
    noise_multiplier: float,
    sampling_rate: float,
    num_steps: int,
    delta: float,
) -> dict:
    """Compute ε upper bound via zCDP accounting.

    Parameters
    ----------
    noise_multiplier : float
        σ (noise std / clipping norm).
    sampling_rate : float
        q = batch_size / dataset_size.
    num_steps : int
        Total training steps T.
    delta : float
        Target δ.

    Returns
    -------
    dict with keys "epsilon", "rho", "method"
    """
    # zCDP composes additively
    rho_total = num_steps * _zcdp_single_step(noise_multiplier, sampling_rate)
    eps = _zcdp_to_eps_delta(rho_total, delta)

    return {
        "epsilon": eps,
        "rho": rho_total,
        "method": "zcdp",
    }
