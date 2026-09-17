"""Privacy Loss Distribution (PLD) accountant via FFT-based numerical composition.

Uses the approach from Koskela et al. (2021) for tight numerical accounting
of the subsampled Gaussian mechanism.

This module wraps Google's `dp_accounting` library when available, and provides
a simplified fallback implementation.

References:
    Koskela et al. (2021). Tight DP for Discrete-Valued Mechanisms and for the
    Subsampled Gaussian Mechanism Using FFT. AISTATS 2021.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def pld_accountant(
    noise_multiplier: float,
    sampling_rate: float,
    num_steps: int,
    delta: float,
) -> dict:
    """Compute ε upper bound via PLD/FFT-based numerical accounting.

    Attempts to use Google's `dp_accounting` library for tight bounds.
    Falls back to a discretized numerical convolution if unavailable.

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
    dict with keys "epsilon", "method"
    """
    try:
        return _pld_via_dp_accounting(noise_multiplier, sampling_rate, num_steps, delta)
    except ImportError:
        logger.warning(
            "dp_accounting not available; falling back to simplified PLD. "
            "Install with: pip install dp-accounting"
        )
        return _pld_fallback(noise_multiplier, sampling_rate, num_steps, delta)


def _pld_via_dp_accounting(
    noise_multiplier: float,
    sampling_rate: float,
    num_steps: int,
    delta: float,
) -> dict:
    """Use Google's dp_accounting library for PLD-based accounting."""
    from dp_accounting.pld import privacy_loss_distribution as pld_lib

    # Create PLD for a single step of subsampled Gaussian
    pld_single = pld_lib.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sensitivity=1.0,
        sampling_prob=sampling_rate,
        use_connect_dots=True,
    )

    # Compose T times (self-composition via FFT)
    pld_composed = pld_single.self_compose(num_steps)

    # Extract epsilon at target delta
    eps = pld_composed.get_epsilon_for_delta(delta)

    return {
        "epsilon": eps,
        "method": "pld",
    }


def _pld_fallback(
    noise_multiplier: float,
    sampling_rate: float,
    num_steps: int,
    delta: float,
) -> dict:
    """Simplified PLD via discretized numerical convolution.

    This is a basic implementation for when dp_accounting is not available.
    Less tight than the full PLD approach but provides a numerical bound.
    """
    # For the Gaussian mechanism, the privacy loss random variable
    # L = log(p(x|in)/p(x|out)) is Gaussian with:
    #   mean = 1/(2σ²), variance = 1/σ²
    # Under subsampling with rate q, we use the mixture PLD.

    sigma = noise_multiplier
    if sigma == 0:
        return {"epsilon": float("inf"), "method": "pld_fallback"}

    # Discretization parameters
    num_points = 2**18  # FFT grid size
    L_max = 20.0  # range of privacy loss values
    dx = 2 * L_max / num_points
    grid = np.linspace(-L_max, L_max, num_points)

    # PLD of single Gaussian mechanism step (sensitivity 1)
    mean_loss = 1.0 / (2.0 * sigma**2)
    std_loss = 1.0 / sigma

    # PDF of privacy loss (Gaussian)
    pld = np.exp(-0.5 * ((grid - mean_loss) / std_loss) ** 2) / (std_loss * np.sqrt(2 * np.pi))

    # Apply subsampling: mixture of delta(0) and mechanism PLD
    # P_sub = (1-q) * delta(0) + q * PLD
    zero_idx = num_points // 2  # index closest to 0
    pld_sub = np.zeros_like(pld)
    pld_sub += (1 - sampling_rate) * 0  # delta at 0 (handled via FFT)
    pld_sub += sampling_rate * pld
    # Add the (1-q) mass at L=0
    pld_sub[zero_idx] += (1 - sampling_rate) / dx

    pld_sub = pld_sub / (pld_sub.sum() * dx)  # normalize

    # Compose via FFT: convolve T times
    pld_fft = np.fft.fft(pld_sub * dx)
    composed_fft = pld_fft**num_steps
    composed = np.real(np.fft.ifft(composed_fft)) / dx

    # Compute epsilon: find smallest ε such that
    # ∫_{L > ε} (1 - e^{ε-L}) * composed(L) dL ≤ δ
    # Simplified: use the hockey-stick divergence
    best_eps = float("inf")
    for eps_candidate in np.linspace(0, 2 * num_steps * mean_loss, 1000):
        # δ(ε) = ∫ max(0, 1 - e^{ε-L}) * p(L) dL
        integrand = np.maximum(0, 1 - np.exp(eps_candidate - grid)) * composed
        delta_at_eps = np.sum(integrand) * dx
        if delta_at_eps <= delta:
            best_eps = eps_candidate
            break

    return {
        "epsilon": best_eps,
        "method": "pld_fallback",
    }
