"""Moments accountant for DP-SGD.

This implements the moments accountant from Abadi et al. (2016) "Deep Learning
with Differential Privacy". The moments accountant tracks the log-moment
generating function of the privacy loss random variable.

References:
    Abadi et al. (2016). Deep Learning with Differential Privacy. CCS 2016.
"""

from __future__ import annotations

import math


def _compute_log_moment(alpha: int, noise_multiplier: float, sampling_rate: float) -> float:
    """Compute the λ-th log moment of the privacy loss for subsampled Gaussian.

    Uses the bound from Abadi et al. (2016) Theorem 1 / Appendix A.
    For integer order λ ≥ 2, the log moment is bounded by:
        α_M(λ) ≤ log(max term in binomial expansion)

    This is equivalent to the RDP computation at integer orders, but
    framed as moments for pedagogical clarity and to match the original paper.
    """
    if noise_multiplier == 0:
        return float("inf")
    if sampling_rate == 0:
        return 0.0

    # For integer λ, compute via the same analytic bound as RDP
    # The λ-th moment bound is: (1/λ) * log E[exp(λ · privacy_loss)]
    # Which for Gaussian mechanism = λ/(2σ²)
    if sampling_rate == 1.0:
        return alpha / (2.0 * noise_multiplier**2)

    # Subsampled case: binomial expansion (same math as RDP for integers)
    log_terms = []
    for j in range(alpha + 1):
        log_binom = math.lgamma(alpha + 1) - math.lgamma(j + 1) - math.lgamma(alpha - j + 1)
        log_q = j * math.log(sampling_rate) + (alpha - j) * math.log(1 - sampling_rate)
        rdp_j = 0.0 if j <= 1 else j * (j - 1) / (2.0 * noise_multiplier**2)
        log_terms.append(log_binom + log_q + rdp_j)

    max_log = max(log_terms)
    return max_log + math.log(sum(math.exp(t - max_log) for t in log_terms))


def moments_accountant(
    noise_multiplier: float,
    sampling_rate: float,
    num_steps: int,
    delta: float,
    max_order: int = 64,
) -> dict:
    """Compute ε upper bound via the moments accountant (Abadi et al. 2016).

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
    max_order : int
        Maximum moment order λ to try.

    Returns
    -------
    dict with keys "epsilon", "best_order", "method"
    """
    best_eps = float("inf")
    best_order = 0

    eps_curve = []
    for lam in range(2, max_order + 1):
        # Moments compose additively across steps
        log_moment = num_steps * _compute_log_moment(lam, noise_multiplier, sampling_rate)
        # Convert to (ε, δ): ε = (log_moment - log(δ)) / λ
        eps = (log_moment - math.log(delta)) / lam
        eps_curve.append((lam, eps))

        if eps < best_eps:
            best_eps = eps
            best_order = lam

    return {
        "epsilon": best_eps,
        "best_order": best_order,
        "eps_curve": eps_curve,
        "method": "moments",
    }
