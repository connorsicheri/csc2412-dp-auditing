"""Rényi Differential Privacy (RDP) accountant for subsampled Gaussian mechanism.

References:
    Mironov (2017). Rényi Differential Privacy. arXiv:1702.07476.
    Mironov, Talwar, Zhang (2019). Rényi DP of the Sampled Gaussian Mechanism. arXiv:1702.07476v3.
"""

from __future__ import annotations

import math
from typing import Sequence


def _compute_rdp_order(alpha: float, noise_multiplier: float, sampling_rate: float) -> float:
    """Compute RDP at a single order α for subsampled Gaussian mechanism.

    Uses the analytic formula from Mironov et al. (2019) Proposition 3:
    For the subsampled Gaussian mechanism with sampling rate q and
    noise multiplier σ, the RDP at order α is bounded.

    For the non-subsampled Gaussian mechanism:
        ε(α) = α / (2σ²)

    For the subsampled version, we use the tight bound via
    log-sum-exp over binomial expansion terms.
    """
    if noise_multiplier == 0:
        return float("inf")

    if sampling_rate == 0:
        return 0.0

    if sampling_rate == 1.0:
        # No subsampling: standard Gaussian mechanism RDP
        return alpha / (2.0 * noise_multiplier**2)

    if alpha <= 1:
        return 0.0

    # Integer orders: use exact binomial expansion (Mironov et al. 2019)
    if isinstance(alpha, int) or alpha == int(alpha):
        alpha = int(alpha)
        # RDP of subsampled mechanism via Proposition 3
        log_terms = []
        for j in range(alpha + 1):
            log_binom = math.lgamma(alpha + 1) - math.lgamma(j + 1) - math.lgamma(alpha - j + 1)
            log_q_term = j * math.log(sampling_rate) + (alpha - j) * math.log(1 - sampling_rate)
            # RDP of Gaussian at order j: j(j-1)/(2σ²) but we use the standard formula
            if j <= 1:
                rdp_j = 0.0
            else:
                rdp_j = j * (j - 1) / (2.0 * noise_multiplier**2)
            log_terms.append(log_binom + log_q_term + rdp_j)

        # Log-sum-exp
        max_log = max(log_terms)
        result = max_log + math.log(sum(math.exp(t - max_log) for t in log_terms))
        return result / (alpha - 1)
    else:
        # Non-integer orders: use upper bound via closest integer orders
        alpha_floor = int(math.floor(alpha))
        alpha_ceil = int(math.ceil(alpha))
        rdp_floor = _compute_rdp_order(alpha_floor, noise_multiplier, sampling_rate)
        rdp_ceil = _compute_rdp_order(alpha_ceil, noise_multiplier, sampling_rate)
        # Linear interpolation (valid upper bound for RDP)
        t = alpha - alpha_floor
        return (1 - t) * rdp_floor + t * rdp_ceil


def _rdp_to_eps_delta(rdp_alpha: float, alpha: float, delta: float) -> float:
    """Convert RDP guarantee to (ε, δ)-DP.

    ε = rdp_α + log(1/δ) / (α - 1)   [Mironov 2017, Proposition 3]

    More precisely (with the improved bound from Balle et al. 2020):
    ε = rdp_α + log(1 - 1/α) - (log(δ) + log(α-1)) / (α - 1)
    """
    if alpha <= 1:
        return float("inf")

    # Standard conversion (Mironov 2017)
    eps_basic = rdp_alpha + math.log(1.0 / delta) / (alpha - 1)

    # Improved conversion (Balle et al. 2020, Proposition 12)
    if alpha > 1:
        eps_improved = (
            rdp_alpha
            + math.log(1 - 1.0 / alpha)
            - (math.log(delta) + math.log(alpha - 1)) / (alpha - 1)
        )
        return min(eps_basic, eps_improved)

    return eps_basic


def rdp_accountant(
    noise_multiplier: float,
    sampling_rate: float,
    num_steps: int,
    delta: float,
    orders: Sequence[float] | None = None,
) -> dict:
    """Compute ε upper bound via RDP accounting for DP-SGD.

    Parameters
    ----------
    noise_multiplier : float
        σ — ratio of noise std to clipping norm.
    sampling_rate : float
        q = batch_size / dataset_size (Poisson subsampling rate).
    num_steps : int
        T — total number of training steps (epochs × steps_per_epoch).
    delta : float
        Target δ.
    orders : sequence of float, optional
        RDP orders α to evaluate. Defaults to a standard grid.

    Returns
    -------
    dict with keys:
        "epsilon": best ε(δ) over all orders
        "best_order": the α achieving the tightest bound
        "rdp_curve": list of (α, ε_rdp(α)) pairs
        "eps_curve": list of (α, ε(δ)) pairs
        "method": "rdp"
    """
    if orders is None:
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512, 1024]

    rdp_curve = []
    eps_curve = []

    for alpha in orders:
        # Composition: RDP composes additively across steps
        rdp_single = _compute_rdp_order(alpha, noise_multiplier, sampling_rate)
        rdp_total = num_steps * rdp_single
        eps = _rdp_to_eps_delta(rdp_total, alpha, delta)
        rdp_curve.append((alpha, rdp_total))
        eps_curve.append((alpha, eps))

    # Best ε over all orders
    best_idx = min(range(len(eps_curve)), key=lambda i: eps_curve[i][1])
    best_eps = eps_curve[best_idx][1]
    best_order = orders[best_idx]

    return {
        "epsilon": best_eps,
        "best_order": best_order,
        "rdp_curve": rdp_curve,
        "eps_curve": eps_curve,
        "method": "rdp",
    }
