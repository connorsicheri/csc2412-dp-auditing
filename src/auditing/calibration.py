"""Calibration module: convert audit outcomes (v, r, m) → ε lower bound.

Implements the **exact** calibration from Appendix D of:

    Steinke, Nasr, Jagielski (2023). Privacy Auditing with One (1) Training Run.
    NeurIPS 2023.  arXiv:2305.08846.

The p-value computation uses binomial tail + a tight δ-correction
(the α · δ · 2m term from Corollary 5.4 / Theorem 5.2 in the paper).
The ε lower bound is obtained via bisection on the p-value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class CalibrationResult:
    """Result of the ε lower bound calibration."""

    eps_lower_bound: float  # ε_audit: calibrated lower bound
    v: int  # number correct
    r: int  # number of non-abstained guesses
    m: int  # total number of canaries
    beta: float  # confidence parameter (bound holds w.p. ≥ 1-β)
    delta: float  # δ parameter
    p_hat: float  # observed success rate v/r
    p_eps: float  # p(ε_audit) = e^ε / (1+e^ε)
    method: str  # "paper_exact"


# ---- Paper's exact calibration (Appendix D) ----


def p_value_DP_audit(m: int, r: int, v: int, eps: float, delta: float) -> float:
    """Compute p-value for the null hypothesis that the algorithm is (ε,δ)-DP.

    Exact implementation from Steinke et al. (2023) Appendix D.

    Parameters
    ----------
    m : int  – total number of canary examples
    r : int  – number of non-abstained guesses
    v : int  – number of correct guesses
    eps : float  – ε of the null hypothesis
    delta : float  – δ of the DP guarantee

    Returns
    -------
    float – p-value in [0, 1].  Small p → reject the null → ε is too small.
    """
    assert 0 <= v <= r <= m
    assert eps >= 0
    assert 0 <= delta <= 1

    q = 1.0 / (1.0 + math.exp(-eps))  # accuracy of ε-DP randomized response
    beta = stats.binom.sf(v - 1, r, q)  # = P[Binomial(r, q) >= v]

    # Tight δ-correction (α term from Corollary 5.4)
    alpha = 0.0
    s = 0.0  # cumulative P[v > Binomial(r,q) >= v-i]
    for i in range(1, v + 1):
        s += stats.binom.pmf(v - i, r, q)
        if s > i * alpha:
            alpha = s / i

    p = beta + alpha * delta * 2 * m
    return min(p, 1.0)


def get_eps_audit(m: int, r: int, v: int, delta: float, p: float) -> float:
    """Find the ε lower bound via bisection on the p-value.

    Exact implementation from Steinke et al. (2023) Appendix D.

    Parameters
    ----------
    m : int  – total canary count
    r : int  – non-abstained guesses
    v : int  – correct guesses
    delta : float  – δ of the DP guarantee
    p : float  – significance level (e.g. 0.05 for 95% confidence)

    Returns
    -------
    float – ε_audit lower bound.  Algorithm is not (ε_audit, δ)-DP.
    """
    assert 0 <= v <= r <= m
    assert 0 <= delta <= 1
    assert 0 < p < 1

    # Trivial case: not even better than random
    if v <= r // 2:
        return 0.0

    eps_min = 0.0  # maintain p_value(eps_min) < p
    eps_max = 1.0  # maintain p_value(eps_max) >= p

    # Expand upper bracket
    while p_value_DP_audit(m, r, v, eps_max, delta) < p:
        eps_max += 1.0

    # 30 iterations of bisection → precision ~2^{-30} ≈ 1e-9
    for _ in range(30):
        eps = (eps_min + eps_max) / 2.0
        if p_value_DP_audit(m, r, v, eps, delta) < p:
            eps_min = eps
        else:
            eps_max = eps

    return eps_min


# ---- Wrapper that returns a CalibrationResult (used by the rest of the code) ----


def _p_of_eps(eps: float) -> float:
    """Compute p(ε) = e^ε / (1 + e^ε) = sigmoid(ε)."""
    if eps > 500:
        return 1.0
    return math.exp(eps) / (1.0 + math.exp(eps))


def compute_eps_lower_bound(
    v: int,
    r: int,
    beta: float = 0.05,
    delta: float = 1e-5,
    m: int | None = None,
    **_kwargs,
) -> CalibrationResult:
    """Compute ε_audit using the paper's exact calibration.

    Parameters
    ----------
    v : int – number of correct guesses
    r : int – number of non-abstained guesses
    beta : float – significance level (1−β confidence)
    delta : float – δ in (ε,δ)-DP
    m : int, optional – total canary count (defaults to r if not given)

    Returns
    -------
    CalibrationResult
    """
    if m is None:
        m = r  # conservative default: no abstentions

    if v <= 0 or r <= 0 or v <= r // 2:
        return CalibrationResult(
            eps_lower_bound=0.0,
            v=v,
            r=r,
            m=m,
            beta=beta,
            delta=delta,
            p_hat=(v / r if r > 0 else 0.0),
            p_eps=0.5,
            method="paper_exact",
        )

    eps_audit = get_eps_audit(m, r, v, delta, beta)

    return CalibrationResult(
        eps_lower_bound=eps_audit,
        v=v,
        r=r,
        m=m,
        beta=beta,
        delta=delta,
        p_hat=v / r,
        p_eps=_p_of_eps(eps_audit),
        method="paper_exact",
    )


# --- Sensitivity analysis utilities ---


def sensitivity_over_beta(
    v: int,
    r: int,
    delta: float = 1e-5,
    betas: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    """Compute ε_audit for a grid of β values (confidence sensitivity)."""
    if betas is None:
        betas = np.logspace(-4, -0.3, 50)
    return [
        (float(b), compute_eps_lower_bound(v, r, beta=float(b), delta=delta).eps_lower_bound)
        for b in betas
    ]


def sensitivity_over_delta(
    v: int,
    r: int,
    beta: float = 0.05,
    deltas: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    """Compute ε_audit for a grid of δ values."""
    if deltas is None:
        deltas = np.logspace(-7, -3, 50)
    return [
        (float(d), compute_eps_lower_bound(v, r, beta=beta, delta=float(d)).eps_lower_bound)
        for d in deltas
    ]


def sensitivity_over_r(
    v_fraction: float,
    beta: float = 0.05,
    delta: float = 1e-5,
    r_values: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """Compute ε_audit for a grid of r values (number of guesses), fixing v/r."""
    if r_values is None:
        r_values = np.array([10, 50, 100, 200, 500, 1000, 2000, 5000])
    results = []
    for r in r_values:
        r = int(r)
        v = int(v_fraction * r)
        if v <= r // 2:
            results.append((r, 0.0))
        else:
            res = compute_eps_lower_bound(v, r, beta=beta, delta=delta)
            results.append((r, res.eps_lower_bound))
    return results
