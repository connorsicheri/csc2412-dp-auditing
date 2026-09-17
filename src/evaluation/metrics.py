"""Tightness metrics: gap, ratio, and profile comparison.

Compares ε_account (upper bound from accountants) vs ε_audit (lower bound
from auditors) to quantify how tight the privacy guarantees are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TightnessMetrics:
    """Summary of tightness between accounting and auditing."""

    gap: float  # Δ = ε_account - ε_audit
    ratio: float  # R = ε_audit / ε_account
    eps_account: float
    eps_audit: float
    delta: float
    accountant_method: str
    auditor_method: str


def compute_gap(eps_account: float, eps_audit: float) -> float:
    """Compute additive gap Δ(δ) = ε_account(δ) - ε_audit(δ).

    A smaller gap means tighter auditing (the audit gets closer to the
    theoretical upper bound).
    """
    return eps_account - eps_audit


def compute_ratio(eps_account: float, eps_audit: float) -> float:
    """Compute ratio R(δ) = ε_audit / ε_account.

    R ∈ [0, 1], where R=1 means perfectly tight auditing.
    """
    if eps_account <= 0:
        return 0.0
    return eps_audit / eps_account


def tightness_summary(
    accounting_results: dict[str, dict],
    audit_results: dict[str, float],
    delta: float,
) -> list[TightnessMetrics]:
    """Compare all accountant/auditor pairs.

    Parameters
    ----------
    accounting_results : dict
        method_name → {"epsilon": float, ...}
    audit_results : dict
        auditor_name → ε_audit (float)
    delta : float
        The δ value used.

    Returns
    -------
    list of TightnessMetrics for each (accountant, auditor) pair.
    """
    metrics = []
    for acc_name, acc_result in accounting_results.items():
        eps_acc = acc_result["epsilon"]
        for aud_name, eps_aud in audit_results.items():
            metrics.append(
                TightnessMetrics(
                    gap=compute_gap(eps_acc, eps_aud),
                    ratio=compute_ratio(eps_acc, eps_aud),
                    eps_account=eps_acc,
                    eps_audit=eps_aud,
                    delta=delta,
                    accountant_method=acc_name,
                    auditor_method=aud_name,
                )
            )
    return metrics


def gap_over_delta(
    noise_multiplier: float,
    sampling_rate: float,
    num_steps: int,
    audit_eps: float,
    deltas: np.ndarray | None = None,
) -> dict:
    """Compute gap/ratio curves as a function of δ.

    Uses a fixed audit ε lower bound and sweeps δ through the accountants.

    Returns
    -------
    dict with "deltas", "gaps", "ratios" for each accountant method.
    """
    from ..accounting import compare_accountants

    if deltas is None:
        deltas = np.logspace(-7, -3, 50)

    results = {"deltas": deltas.tolist()}

    for delta in deltas:
        acc = compare_accountants(
            noise_multiplier=noise_multiplier,
            sampling_rate=sampling_rate,
            num_steps=num_steps,
            delta=float(delta),
        )
        for method, result in acc.items():
            if method not in results:
                results[method] = {"gaps": [], "ratios": [], "epsilons": []}
            eps_acc = result["epsilon"]
            results[method]["gaps"].append(compute_gap(eps_acc, audit_eps))
            results[method]["ratios"].append(compute_ratio(eps_acc, audit_eps))
            results[method]["epsilons"].append(eps_acc)

    return results
