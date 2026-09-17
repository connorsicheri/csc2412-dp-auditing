"""Compare all accountants on a matched DP-SGD workload.

Produces a unified summary of ε upper bounds from RDP, moments, zCDP, and PLD.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from .moments import moments_accountant
from .pld import pld_accountant
from .rdp import rdp_accountant
from .zcdp import zcdp_accountant

logger = logging.getLogger(__name__)


ACCOUNTANT_REGISTRY = {
    "rdp": rdp_accountant,
    "moments": moments_accountant,
    "zcdp": zcdp_accountant,
    "pld": pld_accountant,
}


def compare_accountants(
    noise_multiplier: float,
    sampling_rate: float,
    num_steps: int,
    delta: float,
    methods: Sequence[str] | None = None,
    rdp_orders: Sequence[float] | None = None,
) -> dict[str, dict]:
    """Run all specified accountants and return their ε upper bounds.

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
    methods : sequence of str, optional
        Which accountants to run. Defaults to all.
    rdp_orders : sequence of float, optional
        RDP orders (passed to rdp_accountant only).

    Returns
    -------
    dict mapping method name → result dict (each has "epsilon" key).
    """
    if methods is None:
        methods = list(ACCOUNTANT_REGISTRY.keys())

    results = {}
    for method in methods:
        if method not in ACCOUNTANT_REGISTRY:
            logger.warning(f"Unknown accountant method: {method}")
            continue

        fn = ACCOUNTANT_REGISTRY[method]
        kwargs: dict[str, Any] = {
            "noise_multiplier": noise_multiplier,
            "sampling_rate": sampling_rate,
            "num_steps": num_steps,
            "delta": delta,
        }
        if method == "rdp" and rdp_orders is not None:
            kwargs["orders"] = rdp_orders

        try:
            results[method] = fn(**kwargs)
            logger.info(f"  {method}: ε = {results[method]['epsilon']:.4f}")
        except Exception as e:
            logger.error(f"  {method}: failed — {e}")
            results[method] = {"epsilon": float("inf"), "method": method, "error": str(e)}

    return results
