#!/usr/bin/env python3
"""Compute DP-SGD training parameters for experiment planning.

Given a target epsilon, dataset size, and noise multiplier, finds how many
training steps each accounting method allows, and what batch size matches
the De et al. (2022) subsampling ratio.

Usage:
    python experiments/compute_params.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.accounting.compare import compare_accountants

# =============================================================================
# De et al. (2022) reference parameters for CIFAR-10 (N=50,000)
# =============================================================================
DE_ET_AL = {
    "N": 50_000,
    "B": 4096,
    "C": 1.0,
    "delta": 1e-5,
    # Per-epsilon configs from Table 13
    "configs": {
        1.0: {"sigma": 10.0, "lr": 2.0, "T": 875},
        2.0: {"sigma": 6.0, "lr": 2.0, "T": 1125},
        4.0: {"sigma": 4.0, "lr": 2.0, "T": 1687},
        8.0: {"sigma": 3.0, "lr": 4.0, "T": 2468},
    },
}

DE_Q = DE_ET_AL["B"] / DE_ET_AL["N"]  # 4096/50000 ≈ 0.08192


def find_max_steps(
    noise_multiplier: float,
    sampling_rate: float,
    target_eps: float,
    delta: float,
    method: str = "pld",
    max_T: int = 50_000,
) -> int | None:
    """Binary search for the max number of steps T such that ε(T) ≤ target_eps."""
    from src.accounting.compare import ACCOUNTANT_REGISTRY

    fn = ACCOUNTANT_REGISTRY[method]

    # Check if even 1 step already exceeds target
    result = fn(
        noise_multiplier=noise_multiplier, sampling_rate=sampling_rate, num_steps=1, delta=delta
    )
    if result["epsilon"] > target_eps:
        return 0

    lo, hi = 1, max_T
    # Check if max_T is still under budget
    result = fn(
        noise_multiplier=noise_multiplier, sampling_rate=sampling_rate, num_steps=hi, delta=delta
    )
    if result["epsilon"] <= target_eps:
        return hi  # can do at least max_T steps

    while lo < hi:
        mid = (lo + hi + 1) // 2
        result = fn(
            noise_multiplier=noise_multiplier,
            sampling_rate=sampling_rate,
            num_steps=mid,
            delta=delta,
        )
        if result["epsilon"] <= target_eps:
            lo = mid
        else:
            hi = mid - 1

    return lo


def main():
    delta = DE_ET_AL["delta"]
    methods = ["rdp", "zcdp", "pld"]

    # =================================================================
    # PART 1: Verify De et al. parameters reproduce their claimed ε
    # =================================================================
    print("=" * 80)
    print("PART 1: Verify De et al. Table 13 (N=50,000, B=4096, q=%.4f)" % DE_Q)
    print("=" * 80)

    for target_eps, cfg in sorted(DE_ET_AL["configs"].items()):
        sigma, T = cfg["sigma"], cfg["T"]
        print(f"\n--- Target ε={target_eps}, σ={sigma}, T={T} ---")
        results = compare_accountants(
            noise_multiplier=sigma,
            sampling_rate=DE_Q,
            num_steps=T,
            delta=delta,
            methods=methods,
        )
        for m, r in results.items():
            print(f"  {m:8s}: ε = {r['epsilon']:.4f}")

    # =================================================================
    # PART 2: For our small datasets, match the subsampling ratio
    # =================================================================
    print("\n" + "=" * 80)
    print("PART 2: Matched subsampling ratio for small N")
    print("  De et al. q = B/N = 4096/50000 = %.5f" % DE_Q)
    print("=" * 80)

    for N in [500, 1000, 2000, 5000]:
        B_matched = max(1, round(DE_Q * N))
        q_actual = B_matched / N
        print(f"\n{'='*60}")
        print(f"  N = {N},  B = {B_matched} (q = {q_actual:.5f})")
        print(f"{'='*60}")

        for target_eps, cfg in sorted(DE_ET_AL["configs"].items()):
            sigma = cfg["sigma"]
            print(f"\n  Target ε ≤ {target_eps},  σ = {sigma}:")
            for method in methods:
                T_max = find_max_steps(
                    noise_multiplier=sigma,
                    sampling_rate=q_actual,
                    target_eps=target_eps,
                    delta=delta,
                    method=method,
                )
                epochs = T_max * B_matched / N if T_max else 0
                print(f"    {method:8s}: T_max = {T_max:>6d} steps  ({epochs:>8.1f} epochs)")

    # =================================================================
    # PART 3: What if we use q=1.0 (batch = full dataset)?
    #         How much noise σ do we need for each ε target?
    # =================================================================
    print("\n" + "=" * 80)
    print("PART 3: q=1.0 (full-batch), find σ needed for target ε at various T")
    print("=" * 80)

    def find_min_sigma(
        target_eps,
        sampling_rate,
        num_steps,
        delta,
        method="pld",
        sigma_lo=0.1,
        sigma_hi=200.0,
        tol=0.01,
    ):
        """Binary search for minimum σ such that ε(σ, q, T) ≤ target_eps."""
        from src.accounting.compare import ACCOUNTANT_REGISTRY

        fn = ACCOUNTANT_REGISTRY[method]

        # Check if even sigma_hi is not enough
        result = fn(
            noise_multiplier=sigma_hi, sampling_rate=sampling_rate, num_steps=num_steps, delta=delta
        )
        if result["epsilon"] > target_eps:
            return None

        while sigma_hi - sigma_lo > tol:
            mid = (sigma_lo + sigma_hi) / 2
            result = fn(
                noise_multiplier=mid, sampling_rate=sampling_rate, num_steps=num_steps, delta=delta
            )
            if result["epsilon"] <= target_eps:
                sigma_hi = mid
            else:
                sigma_lo = mid

        return sigma_hi

    for N in [500, 1000]:
        print(f"\n--- N = {N}, q = 1.0 (full batch) ---")
        for target_eps in [1, 2, 4, 8]:
            print(f"\n  Target ε ≤ {target_eps}:")
            for T in [50, 100, 500, 1000, 2500]:
                sigma_needed = find_min_sigma(
                    target_eps=target_eps,
                    sampling_rate=1.0,
                    num_steps=T,
                    delta=delta,
                    method="pld",
                )
                if sigma_needed:
                    print(f"    T={T:>5d}: σ ≥ {sigma_needed:>7.2f} (PLD)")
                else:
                    print(f"    T={T:>5d}: σ > 200 needed (infeasible)")

    # =================================================================
    # PART 4: Summary table — recommended configs
    # =================================================================
    print("\n" + "=" * 80)
    print("PART 4: Recommended experiment configs (N=1000, matched q)")
    print("=" * 80)

    N = 1000
    B_matched = max(1, round(DE_Q * N))
    q = B_matched / N
    print(f"  N={N}, B={B_matched}, q={q:.5f}, δ={delta}")
    print(f"  (De et al.: N=50000, B=4096, q={DE_Q:.5f})")
    print()

    header = f"  {'ε_target':>8s} | {'σ':>5s} | {'Method':>8s} | {'T_max':>7s} | {'Epochs':>8s} | {'ε_actual':>8s}"
    print(header)
    print("  " + "-" * len(header))

    from src.accounting.compare import ACCOUNTANT_REGISTRY

    for target_eps in [1, 2, 4, 8]:
        cfg = DE_ET_AL["configs"][target_eps]
        sigma = cfg["sigma"]
        for method in methods:
            T = find_max_steps(sigma, q, target_eps, delta, method)
            epochs = T * B_matched / N if T else 0
            fn = ACCOUNTANT_REGISTRY[method]
            actual = (
                fn(noise_multiplier=sigma, sampling_rate=q, num_steps=T, delta=delta)
                if T > 0
                else {"epsilon": 0}
            )
            print(
                f"  {target_eps:>8.1f} | {sigma:>5.1f} | {method:>8s} | {T:>7d} | {epochs:>8.1f} | {actual['epsilon']:>8.4f}"
            )
        print()


if __name__ == "__main__":
    main()
