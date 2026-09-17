"""Run accounting + evaluation: compare accountants vs auditors.

Usage:
    python experiments/run_evaluation.py --config configs/mnist_logreg.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.accounting import compare_accountants
from src.auditing.calibration import (
    sensitivity_over_beta,
    sensitivity_over_delta,
    sensitivity_over_r,
)
from src.evaluation.metrics import gap_over_delta, tightness_summary
from src.evaluation.plots import (
    plot_audit_vs_accounting,
    plot_calibration_sensitivity,
    plot_canary_scores,
    plot_epsilon_over_epochs,
    plot_gap_ratio_curves,
)
from src.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluation: Accounting vs Auditing")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir or cfg["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load training metadata ---
    meta_path = output_dir / "training_meta.json"
    if not meta_path.exists():
        logger.error(f"Training metadata not found at {meta_path}. Run training first.")
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)

    # --- Compute accounting upper bounds ---
    logger.info("Computing accounting upper bounds...")
    train_cfg = cfg["training"]
    # Estimate sampling rate and num_steps
    # (In practice, these come from the actual training run)
    n_train = 54000  # approximate (MNIST train size - val split)
    sampling_rate = train_cfg["batch_size"] / n_train
    num_steps = train_cfg["epochs"] * (n_train // train_cfg["batch_size"])
    delta = train_cfg["delta"]

    acc_methods = cfg.get("accounting", {}).get("methods", ["rdp", "moments", "zcdp", "pld"])
    rdp_orders = cfg.get("accounting", {}).get("rdp_orders", None)

    accounting_results = compare_accountants(
        noise_multiplier=train_cfg["noise_multiplier"],
        sampling_rate=sampling_rate,
        num_steps=num_steps,
        delta=delta,
        methods=acc_methods,
        rdp_orders=rdp_orders,
    )

    logger.info("Accounting upper bounds:")
    for method, result in accounting_results.items():
        logger.info(f"  {method}: ε = {result['epsilon']:.4f}")

    # --- Load audit results ---
    audit_results = {}
    audit_file = output_dir / "audit_one_run.json"
    if audit_file.exists():
        with open(audit_file) as f:
            audit_data = json.load(f)
        audit_results["one_run"] = audit_data["eps_audit"]
        logger.info(f"Loaded one-run audit: ε_audit = {audit_data['eps_audit']:.4f}")

    if not audit_results:
        logger.warning("No audit results found. Run auditing first. Will plot accounting only.")
        audit_results = {"none": 0.0}

    # --- Tightness metrics ---
    metrics = tightness_summary(accounting_results, audit_results, delta)
    logger.info("\nTightness Summary:")
    for m in metrics:
        logger.info(
            f"  {m.accountant_method} vs {m.auditor_method}: "
            f"gap={m.gap:.4f}, ratio={m.ratio:.4f}"
        )

    # Save metrics
    metrics_data = [
        {
            "accountant": m.accountant_method,
            "auditor": m.auditor_method,
            "gap": m.gap,
            "ratio": m.ratio,
            "eps_account": m.eps_account,
            "eps_audit": m.eps_audit,
        }
        for m in metrics
    ]
    with open(output_dir / "tightness_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    # --- Plots ---
    logger.info("Generating plots...")

    # 1. Bar chart: accountants vs auditors
    plot_audit_vs_accounting(
        accounting_results,
        audit_results,
        delta,
        save_path=output_dir / "audit_vs_accounting.png",
    )

    # 2. Gap/ratio curves over δ
    if "one_run" in audit_results:
        gap_data = gap_over_delta(
            noise_multiplier=train_cfg["noise_multiplier"],
            sampling_rate=sampling_rate,
            num_steps=num_steps,
            audit_eps=audit_results["one_run"],
        )
        plot_gap_ratio_curves(gap_data, save_path=output_dir / "gap_ratio_curves.png")

    # 3. Calibration sensitivity (if audit data available)
    if audit_file.exists():
        with open(audit_file) as f:
            aud = json.load(f)
        v, r = aud["v"], aud["r"]

        beta_curve = sensitivity_over_beta(v, r, delta=delta)
        delta_curve = sensitivity_over_delta(v, r, beta=aud["beta"])
        r_curve = sensitivity_over_r(v / r, beta=aud["beta"], delta=delta)

        plot_calibration_sensitivity(
            beta_curve=beta_curve,
            delta_curve=delta_curve,
            r_curve=r_curve,
            save_path=output_dir / "calibration_sensitivity.png",
        )

    # 4. Canary score distribution
    scores_file = output_dir / "audit_one_run_scores.npz"
    if scores_file.exists():
        data = np.load(scores_file)
        plot_canary_scores(
            data["scores"],
            data["included"],
            save_path=output_dir / "canary_scores.png",
        )

    # 5. Epsilon convergence over epochs (ε_theoretical vs ε_audit)
    eps_history = meta.get("epsilon_history", [])
    eps_audit_history = meta.get("eps_audit_history", [])
    if eps_history:
        plot_epsilon_over_epochs(
            eps_history,
            eps_audit_history=eps_audit_history if eps_audit_history else None,
            save_path=output_dir / "epsilon_over_epochs.png",
        )

    logger.info(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()
