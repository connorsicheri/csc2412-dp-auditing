"""Run auditing on a trained DP-SGD model.

Usage:
    python experiments/run_audit.py --config configs/mnist_logreg.yaml --auditor one_run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.auditing.one_run import OneRunCanaryAuditor
from src.training import get_input_shape, get_num_classes
from src.training.models import build_model
from src.utils import get_device, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_one_run_audit(cfg: dict, output_dir: Path, device: str) -> dict:
    """Run one-run canary audit on a previously trained model."""
    # Load model
    model = build_model(
        cfg,
        get_input_shape(cfg["dataset"]["name"]),
        get_num_classes(cfg["dataset"]["name"]),
    )
    model_path = output_dir / "model.pt"
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    # Handle Opacus _module. prefix if present
    cleaned = {k.removeprefix("_module."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model = model.to(device)
    logger.info(f"Loaded model from {model_path}")

    # Load canary data
    canary_data = np.load(output_dir / "canary_data.npz")
    included = canary_data["included"]
    canary_dataset = torch.load(output_dir / "canary_dataset.pt", weights_only=False)
    logger.info(f"Loaded canary data: {included.sum()}/{len(included)} included")

    # Load initial model for paper's score: loss(w0) - loss(w_final)
    init_model = None
    init_path = output_dir / "model_init.pt"
    if init_path.exists():
        init_model = build_model(
            cfg,
            get_input_shape(cfg["dataset"]["name"]),
            get_num_classes(cfg["dataset"]["name"]),
        )
        init_state_dict = torch.load(init_path, map_location=device, weights_only=True)
        init_model.load_state_dict(init_state_dict)
        init_model = init_model.to(device)
        logger.info("Loaded initial model for paper-style scoring")
    else:
        logger.warning("No model_init.pt found — falling back to -loss(w_final) scoring")

    # Run audit
    audit_cfg = cfg.get("auditing", {}).get("one_run", {})
    auditor = OneRunCanaryAuditor(
        num_canaries=len(included),
        abstention_threshold=audit_cfg.get("abstention_threshold", 0.0),
        confidence_beta=audit_cfg.get("confidence_beta", 0.05),
        delta=cfg["training"]["delta"],
    )

    result = auditor.audit(model, canary_dataset, included, device=device, init_model=init_model)

    # Save results
    audit_result = {
        "v": result.v,
        "r": result.r,
        "m": result.m,
        "p_hat": result.v / result.r if result.r > 0 else 0,
        "eps_audit": result.calibration.eps_lower_bound if result.calibration else None,
        "beta": audit_cfg.get("confidence_beta", 0.05),
        "delta": cfg["training"]["delta"],
        "method": "one_run",
    }

    result_path = output_dir / "audit_one_run.json"
    with open(result_path, "w") as f:
        json.dump(audit_result, f, indent=2)
    logger.info(f"Saved audit results to {result_path}")

    # Save scores for plotting
    np.savez(
        output_dir / "audit_one_run_scores.npz",
        scores=result.scores,
        included=result.included,
        guesses=result.guesses,
    )

    return audit_result


def main():
    parser = argparse.ArgumentParser(description="DP-SGD Auditing")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--auditor",
        type=str,
        default="one_run",
        choices=["one_run", "worst_case_init", "poisoning", "dpsniper"],
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir or cfg["evaluation"]["output_dir"])
    device = get_device()

    if args.auditor == "one_run":
        result = run_one_run_audit(cfg, output_dir, device)
        logger.info(
            f"One-run audit: ε_audit = {result['eps_audit']:.4f} "
            f"(v={result['v']}, r={result['r']}, β={result['beta']})"
        )
    else:
        logger.error(f"Auditor '{args.auditor}' not yet implemented.")
        sys.exit(1)


if __name__ == "__main__":
    main()
