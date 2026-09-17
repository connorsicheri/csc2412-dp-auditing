"""Run DP-SGD training with optional canary insertion.

Usage:
    python experiments/run_training.py --config configs/mnist_logreg.yaml
    python experiments/run_training.py --config configs/cifar10_cnn.yaml --no-canaries
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training import (
    build_dataloaders,
    get_input_shape,
    get_num_classes,
    load_dataset,
)
from src.training.dpsgd import train_dpsgd
from src.training.models import build_model
from src.utils import get_device, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DP-SGD Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--no-canaries", action="store_true", help="Skip canary insertion")
    parser.add_argument(
        "--no-audit", action="store_true", help="Insert canaries but skip per-epoch auditing"
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.output_dir:
        cfg["evaluation"]["output_dir"] = args.output_dir

    output_dir = Path(cfg["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info(f"Device: {device}")

    # Seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    # Load data
    logger.info(f"Loading dataset: {cfg['dataset']['name']}")
    train_dataset, test_dataset = load_dataset(cfg)
    input_shape = get_input_shape(cfg["dataset"]["name"])
    num_classes = get_num_classes(cfg["dataset"]["name"])

    # Optionally subsample the training set (for m=n experiments with small n)
    subset_n = cfg["dataset"].get("subset_n", None)
    if subset_n is not None and subset_n < len(train_dataset):
        from torch.utils.data import Subset

        indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(seed))[
            :subset_n
        ].tolist()
        train_dataset = Subset(train_dataset, indices)
        logger.info(f"Subsampled training set to {subset_n} examples")

    # Optionally insert canaries
    canary_data = None
    if not args.no_canaries:
        from src.auditing.one_run import OneRunCanaryAuditor

        audit_cfg = cfg.get("auditing", {}).get("one_run", {})
        auditor = OneRunCanaryAuditor(
            num_canaries=audit_cfg.get("num_canaries", 500),
            abstention_threshold=audit_cfg.get("abstention_threshold", 0.0),
            confidence_beta=audit_cfg.get("confidence_beta", 0.05),
            delta=cfg["training"]["delta"],
            seed=seed,
        )
        mislabel = audit_cfg.get("mislabel", True)
        include_base = audit_cfg.get("include_base", True)
        included, canary_dataset = auditor.generate_canaries(
            train_dataset,
            num_classes,
            mislabel=mislabel,
        )
        train_dataset = auditor.build_training_set(
            train_dataset,
            canary_dataset,
            included,
            include_base=include_base,
        )
        canary_data = {
            "auditor": auditor,
            "included": included,
            "canary_dataset": canary_dataset,
        }
        logger.info(
            f"Inserted {included.sum()}/{len(included)} canaries into training set "
            f"(total: {len(train_dataset)} examples)"
        )

    # Build dataloaders
    batch_size = cfg["training"]["batch_size"]
    loaders = build_dataloaders(
        train_dataset,
        test_dataset,
        batch_size,
        val_fraction=cfg["dataset"].get("val_fraction", 0.1),
        seed=seed,
    )

    # Build model
    model = build_model(cfg, input_shape, num_classes)
    logger.info(
        f"Model: {cfg['model']['arch']} ({sum(p.numel() for p in model.parameters())} params)"
    )

    # Save a copy of the init weights for the paper's black-box score:
    # Score(x) = loss(w0, x) - loss(w_final, x)
    import copy

    init_state_dict = copy.deepcopy(model.state_dict())
    torch.save(init_state_dict, output_dir / "model_init.pt")
    logger.info(f"Saved initial model weights to {output_dir / 'model_init.pt'}")

    # Build init model reference for per-epoch auditing
    # Keep on CPU — the background audit thread runs entirely on CPU.
    init_model = None
    if canary_data is not None:
        init_model = build_model(cfg, input_shape, num_classes)
        init_model.load_state_dict(init_state_dict)
        init_model = init_model.cpu().float()
        init_model.eval()

    # Train with DP-SGD
    logger.info("Starting DP-SGD training...")

    # Per-epoch audit callback (if canaries present and auditing not disabled)
    # Now receives a CPU state_dict (not the live model) so it can run in a background thread.
    epoch_callback = None
    if canary_data is not None and init_model is not None and not args.no_audit:
        auditor = canary_data["auditor"]
        _canary_ds = canary_data["canary_dataset"]
        _included = canary_data["included"]
        _init_m = init_model
        _audit_model_template = build_model(cfg, input_shape, num_classes)

        def epoch_callback(state_dict, epoch):
            # Build a fresh model on CPU from the snapshotted weights
            audit_model = build_model(cfg, input_shape, num_classes)
            audit_model.load_state_dict(state_dict)
            audit_model.eval()
            return auditor.quick_audit(
                audit_model,
                _canary_ds,
                _included,
                device="cpu",
                init_model=_init_m,
            )

    result = train_dpsgd(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        noise_multiplier=cfg["training"]["noise_multiplier"],
        delta=cfg["training"]["delta"],
        device=device,
        max_physical_batch_size=cfg["training"].get("max_physical_batch_size", 128),
        optimizer_name=cfg["training"].get("optimizer", "adam"),
        weight_decay=cfg["training"].get("weight_decay", 0.0),
        epoch_callback=epoch_callback,
        eval_every_n_epochs=cfg["training"].get("eval_every_n_epochs", 10),
        accounting_methods=cfg["training"].get("accounting_methods", ["rdp", "zcdp", "pld"]),
        disable_dp=cfg["training"].get("disable_dp", False),
        output_dir=output_dir,
        lr_schedule=cfg["training"].get("lr_schedule", None),
    )

    # Save model (unwrap Opacus GradSampleModule prefix if present)
    model_path = output_dir / "model.pt"
    state_dict = result.model.state_dict()
    # Opacus wraps keys as "_module.<key>"; strip the prefix for clean loading
    cleaned = {k.removeprefix("_module."): v for k, v in state_dict.items()}
    torch.save(cleaned, model_path)
    logger.info(f"Saved model to {model_path}")

    # --- Evaluate on test set ---
    from src.training.dpsgd import _evaluate

    test_loader = loaders["test"]
    test_acc = _evaluate(result.model, test_loader, device)
    logger.info(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Final val accuracy
    final_val_acc = result.val_accuracies[-1] if result.val_accuracies else None
    logger.info(f"Final val accuracy: {final_val_acc:.4f}" if final_val_acc else "No val accuracy")

    # Save training metadata
    # epsilon_history is now {method: [eps_values]} — extract final per method
    final_epsilons = {}
    for method, eps_list in result.epsilon_history.items():
        if eps_list:
            final_epsilons[method] = eps_list[-1]
    meta = {
        "config": cfg,
        "train_losses": result.train_losses,
        "val_accuracies": result.val_accuracies,
        "test_accuracy": test_acc,
        "epsilon_history": result.epsilon_history,
        "eps_audit_history": result.eps_audit_history,
        "audit_epochs": result.audit_epochs,
        "eval_epochs": result.eval_epochs,
        "total_steps": result.total_steps,
        "final_epsilons": final_epsilons,
        "device": device,
    }
    meta_path = output_dir / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Saved training metadata to {meta_path}")

    # Save canary data for later auditing
    if canary_data is not None:
        import numpy as np

        canary_path = output_dir / "canary_data.npz"
        np.savez(canary_path, included=canary_data["included"])
        torch.save(canary_data["canary_dataset"], output_dir / "canary_dataset.pt")
        logger.info(f"Saved canary data to {canary_path}")

    eps_summary = ", ".join(f"{m}={e:.2f}" for m, e in final_epsilons.items())
    logger.info(
        f"Training complete. Final ε: {eps_summary}  "
        f"(δ={cfg['training']['delta']}, steps={result.total_steps})"
    )

    return result, canary_data


if __name__ == "__main__":
    main()
