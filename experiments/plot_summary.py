#!/usr/bin/env env python
"""Generate cross-experiment comparison plots from completed experiments.

Usage:
    python experiments/plot_summary.py
    python experiments/plot_summary.py --results-dir results/experiments
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = sns.color_palette("colorblind")


def load_experiment(exp_dir: Path) -> dict | None:
    """Load training_meta.json and audit_one_run.json from an experiment dir."""
    meta_path = exp_dir / "training_meta.json"
    audit_path = exp_dir / "audit_one_run.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    audit = None
    if audit_path.exists():
        with open(audit_path) as f:
            audit = json.load(f)

    cfg = meta["config"]
    return {
        "name": exp_dir.name,
        "model": cfg["model"]["arch"],
        "epochs": cfg["training"]["epochs"],
        "noise_mult": cfg["training"]["noise_multiplier"],
        "test_acc": meta.get("test_accuracy", 0),
        "eps_theoretical": meta.get("final_epsilon", 0),
        "epsilon_history": meta.get("epsilon_history", []),
        "eps_audit_history": meta.get("eps_audit_history", []),
        "eps_audit": audit.get("eps_audit", 0) if audit else 0,
        "v": audit.get("v", 0) if audit else 0,
        "r": audit.get("r", 0) if audit else 0,
        "p_hat": audit.get("p_hat", 0) if audit else 0,
    }


MODEL_LABELS = {"cnn": "Deep CNN", "wrn": "WRN-16-4", "vit": "ViT-Small"}
MODEL_ORDER = ["cnn", "wrn", "vit"]


def plot_architecture_comparison(exps: list[dict], save_dir: Path) -> None:
    """Grouped bar chart: ε_theoretical vs ε_audit by architecture."""
    # Filter: same epochs (20), same noise (0.7), different models
    arch_exps = [e for e in exps if e["epochs"] == 20 and e["noise_mult"] == 0.7]
    arch_exps.sort(key=lambda e: MODEL_ORDER.index(e["model"]) if e["model"] in MODEL_ORDER else 99)

    if len(arch_exps) < 2:
        logger.warning("Not enough architecture experiments to compare.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    labels = [MODEL_LABELS.get(e["model"], e["model"]) for e in arch_exps]
    eps_theo = [e["eps_theoretical"] for e in arch_exps]
    eps_audit = [e["eps_audit"] for e in arch_exps]

    x = np.arange(len(labels))
    w = 0.35

    bars1 = ax.bar(
        x - w / 2,
        eps_theo,
        w,
        label="ε theoretical",
        color=COLORS[0],
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + w / 2, eps_audit, w, label="ε audit", color=COLORS[1], edgecolor="black", linewidth=0.5
    )

    for b, v in zip(bars1, eps_theo):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.05,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for b, v in zip(bars2, eps_audit):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.05,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ε")
    ax.set_title("Architecture Comparison (CIFAR-10, 20 epochs, σ=0.7)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "comparison_architecture.png", dpi=150, bbox_inches="tight")
    logger.info("Saved comparison_architecture.png")
    plt.close(fig)


def plot_epoch_scaling(exps: list[dict], save_dir: Path) -> None:
    """Line plot: ε_theoretical and ε_audit vs number of epochs."""
    epoch_exps = [e for e in exps if e["model"] == "wrn" and e["noise_mult"] == 0.7]
    epoch_exps.sort(key=lambda e: e["epochs"])

    if len(epoch_exps) < 2:
        logger.warning("Not enough epoch experiments to compare.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = [e["epochs"] for e in epoch_exps]
    eps_theo = [e["eps_theoretical"] for e in epoch_exps]
    eps_audit = [e["eps_audit"] for e in epoch_exps]

    ax.plot(
        epochs, eps_theo, "o-", color=COLORS[0], linewidth=2, markersize=8, label="ε theoretical"
    )
    ax.plot(epochs, eps_audit, "s-", color=COLORS[1], linewidth=2, markersize=8, label="ε audit")

    ax.fill_between(epochs, eps_audit, eps_theo, alpha=0.15, color="gray", label="Gap")

    for ep, et, ea in zip(epochs, eps_theo, eps_audit):
        ax.annotate(
            f"{et:.2f}",
            (ep, et),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )
        ax.annotate(
            f"{ea:.4f}",
            (ep, ea),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=9,
        )

    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("ε")
    ax.set_title("Epoch Scaling (WRN-16-4, σ=0.7)")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(save_dir / "comparison_epochs.png", dpi=150, bbox_inches="tight")
    logger.info("Saved comparison_epochs.png")
    plt.close(fig)


def plot_noise_tradeoff(exps: list[dict], save_dir: Path) -> None:
    """Line plot: ε_theoretical and ε_audit vs noise multiplier σ."""
    noise_exps = [e for e in exps if e["model"] == "wrn" and e["epochs"] == 20]
    noise_exps.sort(key=lambda e: e["noise_mult"])

    if len(noise_exps) < 2:
        logger.warning("Not enough noise experiments to compare.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    sigmas = [e["noise_mult"] for e in noise_exps]
    eps_theo = [e["eps_theoretical"] for e in noise_exps]
    eps_audit = [e["eps_audit"] for e in noise_exps]

    ax.plot(
        sigmas, eps_theo, "o-", color=COLORS[0], linewidth=2, markersize=8, label="ε theoretical"
    )
    ax.plot(sigmas, eps_audit, "s-", color=COLORS[1], linewidth=2, markersize=8, label="ε audit")

    ax.fill_between(sigmas, eps_audit, eps_theo, alpha=0.15, color="gray", label="Gap")

    for s, et, ea in zip(sigmas, eps_theo, eps_audit):
        ax.annotate(
            f"{et:.2f}",
            (s, et),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )
        ax.annotate(
            f"{ea:.4f}",
            (s, ea),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=9,
        )

    ax.set_xlabel("Noise Multiplier σ")
    ax.set_ylabel("ε")
    ax.set_title("Noise-Privacy Tradeoff (WRN-16-4, 20 epochs)")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.invert_xaxis()  # lower σ → higher ε, so reverse for intuition
    fig.tight_layout()
    fig.savefig(save_dir / "comparison_noise.png", dpi=150, bbox_inches="tight")
    logger.info("Saved comparison_noise.png")
    plt.close(fig)


def plot_convergence_overlay(exps: list[dict], save_dir: Path) -> None:
    """Overlay all experiments' ε_theoretical and ε_audit over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, e in enumerate(exps):
        c = COLORS[i % len(COLORS)]
        label = f"{MODEL_LABELS.get(e['model'], e['model'])}, {e['epochs']}ep, σ={e['noise_mult']}"
        epochs = list(range(1, len(e["epsilon_history"]) + 1))

        ax1.plot(
            epochs, e["epsilon_history"], "o-", color=c, markersize=3, linewidth=1.5, label=label
        )

        if e["eps_audit_history"]:
            ax2.plot(
                epochs[: len(e["eps_audit_history"])],
                e["eps_audit_history"],
                "s-",
                color=c,
                markersize=3,
                linewidth=1.5,
                label=label,
            )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ε theoretical")
    ax1.set_title("Theoretical ε Over Training")
    ax1.legend(fontsize=8)
    ax1.set_ylim(bottom=0)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("ε audit")
    ax2.set_title("Audited ε Over Training")
    ax2.legend(fontsize=8)
    ax2.set_ylim(bottom=0)

    fig.suptitle("Privacy Budget Convergence — All Experiments", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "comparison_convergence.png", dpi=150, bbox_inches="tight")
    logger.info("Saved comparison_convergence.png")
    plt.close(fig)


def plot_tightness_ratios(exps: list[dict], save_dir: Path) -> None:
    """Horizontal bar chart: ε_audit / ε_theoretical ratio for all experiments."""
    valid = [e for e in exps if e["eps_theoretical"] > 0]
    valid.sort(key=lambda e: e["eps_audit"] / e["eps_theoretical"] if e["eps_theoretical"] else 0)

    if not valid:
        return

    fig, ax = plt.subplots(figsize=(9, max(4, len(valid) * 0.7)))

    labels = [
        f"{MODEL_LABELS.get(e['model'], e['model'])}\n{e['epochs']}ep, σ={e['noise_mult']}"
        for e in valid
    ]
    ratios = [e["eps_audit"] / e["eps_theoretical"] if e["eps_theoretical"] else 0 for e in valid]

    bars = ax.barh(labels, ratios, color=COLORS[1], edgecolor="black", linewidth=0.5)
    for bar, r in zip(bars, ratios):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{r:.4f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Tightness Ratio (ε_audit / ε_theoretical)")
    ax.set_title("Auditing Tightness Across Experiments")
    ax.set_xlim(0, max(ratios) * 1.3 if ratios else 1)
    fig.tight_layout()
    fig.savefig(save_dir / "comparison_tightness.png", dpi=150, bbox_inches="tight")
    logger.info("Saved comparison_tightness.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Cross-experiment summary plots")
    parser.add_argument(
        "--results-dir",
        default="results/experiments",
        help="Directory containing experiment subdirectories",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    # Load all experiments
    exps = []
    for sub in sorted(results_dir.iterdir()):
        if sub.is_dir():
            data = load_experiment(sub)
            if data:
                exps.append(data)
                logger.info(
                    f"Loaded: {sub.name} (ε_theo={data['eps_theoretical']:.2f}, ε_audit={data['eps_audit']:.4f})"
                )

    if not exps:
        logger.error("No completed experiments found.")
        sys.exit(1)

    logger.info(f"Loaded {len(exps)} experiments. Generating comparison plots...")

    save_dir = results_dir
    plot_architecture_comparison(exps, save_dir)
    plot_epoch_scaling(exps, save_dir)
    plot_noise_tradeoff(exps, save_dir)
    plot_convergence_overlay(exps, save_dir)
    plot_tightness_ratios(exps, save_dir)

    logger.info(f"All summary plots saved to {save_dir}")


if __name__ == "__main__":
    main()
