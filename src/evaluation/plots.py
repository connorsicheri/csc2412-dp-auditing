"""Plotting utilities for auditing vs accounting comparisons.

All functions save to file and optionally display inline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Style
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = sns.color_palette("colorblind")


def _save_fig(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save figure and log the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved figure to {path}")


def plot_audit_vs_accounting(
    accounting_results: dict[str, dict],
    audit_results: dict[str, float],
    delta: float,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Bar chart comparing ε upper bounds (accountants) vs lower bounds (auditors).

    Parameters
    ----------
    accounting_results : dict
        method → {"epsilon": float}
    audit_results : dict
        auditor → ε_audit (float)
    delta : float
        The δ used.
    save_path : str or Path, optional
        Where to save the figure.
    title : str, optional
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Accountants (upper bounds)
    acc_names = list(accounting_results.keys())
    acc_eps = [accounting_results[n]["epsilon"] for n in acc_names]

    # Auditors (lower bounds)
    aud_names = list(audit_results.keys())
    aud_eps = [audit_results[n] for n in aud_names]

    all_names = [f"{n}\n(account)" for n in acc_names] + [f"{n}\n(audit)" for n in aud_names]
    all_eps = acc_eps + aud_eps
    colors = [COLORS[0]] * len(acc_names) + [COLORS[1]] * len(aud_names)

    bars = ax.bar(all_names, all_eps, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, all_eps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel(f"ε (at δ = {delta:.0e})")
    ax.set_title(title or f"Accounting Upper Bounds vs Auditing Lower Bounds (δ = {delta:.0e})")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc=COLORS[0], label="Accountant (upper bound)"),
            plt.Rectangle((0, 0), 1, 1, fc=COLORS[1], label="Auditor (lower bound)"),
        ],
        loc="upper right",
    )

    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_gap_ratio_curves(
    gap_ratio_data: dict,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot gap Δ(δ) and ratio R(δ) curves across δ values.

    Parameters
    ----------
    gap_ratio_data : dict
        Output of `metrics.gap_over_delta()`.
    """
    deltas = gap_ratio_data["deltas"]
    methods = [k for k in gap_ratio_data.keys() if k != "deltas"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, method in enumerate(methods):
        data = gap_ratio_data[method]
        ax1.plot(deltas, data["gaps"], label=method, color=COLORS[i % len(COLORS)], linewidth=2)
        ax2.plot(deltas, data["ratios"], label=method, color=COLORS[i % len(COLORS)], linewidth=2)

    ax1.set_xscale("log")
    ax1.set_xlabel("δ")
    ax1.set_ylabel("Gap Δ(δ) = ε_account − ε_audit")
    ax1.set_title("Additive Gap vs δ")
    ax1.legend()

    ax2.set_xscale("log")
    ax2.set_xlabel("δ")
    ax2.set_ylabel("Ratio R(δ) = ε_audit / ε_account")
    ax2.set_title("Tightness Ratio vs δ")
    ax2.set_ylim(0, 1.05)
    ax2.legend()

    fig.suptitle("Auditing Tightness Analysis", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_calibration_sensitivity(
    beta_curve: list[tuple[float, float]] | None = None,
    delta_curve: list[tuple[float, float]] | None = None,
    r_curve: list[tuple[int, float]] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot sensitivity of ε_audit to (β, δ, r).

    Parameters
    ----------
    beta_curve : list of (β, ε_audit) pairs
    delta_curve : list of (δ, ε_audit) pairs
    r_curve : list of (r, ε_audit) pairs
    """
    n_plots = sum(x is not None for x in [beta_curve, delta_curve, r_curve])
    if n_plots == 0:
        raise ValueError("Provide at least one sensitivity curve.")

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    ax_idx = 0

    if beta_curve is not None:
        betas, eps = zip(*beta_curve)
        axes[ax_idx].plot(betas, eps, "o-", color=COLORS[0], markersize=3)
        axes[ax_idx].set_xscale("log")
        axes[ax_idx].set_xlabel("β (confidence parameter)")
        axes[ax_idx].set_ylabel("ε_audit")
        axes[ax_idx].set_title("Sensitivity to β")
        ax_idx += 1

    if delta_curve is not None:
        deltas, eps = zip(*delta_curve)
        axes[ax_idx].plot(deltas, eps, "o-", color=COLORS[1], markersize=3)
        axes[ax_idx].set_xscale("log")
        axes[ax_idx].set_xlabel("δ")
        axes[ax_idx].set_ylabel("ε_audit")
        axes[ax_idx].set_title("Sensitivity to δ")
        ax_idx += 1

    if r_curve is not None:
        rs, eps = zip(*r_curve)
        axes[ax_idx].plot(rs, eps, "o-", color=COLORS[2], markersize=4)
        axes[ax_idx].set_xscale("log")
        axes[ax_idx].set_xlabel("r (number of guesses)")
        axes[ax_idx].set_ylabel("ε_audit")
        axes[ax_idx].set_title("Sensitivity to r (fixed v/r)")
        ax_idx += 1

    fig.suptitle("ε_audit Calibration Sensitivity", fontsize=13, y=1.02)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_canary_scores(
    scores: np.ndarray,
    included: np.ndarray,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Histogram of canary scores, colored by IN/OUT ground truth.

    Parameters
    ----------
    scores : np.ndarray
        Per-canary scores from the auditor.
    included : np.ndarray of bool
        Ground truth inclusion mask.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(scores[included], bins=50, alpha=0.6, label="IN (included)", color=COLORS[0])
    ax.hist(scores[~included], bins=50, alpha=0.6, label="OUT (excluded)", color=COLORS[1])
    ax.set_xlabel("Canary Score (−loss)")
    ax.set_ylabel("Count")
    ax.set_title("Canary Score Distribution")
    ax.legend()

    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_epsilon_over_epochs(
    epsilon_history: list[float],
    eps_audit_history: list[float] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot ε_theoretical and ε_audit over training epochs.

    Parameters
    ----------
    epsilon_history : list[float]
        Theoretical ε after each epoch (from privacy accounting).
    eps_audit_history : list[float], optional
        Audited ε after each epoch (from per-epoch canary audit).
    save_path : str or Path, optional
        Where to save the figure.
    title : str, optional
        Plot title.
    """
    epochs = list(range(1, len(epsilon_history) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epochs,
        epsilon_history,
        "o-",
        color=COLORS[0],
        linewidth=2,
        markersize=5,
        label="ε theoretical (accounting)",
    )

    if eps_audit_history and len(eps_audit_history) > 0:
        ax.plot(
            epochs[: len(eps_audit_history)],
            eps_audit_history,
            "s-",
            color=COLORS[1],
            linewidth=2,
            markersize=5,
            label="ε audit (one-run)",
        )

        # Shade the gap
        min_len = min(len(epsilon_history), len(eps_audit_history))
        ax.fill_between(
            epochs[:min_len],
            eps_audit_history[:min_len],
            epsilon_history[:min_len],
            alpha=0.15,
            color="gray",
            label="Gap",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("ε")
    ax.set_title(title or "Privacy Budget: Accounting vs Auditing Over Training")
    ax.legend()
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig
