"""Evaluation: metrics and comparison between accountants and auditors."""

from .metrics import compute_gap, compute_ratio, tightness_summary
from .plots import (
    plot_audit_vs_accounting,
    plot_calibration_sensitivity,
    plot_canary_scores,
    plot_gap_ratio_curves,
)

__all__ = [
    "compute_gap",
    "compute_ratio",
    "tightness_summary",
    "plot_gap_ratio_curves",
    "plot_audit_vs_accounting",
    "plot_calibration_sensitivity",
    "plot_canary_scores",
]
