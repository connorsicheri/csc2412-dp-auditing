"""Poisoning-based DP-SGD auditing (Jagielski, Ullman, Oprea 2020).

The idea: craft poisoned examples that, when included in training, cause a
detectable behavior change in the final model. The success rate of detecting
this change translates to privacy lower-bound evidence.

References:
    Jagielski, Ullman, Oprea (2020). Auditing Differentially Private Machine
    Learning. NeurIPS 2020. arXiv:2006.07709.

STATUS: Stub implementation — to be completed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PoisoningAuditResult:
    """Result of poisoning-based auditing."""

    eps_lower_bound: float
    detection_rate: float  # fraction of runs where poisoning was detected
    num_runs: int
    method: str = "poisoning"


class PoisoningAuditor:
    """Poisoning-based auditor for DP-SGD.

    Craft poisoned examples → train with/without → detect behavior change
    → convert detection rate to ε lower bound.

    Parameters
    ----------
    num_poison_samples : int
        Number of poisoned examples to craft.
    poison_strength : float
        Strength/magnitude of the poison perturbation.
    num_runs : int
        Number of training runs (with/without poison) for statistical power.
    confidence_beta : float
        Confidence parameter.
    delta : float
        Target δ.
    """

    def __init__(
        self,
        num_poison_samples: int = 10,
        poison_strength: float = 1.0,
        num_runs: int = 20,
        confidence_beta: float = 0.05,
        delta: float = 1e-5,
    ):
        self.num_poison_samples = num_poison_samples
        self.poison_strength = poison_strength
        self.num_runs = num_runs
        self.confidence_beta = confidence_beta
        self.delta = delta

    def audit(self, **kwargs) -> PoisoningAuditResult:
        """Run poisoning-based audit.

        TODO: Implement the full pipeline:
        1. Craft poison examples targeting a specific class/behavior
        2. Train model with poison (IN) and without (OUT)
        3. Query model on test points to detect behavior change
        4. Use detection success rate to compute ε lower bound
        """
        raise NotImplementedError(
            "Poisoning auditor not yet implemented. "
            "See Jagielski et al. (2020) arXiv:2006.07709 for the approach."
        )
