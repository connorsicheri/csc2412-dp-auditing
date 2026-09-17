"""DP-Sniper: classifier-based black-box violation discovery (Bichsel et al. 2021).

Train a classifier to distinguish the output distributions of a mechanism
on neighboring inputs. If the classifier succeeds, it provides evidence
of a privacy violation (or a lower bound on ε).

References:
    Bichsel, Steffen, Bogunovic, Vechev (2021). DP-Sniper: Black-Box
    Discovery of Differential Privacy Violations using Classifiers.
    IEEE S&P 2021.

STATUS: Stub implementation — to be completed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DPSniperResult:
    """Result of DP-Sniper-style auditing."""

    eps_lower_bound: float
    classifier_accuracy: float  # accuracy of the distinguisher
    num_samples: int
    method: str = "dpsniper"


class DPSniperAuditor:
    """DP-Sniper-style classifier-based auditor.

    Approach:
    1. Sample mechanism outputs on neighboring datasets D, D'
    2. Train a binary classifier to distinguish outputs
    3. Classifier's ROC/accuracy → ε lower bound evidence

    Parameters
    ----------
    num_samples : int
        Number of mechanism output samples per dataset.
    confidence_beta : float
        Confidence parameter.
    delta : float
        Target δ.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        confidence_beta: float = 0.05,
        delta: float = 1e-5,
    ):
        self.num_samples = num_samples
        self.confidence_beta = confidence_beta
        self.delta = delta

    def audit(self, **kwargs) -> DPSniperResult:
        """Run DP-Sniper-style audit.

        TODO: Implement the full pipeline:
        1. Collect model outputs on test queries for D-trained and D'-trained models
        2. Train a logistic regression / neural net classifier on the outputs
        3. Use classifier's success rate to derive ε lower bound
        """
        raise NotImplementedError(
            "DP-Sniper auditor not yet implemented. " "See Bichsel et al. (2021) for the approach."
        )
