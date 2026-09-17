"""Black-box auditors for DP-SGD.

Auditors produce empirical lower bounds ε_audit on the privacy of a trained model.
"""

from .calibration import compute_eps_lower_bound
from .one_run import OneRunCanaryAuditor

__all__ = [
    "OneRunCanaryAuditor",
    "compute_eps_lower_bound",
]
