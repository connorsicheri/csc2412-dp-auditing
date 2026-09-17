"""Tests for the calibration module (ε lower bound inversion).

Validates the core theory deliverable: compute_eps_lower_bound(v, r, β, δ).
Uses the paper's exact calibration from Appendix D (Steinke et al. 2023).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.auditing.calibration import (
    CalibrationResult,
    _p_of_eps,
    compute_eps_lower_bound,
    get_eps_audit,
    p_value_DP_audit,
    sensitivity_over_beta,
    sensitivity_over_r,
)


class TestPOfEps:
    """Test p(ε) = e^ε / (1 + e^ε) = sigmoid(ε)."""

    def test_zero(self):
        assert _p_of_eps(0.0) == pytest.approx(0.5)

    def test_positive(self):
        # p(1) = e/(1+e) ≈ 0.7311
        assert _p_of_eps(1.0) == pytest.approx(math.exp(1) / (1 + math.exp(1)))

    def test_large(self):
        # p(ε) → 1 as ε → ∞
        assert _p_of_eps(50.0) > 0.999

    def test_negative(self):
        # p(-ε) = 1 - p(ε) by symmetry
        assert _p_of_eps(-1.0) == pytest.approx(1 - _p_of_eps(1.0), abs=1e-10)


class TestPaperPValue:
    """Test the paper's exact p-value computation (Appendix D)."""

    def test_paper_example_1(self):
        """Paper example: v=75, m=r=100, ε=log(3), δ=0 → p≈0.553."""
        p = p_value_DP_audit(100, 100, 75, math.log(3), 0)
        assert p == pytest.approx(0.553, abs=0.01)

    def test_high_eps_gives_high_pvalue(self):
        """Large ε → large p-value (observation is not surprising)."""
        p = p_value_DP_audit(100, 100, 60, 10.0, 0)
        assert p > 0.99

    def test_zero_eps_random_guess(self):
        """ε=0, v=50/100 → should have high p-value (not surprising)."""
        p = p_value_DP_audit(100, 100, 50, 0.0, 0)
        assert p > 0.5

    def test_delta_increases_pvalue(self):
        """Adding δ should increase the p-value (harder to reject)."""
        p_no_delta = p_value_DP_audit(100, 100, 75, 1.0, 0)
        p_with_delta = p_value_DP_audit(100, 100, 75, 1.0, 0.01)
        assert p_with_delta >= p_no_delta


class TestGetEpsAudit:
    """Test the paper's bisection ε lower bound."""

    def test_paper_example_95(self):
        """Paper: v=75, m=r=100, δ=0 → ε≥0.702 at 95% confidence."""
        eps = get_eps_audit(100, 100, 75, 0, 0.05)
        assert eps == pytest.approx(0.702, abs=0.01)

    def test_paper_example_with_delta(self):
        """Paper: v=75, m=r=100, δ=1e-4 → ε≥0.699."""
        eps = get_eps_audit(100, 100, 75, 1e-4, 0.05)
        assert eps == pytest.approx(0.699, abs=0.01)

    def test_paper_example_abstentions(self):
        """Paper: v=75, r=100, m=1000, δ=1e-4 → ε≥0.673."""
        eps = get_eps_audit(1000, 100, 75, 1e-4, 0.05)
        assert eps == pytest.approx(0.673, abs=0.01)

    def test_random_guessing_gives_zero(self):
        """v ≤ r/2 → should return 0."""
        eps = get_eps_audit(100, 100, 50, 1e-5, 0.05)
        assert eps == 0.0


class TestComputeEpsLowerBound:
    """Test the wrapper that returns CalibrationResult."""

    def test_trivial_v_zero(self):
        """v=0 correct guesses → ε_audit = 0 (no evidence of leakage)."""
        result = compute_eps_lower_bound(v=0, r=100, beta=0.05, delta=1e-5)
        assert result.eps_lower_bound == 0.0

    def test_perfect_guessing(self):
        """All guesses correct → high ε_audit."""
        result = compute_eps_lower_bound(v=100, r=100, beta=0.05, delta=1e-5)
        assert result.eps_lower_bound > 1.0  # should be very large

    def test_half_correct_is_weak(self):
        """v/r = 0.5 → zero ε_audit (no better than random)."""
        result = compute_eps_lower_bound(v=50, r=100, beta=0.05, delta=1e-5)
        assert result.eps_lower_bound == 0.0

    def test_strong_signal(self):
        """v/r = 0.8, large r → nontrivial ε_audit."""
        result = compute_eps_lower_bound(v=800, r=1000, beta=0.05, delta=1e-5)
        assert result.eps_lower_bound > 0.5
        assert result.method == "paper_exact"

    def test_monotone_in_v(self):
        """More correct guesses → higher ε_audit."""
        eps_low = compute_eps_lower_bound(v=60, r=100, beta=0.05, delta=1e-5).eps_lower_bound
        eps_high = compute_eps_lower_bound(v=80, r=100, beta=0.05, delta=1e-5).eps_lower_bound
        assert eps_high > eps_low

    def test_monotone_in_beta(self):
        """Lower β (more confidence) → smaller ε_audit (harder to rule out)."""
        eps_loose = compute_eps_lower_bound(v=70, r=100, beta=0.10, delta=1e-5).eps_lower_bound
        eps_tight = compute_eps_lower_bound(v=70, r=100, beta=0.01, delta=1e-5).eps_lower_bound
        assert eps_loose >= eps_tight

    def test_m_parameter(self):
        """Larger m (more abstentions) → weaker bound due to δ-correction."""
        eps_small_m = compute_eps_lower_bound(
            v=80, r=100, beta=0.05, delta=1e-4, m=100
        ).eps_lower_bound
        eps_large_m = compute_eps_lower_bound(
            v=80, r=100, beta=0.05, delta=1e-4, m=10000
        ).eps_lower_bound
        assert eps_small_m >= eps_large_m

    def test_delta_sensitivity(self):
        """Larger δ → more δ-correction → lower ε_audit (weaker bound)."""
        eps_small_d = compute_eps_lower_bound(v=80, r=100, beta=0.05, delta=1e-7).eps_lower_bound
        eps_large_d = compute_eps_lower_bound(v=80, r=100, beta=0.05, delta=1e-3).eps_lower_bound
        assert eps_small_d >= eps_large_d

    def test_result_type(self):
        result = compute_eps_lower_bound(v=70, r=100, beta=0.05, delta=1e-5)
        assert isinstance(result, CalibrationResult)
        assert result.v == 70
        assert result.r == 100
        assert result.beta == 0.05
        assert hasattr(result, "m")


class TestSensitivityAnalysis:
    """Test sensitivity sweep utilities."""

    def test_sensitivity_over_beta(self):
        curve = sensitivity_over_beta(v=80, r=100, delta=1e-5, betas=np.array([0.01, 0.05, 0.1]))
        assert len(curve) == 3
        # Higher β → higher ε_audit (easier to rule out)
        betas, eps = zip(*curve)
        assert eps[0] <= eps[1] <= eps[2]  # β=0.01 ≤ β=0.05 ≤ β=0.1

    def test_sensitivity_over_r(self):
        curve = sensitivity_over_r(v_fraction=0.7, r_values=np.array([50, 100, 500]))
        assert len(curve) == 3
        # Larger r with same v/r → tighter bound (higher ε_audit)
        rs, eps = zip(*curve)
        assert eps[0] <= eps[-1]
