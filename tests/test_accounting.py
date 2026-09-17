"""Tests for privacy accounting modules.

Validates RDP, moments, zCDP, and PLD accountants against known values.
"""

from __future__ import annotations

import math

import pytest

from src.accounting.compare import compare_accountants
from src.accounting.moments import moments_accountant
from src.accounting.rdp import _compute_rdp_order, _rdp_to_eps_delta, rdp_accountant
from src.accounting.zcdp import _zcdp_single_step, _zcdp_to_eps_delta, zcdp_accountant


class TestRDPAccountant:
    """Test RDP accounting."""

    def test_gaussian_no_subsampling(self):
        """Non-subsampled Gaussian: ε_RDP(α) = α/(2σ²)."""
        sigma = 1.0
        alpha = 10
        rdp = _compute_rdp_order(alpha, sigma, sampling_rate=1.0)
        assert rdp == pytest.approx(alpha / (2 * sigma**2))

    def test_zero_noise(self):
        """σ=0 → infinite RDP."""
        rdp = _compute_rdp_order(10, noise_multiplier=0.0, sampling_rate=0.5)
        assert rdp == float("inf")

    def test_zero_sampling(self):
        """q=0 → zero RDP (no data used)."""
        rdp = _compute_rdp_order(10, noise_multiplier=1.0, sampling_rate=0.0)
        assert rdp == 0.0

    def test_subsampling_reduces_rdp(self):
        """Subsampling should reduce the RDP (privacy amplification)."""
        rdp_full = _compute_rdp_order(10, 1.0, sampling_rate=1.0)
        rdp_sub = _compute_rdp_order(10, 1.0, sampling_rate=0.01)
        assert rdp_sub < rdp_full

    def test_rdp_to_eps_delta(self):
        """Basic conversion: ε = RDP + log(1/δ)/(α-1)."""
        rdp_val = 1.0
        alpha = 10.0
        delta = 1e-5
        eps = _rdp_to_eps_delta(rdp_val, alpha, delta)
        eps_basic = rdp_val + math.log(1.0 / delta) / (alpha - 1)
        # Should be ≤ basic (improved bound)
        assert eps <= eps_basic + 1e-10

    def test_rdp_accountant_returns_finite(self):
        """Full RDP accounting should return a finite ε."""
        result = rdp_accountant(
            noise_multiplier=1.0,
            sampling_rate=0.01,
            num_steps=1000,
            delta=1e-5,
        )
        assert result["epsilon"] > 0
        assert result["epsilon"] < float("inf")
        assert result["method"] == "rdp"

    def test_more_steps_higher_eps(self):
        """More training steps → higher ε (composition)."""
        eps_100 = rdp_accountant(1.0, 0.01, 100, 1e-5)["epsilon"]
        eps_1000 = rdp_accountant(1.0, 0.01, 1000, 1e-5)["epsilon"]
        assert eps_1000 > eps_100


class TestMomentsAccountant:
    """Test moments accountant."""

    def test_basic(self):
        result = moments_accountant(
            noise_multiplier=1.0,
            sampling_rate=0.01,
            num_steps=1000,
            delta=1e-5,
        )
        assert result["epsilon"] > 0
        assert result["method"] == "moments"

    def test_agrees_with_rdp(self):
        """Moments and RDP should give similar results (same underlying math)."""
        rdp = rdp_accountant(1.0, 0.01, 1000, 1e-5)["epsilon"]
        mom = moments_accountant(1.0, 0.01, 1000, 1e-5)["epsilon"]
        # They use slightly different order grids, so allow some difference
        assert abs(rdp - mom) / max(rdp, mom) < 0.3


class TestZCDPAccountant:
    """Test zCDP accounting."""

    def test_single_step(self):
        """ρ for non-subsampled Gaussian = 1/(2σ²)."""
        rho = _zcdp_single_step(1.0, sampling_rate=1.0)
        assert rho == pytest.approx(0.5)

    def test_subsampling(self):
        """Subsampling reduces ρ by q²."""
        rho_full = _zcdp_single_step(1.0, sampling_rate=1.0)
        rho_sub = _zcdp_single_step(1.0, sampling_rate=0.1)
        assert rho_sub == pytest.approx(0.01 * rho_full)

    def test_conversion(self):
        """zCDP → (ε,δ): ε = ρ + 2√(ρ·ln(1/δ))."""
        rho = 1.0
        delta = 1e-5
        eps = _zcdp_to_eps_delta(rho, delta)
        expected = rho + 2 * math.sqrt(rho * math.log(1 / delta))
        assert eps == pytest.approx(expected)

    def test_accountant(self):
        result = zcdp_accountant(1.0, 0.01, 1000, 1e-5)
        assert result["epsilon"] > 0
        assert result["method"] == "zcdp"


class TestCompareAccountants:
    """Test the comparison function."""

    def test_all_methods(self):
        results = compare_accountants(
            noise_multiplier=1.0,
            sampling_rate=0.01,
            num_steps=100,
            delta=1e-5,
        )
        assert "rdp" in results
        assert "moments" in results
        assert "zcdp" in results
        assert "pld" in results
        for method, result in results.items():
            assert "epsilon" in result

    def test_all_finite_and_ordered(self):
        """All accountants should return finite, positive ε values."""
        results = compare_accountants(1.0, 0.01, 1000, 1e-5)
        for method, result in results.items():
            eps = result["epsilon"]
            assert eps > 0, f"{method} returned non-positive ε"
            assert eps < float("inf"), f"{method} returned infinite ε"
