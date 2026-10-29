"""Tests for the deconvolution parameter resolution."""

import sys
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

from aind_ophys_oasis_event_detection_library.params import (
    DECONVOLVE_KEYS,
    ar_coefficients,
    deconvolve_kwargs,
    resolve_parameters,
    taus_from_ar_coefficients,
)
from aind_ophys_oasis_event_detection_library.settings import OasisSettings

FRAME_RATE = 10.0


def _settings(**kwargs: Any) -> OasisSettings:
    """Construct OasisSettings with a controlled sys.argv."""
    with patch.object(sys, "argv", ["run_capsule.py"]):
        return OasisSettings(**kwargs)


class TestArCoefficients(unittest.TestCase):
    """Time constants to auto-regressive coefficients."""

    def test_unset_tau_ar1(self) -> None:
        """No tau and no rise time leaves a single unset coefficient."""
        self.assertEqual(ar_coefficients(None, 0, FRAME_RATE), (None,))

    def test_unset_tau_ar2(self) -> None:
        """No tau with a rise time leaves two unset coefficients."""
        self.assertEqual(ar_coefficients(None, 0.1, FRAME_RATE), (None, None))

    def test_ar1_value(self) -> None:
        """AR1 coefficient is exp(-1 / (tau * frame_rate))."""
        (g1,) = ar_coefficients(1.0, 0, FRAME_RATE)
        self.assertAlmostEqual(g1, float(np.exp(-1 / FRAME_RATE)))

    def test_ar2_values(self) -> None:
        """AR2 coefficients are (d + r, -d * r)."""
        g1, g2 = ar_coefficients(1.0, 0.5, FRAME_RATE)
        decay = float(np.exp(-1 / (1.0 * FRAME_RATE)))
        rise = float(np.exp(-1 / (0.5 * FRAME_RATE)))
        self.assertAlmostEqual(g1, decay + rise)
        self.assertAlmostEqual(g2, -decay * rise)


class TestTausFromArCoefficients(unittest.TestCase):
    """Inverting the estimated coefficients back to time constants."""

    def test_ar1_round_trip(self) -> None:
        """A 1-D coefficient array inverts to decay times, no rise times."""
        tau = 1.0
        (g1,) = ar_coefficients(tau, 0, FRAME_RATE)
        tau_hat, tau_rise_hat = taus_from_ar_coefficients(np.array([g1, g1]), FRAME_RATE)
        self.assertIsNone(tau_rise_hat)
        np.testing.assert_allclose(tau_hat, [tau, tau], rtol=1e-6)

    def test_ar2_round_trip(self) -> None:
        """A 2-D coefficient array inverts to decay and rise times."""
        tau, tau_rise = 1.0, 0.4
        g1, g2 = ar_coefficients(tau, tau_rise, FRAME_RATE)
        tau_hat, tau_rise_hat = taus_from_ar_coefficients(np.array([[g1, g2]]), FRAME_RATE)
        np.testing.assert_allclose(tau_hat, [tau], rtol=1e-6)
        np.testing.assert_allclose(tau_rise_hat, [tau_rise], rtol=1e-6)


class TestResolveParameters(unittest.TestCase):
    """Parameter resolution and the estimate_parameters=False fallbacks."""

    def test_estimating_keeps_none_values(self) -> None:
        """When estimating, unset lam/baseline stay None for OASIS to fit."""
        params = resolve_parameters(_settings(), FRAME_RATE)
        self.assertIsNone(params["lam"])
        self.assertIsNone(params["b"])
        self.assertEqual(params["g"], (None,))

    def test_field_names_mapped_to_oasis_kwargs(self) -> None:
        """baseline/b_nonneg/optimize_tau map onto OASIS kwargs."""
        params = resolve_parameters(
            _settings(baseline=3.0, b_nonneg=False, optimize_tau=5),
            FRAME_RATE,
        )
        self.assertEqual(params["b"], 3.0)
        self.assertFalse(params["b_nonneg"])
        self.assertEqual(params["optimize_g"], 5)

    def test_not_estimating_without_tau_raises(self) -> None:
        """estimate_parameters=False needs an explicit tau."""
        with self.assertRaises(ValueError) as ctx:
            resolve_parameters(_settings(estimate_parameters=False), FRAME_RATE)
        self.assertIn("tau", str(ctx.exception))

    def test_not_estimating_defaults_lam_and_baseline_to_zero(self) -> None:
        """Unset lam/baseline fall back to 0 when not estimating."""
        params = resolve_parameters(_settings(estimate_parameters=False, tau=1.0), FRAME_RATE)
        self.assertEqual(params["lam"], 0)
        self.assertEqual(params["b"], 0)

    def test_not_estimating_preserves_explicit_values(self) -> None:
        """Explicit lam/baseline are preserved when not estimating."""
        params = resolve_parameters(
            _settings(estimate_parameters=False, tau=1.0, lam=0.5, baseline=2.0),
            FRAME_RATE,
        )
        self.assertEqual(params["lam"], 0.5)
        self.assertEqual(params["b"], 2.0)


class TestDeconvolveKwargs(unittest.TestCase):
    """Selecting the subset OASIS accepts."""

    def test_selects_only_deconvolve_keys(self) -> None:
        """Only the documented OASIS keyword names are passed through."""
        params = resolve_parameters(_settings(), FRAME_RATE)
        self.assertEqual(sorted(deconvolve_kwargs(params)), sorted(DECONVOLVE_KEYS))


if __name__ == "__main__":
    unittest.main()
