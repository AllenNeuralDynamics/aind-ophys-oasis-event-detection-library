"""Deconvolution parameter resolution for OASIS event detection.

Pure, numpy-only logic: converting decay/rise time constants into the
coefficients of the auto-regressive process OASIS fits, and resolving the
parameter set actually handed to ``oasis.functions.deconvolve``. Kept
separate from :mod:`job` and :mod:`events` so the trickiest arithmetic in
this step is unit-tested rather than coverage-omitted.
"""

import logging
from typing import Optional, Tuple

import numpy as np

from aind_ophys_oasis_event_detection_library.settings import OasisSettings

logger = logging.getLogger(__name__)

# Keys ``oasis.functions.deconvolve`` accepts.
DECONVOLVE_KEYS = ("g", "sn", "b", "b_nonneg", "penalty", "optimize_g")


def ar_coefficients(
    tau: Optional[float],
    tau_rise: float,
    frame_rate: float,
) -> Tuple[Optional[float], ...]:
    """Convert time constants to auto-regressive process coefficients.

    ``tau_rise == 0`` selects a single-coefficient AR1 model; any other
    value selects a two-coefficient AR2 model. A ``tau`` of ``None`` leaves
    the coefficients unset (``None``), which instructs OASIS to estimate
    them from the autocovariance of the data.

    Parameters
    ----------
    tau : float or None
        Exponential decay time in seconds, or ``None`` to estimate it.
    tau_rise : float
        Exponential rise time in seconds; ``0`` means negligible rise time.
    frame_rate : float
        Acquisition frame rate in Hz.

    Returns
    -------
    tuple
        ``(g1,)`` for AR1 or ``(g1, g2)`` for AR2. Entries are ``None``
        when ``tau`` is ``None``.
    """
    if tau is None:
        return (None,) if tau_rise == 0 else (None, None)
    decay = float(np.exp(-1 / (tau * frame_rate)))
    if tau_rise == 0:
        return (decay,)
    rise = float(np.exp(-1 / (tau_rise * frame_rate)))
    return (decay + rise, -decay * rise)


def taus_from_ar_coefficients(
    g_hat: np.ndarray,
    frame_rate: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Invert :func:`ar_coefficients` for the coefficients OASIS estimated.

    Parameters
    ----------
    g_hat : numpy.ndarray
        Estimated coefficients: shape ``(n_rois,)`` for AR1 or
        ``(n_rois, 2)`` for AR2.
    frame_rate : float
        Acquisition frame rate in Hz.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray or None)
        Estimated decay times, and estimated rise times for AR2 (``None``
        for AR1).
    """
    g_hat = np.asarray(g_hat)
    if g_hat.ndim == 1:
        return -1 / np.log(g_hat) / frame_rate, None
    half = np.sqrt(g_hat[:, 0] ** 2 + 4 * g_hat[:, 1]) / 2
    tau_hat = -1 / np.log(g_hat[:, 0] / 2 + half) / frame_rate
    tau_rise_hat = -1 / np.log(g_hat[:, 0] / 2 - half) / frame_rate
    return tau_hat, tau_rise_hat


def resolve_parameters(settings: OasisSettings, frame_rate: float) -> dict:
    """Resolve the parameter set passed to OASIS for this run.

    Maps the settings field names onto the keyword names OASIS expects
    (``baseline`` -> ``b``, ``baseline_nonneg`` -> ``b_nonneg``,
    ``optimize_tau`` -> ``optimize_g``) and applies the
    ``estimate_parameters=False`` fallbacks.

    Parameters
    ----------
    settings : OasisSettings
        Parsed run settings.
    frame_rate : float
        Acquisition frame rate in Hz.

    Returns
    -------
    dict
        Keys of :data:`DECONVOLVE_KEYS` plus ``s_min`` and ``lam``.

    Raises
    ------
    ValueError
        If ``estimate_parameters`` is False and no ``tau`` was provided —
        OASIS cannot run without either an explicit decay time or the
        freedom to estimate one.
    """
    lam = settings.lam
    baseline = settings.baseline

    if not settings.estimate_parameters:
        if settings.tau is None:
            raise ValueError(
                "estimate_parameters is False but no decay time constant "
                "'tau' was provided; pass --tau or enable "
                "--estimate_parameters."
            )
        if lam is None:
            lam = 0
            logger.info(
                "estimate_parameters is False and no sparsity penalty 'lam' "
                "was provided; setting lam to 0."
            )
        if baseline is None:
            baseline = 0
            logger.info(
                "estimate_parameters is False and no baseline was provided; "
                "setting baseline to 0."
            )

    return {
        "g": ar_coefficients(settings.tau, settings.tau_rise, frame_rate),
        "sn": settings.sn,
        "b": baseline,
        "b_nonneg": settings.baseline_nonneg,
        "penalty": settings.penalty,
        "optimize_g": settings.optimize_tau,
        "lam": lam,
        "s_min": settings.s_min,
    }


def deconvolve_kwargs(params: dict) -> dict:
    """Select the subset of ``params`` that ``deconvolve`` accepts.

    Parameters
    ----------
    params : dict
        Output of :func:`resolve_parameters`.

    Returns
    -------
    dict
        Only the :data:`DECONVOLVE_KEYS` entries.
    """
    return {key: params[key] for key in DECONVOLVE_KEYS}
