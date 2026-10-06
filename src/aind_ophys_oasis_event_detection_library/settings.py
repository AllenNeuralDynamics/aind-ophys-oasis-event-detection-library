"""CLI/environment settings for OASIS fluorescence event detection.

Kept import-light (pydantic + stdlib only) so it stays cheap to import and
free of oasis / matplotlib / aind-data-schema side effects.

Every field name is at least two characters: pydantic-settings renders a
one-character field as a *short* option (``-b``), which a Code Ocean app
panel can never emit because it only produces ``--name=value``.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class OasisSettings(BaseSettings, cli_parse_args=True):
    """Settings for OASIS event detection using pydantic-settings."""

    input_dir: Path = Field(
        default=Path("../data/"),
        description="Directory containing the input dF/F traces.",
    )
    output_dir: Path = Field(
        default=Path("../results/"),
        description="Directory where output files will be saved.",
    )
    frame_rate: Optional[float] = Field(
        default=None,
        description=(
            "Acquisition frame rate in Hz, used only when neither the core "
            "metadata file nor platform.json supplies one. Event detection "
            "cannot convert the decay/rise times into auto-regressive "
            "coefficients without a frame rate."
        ),
    )
    estimate_parameters: bool = Field(
        default=True,
        description=(
            "Whether to estimate parameters, in particular the sparsity "
            "parameter lam, using the noise constraint, or to use the "
            "parameters provided explicitly."
        ),
    )
    tau: Optional[float] = Field(
        default=None,
        description=(
            "Exponential decay time in seconds (1/e, thus equal to half-life "
            "time divided by ln(2)). Estimated from the autocovariance of the "
            "data if no value is given. Has to be provided explicitly if "
            "estimate_parameters is False."
        ),
    )
    tau_rise: float = Field(
        default=0,
        description=(
            "Exponential rise time in seconds (1/e, thus equal to half-rise "
            "time divided by ln(2)). Zero selects an AR1 model; any other "
            "value selects AR2."
        ),
    )
    optimize_tau: int = Field(
        default=0,
        description=(
            "Number of large, isolated events to consider for further "
            "optimizing tau. No optimization if optimize_tau is 0."
        ),
    )
    baseline: Optional[float] = Field(
        default=None,
        description=(
            "Fluorescence baseline value. If no value is given, the baseline "
            "is optimized when estimate_parameters is True, else 0."
        ),
    )
    baseline_nonneg: bool = Field(
        default=True,
        description="Enforce a strictly non-negative baseline if True.",
    )
    sn: Optional[float] = Field(
        default=None,
        description=(
            "Standard deviation of the noise distribution. If no value is "
            "given, sn is estimated from the data based on power spectral "
            "density."
        ),
    )
    penalty: int = Field(
        default=1,
        description="Sparsity penalty. 1: min |s|_1, 0: min |s|_0.",
    )
    lam: Optional[float] = Field(
        default=None,
        description=(
            "Sparsity penalty parameter. If no value is given, lam is "
            "estimated as the optimal Lagrange multiplier for the noise "
            "constraint under L1 penalty when estimate_parameters is True, "
            "else 0."
        ),
    )
    s_min: float = Field(
        default=0,
        description=("Minimal non-zero activity within each bin (minimal spike size)."),
    )
    no_qc: bool = Field(
        default=False,
        description="Skip generation of the per-ROI QC plots.",
    )
