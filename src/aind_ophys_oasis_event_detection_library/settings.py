"""CLI/env settings for OASIS fluorescence event detection."""

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
        description="Frame rate in Hz, used when the metadata provides none.",
    )
    estimate_parameters: bool = Field(
        default=True,
        description=(
            "Estimate the model parameters, including lam, from the noise "
            "constraint; otherwise use the values provided."
        ),
    )
    tau: Optional[float] = Field(
        default=None,
        description=(
            "Exponential (1/e) decay time in seconds. Estimated from the data "
            "when unset; required when estimate_parameters is False."
        ),
    )
    tau_rise: float = Field(
        default=0,
        description=(
            "Exponential (1/e) rise time in seconds. 0 selects an AR1 model, "
            "any other value AR2."
        ),
    )
    optimize_tau: int = Field(
        default=0,
        description=(
            "Number of large, isolated events used to refine tau; 0 disables "
            "the refinement."
        ),
    )
    baseline: Optional[float] = Field(
        default=None,
        description=(
            "Fluorescence baseline. When unset it is optimized if "
            "estimate_parameters is True, else 0."
        ),
    )
    b_nonneg: bool = Field(
        default=True,
        description="Enforce a strictly non-negative baseline if True.",
    )
    sn: Optional[float] = Field(
        default=None,
        description=(
            "Noise standard deviation. Estimated from the power spectral "
            "density when unset."
        ),
    )
    penalty: int = Field(
        default=1,
        description="Sparsity penalty. 1: min |s|_1, 0: min |s|_0.",
    )
    lam: Optional[float] = Field(
        default=None,
        description=(
            "Sparsity penalty weight. When unset it is estimated from the "
            "noise constraint if estimate_parameters is True, else 0."
        ),
    )
    s_min: float = Field(
        default=0,
        description="Minimal non-zero activity per bin (minimal spike size).",
    )
    no_qc: bool = Field(
        default=False,
        description="Skip generation of the per-ROI QC plots.",
    )
