"""quality_control.json content for OASIS event detection."""

from datetime import datetime as dt
from pathlib import Path
from typing import List

import pytz
from aind_data_schema.core.quality_control import (
    CurationMetric,
    QCMetric,
    QCStatus,
    QualityControl,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_pophys_metadata import quality_control as shared_qc

_SEATTLE_TZ = pytz.timezone("America/Los_Angeles")

EVALUATION = "Events"
METRIC_TYPE = "Operational QC"
CURATION_TYPE = "ROI event traces"


def roi_plot_references(
    experiment_id: str,
    num_rois: int,
) -> List[dict]:
    """Build the per-ROI plot reference payload for the curation metric.

    Each entry references one ROI's dF/F and events figure, relative to the
    asset root.

    Parameters
    ----------
    experiment_id : str
        Plane/experiment identifier.
    num_rois : int
        Number of ROIs deconvolved.

    Returns
    -------
    list of dict
        One ``{"roi_id": ..., "reference": ...}`` entry per ROI.
    """
    return [
        {
            "roi_id": str(roi_id),
            "reference": (f"{experiment_id}/events/plots/" f"{experiment_id}_{roi_id}_oasis.png"),
        }
        for roi_id in range(num_rois)
    ]


def build_events_curation_metric(
    experiment_id: str,
    num_rois: int,
) -> CurationMetric:
    """Construct the per-ROI event-detection curation metric.

    Seeded PASS by ``"Automated"``; a reviewer curates individual ROIs from
    the referenced figures.

    Parameters
    ----------
    experiment_id : str
        Plane/experiment identifier used in the metric name.
    num_rois : int
        Number of ROIs deconvolved; one reference entry is created per ROI.

    Returns
    -------
    CurationMetric
        The event-detection curation metric.
    """
    return CurationMetric(
        name=f"{experiment_id} Event Detection",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        type=CURATION_TYPE,
        value=roi_plot_references(experiment_id, num_rois),
        description="dF/F and per-ROI events detected by OASIS",
        tags={"evaluation": EVALUATION, "type": METRIC_TYPE},
        status_history=[
            QCStatus(
                evaluator="Automated",
                status=Status.PASS,
                timestamp=dt.now(_SEATTLE_TZ).isoformat(),
            )
        ],
    )


def build_quality_control(metrics: List[QCMetric]) -> QualityControl:
    """Wrap metrics into a full QualityControl with the standard grouping."""
    return shared_qc.build_quality_control(metrics)


def write_quality_control_json(
    quality_control: QualityControl,
    output_dir: Path,
) -> Path:
    """Serialize a QualityControl object to ``quality_control.json``.

    Parameters
    ----------
    quality_control : QualityControl
        The QC document to write.
    output_dir : Path
        Directory into which ``quality_control.json`` is written.

    Returns
    -------
    Path
        The path written.
    """
    return shared_qc.write_quality_control_json(quality_control, output_dir)
