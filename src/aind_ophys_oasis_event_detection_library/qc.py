"""quality_control.json generation for OASIS event detection.

Step-specific metric content over the shared ``aind_pophys_metadata``
QC builders; the ``QualityControl`` wrapper (with the project-standard
``default_grouping``) and the writer both live in the shared library.

The per-ROI curation payload is the **core**
``aind_data_schema.core.quality_control.CurationMetric``, placed directly
into ``QualityControl.metrics``: v2 removed ``QCEvaluation`` and qcportal no
longer exports a ``CurationMetric``. The core class requires ``type`` and a
list ``value``, and an empty ``status_history`` makes ``QualityControl``
raise ``IndexError`` while deriving its status, so one status is seeded.
"""

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

    Each entry points a reviewer at one ROI's dF/F + events figure, with
    the path relative to the asset root so the QC portal can resolve it.

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

    Seeded PASS by ``"Automated"`` — deconvolution ran. A reviewer curates
    individual ROIs from the referenced figures.

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
