"""Tests for quality_control.json generation."""

import json
import tempfile
import unittest
from pathlib import Path

from aind_data_schema.core.quality_control import (
    CurationMetric,
    QualityControl,
)

from aind_ophys_oasis_event_detection_library import qc

EXPERIMENT_ID = "plane_0"


class TestRoiPlotReferences(unittest.TestCase):
    """The per-ROI figure references carried as the curation value."""

    def test_one_entry_per_roi(self) -> None:
        """One reference entry is produced per ROI."""
        refs = qc.roi_plot_references(EXPERIMENT_ID, 3)
        self.assertEqual(len(refs), 3)
        self.assertEqual([r["roi_id"] for r in refs], ["0", "1", "2"])

    def test_reference_path_is_asset_relative(self) -> None:
        """References resolve from the asset root, not the output dir."""
        (ref,) = qc.roi_plot_references(EXPERIMENT_ID, 1)
        self.assertEqual(
            ref["reference"],
            "plane_0/events/plots/plane_0_0_oasis.png",
        )

    def test_zero_rois(self) -> None:
        """No ROIs yields an empty reference list, not an error."""
        self.assertEqual(qc.roi_plot_references(EXPERIMENT_ID, 0), [])


class TestEventsCurationMetric(unittest.TestCase):
    """The core CurationMetric replacing the v1 qcportal shape."""

    def test_is_core_curation_metric(self) -> None:
        """The metric is the core class, not the removed qcportal one."""
        metric = qc.build_events_curation_metric(EXPERIMENT_ID, 2)
        self.assertIsInstance(metric, CurationMetric)

    def test_required_v2_fields_populated(self) -> None:
        """``type`` and a list ``value`` are set, as v2 requires."""
        metric = qc.build_events_curation_metric(EXPERIMENT_ID, 2)
        self.assertEqual(metric.type, qc.CURATION_TYPE)
        self.assertIsInstance(metric.value, list)
        self.assertEqual(len(metric.value), 2)

    def test_status_history_is_seeded(self) -> None:
        """An empty status_history breaks QualityControl status derivation."""
        metric = qc.build_events_curation_metric(EXPERIMENT_ID, 1)
        self.assertTrue(metric.status_history)

    def test_carries_evaluation_tag(self) -> None:
        """default_grouping is ["evaluation"], so the tag must be present."""
        metric = qc.build_events_curation_metric(EXPERIMENT_ID, 1)
        self.assertEqual(metric.tags["evaluation"], qc.EVALUATION)
        self.assertEqual(metric.tags["type"], qc.METRIC_TYPE)

    def test_name_includes_experiment_id(self) -> None:
        """The metric name is scoped to the plane."""
        metric = qc.build_events_curation_metric(EXPERIMENT_ID, 1)
        self.assertTrue(metric.name.startswith(EXPERIMENT_ID))


class TestQualityControlDocument(unittest.TestCase):
    """The full QualityControl document and its writer."""

    def test_build_uses_project_default_grouping(self) -> None:
        """The shared standard grouping is applied."""
        metric = qc.build_events_curation_metric(EXPERIMENT_ID, 1)
        document = qc.build_quality_control([metric])
        self.assertIsInstance(document, QualityControl)
        self.assertEqual(document.default_grouping, ["evaluation"])

    def test_write_and_round_trip(self) -> None:
        """The written file validates back into a core v2 QualityControl."""
        metric = qc.build_events_curation_metric(EXPERIMENT_ID, 2)
        document = qc.build_quality_control([metric])
        with tempfile.TemporaryDirectory() as tmp:
            written = qc.write_quality_control_json(document, Path(tmp))
            self.assertTrue(written.exists())
            payload = json.loads(written.read_text())
            self.assertIn("metrics", payload)
            reloaded = QualityControl.model_validate_json(written.read_text())
        self.assertIsInstance(reloaded.metrics[0], CurationMetric)


if __name__ == "__main__":
    unittest.main()
