"""Tests for the event-detection stage body over real compute results.

The traces arrive from an HDF5 file written the way the dF/F step writes
it, so deconvolution, the NaN count, the HDF5 writer and the per-ROI QC
figures all run against numpy arrays at production shapes.
"""

import tempfile
import unittest
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from aind_ophys_oasis_event_detection_library import events, job
from aind_ophys_oasis_event_detection_library.params import resolve_parameters
from aind_ophys_oasis_event_detection_library.settings import OasisSettings
from aind_ophys_oasis_event_detection_library.utils.metadata_utils import OasisMetadata

from .test_events import FRAME_RATE, N_FRAMES, N_ROIS, _settings, synthetic_dff_traces

EXPERIMENT_ID = "plane_0"


def _plan(root: Path, traces: np.ndarray, settings: OasisSettings) -> job.RunPlan:
    """Write ``traces`` as an upstream dF/F file and build the run plan."""
    dff_dir = root / "input" / EXPERIMENT_ID / "dff"
    dff_dir.mkdir(parents=True)
    dff_fp = dff_dir / f"{EXPERIMENT_ID}_dff.h5"
    with h5py.File(dff_fp, "w") as f:
        f.create_dataset("data", data=traces)
    params = resolve_parameters(settings, FRAME_RATE)
    return job.RunPlan(
        dff_fp=dff_fp,
        experiment_id=EXPERIMENT_ID,
        output_dir=events.make_output_directory(root / "results", EXPERIMENT_ID),
        oasis_params=params,
        parameters=params,
        input_data=[str(dff_fp.name)],
    )


class TestRunEventDetection(unittest.TestCase):
    """The stage body, from the upstream read to quality_control.json."""

    def _run(self, traces: np.ndarray, tmp: str, **overrides: Any) -> dict:
        """Run the stage body over ``traces`` and return its summary."""
        settings = _settings(estimate_parameters=True, **overrides)
        plan = _plan(Path(tmp), traces, settings)
        return job._run_event_detection(settings, OasisMetadata(frame_rate=FRAME_RATE), plan)

    def test_populated_plane(self) -> None:
        """Real traces flow through OASIS, the HDF5 writer and the plots."""
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._run(synthetic_dff_traces(), tmp)
            output_dir = Path(tmp) / "results" / EXPERIMENT_ID / "events"
            with h5py.File(output_dir / f"{EXPERIMENT_ID}_events_oasis.h5", "r") as f:
                self.assertEqual(f["events"].shape, (N_ROIS, N_FRAMES))
                self.assertEqual(f["tau_hat"].shape, (N_ROIS,))
            plots = sorted((output_dir / events.PLOTS_SUBDIR).glob("*.png"))
            self.assertEqual(len(plots), N_ROIS)
            self.assertTrue((output_dir / "quality_control.json").exists())
        self.assertEqual(summary["num_rois"], N_ROIS)
        self.assertEqual(summary["num_frames"], N_FRAMES)
        self.assertEqual(summary["num_nan_samples"], 0)

    def test_nan_samples_are_counted(self) -> None:
        """A dropped ROI is reported rather than silently deconvolved."""
        traces = synthetic_dff_traces()
        traces[0, :3] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._run(traces, tmp, no_qc=True)
        self.assertEqual(summary["num_nan_samples"], 3)

    def test_zero_roi_plane(self) -> None:
        """An empty (0, T) matrix writes empty datasets and skips the plots."""
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._run(np.empty((0, N_FRAMES)), tmp)
            output_dir = Path(tmp) / "results" / EXPERIMENT_ID / "events"
            with h5py.File(output_dir / f"{EXPERIMENT_ID}_events_oasis.h5", "r") as f:
                self.assertEqual(f["events"].shape, (0, N_FRAMES))
                self.assertEqual(f["b_hat"].shape, (0,))
            self.assertFalse((output_dir / events.PLOTS_SUBDIR).exists())
            self.assertTrue((output_dir / "quality_control.json").exists())
        self.assertEqual(summary["num_rois"], 0)


if __name__ == "__main__":
    unittest.main()
