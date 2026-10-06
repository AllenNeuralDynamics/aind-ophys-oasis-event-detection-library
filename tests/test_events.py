"""Tests for the upstream-artifact lookup, deconvolution and the outputs.

``events`` is coverage-omitted (heavy compute), but the upstream lookup is
tested anyway: it must fail with the pattern and directory named, never with
a bare ``next(glob(...))`` StopIteration that carries no context.

Every trace and every deconvolution result here is a numpy array at a
production shape. That is what the HDF5 writer, the ``[:, None]`` baseline
broadcast and the per-ROI plot fan-out actually receive; a list-shaped
fixture satisfies none of those operations while still passing the suite.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import h5py
import numpy as np

from aind_ophys_oasis_event_detection_library import events
from aind_ophys_oasis_event_detection_library.params import resolve_parameters
from aind_ophys_oasis_event_detection_library.settings import OasisSettings

EXPERIMENT_ID = "plane_0"
FRAME_RATE = 10.0
N_ROIS = 2
N_FRAMES = 500


def _settings(**kwargs: Any) -> OasisSettings:
    """Construct settings with a clean ``sys.argv``.

    ``cli_parse_args=True`` would otherwise consume the test runner's own
    arguments and abort with SystemExit.
    """
    with patch.object(sys, "argv", ["oasis.py"]):
        return OasisSettings(**kwargs)


def synthetic_dff_traces(n_rois: int = N_ROIS) -> np.ndarray:
    """Return dF/F traces of shape ``(n_rois, N_FRAMES)`` with calcium events.

    Parameters
    ----------
    n_rois : int, optional
        ROI count.

    Returns
    -------
    numpy.ndarray
        Float traces: exponentially decaying transients plus noise.
    """
    rng = np.random.default_rng(0)
    decay = np.exp(-1 / (0.5 * FRAME_RATE))
    traces = np.empty((n_rois, N_FRAMES))
    for roi in range(n_rois):
        spikes = rng.random(N_FRAMES) < 0.02
        calcium = np.zeros(N_FRAMES)
        for frame in range(1, N_FRAMES):
            calcium[frame] = decay * calcium[frame - 1] + spikes[frame]
        traces[roi] = calcium + rng.normal(0, 0.05, N_FRAMES)
    return traces


class TestFindDffTraces(unittest.TestCase):
    """The upstream dF/F lookup must fail with context, never StopIteration."""

    def test_locates_traces_and_experiment_id(self) -> None:
        """The experiment id comes from the dff directory's parent."""
        with tempfile.TemporaryDirectory() as tmp:
            dff_dir = Path(tmp) / EXPERIMENT_ID / "dff"
            dff_dir.mkdir(parents=True)
            expected = dff_dir / f"{EXPERIMENT_ID}_dff.h5"
            expected.touch()
            found, experiment_id = events.find_dff_traces(Path(tmp))
        self.assertEqual(found, expected)
        self.assertEqual(experiment_id, EXPERIMENT_ID)

    def test_missing_dff_directory_raises_with_context(self) -> None:
        """A missing dff directory names the pattern and the directory."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                events.find_dff_traces(Path(tmp))
        message = str(ctx.exception)
        self.assertIn(events.DFF_DIR_GLOB, message)
        self.assertIn(tmp, message)

    def test_missing_dff_file_raises_with_context(self) -> None:
        """An empty dff directory names the file pattern it looked for."""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / EXPERIMENT_ID / "dff").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError) as ctx:
                events.find_dff_traces(Path(tmp))
        self.assertIn(events.DFF_FILE_GLOB, str(ctx.exception))


class TestOutputDirectory(unittest.TestCase):
    """The per-plane output directory layout the pipeline globs depend on."""

    def test_creates_plane_events_subdirectory(self) -> None:
        """Output lands in ``<results>/<experiment_id>/events``."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = events.make_output_directory(Path(tmp), EXPERIMENT_ID)
            self.assertTrue(output_dir.is_dir())
            self.assertEqual(output_dir.relative_to(tmp), Path(EXPERIMENT_ID) / "events")


class TestEventsHdf5(unittest.TestCase):
    """The events HDF5 datasets NWB downstream reads."""

    def test_writes_core_datasets(self) -> None:
        """events and denoised are always written."""
        result = {
            "events": np.zeros((2, 5), dtype="f4"),
            "denoised": np.ones((2, 5), dtype="f4"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            written = events.write_events_h5(Path(tmp), EXPERIMENT_ID, result)
            with h5py.File(written, "r") as f:
                self.assertEqual(sorted(f.keys()), ["denoised", "events"])
        self.assertEqual(written.name, f"{EXPERIMENT_ID}_events_oasis.h5")

    def test_writes_estimated_parameter_datasets(self) -> None:
        """Estimated parameters are written only when present."""
        result = {
            "events": np.zeros((1, 3), dtype="f4"),
            "denoised": np.zeros((1, 3), dtype="f4"),
            "b_hat": np.zeros(1, dtype="f4"),
            "tau_hat": np.ones(1, dtype="f4"),
            "lam_hat": np.zeros(1, dtype="f4"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            written = events.write_events_h5(Path(tmp), EXPERIMENT_ID, result)
            with h5py.File(written, "r") as f:
                self.assertIn("tau_hat", f)
                self.assertNotIn("tau_rise_hat", f)


class TestRunDeconvolutionNoRois(unittest.TestCase):
    """The zero-ROI path must not spawn a pool or index empty arrays."""

    def test_empty_traces_ar1(self) -> None:
        """No ROIs yields zero-length arrays of the populated path's dtype."""
        result = events.run_deconvolution(
            traces=np.empty((0, 7), dtype="f4"),
            params={},
            estimate_parameters=True,
            tau_rise=0,
            frame_rate=10.0,
        )
        self.assertEqual(result["events"].shape, (0, 7))
        for key in ("b_hat", "tau_hat", "lam_hat"):
            self.assertEqual(result[key].shape, (0,))
            self.assertEqual(result[key].dtype, np.dtype("f4"))
        self.assertNotIn("tau_rise_hat", result)

    def test_empty_traces_ar2(self) -> None:
        """AR2 additionally reports a zero-length rise-time array."""
        result = events.run_deconvolution(
            traces=np.empty((0, 4), dtype="f4"),
            params={},
            estimate_parameters=True,
            tau_rise=0.2,
            frame_rate=10.0,
        )
        self.assertEqual(result["tau_rise_hat"].shape, (0,))

    def test_empty_estimates_write_and_plot_as_arrays(self) -> None:
        """The zero-ROI result stays usable by both output writers.

        ``write_events_h5`` stores the estimates as datasets and
        ``write_qc_plots`` indexes ``b_hat`` with ``[:, None]``, so these
        entries have to be arrays rather than lists.
        """
        result = events.run_deconvolution(
            traces=np.empty((0, 6), dtype="f4"),
            params={},
            estimate_parameters=True,
            tau_rise=0.2,
            frame_rate=10.0,
        )
        self.assertEqual(result["b_hat"][:, None].shape, (0, 1))
        with tempfile.TemporaryDirectory() as tmp:
            written = events.write_events_h5(Path(tmp), EXPERIMENT_ID, result)
            with h5py.File(written, "r") as f:
                self.assertEqual(f["tau_rise_hat"].shape, (0,))
                self.assertEqual(f["b_hat"].dtype, np.dtype("f4"))


class TestReadDffTraces(unittest.TestCase):
    """The upstream dF/F read, which hands the whole matrix to compute."""

    def test_reads_the_trace_matrix(self) -> None:
        """Traces come back as a numpy matrix of shape (n_rois, n_frames)."""
        traces = synthetic_dff_traces()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{EXPERIMENT_ID}_dff.h5"
            with h5py.File(path, "w") as f:
                f.create_dataset("data", data=traces)
            read = events.read_dff_traces(path)
        self.assertEqual(read.shape, (N_ROIS, N_FRAMES))
        np.testing.assert_allclose(read, traces)


class TestDeconvolveRealTraces(unittest.TestCase):
    """Deconvolution and QC plotting over real dF/F traces."""

    def test_estimated_parameters_round_trip_to_outputs(self) -> None:
        """OASIS estimates per-ROI parameters, and both writers consume them."""
        traces = synthetic_dff_traces()
        params = resolve_parameters(_settings(), FRAME_RATE)
        result = events.run_deconvolution(
            traces=traces,
            params=params,
            estimate_parameters=True,
            tau_rise=0,
            frame_rate=FRAME_RATE,
            n_jobs=1,
        )
        self.assertEqual(result["denoised"].shape, (N_ROIS, N_FRAMES))
        self.assertEqual(result["events"].shape, (N_ROIS, N_FRAMES))
        self.assertEqual(result["tau_hat"].shape, (N_ROIS,))
        with tempfile.TemporaryDirectory() as tmp:
            events.write_events_h5(Path(tmp), EXPERIMENT_ID, result)
            plots_path = events.write_qc_plots(
                output_dir=Path(tmp),
                experiment_id=EXPERIMENT_ID,
                traces=traces,
                result=result,
                params=params,
                estimate_parameters=True,
                tau=None,
                frame_rate=FRAME_RATE,
                n_jobs=1,
            )
            self.assertEqual(len(sorted(plots_path.glob("*.png"))), N_ROIS)

    def test_a_nan_trace_is_returned_as_nan(self) -> None:
        """A dropped ROI yields all-NaN output instead of raising."""
        trace = synthetic_dff_traces(1)[0]
        trace[10] = np.nan
        calcium, spikes, b_hat, g_hat, lam_hat = events.deconvolve_trace(
            trace,
            resolve_parameters(_settings(), FRAME_RATE),
            estimate_parameters=True,
            tau_rise=0,
        )
        self.assertTrue(np.isnan(calcium).all())
        self.assertTrue(np.isnan(spikes).all())
        self.assertTrue(np.isnan([b_hat, g_hat, lam_hat]).all())


if __name__ == "__main__":
    unittest.main()
