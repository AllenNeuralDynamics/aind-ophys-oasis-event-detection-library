"""Tests for the v1/v2 metadata read layer and this step's Code block."""

import json
import tempfile
import unittest
from pathlib import Path

from aind_data_schema.components.identifiers import Code
from aind_pophys_metadata import processing as shared_processing

from aind_ophys_oasis_event_detection_library.utils import metadata_utils as mu
from tests import fixtures

PLANE_ID = "plane_0"
PROCESSING_JSON = "processing.json"


def _write(directory: Path, name: str, payload: dict) -> None:
    """Write ``payload`` as JSON into ``directory/name``."""
    (directory / name).write_text(json.dumps(payload))


def _upstream_document(*names: str) -> dict:
    """Return a minimal upstream processing.json payload."""
    return {"data_processes": [{"name": name} for name in names]}


class TestLoadOasisMetadata(unittest.TestCase):
    """Version dispatch is by file presence, and only frame rate is required."""

    def test_reads_v1_session(self) -> None:
        """A v1 session.json supplies frame rate, subject and dataset name."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", fixtures.v1_session())
            _write(root, "data_description.json", fixtures.data_description())
            metadata = mu.load_oasis_metadata(root)
        self.assertEqual(metadata.frame_rate, fixtures.FRAME_RATE)
        self.assertEqual(metadata.subject_id, fixtures.SUBJECT_ID)
        self.assertEqual(metadata.dataset_name, fixtures.DATASET_NAME)

    def test_reads_v2_acquisition(self) -> None:
        """A v2 acquisition.json supplies the same fields from v2 paths."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", fixtures.v2_acquisition())
            _write(root, "data_description.json", fixtures.data_description())
            metadata = mu.load_oasis_metadata(root)
        self.assertEqual(metadata.frame_rate, fixtures.FRAME_RATE)
        self.assertEqual(metadata.subject_id, fixtures.SUBJECT_ID)

    def test_optional_files_may_be_absent(self) -> None:
        """No data_description and no subject.json is still processable."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", fixtures.v2_acquisition())
            metadata = mu.load_oasis_metadata(root)
        self.assertIsNone(metadata.dataset_name)
        self.assertEqual(metadata.subject_id, fixtures.SUBJECT_ID)

    def test_subject_id_falls_back_to_subject_json(self) -> None:
        """subject.json is the fallback when the core file omits the id."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                "acquisition.json",
                fixtures.v2_acquisition(subject_id=None),
            )
            _write(root, "subject.json", {"subject_id": "NY129"})
            metadata = mu.load_oasis_metadata(root)
        self.assertEqual(metadata.subject_id, "NY129")

    def test_platform_json_frame_rate_fallback(self) -> None:
        """A v1 platform.json supplies the frame rate when the FOV omits it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                "session.json",
                fixtures.v1_session_without_frame_rate(),
            )
            _write(
                root,
                "platform.json",
                {"imaging_plane_groups": [{"acquisition_framerate_Hz": 11.0}]},
            )
            metadata = mu.load_oasis_metadata(root)
        self.assertEqual(metadata.frame_rate, 11.0)

    def test_cli_override_supplies_the_frame_rate(self) -> None:
        """An asset with no resolvable rate still runs from the CLI value."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                "session.json",
                fixtures.v1_session_without_frame_rate(),
            )
            metadata = mu.load_oasis_metadata(root, cli_frame_rate=7.5)
        self.assertEqual(metadata.frame_rate, 7.5)

    def test_metadata_frame_rate_wins_over_the_cli_override(self) -> None:
        """The override is a fallback, never a replacement."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", fixtures.v1_session())
            metadata = mu.load_oasis_metadata(root, cli_frame_rate=7.5)
        self.assertEqual(metadata.frame_rate, fixtures.FRAME_RATE)

    def test_missing_frame_rate_fails_fast(self) -> None:
        """No frame rate anywhere raises at load, naming file and version."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                "session.json",
                fixtures.v1_session_without_frame_rate(),
            )
            with self.assertRaises(ValueError) as ctx:
                mu.load_oasis_metadata(root)
        message = str(ctx.exception)
        self.assertIn("frame rate", message)
        self.assertIn("session.json", message)
        self.assertIn("v1", message)

    def test_no_core_file_raises(self) -> None:
        """Neither session.json nor acquisition.json is a hard error."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                mu.load_oasis_metadata(Path(tmp))

    def test_both_core_files_is_ambiguous(self) -> None:
        """Both core files present is ambiguous and must not be guessed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", fixtures.v1_session())
            _write(root, "acquisition.json", fixtures.v2_acquisition())
            with self.assertRaises(ValueError):
                mu.load_oasis_metadata(root)


class TestUpstreamProcessNames(unittest.TestCase):
    """Upstream processing.json documents are flat siblings."""

    def test_collects_and_deduplicates(self) -> None:
        """Every upstream name is collected once, in first-seen order."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "VISp_0" / "dff"
            second = root / "VISp_0" / "motion_correction"
            for directory in (first, second):
                directory.mkdir(parents=True)
            _write(first, PROCESSING_JSON, _upstream_document("A: dF/F estimation"))
            _write(
                second,
                PROCESSING_JSON,
                _upstream_document("A: dF/F estimation", "A: Video motion correction"),
            )
            names = mu.upstream_process_names(root)
        self.assertEqual(names, ["A: dF/F estimation", "A: Video motion correction"])

    def test_excludes_this_process(self) -> None:
        """A step can never become its own dependency."""
        own_name = f"{PLANE_ID}: {mu.PROCESS_TYPE.value}"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, PROCESSING_JSON, _upstream_document(own_name))
            self.assertEqual(mu.upstream_process_names(root, PLANE_ID), [])

    def test_no_upstream_documents_is_empty(self) -> None:
        """A task with no staged upstream document reports no dependencies."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(mu.upstream_process_names(Path(tmp)), [])


class TestBuildCode(unittest.TestCase):
    """The Code block identifying which artifact produced the outputs."""

    def test_url_is_the_backing_library_landing_page(self) -> None:
        """Code identity names the backing library, not the wrapper."""
        code = mu.build_code({})
        self.assertIsInstance(code, Code)
        self.assertEqual(
            str(code.url),
            f"https://github.com/AllenNeuralDynamics/{mu.LIBRARY_NAME}",
        )

    def test_version_is_the_installed_library_version(self) -> None:
        """The released library version is the only source of the version."""
        self.assertEqual(
            mu.build_code({}).version,
            shared_processing.library_version(mu.LIBRARY_NAME),
        )

    def test_parameters_and_input_data_recorded(self) -> None:
        """Run parameters and input provenance land on the Code block."""
        code = mu.build_code({"tau": 1.0}, ["plane_0/dff/plane_0_dff.h5"])
        self.assertEqual(code.parameters.model_dump()["tau"], 1.0)
        self.assertEqual(code.input_data[0].name, "plane_0/dff/plane_0_dff.h5")


if __name__ == "__main__":
    unittest.main()
