"""Tests for OasisSettings and the app-panel contract."""

import sys
import typing
import unittest
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import patch

from aind_ophys_oasis_event_detection_library.settings import OasisSettings


def _make_settings(
    argv: Optional[List[str]] = None,
    **kwargs: Any,
) -> OasisSettings:
    """Construct OasisSettings with a controlled sys.argv."""
    with patch.object(sys, "argv", ["run_capsule.py"] + list(argv or [])):
        return OasisSettings(**kwargs)


def _sample_cli_value(annotation: Any) -> str:
    """Return a valid ``--name=value`` payload for a field annotation."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        annotation = args[0]
    if annotation is bool:
        return "true"
    if annotation is int:
        return "2"
    if annotation is float:
        return "1.5"
    if annotation is Path:
        return "/tmp/oasis-test"
    return "sample"


class TestOasisSettingsDefaults(unittest.TestCase):
    """Default values for the run settings fields."""

    def test_defaults(self) -> None:
        """Each field carries its documented default."""
        settings = _make_settings()
        self.assertEqual(settings.input_dir, Path("../data/"))
        self.assertEqual(settings.output_dir, Path("../results/"))
        self.assertTrue(settings.estimate_parameters)
        self.assertIsNone(settings.tau)
        self.assertEqual(settings.tau_rise, 0)
        self.assertEqual(settings.optimize_tau, 0)
        self.assertIsNone(settings.baseline)
        self.assertTrue(settings.b_nonneg)
        self.assertIsNone(settings.sn)
        self.assertEqual(settings.penalty, 1)
        self.assertIsNone(settings.lam)
        self.assertEqual(settings.s_min, 0)
        self.assertFalse(settings.no_qc)

    def test_baseline_replaces_one_character_field(self) -> None:
        """The baseline field has a full name, not a one-character alias."""
        self.assertIn("baseline", OasisSettings.model_fields)
        self.assertNotIn("b", OasisSettings.model_fields)

    def test_bool_string_parsing(self) -> None:
        """Bool fields parse from their string forms."""
        settings = _make_settings(["--estimate_parameters=false"])
        self.assertFalse(settings.estimate_parameters)


class TestSampleCliValue(unittest.TestCase):
    """The helper the app-panel contract test drives every field with."""

    def test_covers_every_annotation_this_settings_class_uses(self) -> None:
        """Each supported annotation yields a parseable payload."""
        self.assertEqual(_sample_cli_value(bool), "true")
        self.assertEqual(_sample_cli_value(int), "2")
        self.assertEqual(_sample_cli_value(float), "1.5")
        self.assertTrue(_sample_cli_value(Path).startswith("/"))
        self.assertEqual(_sample_cli_value(str), "sample")

    def test_optional_unwraps_to_the_inner_type(self) -> None:
        """Optional[float] is driven as a float, not as a string."""
        self.assertEqual(_sample_cli_value(typing.Optional[float]), "1.5")


class TestAppPanelContract(unittest.TestCase):
    """Every field is drivable by a Code Ocean app panel."""

    def test_no_single_character_field_names(self) -> None:
        """No field name is one character long."""
        offenders = [name for name in OasisSettings.model_fields if len(name) == 1]
        self.assertEqual(offenders, [])

    def test_every_field_parses_as_named_argument(self) -> None:
        """Each field accepts the ``--name=value`` form the panel emits."""
        for name, field in OasisSettings.model_fields.items():
            with self.subTest(field=name):
                value = _sample_cli_value(field.annotation)
                settings = _make_settings([f"--{name}={value}"])
                self.assertIsNotNone(getattr(settings, name))

    def test_field_names_use_underscores(self) -> None:
        """pydantic-settings style: no hyphens in field names."""
        offenders = [name for name in OasisSettings.model_fields if "-" in name]
        self.assertEqual(offenders, [])

    def test_every_field_has_a_description(self) -> None:
        """A panel with no descriptions is unusable by an operator."""
        missing = [
            name for name, field in OasisSettings.model_fields.items() if not field.description
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
