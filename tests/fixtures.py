"""Minimal v1 and v2 metadata fixtures shared by the tests.

Deliberately minimal: only the fields the getters actually read, so the
tests prove the library tolerates messy-but-sufficient metadata rather than
requiring a fully-validating document. The v2 discriminator strings are
resolved from the schema classes, matching how the shared read layer works.

Lives in ``tests/`` because ``[tool.coverage.run] source`` includes
``tests``; keep it exercised by the tests that import it.
"""

from typing import Optional

from aind_pophys_metadata.io import object_type_value

FRAME_RATE = 9.48
SUBJECT_ID = "783992"
DATASET_NAME = "multiplane-ophys_783992_2026-07-30_12-00-00"


def v1_session(
    frame_rate: float = FRAME_RATE,
    subject_id: Optional[str] = SUBJECT_ID,
) -> dict:
    """Return a minimal v1 ``session.json`` dict.

    Parameters
    ----------
    frame_rate : float, optional
        Frame rate placed on the first ophys FOV.
    subject_id : str, optional
        Subject identifier; pass ``None`` to omit the key.

    Returns
    -------
    dict
        The raw session dict.
    """
    session = {
        "rig_id": "MESO.1",
        "data_streams": [
            {
                "ophys_fovs": [
                    {
                        "index": 0,
                        "targeted_structure": {"acronym": "VISp"},
                        "frame_rate": frame_rate,
                    }
                ]
            }
        ],
    }
    if subject_id is not None:
        session["subject_id"] = subject_id
    return session


def v1_session_without_frame_rate() -> dict:
    """Return a v1 session whose FOV declares no frame rate."""
    session = v1_session()
    del session["data_streams"][0]["ophys_fovs"][0]["frame_rate"]
    return session


def v2_acquisition(
    frame_rate: float = FRAME_RATE,
    subject_id: Optional[str] = SUBJECT_ID,
) -> dict:
    """Return a minimal v2 ``acquisition.json`` dict.

    Parameters
    ----------
    frame_rate : float, optional
        Frame rate placed on the imaging config's sampling strategy.
    subject_id : str, optional
        Subject identifier; pass ``None`` to omit the key.

    Returns
    -------
    dict
        The raw acquisition dict.
    """
    from aind_data_schema.components.configs import ImagingConfig, PlanarImage

    acquisition = {
        "instrument_id": "MESO.1",
        "data_streams": [
            {
                "configurations": [
                    {
                        "object_type": object_type_value(ImagingConfig),
                        "sampling_strategy": {"frame_rate": frame_rate},
                        "images": [
                            {
                                "object_type": object_type_value(PlanarImage),
                                "planes": [
                                    {
                                        "plane_index": 0,
                                        "targeted_structure": {"acronym": "VISp"},
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ],
    }
    if subject_id is not None:
        acquisition["subject_id"] = subject_id
    return acquisition


def data_description(name: str = DATASET_NAME) -> dict:
    """Return a minimal ``data_description.json`` dict."""
    return {"name": name}
