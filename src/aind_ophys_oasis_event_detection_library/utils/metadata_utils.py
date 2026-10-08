"""Acquisition metadata and Code identity for OASIS event detection."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from aind_data_schema.components.identifiers import Code
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import CoreMetadata
from aind_pophys_metadata import processing as shared_processing

logger = logging.getLogger(__name__)

PROCESS_NAME = "aind-ophys-oasis-event-detection"
PROCESS_TYPE = ProcessName.FLUORESCENCE_EVENT_DETECTION


@dataclass(frozen=True)
class OasisMetadata:
    """Minimal metadata OASIS event detection consumes.

    Only ``frame_rate`` is required; the identifiers are optional and used
    for logging context.

    Attributes
    ----------
    frame_rate : float
        Acquisition frame rate in Hz.
    subject_id : str or None
        Subject identifier, for logging context.
    dataset_name : str or None
        Derived asset name, for logging context.
    """

    frame_rate: float
    subject_id: Optional[str] = None
    dataset_name: Optional[str] = None


def load_oasis_metadata(
    input_dir: Path,
    cli_frame_rate: Optional[float] = None,
) -> OasisMetadata:
    """Read the acquisition metadata event detection consumes from ``input_dir``.

    The shared loader locates and reads the core file; the frame rate falls
    back to ``platform.json`` and then to ``cli_frame_rate``.

    Parameters
    ----------
    input_dir : Path
        Directory searched recursively for the metadata files.
    cli_frame_rate : float, optional
        Operator-supplied frame rate, used only when no metadata source
        provides one.

    Returns
    -------
    OasisMetadata
        The minimal fields this step consumes.

    Raises
    ------
    FileNotFoundError
        If no core metadata file is found.
    ValueError
        If the input is ambiguous or malformed, or if no source supplies a
        frame rate.
    """
    core = CoreMetadata.load(Path(input_dir))
    return OasisMetadata(
        frame_rate=core.get_frame_rate(cli_override=cli_frame_rate, required=True),
        subject_id=core.get_subject_id(),
        dataset_name=core.get_dataset_name(),
    )


def upstream_process_names(
    input_dir: Path,
    plane_id: Optional[str] = None,
) -> List[str]:
    """Collect the upstream process names for the dependency graph.

    Every ``processing.json`` staged into this task is a flat sibling
    upstream: no ordering is inferred and no parent is distinguished from a
    grandparent.

    Parameters
    ----------
    input_dir : Path
        Input directory, searched recursively.
    plane_id : str, optional
        Plane id this task ran on; this process's own name is excluded.

    Returns
    -------
    list of str
        Deduplicated upstream process names.
    """
    own_name = PROCESS_TYPE.value
    if plane_id:
        own_name = f"{plane_id}: {own_name}"
    return shared_processing.collect_upstream_process_names(input_dir, exclude=own_name)


def build_code(
    parameters: dict,
    input_data: Optional[List[str]] = None,
) -> Code:
    """Construct the v2 ``Code`` block for this step.

    Parameters
    ----------
    parameters : dict
        Resolved run parameters recorded on the code block.
    input_data : list of str, optional
        Input data path(s)/asset name(s) for provenance.

    Returns
    -------
    Code
        The populated code block; name, version and url are this library's
        installed package metadata.
    """
    return shared_processing.build_code(
        __name__,
        parameters=parameters,
        input_data=input_data,
    )
