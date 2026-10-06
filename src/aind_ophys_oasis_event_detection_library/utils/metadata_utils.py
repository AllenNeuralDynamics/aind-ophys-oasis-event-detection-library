"""aind-data-schema v1/v2 metadata reads for OASIS event detection.

A thin, step-specific layer over the shared ``aind_pophys_metadata``
library: the v1/v2 dict paths, schema-version dispatch, the frame-rate
resolution order and the ``Processing`` builders all live in the shared
library, so this module only assembles the ``OasisMetadata`` view and this
step's ``Code`` block.
"""

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
LIBRARY_NAME = "aind-ophys-oasis-event-detection-library"
LIBRARY_URL = "https://github.com/AllenNeuralDynamics/aind-ophys-oasis-event-detection-library"
PROCESS_TYPE = ProcessName.FLUORESCENCE_EVENT_DETECTION


@dataclass(frozen=True)
class OasisMetadata:
    """Minimal metadata OASIS event detection consumes.

    ``frame_rate`` is the only field the computation needs — it converts the
    configured decay/rise times into auto-regressive coefficients. The other
    fields are identifiers used for logging context, so all stay optional (a
    partial-metadata asset must still be processable).

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
    back to ``platform.json`` and then to ``cli_frame_rate``. A missing frame
    rate is fatal here rather than deep in the run: event detection cannot
    convert time constants into auto-regressive coefficients without one.

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
        Plane id this task ran on, used to compose this process's own name
        so it can never appear as its own dependency.

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
        Resolved run parameters (the inputs to this step). Output summary
        statistics belong in ``DataProcess.output_parameters``, never here.
    input_data : list of str, optional
        Input data path(s)/asset name(s) for provenance.

    Returns
    -------
    Code
        The populated code block.
    """
    return shared_processing.build_code(
        name=PROCESS_NAME,
        library_name=LIBRARY_NAME,
        url=LIBRARY_URL,
        parameters=parameters,
        input_data=input_data,
    )
