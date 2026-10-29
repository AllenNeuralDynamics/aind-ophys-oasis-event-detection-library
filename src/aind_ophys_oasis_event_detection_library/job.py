"""OASIS event-detection entry point."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from aind_pophys_metadata import relative_to_root
from aind_pophys_metadata.runner import stage_guard
from log_schema.main import setup_logging

from . import events, qc
from .params import resolve_parameters
from .settings import OasisSettings
from .utils import metadata_utils as mu

logger = logging.getLogger("Ophys event detection")

PROCESS_NAME = mu.PROCESS_NAME


@dataclass(frozen=True)
class RunPlan:
    """Everything resolved from the inputs before any compute starts.

    Attributes
    ----------
    dff_fp : Path
        The upstream dF/F HDF5 file.
    experiment_id : str
        Plane id, recovered from the upstream directory layout.
    output_dir : Path
        This plane's ``events`` output directory.
    oasis_params : dict
        Parameters handed to OASIS (see :func:`~.params.resolve_parameters`).
    parameters : dict
        The settings plus the OASIS parameters, recorded on
        ``code.parameters``.
    input_data : list of str
        Input provenance, relative to the input root.
    """

    dff_fp: Path
    experiment_id: str
    output_dir: Path
    oasis_params: dict
    parameters: dict
    input_data: List[str]


def _worker_count() -> Optional[int]:
    """Return the Code Ocean CPU allocation, or ``None`` if unset."""
    co_cpus = os.environ.get("CO_CPUS")
    return int(co_cpus) if co_cpus else None


def _write_quality_control(output_dir: Path, experiment_id: str, num_rois: int) -> None:
    """Emit one full quality_control.json for this plane."""
    metric = qc.build_events_curation_metric(
        experiment_id=experiment_id,
        num_rois=num_rois,
    )
    qc.write_quality_control_json(qc.build_quality_control([metric]), output_dir)


def _plan_run(
    settings: OasisSettings,
    metadata: mu.OasisMetadata,
    input_dir: Path,
    results_root: Path,
) -> RunPlan:
    """Resolve inputs, plane id, output directory and OASIS parameters.

    Called before the stage guard, which needs the plane id and output path.

    Parameters
    ----------
    settings : OasisSettings
        Parsed run settings.
    metadata : OasisMetadata
        The metadata view for this asset.
    input_dir : Path
        Resolved input directory.
    results_root : Path
        Resolved results root.

    Returns
    -------
    RunPlan
        The resolved plan.
    """
    dff_fp, experiment_id = events.find_dff_traces(input_dir)
    output_dir = events.make_output_directory(results_root, experiment_id)
    params = resolve_parameters(settings, metadata.frame_rate)
    return RunPlan(
        dff_fp=dff_fp,
        experiment_id=experiment_id,
        output_dir=output_dir,
        oasis_params=params,
        parameters={**settings.model_dump(mode="json"), **params},
        input_data=[relative_to_root(input_dir, dff_fp)],
    )


def _run_event_detection(
    settings: OasisSettings,
    metadata: mu.OasisMetadata,
    plan: RunPlan,
) -> dict:
    """Deconvolve the upstream dF/F traces and write all outputs.

    Parameters
    ----------
    settings : OasisSettings
        Parsed run settings.
    metadata : OasisMetadata
        The metadata view for this asset.
    plan : RunPlan
        The resolved run plan.

    Returns
    -------
    dict
        Summary statistics of this step's outputs, written to
        ``DataProcess.output_parameters``.
    """
    traces = events.read_dff_traces(plan.dff_fp)
    num_rois, num_frames = traces.shape
    nan_count = int((traces != traces).sum())
    if nan_count:
        logger.info("Traces contain %s NaN samples in %s", nan_count, plan.experiment_id)

    result = events.run_deconvolution(
        traces=traces,
        params=plan.oasis_params,
        estimate_parameters=settings.estimate_parameters,
        tau_rise=settings.tau_rise,
        frame_rate=metadata.frame_rate,
        n_jobs=_worker_count(),
    )
    events.write_events_h5(plan.output_dir, plan.experiment_id, result)

    if num_rois and not settings.no_qc:
        events.write_qc_plots(
            output_dir=plan.output_dir,
            experiment_id=plan.experiment_id,
            traces=traces,
            result=result,
            params=plan.oasis_params,
            estimate_parameters=settings.estimate_parameters,
            tau=settings.tau,
            frame_rate=metadata.frame_rate,
            n_jobs=_worker_count(),
        )

    _write_quality_control(plan.output_dir, plan.experiment_id, num_rois)
    return {
        "num_rois": num_rois,
        "num_frames": num_frames,
        "num_nan_samples": nan_count,
    }


def run(settings: Optional[OasisSettings] = None) -> None:
    """Run OASIS fluorescence event detection end to end.

    Parameters
    ----------
    settings : OasisSettings, optional
        Parsed settings; constructed from CLI/env when omitted.
    """
    settings = settings or OasisSettings()
    metadata = mu.load_oasis_metadata(settings.input_dir, cli_frame_rate=settings.frame_rate)
    setup_logging(
        use_cloudwatch=True,
        model={
            "acquisition_name": metadata.dataset_name or metadata.subject_id,
            "process_name": PROCESS_NAME,
        },
    )
    results_root = settings.output_dir.resolve()
    input_dir = settings.input_dir.resolve()

    try:
        plan = _plan_run(settings, metadata, input_dir, results_root)
    except BaseException:
        # Plane not known yet: write the document at the results root.
        with stage_guard(
            results_root,
            mu.PROCESS_TYPE,
            mu.build_code(settings.model_dump(mode="json")),
            logger=logger,
        ):
            raise

    with stage_guard(
        plan.output_dir,
        mu.PROCESS_TYPE,
        mu.build_code(plan.parameters, plan.input_data),
        plane_id=plan.experiment_id,
        output_path=relative_to_root(results_root, plan.output_dir),
        logger=logger,
    ) as ctx:
        ctx.upstream_names = mu.upstream_process_names(input_dir, plan.experiment_id)
        ctx.output_parameters.update(_run_event_detection(settings, metadata, plan))
