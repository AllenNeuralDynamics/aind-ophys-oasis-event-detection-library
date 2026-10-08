"""OASIS deconvolution, HDF5 output and per-ROI QC plots."""

from datetime import datetime as dt
import functools
import logging
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Optional, Tuple
from typing import Union

import h5py
import matplotlib
from aind_pophys_metadata import find_one

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from oasis.functions import deconvolve  # noqa: E402
from oasis.oasis_methods import (  # noqa: E402
    oasisAR1,
    oasisAR1_f32,
    oasisAR2,
)

from .params import (  # noqa: E402
    deconvolve_kwargs,
    taus_from_ar_coefficients,
)

logger = logging.getLogger(__name__)

DFF_DIR_GLOB = "*/dff/"
DFF_FILE_GLOB = "*dff.h5"
EVENTS_SUBDIR = "events"
PLOTS_SUBDIR = "plots"


def write_data_process(
    metadata: dict,
    input_fp: Union[str, Path],
    output_fp: Union[str, Path],
    unique_id: str,
    start_time: dt,
    end_time: dt,
) -> None:
    """Writes output metadata to plane processing.json

    Parameters
    ----------
    metadata: dict
        parameters from suite2p motion correction
    raw_movie: str
        path to raw movies
    output_fp: str
        path to motion corrected movies
    unique_id: str
        unique identifier for the processing
    start_time: dt
        start time of processing
    end_time: dt
        end time of processing
    """
    data_proc = DataProcess(
        name=ProcessName.FLUORESCENCE_EVENT_DETECTION,
        software_version=os.getenv("VERSION", ""),
        start_date_time=start_time.isoformat(),
        end_date_time=end_time.isoformat(),
        input_location=str(input_fp),
        output_location=str(output_fp),
        code_url=(os.getenv("REPO_URL", "")),
        parameters=metadata,
    )
    if isinstance(output_fp, str):
        output_dir = Path(output_fp).parent
    else:
        output_dir = output_fp.parent
    with open(output_dir / f"{unique_id}_oasis_events_data_process.json", "w") as f:
        json.dump(json.loads(data_proc.model_dump_json()), f, indent=4)


def find_dff_traces(input_dir: Path) -> Tuple[Path, str]:
    """Locate the upstream dF/F traces and derive the experiment id.

    The dF/F step publishes ``<experiment_id>/dff/<experiment_id>_dff.h5``.

    Parameters
    ----------
    input_dir : Path
        Input directory (the mounted upstream results).

    Returns
    -------
    tuple of (Path, str)
        Path to the dF/F HDF5 file, and the experiment id.

    Raises
    ------
    FileNotFoundError
        If no ``dff`` directory or no dF/F HDF5 file is found.
    """
    input_dir = Path(input_dir)
    dff_dir = find_one(input_dir, DFF_DIR_GLOB)
    dff_fp = find_one(dff_dir, DFF_FILE_GLOB, recursive=False)
    return dff_fp, dff_dir.parent.name


def make_output_directory(output_dir: Path, experiment_id: str) -> Path:
    """Create and return ``<output_dir>/<experiment_id>/events``.

    Parameters
    ----------
    output_dir : Path
        Results root.
    experiment_id : str
        Plane/experiment identifier.

    Returns
    -------
    Path
        The created output directory.
    """
    output_dir = Path(output_dir) / experiment_id / EVENTS_SUBDIR
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def read_dff_traces(dff_fp: Path) -> np.ndarray:
    """Read the dF/F trace matrix from the upstream HDF5 file.

    Parameters
    ----------
    dff_fp : Path
        Path to the dF/F HDF5 file.

    Returns
    -------
    numpy.ndarray
        Traces of shape ``(n_rois, n_frames)``.
    """
    with h5py.File(dff_fp, "r") as f:
        return f["data"][:]


def deconvolve_trace(
    trace: np.ndarray,
    params: dict,
    estimate_parameters: bool,
    tau_rise: float,
) -> tuple:
    """Deconvolve one dF/F trace into denoised calcium and events.

    A trace containing any NaN is returned as all-NaN.

    Parameters
    ----------
    trace : numpy.ndarray
        One ROI's dF/F trace.
    params : dict
        Resolved parameters from
        :func:`~.params.resolve_parameters`.
    estimate_parameters : bool
        Whether OASIS estimates the model parameters.
    tau_rise : float
        Rise time; ``0`` selects AR1, anything else AR2.

    Returns
    -------
    tuple
        ``(calcium, events, b_hat, g_hat, lam_hat)`` when estimating,
        else ``(calcium, events)``.
    """
    if np.isnan(trace).any():
        nan_like = np.full_like(trace, np.nan)
        empty = (nan_like, nan_like)
        return empty + (np.nan, np.nan, np.nan) if estimate_parameters else empty

    if estimate_parameters:
        return deconvolve(trace, **deconvolve_kwargs(params))

    baseline = params["b"]
    coefficients = params["g"]
    if tau_rise == 0:
        solver = oasisAR1_f32 if trace.dtype == np.float32 else oasisAR1
        return solver(
            trace - baseline,
            coefficients[0],
            s_min=params["s_min"],
            lam=params["lam"],
        )
    return oasisAR2(
        trace.astype(float) - baseline,
        coefficients[0],
        coefficients[1],
        s_min=params["s_min"],
        lam=params["lam"],
    )


def run_deconvolution(
    traces: np.ndarray,
    params: dict,
    estimate_parameters: bool,
    tau_rise: float,
    frame_rate: float,
    n_jobs: Optional[int] = None,
) -> dict:
    """Deconvolve every trace in parallel and assemble the result arrays.

    Parameters
    ----------
    traces : numpy.ndarray
        dF/F traces of shape ``(n_rois, n_frames)``.
    params : dict
        Resolved parameters from :func:`~.params.resolve_parameters`.
    estimate_parameters : bool
        Whether OASIS estimates the model parameters.
    tau_rise : float
        Rise time; ``0`` selects AR1, anything else AR2.
    frame_rate : float
        Acquisition frame rate in Hz, used to invert estimated
        coefficients back into time constants.
    n_jobs : int, optional
        Worker count; ``None`` lets multiprocessing choose.

    Returns
    -------
    dict
        ``denoised`` and ``events``, both ``f4`` arrays of shape
        ``(n_rois, n_frames)``, and, when estimating, the ``f4`` per-ROI
        arrays ``b_hat``, ``tau_hat``, ``lam_hat`` and (AR2 only)
        ``tau_rise_hat``, each of shape ``(n_rois,)``. A zero-ROI plane
        returns the same keys at length 0.
    """
    n_rois, n_frames = traces.shape
    if not n_rois:
        result = {
            "denoised": np.empty((0, n_frames), dtype="f4"),
            "events": np.empty((0, n_frames), dtype="f4"),
        }
        if estimate_parameters:
            # f4 arrays, not lists: the HDF5 writer stores the dtype and the
            # QC plotter indexes b_hat with [:, None].
            empty = np.empty((0,), dtype="f4")
            result.update(b_hat=empty.copy(), tau_hat=empty.copy(), lam_hat=empty.copy())
            if tau_rise != 0:
                result["tau_rise_hat"] = empty.copy()
        return result

    work = functools.partial(
        deconvolve_trace,
        params=params,
        estimate_parameters=estimate_parameters,
        tau_rise=tau_rise,
    )
    with Pool(n_jobs) as pool:
        per_roi = pool.map(work, traces)

    denoised, events = (np.array([roi[i] for roi in per_roi], dtype="f4") for i in (0, 1))
    result = {"denoised": denoised, "events": events}
    if estimate_parameters:
        b_hat, g_hat, lam_hat = (
            np.array([roi[i] for roi in per_roi], dtype="f4") for i in (2, 3, 4)
        )
        tau_hat, tau_rise_hat = taus_from_ar_coefficients(g_hat, frame_rate)
        result.update(b_hat=b_hat, tau_hat=tau_hat, lam_hat=lam_hat)
        if tau_rise_hat is not None:
            result["tau_rise_hat"] = tau_rise_hat
    return result


def write_events_h5(
    output_dir: Path,
    experiment_id: str,
    result: dict,
) -> Path:
    """Write the OASIS output HDF5 file.

    Parameters
    ----------
    output_dir : Path
        Directory to write into.
    experiment_id : str
        Plane/experiment identifier used in the filename.
    result : dict
        Output of :func:`run_deconvolution`.

    Returns
    -------
    Path
        Path to the written HDF5 file.
    """
    events_h5 = Path(output_dir) / f"{experiment_id}_events_oasis.h5"
    with h5py.File(events_h5, "w") as f:
        f.create_dataset("events", data=result["events"], compression="gzip")
        f.create_dataset("denoised", data=result["denoised"], compression="gzip")
        for key in ("b_hat", "tau_hat", "tau_rise_hat", "lam_hat"):
            if key in result:
                f.create_dataset(key, data=result[key])
    return events_h5


def plot_trace_and_events_png(
    trace: np.ndarray,
    calcium: np.ndarray,
    spikes: np.ndarray,
    timestamps: np.ndarray,
    roi_id: int,
    tau: float,
    plots_path: Path,
    experiment_id: str,
) -> Path:
    """Write one ROI's dF/F, denoised trace and event-rate figure.

    Parameters
    ----------
    trace : numpy.ndarray
        Raw dF/F trace.
    calcium : numpy.ndarray
        Denoised calcium trace (baseline added back).
    spikes : numpy.ndarray
        Detected event rate.
    timestamps : numpy.ndarray
        Frame timestamps in seconds.
    roi_id : int
        ROI index, used in the title and filename.
    tau : float
        Decay time constant shown in the legend.
    plots_path : Path
        Directory to write the PNG into.
    experiment_id : str
        Plane/experiment identifier used in the filename.

    Returns
    -------
    Path
        Path to the written PNG.
    """
    sns.set_context("talk")
    fig, ax = plt.subplots(2, 1, figsize=(20, 5), sharex=True)
    ax[0].plot(timestamps, 100 * trace, color="C0", label=r"raw $\Delta$F/F")
    ax[0].plot(timestamps, 100 * calcium, color="C1", label="denoised")
    end = min(580, timestamps[-1])  # arbitrary window to eyeball
    ax[0].set_xlim(max(0, end - 180), end)
    ax[0].legend()
    ax[0].set_ylabel(r"$\Delta$F/F [%]")
    ax[0].set_title(f"cell_roi_id: {roi_id}")
    ax[1].plot(timestamps, spikes, color="C2", label=f"events, tau={tau:.4f}s")
    ax[1].legend()
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Spike rate [a.u.]")
    plt.tight_layout(pad=0.2)
    png = Path(plots_path) / f"{experiment_id}_{roi_id}_oasis.png"
    fig.savefig(png)
    plt.close(fig)
    return png


def write_qc_plots(
    output_dir: Path,
    experiment_id: str,
    traces: np.ndarray,
    result: dict,
    params: dict,
    estimate_parameters: bool,
    tau: Optional[float],
    frame_rate: float,
    n_jobs: Optional[int] = None,
) -> Path:
    """Write one QC figure per ROI into ``<output_dir>/plots``.

    Parameters
    ----------
    output_dir : Path
        The events output directory.
    experiment_id : str
        Plane/experiment identifier.
    traces : numpy.ndarray
        Raw dF/F traces.
    result : dict
        Output of :func:`run_deconvolution`.
    params : dict
        Resolved parameters from :func:`~.params.resolve_parameters`.
    estimate_parameters : bool
        Whether OASIS estimated the model parameters.
    tau : float or None
        The configured decay time, used in the legend when OASIS did not
        estimate per-ROI time constants.
    frame_rate : float
        Acquisition frame rate in Hz.
    n_jobs : int, optional
        Worker count; ``None`` lets multiprocessing choose.

    Returns
    -------
    Path
        The plots directory.
    """
    n_rois, n_frames = traces.shape
    plots_path = Path(output_dir) / PLOTS_SUBDIR
    plots_path.mkdir(exist_ok=True, parents=True)
    timestamps = np.arange(n_frames) / frame_rate

    if estimate_parameters:
        baseline = result["b_hat"][:, None]
        taus = result["tau_hat"]
    else:
        baseline = params["b"]
        taus = [tau] * n_rois

    with Pool(n_jobs) as pool:
        pool.starmap(
            plot_trace_and_events_png,
            zip(
                traces,
                result["denoised"] + baseline,
                result["events"],
                [timestamps] * n_rois,
                range(n_rois),
                taus,
                [plots_path] * n_rois,
                [experiment_id] * n_rois,
            ),
        )
    return plots_path


if __name__ == "__main__":
    args = parser.parse_args()
    params = vars(args)
    start_time = dt.now()
    output_dir = Path(args.output_dir).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = make_output_directory(output_dir, experiment_id)
    setup_logging(
        "aind-ophys-oasis-event-detection", mouse_id=subject_id, session_name=name
    )
    try:
        print(f"Performing Event Detection for {experiment_id}")

        logging.info(f"SUCCESS: {experiment_id}")
    except Exception as e:
        logging.error(f"FAILED: {experiment_id}")
        raise e

    write_data_process(
        params,
        dff_fp,
        oasis_h5,
        experiment_id,
        start_time,
        end_time=dt.now(),
    )

    write_qc_evalutation(output_dir, experiment_id, N)
