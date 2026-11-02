# aind-ophys-oasis-event-detection-library

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python->=3.10,<3.13-blue?logo=python)

Backing library for the `aind-ophys-oasis-event-detection` Code Ocean capsule. It reads one
plane's dF/F traces, deconvolves them with [OASIS](https://github.com/j-friedrich/OASIS) into
denoised calcium and fluorescence events, and emits the events HDF5, per-ROI QC figures and the
`aind-data-schema` v2 metadata documents. The capsule is a thin wrapper that calls
`job.run()`; all logic lives here and is unit-tested.

## Usage

```python
from aind_ophys_oasis_event_detection_library.job import run

run()  # settings come from CLI arguments / environment
```

Settings are parsed by pydantic-settings, so every field is also a `--name=value` command-line
option (`--tau`, `--tau_rise`, `--estimate_parameters`, `--no_qc`, ...). See `settings.py` for the
full list and defaults.

### Input

The upstream dF/F step's results, mounted at `--input_dir` (default `../data/`):

- `<experiment_id>/dff/<experiment_id>_dff.h5`, with the trace matrix in the `data` dataset of
  shape `(n_rois, n_frames)`
- the acquisition metadata for the asset: `acquisition.json` (v2), `session.json` (v1) or a
  minimal `metadata.json`. The frame rate is required, and is resolved from the core file, then
  `platform.json`, then the `--frame_rate` override.

### Output

Everything is written under `<output_dir>/<experiment_id>/events/`:

- `<experiment_id>_events_oasis.h5` — `events` and `denoised`, both `f4` and shaped
  `(n_rois, n_frames)`. When OASIS estimates the model parameters, the per-ROI `f4` arrays
  `b_hat`, `tau_hat`, `lam_hat` and (AR2 only) `tau_rise_hat` are written as well.
- `plots/<experiment_id>_<roi_id>_oasis.png` — one dF/F, denoised-trace and event-rate figure per
  ROI, skipped with `--no_qc`.
- `processing.json` — a full `Processing` document, written on both the success and the error path.
- `quality_control.json` — a full `QualityControl` document holding the per-ROI curation metric
  that references the figures above.

## Modules

### `job.py`
`run()`, the entry point: loads metadata, resolves the run plan, runs the stage under the shared
`stage_guard`, which owns the `processing.json` lifecycle.

### `settings.py`
`OasisSettings`, the CLI/env settings model; the Code Ocean app panel is generated from it.

### `params.py`
`ar_coefficients()`, `taus_from_ar_coefficients()`, `resolve_parameters()` and
`deconvolve_kwargs()` — converting decay/rise time constants to auto-regressive coefficients and
back, and mapping the settings onto the keyword names OASIS expects.

### `events.py`
The compute and the outputs: `find_dff_traces()`, `read_dff_traces()`, `make_output_directory()`,
`run_deconvolution()` (one trace per worker process), `write_events_h5()`, `write_qc_plots()` and
`plot_trace_and_events_png()`.

### `qc.py`
`build_events_curation_metric()`, `roi_plot_references()`, `build_quality_control()` and
`write_quality_control_json()` — the `quality_control.json` content.

### `utils/metadata_utils.py`
`load_oasis_metadata()` (metadata reads via `aind-pophys-metadata`'s `CoreMetadata`),
`upstream_process_names()` and `build_code()`.

## Level of Support
Please indicate a level of support:
 - [ ] Supported: We are releasing this code to the public as a tool we expect others to use. Issues are welcomed, and we expect to address them promptly; pull requests will be vetted by our staff before inclusion.
 - [ ] Occasional updates: We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.
 - [ ] Unsupported: We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

## Release Status
GitHub's tags and Release features can be used to indicate a Release status.

 - Stable: v1.0.0 and above. Ready for production.
 - Beta:  v0.x.x or indicated in the tag. Ready for beta testers and early adopters.
 - Alpha: v0.x.x or indicated in the tag. Still in early development.

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e . --group dev
```
Note: --group flag is available only in pip versions >=25.1

Alternatively, if using [uv](https://docs.astral.sh/uv/), run
```bash
uv sync
```

## Tests and linters

Tests are `unittest`; coverage and interrogate are both gated at 100%.

```bash
coverage run -m unittest discover
coverage report
flake8 .
interrogate --verbose .
```
