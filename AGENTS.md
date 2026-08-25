# Repository Guidelines

## Project Structure & Module Organization

Application code lives in `src/sorting_vision/`. Keep camera adapters in `camera.py`, motion gating in `interlock.py`, JSON/TCP behavior in `server.py`, and 2-D/RGB-D processing in `pipeline.py` and `pipeline3d.py`. Geometry backends and edge-topology code use the `geometry_*.py` modules. Tests mirror these modules under `tests/` with names such as `test_geometry_edges.py`. Runtime defaults belong in `config/default.yaml`; trained artifacts belong in `models/`. Treat `data/` and folders named `几何测试_*` as local datasets or generated review artifacts, not source code.

## Build, Test, and Development Commands

Use Python 3.10 or newer. On PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m sorting_vision.cli --help
```

Install `.[realsense]`, `.[cnn]`, or `.[cnn-train]` only for the relevant hardware or model workflow. Run a focused test while iterating, for example `python -m pytest tests/test_geometry_edges.py -q`. Use `camera-live --source uvc --camera-index 1` for local USB preview; RGB-only results must never become executable grasps.

## Coding Style & Naming Conventions

Follow standard Python style: four-space indentation, `snake_case` functions and modules, `PascalCase` classes, and uppercase constants. Add type hints to public APIs and dataclasses. Prefer small, deterministic NumPy/OpenCV operations; keep optional SDK imports lazy. No formatter or linter is enforced, so run `python -m py_compile` and `git diff --check` before committing. Preserve schema compatibility when changing model or TCP formats.

## Testing Guidelines

Pytest is configured in `pyproject.toml`; new behavior requires a matching `test_*.py` test. Cover normal input, empty or corrupt input, low-confidence rejection, save/load compatibility, and safety interlocks. Hardware-dependent tests must skip clearly when optional packages or devices are unavailable. Dataset reports must disclose duplicate removal and whether evaluation is same-batch, holdout, or training replay.

## Commit & Pull Request Guidelines

History uses short imperative subjects, for example `Add edge-topology geometry classifier`. Keep commits focused and do not commit camera captures unless deliberately adding a reviewed fixture. Pull requests should describe behavior changes, commands run, test results, model/data provenance, and latency or accuracy changes. Include annotated before/after images for visual-pipeline changes and call out any protocol, configuration, or model-version migration.

## Safety & Configuration

Do not embed machine-specific calibration, credentials, or absolute paths in source. Preserve the rule that only RGB-D results with `status=PICKABLE` and `selected=true` may drive motion; RGB development output remains `DEPTH_REQUIRED`.
