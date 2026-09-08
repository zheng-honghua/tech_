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

Install `.[realsense]`, `.[cnn]`, or `.[cnn-train]` only for the relevant hardware or model workflow. Run a focused test while iterating, for example `python -m pytest tests/test_geometry_edges.py -q`. Use `camera-live --source uvc --camera-index 0` for local USB preview and change the index to the actual enumerated device; RGB-only results must never become executable grasps.

## Coding Style & Naming Conventions

Follow standard Python style: four-space indentation, `snake_case` functions and modules, `PascalCase` classes, and uppercase constants. Add type hints to public APIs and dataclasses. Prefer small, deterministic NumPy/OpenCV operations; keep optional SDK imports lazy. No formatter or linter is enforced, so run `python -m compileall -q src tests` and `git diff --check` before committing. Preserve schema compatibility when changing model or TCP formats.

## Testing Guidelines

Pytest is configured in `pyproject.toml`; new behavior requires a matching `test_*.py` test. Cover normal input, empty or corrupt input, low-confidence rejection, save/load compatibility, and safety interlocks. Hardware-dependent tests must skip clearly when optional packages or devices are unavailable. Dataset reports must disclose duplicate removal and whether evaluation is same-batch, holdout, or training replay.

Every dataset run, model change, or visual-pipeline change must also receive an actual visual review; statistics alone are insufficient. Generate RGB/depth contact sheets with `rgbd-review`, inspect ROI boundaries, segmentation, labels, missing depth, merged/split objects, and abnormal frames, then record reviewed batches, frame IDs, and findings in the accompanying Markdown report. Do not mark the review complete merely because the contact sheets were generated.

## Algorithm Change Records

Every code update must also update `算法更新记录.md`. Preserve its chronological 19-stage baseline and append later algorithm changes as stage 20, 21, and so on; never renumber or replace the earlier stages. Record the changed algorithm, the problem and purpose, expected effect, actual test result, visual-review finding, affected files/models, and known limitations. Do not describe an expected improvement as an achieved result. For documentation-only or infrastructure changes, explicitly write `算法变化：无` and do not consume an algorithm-stage number. If an update needs a longer report, add a dated Markdown document and link it from the log. An implementation is not complete until its record and relevant usage commands are current.

## Commit & Pull Request Guidelines

History uses short imperative subjects, for example `Add edge-topology geometry classifier`. Keep commits focused and do not commit camera captures unless deliberately adding a reviewed fixture. Pull requests should describe behavior changes, commands run, test results, model/data provenance, and latency or accuracy changes. Include annotated before/after images for visual-pipeline changes and call out any protocol, configuration, or model-version migration.

## Safety & Configuration

Do not embed machine-specific calibration, credentials, or absolute paths in source. Preserve the rule that only RGB-D results with `status=PICKABLE` and `selected=true` may drive motion; RGB development output remains `DEPTH_REQUIRED`.
