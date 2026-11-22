# Local verification record

This record distinguishes executed checks from what was not tested.

## Model choice, FlurSeg and app, 22 November 2026

All checks of this section ran on Windows **without GPU** (CUDA hidden), without the Metashape
application and without installing anything. Of Metashape only the bundled `python.exe` was run,
as described below. The test environment was **Python 3.13, NumPy 2.2.6, PyTorch
2.14.0 (used on CPU), Transformers 5.22.1 and NiceGUI 3.17.1**. This is not the production
environment of Python 3.12, NumPy 1.26.4 and PyTorch 2.7.1+cu128.

### Executed

- `pytest`: 256 passed (the earlier 57 and the new ones). `ruff check .`: passed, with ruff
  0.17.3 instead of the pinned 0.11.13. The tests replace every CUDA query with a fake.
- 209 of these tests also passed on Python 3.12, NumPy 1.26.4 and PyTorch 2.11 (CPU). That
  environment had no Transformers, NiceGUI or pyproj, so the app tests and the two real-model
  tests were left out and pyproj was replaced by a stub.
- A tiny random EoMT-DINOv3 runs a real forward through the backend on CPU. The band buffer
  equals the full-buffer result; padding, label order and out-of-memory back-off (simulated)
  are tested.
- **The real v1 weights** (SHA-256 `83ad8990…a789a9`) were loaded once on CPU with six threads,
  without missing or unexpected keys. One 768-pixel tile of a drone photo was segmented: the
  class order is correct, and the labels and confidences are identical to SAMcloud's own
  inference. Only this one tile was checked. It contained no marking, kerb or cover.
- SAM 3 regression against commit 647c69f on synthetic data with a fake backend: smoothing and
  post-filter of 24 random scenes, and one complete stage 2 with default settings, gave
  identical codes, LAS point records, CSV legend and assist counts.
- The embedded scripts were run unchanged against a **fake** Metashape module, through export,
  worker, import and final LAS, for both models in one project. They parse as Python 3.9.
- **The scenario harness for the embedded scripts ran on Python 3.9.13 and on Python 3.12**,
  each as `python.exe -I -S -B`: 34 of 34 scenarios passed on both. It executed unchanged
  copies of `start_in_metashape.py` and all of `metashape_scripts/` in a scratch folder. The
  3.9.13 interpreter is the one bundled with the installed Metashape 2.1.0; 3.12 is the
  python.org installation of this PC (3.12.9). `Metashape` was a fake module in plain
  Python, reporting 2.2.2 under 3.9 and 2.3.1 under 3.12, and the venv's processes (install
  check, app, worker) were stubs. The scenarios:
  - success with the `depth_maps` branch, legends, two chunks, a geographic CRS and z-buffer
    mode, for both models, including reuse of an imported asset;
  - cancel before Start, after preparation, as the worker starts, during and at the end of
    classification;
  - errors of preconditions, export, worker, import and finalize, a stale install, an app crash;
  - install-check timeout, startup deadline, lost heartbeat and an interrupted script, each
    ending a real sleeping child process (with a simulated clock for the waiting time);
  - stage 1 and stage 3 as headless scripts, and consoles with ASCII, cp932 and cp1251.
  On 3.9 the scenarios executed every line of the starter, `common`, `gui` and
  `stage3_import`, and 262 of 263 lines of `stage1_export`. On 3.12 they executed every line of
  all five files. Nothing in the scripts had to be changed for 3.12.
- The same chain with real processes (app under `pythonw`, worker) and stubbed inference,
  driven by real clicks in headless Edge: start, reload, cancel of a running worker, a second
  and third run, no process left over.
- The app's pages and states at 1920×1080 and at a 375×812 layout viewport, without horizontal
  scrolling and without any request to another host.

### Not executed

- **No run in Metashape.** The new launcher, the window opened from Metashape and the import of
  the new codes 71, 75 and 82 are untested there.
- **No GPU inference.** fp16 autocast, VRAM use, real out-of-memory back-off and all speeds are
  untested. The 15× and 7.6× figures come from other tests, not from this pipeline.
- **torch 2.7.1 with Transformers 5.22.1 was not run locally.** The CPU tests in CI check it,
  once CI has run for this version.
- Metashape's real Python API, its console and its process environment, in every version. The
  harness used a fake module and stubbed processes, and it did not run the venv's code.
- Any Metashape 2.2.x or 2.3.x interpreter: none is installed on this PC. The installed 2.1.0 is
  below the supported minimum of 2.2, and the launcher refuses it.
- A real CUDA check. The early check was tested only with an injected fake GPU and without one.
- `install.bat` in its new form, the download commands and a real `--app` browser window.
- SAM 3 with real weights in this version, and the semantic accuracy of either model on a point
  cloud.

### Metashape versions and their Python

| Fact | Basis |
|---|---|
| The Metashape installed on this PC on 22 November 2026 is 2.1.0 | observed: uninstall registry, exe dated 2024-01-07 |
| Its bundled Python is 3.9.13 | observed: interpreter run |
| Metashape 2.3.x embeds Python 3.12 | documented, not observed: Agisoft's changelog, "Version 2.3.0 build 21378 preview: Updated Python to version 3.12" |
| Metashape 2.2.x embeds Python 3.9 | inferred, not observed: the changelog lists no Python update between 2.1 and 2.3 |

The October record below names 2.3.1, which was the installation at that time.

### Found during this work

Stage 1 called `zip(..., strict=False)`. That keyword exists from Python 3.10 on.

- Observed: on Python 3.9.13 it raises `TypeError: zip() takes no keyword arguments`. With the
  old call put back into a scratch copy, 16 of the 34 scenarios failed on 3.9 and none on 3.12.
- Inferred: an export in the default `depth_maps` mode would fail on Metashape 2.2.x.
  Metashape 2.3.x with Python 3.12 is not affected.

The call was changed for compatibility with the supported 2.2.x line. No failed export in a
real Metashape was observed.

The harness also showed that a launcher message with umlauts aborted the whole run on a console
whose code page cannot encode it. Such messages are now printed with escapes instead.

## Earlier checks of the SAM-only version

Verified on Windows with Python 3.12.6, Metashape Professional 2.3.1 build 22580,
PyTorch 2.7.1+cu128, and an NVIDIA RTX 4090. They do not cover the changes above.

### Completed checks, 8–9 October 2026

- Fresh virtual environment and editable package installation succeeded.
- `install.bat --no-pause` completed, including official SAM 3 imports, CUDA detection,
  dependency consistency and the class-list command. Rerunning repairs CPU-only PyTorch.
- `ruff check .`: passed. `pytest -q`: 30 passed. `pip check`: no broken requirements.
- The synthetic pipeline recovered more than 95% of known labels, preserved coordinates
  and colors, and passed cache corruption, interruption/resume, projection, visibility,
  class-code and LAS-partition checks without requiring model weights or Metashape.
- The user-supplied `models/sam3.pt` loaded on CUDA. A real 1008×1008 drone-image tile
  produced finite probabilities for all 16 classes, approximately 0 to 0.965. Batched
  prompt inference took 1.66 seconds in this isolated measurement, excluding model loading.
  This is not a whole-project runtime estimate or an accuracy measurement.
- A complete 5280×3956 drone photograph passed the production tiling and maximum-merge
  path with all 16 classes in 46.92 seconds, excluding model loading. All output
  probabilities were finite; 1.13% of pixels remained unknown at the default threshold.
  Label and confidence PNGs were written successfully and the scratch mapping was removed.
  These are image-level execution and coverage checks, not point-cloud accuracy results.
- Live Metashape probes confirmed camera orientation and extended calibration behavior,
  headless invocation, metadata access, and separate point-cloud assets. A synthetic
  128-point stage-3 import/export/save check preserved class counts for ground and a custom
  code. Tests of codes 0–255 established the import/export limitations documented in
  [method.md](method.md#custom-las-codes-observed-behavior).
- The direct-launcher module imported successfully in Metashape Professional 2.3.1's embedded
  Python interpreter without opening a project. It uses no custom Qt widget or second
  Metashape process. A complete interactive run remains subject to the real-project acceptance
  test below.

### Real-project acceptance test: blocked by licensing

The real project was backed up before testing. Stage 1 recognized 291 aligned cameras.
The initial run exposed homogeneous four-component tie-point coordinates; the exporter
now normalizes these before projection. The next run reached dense-cloud export, which
failed with `OSError: No license found`. A separate license probe reported that the local
license had expired and all floating licenses were occupied. The adapter now checks for
an active license before opening the project and reports a direct remediation message.

No classified real-project asset was saved and the original project was not modified.
Full geometry validation, multi-view classification and real-project import remain
unverified. Once a licensed headless process is available, run `validate`, inspect its
checks, then run the complete pipeline. Follow the remaining dataset-specific checks in
[method.md](method.md#to-be-verified-on-first-real-run).

Model weights, survey imagery, test LAS files, private project paths and local logs are
excluded from the committed repository.
