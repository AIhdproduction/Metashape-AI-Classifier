# Local verification record

Verified on Windows with Python 3.12.6, Metashape Professional 2.3.1 build 22580,
PyTorch 2.7.1+cu128, and an NVIDIA RTX 4090. This record distinguishes executable checks
from the outstanding real-project acceptance test.

## Completed checks, 8–9 October 2026

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

## Real-project acceptance test: blocked by licensing

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
