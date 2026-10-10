# Metashape SAM 3 Point Cloud Classifier

**AIhdproduction** · Outdoor drone photogrammetry · Python package `msam3`

Classify a finished Metashape dense point cloud from its original photographs using SAM 3.
The tool creates a **new point-cloud asset** in the same chunk, saves the project, and writes
a classified LAS 1.4 file with class legends. The original point cloud is preserved.

**License:** free for noncommercial education and research. Commercial use requires
prior written permission from **AIhdproduction**. See [Licenses](#licenses) and the
[German summary / deutsche Zusammenfassung](docs/licensing.de.md).

```powershell
python -m msam3 run --project "D:\jobs\site\site.psx"
```

## How it works

1. **Export:** headless Metashape exports the dense cloud, frame-camera calibrations,
   transforms and independent tie-point projection samples. This stage does not save the project.
2. **Classify:** geometry must pass validation before CUDA processing. SAM 3 segments image
   tiles. Visible points collect confidence-weighted votes from multiple cameras. Labels are
   propagated from a voxel voting cloud to every original point.
3. **Import:** headless Metashape imports a separate classified asset, verifies class counts,
   saves the project, and exports class partitions. The driver assembles a full-code LAS 1.4
   in the project's CRS. This last step avoids a native Metashape LAS export limitation.

The fusion method is an independent implementation based on
[SegEarth-OV3, Li et al., arXiv:2512.08730](https://arxiv.org/abs/2512.08730).
See [method and verified APIs](docs/method.md) for implementation details and checks.

## Requirements

- Windows 10/11, 64-bit.
- **Agisoft Metashape Professional 2.2 or newer within the 2.x series.** Standard has no
  Python scripting. Although the architecture uses 2.x APIs, the required import attributes
  were added in 2.2; 2.0 and 2.1 are rejected. Local API probes used 2.3.1 build 22580.
- NVIDIA CUDA GPU with **at least 16 GB VRAM**, a current NVIDIA driver, and sufficient RAM.
- **Python 3.12, 64-bit**, including the Windows Python launcher; **Git for Windows**.
- Free disk space for the exported cloud, classified cloud, temporary class exports, voxel
  partitions and image caches. Large jobs can need several times their source LAS size.
- Permission to use the separately downloaded SAM 3 checkpoint.

## Installation

Open PowerShell in the directory where you want the repository and run:

```powershell
git clone https://github.com/AIhdproduction/metashape-sam3-classifier.git
cd metashape-sam3-classifier
.\install.bat
```

The installer creates `.venv`, checks Python's version, installs CUDA PyTorch and official
SAM 3 at the verified commit, and checks imports and CUDA availability. A second run reuses
the environment and repairs a CPU-only PyTorch installation. It stops on errors. It does
not download model weights or alter Metashape's embedded Python. Use
`install.bat --no-pause` for a terminal run without the final keypress prompt.

To activate the environment in **Command Prompt**:

```bat
.venv\Scripts\activate.bat
```

In **PowerShell**, use `.\.venv\Scripts\Activate.ps1`. If your existing execution policy
blocks activation, run `.\.venv\Scripts\python.exe` instead of `python`; activation is optional.

Manual installation, equivalent to the installer:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade "pip>=25,<27" "setuptools>=75,<81" wheel
.\.venv\Scripts\python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m pip install -e ".[sam]"
.\.venv\Scripts\python -m pip check
```

`.[sam]` installs Meta's official `sam3` repository, not an unrelated PyPI implementation.
Windows needs the additional `triton-windows` runtime because SAM 3 imports its tracker
modules even for image inference. This extra is included automatically.
The CUDA wheel pairing follows [PyTorch's installation matrix](https://pytorch.org/get-started/previous-versions/).

## Model download — required separately

1. Visit [facebook/sam3 on Hugging Face](https://huggingface.co/facebook/sam3).
2. Request access and accept **Meta's SAM License**.
3. Download the **SAM 3** checkpoint and save it as `models/sam3.pt`.
4. Do not substitute the SAM 3.1 video checkpoint.

**Weights are not in this repository and must never be committed.** The model is loaded
locally; the tool never requests a token or downloads a checkpoint automatically.

## Configure Metashape

Edit `config/default.yml` if Metashape is installed somewhere else:

```yaml
metashape:
  exe: "C:/Program Files/Agisoft/Metashape Pro/metashape.exe"
```

Relative configuration paths resolve from the repository root. A relative CLI `--work-dir`
resolves from your current directory. `metashape.offscreen: auto` uses `-platform offscreen`
where available; on Windows installations without that Qt plugin, `-r` alone runs headlessly.

## Prepare the project

Align the cameras, build the dense point cloud, and retain valid tie points for validation.
Save and **close the project in the Metashape GUI**. Images must still exist at their stored
locations. If they moved, configure prefix replacements:

```yaml
paths:
  image_path_remap: {"D:/old/": "E:/new/"}
```

Only enabled, aligned frame cameras are processed. Fisheye, spherical, unaligned and disabled
cameras are counted in the logs. A chunk without an eligible camera or dense cloud is skipped.
Keep a normal project backup, particularly for the first real run.

## Start and resume

Choose one of the following options. **Option 1 is the recommended workflow**: it keeps the
open Metashape project under Metashape's control, imports the result automatically and writes
the final LAS beside the project.

### Option 1 — Metashape: Tools → Run Script

1. Open the saved project in Metashape Professional.
2. Select **Tools → Run Script**.
3. Choose [`start_in_metashape.py`](start_in_metashape.py) in this repository.
4. The window **SAM 3 – Start und Einstellungen** opens. Choose the options and select
   **Klassifikation starten**.
5. Leave Metashape open until the completion message appears.

The start window applies its choices **only to this run**; it does not modify
`config/default.yml`. It provides:

- **Schnell**, **Ausgewogen** and **Hohe Genauigkeit** profiles.
- SAM image resolution, exact tile overlap, minimum image observations and GPU prompt batch.
- Switches for Metashape ground assistance, geometric postprocessing, DTM vegetation split,
  road-surface repair and optional kNN smoothing.
- **Auto: maximal sicher**, which measures currently free CUDA memory and selects the largest
  conservative prompt batch from 4, 8, 12 and 16. It reserves VRAM for SAM, Windows and
  Metashape. High GPU utilization can make the computer temporarily less responsive.
- A run-time estimate. It uses the active camera image sizes, the resulting exact number of
  1008-pixel SAM tiles, the selected batch, and compatible previous run logs for the same project.
  It updates immediately when resolution, overlap or batch changes.

After the start choice, a separate topmost **SAM 3 – Metashape-Klassifikation** progress window
shows the current phase, image counter, percentage, elapsed time and remaining-time estimate.
Its **Abbrechen** button stops the background worker cleanly. During the run Metashape is protected
from project edits; use the external window for status and cancellation. The script saves the open
project, exports the source photo-colored point cloud, validates geometry, runs SAM 3, imports a
new **SAM3 classified** point-cloud asset, and then selects it with the original photo RGB colors.
The source point cloud remains unchanged.

### Option 2 — PowerShell

Open PowerShell in the repository and run the geometry check first. This does not require model
weights or a GPU:

```powershell
cd "C:\path\to\metashape-sam3-classifier"
.\.venv\Scripts\python.exe -m msam3 validate --project "D:\jobs\site\site.psx"
.\.venv\Scripts\python.exe -m msam3 run --project "D:\jobs\site\site.psx"
```

For PowerShell, save and close the project in Metashape first. The direct Metashape option above
keeps it open instead.

To process a particular chunk or use different settings:

```powershell
python -m msam3 run --project "D:\jobs\site\site.psx" --chunk "Block A" --config config/default.yml --classes config/classes.yml --work-dir "E:\msam3-work"
```

Repeat `--chunk` for multiple labels. Without it, the configured list is used; an empty list
selects all enabled chunks. Chunk labels must identify the intended chunks unambiguously.

After interruption, rerun the same command. Completed image caches are reused if the model,
segmentation configuration and source image identity still match. Each project has a separate
work directory identified by a hash of its absolute path. Voting is recomputed from scratch
to avoid double-counting partially accumulated observations.

You can run stages separately:

```powershell
python -m msam3 stage1 --project "D:\jobs\site\site.psx"
python -m msam3 stage2 --project "D:\jobs\site\site.psx"
python -m msam3 stage2 --project "D:\jobs\site\site.psx" --voting-only
python -m msam3 stage3 --project "D:\jobs\site\site.psx"
```

Changing voting, visibility or smoothing settings does not invalidate segmentation caches.
`--voting-only` requires all image caches to be valid and still requires the same local
checkpoint for hash verification. Changes to the saved project invalidate the export.

## Results

For the Metashape start-window workflow, the result is imported as a new point-cloud asset named
**SAM3 classified** in every processed chunk. It is selected with the original photo RGB colors,
so the point cloud keeps its familiar image appearance. Switch Metashape manually to a
classification display only when you want to inspect numeric LAS classes.

The complete classified LAS is written beside the Metashape project:

```text
<project folder>\export\punktwolke_class.las
```

For example, `D:\Test Script\Test_script.psx` produces
`D:\Test Script\export\punktwolke_class.las`. The import and LAS assembly must both succeed
before the workflow reports completion. Intermediate processing data stays below the repository's
`work/` directory; the project folder receives only the promised `export` output.

For the PowerShell workflow, final exports remain under
`output/<project>-<path-hash>/`. The following artifacts are available for both workflows:

- `class_legend.json` and `class_legend.csv`: class names, codes, colors and prompts.
- Per-output `.report.json`: coverage, class counts, timing and validation results.
- `work/<project>-<path-hash>/chunk-<key>/post_sam_filter/postprocess_report.json`:
  geometry-cleanup and DTM vegetation decisions.
- `msam3.log`, `stage1_metashape.log`, `stage2.log` and `stage3_metashape.log`: progress and
  diagnostic logs.

The postprocessor works on SAM votes after image-to-point assignment. It can remove isolated
points, fill small supported holes, restore flat road patches, reject detached ground points,
repair flat roofs, and split connected mixed vegetation using an in-memory terrain model.
The vegetation terrain support prioritizes ground, water, road, sidewalk and curb classes;
wire and guard-rail classes are used only as a fallback. A connected SAM tree cluster with a
supported crown at least 1.5 m high remains high vegetation as a whole.

**Metashape import limitation:** project-import class codes must stay below 128. The tool checks
class counts before saving and assembles the final LAS from class-filtered Metashape exports to
avoid native custom-code export loss. Consult the CSV or JSON legend for full class names.

## Classes

```powershell
python -m msam3 classes
```

| Class | LAS code | Meaning |
|---|---:|---|
| Unclassified | 1 | Undecided or rejected point |
| ground | 2 | Ground, terrain, soil, bare earth |
| low_vegetation | 3 | Grass, shrubs and low vegetation |
| tree | 5 | High vegetation and tree clusters |
| building | 6 | Building, house, facade or exterior wall |
| water | 9 | Pond, stream, puddle or water surface |
| road | 11 | Vehicle carriageway and asphalt road |
| power_line | 14 | Overhead wire, cable or conductor |
| construction_machinery | 67 | Excavator, loader, bulldozer or similar |
| sidewalk | 73 | Pedestrian sidewalk or paved footway |
| curb | 74 | Curb, kerb or edge stone |
| construction_material | 76 | Container or construction material |
| fence | 77 | Fence, barrier, railing or gate |
| traffic_sign | 81 | Road sign or signpost |
| utility_pole | 84 | Utility pole, streetlight or lamp post |
| vehicle | 91 | Vehicle, car, van or truck |
| scaffolding | 110 | Construction scaffold or steel structure |

The default codes follow the usable Cyclone/ASPRS mapping in
[`config/classes.yml`](config/classes.yml). Metashape displays the imported point cloud with
photo RGB colors by default; colors in the class legend are reference colors for exports and
classification views. New custom classes begin at code 120. Keep every project-import code below
128 and assign codes explicitly before changing an established class list.

## Tuning

Use the **Metashape start window** for normal production runs. The selected settings apply only
to that run. Use `config/default.yml` when you need a permanent default or the PowerShell workflow.

| Aim | Start-window choice or setting |
|---|---|
| Shortest reliable run | **Schnell**, lower image resolution and 10% tile overlap. |
| Better small-object boundaries | **Hohe Genauigkeit**, 100% resolution and greater tile overlap. |
| Use the GPU efficiently | Click **Auto: maximal sicher**; it recalculates from current free VRAM. |
| Lower GPU memory use | Select a lower prompt batch; close other GPU applications. |
| More conservative point acceptance | Increase image observations per point or raise `voting.min_share`. |
| Correct uncertain small spots | Enable kNN smoothing only after checking that it does not soften edges. |
| Improve ground/road support | Keep Metashape ground assistance and geometric postprocessing enabled. |
| Improve vegetation height classes | Keep DTM vegetation splitting enabled; high vegetation starts at 50 cm,
  with a 45–55 cm SAM-preserving transition band. |

The estimate is a planning value, not a guarantee: it learns from compatible earlier logs and
recalculates exact tile counts from the active images. It becomes more accurate after successful
runs with the same project, GPU and model. Large splats or depth tolerances can erase thin objects
or admit occluded surfaces. Geometry validation thresholds are error gates, not coverage controls.

## Troubleshooting

| Problem | What to do |
|---|---|
| No start window appears | Run `install.bat` again, then start `start_in_metashape.py` through **Tools → Run Script**. |
| `No module named 'msam3'` | The venv package is missing or outdated; run `install.bat` from the repository. |
| Missing `sam3.pt` | Follow the model-download section; verify `paths.model`. |
| Project locked | For PowerShell, save and close Metashape. For the start-window workflow, keep Metashape open. |
| No active Metashape license | Ensure a valid Professional license is available before starting. |
| Out of VRAM | Close GPU-heavy applications, click **Auto: maximal sicher**, or select a lower GPU prompt batch. |
| PC becomes slow during classification | This is expected with a high auto batch. Choose a lower batch if you need to work on the PC. |
| Reprojection gate fails | Check rolling shutter, frame calibration and unsupported film corrections; enable `debug.save_overlays`. |
| Tie-to-dense gate fails | Check CRS, scale, cloud cropping and whether tie points and dense cloud cover the same area. |
| Image not found | Fix the image paths or configure `image_path_remap`. |
| bf16 unsupported | Set `gpu.dtype: float16`. |
| CUDA unavailable after installation | Update the NVIDIA driver; rerun the installer, which explicitly installs `+cu128` wheels. |
| SAM import error | Rerun the installer; inspect its first error. The pinned upstream import needs Triton, psutil and setuptools below 81. |
| Exit 0 but no marker | The stage failed despite the exit code; inspect the captured Metashape log and license status. |
| Unsupported class code | Keep project-import codes below 128. The stage-2 LAS remains available. |

## Limitations and verification

Outdoor drone imagery only. Frame cameras only. Coverage depends on overlap and visibility;
thin objects, reflections, water and vegetation motion are difficult. Prompt-based labels are
not survey certification or height-rule vegetation classification. Check the classified cloud
before delivery. No cloud upload is required for inference.

The CPU tests do not require Metashape or weights. The dense cloud is streamed, but voting
representatives, spatial indices and their kNN tree still need RAM. `voting.max_ram_gb` governs
the **vote matrix**, not every allocation. Extremely large voting clouds may need larger voxels
and more system memory. Native final LAS coordinate quantization is controlled by Metashape;
the stage-2 LAS preserves the original stage-1 coordinates exactly.

For the precise scope of tested behavior and remaining field checks, see
[method.md](docs/method.md#to-be-verified-on-first-real-run) and the
[local verification record](docs/verification.md).

## Development

Install the optional development tools and run the local checks:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ruff check src tests metashape_scripts start_in_metashape.py
.\.venv\Scripts\python.exe -m pytest -q
```

CI runs the CPU tests on Windows and Linux. Model weights, imagery, Metashape projects,
exports, logs and `work/` data are excluded from Git. Keep credentials and model weights out
of commits. For changes intended for the project, use the repository's normal contribution and
review process.

## Licenses

Original project material is licensed under the custom
**[AIhdproduction Education and Research License 1.0](LICENSE)**. AIhdproduction retains
its rights in that material. This is source-available software with purpose restrictions.

- **Free without an individual request:** noncommercial teaching, coursework, theses and
  research, including modification and free redistribution under the license conditions.
- **Prior written agreement required:** commercial use, paid client work, internal business
  use, commercial R&D, products, hosted services and commercial exploitation of results.
  Academic status alone does not exempt a commercial project.
- **Your data and results:** AIhdproduction claims no ownership of them. Noncommercial
  research results may be published and shared; the software license does not automatically
  become the license of those results. Commercial generation or exploitation of results by
  the licensee requires permission under the license.
- **Requests:** contact AIhdproduction through the
  [project issue tracker](https://github.com/AIhdproduction/metashape-sam3-classifier/issues).
  Permission must be expressly granted in writing before the proposed use.

The complete [LICENSE](LICENSE) controls; a [German explanation](docs/licensing.de.md)
is available. Publications must acknowledge this project and SAM 3 as applicable.

SAM 3 code and weights are separately subject to
**[Meta's SAM License](https://github.com/facebookresearch/sam3/blob/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc/LICENSE)**.
Weights are not redistributed here; upstream SAM 3 code is installed separately.
Using only generated outputs does not exempt the person running SAM 3 from Meta's terms.
Metashape requires its own license. The method cites SegEarth-OV3; no SegEarth repository
code was copied because its license could not be established during implementation.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
