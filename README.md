# Metashape SAM 3 Point Cloud Classifier

**AIhdproduction** · Outdoor drone photogrammetry · Python package `msam3`

Classify a finished Metashape dense point cloud from its original photographs using SAM 3.
The tool creates a **new point-cloud asset** in the same chunk, saves the project, and writes
a classified LAS 1.4 file with class legends. The original point cloud is preserved.

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

Choose one of the following options. **Option 1 is the easiest option** and starts directly
inside Metashape.

### Option 1 — Metashape: Tools → Run Script

1. Open the saved project in Metashape Professional.
2. Select **Tools → Run Script**.
3. Choose [`start_in_metashape.py`](start_in_metashape.py) in this repository.
4. Leave Metashape open until the completion message appears.

The script saves the open project, exports its point cloud, validates geometry, runs SAM 3, and
imports a new **SAM3 classified** point-cloud asset into that same project. At the end it selects
the asset and switches the model view to **Classification**. It uses the already installed
`.venv`; it does not start another Metashape instance or reopen the project. The Metashape window
is unavailable while the GPU is classifying images, which protects the project from edits during
processing. The original point cloud is preserved.

### Option 2 — PowerShell

Open PowerShell in the repository and run the geometry check first. This does not require model
weights or a GPU:

```powershell
cd "D:\Help\01_Privat\metashape-sam3-classifier"
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

Reopen the project. The new asset is named **SAM3 classified** under each processed chunk.
Select it and switch the point-cloud view to **Classification**. The original asset remains
the default unless `metashape.make_default: true` is configured.

- `output/<project>-<path-hash>/<project>_<chunk>_<key>_classified.las`: final LAS 1.4 in the chunk CRS.
- `class_legend.json` and `class_legend.csv`: complete class names and prompts alongside LAS files.
- Per-output `.report.json`: coverage, class counts, timing and validation results.
- `work/<project>-<path-hash>/chunk-<key>/classified_internal.las`: full-resolution stage-2 result
  in the export CRS. The filename does **not** mean its coordinates are internal; they are
  geocentric WGS 84 for georeferenced projects, or the local project CRS.
- `msam3.log`, `stage1_metashape.log`, `stage3_metashape.log`: progress and diagnostic logs.

**Custom-code limitation verified in Metashape 2.3.1:** asset codes 64–72 are retained,
but native LAS export truncates them. This tool repairs that through class-filtered exports
and LAS 1.4 assembly. Codes 128–255 were dropped on import; stage 3 rejects these codes before
import and checks every class count before saving. The standalone stage-2 LAS supports all
0–255 codes. Custom display names in Metashape may remain numeric/generic; consult the legend.

## Classes

```powershell
python -m msam3 classes
```

| Class | Code | Prompts |
|---|---:|---|
| Unclassified | 1 | Undecided points |
| ground | 2 | ground, terrain, soil, bare earth |
| road | 11 | road, street, asphalt, carriageway |
| sidewalk | 64 | sidewalk, footpath, pedestrian path, pavement |
| curb | 65 | curb, kerb, curb stone, edge stone |
| low_vegetation | 3 | low vegetation, grass, shrubs, bushes |
| tree | 5 | tree, tall vegetation, tree canopy, foliage |
| building | 6 | building, house, facade, exterior wall |
| vehicle | 66 | vehicle, car, van, truck |
| construction_machinery | 67 | construction machinery, excavator, digger, bulldozer, loader |
| utility_pole | 68 | utility pole, light pole, streetlight, lamp post |
| traffic_sign | 69 | traffic sign, road sign, signpost |
| fence | 70 | fence, barrier, railing, gate |
| power_line | 14 | overhead power line, utility wire, cable |
| scaffolding | 71 | scaffolding, construction scaffold |
| construction_material | 72 | construction container, site container, building material, construction material |
| water | 9 | water, pond, stream, puddle |

Append a block to `config/classes.yml` to add a class. A class without an explicit code gets
the next free number from 64. Appending one to the default list gives **73**. Existing explicit
codes are reserved before automatic allocation. Do not reorder existing classes after
classified data exists. Freeze codes explicitly before inserting/reordering existing entries.
LAS 1.4 lookup descriptions are limited to 15 bytes; the JSON and CSV contain full names.

## Tuning

| Aim | Settings to consider |
|---|---|
| More coverage | Reduce `voting.min_views` from 3 to 2; cautiously lower `sam.background_threshold`. |
| Fewer incorrect decisions | Raise `voting.min_share`, `voting.min_views`, or `sam.background_threshold`. |
| Power lines, fences, curbs | Keep `sam.image_scale: 1`, raise `cache.scale` toward 1, reduce voxel size and z-buffer splat radius. |
| Lower GPU memory use | Reduce `sam.prompt_batch_size` to 1 and `voting.batch_points`; SAM tiles remain 1008 pixels. Close other GPU applications. |
| Lower vote memory use | Reduce `voting.max_ram_gb` to spill votes to disk; increase voxel size to reduce representatives. |
| Smooth uncertain areas | Enable kNN smoothing. High-share and unclassified points remain unchanged. |

Large splats or depth tolerances can erase thin objects or admit occluded surfaces. Small
splats can leave holes. Geometry validation thresholds are error gates, not coverage controls;
investigate a failed gate before changing them. Optional `depth_maps` requires retained maps
and uses their own calibration. Incidence weighting requires valid exported LAS normal vectors;
it fails clearly when unavailable instead of inventing normals.

## Troubleshooting

| Problem | What to do |
|---|---|
| Missing `sam3.pt` | Follow the model-download section; verify `paths.model`. |
| Project locked | Save and close it in Metashape. Do not bypass or delete a live lock. |
| No active license | Ensure a valid Professional license is available to the headless process. Closing the project alone may leave a floating seat occupied by the GUI; close the application if needed. |
| Reprojection gate fails | Check rolling shutter, frame calibration and unsupported film corrections; enable `debug.save_overlays`. |
| Tie-to-dense gate fails | Check CRS, scale, cloud cropping and whether tie points and dense cloud cover the same area. |
| Image not found | Fix the image paths or configure `image_path_remap`. |
| Out of VRAM | Close GPU-heavy applications, reduce point batch size, use the recommended GPU. |
| bf16 unsupported | Set `gpu.dtype: float16`. |
| CUDA unavailable after installation | Update the NVIDIA driver; rerun the installer, which explicitly installs `+cu128` wheels. |
| SAM import error | Rerun the installer; inspect its first error. The pinned upstream import needs Triton, psutil and setuptools below 81. |
| Exit 0 but no marker | The stage failed despite the exit code; inspect the captured Metashape log and license status. |
| Stale stage 1 | Run `stage1` or `run` again with the same chunk selection and work directory. |
| Unsupported custom code | Keep project-import codes below 128. The stage-2 LAS remains available. |

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

## Development and publishing

```powershell
python -m pip install -e ".[dev]"
ruff check .
pytest
```

CI runs the tests on CPU on Windows and Linux. Model weights, imagery, project data and logs
are excluded from Git. Work stays on `main`. Nothing is pushed automatically.

Before a user-authorized push, verify the GitHub identity is **AIhdproduction**. With GitHub CLI:

```powershell
gh auth status
gh api user --jq .login
# Continue only when the authenticated login is AIhdproduction.
git remote -v
git status
git push -u origin main
```

Git's commit author and GitHub authentication are separate. This checkout uses
`AIhdproduction <260610871+AIhdproduction@users.noreply.github.com>` for local commits.

## Licenses

Repository: **All rights reserved**, AIhdproduction; the owner may replace the placeholder
[LICENSE](LICENSE). SAM 3 code and weights are subject to **Meta's SAM License**; weights are
not redistributed here. The method cites SegEarth-OV3. No SegEarth repository code was copied
because its license could not be established. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
