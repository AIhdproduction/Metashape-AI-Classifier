# Metashape AI Classifier

**AIhdproduction** · Outdoor drone photogrammetry · Python package `msam3`

Classify a finished Metashape dense point cloud from its original photographs. You choose the model:

- **FlurSeg** (default): our model with 11 fixed classes, designed for fast outdoor classification.
- **SAM 3** (option): finds **your own classes** through English text prompts.

Everything else is the same for both. The tool creates a **new point-cloud asset** in the same
chunk, saves the project and writes a classified LAS 1.4 file with class legends. The original
point cloud is preserved. Everything runs locally and offline.

**License 1.1:** free for noncommercial private use (by an individual for personal purposes,
without payment, not for a business, employer or client), education and research. Every other use
requires prior written permission from **AIhdproduction**. See [Licences](#licences) and the
[German summary / deutsche Zusammenfassung](docs/licensing.de.md).

Contents: [Quick start](#quick-start) ·
[Update](#update-an-existing-installation) ·
[Choose a model](#choose-a-model) ·
[FlurSeg download](#download-flurseg) ·
[SAM 3](#sam-3-optional) ·
[Classes](#classes-and-las-codes) ·
[App](#the-app) ·
[CLI](#command-line) ·
[Troubleshooting](#troubleshooting)

## Quick start

1. Install Python 3.12 (64-bit, with the Python launcher), Git for Windows and a current NVIDIA
   driver. Then, in PowerShell:

   ```powershell
   git clone https://github.com/AIhdproduction/Metashape-AI-Classifier.git
   cd Metashape-AI-Classifier
   .\install.bat
   ```

2. Download FlurSeg into `models\flurseg\v1` (about 1.26 GB), in the same folder:

   ```powershell
   .venv\Scripts\hf.exe download AIhdproduction/Flurseg --revision 5df53dbbf721778f1ecd4903369e9465926ae2fd --include "v1/*" --local-dir models/flurseg
   ```

3. In Metashape Professional open the saved project, choose **Tools → Run Script** and select
   [`start_in_metashape.py`](start_in_metashape.py).
4. The app opens. FlurSeg is already selected. Press **Klassifikation starten**
   (Start classification).
5. Keep Metashape open until it reports completion. It does not respond during the run; watch
   the app window for progress and use it to cancel.

## Update an existing installation

Close Metashape and any app window, then run in the repository:

```powershell
git pull
.\install.bat
```

**Run `install.bat` after the pull.** This version needs new packages (NiceGUI, Transformers).
If you forget, the script in Metashape tells you so and starts nothing.

- Your weights in `models\`, your own classes and your work data are not touched by either command.
- An existing `models\sam3.pt` keeps working. To use FlurSeg, download it once as described below.
- An older configuration file without the model keys still loads and uses FlurSeg. Set
  `segmentation.backend: sam3` to make SAM 3 the default again.
- If Git refuses the pull because you edited `config/default.yml`, save a copy of your changes
  first and enter them again afterwards.
- Segmentation caches of older versions are not reused. The first run computes the images again.

## Requirements

- Windows 10/11, 64-bit.
- **Agisoft Metashape Professional 2.2 or newer within 2.x.** 2.1 and older are refused.
  Standard has no Python scripting.
- NVIDIA GPU with CUDA and a current driver. For SAM 3 at least 16 GB VRAM is recommended.
  FlurSeg halves its window batch by itself when memory is short.
- **Python 3.12, 64-bit**, and **Git for Windows**.
- Microsoft Edge or Google Chrome for the app window. Otherwise the default browser is used.
- Disk space for the exported cloud, the classified cloud, class exports and image caches while
  a run is going. Large jobs can need several times the size of their source LAS (462 photos:
  about 15 GB). After a successful run this work data is deleted; see
  [Work data](#work-data).

`install.bat` creates `.venv`, installs CUDA PyTorch 2.7.1, the app, Transformers for FlurSeg and
the official SAM 3 code at a pinned commit, and verifies both code stacks and CUDA. It needs no
weights and downloads none. It does not change Metashape's embedded Python.
Use `install.bat --no-pause` to skip the final key press.

<details>
<summary>Manual installation, equivalent to the installer</summary>

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade "pip>=25,<27" "setuptools>=75,<81" wheel
.\.venv\Scripts\python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e ".[sam]"
.\.venv\Scripts\python -m pip check
```

</details>

## Choose a model

| | **FlurSeg v1** (default) | **SAM 3** (option) |
|---|---|---|
| Classes | 11, fixed in the weights | 18 standard classes, editable, plus your own |
| Own classes or prompts | no | yes, English prompts |
| Thin objects (kerbs, markings, covers) | weaker, sensitive to resolution | usually better |
| Weights | public, 1.26 GB | access request at Meta |
| Files | `models/flurseg/v1/` | `models/sam3.pt` |
| Name of the result in Metashape | `FlurSeg classified` | `SAM3 classified` |
| Weight licence | DINOv3 License | Meta SAM License |

Take **FlurSeg** for normal production runs. Take **SAM 3** when you need a class that FlurSeg
does not have, or when accuracy on small objects matters more than time.

### Latest Metashape benchmark (27 November 2026)

Both models processed the same **462 photos** (5280 × 3956) and **74,913,474 points**
(60,829,352 voting points) on an **NVIDIA RTX 4090**. All photos were processed fresh.
FlurSeg used the 4,000-pixel profile, FP16 and batches of four windows. SAM 3 used 75% image
scale, BF16 and batches of eight prompts (78 prompts across 18 classes).

| Step | FlurSeg v1 | SAM 3 |
|---|---:|---:|
| Preparation | 00:00:02 | 00:00:02 |
| Metashape export | 00:03:48 | 00:03:46 |
| Classification stage (including voting and post-filter) | 00:39:40 | 03:21:22 |
| ↳ Image segmentation and mask preparation | 00:22:34 | 03:03:13 |
| ↳ Point voting | 00:09:27 | 00:09:26 |
| ↳ Post-filter | 00:05:17 | 00:06:23 |
| Metashape import and final LAS | 00:01:57 | 00:02:25 |
| **Total wall time** | **00:45:28** | **03:27:35** |
| Segmentation and voting per photo | 4.16 s | 25.02 s |

On this project, FlurSeg processed about **6.0× as many photos per second** during segmentation
and voting. Its complete run took **22% of the SAM 3 time**. Segmentation includes image loading,
mask generation, cache writes and mask preparation. Classification-stage totals also include
worker setup and intermediate point-cloud output. The app estimates time from the image window
or tile count and previous runs.

These runs used the updated inference paths. They were timed before the final Ground-to-Building
roof filter correction. The measurements describe speed on this project. The two models have
different class sets and the runs were not scored for class accuracy.
[Recorded timings](docs/benchmarks/metashape-462-2026-11-27.json)

### FlurSeg resolution profiles

| Profile | Long image side | Windows for 5280×3956 | Use |
|---|---:|---:|---|
| Schnell (fast) | 3200 px | 48 | quick overview |
| **Ausgewogen** (balanced, default) | 4000 px | 70 | the same image size limit as SAMcloud |
| Hohe Genauigkeit (high accuracy) | original | 130 | recommended by the model card |

Images are never enlarged. **The reduced profiles trade accuracy on thin classes** (`curb`,
`road_marking`, `utility_cover`) for time: a marking that is a few pixels wide in the photo can
disappear when the photo is made smaller. Use high accuracy when these classes matter.

## Download FlurSeg

Model: [AIhdproduction/Flurseg](https://huggingface.co/AIhdproduction/Flurseg). Use **`v1`**.
`v2` is experimental and is not better than v1. The tool never downloads a model by itself.

### Option A: command

In PowerShell, in the repository folder, after `install.bat`:

```powershell
cd "C:\path\to\Metashape-AI-Classifier"
.venv\Scripts\hf.exe download AIhdproduction/Flurseg --revision 5df53dbbf721778f1ecd4903369e9465926ae2fd --include "v1/*" --local-dir models/flurseg
```

`--local-dir` is `models/flurseg`, without `v1`: the download creates the `v1` folder itself.

### Option B: browser

1. Open the [v1 folder on Hugging Face](https://huggingface.co/AIhdproduction/Flurseg/tree/5df53dbbf721778f1ecd4903369e9465926ae2fd/v1).
2. Download these three files with the download arrow next to each:
   `config.json`, `model.safetensors`, `preprocessor_config.json`.
3. Create the folder `models\flurseg\v1` in the repository and move the files into it. The
   browser must not rename them: `config (1).json` is not found.

### The exact folder tree

```text
Metashape-AI-Classifier/
├── install.bat
├── start_in_metashape.py
├── config/
└── models/
    ├── README.md
    ├── flurseg/
    │   └── v1/
    │       ├── config.json
    │       ├── model.safetensors          1,259,264,920 bytes
    │       └── preprocessor_config.json
    └── sam3.pt                            only for SAM 3
```

### Check the download

In PowerShell, in the repository folder. The last command must print `True`:

```powershell
(Get-Item .\models\flurseg\v1\model.safetensors).Length
(Get-FileHash .\models\flurseg\v1\model.safetensors -Algorithm SHA256).Hash -eq "83ad8990953cadbdbc4ae831eb41eeddf100312709cee8042ff412f211a789a9"
```

The app's **Modelle** page shows the same: installed or missing, the exact path, the command and
whether the weights are v1. Press **Erneut prüfen** after a download. Other weights in the same
layout, such as v2, are allowed and shown as "nicht v1 / unbekannte Gewichte". Weights with other
classes or another class order are rejected.

To keep the files elsewhere, set `paths.flurseg_model` in `config/default.yml` to the folder
that contains the three files.

## SAM 3 (optional)

Only needed when you select SAM 3. Missing SAM weights never block FlurSeg, and the reverse.

1. Visit [facebook/sam3](https://huggingface.co/facebook/sam3), request access and accept
   **Meta's SAM License**.
2. Save the **SAM 3** checkpoint as `models/sam3.pt`, by browser or with:

   ```powershell
   .venv\Scripts\hf.exe auth login
   .venv\Scripts\hf.exe download facebook/sam3 sam3.pt --local-dir models
   ```

3. Do not substitute the SAM 3.1 video checkpoint.

**Weights are not in this repository and must never be committed.**

## Prepare the project

Align the cameras, build the dense point cloud and keep valid tie points: they validate the
geometry before any GPU work.

- **Keep the depth maps** when you build the point cloud. The default visibility test uses them.
  Without depth maps set `visibility.mode: zbuffer` in `config/default.yml`.
- Save the project. The photos must still be at their stored locations. If they moved:

  ```yaml
  paths:
    image_path_remap: {"D:/old/": "E:/new/"}
  ```

- Only enabled, aligned frame cameras are processed. Fisheye, spherical, unaligned and disabled
  cameras are skipped and counted in the logs.
- Keep a normal project backup, particularly for the first real run.

## How it works

1. **Prepare:** check the options and the selected model.
2. **Export:** Metashape exports the dense cloud, camera calibrations, depth maps, transforms and
   independent tie-point projection samples.
3. **Segmentation and voting:** the geometry must pass validation before CUDA work. The model
   segments each photo, visible points collect confidence-weighted votes from several photos, the
   post-filter cleans up, and the labels are propagated to every original point.
4. **Import:** Metashape imports a separate classified asset, checks the class counts and saves
   the project. The worker assembles the final LAS 1.4 with the full codes in the project's CRS.

Only step 3 depends on the model. An export can be reused by both models; nothing after it is.
Switching model, weights or classes never reuses labels and never imports an older result.
Details: [docs/method.md](docs/method.md).

## The app

One local window replaces the two former windows. It is German by default, English under
**Einstellungen → Sprache**.

| Page | Content |
|---|---|
| **Übersicht** (Overview) | project, model switch, summary, estimate, start and cancel; on the right the four steps with percent, times and storage; at the end the result with classes, shares and output path |
| **Modelle** (Models) | both models with speed, classes, state, exact path, download link and command |
| **Klassen** (Classes) | FlurSeg's fixed list; the editor for SAM 3 |
| **Einstellungen** (Settings) | profile, options of the selected model, assists, filters, expert values, language |
| **Protokoll** (Log) | live log of the run |

- Choices apply **to this run only**. `config/default.yml` is never written.
- SAM 3 has **Auto: maximal sicher** (maximum safe): it reads the free VRAM from the NVIDIA
  driver and picks the largest conservative prompt batch of 4, 8, 12 or 16.
- Closing the window **before Start** cancels. A reload does not.
- **During a run** a reload, sleep or closed window never cancels. Cancel with **Abbrechen**,
  possible during preparation and classification, not during export and import. A cancelled run
  imports nothing.
- The app listens on `127.0.0.1` only, on a random port, protected by a token of the session. It
  loads nothing from the internet, has no telemetry, loads no model and uses no VRAM.
- Started without Metashape, it shows recent projects and lets you edit classes and check models:

  ```powershell
  .\.venv\Scripts\python.exe -m msam3.app
  ```

- To reopen the window of a running session:

  ```powershell
  .\.venv\Scripts\python.exe -m msam3.gui_progress --session "work\gui-sessions\<session>"
  ```

- `--no-browser`, or the environment variable `MSAM3_NO_BROWSER=1`, opens no window. The link
  is then in the session's `ui.json`, or printed when there is no session.

Sessions stay in `work/gui-sessions/` for "recent projects" and the time estimates. You can
delete old session folders by hand.

## Classes and LAS codes

Code 1 is always **Unclassified**: undecided or rejected points.

### FlurSeg: 11 fixed classes

They are part of the weights and cannot be changed. File: [`config/classes.flurseg.yml`](config/classes.flurseg.yml).

| ID | Class | LAS code | Colour |
|---:|---|---:|---|
| 0 | other | 64 | `#9E9E9E` |
| 1 | ground | 2 | `#A0522D` |
| 2 | paved_surface | 71 | `#5C6BC0` |
| 3 | low_vegetation | 3 | `#9CCC65` |
| 4 | tree | 5 | `#1B5E20` |
| 5 | building | 6 | `#E53935` |
| 6 | vehicle | 91 | `#FFEB3B` |
| 7 | construction | 67 | `#FF9800` |
| 8 | curb | 74 | `#00BCD4` |
| 9 | road_marking | 82 | `#FFFFFF` |
| 10 | utility_cover | 75 | `#E040FB` |

The codes follow the SAMcloud profile, with one exception: Cyclone's code for construction is
151, but **Metashape 2.3.1 drops codes 128 to 255 on import**. Code 67 is used, as for
construction machinery with SAM 3.

### SAM 3: standard classes and your own

| Class | LAS code | Meaning |
|---|---:|---|
| ground | 2 | terrain, soil, bare earth |
| low_vegetation | 3 | grass and low vegetation |
| tree | 5 | high vegetation, hedges, shrubs |
| building | 6 | building, facade, exterior wall |
| water | 9 | pond, stream, puddle |
| road | 11 | vehicle carriageway, asphalt road |
| power_line | 14 | overhead wire, cable |
| construction_machinery | 67 | excavator, loader, bulldozer |
| sidewalk | 73 | pedestrian sidewalk, paved footway |
| curb | 74 | kerb, edge stone |
| construction_material | 76 | container, construction material |
| fence | 77 | fence, barrier, railing, gate |
| traffic_sign | 81 | road sign, signpost |
| utility_pole | 84 | pole, streetlight, lamp post |
| vehicle | 91 | car, van, truck |
| scaffolding | 110 | construction scaffold |

On the **Klassen** page you can:

- add, edit and remove your own classes: name, English prompts, colour and LAS code. An empty
  code is assigned automatically (from 120) and then stays fixed. Codes must be below 128.
- edit the prompts of standard classes, or switch them off. Their names and codes stay stable.
- reset everything to the standard.

Your edits are saved in `config/classes.user.yml`, which Git ignores. `config/classes.yml` is
never rewritten. A run takes a snapshot of the classes at Start. The CLI uses your edits too,
unless you pass `--classes`. Switching off a class can switch off an assist that needs it (for
example road repair without `road`); the app says so. Your own classes get the post-filter of
small objects.

## Command line

Save and **close the project in Metashape first**; the app workflow above keeps it open instead.

```powershell
cd "C:\path\to\Metashape-AI-Classifier"
$py = ".\.venv\Scripts\python.exe"
& $py -m msam3 validate --project "D:\jobs\site\site.psx"          # geometry, needs no GPU or weights
& $py -m msam3 run --project "D:\jobs\site\site.psx"               # FlurSeg, the default
& $py -m msam3 run --project "D:\jobs\site\site.psx" --model sam3  # SAM 3
& $py -m msam3 classes --model flurseg                             # classes of a model
```

All commands (`run`, `stage1`, `stage2`, `stage3`, `validate`, `classes`) accept `--model`,
`--config` and `--classes`. All but `classes` also accept `--project`, `--chunk` (repeatable) and
`--work-dir`. Without `--chunk` the configured list is used; an empty list means all enabled
chunks. A relative `--work-dir` resolves from your current folder, configuration paths from the
repository.

```powershell
& $py -m msam3 stage1 --project "D:\jobs\site\site.psx"
& $py -m msam3 stage2 --project "D:\jobs\site\site.psx" --model sam3
& $py -m msam3 stage2 --project "D:\jobs\site\site.psx" --model sam3 --voting-only
& $py -m msam3 stage3 --project "D:\jobs\site\site.psx" --model sam3
```

- Use the **same `--model`** for stage 2 and stage 3. Stage 3 refuses a stage-2 result of another
  model, other weights, classes or settings. It needs the weights to compare them, but no GPU.
- `--classes` with FlurSeg is only accepted for a file with the same 11 classes in the same order
  (to change colours or codes).
- After an interruption run the same command again: finished images are reused. Changing voting,
  visibility or smoothing does not invalidate them. Changes to the saved project invalidate the
  export.

If Metashape is installed elsewhere, set `metashape.exe` in `config/default.yml` for the CLI.

## Results

**In Metashape:** a new point cloud named `FlurSeg classified` or `SAM3 classified` (or the
`metashape.imported_label` that you set), selected with the original photo colours. Switch to the
classification display to see the classes. A classified cloud of either model is never used as
the input of the next run.

**LAS:** the app writes beside the project

```text
<project folder>\export\punktwolke_class.las
```

With several chunks there is one LAS per chunk, named after project and chunk. A later run
replaces the file, also one of the other model. The CLI writes to
`output/<project>-<path-hash>/`. Both write:

- `class_legend.csv` (columns `name;code;color;prompts`) and `class_legend.json`, which also
  names the model of each row.
- The CLI also writes `report.json` with the model, weights checksum, coverage, counts, timings
  and validation next to the LAS.
- Logs: `ui.log`, `prepare.log`, `stage2.log` and `finalize.log` in the session; `msam3.log` in
  `work/<project>-<path-hash>/` for the CLI.

### Work data

During a run, `work/<project>-<path-hash>/` holds the exported cloud, depth maps, image caches
and class parts. As soon as the final LAS is written and checked, this folder is emptied; only the
small lock file and, for the CLI, `msam3.log` stay. This happens silently; `finalize.log` records
the freed space. The classified cloud in the Metashape project and the delivered LAS are not affected.

- A run that stops (error, cancel, PC restart) keeps its work data. Starting again then takes
  the finished photos from the cache.
- To keep the data after success, for example to rerun only the filters with `--voting-only`,
  set `paths.keep_work_data: true` in `config/default.yml`. A rerun then also skips the export.

LAS 1.4 lookup names have at most 15 characters; the full names are in the legends.

When the run is complete, the app's result page, the Metashape message and the CLI show the full
path of this file together with its checked format (LAS 1.4, point format 7). Use this file. A
LAS/LAZ export of your own from Metashape is LAS 1.2, which keeps only classes 0–31: FlurSeg
`construction` (67), for example, then appears as `low_vegetation` (3).

**Slow start of the window:** `install.bat` checks where the Python of `.venv` comes from. On a
network drive every process loaded the standard library over the network (in one office: 8.5 s
to import the window's web framework instead of 0.8 s). It then copies that Python once into
`.python-local\` (about 40 MB, without packages) and points `.venv` at it; the installed
packages stay. The previous setting is kept as `.venv\pyvenv.cfg.network`.

## Configuration and tuning

`config/default.yml` holds the defaults. The important keys:

```yaml
segmentation:
  backend: flurseg          # or sam3
paths:
  flurseg_model: "models/flurseg/v1"
  model: "models/sam3.pt"   # SAM 3
flurseg:
  resolution: balanced      # fast, balanced or accurate
  window_batch_size: 4
```

Class names and class codes in the assist and post-filter sections describe the SAM 3 classes.
With FlurSeg, [`config/profile.flurseg.yml`](config/profile.flurseg.yml) replaces those entries.
Switches and all other thresholds apply to both models, except
`visibility.min_mask_confidence`, which FlurSeg replaces with `flurseg.min_mask_confidence`.

| Aim | What to change |
|---|---|
| Shortest run | FlurSeg, profile Schnell |
| Best thin classes | FlurSeg with Hohe Genauigkeit, or SAM 3 |
| A class of your own | SAM 3, page Klassen |
| Less GPU memory | smaller window batch (FlurSeg) or prompt batch (SAM 3) |
| More cautious labels | more image observations per point, or higher `voting.min_share` |
| Fewer speckles | kNN smoothing, after checking the edges |

## Troubleshooting

| Problem | What to do |
|---|---|
| "Die Installation in .venv ist nicht aktuell" | Run `install.bat` again. It is needed after a `git pull`. |
| The window does not open, or closes unexpectedly | The message shows the end of `work/gui-sessions/<session>/ui.log`. Check that Edge, Chrome or a default browser works, then run `install.bat` again. |
| FlurSeg shows "Fehlt" (missing) | The three files must be directly in `models/flurseg/v1`, not in `v1/v1` and without renamed files. Press "Erneut prüfen". |
| FlurSeg shows "Ungültig" (invalid) | Usually an interrupted download. Download again and check the checksum. |
| "nicht v1 / unbekannte Gewichte" | Allowed, but these are not the recommended weights. |
| `sam3.pt` missing | Only relevant for SAM 3. Follow the [SAM 3 section](#sam-3-optional). |
| "depth_maps mode requires retained Metashape depth maps" | Build the point cloud with kept depth maps, or set `visibility.mode: zbuffer`. |
| "ist nicht rechtzeitig gestartet" / "reagiert nicht mehr" | The app hung and was ended (180 s without starting, or 120 s without a sign of life). See the `ui.log` named in the message, then start the script again. |
| Metashape closed or crashed during a run | The background process may still use the GPU. The app then offers **Hintergrundprozess beenden**. If its window is closed, run `.\.venv\Scripts\python.exe -m msam3.app` and open the project under "Letzte Projekte". |
| "requires CUDA" right after Start | It is checked before the export. Update the NVIDIA driver and run `install.bat` again. |
| Metashape does not respond | Normal during a run. The app shows progress and can cancel. |
| Out of VRAM | FlurSeg halves its batch by itself; at one window choose a lower profile. SAM 3: use "Auto: maximal sicher" or a smaller batch. Close other GPU programs. |
| "FlurSeg returned invalid scores" | Set the precision to fp32 under Einstellungen → Experte. |
| bf16 unsupported (SAM 3) | Set `gpu.dtype: float16`. |
| CUDA unavailable | Update the NVIDIA driver and run `install.bat` again. |
| Reprojection check fails | Check rolling shutter and calibration; enable `debug.save_overlays`. |
| Tie-to-dense check fails | Check CRS, scale and cropping of the dense cloud. |
| Image not found | Configure `paths.image_path_remap`. |
| Stage 3: "changed since stage 2" | Model, weights, classes or settings differ. Run stage 2 again. |
| Project locked | CLI: close the project in Metashape. App: keep it open. |
| "Metashape 2.2+ is required" | Your Metashape is 2.1 or older (see Help → About). Update it to 2.2 or newer. |
| No active Metashape licence | Make a Professional licence available. |
| Unsupported class code | Keep codes below 128. The stage-2 LAS remains intact. |

## Limitations and verification

Outdoor drone imagery and frame cameras only. Coverage depends on overlap and visibility; thin
objects, reflections, water and moving vegetation are difficult. FlurSeg was trained mostly on
European imagery and for views from above. The labels are not a survey certification: check the
classified cloud before delivery.

Voting representatives and their search trees need RAM; `voting.max_ram_gb` governs the vote
matrix only. Very large clouds may need larger voxels.

The latest benchmark above completed the real Metashape workflow for both models. The
automated verification record is in [docs/verification.md](docs/verification.md). Additional
dataset-specific checks are listed in [docs/method.md](docs/method.md#to-be-verified-on-first-real-run).

## Development

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

The tests need no GPU, weights, SAM 3, Metashape or browser. CI runs them on Windows and Linux.
`metashape_scripts/` runs in Metashape's embedded Python and must support **both 3.9 and 3.12**:
Metashape 2.3 embeds 3.12 according to Agisoft's changelog, and 2.2 is inferred to embed 3.9.
So: standard library and Metashape only, 3.9 syntax and no newer functions (for example no
`zip(strict=...)`).

## Licences

Original project material is licensed under the custom
**[AIhdproduction Education and Research License 1.1](LICENSE)**. AIhdproduction retains its
rights. This is source-available software with purpose restrictions.

- **Free without an individual request:** noncommercial private use, teaching, coursework, theses
  and research, including modification and free redistribution under the licence conditions.
  Private use means an individual's personal purposes, such as hobby projects or their own home,
  without payment and not for a business, employer or client.
- **Prior written permission required for every other use,** for example commercial use, paid
  client work, internal business use, commercial R&D, products, hosted services and commercial
  exploitation of results. Academic status alone does not exempt a commercial project.
- **Your data and results:** AIhdproduction claims no ownership of them. Results of noncommercial
  private use, education and research may be published and shared; the software licence does not
  automatically become the licence of those results. Commercial generation or exploitation of
  results by the licensee requires permission.
- **Requests:** through the
  [issue tracker](https://github.com/AIhdproduction/Metashape-AI-Classifier/issues). Permission
  must be granted expressly in writing before the proposed use.

The complete [LICENSE](LICENSE) controls; a [German explanation](docs/licensing.de.md) is
available.

Third-party components keep their own terms:

- **FlurSeg weights:** [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)
  ([copy](licenses/FLURSEG_DINOV3_LICENSE.md)).
  **Built with DINOv3.** The architecture is [EoMT](https://github.com/tue-mps/eomt) (MIT).
- **SAM 3 code and weights:**
  [Meta's SAM License](https://github.com/facebookresearch/sam3/blob/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc/LICENSE)
  ([copy](licenses/SAM3_LICENSE.txt)).
- Both models must be acknowledged in research publications that use them. Using only generated
  outputs does not exempt the person running a model from its terms.
- **NiceGUI** (MIT) and **Transformers** (Apache-2.0). Metashape needs its own licence.
- SegEarth-OV3 provided scientific inspiration for the SAM 3 fusion; none of its source code is
  included.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
