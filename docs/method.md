# Method, API verification and operational constraints

## Sources and scope

- [Metashape Python Reference 2.2.2](https://www.agisoft.com/pdf/metashape_python_api_2_2_2.pdf).
- [Metashape Professional manual 2.2, Appendix D](https://www.agisoft.com/pdf/metashape-pro_2_2_en.pdf),
  printed pages 242–243, camera axes, raw pixels and Brown frame projection.
- [Agisoft headless scripting instructions](https://agisoft.freshdesk.com/support/solutions/articles/31000133141-how-to-run-the-script-in-headless-mode-from-the-command-line).
- [SegEarth-OV3 paper](https://arxiv.org/abs/2512.08730) and
  [reference segmentor](https://github.com/earth-insights/SegEarth-OV-3/blob/main/segearthov3_segmentor.py).
- [FlurSeg model card](https://huggingface.co/AIhdproduction/Flurseg) (revision
  `5df53dbbf721778f1ecd4903369e9465926ae2fd`), [EoMT](https://github.com/tue-mps/eomt) and
  [Transformers EoMT-DINOv3](https://huggingface.co/docs/transformers/model_doc/eomt_dinov3).
- [SAM 3 model builder](https://github.com/facebookresearch/sam3/blob/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc/sam3/model_builder.py),
  [processor](https://github.com/facebookresearch/sam3/blob/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc/sam3/model/sam3_image_processor.py),
  [image model](https://github.com/facebookresearch/sam3/blob/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc/sam3/model/sam3_image.py),
  [segmentation head](https://github.com/facebookresearch/sam3/blob/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc/sam3/model/maskformer_segmentation.py).
- [ASPRS classification lookup registry](https://github.com/ASPRSorg/LAS/wiki/Standard-VLRs).

The requested architecture is retained. Verified API behavior required three clarifications:
Metashape 2.2+ is the supported 2.x subset; Windows can run headlessly with `-r` alone;
final LAS output requires class-partition assembly to preserve custom codes.

## Coordinates and camera model

Chunk internal coordinates are transformed in float64. For georeferenced projects, stage 1
exports EPSG:4978. `chunk.transform.matrix` maps internal points into the chunk datum's
geocentric frame. When that datum differs from WGS 84, stage 1 composes the documented
`CoordinateSystem.datumTransform` matrix and records `export_transform` separately from
`transform`. Independent `CoordinateSystem.transform` sample checks must agree within 1 mm.
A non-affine datum transformation exceeding that limit is rejected rather than approximated
silently. Local projects use their original CRS and its linear unit conversion.

External LAS points pass through `inverse(export_transform)` in float64, then become float32
coordinates relative to a local internal origin, multiplied by metres per internal unit.
Voxel sizes, spatial blocks and visibility tolerances consequently use **metres**, including
projects with nonunit chunk scales. The camera matrix is shifted and scaled in float64 before
GPU conversion, preventing ECEF-sized float32 cancellation. Shear and nonuniform chunk scale
are rejected.

For camera-space `(X,Y,Z)`, `x=X/Z`, `y=Y/Z`, `r2=x*x+y*y`. The model applies four radial
terms, the specified tangential terms with the `1+p3*r2+p4*r2*r2` multiplier, and then focal,
principal-point, affinity and skew terms. Points with nonpositive depth, nonfinite projections
or coordinates outside the sensor dimensions are invalid. NumPy and torch implementations
share the scalar distortion expression and have independent numerical tests.

The manual prints the usual p1/p2 model; p3/p4 are listed in the Python calibration API.
A local 2.3.1 `Calibration.project` probe verified the extended expression: for width 800,
height 600, f=500, p1=.002, p2=.003, p3=.3, p4=.2 and camera point `(1,2,10)`, both
implementations return `(450.132015,400.2386425)`.

A separate local `Camera.project` probe returned identical pixels for orientation flags
1, 3, 6 and 8. Images are loaded in their raw stored orientation, without EXIF transpose.
Image dimensions must match adjusted calibration dimensions. Debug overlays display the
predicted sample positions on those same raw pixels. Rolling-shutter sensors produce warnings;
the frame-model validation gate must still pass.

## Validation gate

Up to 500 valid tie points are selected by deterministic reservoir sampling over the complete
tie cloud. Up to 20 cameras are spread across the exported camera list. Stage 1 records both
internal/world coordinates and `Camera.project` pixels. Stage 2 reproduces all recorded
projections, stopping above 0.5 px maximum error. Missing samples are an error, not a pass.

The transformed samples must lie inside the exported LAS bounds, with a two-quantization-step
boundary tolerance. Their exact median nearest distance to the full dense cloud must be at
most 0.10 m. A temporary cKDTree for each streamed LAS block is queried for every sample;
minimum distances are accumulated across blocks, avoiding a full dense-cloud tree in RAM.
Every selected chunk passes this gate before any model is loaded.

That CUDA exists is checked much earlier, without loading a model: by the worker's `prepare`
and by `run` and `stage2` of the CLI, so a PC without CUDA fails before the long export. The app
process never queries CUDA.

## Model selection

`msam3.backends` is the one registry that the Metashape worker, the CLI driver, the pipeline and
the app use. For each model it supplies the file and package check, the weights hash, the cache
and run identity, the class set, the post-filter profile and the backend. `segmentation.backend`
or `--model` selects it; FlurSeg is the default.

Stage 1 does not depend on the model. Its identity holds only the project, chunks, depth-map
choice and the ground-assist settings of `default.yml`, so both models share an export. The
class-dependent settings of FlurSeg are applied later, inside stage 2. Everything after the
export records and checks the model.

## FlurSeg adapter

FlurSeg v1 is `EomtDinov3ForUniversalSegmentation` (image size 768, 100 queries), trained with
Transformers 5.22.1. It is loaded with `local_files_only=True`, without remote code and only from
safetensors. Missing, unexpected or mismatched keys are errors, so a wrong file cannot leave
randomly initialised layers. `id2label` must contain the eleven names in their trained order.
Production requires CUDA; only tests construct the backend with a CPU model.

Photos are limited by their long side (fast 3200, balanced 4000, accurate original) with
bilinear, antialiased reduction, and are never enlarged. RGB is divided by 255 and normalised
with the preprocessor's ImageNet mean and standard deviation. Windows are 768 pixels with stride
384; the last window is aligned to the image edge. Missing pixels of small images are zero after
normalisation, as in training.

Per window the scores are `einsum("qc,qhw->chw", class_logits.softmax(-1)[..., :-1],
mask_logits.sigmoid())` at native mask resolution, then bilinearly upsampled to the window,
multiplied by a Hann window plus 1e-6 and accumulated. The confidence is the winner's share of
the eleven accumulated scores. Below `flurseg.background_threshold` (0.3) a pixel is 255.

A full score image of a 45 MP photo would need about 2 GB of VRAM. Instead the device keeps a
band of one window height × image width × 11. After each row of windows, the GPU decides the rows
that no later window can touch, and the band moves down. If the RGB input and the label and
confidence maps fit within 128 MiB, the GPU buffers them so transfers happen once per image
instead of once per strip. Larger images stream by row.

Inference uses float16 autocast by default (`flurseg.precision`), with converted weights reused
within an image and scores computed in float32. On CUDA out of memory, a buffered run first
retries by streaming rows. A streaming run halves the window batch and retries. At one window,
the run stops with a clear error. The batch is not part of the cache identity.

The exported v1 weights keep a mask-annealing probability of 1e-4 in the last block. Transformers
then rarely masks a query at random, as in the owner's reference code, so two fresh inferences can
differ in a few pixels. Cached labels are stable.

Thresholds: on the test tile a normalised winner share of 0.5 rejected up to 16 % of a small
class's pixels, while 0.4 rejected at most 6 % (markings, kerbs and covers were not in it). So
FlurSeg votes from 0.4 (`flurseg.min_mask_confidence`), the label threshold that SAMcloud uses.
`voting.min_share` stays 0.4 because it is a share of the accumulated votes, not a model score.

## SAM 3 raw-output adapter

The optional package is pinned to revision `0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`.
`build_sam3_image_model(checkpoint_path=..., load_from_HF=False, enable_segmentation=True,
enable_inst_interactivity=False, compile=False)` loads the local checkpoint. The model is
moved explicitly to the selected CUDA device and put in evaluation mode.

The public `Sam3Processor.set_image` computes image features once per tile. The adapter calls
`model.backbone.forward_text(prompts, device=...)` once per prompt list and retains the text
features across tiles. It updates `backbone_out` and constructs `sam3.model.data_misc.FindStage`
with repeated image index zero and the batch's text indices. `model.forward_grounding` receives
this object, `model._get_dummy_prompt(num_prompts=batch_size)` and `find_target=None`. These
internals replace the reference's modified processor. No upstream package files are patched.

Raw keys are `semantic_seg` (B×1×H×W), `pred_masks` (B×N×H×W), `pred_logits` (B×N×1)
and `presence_logit_dec` (one value per prompt). Shapes are checked, logits receive sigmoid,
and mask logits are bilinearly resized with `align_corners=False` before sigmoid. Unexpected
joint instance/presence scoring is rejected to avoid applying the presence score twice.

For each prompt, retained instances have sigmoid confidence at least 0.3. Their probability
masks are weighted by instance confidence and combined by a maximum. The result is maximized
with the semantic probability and multiplied by sigmoid presence exactly once. Synonyms
are merged by per-class maximum. The final pixel argmax uses all classes and a 0.3 minimum
confidence; otherwise the cached index is 255 (unknown).

Tiles are 1008×1008 with 20% overlap. Edge tiles retain all image pixels. Overlapping
**per-class probabilities are maximized before argmax** as requested; the reference segmentor
averages overlaps, so this is a documented task-specific difference. Per-class image scores
use a CPU disk mapping. The GPU resizes up to eight retained instance masks at once and merges
each group into the running maximum. All inference uses `torch.inference_mode` and bf16/fp16 CUDA
autocast. The shipped default.yml runs eight prompts at a time. A smaller
`sam.prompt_batch_size` reduces memory use. Batched grounding and all four raw output shapes
were checked on the local checkpoint with a real 1008-pixel drone-image tile on an RTX 4090.
CUDA cache is released between newly segmented images.

## Spatial reduction, visibility and votes

Voxel keys are computed in metric coordinates and distributed over disk buckets. NumPy
selects the first original index per voxel in each bucket, including across I/O batch
boundaries. Bucket arrays and voting-cloud indices need RAM; raise the bucket count or voxel
size for exceptionally large data. The full original cloud is a disk mapping.

Spatial blocks have 10 m edges. Blocks behind the camera are rejected. For an undistorted
camera, the projected bounding rectangle of the block is compared with the image rectangle;
a block containing the entire image must remain even when none of its corners is inside.
Distorted blocks currently use conservative depth-only culling to avoid false exclusions from
nonlinear distortion. This costs extra projections but preserves correct visibility.

The GPU z-buffer uses `scatter_reduce_(reduce="amin")`, reduced image resolution, and a square
splat footprint whose configured size is a **radius in z-buffer pixels**. All surviving
blocks contribute before any votes are read. This prevents a far block from voting before
its occluder has been processed. Visibility uses metric camera Z and the specified relative
and absolute tolerances. Optional retained depth maps use their own calibration, raw internal
Z values scaled to metres, and reject missing/zero/nonfinite depth.

Known pixel observations add quantized confidence times view weight to float32 class totals
and increment the valid-view counter. The default weight is 1. Incidence mode requires
normal-vector LAS extra dimensions; normals are transformed as covectors, normalized and
compared with the viewing ray. Missing or invalid normals fail explicitly.

The vote matrix becomes a NumPy memmap when it exceeds `max_ram_gb`. Decisions are made in
batches: one positive known observation is sufficient, and the winning class is the class with
the largest accumulated confidence across all visible images; its share must be at least 0.4.
Repeated observations therefore add confidence to the same class. Optional kNN smoothing changes only low-share already-decided representatives
when neighbours have a strict majority. A chunked cKDTree query propagates the resulting
codes to every original point, preserving source point order and attributes in stage 2.

## Post-filter

After voting, the enabled `postprocess` configuration applies an NPI-style neighbourhood filter
to the representative cloud. Per-class rules remove low-confidence isolated points and fill only
small, low-confidence holes surrounded by a sufficient share of one class. The road rule rejects
local planes steeper than 40 degrees before filling road holes. The last step reclassifies Ground
inside a flat or pitched roof plane as Building. It requires nearby Building points on the same
face to surround the candidate, an agreeing surface normal, and lower terrain in several nearby
cells. It reads the existing Building labels for a single pass and uses the true vertical. A report
is written below each chunk as
`post_sam_filter/postprocess_report.json` before labels are propagated back to the full cloud.

### Ground from a terrain model

Photogrammetry sees the top of a meadow, not the soil, so a model labels most of it low
vegetation. The last post-filter step (`postprocess.dtm_ground`, both models) builds a terrain
model from all ground and low-vegetation points: per 1 m cell the point at the 5th height
percentile along the vertical estimated from the cloud; cells far below (25 cm) or above (1 m)
their neighbours are dropped; each remaining node gets a plane through its neighbours. Low
vegetation at most 5 cm above that plane becomes ground. In the 291-photo project this raised
ground from 6% to 54% of the voting points in 9 s; the remaining low vegetation stands higher than
the meadow surface. SAM 3 labels these meadows low vegetation as well, so agreement with SAM 3 is
no measure for this step.

### Small flat classes

`postprocess.compact_classes` checks classes of small, flat objects; by default
`utility_cover` (manhole and drain covers) for both models. Points closer than 10 cm form one
object, measured along the vertical from the cloud: length along its main horizontal axis (2nd
to 98th percentile) between 0.2 and 1.5 m, height span (5th to 95th percentile) at most 15 cm.
Other groups take the most common class of their neighbours within 0.5 m at the same height
(±20 cm), or become unclassified. In the 291-photo project FlurSeg labelled 160 groups; 75
fragments, 10 groups longer than 1.5 m (strips of up to 5 x 0.9 m along facades, which hold
most cover points) and 12 raised groups were moved, 63 cover objects stayed.

SAM 3 can only find what its class list names. Its default list had neither covers nor road
markings, so SAM 3 runs never produced them. `config/classes.yml` now has both, with the LAS
codes and colours of FlurSeg: `road_marking` (82; road marking, painted road line, zebra
crossing, painted arrow on road) and `utility_cover` (75; manhole cover, utility cover, drain
grate, sewer grate, gully cover, valve cover). Kerbs, markings and covers are protected for
SAM 3 as for FlurSeg: no mask erosion, never overwritten by road repair, hole filling or
geometry rules, two neighbours suffice for the cluster filter, and they support the DTM.
The new classes change the SAM 3 segmentation identity: the next SAM 3 run segments all photos
again, and the ten extra prompts add to its time per photo.

### Per-class geometry review

`postprocess.geometry_rules` runs last and holds one rule per target class. A rule names the
classes it may take points from (`source_classes`, plus unclassified with `include_unknown`).
A candidate moves to the target only when all of these hold:

- at least 8 of its 16 nearest target points (thinned to one per 12 cm voxel) lie within the
  search radius, and they span a plane (second eigenvalue at least 8 cm);
- the candidate lies on that plane (12 cm; unclassified 8 cm) and its own Metashape normal
  agrees with the plane normal (|cos| at least 0.8; unclassified 0.85);
- unclassified points must also touch the surface (nearest target point within 20 cm) and be
  surrounded by it: target points in at least 7 of 8 directions, measured inside the plane. A
  hole in a facade is filled; the plane is never extended past its edge;
- the candidate is more than 25 cm away from a protected thin class (kerb, marking, cover).
  Protected classes are never sources, and a rule for a protected target is ignored.

`surfaces: [horizontal, vertical]` fits roofs and facades separately (normal within 56.6
degrees of vertical counts as horizontal). `source_surfaces` restricts a source class to points
whose own normal faces one way. All rules read the labels from before the step; a point
accepted by two targets goes to the closer plane. Without normals in the source LAS the step is
skipped with a warning.

The vertical comes from the point cloud itself, for every project, georeferenced or not.
Metashape's internal axes and the project CRS are not trusted:

1. PCA normals of planar neighbourhoods are computed from the coordinates of 200,000 sample
   points (neighbours from a 2-million-point subsample, 0.5 m radius).
2. The most common normal direction (terrain, roads, roofs) gives a first estimate.
3. Walls are vertical also on a hillside. When the cloud has walls in at least two directions
   (found within 40 degrees, then 15 degrees of the horizontal), the vertical is the direction
   perpendicular to all of them.
4. The camera centres lie above the scene and fix the sign.

On the 291-photo project this took 2.3 s and lay 0.5–0.6 degrees from the vertical of the
geocentric export; terrain alone would have leant 2.7 degrees with the slope. The report
`post_sam_filter/postprocess_report.json` records the method, the tilt of the internal frame and
the angle to the CRS vertical. Only a cloud without planar surfaces falls back to the CRS.

`default.yml` (SAM 3) restores only unclassified building points through the per-class rules.
Metashape's internal frame is not levelled: 31.7 degrees in the 291-photo project.
The older filters (DTM height and road slope) still work in that frame because their thresholds were calibrated there;
`postprocess.level_to_vertical: true` rotates all of them to the true vertical.

### FlurSeg profile

The class names and codes in `default.yml` are those of SAM 3. `config/profile.flurseg.yml`
replaces them while FlurSeg is selected:

- rules for all eleven classes; `paved_surface` (71) takes the place of road;
- ground assist chooses between `ground` and `paved_surface`; vegetation assist is unchanged;
- trusted DTM codes are 2, 71, 82, 75 and 74;
- planar filling uses ground, paved surface and building.
- a colour check restores non-green `low_vegetation` to `ground` only where Metashape's
  source classified the point as terrain; a 0.75 m 3D building-neighbour check restores
  nearby `other` points to `building`, except within 25 cm of a kerb, marking or cover.
  Both rules can be disabled in the profile;
- the geometry review also takes `other` on facades and `construction` on roofs to `building`.
  Flat `other` and facade `construction` (often scaffolding) stay, and `low_vegetation` is never
  a source: against SAM 3 it was almost always wrong. Measurements are in
  [the benchmark](benchmarks/flurseg-filter-291-images.md).

`curb`, `road_marking` and `utility_cover` are thin or small and lie in the road plane. Four
protections keep the road from absorbing them: their masks are not eroded before voting (two
pixels on each side would remove a whole marking); road repair and hole filling never overwrite
them; the cluster filter needs two same-class neighbours for them instead of four; kNN smoothing
never changes them. A single marking point without neighbours is still removed and may then be
filled as pavement.

### SAM 3 user classes

`config/classes.user.yml` stores the edits as a difference to `config/classes.yml`. A step whose
class is switched off is skipped, and so are rules for missing classes. Own classes without a
rule get the rule of a small object (2 same neighbours, fill from 5 at 70%). Without user edits
no setting differs from the SAM-only version. The ground classification before export still
runs when the ground class is switched off; only the assist after voting is then inactive.

## Cache and stage integrity

Each camera has an atomic PNG label/confidence pair and a JSON sidecar written last. The
sidecar records the weights hash (for FlurSeg also both JSON files), a hash of backend, backend
version, model options and classes, image path, size, modification time,
dimensions, orientation and PNG checksums. Images are cached at half raw resolution by
default. Both arrays use identical nearest-neighbour resampling; confidence is not blended
across labels. Corrupt, incomplete or stale caches are recomputed, or rejected in voting-only
mode. Voting settings do not enter the segmentation hash. Each model has its own cache folder.

Stage 2 records backend, weights hash, segmentation identity and a hash of the result-affecting
visibility, voting, assist, smoothing and post-filter settings. Stage 3 compares all of it, and
the classes, before an import. The import ID contains backend and weights, so identical labels
of two models still become two correctly named assets.

The job file contains resolved settings and paths. Stage 1 reuse additionally requires a
matching saved-project fingerprint and all exported package files. Stage 3 checks the source
cloud key/count, transform, CRS and camera records against stage 1. A per-work-directory OS
lock prevents overlapping msam3 processes. Metashape lock files are never removed or ignored.
Subprocess success requires both zero exit status and a fresh completion marker. Logs are
captured and diagnostic tails included on failure.

### Embedded Python versions

The scripts in `metashape_scripts/` support Python 3.9 and 3.12, because the supported Metashape
versions differ. Agisoft's changelog documents Python 3.12 from 2.3.0 build 21378 (preview).
It lists no Python update between 2.1 and 2.3, so 3.9 for 2.2.x is an inference from that and
from the 3.9.13 observed in an installed 2.1.0. No 2.2.x interpreter has been observed.

### Metashape GUI launcher

`start_in_metashape.py` is a second entry point for users who prefer **Tools → Run Script**.
It deliberately does not use a second Metashape process or a custom Qt event loop. The open GUI
document saves once, runs the existing stage-1 exporter against that document, waits while a
separate `.venv` Python process performs stage 2 only, then runs stage 3 against the same open
document. The worker never imports Metashape and never opens the project. This prevents
project-lock contention and avoids nested Qt loops in Metashape's embedded interpreter. The
script selects the verified imported asset and uses its configured display mode after stage 3.
The default `image_colors` mode shows the photo colours; `classification` uses Metashape's
built-in class-colour view.

Options, progress and result are one NiceGUI app in a separate process. The launcher starts
`msam3.gui_options --session <dir>` and continues as soon as that process has written
`options.json`; the app keeps running. All communication is by files in the session:

| File | Writer | Meaning |
|---|---|---|
| `session.json` | launcher | project, image sizes, PID of Metashape |
| `options.json` | app | the choice at Start, with the class snapshot for SAM 3 |
| `options_cancelled.json` | app | cancelled before Start |
| `launcher_status.json` | launcher | phase and the time of each phase |
| `job.json` | worker | settings, model and classes of the run |
| `worker.json`, `<action>_status.json` | launcher | running and finished worker |
| `cancel_requested.json` | app | cancel request |
| `finalize_done.json` | worker | outputs and class counts |
| `ui.json` | app | local link with the session token; its existence means the app is up |
| `ui_alive` | app | touched every second while monitor and event loop respond |
| `ui.log`, `prepare.log`, `stage2.log`, `finalize.log` | processes | logs |

Before the app starts, `msam3.install_check` detects a `.venv` that is older than the checkout.
If the app ends without a choice, the launcher reports the end of `ui.log` instead of waiting.
It also ends and reports an app that has no `ui.json` after 180 seconds, or whose `ui_alive`
stands still for 120 seconds before Start. Time in which the PC sleeps is not counted, and an app
that is alive may wait for the user without limit. The install check is ended after 180 seconds.

A cancel request is checked again immediately before the export. An interrupted launcher ends
its worker. If Metashape itself died, the app offers to stop the worker, but only for a running
process whose command line names `msam3.gui_worker` and this session.
The app ends itself when no window connects within 120 seconds, when the window is closed
before Start, or after the final state once the window is closed (after two minutes if nobody
saw that state). During a run it never ends or cancels because a socket was lost.

The app offers **Abbrechen** while the independent preparation or classification worker is
running. It terminates that worker, leaves the original point cloud selected and imports
no partial asset. A successful one-click run writes the lossless LAS file beside the saved project
as `export/punktwolke_class.las`, then imports the same classification into Metashape. Temporary
work data remains in the repository work directory, rather than beside the project.

Each semantic class and the unknown class has a `#RRGGBB` `color` in its class file. These are
reference colours for the legends and the app. The LAS keeps the original photo RGB.

## Metashape calls and attributes used

All entries below refer to the named sections in the 2.2.2 Python Reference; page numbers
are printed pages, not PDF viewer indices. Metadata key access was also probed on 2.3.1.

| API | Reference section |
|---|---|
| `Metashape.app.version` | Application, p.17 |
| `Metashape.Document()`; `open(read_only=False, ignore_lock=False)` | Document, pp.82–84 |
| `document.chunks`, `findChunk(key)`, `save()` | Document, pp.83–85 |
| `chunk.enabled`, `key`, `label`, `cameras`, `crs` | Chunk properties |
| `chunk.transform.matrix` | Chunk / Transform |
| `chunk.point_cloud`, `point_clouds`, default assignment | Chunk properties |
| `chunk.tie_points.points`; `point.valid`, `coord` | TiePoints, pp.296–300 |
| `camera.enabled`, `transform`, `photo.path`, `calibration` | Camera / Photo |
| `camera.sensor`, `orientation`, `project(point)` | Camera, pp.25–27 |
| `Sensor.Type.Frame`, `sensor.calibration`, `rolling_shutter` | Sensor, pp.142–147 |
| `Shutter.Model.Disabled` | Shutter, p.151 |
| calibration width/height, f/cx/cy/b1/b2/k1–k4/p1–p4 | Calibration, pp.19–23 |
| `CoordinateSystem(...)`, `wkt`, `geoccs` | CoordinateSystem, pp.77–80 |
| `CoordinateSystem.datumTransform`, `transform` | CoordinateSystem, pp.78–79 |
| Matrix multiplication/indexing, `mulp`; Vector indexing, `norm` | Matrix / Vector |
| `chunk.exportPointCloud` with explicit cloud keys | Chunk, pp.49–50 |
| `chunk.importPointCloud(..., replace_asset=False, precision=...)` | Chunk, pp.59–60 |
| `PointCloudData`, `PointCloudFormatLAS` | DataSource / PointCloudFormat |
| point-cloud key/label/meta/count/count_by_class | PointCloud, pp.132–133 |
| `MetaData.keys()` and string key read/write | Metadata mapping; live 2.3.1 probe |
| `chunk.depth_maps`, `keys()`, camera lookup | DepthMaps, pp.81–82 |
| `DepthMap.calibration`, `image()`, `Image.save(path)` | DepthMap / Image |

Export explicitly disables boundary and region clipping and requests color, classification,
confidence and normals. Import creates an additional asset, specifies 1 mm or finer precision,
and verifies class histograms before saving. `point_clouds=[asset.key]` selects the asset for
export without changing the default. Native final class partitions are finalized externally
because embedded scripts intentionally import only the standard library and Metashape.

## Custom LAS codes: observed behavior

On the local Metashape 2.3.1 installation, an isolated synthetic LAS 1.4/format-7 cloud
containing codes 0–255 imported only codes 0–127. Native LAS export produced LAS 1.2/format 2
with five-bit classifications. Separate filters for codes 64, 65, 66 and 72 selected the
correct points, despite truncation in the resulting LAS records. Original/default asset
restoration using `chunk.point_cloud = original` was verified.

Consequently, stage 3 rejects observed codes ≥128, imports the remaining classes, checks
every count, and exports one partition per code. The external writer converts partitions to
LAS 1.4 format 7/8 with known full class codes and a common coordinate quantization, then
writes the lookup VLR and full JSON/CSV legends. Input RGB, standard attributes and extra
dimensions are preserved where representable; legacy scan angles are converted to format-7
units. LAS 1.4 format-7 CRS metadata uses WKT. Legacy lookup names have a 15-byte limit;
the full names are available in legends.

Metashape writes the whole code into the format-2 byte that holds a five-bit class and the
synthetic, key-point and withheld flags: 67 reads as class 3 with the key-point flag, codes from
96 also set synthetic and from 128 withheld. The writer clears these three flags for parts with
codes of 32 or more. After writing, it checks the header (LAS 1.4, point format 6 or higher,
point count) and the first point of every class part. The app, the Metashape completion message
and the CLI name this file by its full path. A LAS/LAZ export of your own from Metashape is again
LAS 1.2 and truncates codes above 31, so FlurSeg `construction` (67) appears as
`low_vegetation` (3). A final name ending in `.laz` is compressed with `lazrs` when installed
and refused otherwise; `install.bat` installs `lazrs`.

## To be verified on first real run

These checks are specific to the real dataset, installed build and licensed model:

1. Run `validate`; inspect mean/max reprojection error, tie-to-dense distance and overlays,
   especially rotated imagery and any rolling-shutter camera. Do not override a failed gate
   without understanding its cause.
2. On a non-WGS datum or vertical-geoid project, verify the exported ECEF samples and final
   project-CRS coordinates against control points. Check geoid availability in headless mode.
3. SAM 3: check semantic/instance/presence tensor shapes and probabilities with the authorized
   local checkpoint; run a representative real image and inspect labels at tile seams and thin
   objects. FlurSeg: confirm in `stage2.log` that no invalid-score or batch warning appears with
   torch 2.7.1+cu128 and fp16, then inspect markings, kerbs and covers in the point cloud and
   compare the profiles balanced and accurate.
4. Reopen the saved project, verify the original cloud remains, inspect codes 64–72 in
   Classification view, and confirm which custom names the installed GUI displays.
5. Compare final LAS point/class counts with the stage-2 report in independent LAS software.
   Check native-export coordinate precision against delivery requirements.
6. If retained depth maps are enabled, compare calibrated depth sampling on known visible and
   occluded points. If incidence weighting is enabled, verify the exported normal dimension
   names and direction convention. These optional paths need dataset-specific evidence.
7. Interrupt a real multi-image job and resume; verify only unfinished images invoke the model.
   Benchmark RAM, disk, GPU use and throughput before a 100-million-point production run.
8. With the app: check that the window opens from Metashape, that Metashape continues after
   Start, and that a second run in the same Metashape works and starts from the photo cloud.

The automated synthetic suite verifies the mathematical pipeline and >95% known-label
recovery. It does not establish semantic accuracy for a particular survey site.
