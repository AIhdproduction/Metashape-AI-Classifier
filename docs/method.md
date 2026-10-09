# Method, API verification and operational constraints

## Sources and scope

- [Metashape Python Reference 2.2.2](https://www.agisoft.com/pdf/metashape_python_api_2_2_2.pdf).
- [Metashape Professional manual 2.2, Appendix D](https://www.agisoft.com/pdf/metashape-pro_2_2_en.pdf),
  printed pages 242–243, camera axes, raw pixels and Brown frame projection.
- [Agisoft headless scripting instructions](https://agisoft.freshdesk.com/support/solutions/articles/31000133141-how-to-run-the-script-in-headless-mode-from-the-command-line).
- [SegEarth-OV3 paper](https://arxiv.org/abs/2512.08730) and
  [reference segmentor](https://github.com/earth-insights/SegEarth-OV-3/blob/main/segearthov3_segmentor.py).
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
Every selected chunk passes this gate before CUDA work begins.

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
use a CPU disk mapping, while instance resizing is sequential to avoid hundreds of full-size
GPU masks. All inference uses `torch.inference_mode` and bf16/fp16 CUDA autocast. Prompts are
batched in groups of four by default (`sam.prompt_batch_size`); set this to one to reduce
memory use. Batched grounding and all four raw output shapes were checked on the local
checkpoint with a real 1008-pixel drone-image tile on an RTX 4090. CUDA cache is released
between newly segmented images.

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

## Post-SAM geometry filter

After voting, the enabled `postprocess` configuration applies an NPI-style neighbourhood filter
to the representative cloud. Per-class rules remove low-confidence isolated points and fill only
small, low-confidence holes surrounded by a sufficient share of one class. The road rule rejects
local planes steeper than 40 degrees before filling road holes. The building rule changes a point
labelled ground to building only when it is horizontally close to several building points at least
one metre below it, which protects flat roofs. A report is written below each chunk as
`post_sam_filter/postprocess_report.json` before labels are propagated back to the full cloud.

## Cache and stage integrity

Each camera has an atomic PNG label/confidence pair and a JSON sidecar written last. The
sidecar records model SHA-256, segmentation/classes hash, image path, size, modification time,
dimensions, orientation and PNG checksums. Images are cached at half raw resolution by
default. Both arrays use identical nearest-neighbour resampling; confidence is not blended
across labels. Corrupt, incomplete or stale caches are recomputed, or rejected in voting-only
mode. Voting settings do not enter the segmentation hash.

The job file contains resolved settings and paths. Stage 1 reuse additionally requires a
matching saved-project fingerprint and all exported package files. Stage 3 checks the source
cloud key/count, transform, CRS and camera records against stage 1. A per-work-directory OS
lock prevents overlapping msam3 processes. Metashape lock files are never removed or ignored.
Subprocess success requires both zero exit status and a fresh completion marker. Logs are
captured and diagnostic tails included on failure.

### Metashape GUI launcher

`start_in_metashape.py` is a second entry point for users who prefer **Tools → Run Script**.
It deliberately does not use a second Metashape process or a custom Qt event loop. The open GUI
document saves once, runs the existing stage-1 exporter against that document, waits while a
separate `.venv` Python process performs stage 2 only, then runs stage 3 against the same open
document. A separate topmost Windows progress window displays the image count, percentage,
elapsed time and estimate of remaining time while the GPU is working. The worker never imports
Metashape and never opens the project. This prevents project-lock contention and avoids nested Qt
loops in Metashape's embedded interpreter. The GUI script selects the verified imported asset and
uses its configured display mode after stage 3. The default `class_palette` mode shows RGB values
from `config/classes.yml` in `PointCloudViewColor`; `classification` retains Metashape's built-in
class-colour view.

The status window offers **Abbrechen** while the independent preparation or GPU-classification
worker is running. It terminates that worker, leaves the original point cloud selected and imports
no partial asset. A successful one-click run writes the lossless LAS file beside the saved project
as `export/punktwolke_class.las`, then imports the same classification into Metashape. Temporary
work data remains in the repository work directory, rather than beside the project.

Each semantic class and the unknown class has a `#RRGGBB` `color` in `config/classes.yml`. Those
values are written into the LAS RGB dimensions during classification and final class-partition
assembly, so the same palette is present in the standalone LAS and in the imported asset.

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

## To be verified on first real run

These checks are specific to the real dataset, installed build and licensed model:

1. Run `validate`; inspect mean/max reprojection error, tie-to-dense distance and overlays,
   especially rotated imagery and any rolling-shutter camera. Do not override a failed gate
   without understanding its cause.
2. On a non-WGS datum or vertical-geoid project, verify the exported ECEF samples and final
   project-CRS coordinates against control points. Check geoid availability in headless mode.
3. Check semantic/instance/presence tensor shapes and probabilities with the authorized local
   checkpoint; run a representative real image and inspect labels at tile seams and thin objects.
4. Reopen the saved project, verify the original cloud remains, inspect codes 64–72 in
   Classification view, and confirm which custom names the installed GUI displays.
5. Compare final LAS point/class counts with the stage-2 report in independent LAS software.
   Check native-export coordinate precision against delivery requirements.
6. If retained depth maps are enabled, compare calibrated depth sampling on known visible and
   occluded points. If incidence weighting is enabled, verify the exported normal dimension
   names and direction convention. These optional paths need dataset-specific evidence.
7. Interrupt a real multi-image job and resume; verify only unfinished images invoke SAM 3.
   Benchmark RAM, disk, GPU use and throughput before a 100-million-point production run.

The automated synthetic suite verifies the mathematical pipeline and >95% known-label
recovery. It does not establish semantic accuracy for a particular survey site.
