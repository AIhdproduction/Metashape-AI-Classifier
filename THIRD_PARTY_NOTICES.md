# Third-party notices

## Original project development

The project-specific application code was independently developed for AIhdproduction's
Metashape AI Classifier (formerly Metashape SAM 3 Point Cloud Classifier). This includes the
Metashape workflow, the point-cloud classification pipeline, the app and the integration with
separately installed FlurSeg and SAM 3 models. The FlurSeg inference follows AIhdproduction's
own training and SAMcloud code.
The project's original material is covered by [LICENSE](LICENSE).

The references below acknowledge scientific inspiration and identify external software
used by the application. A methodological reference does not mean that its source code
is included in this repository. External libraries and models remain the work of their
respective authors and are subject to their own licenses.

## SegEarth-OV3 — scientific inspiration

Li et al., *SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic Segmentation in Remote
Sensing Images*, arXiv:2512.08730. https://arxiv.org/abs/2512.08730

Reference project: https://github.com/earth-insights/SegEarth-OV-3

SegEarth-OV3 served as scientific and methodological inspiration for the use of SAM 3 in
remote-sensing image segmentation. Its paper and reference implementation were consulted
to understand the approach. No source code from the SegEarth-OV3 repository was copied
into or is distributed with this project.

The probability-fusion and tiling code was independently written for this application
using the described method and this project's requirements. The implementation also makes
its own design choices: overlapping tiles are combined by per-class maximum, whereas the
reference implementation averages overlaps. The citation credits the underlying research;
it does not claim authorship of that research or imply affiliation with its authors.

## SAM 3 — external model and software dependency

Official implementation: https://github.com/facebookresearch/sam3

Verified source revision: `0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`.

SAM 3 is subject to Meta's SAM License:
https://github.com/facebookresearch/sam3/blob/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc/LICENSE
(copy from revision `0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`:
[licenses/SAM3_LICENSE.txt](licenses/SAM3_LICENSE.txt)).

The checkpoint is not included or redistributed. Users must request access and accept the
applicable terms at https://huggingface.co/facebook/sam3. Upstream package metadata contains
an MIT classifier; the actual upstream LICENSE governs, not that metadata label.

The installer fetches SAM 3 code from the official upstream repository separately. Local
inference is still use of SAM Materials under Meta's license, even when only classified
point clouds, masks or other generated results are shared. Meta's section 1.b.ii requires
acknowledging SAM Materials when publishing research results produced using them.

Sharing generated results should be distinguished from distributing SAM code, model weights
or derivatives of those materials. Section 1.b.i requires a copy of Meta's agreement when
SAM Materials or their derivatives are distributed; a link alone does not satisfy that
requirement. Anyone bundling those materials must handle those obligations separately.
These notes do not grant rights in outputs or override Meta's applicable terms.

## FlurSeg — external model weights

Built with DINOv3.

Model and model card: https://huggingface.co/AIhdproduction/Flurseg

Verified revision: `5df53dbbf721778f1ecd4903369e9465926ae2fd`, folder `v1`.

FlurSeg was trained by AIhdproduction. Its weights are released under Meta's DINOv3 License,
because the backbone is DINOv3:
https://ai.meta.com/resources/models-and-libraries/dinov3-license/
(copy from FlurSeg revision `5df53dbbf721778f1ecd4903369e9465926ae2fd`:
[licenses/FLURSEG_DINOV3_LICENSE.md](licenses/FLURSEG_DINOV3_LICENSE.md)).

The weights are not included or redistributed here. Users download them separately and are
bound by that agreement. Among other conditions it requires providing a copy of the agreement
when distributing DINO Materials or derivatives, acknowledging DINO Materials in research
publications, and it prohibits certain uses, including military ones. The original licence file
and the attribution of the training data are in the model repository (`LICENSE`, `NOTICE.md`).

The model architecture is EoMT (Kerssies et al., "Your ViT is Secretly an Image Segmentation
Model", CVPR 2025), MIT licence: https://github.com/tue-mps/eomt

DINOv3: Simeoni et al., "DINOv3", Meta AI, 2025. https://github.com/facebookresearch/dinov3

SAM 3 was used as a label teacher during FlurSeg's training. No SAM 3 weights are part of FlurSeg.

## Transformers — software dependency

Hugging Face Transformers 5.22.1, Apache License 2.0:
https://github.com/huggingface/transformers

It provides the EoMT-DINOv3 implementation. It is installed from PyPI and not copied into this
repository. The `hf` download command comes from huggingface_hub (Apache License 2.0).

## NiceGUI — software dependency of the app

NiceGUI 3.17.1, MIT licence: https://github.com/zauberzeug/nicegui

The app serves the files bundled with the installed NiceGUI package locally, including
Quasar (MIT), Vue (MIT), the Roboto font and the Material Icons font, each under its upstream
licence. No files are loaded from a CDN, and none are copied into this repository. The app's
logo is an original drawing of this project.

## Other dependencies

Metashape is proprietary Agisoft software, installed and licensed separately. No Agisoft
binaries are distributed. NumPy, SciPy, PyTorch, torchvision, laspy, pyproj, Pillow, Pydantic,
PyYAML, psutil (BSD-3-Clause), safetensors and the optional SAM dependencies retain their
respective upstream licenses.
Windows Triton: https://github.com/triton-lang/triton-windows (MIT).
Dependency installation does not transfer ownership of those projects to AIhdproduction.
