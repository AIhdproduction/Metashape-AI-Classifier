# Third-party notices

The project's [LICENSE](LICENSE) covers original AIhdproduction material only. It does not
replace any third-party license or grant ownership of third-party code, models or methods.

## SAM 3

Official implementation: https://github.com/facebookresearch/sam3

Verified source revision: `0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`.

SAM 3 is subject to Meta's SAM License:
https://github.com/facebookresearch/sam3/blob/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc/LICENSE

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

## SegEarth-OV3

Li et al., *SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic Segmentation in Remote
Sensing Images*, arXiv:2512.08730. https://arxiv.org/abs/2512.08730

Reference implementation studied: https://github.com/earth-insights/SegEarth-OV-3

The GitHub license endpoint returned 404 during inspection. No source code from that
repository is included. Probability fusion and tiling were independently implemented from
the task specification and method description. The reference's overlap averaging differs
from this tool's explicitly requested per-class maximum merge.

## Other dependencies

Metashape is proprietary Agisoft software, installed and licensed separately. No Agisoft
binaries are distributed. NumPy, SciPy, PyTorch, torchvision, laspy, pyproj, Pillow, Pydantic,
PyYAML and the optional SAM dependencies retain their respective upstream licenses.
Windows Triton: https://github.com/triton-lang/triton-windows (MIT).
Dependency installation does not transfer ownership of those projects to AIhdproduction.
