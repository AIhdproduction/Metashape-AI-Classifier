"""Thin, version-checked adapter to official SAM 3 raw grounding outputs."""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from msam3.config import Classes, GPUConfig, SAMConfig
from msam3.segmentation.segearth_fusion import fuse_prompt

LOG = logging.getLogger(__name__)
SAM_REVISION = "0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc"
BACKEND_VERSION = "raw-grounding-v2-batched"


def require_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"SAM 3 checkpoint missing: {path}. Request access at "
            "https://huggingface.co/facebook/sam3 (access request required), accept Meta's "
            "SAM License, download the checkpoint and place it at models/sam3.pt "
            "(or set paths.model). Weights are not included in this repository."
        )


def check_gpu(config: GPUConfig) -> None:
    if not config.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "SAM 3 requires CUDA. Install the CUDA PyTorch wheels and NVIDIA driver."
        )
    device = torch.device(config.device)
    properties = torch.cuda.get_device_properties(device)
    gib = properties.total_memory / 1024**3
    LOG.info("GPU: %s, %.1f GiB VRAM", properties.name, gib)
    if gib < config.min_vram_gb:
        LOG.warning("GPU has %.1f GiB; at least %.1f GiB is recommended", gib, config.min_vram_gb)
    with torch.cuda.device(device):
        if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU lacks bfloat16 support; set gpu.dtype to float16.")


def raw_probabilities(outputs: dict, shape: tuple[int, int], threshold: float):
    """Sigmoid logits once; do not apply the processor's extra presence weighting."""
    required = {"semantic_seg", "pred_masks", "pred_logits", "presence_logit_dec"}
    if not required.issubset(outputs):
        raise RuntimeError(
            f"Incompatible SAM 3 outputs: missing {sorted(required - outputs.keys())}"
        )
    semantic = outputs["semantic_seg"]
    masks = outputs["pred_masks"]
    logits = outputs["pred_logits"]
    presence = outputs["presence_logit_dec"].sigmoid().float().reshape(-1)
    if semantic.ndim != 4 or semantic.shape[:2] != (1, 1):
        raise RuntimeError(f"Unexpected semantic_seg shape: {semantic.shape}")
    if masks.ndim != 4 or masks.shape[0] != 1 or logits.shape != (*masks.shape[:2], 1):
        raise RuntimeError("Unexpected SAM 3 instance/logit shape")
    if presence.numel() != 1:
        raise RuntimeError("Expected one SAM 3 presence score per prompt")
    sem = functional.interpolate(semantic.float(), shape, mode="bilinear", align_corners=False)[
        0, 0
    ].sigmoid()
    scores = logits[0, :, 0].sigmoid().float()
    result = sem.clone()
    # Resize one retained instance at a time: hundreds of full-size masks can exhaust VRAM.
    for index in torch.nonzero(scores >= threshold).flatten():
        mask = functional.interpolate(
            masks[:, index : index + 1].float(), shape, mode="bilinear", align_corners=False
        )[0, 0].sigmoid()
        result = torch.maximum(result, mask * scores[index])
    empty = result.new_empty((0, *shape))
    return fuse_prompt(result, empty, scores[:0], presence[0], threshold)


class SAM3Backend:
    def __init__(self, checkpoint: Path, gpu: GPUConfig, settings: SAMConfig):
        require_checkpoint(checkpoint)
        check_gpu(gpu)
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as error:
            raise RuntimeError(
                'Cannot import official SAM 3. Install pip install -e ".[sam]" and the '
                "platform dependencies described in README.md. " + str(error)
            ) from error
        self.device = torch.device(gpu.device)
        self.dtype = getattr(torch, gpu.dtype)
        self.settings = settings
        # The upstream builder special-cases the literal 'cuda'; move explicitly for cuda:N.
        self.model = (
            build_sam3_image_model(
                checkpoint_path=str(checkpoint),
                device="cpu",
                load_from_HF=False,
                enable_segmentation=True,
                enable_inst_interactivity=False,
                compile=False,
            )
            .to(self.device)
            .eval()
        )
        if self.model.supervise_joint_box_scores:
            raise RuntimeError("Unexpected joint SAM 3 scores: refusing double presence gating")
        self.processor = Sam3Processor(self.model, resolution=1008, device=str(self.device))
        self._prompt_key = None
        self._text_features = None

    @torch.inference_mode()
    def predict_tile(self, image: Image.Image, classes: Classes) -> np.ndarray:
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            state = self.processor.set_image(image)
            result = np.zeros((len(classes.classes), image.height, image.width), np.float32)
            from sam3.model.data_misc import FindStage

            prompts = tuple(prompt for cls in classes.classes for prompt in cls.prompts)
            class_indices = [i for i, cls in enumerate(classes.classes) for _ in cls.prompts]
            if self._prompt_key != prompts:
                self._text_features = self.model.backbone.forward_text(
                    list(prompts), device=str(self.device)
                )
                self._prompt_key = prompts
            state["backbone_out"].update(self._text_features)
            size = self.settings.prompt_batch_size
            for start in range(0, len(prompts), size):
                end = min(start + size, len(prompts))
                find = FindStage(
                    img_ids=torch.zeros(end - start, dtype=torch.long, device=self.device),
                    text_ids=torch.arange(start, end, device=self.device),
                    input_boxes=None,
                    input_boxes_mask=None,
                    input_boxes_label=None,
                    input_points=None,
                    input_points_mask=None,
                )
                outputs = self.model.forward_grounding(
                    backbone_out=state["backbone_out"],
                    find_input=find,
                    geometric_prompt=self.model._get_dummy_prompt(num_prompts=end - start),
                    find_target=None,
                )
                required = ("semantic_seg", "pred_masks", "pred_logits", "presence_logit_dec")
                if any(
                    key not in outputs or outputs[key].shape[0] != end - start for key in required
                ):
                    raise RuntimeError("Unexpected SAM 3 batched grounding output contract")
                for offset, index in enumerate(class_indices[start:end]):
                    single = {key: outputs[key][offset : offset + 1] for key in required}
                    probability = raw_probabilities(
                        single, (image.height, image.width), self.settings.instance_conf_threshold
                    )
                    np.maximum(result[index], probability.cpu().numpy(), out=result[index])
                    del single, probability
                del outputs
            del state
        return result

    def release_image(self) -> None:
        torch.cuda.empty_cache()
