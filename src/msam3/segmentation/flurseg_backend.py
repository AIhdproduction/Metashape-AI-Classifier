"""FlurSeg (EoMT-DINOv3) inference from local files, in bounded GPU memory.

Scores of overlapping windows are blended in a band that is one window high. Rows that no later
window can touch are decided on the device; only labels and confidence travel to the CPU.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from msam3.config import FlurSegConfig, GPUConfig
from msam3.segmentation.model_files import check_labels, read_flurseg_metadata, require_flurseg
from msam3.segmentation.windows import scaled_size, window_starts

LOG = logging.getLogger(__name__)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def check_cuda(gpu: GPUConfig) -> None:
    if not gpu.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "FlurSeg requires CUDA. Install the CUDA PyTorch wheels (install.bat) and a current "
            "NVIDIA driver."
        )
    properties = torch.cuda.get_device_properties(torch.device(gpu.device))
    LOG.info("GPU: %s, %.1f GiB VRAM", properties.name, properties.total_memory / 1024**3)


def load_model(directory: Path, device: str | torch.device):
    """Load exactly the local weights: no download, no remote code, no partly random model."""
    require_flurseg(directory)
    try:
        from transformers import EomtDinov3ForUniversalSegmentation
        from transformers.utils import logging as transformers_logging
    except ImportError as error:
        raise RuntimeError(
            "Cannot import transformers 5.22.1 for FlurSeg. Run install.bat again. " + str(error)
        ) from error
    transformers_logging.disable_progress_bar()
    model, report = EomtDinov3ForUniversalSegmentation.from_pretrained(
        str(directory),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        output_loading_info=True,
    )
    broken = {
        key: sorted(map(str, report[key]))[:5]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys")
        if report.get(key)
    }
    if broken:
        raise ValueError(f"The weights in {directory} do not fit FlurSeg: {broken}")
    return model.to(device).eval()


def semantic_scores(output, size: int) -> torch.Tensor:
    """Combine the query masks at native resolution, then upsample only the class maps."""
    classes = output.class_queries_logits.float().softmax(-1)[..., :-1]
    masks = output.masks_queries_logits.float().sigmoid()
    scores = torch.einsum("bqc,bqhw->bchw", classes, masks)
    return functional.interpolate(scores, (size, size), mode="bilinear", align_corners=False)


class FlurSegBackend:
    def __init__(
        self,
        model,
        settings: FlurSegConfig,
        device: str | torch.device,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ):
        check_labels({int(key): value for key, value in model.config.id2label.items()})
        self.model = model
        self.settings = settings
        self.device = torch.device(device)
        self.window = int(model.config.image_size)
        self.stride = self.window // 2
        self.classes = len(model.config.id2label)
        self.batch_size = settings.window_batch_size
        self.mean = torch.tensor(mean, device=self.device, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, device=self.device, dtype=torch.float32).view(3, 1, 1)
        hann = torch.hann_window(self.window, device=self.device)
        self.hann = torch.outer(hann, hann) + 1e-6

    @classmethod
    def load(cls, directory: Path, gpu: GPUConfig, settings: FlurSegConfig):
        check_cuda(gpu)
        mean, std = read_flurseg_metadata(directory)
        LOG.info("Loading FlurSeg from %s", directory)
        return cls(load_model(directory, gpu.device), settings, gpu.device, mean, std)

    @torch.inference_mode()
    def segment(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Return uint8 labels (255 = unknown) and float32 confidence at the profile resolution."""
        image = image.convert("RGB")
        size = scaled_size(image.width, image.height, self.settings.resolution)
        if size != image.size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        return self._segment_pixels(np.asarray(image))

    def _segment_pixels(self, pixels: np.ndarray):
        height, width = pixels.shape[:2]
        window, band_width = self.window, max(width, self.window)
        rows = window_starts(height, window, self.stride)
        columns = window_starts(width, window, self.stride)
        band = torch.zeros((self.classes, window, band_width), device=self.device)
        weight = torch.zeros((window, band_width), device=self.device)
        labels = np.empty((height, width), np.uint8)
        confidence = np.empty((height, width), np.float32)
        for index, top in enumerate(rows):
            strip = torch.from_numpy(np.array(pixels[top : top + window]))
            self._add_row(strip.permute(2, 0, 1).to(self.device), columns, band, weight)
            last = index + 1 == len(rows)
            done = height - top if last else rows[index + 1] - top
            decided = self._decide(band[:, :done, :width], weight[:done, :width])
            labels[top : top + done], confidence[top : top + done] = decided
            if not last:
                keep = window - done
                band[:, :keep] = band[:, done:].clone()
                band[:, keep:] = 0
                weight[:keep] = weight[done:].clone()
                weight[keep:] = 0
        return labels, confidence

    def _add_row(self, strip, columns, band, weight):
        done = 0
        while done < len(columns):
            batch = columns[done : done + self.batch_size]
            scores = None
            try:
                scores = self._predict(strip, batch)
            except torch.OutOfMemoryError:
                # Retry outside the handler: its traceback still holds the failed tensors.
                pass
            if scores is None:
                if len(batch) == 1:
                    raise RuntimeError(
                        "Out of GPU memory although FlurSeg already runs one window at a time. "
                        "Close other GPU applications or choose a lower resolution profile."
                    )
                self.batch_size = len(batch) // 2
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                LOG.warning("GPU memory is short: FlurSeg window batch is now %d", self.batch_size)
                continue
            for window_scores, left in zip(scores, batch, strict=True):
                area = slice(left, left + self.window)
                band[:, :, area] += window_scores * self.hann
                weight[:, area] += self.hann
            done += len(batch)
            del scores

    def _predict(self, strip, lefts):
        tiles = []
        for left in lefts:
            tile = strip[:, :, left : left + self.window].float() / 255
            tile = (tile - self.mean) / self.std
            # Zero after normalisation, i.e. the mean colour, as in the training pipeline.
            padding = (0, self.window - tile.shape[2], 0, self.window - tile.shape[1])
            tiles.append(functional.pad(tile, padding))
        half = self.settings.precision == "fp16" and self.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.float16, enabled=half):
            output = self.model(pixel_values=torch.stack(tiles))
        scores = semantic_scores(output, self.window)
        if scores.shape[1] != self.classes:
            raise RuntimeError(f"FlurSeg returned {scores.shape[1]} instead of {self.classes}")
        return scores

    def _decide(self, totals, weight):
        scores = totals / weight.clamp_min(1e-6)
        best, labels = scores.max(dim=0)
        confidence = best / scores.sum(dim=0).clamp_min(1e-6)
        if not torch.isfinite(confidence).all():
            raise RuntimeError("FlurSeg returned invalid scores; try flurseg.precision: fp32")
        labels = labels.to(torch.uint8)
        labels[confidence < self.settings.background_threshold] = 255
        return labels.cpu().numpy(), confidence.float().cpu().numpy()
