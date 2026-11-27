import numpy as np
import pytest
import torch
from PIL import Image

from msam3.config import Classes, SAMConfig
from msam3.segmentation.sam3_backend import raw_probabilities
from msam3.segmentation.segearth_fusion import decide, fuse_prompt, merge_prompts
from msam3.segmentation.tiling import segment_image, tile_boxes


def test_fusion_gating_prompts_and_unknown():
    semantic = torch.tensor([[0.1, 0.6], [0.2, 0.1]])
    masks = torch.tensor([[[1.0, 0.1], [0.9, 0.3]], [[1.0, 1.0], [1.0, 1.0]]])
    scores = torch.tensor([0.8, 0.2])
    result = fuse_prompt(semantic, masks, scores, torch.tensor(0.5), 0.3)
    torch.testing.assert_close(result, torch.tensor([[0.4, 0.3], [0.36, 0.12]]))
    maximum = merge_prompts([result, torch.full((2, 2), 0.2)])
    labels, confidence = decide(torch.stack([maximum, torch.full((2, 2), 0.1)]), 0.3)
    assert labels.tolist() == [[0, 0], [0, 255]]
    assert confidence[1, 1].item() == pytest.approx(0.2)
    empty = fuse_prompt(semantic, masks[:0], scores[:0], torch.tensor(0.0))
    assert not torch.any(empty)


def test_raw_logits_presence_applied_once():
    outputs = {
        "semantic_seg": torch.full((1, 1, 2, 2), torch.logit(torch.tensor(0.2))),
        "pred_masks": torch.full((1, 1, 2, 2), torch.logit(torch.tensor(0.9))),
        "pred_logits": torch.full((1, 1, 1), torch.logit(torch.tensor(0.8))),
        "presence_logit_dec": torch.tensor([[0.0]]),
    }
    result = raw_probabilities(outputs, (4, 4), 0.3)
    torch.testing.assert_close(result, torch.full((4, 4), 0.36))
    with pytest.raises(RuntimeError, match="missing"):
        raw_probabilities({}, (4, 4), 0.3)


@pytest.mark.parametrize("threshold", [0.3, 0.9])
def test_raw_logits_merge_multiple_mask_batches_and_keep_semantic_fallback(threshold):
    # Eleven masks pass at 0.3. One winner sits in the first batch, another in the second.
    mask_probs = torch.full((1, 12, 1, 3), 0.1)
    mask_probs[0, 0, 0, 0] = 0.9
    mask_probs[0, 9, 0, 1] = 0.9
    mask_probs[0, 11, 0, 2] = 0.99
    instance_scores = torch.full((1, 12, 1), 0.8)
    instance_scores[0, 11, 0] = 0.2
    outputs = {
        "semantic_seg": torch.full((1, 1, 1, 3), torch.logit(torch.tensor(0.1))),
        "pred_masks": torch.logit(mask_probs),
        "pred_logits": torch.logit(instance_scores),
        "presence_logit_dec": torch.tensor([[0.0]]),
    }

    probabilities = raw_probabilities(outputs, (2, 3), threshold)

    row = [0.36, 0.36, 0.05] if threshold == 0.3 else [0.05, 0.05, 0.05]
    torch.testing.assert_close(probabilities, torch.tensor([row, row]))


def test_tile_coverage_and_max_merge(tmp_path):
    width, height = 1010, 20
    covered = np.zeros((height, width), int)
    for x0, y0, x1, y1 in tile_boxes(width, height, 1008, 0.2):
        covered[y0:y1, x0:x1] += 1
    assert covered.min() > 0
    classes = Classes.model_validate(
        {"classes": [{"name": "a", "prompts": ["a"]}, {"name": "b", "prompts": ["b"]}]}
    )

    class Backend:
        calls = 0

        def predict_tile(self, image, classes):
            self.calls += 1
            result = np.zeros((2, image.height, image.width), np.float32)
            result[0] = 0.9 if self.calls == 1 else 0.1
            result[1] = 0.8
            return result

    labels, _ = segment_image(
        Image.new("RGB", (width, height)), Backend(), classes, SAMConfig(), tmp_path / "scores.dat"
    )
    assert np.all(labels[:, 2:1008] == 0)
    assert np.all(labels[:, 1008:] == 1)
    assert not (tmp_path / "scores.dat").exists()
