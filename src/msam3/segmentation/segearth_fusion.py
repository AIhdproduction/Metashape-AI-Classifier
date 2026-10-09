"""Independent implementation of the requested SegEarth-OV3 probability fusion."""

import torch


def fuse_prompt(semantic, instances, scores, presence, threshold: float = 0.3):
    result = semantic.float().clone()
    keep = scores >= threshold
    if torch.any(keep):
        # Iterate masks to avoid a second N_instances x H x W allocation.
        for mask, score in zip(instances[keep], scores[keep], strict=True):
            result = torch.maximum(result, mask.float() * score.float())
    return result * presence.float()


def merge_prompts(probabilities: list[torch.Tensor]) -> torch.Tensor:
    if not probabilities:
        raise ValueError("At least one prompt is required")
    result = probabilities[0].clone()
    for probability in probabilities[1:]:
        result = torch.maximum(result, probability)
    return result


def decide(probabilities: torch.Tensor, background_threshold: float):
    confidence, labels = probabilities.max(dim=0)
    labels = labels.to(torch.uint8)
    labels[confidence < background_threshold] = 255
    return labels, confidence
