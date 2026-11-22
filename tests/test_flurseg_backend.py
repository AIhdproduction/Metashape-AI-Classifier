import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from msam3.config import FlurSegConfig, GPUConfig
from msam3.segmentation import flurseg_backend
from msam3.segmentation.flurseg_backend import FlurSegBackend, load_model, semantic_scores
from msam3.segmentation.model_files import FLURSEG_LABELS
from msam3.segmentation.windows import scaled_size, window_count, window_starts

LABELS = dict(enumerate(FLURSEG_LABELS))


class FakeModel:
    """Random but repeatable query outputs that depend on the window's pixels."""

    def __init__(self, window=32, fail_above=None, labels=None, favourite=None):
        self.config = SimpleNamespace(image_size=window, id2label=labels or LABELS)
        self.fail_above, self.favourite = fail_above, favourite
        self.batches, self.inputs = [], []
        rng = torch.Generator().manual_seed(3)
        self.class_mix = torch.randn((7, len(LABELS) + 1, 3), generator=rng)
        self.mask_mix = torch.randn((7, 3), generator=rng)

    def __call__(self, pixel_values):
        self.batches.append(len(pixel_values))
        if self.fail_above is not None and len(pixel_values) > self.fail_above:
            raise torch.OutOfMemoryError("synthetic out of memory")
        self.inputs.append(pixel_values.clone())
        mean = pixel_values.mean(dim=(2, 3))
        classes = torch.einsum("qck,bk->bqc", self.class_mix, mean)
        if self.favourite is not None:
            classes[:, :, self.favourite] += 30
        small = torch.nn.functional.avg_pool2d(pixel_values, 4)
        masks = torch.einsum("qk,bkhw->bqhw", self.mask_mix, small)
        return SimpleNamespace(class_queries_logits=classes, masks_queries_logits=masks)


def backend(model=None, **settings):
    values = {"resolution": "accurate", "background_threshold": 0.0, **settings}
    return FlurSegBackend(model or FakeModel(), FlurSegConfig(**values), "cpu")


def photo(width, height, seed=1):
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def full_buffer(engine, image):
    """The straightforward reference: every class for the whole image in memory."""
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
    pixels = (pixels - engine.mean) / engine.std
    size, (height, width) = engine.window, pixels.shape[1:]
    total = torch.zeros((engine.classes, height, width))
    weight = torch.zeros((height, width))
    for top in window_starts(height, size, engine.stride):
        for left in window_starts(width, size, engine.stride):
            tile = pixels[:, top : top + size, left : left + size]
            h, w = tile.shape[1:]
            tile = torch.nn.functional.pad(tile, (0, size - w, 0, size - h))
            scores = semantic_scores(engine.model(tile[None]), size)[0]
            total[:, top : top + h, left : left + w] += scores[:, :h, :w] * engine.hann[:h, :w]
            weight[top : top + h, left : left + w] += engine.hann[:h, :w]
    scores = total / weight
    return scores.argmax(0).numpy(), (scores.max(0).values / scores.sum(0)).numpy()


@pytest.mark.parametrize(
    "width,height",
    [(101, 119), (32, 32), (33, 97), (19, 23), (80, 64), (31, 131), (145, 17)],
)
@pytest.mark.parametrize("batch", [1, 4])
def test_band_buffer_equals_full_buffer(width, height, batch):
    engine = backend(window_batch_size=batch)
    image = photo(width, height)
    labels, confidence = engine.segment(image)
    expected_labels, expected_confidence = full_buffer(engine, image)
    assert labels.dtype == np.uint8 and confidence.dtype == np.float32
    assert labels.shape == (height, width)
    assert np.mean(labels == expected_labels) > 0.999  # Only float rounding may differ.
    np.testing.assert_allclose(confidence, expected_confidence, atol=2e-5)


def test_small_image_is_padded_with_the_mean_colour_after_normalisation():
    model = FakeModel()
    engine = backend(model)
    image = photo(19, 23)
    labels, _ = engine.segment(image)
    tile = model.inputs[0][0]
    assert labels.shape == (23, 19) and tile.shape == (3, 32, 32)
    assert torch.all(tile[:, 23:] == 0) and torch.all(tile[:, :, 19:] == 0)
    pixels = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(tile[:, :23, :19], (pixels - engine.mean) / engine.std)


@pytest.mark.parametrize("index", range(len(FLURSEG_LABELS)))
def test_label_index_is_the_trained_class_order(index):
    labels, confidence = backend(FakeModel(favourite=index)).segment(photo(70, 45))
    assert np.all(labels == index)
    assert confidence.min() > 0.99


def test_score_memory_is_one_window_high(monkeypatch):
    shapes, zeros = [], torch.zeros

    def record(*size, **options):
        result = zeros(*size, **options)
        shapes.append(tuple(result.shape))
        return result

    engine = backend()
    monkeypatch.setattr(flurseg_backend.torch, "zeros", record)
    engine.segment(photo(90, 700))
    assert (len(LABELS), 32, 90) in shapes
    assert all(700 not in shape for shape in shapes)


def test_unknown_below_the_threshold():
    labels, confidence = backend(background_threshold=1.0).segment(photo(50, 50))
    assert np.all(labels == 255) and np.all(confidence < 1)


def test_out_of_memory_halves_the_batch_and_keeps_the_result():
    image = photo(120, 75)
    expected, _ = backend().segment(image)
    model = FakeModel(fail_above=1)
    engine = backend(model, window_batch_size=8)
    labels, _ = engine.segment(image)
    assert engine.batch_size == 1
    assert model.batches[0] > 1
    assert np.mean(labels == expected) > 0.999


def test_out_of_memory_at_one_window_is_a_clear_error():
    with pytest.raises(RuntimeError, match="one window at a time"):
        backend(FakeModel(fail_above=0)).segment(photo(70, 70))


def test_other_label_order_is_refused():
    labels = dict(LABELS)
    labels[2], labels[4] = labels[4], labels[2]
    with pytest.raises(ValueError, match="order"):
        backend(FakeModel(labels=labels))


def test_resolution_profiles_never_upscale():
    assert scaled_size(8192, 5464, "fast") == (3200, 2134)
    assert scaled_size(5464, 8192, "balanced") == (2668, 4000)
    assert scaled_size(8192, 5464, "accurate") == (8192, 5464)
    assert all(scaled_size(1000, 700, name) == (1000, 700) for name in ("fast", "balanced"))
    assert window_count(4000, 3000, "balanced") == 70
    assert window_count(500, 400, "accurate") == 1


def test_segment_uses_the_profile_size():
    labels, _ = backend(FakeModel(768), resolution="fast").segment(photo(3300, 120))
    assert labels.shape == (116, 3200)


def test_production_requires_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(flurseg_backend.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        FlurSegBackend.load(tmp_path, GPUConfig(), FlurSegConfig())


@pytest.fixture(scope="module")
def tiny_model_directory(tmp_path_factory):
    """A real, very small random EoMT-DINOv3 in FlurSeg's layout, built without a download."""
    from transformers import EomtDinov3Config, EomtDinov3ForUniversalSegmentation

    torch.manual_seed(11)
    config = EomtDinov3Config(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_blocks=2,
        num_queries=6,
        num_register_tokens=2,
        num_upscale_blocks=2,
        patch_size=16,
        image_size=768,
        id2label=LABELS,
        label2id={name: index for index, name in LABELS.items()},
    )
    directory = tmp_path_factory.mktemp("tiny-flurseg")
    EomtDinov3ForUniversalSegmentation(config).eval().save_pretrained(directory)
    processor = {"image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225]}
    (directory / "preprocessor_config.json").write_text(json.dumps(processor), encoding="utf-8")
    return directory


def test_real_eomt_dinov3_forward_through_the_backend(tiny_model_directory):
    """This is how CI proves torch 2.7.1 together with transformers 5.22.1."""
    model = load_model(tiny_model_directory, "cpu")
    assert not model.training
    engine = FlurSegBackend(model, FlurSegConfig(background_threshold=0), "cpu")
    labels, confidence = engine.segment(photo(800, 420))
    assert labels.shape == confidence.shape == (420, 800)
    assert labels.max() < len(LABELS)
    assert np.all(np.isfinite(confidence)) and confidence.min() >= 1 / len(LABELS) - 1e-4


def test_weights_that_do_not_fit_are_refused(tiny_model_directory, tmp_path):
    from safetensors.torch import load_file, save_file

    for name in ("config.json", "preprocessor_config.json"):
        (tmp_path / name).write_bytes((tiny_model_directory / name).read_bytes())
    state = load_file(tiny_model_directory / "model.safetensors")
    del state["class_predictor.weight"]
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(ValueError, match="do not fit FlurSeg"):
        load_model(tmp_path, "cpu")
