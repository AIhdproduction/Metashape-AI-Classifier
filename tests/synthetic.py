"""A noisy synthetic site, stand-in weights and fake segmentation backends for the tests."""

import json

import numpy as np
from PIL import Image

from msam3.camera_model import Camera, project_numpy
from msam3.segmentation.model_files import FLURSEG_LABELS

CAMERA_HEIGHT = 5.0
WIDTH, HEIGHT, FOCAL = 240, 180, 200.0
# The same site, named with the SAM 3 classes.
SAM_NAMES = {
    "other": "water",
    "ground": "ground",
    "paved_surface": "road",
    "low_vegetation": "low_vegetation",
    "tree": "tree",
    "building": "building",
    "vehicle": "vehicle",
    "construction": "construction_machinery",
    "curb": "curb",
    "road_marking": "sidewalk",
    "utility_cover": "utility_pole",
}


def site_class(x, y):
    """Index of the FlurSeg class at a position in metres."""
    x, y = np.asarray(x), np.asarray(y)
    index = {name: i for i, name in enumerate(FLURSEG_LABELS)}
    labels = np.full(x.shape, index["paved_surface"], np.uint8)
    labels[x < -1.2] = index["ground"]
    labels[(x >= 1.23) & (x < 1.41)] = index["curb"]
    labels[x >= 1.41] = index["low_vegetation"]
    labels[(np.abs(x - 0.31) < 0.085) & (y < 1.1)] = index["road_marking"]
    labels[(x + 0.53) ** 2 + (y + 0.22) ** 2 < 0.31**2] = index["utility_cover"]
    top = y >= 1.37
    for low, high, name in (
        (-9, -1.31, "other"),
        (-1.31, -0.07, "vehicle"),
        (-0.07, 1.23, "construction"),
        (1.41, 9, "building"),
    ):
        labels[top & (x >= low) & (x < high)] = index[name]
    return labels


def site_points(seed=7, spacing=0.05):
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(-2.5, 2.5, spacing), np.arange(-1.8, 1.8, spacing))
    # Noise everywhere: no equal distances and no point on a voxel or pixel border.
    points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, CAMERA_HEIGHT)))
    points[:, :2] += 0.0113 + rng.uniform(-0.012, 0.012, (len(points), 2))
    points[:, 2] += rng.normal(0, 0.004, len(points))
    return points, site_class(points[:, 0], points[:, 1])


def site_cameras(directory, count=3):
    """Write nadir photos whose colour encodes the class; return the camera records."""
    records = []
    for key in range(count):
        x, y = (key - (count - 1) / 2) * 0.43, key * 0.017
        transform = np.eye(4)
        transform[:3, 3] = [x, y, 0]
        uu, vv = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
        labels = site_class(
            (uu - WIDTH / 2) / FOCAL * CAMERA_HEIGHT + x,
            (vv - HEIGHT / 2) / FOCAL * CAMERA_HEIGHT + y,
        )
        pixels = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        pixels[:, :, 0] = labels * 20 + 10
        path = directory / f"photo-{key}.png"
        Image.fromarray(pixels).save(path)
        records.append(
            {
                "key": key,
                "label": f"DJI ({key}) Süd",
                "image_path": str(path),
                "width": WIDTH,
                "height": HEIGHT,
                "transform": transform.tolist(),
                "calibration": {"f": FOCAL},
                "orientation": 1,
            }
        )
    return records


def decode(image):
    red = np.asarray(image.convert("RGB"), dtype=float)[:, :, 0]
    return np.clip(np.rint((red - 10) / 20), 0, len(FLURSEG_LABELS) - 1).astype(np.uint8)


class FakeFlurSeg:
    def __init__(self):
        self.calls = 0

    def segment(self, image):
        self.calls += 1
        labels = decode(image)
        # Uneven confidence, so that no class totals are exactly equal.
        noise = np.random.default_rng(self.calls).uniform(0.0, 0.09, labels.shape)
        return labels, (0.88 + noise).astype(np.float32)


class FakeSam:
    def __init__(self):
        self.calls = 0

    def predict_tile(self, image, classes):
        self.calls += 1
        labels = decode(image)
        names = [item.name for item in classes.classes]
        rng = np.random.default_rng(self.calls)
        result = rng.uniform(0.01, 0.05, (len(names), *labels.shape)).astype(np.float32)
        for index, name in enumerate(FLURSEG_LABELS):
            if SAM_NAMES[name] in names:
                mask = labels == index
                result[names.index(SAM_NAMES[name])][mask] = rng.uniform(0.85, 0.97)
        return result


def write_flurseg_files(directory, marker="v1", labels=FLURSEG_LABELS):
    """Files that pass every check before loading, a few bytes small."""
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "eomt_dinov3",
        "image_size": 768,
        "id2label": {str(index): name for index, name in enumerate(labels)},
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    processor = {"image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225]}
    (directory / "preprocessor_config.json").write_text(json.dumps(processor), encoding="utf-8")
    write_weights(directory / "model.safetensors", marker)
    return directory


def write_weights(path, marker):
    data = marker.encode().ljust(8, b"_")
    header = {"w": {"dtype": "U8", "shape": [len(data)], "data_offsets": [0, len(data)]}}
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + data)


def projections(points, records):
    result = []
    for record in records:
        uv, _, valid = project_numpy(points, Camera.from_dict(record))
        for index in np.flatnonzero(valid):
            result.append(
                {"camera_key": record["key"], "sample": int(index), "pixel": uv[index].tolist()}
            )
    return result
