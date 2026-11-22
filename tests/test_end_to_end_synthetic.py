import json

import laspy
import numpy as np
import pytest
from PIL import Image

from msam3.cache import atomic_json
from msam3.camera_model import Camera, project_numpy
from msam3.config import Classes, Settings
from msam3.pipeline import process_chunk
from msam3.validation import validate_package


@pytest.fixture
def package(tmp_path):
    width, height, focal, depth = 80, 60, 70.0, 20.0
    xx, yy = np.meshgrid(np.linspace(-6, 6, 41), np.linspace(-4, 4, 21))
    points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, depth)))
    # Nonunit scale and ECEF-sized offsets catch using internal units as metres/float32 ECEF.
    transform = np.diag([2.0, 2.0, 2.0, 1.0])
    transform[:3, 3] = [4e6, 1e6, 4.8e6]
    world = points @ transform[:3, :3].T + transform[:3, 3]
    las = laspy.create(point_format=7, file_version="1.4")
    las.header.offsets = world.mean(axis=0)
    las.header.scales = np.full(3, 0.001)
    las.x, las.y, las.z = world.T
    las.red = np.full(len(points), 12345, np.uint16)
    las.classification = np.ones(len(points), np.uint8)
    las.write(tmp_path / "dense.las")
    samples = [
        {"internal": point.tolist(), "world": coordinate.tolist()}
        for point, coordinate in zip(points[::20], world[::20], strict=True)
    ]
    projections, cameras = [], []
    for key, offset in enumerate((-1.0, 0.0, 1.0)):
        matrix = np.eye(4)
        matrix[0, 3] = offset
        path = tmp_path / f"camera-{key}.png"
        pixels = np.zeros((height, width, 3), np.uint8)
        wx = (np.arange(width) + 0.5 - width / 2) / focal * depth + offset
        pixels[:, wx < 0, 0] = 255
        pixels[:, wx >= 0, 2] = 255
        Image.fromarray(pixels).save(path)
        record = {
            "key": key,
            "label": str(key),
            "image_path": str(path),
            "width": width,
            "height": height,
            "transform": matrix.tolist(),
            "calibration": {"f": focal},
            "orientation": 6,
        }
        cameras.append(record)
        uv, _, valid = project_numpy(points[::20], Camera.from_dict(record))
        for index in np.flatnonzero(valid):
            projections.append(
                {"camera_key": key, "sample": int(index), "pixel": uv[index].tolist()}
            )
    atomic_json(tmp_path / "cameras.json", cameras)
    atomic_json(
        tmp_path / "chunk.json",
        {
            "key": 0,
            "label": "synthetic",
            "point_count": len(points),
            "transform": transform.tolist(),
        },
    )
    atomic_json(
        tmp_path / "validation_samples.json", {"samples": samples, "projections": projections}
    )
    classes = Classes.model_validate(
        {
            "classes": [
                {"name": "ground", "code": 2, "prompts": ["ground"]},
                {"name": "custom", "code": 64, "prompts": ["custom"]},
            ]
        }
    )
    settings = Settings()
    settings.segmentation.backend = "sam3"  # The fixture's fake backend has the SAM 3 contract.
    settings.visibility.splat_px = 0
    settings.visibility.zbuffer_scale = 1
    settings.visibility.mask_interior_px = 0
    settings.cache.scale = 1
    settings.voting.batch_points = 107
    settings.voting.voxel_buckets = 4
    # The synthetic 30 cm grid is intentionally too sparse for the real 45 cm
    # all-class cluster filter; this fixture validates projection and LAS I/O instead.
    settings.postprocess.cluster.enabled = False
    return tmp_path, settings, classes, np.where(points[:, 0] < 0, 2, 64)


class Backend:
    calls = 0

    def predict_tile(self, image, classes):
        self.calls += 1
        pixels = np.array(image)
        return np.stack((pixels[:, :, 0], pixels[:, :, 2])).astype(np.float32) / 255


def test_stage2_end_to_end_and_voting_only(package):
    directory, settings, classes, expected = package
    backend = Backend()
    result = process_chunk(
        directory, settings, classes, "mock-model", lambda: backend, device="cpu"
    )
    las = laspy.read(result["classified_las"])
    assert np.mean(las.classification == expected) > 0.95
    original = laspy.read(directory / "dense.las")
    np.testing.assert_array_equal(las.X, original.X)
    # Classification changes only the LAS class code; photogrammetric RGB is preserved.
    np.testing.assert_array_equal(las.red, original.red)
    np.testing.assert_array_equal(las.green, original.green)
    np.testing.assert_array_equal(las.blue, original.blue)
    assert str(las.header.version) == "1.4" and las.header.point_format.id == 7
    lookup = next(v for v in las.vlrs if v.user_id == "LASF_Spec" and v.record_id == 0)
    assert lookup[64] == "custom"
    assert backend.calls == 3
    settings.voting.min_views = 2
    process_chunk(directory, settings, classes, "mock-model", voting_only=True, device="cpu")
    assert backend.calls == 3
    settings.sam.background_threshold = 0.6
    with pytest.raises(RuntimeError, match="stale"):
        process_chunk(directory, settings, classes, "mock-model", voting_only=True, device="cpu")


def test_validation_prevents_inference(package):
    directory, settings, classes, _ = package
    path = directory / "validation_samples.json"
    samples = json.loads(path.read_text())
    samples["projections"][0]["pixel"][0] += 4
    atomic_json(path, samples)
    backend = Backend()
    with pytest.raises(ValueError, match="Projection validation failed"):
        process_chunk(directory, settings, classes, "mock", lambda: backend, device="cpu")
    assert backend.calls == 0


def test_validation_accepts_a_cropped_dense_cloud(package):
    directory, settings, _, _ = package
    path = directory / "dense.las"
    las = laspy.read(path)
    keep = np.asarray(las.x) > np.quantile(np.asarray(las.x), 0.1)
    las.points = las.points[keep]
    las.write(path)
    metadata_path = directory / "chunk.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["point_count"] = int(keep.sum())
    atomic_json(metadata_path, metadata)

    report = validate_package(directory, settings)

    assert (
        report["tie_points_inside_dense"] + report["tie_points_outside_dense"]
        == report["tie_point_count"]
    )
    assert report["tie_points_outside_dense"] > 0


def test_tie_dense_distance_gate(package):
    directory, settings, _, _ = package
    path = directory / "dense.las"
    las = laspy.read(path)
    las.z = np.asarray(las.z) + 0.5
    las.write(path)
    with pytest.raises(ValueError, match="bounding box"):
        validate_package(directory, settings)


def test_resume_after_interrupted_segmentation(package):
    directory, settings, classes, _ = package

    class InterruptedBackend(Backend):
        def predict_tile(self, image, classes):
            if self.calls == 1:
                raise RuntimeError("synthetic interruption")
            return super().predict_tile(image, classes)

    interrupted = InterruptedBackend()
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        process_chunk(directory, settings, classes, "mock", lambda: interrupted, device="cpu")
    resumed = Backend()
    process_chunk(directory, settings, classes, "mock", lambda: resumed, device="cpu")
    assert resumed.calls == 2  # First image survived the interruption.
