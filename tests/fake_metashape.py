"""Just enough of the Metashape 2.x API to run the real embedded scripts without Metashape."""

import types
from pathlib import Path

import laspy
import numpy as np
from PIL import Image
from synthetic import CAMERA_HEIGHT, FOCAL, HEIGHT, WIDTH, site_cameras, site_points

from msam3.camera_model import Camera as CameraModel
from msam3.camera_model import project_numpy

WKT = 'LOCAL_CS["Local",UNIT["metre",1]]'


class Vector(list):
    x = property(lambda self: self[0])
    y = property(lambda self: self[1])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self, other, strict=True)])

    def norm(self):
        return float(np.linalg.norm(self))


class Matrix:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def __getitem__(self, position):
        return self.values[position]

    def mulp(self, point):
        return Vector((self.values[:3, :3] @ np.asarray(point) + self.values[:3, 3]).tolist())


class CoordinateSystem:
    geoccs = None

    def __init__(self, wkt=WKT):
        self.wkt = wkt


class Calibration:
    def __init__(self):
        for name in ("cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4", "b1", "b2"):
            setattr(self, name, 0.0)
        self.f, self.width, self.height = FOCAL, WIDTH, HEIGHT


class Camera:
    def __init__(self, module, record):
        self.enabled = True
        self.key, self.label = record["key"], record["label"]
        self.transform = Matrix(record["transform"])
        self.calibration = Calibration()
        self.orientation = 1
        self.photo = types.SimpleNamespace(path=record["image_path"])
        self.sensor = types.SimpleNamespace(
            type=module.Sensor.Type.Frame,
            calibration=self.calibration,
            rolling_shutter=module.Shutter.Model.Disabled,
        )
        self._model = CameraModel.from_dict(record)

    def project(self, point):
        uv, depth, _ = project_numpy(np.array([point], dtype=float), self._model)
        return Vector(uv[0].tolist()) if depth[0] > 0 else None


class PointCloud:
    def __init__(self, key, label, xyz, codes):
        self.key, self.label = key, label
        self.xyz, self.codes = xyz, codes
        self.meta = {}

    @property
    def point_count(self):
        return len(self.codes)

    @property
    def point_count_by_class(self):
        codes, counts = np.unique(self.codes, return_counts=True)
        return {int(code): int(count) for code, count in zip(codes, counts, strict=True)}

    def classifyGroundPoints(self, **options):
        # Metashape finds a little more terrain in the points it has not classified yet.
        if options["source_class"] == 0:
            left = self.xyz[:, 0] < -2.2
            self.codes[left & (self.codes == 0)] = 2


class DepthMap:
    def __init__(self):
        self.calibration = Calibration()

    def image(self):
        # Slightly behind the noisy surface, so that the surface counts as visible.
        depth = np.full((HEIGHT, WIDTH), CAMERA_HEIGHT + 0.03, np.float32)
        return types.SimpleNamespace(save=lambda path: Image.fromarray(depth).save(path))


class Chunk:
    def __init__(self, module, directory):
        self.key, self.label, self.enabled = 0, "Süd", True
        self.crs = CoordinateSystem()
        self.transform = types.SimpleNamespace(matrix=Matrix(np.eye(4)))
        self.cameras = [Camera(module, record) for record in site_cameras(directory)]
        self.depth_maps = {camera: DepthMap() for camera in self.cameras}
        points, _ = site_points()
        points = np.round(points, 3)  # The LAS export stores millimetres.
        cloud = PointCloud(7, "Dense Cloud", points, np.zeros(len(points), np.uint8))
        self.point_clouds = [cloud]
        self.point_cloud = cloud
        ties = [types.SimpleNamespace(valid=True, coord=Vector(p.tolist())) for p in points[::53]]
        self.tie_points = types.SimpleNamespace(points=ties)
        self.imports = []
        self.full_exports = []  # Keys of the clouds exported whole, i.e. as stage 1 input.

    def exportPointCloud(self, path, point_clouds, classes=None, **options):
        cloud = next(item for item in self.point_clouds if item.key == point_clouds[0])
        if classes is None:
            self.full_exports.append(cloud.key)
        keep = np.ones(len(cloud.codes), bool) if classes is None else np.isin(cloud.codes, classes)
        # Like Metashape 2.3.1: LAS 1.2 point format 2, with the whole code in the byte that
        # holds a five-bit class and the synthetic/key-point/withheld flags. 67 reads as 3.
        las = laspy.create(point_format=2, file_version="1.2")
        las.header.offsets, las.header.scales = np.zeros(3), np.full(3, 0.001)
        las.x, las.y, las.z = cloud.xyz[keep].T
        las.red = np.full(int(keep.sum()), 12345, np.uint16)
        las.points.array["raw_classification"] = cloud.codes[keep]
        las.write(path)

    def importPointCloud(self, path, **options):
        las = laspy.read(path)
        xyz = np.column_stack((las.x, las.y, las.z))
        key = max(item.key for item in self.point_clouds) + 1
        asset = PointCloud(key, "Point Cloud", xyz, np.asarray(las.classification).copy())
        self.point_clouds.append(asset)
        self.point_cloud = asset
        self.imports.append(Path(path))


class Document:
    def __init__(self, module, directory):
        self.path = str(directory / "Site.psx")
        Path(self.path).write_bytes(b"project")
        self.read_only = False
        self.chunks = [Chunk(module, directory)]
        self.chunk = self.chunks[0]
        self.saves = 0

    def findChunk(self, key):
        return next((chunk for chunk in self.chunks if chunk.key == key), None)

    def save(self):
        self.saves += 1
        Path(self.path).write_bytes(b"project" + b"+" * self.saves)


def build(directory):
    """Return a module that can stand in for ``import Metashape``, with an open document."""
    space = types.SimpleNamespace
    module = types.ModuleType("Metashape")
    module.Vector = Vector
    module.CoordinateSystem = CoordinateSystem
    module.PointCloudData = "PointCloudData"
    module.PointCloudFormatLAS = "LAS"
    module.PointClass = space(Created=0, Unclassified=1)
    module.Sensor = space(Type=space(Frame="Frame"))
    module.Shutter = space(Model=space(Disabled="Disabled"))
    module.Application = space(
        ModelView=space(PointCloudViewColor="colour", PointCloudViewClassification="classes")
    )
    messages = []
    module.app = space(
        version="2.3.1",
        activated=True,
        messageBox=messages.append,
        messages=messages,
        update=lambda: None,
        model_view=space(view_mode=None, point_cloud_view_mode=None),
    )
    module.app.document = Document(module, directory)
    return module
