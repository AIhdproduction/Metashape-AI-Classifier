"""Import a separate classified asset and export each class without LAS bit truncation."""

import json
import logging
import sys
from pathlib import Path

import Metashape

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metashape_scripts.common import (
    camera_record,
    export_cloud,
    open_document,
    run_guarded,
    safe_name,
    source_signature,
    write_json,
    write_legends,
)

LOG = logging.getLogger("msam3.stage3")


def final_las_path(job, chunk, chunk_count):
    """Choose the promised simple GUI filename without collisions for multi-chunk jobs."""
    requested = job.get("final_las_name")
    if requested and chunk_count == 1:
        return Path(job["output_dir"]) / requested
    filename = (
        safe_name(Path(job["project"]).stem)
        + "_"
        + safe_name(chunk.label)
        + "_"
        + str(chunk.key)
        + "_classified.las"
    )
    return Path(job["output_dir"]) / filename


def main(job, document=None, progress=None):
    if document is None:
        document = open_document(job["project"])
    results = []
    chunk_count = len(job["stage3_inputs"])
    for entry in job["stage3_inputs"]:
        directory = Path(entry["directory"])
        metadata = json.loads((directory / "chunk.json").read_text(encoding="utf-8"))
        chunk = document.findChunk(metadata["key"])
        if chunk is None:
            raise RuntimeError("Chunk no longer exists: " + str(metadata["key"]))
        original = next(
            (p for p in chunk.point_clouds if p.key == metadata["point_cloud_key"]), None
        )
        if original is None:
            raise RuntimeError("Original cloud no longer exists; rerun stage 1")
        previous_default = chunk.point_cloud
        chunk.point_cloud = original
        camera_data = json.loads((directory / "cameras.json").read_text(encoding="utf-8"))
        camera_keys = {item["key"] for item in camera_data}
        current = [camera_record(c, job["project"]) for c in chunk.cameras if c.key in camera_keys]
        if source_signature(chunk, current) != metadata["source_signature"]:
            raise RuntimeError("Chunk geometry or calibration changed since export; rerun stage 1")
        expected = {int(k): int(v) for k, v in entry["counts"].items() if v}
        if any(code >= 128 for code in expected):
            raise RuntimeError(
                "Metashape 2.3.1 drops class codes 128..255. The standalone LAS "
                "is intact; choose class codes below 128 for project import."
            )
        asset = next(
            (
                p
                for p in chunk.point_clouds
                if "msam3.import_id" in p.meta.keys()
                and p.meta["msam3.import_id"] == entry["import_id"]
            ),
            None,
        )
        if asset is None:
            before = {p.key for p in chunk.point_clouds}
            chunk.importPointCloud(
                path=entry["classified_las"],
                format=Metashape.PointCloudFormatLAS,
                crs=Metashape.CoordinateSystem(metadata["export_crs_wkt"]),
                load_point_classification=True,
                load_point_color=True,
                replace_asset=False,
                calculate_normals=False,
                precision=entry["precision_m"],
                **({"progress": progress} if progress is not None else {}),
            )
            added = [p for p in chunk.point_clouds if p.key not in before]
            if len(added) != 1:
                raise RuntimeError("Expected exactly one new point-cloud asset after import")
            asset = added[0]
            asset.meta["msam3.import_id"] = entry["import_id"]
        # The GUI selects this asset for display. Retain the exact source so a later run
        # cannot accidentally classify a prior result instead of the photo cloud.
        asset.meta["msam3.source_point_cloud_key"] = str(original.key)
        actual = {int(k): int(v) for k, v in asset.point_count_by_class.items() if v}
        if actual != expected or asset.point_count != sum(expected.values()):
            LOG.error("Class preservation failure: expected %s, imported %s", expected, actual)
            raise RuntimeError(
                "Metashape changed point counts or classification codes; "
                "project was not saved. Standalone classified LAS remains valid."
            )
        asset.label = job["settings"]["metashape"]["imported_label"]
        if job["settings"]["metashape"]["make_default"]:
            chunk.point_cloud = asset
        else:
            chunk.point_cloud = previous_default
        parts_dir = directory / "class_exports"
        parts_dir.mkdir(exist_ok=True)
        parts = []
        LOG.warning(
            "Native Metashape LAS export truncates custom codes; exporting class "
            "partitions for lossless LAS 1.4 assembly by msam3"
        )
        for code, count in sorted(expected.items()):
            path = parts_dir / ("class-" + str(code) + ".las")
            export_cloud(chunk, path, chunk.crs, asset.key, classes=[code], progress=progress)
            parts.append({"code": code, "count": count, "path": str(path)})
        write_legends(parts_dir, job["classes"], job.get("model", {}).get("title"))
        output = final_las_path(job, chunk, chunk_count)
        results.append(
            {
                "chunk_key": chunk.key,
                "asset_key": asset.key,
                "parts": parts,
                "crs_wkt": chunk.crs.wkt,
                "directory": str(directory),
                "output": str(output),
            }
        )
    # Every imported cloud has passed class preservation checks before saving.
    document.save()
    write_json(Path(job["work_dir"]) / "stage3_done.json", {"chunks": results})


if __name__ == "__main__":
    run_guarded(main)
