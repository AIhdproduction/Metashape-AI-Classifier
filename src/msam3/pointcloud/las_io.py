"""Chunked LAS I/O, preserved attributes, and the LAS 1.4 classification lookup."""

import copy
import csv
from pathlib import Path

import laspy
import numpy as np
from laspy.vlrs.known import ClassificationLookupVlr
from pyproj import CRS

from msam3.cache import atomic_json
from msam3.config import Classes


def legend(classes: Classes) -> list[dict]:
    return [
        {"name": "Unclassified", "code": classes.unknown_code, "prompts": []},
        *[cls.model_dump() for cls in classes.classes],
    ]


def write_legends(directory: Path, classes: Classes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = legend(classes)
    atomic_json(directory / "class_legend.json", rows)
    with (directory / "class_legend.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "code", "prompts"])
        writer.writeheader()
        writer.writerows({**row, "prompts": " | ".join(row["prompts"])} for row in rows)


def classified_header(
    source: laspy.LasHeader, classes: Classes, crs_wkt: str | None = None
) -> laspy.LasHeader:
    header = copy.deepcopy(source)
    point_format = laspy.PointFormat(8 if "nir" in source.point_format.dimension_names else 7)
    for dimension in source.point_format.extra_dimensions:
        point_format.add_extra_dimension(
            laspy.ExtraBytesParams(
                name=dimension.name,
                type=dimension.dtype,
                description=dimension.description,
                scales=dimension.scales,
                offsets=dimension.offsets,
            )
        )
    header.set_version_and_point_format(laspy.header.Version(1, 4), point_format)
    header.vlrs = [v for v in header.vlrs if not (v.user_id == "LASF_Spec" and v.record_id == 0)]
    lookup = ClassificationLookupVlr()
    # LAS 1.4 lookup descriptions have exactly 15 bytes; full names are in the legends.
    names = {row["code"]: row["name"] for row in legend(classes)}
    for code in range(256):
        lookup[code] = names.get(code, "")[:15]
    header.vlrs.append(lookup)
    crs = CRS.from_wkt(crs_wkt) if crs_wkt else source.parse_crs()
    if crs is not None:
        # Formats 6+ use WKT, even when the input stored its CRS in legacy GeoTIFF VLRs.
        header.add_crs(crs)
    header.generating_software = "AIhdproduction msam3"
    return header


def convert_records(records, header):
    converted = laspy.ScaleAwarePointRecord.zeros(len(records), header=header)
    converted.copy_fields_from(records)
    if "scan_angle_rank" in records.point_format.dimension_names:
        converted.scan_angle = np.rint(np.asarray(records.scan_angle_rank) / 0.006).astype(np.int16)
    # XYZ integers depend on source offsets/scales, so always assign physical coordinates.
    converted.x, converted.y, converted.z = records.x, records.y, records.z
    return converted


def write_classified(source: Path, output: Path, codes, classes: Classes, batch_size: int):
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp.las")
    counts = np.zeros(256, dtype=np.int64)
    with laspy.open(source) as reader:
        if reader.header.point_count != len(codes):
            raise ValueError("Classification count does not match the source LAS")
        header = classified_header(reader.header, classes)
        with laspy.open(temp, mode="w", header=header) as writer:
            offset = 0
            for points in reader.chunk_iterator(batch_size):
                converted = convert_records(points, header)
                labels = np.asarray(codes[offset : offset + len(points)], dtype=np.uint8)
                converted.classification = labels
                writer.write_points(converted)
                counts += np.bincount(labels, minlength=256)
                offset += len(points)
    temp.replace(output)
    write_legends(output.parent, classes)
    return counts


def merge_class_exports(
    parts: list[dict], output: Path, classes: Classes, crs_wkt: str, batch_size: int
) -> np.ndarray:
    """Repair Metashape's five-bit LAS export using separately exported class partitions."""
    if not parts:
        raise ValueError("Metashape returned no nonempty class exports")
    headers = []
    for part in parts:
        with laspy.open(part["path"]) as reader:
            if reader.header.point_count != part["count"]:
                raise ValueError(f"Class {part['code']}: exported point count changed")
            headers.append(reader.header)
    header = classified_header(headers[0], classes, crs_wkt)
    # One shared scale/offset must cover every partition without integer overflow.
    low = np.min([h.mins for h in headers], axis=0)
    high = np.max([h.maxs for h in headers], axis=0)
    header.offsets = (low + high) / 2
    header.scales = np.maximum(
        np.max([h.scales for h in headers], axis=0), (high - low) / (2**32 - 2)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp.las")
    counts = np.zeros(256, dtype=np.int64)
    with laspy.open(temp, mode="w", header=header) as writer:
        for part in parts:
            with laspy.open(part["path"]) as reader:
                for points in reader.chunk_iterator(batch_size):
                    converted = convert_records(points, header)
                    converted.classification = np.full(len(points), part["code"], dtype=np.uint8)
                    writer.write_points(converted)
                    counts[part["code"]] += len(points)
    temp.replace(output)
    write_legends(output.parent, classes)
    return counts
