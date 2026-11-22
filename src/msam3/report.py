"""Machine-readable quality, coverage and runtime summaries."""

from pathlib import Path

import numpy as np

from msam3.cache import atomic_json
from msam3.config import Classes


def write_report(
    path: Path,
    counts: np.ndarray,
    classes: Classes,
    images: int,
    views: np.ndarray,
    validation: dict,
    timings: dict,
    voting_points: int,
    ground_assist: dict | None = None,
    vegetation_assist: dict | None = None,
    model: dict | None = None,
):
    total = int(counts.sum())
    report = {
        "point_count": total,
        "voting_point_count": voting_points,
        "unknown_share": float(counts[classes.unknown_code] / total),
        "images_used": images,
        "mean_views_per_point": float(views.mean()),
        "mean_views_population": "voxel voting points",
        "ground_assist": ground_assist or {"metashape_ground_points": 0, "resolved": {}},
        "vegetation_assist": vegetation_assist
        or {"metashape_low_points": 0, "metashape_high_points": 0, "resolved": {}},
        "timing_seconds": timings,
        "validation": validation,
        "classes": [
            {
                "name": cls.name,
                "code": cls.code,
                "count": int(counts[cls.code]),
                "percentage": float(counts[cls.code] / total * 100),
            }
            for cls in classes.classes
        ],
    }
    report["classes"].append(
        {
            "name": "Unclassified",
            "code": classes.unknown_code,
            "count": int(counts[classes.unknown_code]),
            "percentage": float(report["unknown_share"] * 100),
        }
    )
    if model:
        report["model"] = model
    atomic_json(path, report)
    return report
