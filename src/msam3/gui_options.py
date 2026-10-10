"""External start window for one Metashape SAM 3 run."""

import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, IntVar, StringVar, Tk, ttk

from msam3.cache import atomic_json

IMAGE_LINE = re.compile(
    r"^(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ INFO .* image "
    r"(?P<done>\d+)/(?P<total>\d+) \([^)]+\) complete$"
)
PROFILES = {
    "Schnell": {"image_scale": 0.50, "tile_overlap": 0.10, "min_views": 1},
    "Ausgewogen": {"image_scale": 0.75, "tile_overlap": 0.20, "min_views": 1},
    "Hohe Genauigkeit": {"image_scale": 1.00, "tile_overlap": 0.30, "min_views": 2},
}


def historical_seconds_per_image(sessions_root: Path) -> float:
    """Use recent completed image timings as an estimate for this computer and GPU."""
    samples = []
    for log_path in sessions_root.glob("*/stage2.log"):
        records = []
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = IMAGE_LINE.match(line)
            if match:
                records.append(match)
        if len(records) < 8:
            continue
        first = datetime.strptime(records[0]["time"], "%Y-%m-%d %H:%M:%S")
        last = datetime.strptime(records[-1]["time"], "%Y-%m-%d %H:%M:%S")
        images = int(records[-1]["done"]) - int(records[0]["done"])
        if images > 0:
            seconds = (last - first).total_seconds() / images
            if 1 <= seconds <= 900:
                samples.append(seconds)
    return statistics.median(samples) if samples else 45.0


def estimate_seconds(
    image_count: int, image_scale: float, tile_overlap: float, min_views: int, baseline: float
) -> float:
    """Give a deliberately conservative wall-clock estimate, including Metashape I/O."""
    resolution_factor = (image_scale / 0.75) ** 2
    overlap_factor = ((1 - 0.20) / (1 - tile_overlap)) ** 2
    view_factor = 1 + max(0, min_views - 1) * 0.08
    classification = image_count * baseline * resolution_factor * overlap_factor * view_factor
    return classification + max(180, min(1200, image_count * 2.0))


def format_duration(seconds: float) -> str:
    minutes = max(1, round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min" if hours else f"ca. {minutes} min"


def settings_override(
    image_scale: float,
    tile_overlap: float,
    prompt_batch_size: int,
    min_views: int,
    ground_assist: bool,
    postprocess: bool,
    vegetation: bool,
    road: bool,
    smoothing: bool,
) -> dict:
    """Translate the visible controls to the validated job settings schema."""
    return {
        "sam": {
            "image_scale": image_scale,
            "tile_overlap": tile_overlap,
            "prompt_batch_size": prompt_batch_size,
        },
        "voting": {"min_views": min_views},
        "ground_assist": {"enabled": ground_assist},
        "postprocess": {
            "enabled": postprocess,
            "vegetation": {"enabled": vegetation},
            "road": {"enabled": road},
        },
        "smoothing": {"enabled": smoothing},
    }


class OptionsWindow:
    def __init__(self, session: Path):
        self.session = session
        session_data = json.loads((session / "session.json").read_text(encoding="utf-8"))
        self.image_count = int(session_data.get("image_count", 0))
        self.baseline = historical_seconds_per_image(session.parent)
        self.root = Tk()
        self.root.title("SAM 3 – Start und Einstellungen")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

        self.profile = StringVar(value="Ausgewogen")
        self.image_scale = StringVar(value="75 %")
        self.overlap = StringVar(value="20 %")
        self.prompt_batch_size = IntVar(value=8)
        self.min_views = IntVar(value=1)
        self.ground_assist = BooleanVar(value=True)
        self.postprocess = BooleanVar(value=True)
        self.vegetation = BooleanVar(value=True)
        self.road = BooleanVar(value=True)
        self.smoothing = BooleanVar(value=False)
        self.estimate = StringVar()
        self._build()
        self._refresh_estimate()

    def _build(self):
        frame = ttk.Frame(self.root, padding=18)
        frame.grid()
        ttk.Label(
            frame,
            text="SAM 3 Klassifikation starten",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            frame,
            text=f"Projekt: {self.image_count} aktivierte Kamerabilder",
        ).grid(row=1, column=0, columnspan=2, pady=(4, 12), sticky="w")

        quality = ttk.LabelFrame(frame, text="Bildqualität und Aufteilung", padding=12)
        quality.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(quality, text="Profil").grid(row=0, column=0, sticky="w")
        profile = ttk.Combobox(
            quality, textvariable=self.profile, values=list(PROFILES), state="readonly", width=22
        )
        profile.grid(row=0, column=1, padx=(18, 0), sticky="ew")
        profile.bind("<<ComboboxSelected>>", self._apply_profile)
        ttk.Label(quality, text="Bildauflösung für SAM").grid(
            row=1, column=0, pady=(8, 0), sticky="w"
        )
        scale = ttk.Combobox(
            quality,
            textvariable=self.image_scale,
            values=("50 %", "75 %", "100 %"),
            state="readonly",
            width=22,
        )
        scale.grid(row=1, column=1, padx=(18, 0), pady=(8, 0), sticky="ew")
        scale.bind("<<ComboboxSelected>>", self._changed)
        ttk.Label(quality, text="Kachelüberlappung").grid(row=2, column=0, pady=(8, 0), sticky="w")
        overlap = ttk.Combobox(
            quality,
            textvariable=self.overlap,
            values=("10 %", "20 %", "30 %"),
            state="readonly",
            width=22,
        )
        overlap.grid(row=2, column=1, padx=(18, 0), pady=(8, 0), sticky="ew")
        overlap.bind("<<ComboboxSelected>>", self._changed)
        ttk.Label(quality, text="Bildbeobachtungen je Punkt").grid(
            row=3, column=0, pady=(8, 0), sticky="w"
        )
        views = ttk.Combobox(
            quality, textvariable=self.min_views, values=(1, 2, 3), state="readonly", width=22
        )
        views.grid(row=3, column=1, padx=(18, 0), pady=(8, 0), sticky="ew")
        views.bind("<<ComboboxSelected>>", self._changed)
        ttk.Label(quality, text="GPU-Prompt-Batch").grid(row=4, column=0, pady=(8, 0), sticky="w")
        batch = ttk.Combobox(
            quality,
            textvariable=self.prompt_batch_size,
            values=(4, 8, 12, 16),
            state="readonly",
            width=22,
        )
        batch.grid(row=4, column=1, padx=(18, 0), pady=(8, 0), sticky="ew")
        batch.bind("<<ComboboxSelected>>", self._changed)

        filters = ttk.LabelFrame(frame, text="Hilfe und Nachfilter", padding=12)
        filters.grid(row=3, column=0, columnspan=2, pady=(12, 0), sticky="ew")
        ttk.Checkbutton(filters, text="Metashape-Bodenhilfe", variable=self.ground_assist).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Checkbutton(
            filters, text="Geometrischen Nachfilter verwenden", variable=self.postprocess
        ).grid(row=1, column=0, pady=(5, 0), sticky="w")
        ttk.Checkbutton(
            filters, text="Vegetation nach DTM-Höhe trennen", variable=self.vegetation
        ).grid(row=2, column=0, pady=(5, 0), sticky="w")
        ttk.Checkbutton(filters, text="Straßenflächen korrigieren", variable=self.road).grid(
            row=3, column=0, pady=(5, 0), sticky="w"
        )
        ttk.Checkbutton(filters, text="Zusätzliche kNN-Glättung", variable=self.smoothing).grid(
            row=4, column=0, pady=(5, 0), sticky="w"
        )

        ttk.Label(frame, textvariable=self.estimate, font=("Segoe UI", 10, "bold")).grid(
            row=4, column=0, columnspan=2, pady=(14, 2), sticky="w"
        )
        ttk.Label(
            frame,
            text=(
                "Der Richtwert enthält Klassifikation, Export und Import. "
                "Die Auswahl gilt nur für diesen Lauf."
            ),
            wraplength=590,
        ).grid(row=5, column=0, columnspan=2, sticky="w")
        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=2, pady=(16, 0), sticky="e")
        ttk.Button(buttons, text="Abbrechen", command=self.cancel).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(buttons, text="Klassifikation starten", command=self.start).grid(row=0, column=1)

    def _apply_profile(self, _event=None):
        values = PROFILES[self.profile.get()]
        self.image_scale.set(f"{int(values['image_scale'] * 100)} %")
        self.overlap.set(f"{int(values['tile_overlap'] * 100)} %")
        self.min_views.set(values["min_views"])
        self._refresh_estimate()

    def _changed(self, _event=None):
        self.profile.set("Benutzerdefiniert")
        self._refresh_estimate()

    def _values(self):
        return (
            int(self.image_scale.get().split()[0]) / 100,
            int(self.overlap.get().split()[0]) / 100,
            int(self.prompt_batch_size.get()),
            int(self.min_views.get()),
        )

    def _refresh_estimate(self):
        image_scale, overlap, _, min_views = self._values()
        estimate = estimate_seconds(
            self.image_count, image_scale, overlap, min_views, self.baseline
        )
        self.estimate.set(f"Geschätzte Laufzeit: {format_duration(estimate)}")

    def start(self):
        image_scale, overlap, prompt_batch_size, min_views = self._values()
        atomic_json(
            self.session / "options.json",
            {
                "profile": self.profile.get(),
                "estimated_seconds": round(
                    estimate_seconds(
                        self.image_count, image_scale, overlap, min_views, self.baseline
                    )
                ),
                "settings_override": settings_override(
                    image_scale,
                    overlap,
                    prompt_batch_size,
                    min_views,
                    self.ground_assist.get(),
                    self.postprocess.get(),
                    self.vegetation.get(),
                    self.road.get(),
                    self.smoothing.get(),
                ),
            },
        )
        self.root.destroy()

    def cancel(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    OptionsWindow(args.session.resolve()).run()
    return 0 if (args.session / "options.json").is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
