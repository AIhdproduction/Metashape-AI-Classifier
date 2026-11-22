"""English command-line interface; errors are explicit and return a nonzero exit status."""

import argparse
import logging
import sys
from pathlib import Path

from msam3 import backends
from msam3.config import load_settings


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="msam3", description="Metashape AI point-cloud classifier (FlurSeg, SAM 3)"
    )
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("run", "stage1", "stage2", "stage3", "validate", "classes"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "--config", type=Path, help="Settings YAML (defaults to config/default.yml)"
        )
        sub.add_argument(
            "--model",
            choices=tuple(backends.MODELS),
            help="Segmentation model (defaults to segmentation.backend, normally flurseg)",
        )
        sub.add_argument(
            "--classes",
            type=Path,
            help="Class YAML (defaults to the selected model's classes, with your SAM 3 edits)",
        )
        if name != "classes":
            sub.add_argument("--project", type=Path, required=True, help="Saved .psx/.psz project")
            sub.add_argument("--chunk", action="append", help="Chunk label; may be repeated")
            sub.add_argument("--work-dir", type=Path, help="Override intermediate work root")
        if name == "stage2":
            sub.add_argument(
                "--voting-only", action="store_true", help="Require valid image caches"
            )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        settings = load_settings(args.config)
        if args.model is not None:
            settings.segmentation.backend = args.model
        segmenter = backends.selected(settings)
        classes = segmenter.classes(args.classes)
        if args.command == "classes":
            sys.stdout.write(f"Model: {segmenter.title}\n")
            sys.stdout.write(f"{'Name':28} {'Code':>4}  Prompts\n")
            sys.stdout.write(f"{'Unclassified':28} {classes.unknown_code:4}  (undecided points)\n")
            for cls in classes.classes:
                prompts = ", ".join(cls.prompts) or "(fixed class of the model)"
                sys.stdout.write(f"{cls.name:28} {cls.code:4}  {prompts}\n")
            return 0
        if args.chunk is not None:
            settings.metashape.chunks = args.chunk
        if args.work_dir is not None:
            settings.paths.work_dir = args.work_dir.resolve()
        from msam3.driver import execute

        execute(
            args.command,
            args.project,
            settings,
            classes,
            voting_only=getattr(args, "voting_only", False),
        )
        return 0
    except KeyboardInterrupt:
        logging.error("Interrupted. Completed image caches will be reused on the next run.")
        return 130
    except Exception as error:
        logging.error("%s", error)
        logging.debug("Failure details", exc_info=True)
        return 1
