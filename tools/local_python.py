"""Run the .venv on a local copy of its base Python when that Python lives on a network drive.

Every process of the app (installation check, window, workers) loads the standard library from
the base Python. From a network share this took 5 s for asyncio/logging/email alone and 8.5 s
for the window's web framework; from a local disk 0.15 s and 0.8 s. A venv finds its base
Python through ``home`` in pyvenv.cfg, so a local copy of the same version is enough: the
installed packages stay untouched. Standard library only; run by install.bat with the venv's
python. Safe to run again: it refreshes the copy when the base Python changed.
"""

import ctypes
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / ".python-local"
CONFIG = ROOT / ".venv" / "pyvenv.cfg"
SKIPPED = {"site-packages", "test", "idlelib", "__pycache__"}


def read_config():
    lines = CONFIG.read_text(encoding="utf-8").splitlines()
    values = {}
    for line in lines:
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return lines, values


def is_remote(path: Path) -> bool:
    text = str(path)
    if text.startswith("\\\\"):
        return True
    if os.name != "nt" or len(text) < 2 or text[1] != ":":
        return False
    return ctypes.windll.kernel32.GetDriveTypeW(text[:2] + "\\") == 4  # DRIVE_REMOTE


def signature(home: Path) -> str:
    """Version file sizes and times: a changed base Python means a new copy."""
    parts = []
    for name in ("python.exe", f"python{sys.version_info.major}{sys.version_info.minor}.dll"):
        path = home / name
        if path.is_file():
            stat = path.stat()
            parts.append(f"{name}:{stat.st_size}:{int(stat.st_mtime)}")
    return ";".join(parts)


def copy_runtime(source: Path, target: Path) -> None:
    temporary = target.with_name(target.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    for item in source.iterdir():
        if item.is_file() and item.suffix.lower() in (".exe", ".dll"):
            shutil.copy2(item, temporary / item.name)
    for folder in ("DLLs", "Lib"):
        shutil.copytree(
            source / folder,
            temporary / folder,
            ignore=lambda directory, names: [name for name in names if name in SKIPPED],
        )
    (temporary / "Lib" / "site-packages").mkdir(exist_ok=True)
    (temporary / "source.txt").write_text(
        str(source) + "\n" + signature(source) + "\n", encoding="utf-8"
    )
    shutil.rmtree(target, ignore_errors=True)
    temporary.rename(target)


def main() -> int:
    if os.name != "nt" or not CONFIG.is_file():
        return 0
    lines, values = read_config()
    home = Path(values.get("home", ""))
    origin_file = LOCAL / "source.txt"
    if home == LOCAL:
        # Already local: refresh when the original base Python changed since the copy.
        source = Path(origin_file.read_text(encoding="utf-8").splitlines()[0])
        if not source.is_dir():
            return 0
    elif is_remote(home):
        source = home
    else:
        print(f"Base Python is local: {home}")
        return 0
    stored = origin_file.read_text(encoding="utf-8").splitlines() if origin_file.is_file() else []
    if stored[:2] != [str(source), signature(source)]:
        print(f"Copying the base Python from {source} to {LOCAL} (network drives start slowly) ...")
        copy_runtime(source, LOCAL)
    if home != LOCAL:
        backup = CONFIG.with_name("pyvenv.cfg.network")
        if not backup.exists():
            shutil.copy2(CONFIG, backup)
        replace = {"home": f"home = {LOCAL}", "executable": f"executable = {LOCAL / 'python.exe'}"}
        updated = [replace.get(line.partition("=")[0].strip(), line) for line in lines]
        CONFIG.write_text("\n".join(updated) + "\n", encoding="utf-8")
        print(f"The .venv now starts from {LOCAL}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
