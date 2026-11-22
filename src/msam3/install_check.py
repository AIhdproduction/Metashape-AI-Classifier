"""Quick check that .venv matches this checkout. Standard library only: it must work when stale."""

import sys
from importlib import metadata
from importlib.util import find_spec

# module -> (distribution, exact version or None)
REQUIRED = {
    "nicegui": ("nicegui", "3.17.1"),
    "transformers": ("transformers", "5.22.1"),
    "psutil": ("psutil", None),
    "torch": ("torch", None),
}


def problems(required=None) -> list[str]:
    found = []
    for module, (distribution, wanted) in (required or REQUIRED).items():
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            found.append(f"{distribution} fehlt")
            continue
        if find_spec(module) is None:
            found.append(f"{distribution} ist unvollständig installiert")
        elif wanted and version != wanted:
            found.append(f"{distribution} {version} statt {wanted}")
    return found


def main() -> int:
    found = problems()
    if not found:
        return 0
    sys.stdout.write("Die Installation ist veraltet: " + ", ".join(found) + "\n")
    sys.stdout.write("Bitte install.bat im Repository erneut ausführen.\n")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
