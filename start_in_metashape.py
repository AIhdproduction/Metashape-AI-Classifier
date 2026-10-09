"""Select this file in Metashape Professional: Tools > Run Script (no arguments)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metashape_scripts.gui import start  # noqa: E402

if __name__ == "__main__":
    start()
