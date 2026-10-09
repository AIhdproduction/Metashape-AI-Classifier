"""Select this file in Metashape Professional: Tools > Run Script (no arguments)."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metashape_scripts.gui as gui  # noqa: E402

# Metashape retains imported modules between Tools > Run Script invocations. Reload the
# launcher so an updated script takes effect without requiring the operator to restart.
gui = importlib.reload(gui)

if __name__ == "__main__":
    gui.start()
