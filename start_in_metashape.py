"""Select this file in Metashape Professional: Tools > Run Script (no arguments)."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metashape_scripts import common, gui, stage1_export, stage3_import  # noqa: E402

# Metashape retains every imported module between Tools > Run Script invocations. Reload the
# complete embedded stack, in dependency order, so fixes take effect without a restart.
importlib.invalidate_caches()
common = importlib.reload(common)
stage1_export = importlib.reload(stage1_export)
stage3_import = importlib.reload(stage3_import)
gui = importlib.reload(gui)

if __name__ == "__main__":
    gui.start()
