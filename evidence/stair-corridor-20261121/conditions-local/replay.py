"""Local-repair T4 entry point; delegates all work to the tracked replay utility."""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE.with_name("conditions-tracked") / "replay.py"
WALK = HERE.parent / "local-repair" / "walk50.npz"

if "--walk" not in sys.argv:
    sys.argv += ["--walk", str(WALK)]
if "--evidence-root" not in sys.argv:
    sys.argv += ["--evidence-root", str(HERE)]
runpy.run_path(str(COMMON), run_name="__main__")
