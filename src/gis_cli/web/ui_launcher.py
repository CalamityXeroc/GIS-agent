from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run() -> None:
    app_path = Path(__file__).with_name("streamlit_app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    raise SystemExit(subprocess.call(cmd))
