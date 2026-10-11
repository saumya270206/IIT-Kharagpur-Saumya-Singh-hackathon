"""Launch the Streamlit dashboard (Module A home page + Module B / Engine pages).

    python run_dashboard.py      -> http://localhost:8501
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent / "dashboard" / "app.py"

if __name__ == "__main__":
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(APP),
         "--server.port", "8501", *sys.argv[1:]]))
