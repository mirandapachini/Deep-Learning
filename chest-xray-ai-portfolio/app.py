"""
Lightweight shim to run relocated app in `src/` while preserving paths.
This file keeps the original execution command `streamlit run app.py` working.
"""

import os
import runpy

# When running this shim, set the env var so the relocated app can locate `data/` and logs
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_APP_DIR = os.path.join(HERE)
os.environ.setdefault("CHEST_XRAY_APP_DIR", SRC_APP_DIR)

# Execute the relocated app module
runpy.run_path(os.path.join(os.path.dirname(HERE), "src", "chest_xray_ai_portfolio", "app.py"), run_name="__main__")
