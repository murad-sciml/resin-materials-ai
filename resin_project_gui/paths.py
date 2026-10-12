"""
Centralizes filesystem paths and Python import path setup so the GUI works
regardless of the current working directory it's launched from, and so the
four core-function packages (which each use plain, non-package-qualified
imports internally, e.g. `import schema`) can be imported side by side
without needing to turn them into installed packages.
"""
from __future__ import annotations

import os
import sys

GUI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(GUI_DIR, ".."))

CORE1_DIR = os.path.join(PROJECT_ROOT, "core1_reverse_design")
CORE2_DIR = os.path.join(PROJECT_ROOT, "core2_data_integration")
CORE3_DIR = os.path.join(PROJECT_ROOT, "core3_validation_protocol")
CORE4_DIR = os.path.join(PROJECT_ROOT, "core4_autonomous_experiment")

for _d in (CORE1_DIR, CORE2_DIR, CORE3_DIR, CORE4_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

CORE2_SAMPLE_DATA = os.path.join(CORE2_DIR, "sample_data")
CORE1_SAMPLE_DATA = os.path.join(CORE1_DIR, "sample_data")
os.makedirs(CORE2_SAMPLE_DATA, exist_ok=True)
os.makedirs(CORE1_SAMPLE_DATA, exist_ok=True)

CORE2_DB_FILE = os.path.join(CORE2_DIR, "sample_data_ingested.db")
CORE2_DB_URL = f"sqlite:///{CORE2_DB_FILE}"

MATERIALS_WIDE_CSV = os.path.join(CORE2_SAMPLE_DATA, "materials_wide.csv")
MATERIALS_WIDE_WITH_COST_CSV = os.path.join(CORE1_SAMPLE_DATA, "materials_wide_with_cost.csv")

# Shared design-space bounds: (resin_voc_ratio, sio2_na2o_ratio, curing_time)
CONTINUOUS_BOUNDS = [(0.05, 0.4), (1.0, 3.5), (1.0, 48.0)]
