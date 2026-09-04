"""Test configuration for the source-layout package."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Test the checkout under development even when another ``lifts`` release is
# installed in the active environment.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Keep tests deterministic, CPU-friendly, and independent of a display server.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
