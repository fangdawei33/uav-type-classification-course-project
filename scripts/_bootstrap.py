"""Shared script bootstrap. This file is imported, not run as a workflow stage."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_project.common import run_stage
