"""Restore the latest frozen Qwen prediction prefix, then continue evaluation."""
import os
from pathlib import Path
import runpy

from _bootstrap import PROJECT_ROOT


if __name__ == "__main__":
    progress_root = os.environ.get("UAV12_PROGRESS_ROOT")
    if not progress_root:
        raise RuntimeError("Initialize the Colab session before resuming evaluation")
    restore = runpy.run_path(str(PROJECT_ROOT / "scripts/10r_restore_uav12_predictions.py"))
    if (Path(progress_root) / "qwen_progress.json").is_file():
        restore["restore_prediction_files"](PROJECT_ROOT, progress_root)
    from uav_project.uav12_protocol import log_stage
    from uav_project.uav12_evaluation import evaluate_uav12
    log_stage("evaluate_uav12_holdout", evaluate_uav12)
