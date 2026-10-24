"""Evaluate all frozen traditional and Qwen model variants on the sealed holdout."""
from _bootstrap import PROJECT_ROOT
from uav_project.uav12_protocol import log_stage
from uav_project.uav12_evaluation import evaluate_uav12

if __name__ == "__main__":
    log_stage("evaluate_uav12_holdout", evaluate_uav12)
