"""Freeze every declared 12-label model before feature extraction on holdout."""
from _bootstrap import PROJECT_ROOT
from uav_project.uav12_protocol import log_stage
from uav_project.uav12_models import freeze_uav12

if __name__ == "__main__":
    log_stage("freeze_uav12_experiment", freeze_uav12)
