"""09. Freeze data, code, hyperparameters and trained weights before testing."""
from _bootstrap import run_stage
from uav_project.evaluation import freeze

if __name__ == "__main__":
    run_stage("freeze_experiment", freeze)
