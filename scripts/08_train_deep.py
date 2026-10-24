"""08. GPU stage: EfficientNet fine tuning, repeated seeds and ablations."""
from _bootstrap import run_stage
from uav_project.deep_model import run_deep

if __name__ == "__main__":
    run_stage("train_deep", run_deep)
