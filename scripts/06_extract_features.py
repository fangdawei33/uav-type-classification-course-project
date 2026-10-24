"""06. Cache hand-crafted training/validation features; test images stay unused."""
from _bootstrap import run_stage
from uav_project.features import extract_training_features

if __name__ == "__main__":
    run_stage("extract_features", extract_training_features)
