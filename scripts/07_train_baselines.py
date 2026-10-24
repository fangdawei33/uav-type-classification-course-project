"""07. Train four traditional baselines, grouped CV and grid search; CPU only."""
from _bootstrap import run_stage
from uav_project.baseline_models import run_baselines

if __name__ == "__main__":
    run_stage("train_baselines", run_baselines)
