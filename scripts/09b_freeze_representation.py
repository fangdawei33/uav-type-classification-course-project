"""09b. Freeze declared multiclass models before opening the holdout."""
import argparse
from _bootstrap import run_stage
from uav_project.representation_training import freeze_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["auxiliary", "seven"], required=True)
    args = parser.parse_args()
    run_stage("freeze_representation", lambda: freeze_experiment(args.dataset))
