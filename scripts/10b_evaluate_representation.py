"""10b. Evaluate pre-frozen models once; no hyperparameter selection."""
import argparse
from _bootstrap import run_stage
from uav_project.representation_evaluation import evaluate_representation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["auxiliary", "seven"], required=True)
    args = parser.parse_args()
    run_stage("evaluate_representation", lambda: evaluate_representation(args.dataset))
