"""07b. Train four multiclass baselines and tune the CV-selected family."""
import argparse
from _bootstrap import run_stage
from uav_project.representation_training import train_classifiers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["auxiliary", "seven"], required=True)
    parser.add_argument("--representation", choices=["handcrafted", "siglip2"], required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    run_stage("train_representation", lambda: train_classifiers(args.dataset, args.representation, args.jobs))
