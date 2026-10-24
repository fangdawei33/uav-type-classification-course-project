"""06b. Extract fixed features; SigLIP2 requires GPU. No model fitting."""
import argparse
from _bootstrap import run_stage
from uav_project.representation_features import extract_representation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["auxiliary", "seven", "candidates"], required=True)
    parser.add_argument("--representation", choices=["handcrafted", "siglip2"], required=True)
    parser.add_argument("--partition", choices=["development", "holdout"], default="development")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    run_stage("extract_representation", lambda: extract_representation(args.dataset, args.representation, args.partition, args.batch_size))
