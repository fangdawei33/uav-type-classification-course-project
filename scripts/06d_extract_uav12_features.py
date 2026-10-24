"""Extract fixed hand-crafted or frozen SigLIP2 features for the v3 protocol."""
import argparse
from _bootstrap import PROJECT_ROOT
from uav_project.uav12_protocol import log_stage
from uav_project.uav12_features import extract_uav12_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=["handcrafted", "siglip2"], required=True)
    parser.add_argument("--partition", choices=["development", "holdout"], default="development")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    stage = f"extract_uav12_{args.representation}_{args.partition}"
    log_stage(stage, lambda: extract_uav12_features(args.representation, args.partition, args.batch_size))
