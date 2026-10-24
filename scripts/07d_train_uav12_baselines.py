"""Train four distinct classifiers and tune the CV-selected family (development only)."""
import argparse
from _bootstrap import PROJECT_ROOT
from uav_project.uav12_protocol import log_stage
from uav_project.uav12_models import train_uav12_baselines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=["handcrafted", "siglip2"], required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    log_stage(f"train_uav12_baselines_{args.representation}",
              lambda: train_uav12_baselines(args.representation, args.jobs))
