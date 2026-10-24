"""05b. Prepare a versioned multiclass protocol; no training is performed."""
import argparse
from _bootstrap import run_stage
from uav_project.representation_protocol import prepare_protocol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["auxiliary", "seven"], required=True)
    args = parser.parse_args()
    run_stage("prepare_representation", lambda: prepare_protocol(args.dataset))
