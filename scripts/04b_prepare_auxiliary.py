"""04b. Prepare auxiliary-only split manifests and quarantine exact pixel leakage."""
import argparse
from _bootstrap import run_stage
from uav_project.auxiliary_catalog import prepare_auxiliary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="cranfield_synthetic_classification")
    args = parser.parse_args()
    run_stage("clean_data", lambda: prepare_auxiliary(args.source))
