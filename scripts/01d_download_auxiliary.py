"""01d. Download a pinned auxiliary dataset archive, without extracting or training."""
import argparse
from _bootstrap import run_stage
from uav_project.auxiliary_data import download_auxiliary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="cranfield_synthetic_classification")
    args = parser.parse_args()
    run_stage("download_public", lambda: download_auxiliary(args.source))
