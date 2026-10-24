"""03e. Decode and catalog auxiliary images, keeping original source labels/splits."""
import argparse
from _bootstrap import run_stage
from uav_project.auxiliary_catalog import catalog_auxiliary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="cranfield_synthetic_classification")
    args = parser.parse_args()
    run_stage("review_data", lambda: catalog_auxiliary(args.source))
