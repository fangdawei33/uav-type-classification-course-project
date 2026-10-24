"""02e. Save bounded public platform metadata; never download dataset media."""
import argparse
from _bootstrap import run_stage
from uav_project.platform_inventory import inventory_platforms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["huggingface", "kaggle", "educoder"])
    parser.add_argument("--source", action="append", help="Repeat for bounded follow-up sources")
    args = parser.parse_args()
    run_stage("crawl_external", lambda: inventory_platforms(args.platform, args.source))

