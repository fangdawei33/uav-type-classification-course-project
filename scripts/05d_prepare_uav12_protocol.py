"""Freeze the v3 group-aware 12-label diagnostic split and development CV folds."""
import argparse
from _bootstrap import PROJECT_ROOT
from uav_project.uav12_protocol import prepare_protocol, log_stage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-before-training", action="store_true",
                        help="Rewrite only protocol files when no features/models exist")
    args = parser.parse_args()
    log_stage("prepare_uav12_protocol", lambda: prepare_protocol(args.refresh_before_training))
