"""12. Prepare explicit T4 upload file list; does not upload, train, test or archive."""
import argparse
from _bootstrap import run_stage
from uav_project.gpu_handoff import prepare_gpu_handoff

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auxiliary-only", action="store_true", help="Omit optional unapproved seven-class crop feature-cache inputs")
    args = parser.parse_args()
    run_stage("prepare_gpu_handoff", lambda: prepare_gpu_handoff(not args.auxiliary_only))
