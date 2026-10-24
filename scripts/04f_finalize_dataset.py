"""04f. Finalize current acquired data or check the frozen data-file hashes."""
import argparse
from _bootstrap import run_stage
from uav_project.dataset_completion import build_dataset_completion, audit_dataset_completion

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true', help='Read and verify frozen data files only')
    args = parser.parse_args()
    run_stage('clean_data', audit_dataset_completion if args.audit_only else build_dataset_completion)
