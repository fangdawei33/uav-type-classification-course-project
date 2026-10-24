"""03b. Prepare external review evidence or summarize current decisions."""
import argparse
from _bootstrap import run_stage
from uav_project.external_review import prepare_external_review, summarize_external_review

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-only", action="store_true", help="Update current status without changing evidence images")
    parser.add_argument("--audit-integrity", action="store_true", help="Decode acquired images and verify saved files/evidence; no software tests or models")
    args = parser.parse_args()
    def action():
        if args.audit_integrity:
            from uav_project.external_integrity import audit_external_integrity
            return audit_external_integrity()
        if not args.summary_only:
            prepare_external_review()
        return summarize_external_review()
    run_stage("review_data", action)
