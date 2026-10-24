"""11b. Export actual multiclass metrics, plots and Markdown; no PDF generation."""
import argparse
from _bootstrap import run_stage
from uav_project.representation_evaluation import export_representation_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["auxiliary", "seven"], required=True)
    args = parser.parse_args()
    run_stage("report_representation", lambda: export_representation_report(args.dataset))
