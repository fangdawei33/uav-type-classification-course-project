"""04c. Export reviewed target crops, preserving the formal source-approval gate."""
from _bootstrap import run_stage
from uav_project.candidate_crops import export_candidate_crops

if __name__ == "__main__":
    run_stage("clean_data", export_candidate_crops)
