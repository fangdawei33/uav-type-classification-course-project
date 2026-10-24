"""04d. Export externally sourced candidate target crops and provenance."""
from _bootstrap import run_stage
from uav_project.external_candidate_crops import export_external_candidate_crops

if __name__ == '__main__':
    run_stage('clean_data', export_external_candidate_crops)
