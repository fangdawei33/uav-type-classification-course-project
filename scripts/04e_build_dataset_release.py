"""04e. Materialize and freeze the completed local dataset version."""
from _bootstrap import run_stage
from uav_project.dataset_release import build_dataset_release

if __name__=='__main__':
    run_stage('clean_data',build_dataset_release)
