"""01e. Download the fixed author VisioDECT release, separate from existing data."""
from _bootstrap import run_stage
from uav_project.expansion_data import download_visiodect

if __name__ == '__main__':
    run_stage('download_public', download_visiodect)
