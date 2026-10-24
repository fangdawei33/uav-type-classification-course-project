"""03g. Inventory acquired expansion images/annotations without model execution."""
from _bootstrap import run_stage
from uav_project.expansion_data import catalog_visiodect

if __name__ == '__main__':
    run_stage('review_data', catalog_visiodect)
