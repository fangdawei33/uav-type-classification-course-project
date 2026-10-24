"""05. Persist one group-aware train/validation/test/external-test assignment."""
from _bootstrap import run_stage
from uav_project.splitting import split_data

if __name__ == "__main__":
    run_stage("split_data", split_data)
