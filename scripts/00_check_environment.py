"""00. Print installed packages and hardware; does not install or train anything."""
from _bootstrap import run_stage
from uav_project.common import environment

if __name__ == "__main__":
    run_stage("check_environment", environment)
