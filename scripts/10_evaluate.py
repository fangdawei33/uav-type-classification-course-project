"""10. Evaluate frozen models on both held-out sets; never tune on results."""
from _bootstrap import run_stage
from uav_project.evaluation import evaluate

if __name__ == "__main__":
    run_stage("evaluate", evaluate)
