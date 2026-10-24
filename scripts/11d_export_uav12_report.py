"""Export tables, figures and Markdown results after the one-time frozen evaluation."""
from _bootstrap import PROJECT_ROOT
from uav_project.uav12_protocol import log_stage
from uav_project.uav12_evaluation import export_uav12_report

if __name__ == "__main__":
    log_stage("export_uav12_report", export_uav12_report)
