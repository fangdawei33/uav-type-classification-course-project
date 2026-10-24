"""03c. Export visual screening CSVs, actual data statistics and audit figures."""
from _bootstrap import run_stage
from uav_project.data_audit_export import export_data_audit

if __name__ == "__main__":
    run_stage("review_data", export_data_audit)
