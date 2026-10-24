"""03d. Export the data-stage PDF only; requires reportlab and Windows SimHei."""
from _bootstrap import run_stage
from uav_project.data_report_pdf import export_data_report_pdf

if __name__ == '__main__':
    run_stage('review_data', export_data_report_pdf)
