"""Prepare modular code + image transport archives and a per-stage Colab notebook."""
from _bootstrap import PROJECT_ROOT
from uav_project.uav12_protocol import log_stage
from uav_project.uav12_transfer import export_uav12_transfer

if __name__ == "__main__":
    log_stage("export_uav12_transfer", export_uav12_transfer)
