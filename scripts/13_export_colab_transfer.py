"""13. Export separate code/image transfer archives plus stepwise Colab notebook."""
from _bootstrap import run_stage
from uav_project.colab_transfer import export_colab_transfer

if __name__ == "__main__":
    run_stage("export_colab_transfer", export_colab_transfer)
