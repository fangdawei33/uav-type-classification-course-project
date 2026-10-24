"""11. Create tables, figures and paper inputs from measured experiment results."""
import argparse
from _bootstrap import run_stage
from uav_project.reporting import export_results
from uav_project.common import bundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", action="store_true", help="可选：压缩结果材料用于回传；不把源码合成单文件")
    args = parser.parse_args()
    run_stage("export_results", export_results)
    if args.zip:
        bundle("results")
