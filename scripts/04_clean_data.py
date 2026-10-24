"""04. Prepare seven-class cleaning evidence; apply only reviewed decisions."""
import argparse
from _bootstrap import run_stage
from uav_project.cleaning import clean_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="仅在审核与来源分组完成后生成清洗数据")
    parser.add_argument("--view", choices=["full_frame", "target_crop"], help="应用时明确分类输入协议")
    args = parser.parse_args()
    if args.apply and args.view is None:
        parser.error("--apply必须同时指定--view")
    run_stage("clean_data", lambda: clean_data(args.apply, args.view))
