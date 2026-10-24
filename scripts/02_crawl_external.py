"""02. Collect seven-class image candidates from explicitly configured public pages."""
import argparse
from _bootstrap import run_stage
from uav_project.crawler import crawl_external

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-id", type=int, choices=range(7), help="只采集一个类别；默认使用全部已配置来源")
    parser.add_argument("--max-per-class", type=int, help="每类候选原图上限；默认读取配置，不是独立测试样本数")
    args = parser.parse_args()
    if args.max_per_class is not None and args.max_per_class < 1:
        parser.error("--max-per-class必须为正数")
    run_stage("crawl_external", lambda: crawl_external(args.class_id, args.max_per_class))
