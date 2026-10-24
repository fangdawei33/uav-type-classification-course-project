"""01. List or download Drone-type-Set images and paired annotations."""
import argparse
from _bootstrap import run_stage
from uav_project.download import PublicDownloadUnavailable, download_public

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-only", action="store_true", help="只统计七类文件，不下载图片")
    parser.add_argument("--refresh-index", action="store_true", help="重新读取目录；变更列表另存，不自动混用")
    parser.add_argument("--accept-listed-subset", action="store_true", help="明确接受当前可访问子集，保留与论文的数量差异")
    args = parser.parse_args()
    try:
        run_stage("download_public", lambda: download_public(args.list_only, args.refresh_index, args.accept_listed_subset))
    except PublicDownloadUnavailable as exc:
        # Full diagnostic trace is retained by run_stage; keep the terminal actionable.
        print(str(exc), flush=True)
        raise SystemExit(2)
