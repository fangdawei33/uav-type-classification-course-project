"""03. Audit imported images, generate review batches, or import explicit decisions."""
import argparse
from pathlib import Path
from _bootstrap import run_stage
from uav_project.review import make_review, apply_review, summarize_review
from uav_project.review_batches import make_batches
from uav_project.review_integrity import audit_review_integrity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--review-file", type=Path, help="导入人工审核CSV")
    mode.add_argument("--summary-only", action="store_true", help="只更新审核统计，不生成预览或改变决定")
    mode.add_argument("--audit-integrity", action="store_true", help="核对原图、标注、证据哈希及最新审核导入")
    mode.add_argument("--batches", action="store_true", help="生成可追溯的分批审核图")
    parser.add_argument("--class-id", type=int, choices=range(7))
    parser.add_argument("--flag", help="质量表中的完整标记，如not_single_target")
    parser.add_argument("--start", type=int, help="源文件编号起点（含）")
    parser.add_argument("--end", type=int, help="源文件编号终点（含）")
    args = parser.parse_args()
    if not args.batches and any(x is not None for x in (args.class_id, args.flag, args.start, args.end)):
        parser.error("筛选参数需配合--batches")
    if args.batches:
        action = lambda: make_batches(args.class_id, args.flag, args.start, args.end)
    elif args.audit_integrity:
        action = audit_review_integrity
    elif args.summary_only:
        action = summarize_review
    elif args.review_file:
        action = lambda: apply_review(args.review_file)
    else:
        action = make_review
    run_stage("review_data", action)
