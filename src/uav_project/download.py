"""Download Drone-type-Set images and labels, separate from training."""
from __future__ import annotations

import importlib.metadata
import json
import math
import re
import time
from pathlib import PurePosixPath

from PIL import Image

from .common import (
    ROOT, META, OUT, config, digest, now, protect_frozen,
    read_json, relative, resolve, save_json, setup, write_rows,
)

RAW = ROOT / "data" / "raw" / "drone_type_set"
INDEX = META / "drone_type_set_file_index.json"
SUMMARY = OUT / "drone_type_set_download_summary.json"
JOURNAL = OUT / "logs" / "drone_type_set_download.jsonl"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
GDOWN_VERSION = "6.4.2"


class PublicDownloadUnavailable(RuntimeError):
    """The provider did not expose a downloadable URL; do not guess the cause."""


def normalized(value):
    return " ".join(value.strip().split()).casefold()


def load_downloader():
    try:
        version = importlib.metadata.version("gdown")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("请先安装requirements.txt中的gdown==6.4.2") from exc
    if version != GDOWN_VERSION:
        raise RuntimeError(f"当前gdown={version}，请安装本项目指定的{GDOWN_VERSION}")
    import gdown
    return gdown


def safe_source_path(value):
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(p in {"..", "."} for p in path.parts):
        raise ValueError("非法源路径：" + str(value))
    for part in path.parts:
        if any(c in part for c in '<>:"|?*') or part.endswith((" ", ".")):
            raise ValueError("不能安全保存的文件名：" + str(value))
        stem = part.split(".")[0].upper()
        if stem in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?:COM|LPT)[1-9]", stem):
            raise ValueError("Windows保留文件名：" + str(value))
    return path


def check_listing(entries, spec):
    """Check directory completeness; filenames alone never establish label quality."""
    classes = {normalized(name): name for name in spec["classes_in_expected_id_order"]}
    counts = {name: {"images": 0, "labels": 0} for name in classes.values()}
    pairs, issues = {}, []
    seen_paths, seen_ids = set(), set()
    root_labels = 0
    for item in entries:
        path = safe_source_path(item["source_path"])
        key = str(path).casefold()
        if key in seen_paths or item["file_id"] in seen_ids:
            issues.append({"path": str(path), "reason": "duplicate_file_identity_or_path"})
            continue
        seen_paths.add(key)
        seen_ids.add(item["file_id"])
        if len(path.parts) == 1 and normalized(path.name) == "annotations.txt":
            root_labels += 1
            continue
        if len(path.parts) != 2 or normalized(path.parts[0]) not in classes:
            issues.append({"path": str(path), "reason": "unexpected_directory_structure"})
            continue
        name = classes[normalized(path.parts[0])]
        suffix = path.suffix.lower()
        if suffix not in IMAGE_SUFFIXES | {".txt"}:
            issues.append({"path": str(path), "reason": "unexpected_file_type"})
            continue
        kind = "image" if suffix in IMAGE_SUFFIXES else "label"
        counts[name]["images" if kind == "image" else "labels"] += 1
        pair = pairs.setdefault((name, path.stem.casefold()), {})
        if kind in pair:
            issues.append({"path": str(path), "reason": "ambiguous_pair"})
        pair[kind] = item
    if root_labels != 1:
        issues.append({"path": "Annotations.txt", "reason": "missing_or_duplicate_class_list"})
    expected = int(spec["expected_images_per_class"])
    for name, values in counts.items():
        if values["images"] == 0 or values["labels"] == 0:
            issues.append({"path": name, "reason": "missing_class_images_or_labels"})
        if values != {"images": expected, "labels": expected}:
            issues.append({"path": name, "reason": f"count_mismatch_expected_{expected}_pairs"})
    for (name, stem), pair in pairs.items():
        if set(pair) != {"image", "label"}:
            issues.append({"path": name + "/" + stem, "reason": "missing_image_or_label"})
    return counts, issues


def discover(gdown, spec, refresh=False):
    if INDEX.exists() and not refresh:
        index = read_json(INDEX)
        if index.get("folder_url") != spec["folder_url"] or index.get("schema_version") != 1:
            raise RuntimeError("已有目录清单不属于当前来源，不自动覆盖")
    else:
        # Public directory listing only; no login/browser cookies are imported.
        files = gdown.download_folder(
            url=spec["folder_url"], output=str(RAW), skip_download=True,
            quiet=True, use_cookies=False, verify=True,
            user_agent=config()["user_agent"], timeout=(15, 90),
        )
        if not files:
            raise RuntimeError("公开目录没有返回文件，请回传错误；不绕过访问限制")
        entries = []
        for item in files:
            source = safe_source_path(item.path)
            target = (RAW / str(source)).resolve()
            if not target.is_relative_to(RAW.resolve()):
                raise ValueError("目标路径超出本数据集目录")
            file_id = str(item.id)
            entries.append({
                "file_id": file_id, "source_path": source.as_posix(),
                "path": relative(target),
                "source_url": f"https://drive.google.com/file/d/{file_id}/view",
            })
        index = {
            "schema_version": 1, "dataset": "Drone-type-Set",
            "folder_url": spec["folder_url"], "listed_at": now(),
            "gdown_version": GDOWN_VERSION, "files": entries,
            "total_bytes": None,
            "size_note": "Folder listing does not expose reliable byte sizes.",
        }
        if INDEX.exists():
            old = read_json(INDEX)
            old_pairs = {(r["file_id"], r["path"]) for r in old["files"]}
            new_pairs = {(r["file_id"], r["path"]) for r in entries}
            if old_pairs != new_pairs:
                # Before any payload download, refreshing an incomplete listing is safe.
                # Once download has started, never silently substitute its source snapshot.
                has_payloads = RAW.exists() and any(p.is_file() for p in RAW.rglob("*"))
                if JOURNAL.exists() or has_payloads:
                    save_json(META / "drone_type_set_changed_index.json", index)
                    raise RuntimeError("下载开始后目录有变化，候选清单已另存；不自动混合两个快照")
                history = META / "drone_type_set_index_history" / (digest(INDEX) + ".json")
                if not history.exists():
                    save_json(history, old)
                print("尚未下载原图：旧目录快照已保留，采用本次重新获取的清单。")
        save_json(INDEX, index)

    for item in index["files"]:
        source = safe_source_path(item["source_path"])
        if not re.fullmatch(r"[-A-Za-z0-9_]+", item["file_id"]):
            raise ValueError("非法文件ID")
        if resolve(item["path"]) != (RAW / str(source)).resolve():
            raise ValueError("清单目标路径不一致")
    counts, issues = check_listing(index["files"], spec)
    structural_issues = [r for r in issues if not r["reason"].startswith("count_mismatch_")]
    count_discrepancies = [r for r in issues if r["reason"].startswith("count_mismatch_")]
    write_rows(META / "drone_type_set_listing_issues.csv", [
        {**r, "severity": "count_discrepancy" if r["reason"].startswith("count_mismatch_") else "blocking_structure"}
        for r in issues
    ], ["path", "reason", "severity"])
    write_rows(META / "drone_type_set_counts.csv",
               [{"class_name": name, **v} for name, v in counts.items()])
    validation = {
        "status": "invalid_structure" if structural_issues else (
            "listed_with_paper_count_discrepancy" if count_discrepancies else "matches_paper_counts"
        ),
        "listed_at": index["listed_at"], "classes": counts,
        "listed_files": len(index["files"]), "total_bytes": None,
        "listed_images": sum(v["images"] for v in counts.values()),
        "paper_reported_images": len(counts) * int(spec["expected_images_per_class"]),
        "structural_issues": len(structural_issues),
        "count_discrepancies": len(count_discrepancies),
        "upstream_completeness": "not_independently_proven",
        "matches_paper_counts": not count_discrepancies,
    }
    save_json(OUT / "drone_type_set_listing_summary.json", validation)
    print("来源：", spec["folder_url"])
    for name, values in counts.items():
        print(f"{name}: 图片 {values['images']}，标注 {values['labels']}")
    print("文件总数：", len(index["files"]), "；字节总量：目录接口未提供，不估算。")
    print("目录统计：", META / "drone_type_set_counts.csv")
    if structural_issues:
        raise RuntimeError("目录存在配对缺失、重复或未知结构；请查看listing_issues.csv，暂不下载。")
    if count_discrepancies:
        print("注意：图片/标注配对正常，但数量与论文不一致。当前清单不等于论文完整数据。")
    index["listing_validation"] = validation
    return index


def journal_records():
    result = {}
    if JOURNAL.exists():
        for line in JOURNAL.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                if record.get("status") == "ok":
                    result[record["file_id"]] = record
            except json.JSONDecodeError:
                continue
    return result


def inspect_payload(path, entry):
    """Validate content, not semantic labels. SHA256 records local provenance."""
    if path.stat().st_size == 0:
        raise ValueError("空文件：" + str(path))
    result = {"bytes": path.stat().st_size, "width": None, "height": None}
    suffix = PurePosixPath(entry["source_path"]).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            result.update(width=image.width, height=image.height)
    else:
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("标注文本异常过大")
        text = path.read_text(encoding="utf-8-sig")
        if "<html" in text.lower() or "<!doctype" in text.lower():
            raise ValueError("下载内容是网页，不是标注")
    result["sha256"] = digest(path)
    return result


def fetch_one(gdown, entry, previous, spec):
    path = resolve(entry["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = previous.get(entry["file_id"])
    if path.exists():
        info = inspect_payload(path, entry)
        if prior is None or prior["path"] != entry["path"] or prior["sha256"] != info["sha256"]:
            raise ValueError("已有文件无匹配记录或已改变，不覆盖：" + str(path))
        return {**entry, **info, "status": "ok", "action": "reused", "time": now()}
    partial = path.with_name(path.name + ".download")
    time.sleep(float(spec["download_delay_seconds"]))
    from gdown.exceptions import FileURLRetrievalError
    try:
        # gdown retries only transient transport failures here, not permissions/quota errors.
        received = gdown.download(
            id=entry["file_id"], output=str(partial), quiet=True,
            resume=True, use_cookies=False, verify=True,
            retries=int(spec.get("network_retries", 2)),
            user_agent=config()["user_agent"], timeout=(15, 90),
        )
    except FileURLRetrievalError as exc:
        save_json(OUT / "drone_type_set_blocked_file.json", {
            "time": now(), "file_id": entry["file_id"],
            "source_path": entry["source_path"], "source_url": entry["source_url"],
            "error_type": type(exc).__name__, "error": str(exc),
            "cause": "undetermined_access_quota_or_page_parsing",
            "automatic_retry": False,
            "recovery": "Keep previous files and journal. Retry the same command later; if repeated, inspect the author's normal browser download.",
        })
        raise PublicDownloadUnavailable(
            "未取得下载链接：" + entry["source_path"] + "。"
            "原因可能是限流、权限或页面解析，暂未确定。"
            "已保留成功文件与断点；不要清空目录或重装环境。"
            "详情见outputs/drone_type_set_blocked_file.json。"
        ) from exc
    if received is None or not partial.is_file():
        raise RuntimeError("文件未完成：" + entry["source_path"])
    info = inspect_payload(partial, entry)
    record = {**entry, **info, "status": "ok", "action": "downloaded", "time": now()}
    # Write-ahead journal allows recovery if interrupted immediately before rename.
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
    partial.replace(path)
    return record


def annotation_audit(index, spec):
    """Audit syntax/class consistency; this is not human semantic approval."""
    entries = {r["source_path"].casefold(): r for r in index["files"]}
    root = next(r for r in index["files"] if normalized(r["source_path"]) == "annotations.txt")
    names = [s.strip() for s in resolve(root["path"]).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if list(map(normalized, names)) != list(map(normalized, spec["classes_in_expected_id_order"])):
        raise ValueError("Annotations.txt与类别表不一致，停止自动映射")
    mapping = {normalized(name): i for i, name in enumerate(names)}
    images, issues = [], []
    for _, item in sorted(entries.items()):
        source = PurePosixPath(item["source_path"])
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_entry = entries[str(source.with_suffix(".txt")).casefold()]
        lines = [s.strip() for s in resolve(label_entry["path"]).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        expected_id = mapping[normalized(source.parts[0])]
        image_issues = []
        if not lines:
            image_issues.append("empty_label")
        for line_number, line in enumerate(lines, 1):
            try:
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError("expected_5_fields")
                class_id = int(fields[0])
                cx, cy, width, height = map(float, fields[1:])
                if class_id != expected_id:
                    raise ValueError("class_id_folder_mismatch")
                if not all(math.isfinite(v) for v in (cx, cy, width, height)):
                    raise ValueError("nonfinite_coordinates")
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError("coordinates_out_of_range")
                if min(cx-width/2, cy-height/2) < -0.001 or max(cx+width/2, cy+height/2) > 1.001:
                    raise ValueError("box_exceeds_image")
            except ValueError as exc:
                image_issues.append(f"line_{line_number}:{exc}")
        for reason in image_issues:
            issues.append({"image": item["source_path"], "annotation": label_entry["source_path"], "reason": reason})
        images.append({
            "sample_id": "dts_" + item["file_id"], "path": item["path"],
            "annotation_path": label_entry["path"], "class_id": expected_id,
            "class_name": names[expected_id], "source_url": item["source_url"],
            "boxes": len(lines),
            "annotation_status": "needs_review" if image_issues else "syntax_checked",
            "group_id": "", "human_review": "pending", "source_dataset": "Drone-type-Set",
        })
    write_rows(META / "drone_type_set_images.csv", images)
    write_rows(META / "drone_type_set_annotation_issues.csv", issues, ["image", "annotation", "reason"])
    save_json(META / "drone_type_set_classes.json", {
        "origin": root["source_url"], "classes": names,
        "mapping_basis": "Annotations.txt and per-file audit; human identity review pending",
    })
    return len(images), len(issues)


def download_public(list_only=False, refresh_index=False, accept_listed_subset=False):
    protect_frozen()
    setup()
    spec = config()["drone_type_set"]
    gdown = load_downloader()
    index = discover(gdown, spec, refresh=refresh_index)
    if list_only:
        print("目录检查结束；未下载图片。请回传七类数量及数量差异，再确定下载范围。")
        return
    validation = index["listing_validation"]
    if validation["count_discrepancies"] and not accept_listed_subset:
        raise RuntimeError(
            f"当前可访问清单为{validation['listed_images']}张，论文报告{validation['paper_reported_images']}张。"
            "若明确按当前子集开展实验，使用--accept-listed-subset；不能将其称为完整论文数据集。"
        )
    scope = "listed_subset_only" if validation["count_discrepancies"] else "listing_matching_paper_counts"
    acquisition = {
        "time": now(), "folder_url": spec["folder_url"], "index_sha256": digest(INDEX),
        "scope": scope, "listed_images": validation["listed_images"],
        "paper_reported_images": validation["paper_reported_images"],
        "classes": validation["classes"], "subset_explicitly_accepted": bool(accept_listed_subset),
        "upstream_completeness": "not_independently_proven",
    }
    save_json(OUT / "drone_type_set_acquisition_scope.json", acquisition)
    if scope == "listed_subset_only":
        print("本次下载范围：当前可访问子集。论文报告数量及差异会保留，不视为完整复现。")
    previous = journal_records()
    completed = []
    current_entry = None
    try:
        ordered = sorted(index["files"], key=lambda r: (normalized(r["source_path"]) != "annotations.txt", r["source_path"]))
        for number, entry in enumerate(ordered, 1):
            current_entry = entry
            record = fetch_one(gdown, entry, previous, spec)
            previous[entry["file_id"]] = record
            completed.append(record)
            print(f"[{number}/{len(ordered)}] {record['action']}: {entry['source_path']}", flush=True)
            if number % 100 == 0:
                write_rows(META / "drone_type_set_download_log.csv", completed)
        write_rows(META / "drone_type_set_download_log.csv", completed)
        current_entry = None
        count, issues = annotation_audit(index, spec)
    except BaseException as exc:
        write_rows(META / "drone_type_set_download_log.csv", completed)
        save_json(SUMMARY, {
            "status": "blocked_source_access" if isinstance(exc, PublicDownloadUnavailable) else "interrupted_or_failed",
            "completed_this_run": len(completed),
            "failed_file": current_entry,
            "expected_files": len(index["files"]), "listed_files_download_complete": False,
            "scope": scope, "paper_reported_images": validation["paper_reported_images"],
            "instruction": "保留现有文件并回传错误。相同命令会复用已校验下载。",
        })
        raise
    save_json(SUMMARY, {
        "status": "downloaded_subset_review_pending" if scope == "listed_subset_only" else "downloaded_review_pending",
        "listed_files_download_complete": True, "scope": scope,
        "matches_paper_counts": validation["matches_paper_counts"],
        "paper_reported_images": validation["paper_reported_images"],
        "file_count": len(completed), "image_count": count,
        "local_bytes": sum(r["bytes"] for r in completed), "annotation_issues": issues,
        "source_listing_time": index["listed_at"], "finished_at": now(),
        "human_review_complete": False, "training_ready": False,
        "upstream_checksums": "not_provided; sha256 records local bytes only",
    })
    print("图片下载完成：", count, "张；标注待复核项：", issues)
    print("摘要：", SUMMARY)
    print("尚未审核或划分训练集；旧public_raw.csv未覆盖。请先回传摘要，不运行旧二分类脚本。")


