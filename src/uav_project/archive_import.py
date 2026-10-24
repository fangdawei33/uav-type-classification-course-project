"""Import the author's browser-downloaded ZIP files into the frozen file listing.

No network requests and no extractall(): archive members must map to indexed
image/label paths. Existing original files are reused only when bytes agree.
"""
from __future__ import annotations

import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .common import (
    META, OUT, config, digest, now, protect_frozen, read_json, relative,
    resolve, save_json, setup, write_rows,
)
from .download import (
    INDEX, JOURNAL, SUMMARY, annotation_audit, check_listing,
    inspect_payload, journal_records, normalized, safe_source_path,
)

MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024 * 1024
IMPORT_STATUS = OUT / "drone_type_set_zip_import_summary.json"


def archive_member_path(name):
    raw = name.replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
        raise ValueError("压缩包包含不安全路径：" + name)
    return path


def scan_archives(archives, lookup):
    """Prepare an allowlisted import plan before writing dataset files."""
    plan, unindexed = [], []
    total = 0
    for archive in archives:
        with zipfile.ZipFile(archive) as package:
            for number, info in enumerate(package.infolist()):
                source = archive_member_path(info.filename)
                if info.is_dir() or not source.parts or "__MACOSX" in source.parts:
                    continue
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                    raise ValueError("不导入压缩包中的链接或特殊文件：" + info.filename)
                if info.flag_bits & 1:
                    raise ValueError("不支持加密的数据包：" + archive.name)
                if info.file_size > MAX_MEMBER_BYTES:
                    raise ValueError("单个图片/标注文件异常过大：" + info.filename)
                # Google Drive can add a root folder or split content across ZIPs.
                candidates = [source.as_posix().casefold()]
                if len(source.parts) >= 2:
                    candidates.append(PurePosixPath(*source.parts[-2:]).as_posix().casefold())
                if normalized(source.name) == "annotations.txt":
                    candidates.append("annotations.txt")
                matches = {lookup[key]["file_id"]: lookup[key] for key in candidates if key in lookup}
                if not matches:
                    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".txt"}:
                        if source.name.lower() not in {"readme.txt", "license.txt"}:
                            unindexed.append({"archive": archive.name, "member": info.filename,
                                              "reason": "not_in_frozen_file_listing"})
                    continue
                if len(matches) != 1:
                    raise ValueError("压缩包成员映射不唯一：" + info.filename)
                total += info.file_size
                if total > MAX_TOTAL_BYTES:
                    raise ValueError("导入总展开量超过20GiB，停止以检查数据包")
                plan.append({"archive": archive, "member_number": number,
                             "member_name": info.filename, "entry": next(iter(matches.values()))})
    write_rows(META / "drone_type_set_zip_unindexed.csv", unindexed, ["archive", "member", "reason"])
    if unindexed:
        raise RuntimeError("ZIP含当前清单之外的图片/标注，先核对是否下载了不同版本；详见zip_unindexed.csv，不静默忽略。")
    if not plan:
        raise RuntimeError("ZIP中没有匹配的原图/标注。不要放入日志回传包，请使用作者目录下载的原始ZIP。")
    return plan, total


def append_receipt(record):
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def import_member(package, info, entry, archive_name, archive_sha256):
    target = resolve(entry["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # Only our own temporary file is removed in finally; original files are not deleted.
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".import-", delete=False) as handle:
            temporary = Path(handle.name)
            copied = 0
            with package.open(info) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    copied += len(chunk)
                    if copied > MAX_MEMBER_BYTES:
                        raise ValueError("ZIP成员实际展开量超过限制")
                    handle.write(chunk)
        if copied != info.file_size:
            raise ValueError("ZIP成员长度不一致：" + info.filename)
        properties = inspect_payload(temporary, entry)
        receipt = {
            **entry, **properties, "status": "ok", "time": now(),
            "action": "imported_browser_zip", "archive_name": archive_name,
            "archive_sha256": archive_sha256, "archive_member": info.filename,
            "provenance_note": "Filename matched to source listing; SHA256 records local bytes, not an upstream signature.",
        }
        if target.exists():
            if digest(target) != properties["sha256"]:
                raise ValueError("ZIP与已有原图字节不同，保留原图并停止：" + entry["source_path"])
            receipt["action"] = "reused_identical"
            append_receipt(receipt)
        else:
            # User must stop the online downloader before starting this import.
            append_receipt(receipt)
            temporary.replace(target)
            temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def import_public_zips(archives, accept_listed_subset=False):
    protect_frozen()
    setup()
    spec = config()["drone_type_set"]
    if not INDEX.exists():
        raise RuntimeError("缺少原始目录清单，请先完成01_download_public.py --list-only")
    index = read_json(INDEX)
    if index.get("folder_url") != spec["folder_url"]:
        raise ValueError("目录清单来源不匹配")
    counts, issues = check_listing(index["files"], spec)
    structural = [r for r in issues if not r["reason"].startswith("count_mismatch_")]
    discrepancies = [r for r in issues if r["reason"].startswith("count_mismatch_")]
    if structural:
        raise ValueError("冻结清单有结构错误，不能导入")
    if discrepancies and not accept_listed_subset:
        raise ValueError("当前清单数量与论文不同；确认按当前子集使用时需传--accept-listed-subset")
    scope = "listed_subset_only" if discrepancies else "listing_matching_paper_counts"
    archives = sorted({Path(p).resolve() for p in archives})
    if not archives or any(not p.is_file() or p.suffix.lower() != ".zip" for p in archives):
        raise ValueError("没有找到完整的.zip文件；请先完成浏览器下载，不用解压")
    lookup = {}
    for entry in index["files"]:
        source = safe_source_path(entry["source_path"])
        expected = (Path("data/raw/drone_type_set") / str(source)).as_posix()
        if resolve(entry["path"]) != resolve(expected):
            raise ValueError("源清单目标路径不一致")
        lookup[source.as_posix().casefold()] = entry
    plan, total = scan_archives(archives, lookup)
    print("ZIP数量：", len(archives), "；匹配成员：", len(plan), f"；展开量约{total/1024**2:.1f} MiB")
    print("本步骤只读本地ZIP，不访问网络。请勿同时运行逐文件下载脚本。")
    receipts = journal_records()
    imported = []
    archive_records = []
    try:
        for archive in archives:
            fingerprint = digest(archive)
            archive_records.append({"name": archive.name, "bytes": archive.stat().st_size, "sha256": fingerprint})
            selection = [p for p in plan if p["archive"] == archive]
            with zipfile.ZipFile(archive) as package:
                members = package.infolist()
                for item in selection:
                    record = import_member(package, members[item["member_number"]], item["entry"], archive.name, fingerprint)
                    receipts[record["file_id"]] = record
                    imported.append(record)
                    if len(imported) % 100 == 0:
                        print(f"本地导入 {len(imported)}/{len(plan)}", flush=True)
        completed, missing = [], []
        for entry in index["files"]:
            target = resolve(entry["path"])
            record = receipts.get(entry["file_id"])
            if not target.is_file() or record is None:
                missing.append({"file_id": entry["file_id"], "source_path": entry["source_path"], "reason": "missing_file_or_receipt"})
                continue
            if record["path"] != entry["path"] or digest(target) != record["sha256"]:
                raise ValueError("导入后校验记录不符：" + entry["source_path"])
            completed.append(record)
        write_rows(META / "drone_type_set_download_log.csv", completed)
        write_rows(META / "drone_type_set_zip_missing.csv", missing, ["file_id", "source_path", "reason"])
        base = {
            "time": now(), "method": "browser_zip_import", "scope": scope,
            "index_sha256": digest(INDEX), "archives": archive_records,
            "completed_files": len(completed), "expected_files": len(index["files"]),
            "missing_files": len(missing), "local_bytes": sum(r["bytes"] for r in completed),
            "paper_reported_images": len(counts)*int(spec["expected_images_per_class"]),
            "matches_paper_counts": not discrepancies,
        }
        if missing:
            save_json(SUMMARY, {**base, "status": "partial_zip_import", "listed_files_download_complete": False})
            raise RuntimeError(f"本次已导入并复用{len(completed)}个文件，尚缺{len(missing)}个；补齐其余ZIP再运行，勿开始训练。")
        image_count, annotation_issues = annotation_audit(index, spec)
        result = {
            **base, "status": "downloaded_subset_review_pending" if discrepancies else "downloaded_review_pending",
            "image_count": image_count, "annotation_issues": annotation_issues,
            "listed_files_download_complete": True, "human_review_complete": False, "training_ready": False,
        }
        save_json(SUMMARY, result)
        save_json(IMPORT_STATUS, result)
        print("当前清单导入完成：", image_count, "张图片；标注待复核：", annotation_issues)
        print("摘要：", SUMMARY)
    except BaseException as exc:
        save_json(IMPORT_STATUS, {"status": "failed_or_partial", "time": now(), "imported_this_run": len(imported),
                                 "archives": archive_records, "error": repr(exc), "scope": scope})
        raise
