"""Decode actual auxiliary files, with safe extraction and explicit domain labels."""
import hashlib
import stat
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import PurePosixPath
from PIL import Image, ImageDraw, ImageOps
from .common import ROOT, OUT, META, digest, relative, now, rows, write_rows, save_json, protect_frozen
from .auxiliary_data import auxiliary_spec


def catalog_auxiliary(source_id):
    protect_frozen()
    spec = auxiliary_spec(source_id)
    archive = ROOT / "data/downloads/auxiliary" / source_id / spec["filename"]
    if archive.stat().st_size != spec["expected_bytes"] or digest(archive) != spec["expected_sha256"]:
        raise ValueError("Archive differs from pinned public metadata")
    destination = (ROOT / "data/raw/auxiliary" / source_id).resolve()
    output = OUT / "auxiliary" / source_id
    output.mkdir(parents=True, exist_ok=True)
    public_hashes = {r["sha256"] for r in rows(META / "drone_type_set_quality.csv")}
    records, skipped, failures = [], [], []
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        if len(members) > 100_000 or sum(m.file_size for m in members) > 6_000_000_000:
            raise ValueError("Unexpected archive size; extraction stopped")
        targets = set()
        # Preflight all members before writing, rejecting traversal and symlinks.
        for member in members:
            parts = PurePosixPath(member.filename).parts
            if (PurePosixPath(member.filename).is_absolute() or "\\" in member.filename
                    or any(p in {".", ".."} or ":" in p for p in parts)
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError("Unsafe archive member: " + member.filename)
            if not member.is_dir():
                if member.filename in targets:
                    raise ValueError("Repeated archive member name")
                targets.add(member.filename)
        for member in members:
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            if len(parts) != 4 or parts[1] not in {"train", "val"} or parts[2] not in {
                    "dji_inspire", "dji_mavic", "dji_phantom", "no_drone"} or not parts[-1].lower().endswith(".png"):
                skipped.append({"member": member.filename, "reason": "outside_expected_source_label_split_layout"})
                continue
            try:
                if member.file_size > 25_000_000:
                    raise ValueError("Individual image exceeds collection bound")
                payload = handle.read(member)  # ZIP CRC is checked during the read.
                sha = hashlib.sha256(payload).hexdigest()
                with Image.open(BytesIO(payload)) as image:
                    image.load()
                    if getattr(image, "n_frames", 1) != 1:
                        raise ValueError("Multiframe image")
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    width, height = rgb.size
                    pixel_sha = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                path = destination.joinpath(*parts[1:]).resolve()
                if not path.is_relative_to(destination):
                    raise ValueError("Extraction target escaped source directory")
                if path.exists():
                    if digest(path) != sha:
                        raise ValueError("Existing raw image differs; never overwrite")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as stream:
                        stream.write(payload)
                records.append({"sample_id": "aux_" + hashlib.sha256((source_id+"|"+member.filename).encode()).hexdigest()[:20],
                    "path": relative(path), "source_id": source_id, "source_member": member.filename,
                    "source_split": parts[1], "source_label": parts[2], "data_origin": spec["origin"],
                    "is_synthetic": True, "source_revision": spec["revision"], "archive_sha256": spec["expected_sha256"],
                    "sha256": sha, "pixel_sha256": pixel_sha, "width": width, "height": height,
                    "original_capture_group": "unknown", "render_parent_id": "unknown",
                    "seven_class_label": "", "seven_class_mapping": "not_established",
                    "intended_role": spec["intended_role"], "actual_training_used": False,
                    "visual_review_status": "not_individually_reviewed", "exact_public_file_match": sha in public_hashes})
                if len(records) % 1000 == 0:
                    print("Auxiliary decoded and cataloged:", len(records), flush=True)
            except Exception as exc:
                failures.append({"member": member.filename, "error": repr(exc)})
    by_pixel = defaultdict(list)
    for row in records:
        by_pixel[row["pixel_sha256"]].append(row)
    duplicates = []
    for key, group in by_pixel.items():
        if len(group) < 2:
            continue
        cross = len({r["source_split"] for r in group}) > 1
        conflicting_labels = len({r["source_label"] for r in group}) > 1
        for row in group:
            duplicates.append({"sample_id": row["sample_id"], "pixel_sha256": key,
                "source_split": row["source_split"], "source_label": row["source_label"],
                "group_images": len(group), "cross_source_split": cross, "conflicting_labels": conflicting_labels})
    duplicate_by_id = {r["sample_id"]: r for r in duplicates}
    for row in records:
        d = duplicate_by_id.get(row["sample_id"], {})
        row["pixel_duplicate_group_size"] = d.get("group_images", 1)
        row["cross_source_split_pixel_duplicate"] = d.get("cross_source_split", False)
        row["conflicting_duplicate_labels"] = d.get("conflicting_labels", False)
        row["source_split_eligible_candidate"] = not (d.get("cross_source_split", False) or d.get("conflicting_labels", False)
            or row["exact_public_file_match"])
    manifest = META / (source_id + "_images.csv")
    write_rows(manifest, records)
    write_rows(META / (source_id + "_pixel_duplicates.csv"), duplicates,
        ["sample_id","pixel_sha256","source_split","source_label","group_images","cross_source_split","conflicting_labels"])
    # This is a reproducible preview sample, not approval of every image.
    preview_members = []
    canvas = Image.new("RGB", (1280, 1200), "white")
    for ci, label in enumerate(("dji_inspire", "dji_mavic", "dji_phantom", "no_drone")):
        candidates = [r for r in records if r["source_label"] == label]
        chosen = []
        for split in ("train", "val"):
            group = sorted((r for r in candidates if r["source_split"] == split), key=lambda r:r["source_member"])
            chosen.extend([group[0], group[len(group)//2]] if len(group)>1 else group)
        for col, r in enumerate(chosen):
            with Image.open(ROOT/r["path"]) as im:
                thumb = ImageOps.contain(im.convert("RGB"), (310, 260))
            canvas.paste(thumb, (col*320+(320-thumb.width)//2, ci*300))
            ImageDraw.Draw(canvas).text((col*320+5, ci*300+265), label+" / "+r["source_split"], fill="black")
            ImageDraw.Draw(canvas).text((col*320+5, ci*300+282), r["sample_id"], fill="black")
            preview_members.append(r)
    preview = output / "sample_preview.jpg"
    canvas.save(preview, quality=95)
    write_rows(output / "sample_preview_manifest.csv", preview_members)
    status = {"recorded_at": now(), "source_id": source_id, "archive_sha256": spec["expected_sha256"],
        "status": "decoded_catalog_only_not_training_approval", "archive_file_records": len(targets),
        "decoded_images": len(records), "counts_by_split_label": dict(Counter(r["source_split"]+"/"+r["source_label"] for r in records)),
        "dimensions": dict(Counter(str((r["width"],r["height"])) for r in records)),
        "skipped": skipped, "decode_or_extraction_failures": failures,
        "unique_file_hashes": len({r["sha256"] for r in records}), "unique_pixel_hashes": len(by_pixel),
        "pixel_duplicate_groups": len({r["pixel_sha256"] for r in duplicates}),
        "cross_split_pixel_duplicate_groups": len({r["pixel_sha256"] for r in duplicates if r["cross_source_split"]}),
        "conflicting_label_pixel_duplicate_groups": len({r["pixel_sha256"] for r in duplicates if r["conflicting_labels"]}),
        "exact_public_file_matches": sum(r["exact_public_file_match"] for r in records),
        "manifest": relative(manifest), "manifest_sha256": digest(manifest),
        "preview": relative(preview), "preview_sample_count": len(preview_members), "preview_inspected_by_command": False,
        "all_images_visually_reviewed": False, "original_split_independence_verified": False,
        "seven_class_mapping_established": False, "training_run": False, "software_tests_run": False, "gpu_used": False}
    save_json(output / "catalog_current.json", status)
    save_json(output / ("catalog_"+status["recorded_at"].replace(":", "").replace("+", "_")+".json"), status)
    print("Auxiliary actual decoded images:", len(records), "failures:", len(failures), flush=True)
    if failures:
        raise ValueError("Auxiliary file findings saved; incomplete rows are not training candidates")
    return status


def prepare_auxiliary(source_id):
    """Use the verified catalog to prepare source-label pretraining candidates.

    This is separate from the formal seven-class cleaning gate. The source's val
    split remains auxiliary validation, never an independent real-world test.
    """
    from .common import read_json
    protect_frozen()
    spec = auxiliary_spec(source_id)
    folder = OUT / "auxiliary" / source_id
    catalog = read_json(folder / "catalog_current.json")
    manifest_path = META / (source_id + "_images.csv")
    if (catalog["archive_sha256"] != spec["expected_sha256"]
            or catalog["manifest_sha256"] != digest(manifest_path)
            or catalog["decode_or_extraction_failures"]):
        raise ValueError("Current auxiliary catalog is incomplete or changed")
    candidates = rows(manifest_path)
    accepted, rejected = [], []
    for row in candidates:
        if row["source_split"] not in {"train", "val"} or row["data_origin"] != spec["origin"]:
            raise ValueError("Unexpected source split/origin")
        if row["seven_class_label"]:
            raise ValueError("Auxiliary labels cannot populate the final seven-class field")
        reasons = []
        for flag, reason in (("cross_source_split_pixel_duplicate", "exact_pixel_duplicate_across_source_splits"),
                             ("conflicting_duplicate_labels", "conflicting_labels_for_identical_pixels"),
                             ("exact_public_file_match", "exact_file_overlap_with_primary_public_data")):
            if row[flag] == "True":
                reasons.append(reason)
        if reasons:
            rejected.append({**row, "exclusion_reason": "|".join(reasons)})
        else:
            accepted.append({**row, "auxiliary_partition": "auxiliary_" + row["source_split"],
                "experiment_scope": "source_four_class_pretraining_only",
                "counts_as_real_crawled_sample": False, "is_final_real_test": False})
    names = {}
    for split in ("train", "val"):
        records = [r for r in accepted if r["source_split"] == split]
        path = META / (source_id + "_auxiliary_" + split + ".csv")
        write_rows(path, records)
        names[split] = {"manifest": relative(path), "sha256": digest(path), "images": len(records),
            "classes": dict(Counter(r["source_label"] for r in records))}
    exclusion_path = META / (source_id + "_excluded.csv")
    write_rows(exclusion_path, rejected, list(candidates[0])+["exclusion_reason"])
    # A complete source-label table is kept separate from any main task label map.
    labels = {name: i for i, name in enumerate(sorted({r["source_label"] for r in accepted}))}
    save_json(META / (source_id + "_auxiliary_classes.json"), {"scope": "auxiliary_pretraining_only", "source_labels": labels,
        "seven_class_mapping": None, "negative_background_class": "no_drone"})
    summary = {"recorded_at": now(), "source_id": source_id,
        "status": "auxiliary_manifests_prepared_not_trained", "raw_decoded_images": len(candidates),
        "retained_auxiliary_candidates": len(accepted), "quarantined_images": len(rejected),
        "excluded_reasons": dict(Counter(r["exclusion_reason"] for r in rejected)), "splits": names,
        "excluded_manifest": relative(exclusion_path), "excluded_manifest_sha256": digest(exclusion_path),
        "same_split_near_duplicates_checked": False, "cross_split_near_duplicates_checked": False,
        "original_render_parent_identity_known": False, "source_split_independence_claimed": False,
        "source_dimensions_below_64px_candidates": sum(min(int(r["width"]),int(r["height"]))<64 for r in accepted),
        "primary_seven_class_dataset_changed": False, "training_run": False, "gpu_used": False}
    save_json(folder / "preparation_current.json", summary)
    save_json(folder / ("preparation_"+summary["recorded_at"].replace(":", "").replace("+", "_")+".json"), summary)
    export_auxiliary_report(summary, catalog)
    print("Auxiliary candidates:", len(accepted), "quarantined:", len(rejected), "source labels:", labels, flush=True)
    return summary



def export_auxiliary_report(preparation, catalog):
    """Export source-label counts and observed data issues, never model results."""
    from .common import read_json
    source_id = preparation["source_id"]
    folder = OUT / "auxiliary" / source_id
    observation_path = folder / "sample_visual_observation_2026-10-22.json"
    observation = read_json(observation_path) if observation_path.exists() else {}
    observation_current = bool(observation.get("all_16_preview_cells_actually_viewed")
        and observation.get("evidence_sha256") == digest(folder/"sample_preview.jpg")
        and observation.get("manifest_sha256") == digest(folder/"sample_preview_manifest.csv"))
    train, val = preparation["splits"]["train"], preparation["splits"]["val"]
    labels = sorted(train["classes"])
    table = "\n".join(f"| {label} | {catalog['counts_by_split_label'].get('train/'+label,0)} | "
        f"{catalog['counts_by_split_label'].get('val/'+label,0)} | {train['classes'].get(label,0)} | "
        f"{val['classes'].get(label,0)} |" for label in labels)
    text = f"""# Cranfield合成辅助数据：获取、核查与准备记录

生成时间：{preparation['recorded_at']}。本报告只记录已下载和处理的数据，不包含训练指标。

## 数据来源和用途

来源为Hugging Face `mazqtpopx/cranfield-synthetic-drone-classification`，固定revision `61f3f1858aa16eb6fbb763b0814d96dd58c0716a`。ZIP 855,922,191字节，SHA256为`{catalog['archive_sha256']}`，与公开LFS元数据一致。对应论文DOI为10.3390/jimaging8080218；原始数据DOI为10.17862/cranfield.rd.19423925。当前取得的是镜像包，没有把它当作第二个独立采集来源。

来源数据卡声明CC BY 4.0，已保存同版本README。这是第三方三维渲染合成域数据，不是本项目用大模型生成的图片，也不是老师要求的真实网页爬虫样本。

标签为三个DJI机型系列和No Drone。Phantom系列不能无证据映射为Phantom 3或Phantom 4 Pro，No Drone不属于既有给定区域七分类。全部辅助清单的seven_class_label保持空值，单独保留四类预训练标签。尚未使用它训练，主数据集与类别范围未改。

## 实际规模与隔离处理

已读取ZIP并完成安全路径检查、CRC读取、全量图像解码、文件及RGB像素哈希。{catalog['decoded_images']}张全部解码成功，解码/提取失败{len(catalog['decode_or_extraction_failures'])}项；全部原图保留。

| 来源类别 | 原始train | 原始val | 隔离后辅助train候选 | 隔离后辅助val候选 |
|---|---:|---:|---:|---:|
{table}
| 合计 | 4001 | 4000 | {train['images']} | {val['images']} |

发现一组3张No Drone图片RGB像素完全相同，分布为train 2张、val 1张。文件编码不同，单查文件SHA256会漏掉。三张全部从辅助候选清单隔离，记录在`data/metadata/{source_id}_excluded.csv`，原图和原始split不改。共保留{preparation['retained_auxiliary_candidates']}张辅助候选，不把8001张都写成训练图，也不把原val称为真实外部测试。

该包有{catalog['unique_file_hashes']}个文件哈希、{catalog['unique_pixel_hashes']}种不同RGB像素内容。与2973张主公开图像的文件哈希精确匹配为{catalog['exact_public_file_matches']}项；这不等于跨数据集近重复或来源独立核实。尚未完成裁剪变体、相近渲染父样本和背景复用检查。

## 尺寸与实际抽查

保留候选中最短边不足64像素的有{preparation['source_dimensions_below_64px_candidates']}张。本轮不因分辨率低就按模型效果删图，后续需要报告尺度分布与低清条件。放大不能恢复不存在的机体细节。

![图1 16张确定性抽查预览](../outputs/auxiliary/{source_id}/sample_preview.jpg)

图1：每类各选train两张、val两张，共16张。预览及清单哈希与实际查看记录匹配：{observation_current}。可见不同背景、视角和模糊程度；这是接触表抽查，不是8001张的逐图视觉审核，也不证明具体机型代际或原始渲染父样本独立。

## 可复现入口

1. `scripts/01d_download_auxiliary.py`：仅下载固定版本ZIP及数据卡、核对大小与SHA256。下载客户端版本固定在`requirements-auxiliary.txt`。
2. `scripts/03e_catalog_auxiliary.py`：解码、保留原图、生成原始清单和重复关系，不训练模型。
3. `scripts/04b_prepare_auxiliary.py`：按当前精确重复规则隔离3张，输出辅助train/val清单及本报告；不触碰主七类正式清洗准入。

本轮已执行上述数据步骤，不需要用户重复运行。关键输出：`{catalog['manifest']}`；`{train['manifest']}`；`{val['manifest']}`；`outputs/auxiliary/{source_id}/preparation_current.json`。

原始生成参数、父渲染ID及背景ID未知，记录unknown。辅助train/val保留来源划分，只能作为辅助开发划分，不能声称来源完全独立。已有数据的合成预训练与未来结构保持背景生成增强是两种不同实验，应分开比较和报告计算量。

本轮没有运行软件测试、模型训练、模型评估或图片生成，无需GPU；未生成PDF。
"""
    path = ROOT / "reports/auxiliary_data_report.md"
    path.write_text(text, encoding="utf-8")
    return path
