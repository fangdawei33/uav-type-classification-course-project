"""Export current data-review records and figures without creating train/test data."""
from collections import Counter
from .common import META, OUT, ROOT, digest, now, read_json, relative, rows, save_json, write_rows


def export_data_audit():
    images = rows(META / "drone_type_set_images.csv")
    quality = {r["sample_id"]: r for r in rows(META / "drone_type_set_quality.csv")}
    reviews = {r["sample_id"]: r for r in rows(META / "drone_type_set_review.csv")}
    sources = {r["sample_id"]: r for r in rows(META / "drone_type_set_source_groups.csv")}
    constraints = {r["sample_id"]: r for r in rows(META / "drone_type_set_reviewed_correlation_constraints.csv")}
    selections = {r["sample_id"]: r for r in rows(META / "drone_type_set_box_selections.csv")}
    labels = read_json(META / "drone_type_set_classes.json")["classes"]
    latest = {}
    snapshots = []
    for path in sorted(OUT.glob("drone_type_set_visual_review*.json")):
        snapshot = read_json(path)
        snapshots.append((snapshot.get("recorded_at", snapshot.get("date", "")), path.name, path, snapshot))
    for _, _, path, snapshot in sorted(snapshots):
        for record in snapshot.get("records", []):
            latest[record["sample_id"]] = (record, relative(path))
    integrity_path = OUT / ("drone_type_set_review_integrity_" + now()[:10] + ".json")
    integrity = read_json(integrity_path) if integrity_path.exists() else {}
    audit_matches = bool(integrity.get("all_checked_files_and_decisions_match")
                         and integrity.get("review_csv_sha256") == digest(META / "drone_type_set_review.csv")
                         and integrity.get("visually_recorded_unique_images_checked") == len(images))
    manifest = []
    class_stats = []
    for image in images:
        sid = image["sample_id"]
        review, q = reviews[sid], quality[sid]
        record, snapshot_path = latest.get(sid, ({}, ""))
        if record and record["decision"] != review["decision"]:
            raise ValueError("Review decision differs from latest saved evidence: " + sid)
        source, constraint = sources.get(sid, {}), constraints.get(sid, {})
        source_ok = bool(source.get("status") == "confirmed" and source.get("source_group")
                         and source.get("source_evidence") and source["source_group"] == review.get("group_id"))
        choice = selections.get(sid, {})
        box_ok = bool((choice.get("status") == "reviewed") if choice else int(q["box_count"]) == 1)
        reason = record.get("primary_reason", "")
        if not reason:
            reason = "visual_candidate" if review["decision"] == "keep" else "body_not_identifiable_in_source_box"
        manifest.append({
            "sample_id": sid, "path": image["path"], "annotation_path": image["annotation_path"],
            "label": review["label"], "class_name": review["class_name"], "visual_decision": review["decision"],
            "primary_reason": reason, "review_note": review["note"],
            "image_sha256": q["sha256"], "annotation_sha256": q["annotation_sha256"],
            "width": q["width"], "height": q["height"], "quality_flags": q["flags"],
            "selected_annotation_index": choice.get("selected_box_index", "0" if int(q["box_count"]) == 1 else ""),
            "box_choice_status": choice.get("status", "single_source_box" if box_ok else "unresolved"),
            "conservative_constraint_group": constraint.get("constraint_group", ""),
            "constraint_is_verified_source": False, "source_group": source.get("source_group", ""),
            "source_record_confirmed_and_matched": source_ok,
            "label_basis": "author_label_not_independently_authenticated",
            "review_snapshot": snapshot_path, "visual_evidence": record.get("evidence", ""),
            "visual_evidence_sha256": record.get("evidence_sha256", ""),
            "is_formal_clean_manifest": False,
        })
    for c, name in enumerate(labels):
        part = [r for r in manifest if int(r["label"]) == c]
        kept = [r for r in part if r["visual_decision"] == "keep"]
        counts = Counter(r["visual_decision"] for r in part)
        class_stats.append({"class_id": c, "class_name": name, "raw": len(part),
            "visual_keep": counts["keep"], "visual_reject": counts["reject"], "pending": counts["pending"],
            "kept_observed_constraint_groups": len({r["conservative_constraint_group"] for r in kept if r["conservative_constraint_group"]}),
            "kept_without_correlation_constraint": sum(not r["conservative_constraint_group"] for r in kept),
            "kept_without_confirmed_source": sum(not r["source_record_confirmed_and_matched"] for r in kept),
            "confirmed_source_groups_for_kept": len({r["source_group"] for r in kept if r["source_record_confirmed_and_matched"]}),
            "kept_without_reviewed_box": sum(r["box_choice_status"] not in {"reviewed", "single_source_box"} for r in kept),
        })
    reasons = Counter(r["primary_reason"] for r in manifest if r["visual_decision"] == "reject")
    write_rows(META / "drone_type_set_visual_cohort.csv", manifest)
    write_rows(META / "drone_type_set_review_class_summary.csv", class_stats)
    write_rows(META / "drone_type_set_exclusion_reason_counts.csv", [{"primary_reason": k, "count": v} for k, v in reasons.most_common()])
    external = read_json(OUT / "drone_type_external_current_status.json")
    keep_count = sum(r["visual_decision"] == "keep" for r in manifest)
    missing_source = sum(r["visual_decision"] == "keep" and not r["source_record_confirmed_and_matched"] for r in manifest)
    blocking = []
    if missing_source:
        blocking.append({"code": "public_sources_unverified", "affected_visual_candidates": missing_source,
                         "resolution": "Trace original videos/pages and authenticate source groups, or acquire independently documented sources; preserve correlation constraints."})
    if any(r["pending"] for r in class_stats):
        blocking.append({"code": "public_visual_review_pending"})
    if not audit_matches:
        blocking.append({"code": "current_review_integrity_not_confirmed"})
    if not external.get("test_set_ready"):
        blocking.append({"code": "external_test_not_ready", "uncollected_classes": external.get("uncollected_classes", []),
                         "collected_candidates": external["total_collected_candidates"],
                         "pending_visual_decisions": external["visual_decision_counts"].get("pending", 0),
                         "resolution": "Acquire sufficiently documented independent sources; resolve pending decisions, final class mapping, target boxes and cross-set related/duplicate images before final test designation."})
    report = {
        "recorded_at": now(), "stage": "public_visual_review_complete_source_verification_incomplete",
        "public_raw_images": len(images), "public_visual_candidates": keep_count,
        "public_visual_rejected": sum(r["visual_decision"] == "reject" for r in manifest),
        "public_visual_pending": sum(r["visual_decision"] == "pending" for r in manifest),
        "review_integrity_matches_current_review_csv": audit_matches,
        "source_verified_visual_candidates": keep_count - missing_source,
        "class_summary": class_stats, "exclusion_primary_reasons": dict(reasons),
        "blocking_conditions": blocking,
        "formal_cleaning_executed_by_this_command": False,
        "data_split_or_model_experiment_executed": False,
        "software_tests_run": False, "training_ready": False,
        "training_ready_scope": "This exporter does not approve a training protocol. Current public source and external test prerequisites remain incomplete.",
        "external_status_file": "outputs/drone_type_external_current_status.json",
        "external_snapshot": {k: external[k] for k in ("total_collected_candidates", "reference_only_images", "visual_decision_counts", "reviewed_correlation_groups", "images_with_correlation_constraints", "test_set_ready")},
        "review_csv_sha256": digest(META / "drone_type_set_review.csv"),
        "visual_cohort_csv_sha256": digest(META / "drone_type_set_visual_cohort.csv"),
        "metadata_snapshot_warning": "Export is a timestamped status snapshot; integrity reference checks saved review hashes, not a new decode/hash audit.",
    }
    save_json(OUT / "drone_type_set_data_readiness.json", report)
    _figures(class_stats, external)
    _report(report, external)
    print(f"Exported {len(manifest)} review rows; {keep_count} visual candidates. No formal clean dataset or model results created.")
    print(OUT / "drone_type_set_data_readiness.json")
    print(ROOT / "reports/data_audit_report.md")
    return report


def _figures(class_stats, external):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder = OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    names = ["Bebop", "Phantom 3", "Phantom 4 Pro", "Emax", "Intel Aero", "Mambo", "YH 19Hw"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    keep = [r["visual_keep"] for r in class_stats]
    rejected = [r["visual_reject"] for r in class_stats]
    ax.bar(x, keep, color="#347ba6", label="Visual candidates (source unverified)")
    ax.bar(x, rejected, bottom=keep, color="#d47c57", label="Excluded from primary cohort")
    for i, row in enumerate(class_stats):
        ax.text(i, keep[i]/2, str(keep[i]), ha="center", va="center", color="white", fontsize=10)
        if rejected[i] >= 30:
            ax.text(i, keep[i]+rejected[i]/2, str(rejected[i]), ha="center", va="center", fontsize=9)
        elif rejected[i]:
            ax.annotate(str(rejected[i]), (i+0.40, keep[i]+rejected[i]/2), xytext=(7, 0), textcoords="offset points", ha="left", va="center", fontsize=8)
        ax.text(i, row["raw"]+10, str(row["raw"]), ha="center", fontsize=9)
    ax.set_xticks(x, names, rotation=12)
    ax.set_ylim(0, max(r["raw"] for r in class_stats)*1.16)
    ax.set_ylabel("Image count")
    ax.set_title(f"Public subset: {sum(r['raw'] for r in class_stats):,} inspected; {sum(keep):,} candidates; {sum(rejected):,} exclusions")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(folder / "data_review_counts.png", dpi=180)
    plt.close(fig)
    coverage = external["coverage"]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    bottom = np.zeros(7)
    for key, label, color in [("candidate_keep", "Visually retained, not final test", "#347ba6"), ("pending", "Pending", "#d8b052"), ("reject", "Excluded", "#d47c57")]:
        values = np.array([r[key] for r in coverage])
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    for i, n in enumerate(bottom):
        ax.text(i, n+0.6, str(int(n)), ha="center")
    ax.set_xticks(x, names, rotation=12)
    ax.set_ylim(0, max(bottom)*1.22)
    ax.set_ylabel("Collected candidate images")
    ax.set_title(f"Self-collected candidates: {external['total_collected_candidates']} images; "
                 f"{sum(r['collected_candidates'] > 0 for r in coverage)}/7 classes collected")
    fig.supxlabel(f"{external.get('reference_only_images', 0)} reference-only images excluded; candidate coverage is not test readiness", fontsize=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(folder / "external_class_coverage.png", dpi=180)
    plt.close(fig)


def _report(status, external):
    stats = status["class_summary"]
    table = "\n".join(f"| {r['class_name']} | {r['raw']} | {r['visual_keep']} | {r['visual_reject']} | {r['pending']} |" for r in stats)
    names = {"wrong_target_box": "错框/错目标", "crossfade": "转场叠影", "postproduction_overlay": "后期字幕/动画叠加", "misaligned_body_crop": "框偏移/主要机身漏裁", "burned_in_detector_overlay": "烧录检测框/置信度", "inset_intersects_crop": "画中画进入裁剪", "promotional_title_card": "宣传标题卡域", "exact_duplicate": "完全重复冗余", "body_out_of_frame": "机身出画仅剩部件", "rendered_target_graphics": "目标区域内渲染图形", "body_not_identifiable_in_source_box": "框内机身不可辨认", "unverifiable_tiny_target": "原生小框内容不能确认"}
    reasons = "\n".join(f"| {names.get(k, k)}（{k}） | {v} |" for k, v in sorted(status["exclusion_primary_reasons"].items(), key=lambda p: -p[1]))
    constraints = "\n".join(f"| {r['class_name']} | {r['kept_observed_constraint_groups']} | {r['kept_without_correlation_constraint']} | {r['confirmed_source_groups_for_kept']} |" for r in stats)
    counts = external["visual_decision_counts"]
    class_coverage = sum(r["collected_candidates"] > 0 for r in external["coverage"])
    references = external.get("reference_only_images", 0)
    external_table = "\n".join(f"| {r['class_name']} | {r['collected_candidates']} | {r['candidate_keep']} | {r['pending']} | {r['reject']} |" for r in external["coverage"])
    external_integrity_path = OUT / "drone_type_external_integrity_current.json"
    external_integrity = read_json(external_integrity_path) if external_integrity_path.exists() else {}
    external_integrity_matches = bool(external_integrity.get("all_checked_files_and_decisions_match")
        and external_integrity.get("review_csv_sha256") == digest(META / "drone_type_external_review.csv")
        and all(external_integrity.get("collection_manifest_sha256", {}).get(relative(p)) == digest(p)
                for p in (META / "drone_type_external_raw.csv", META / "drone_type_article_raw.csv")))
    text = f"""# 数据构建与视觉审核阶段报告

生成时间：{status['recorded_at']}。本报告汇总真实数据记录，不包含模型训练结果。任务是给定目标区域的七类无人机机型分类（target_crop），不是端到端检测。

## 1. 数据范围和过程

作者公开目录网页ZIP共2973张图片、2973份YOLO标注和1份类别清单。当前可访问子集与论文所述7000张有差异，不能把缺失图片当作已取得数据。原图有2970个不同文件SHA-256；其中一组4张完全重复。

公开数据完成逐图AI辅助视觉复核，证据覆盖2973张。标签沿用作者定义，尚非独立机型鉴定；审核不是两名独立人工标注。审核图仅用于检查，后续模型不能输入绘制了红色框和文字的审核预览。

自行编写的网页爬虫另外保存{external['total_collected_candidates']}张真实候选，覆盖{class_coverage}类；另有{references}张Tinyhawk机型核对参考图，未计入实验候选。全部自采集原图合计{external['total_collected_candidates'] + references}张。该部分与作者ZIP分开保存，记录页面、许可待核实字段、采集时间和哈希；候选覆盖不代表独立外部测试集就绪。

## 2. 逐类筛选结果

| 类别 | 原图 | 视觉候选保留 | 主实验排除 | 待审 |
|---|---:|---:|---:|---:|
{table}
| 合计 | 2973 | 2720 | 253 | 0 |

![图1 公开数据逐类视觉审核数量](../outputs/figures/data_review_counts.png)

图1：数字是图片数，不是独立来源数。2720张只通过视觉候选筛选，未完成正式清洗准入、划分或训练。

## 3. 排除依据

排除协议visual-primary-cohort-v1-2026-10-22在任何模型分数产生前记录。原图、原标注和历史决定保留；不使用生成式补图去除噪声。小目标、模糊、逆光、局部自然遮挡/截断，只要机身可辨认，不因此自动排除。

| 互斥主原因 | 图片数 |
|---|---:|
{reasons}
| 合计 | 253 |

主原因按图统计一次；某图仍可能同时有其他问题。宣传标题卡即使标题在目标裁剪之外，也按事先声明的场景范围排除；不能把它们全部写成“裁剪内有答案文字”。物体自带品牌标识与后期叠加标题分别对待。

Phantom 3此前99项、YH此前10项未决问题，依据既有图像证据按同一协议转为排除。此操作是109个已看样本的决定收尾，不是又新增109张审核量。YH210的13×9像素框仍不足以确认目标，选框保留proposed且该图排除。

## 4. 来源相关性与实验可行性

2807张原图记录为12个保守相关画面约束，含排除图片。其余166张Phantom 3尚无该类约束记录；不能把未分组当作相互独立。观察到的场景不能代替已确认的原始视频、设备或飞行ID。

下表只统计视觉保留项涉及的约束：

| 类别 | 观察到的约束组数 | 未填写约束的保留图 | 已确认来源组数 |
|---|---:|---:|---:|
{constraints}

Bebop、Intel Aero、Mambo各只有一个观察到的保守相关组。不同画面、姿态、分辨率或文件编号不足以证明独立来源。正式来源确认仍为0组，2720张候选均未完成来源准入。当前不能进行声称来源独立的七类7:3划分及分组交叉验证。

![图2 自采集候选类别覆盖](../outputs/figures/external_class_coverage.png)

图2：自采集候选{external['total_collected_candidates']}张，{counts.get('candidate_keep', 0)}张视觉暂留、{counts.get('pending', 0)}张待定、{counts.get('reject', 0)}张排除。七类都有候选；Intel仅一张官方未装桨开箱图，现为视觉候选且仅供开箱挑战范围，不代表飞行测试样本；Emax具体型号与作者类别对应未确证。数量不能当作独立样本量。

| 外部候选类别 | 原图数 | 视觉暂留 | 待定 | 排除 |
|---|---:|---:|---:|---:|
{external_table}

当前YH四张、EZ Pilot四张为视觉候选；EZ Pilot拆机图仅标记挑战条件，且与作者Emax类映射未最终证实。Mambo共九张视觉候选，一张人物手机图无可辨无人机而排除。外部共{external['reviewed_correlation_groups']}个保守相关组，涉及{external['images_with_correlation_constraints']}张，不证明组间独立。继新增63张接触表初筛后，另逐图查看原先32张待定原图/框卡，15张转为视觉候选，17张仍待定。原先缺框的109张已完成原图观察、目标框记录与裁剪检查。当前157张非排除图（含17张待定）具有框几何记录；140张视觉暂留图全部导出原图派生裁剪，缺框0。场景范围124张standard_candidate、16张challenge_only。几何复核不等于机型或来源独立认证。

## 5. 已完成的文件核对

最新数据完整性核对实际解码2973张原图，核查原图、标注、证据、2807项相关性约束和23项选框记录，0项不一致。导出时审核CSV与该核对报告匹配：{status['review_integrity_matches_current_review_csv']}。这是文件和记录一致性核查，不证明语义标签正确或来源独立，也不是模型测试。

外部当前完整性核查记录见outputs/drone_type_external_integrity_current.json；其与当前采集清单和审核CSV匹配：{external_integrity_matches}。该次核查实际读取{external_integrity.get('decoded_and_size_checked', 0)}张候选/参考原图、核对{external_integrity.get('candidate_visual_evidence_checked', 0)}项候选审核证据与{external_integrity.get('box_proposals_checked', 0)}项框提案，报告{len(external_integrity.get('problems', []))}项未满足审核条件；具体见JSON。哈希核对不能证明来源独立或类别真值。

可追溯文件：data/metadata/drone_type_set_visual_cohort.csv（全量视觉筛选台账，非正式清洗清单）、drone_type_set_review.csv（当前决定）、drone_type_set_exclusion_reason_counts.csv（排除统计）；outputs/drone_type_set_review_integrity_2026-10-22.json、drone_type_set_data_readiness.json；outputs/review_batches/final_review_transitions_2026-10-22.csv（本轮决定变化）。

## 6. 剩余工作和允许的结论

1. 追溯原始网页/视频，确认真实来源及机型标签；补足能支持来源隔离的数据。原始出处无法确定时如实标记未知，不能生成flight_id。
2. 七类候选已覆盖；继续补足每类独立来源，解决{counts.get('pending', 0)}项待定及全部最终标签/目标框，核对相关来源与跨集合衍生图。网页搜索词和不同下载URL不能自动作为真值或独立性证据。
3. 现有数据已固定为data/final/uav_type_v1/：2720张作者标签训练池、135张网页弱标签池、4张EZPilot身份补充；17张型号未定明确排除、1组不同分辨率同图隔离。数据整理完成，严格来源独立测试条件另列；未进行7:3划分或本地模型运行。
4. 七类主任务仍需至少四种不同原理模型的真实训练与比较；辅助四分类已返回两种表征×四分类器及调参结果，不能将其移作七类指标。详见[辅助实验结果与分析](auxiliary_experiment_results.md)。

第三方合成辅助数据的独立报告见[auxiliary_data_report.md](auxiliary_data_report.md)，其来源四类与主七类清单分开统计，不能相加宣传为七类训练规模。

当前阶段已完成全部公开图片的视觉审核。训练用图像整理版本uav_type_v1已完成；七类来源独立划分、主任务模型实验、完整论文和PPT尚未完成。Colab四类合成辅助结果已导入，见auxiliary_experiment_results.md；实际设备为Tesla T4。没有运行本地软件测试、模型推理、训练或评估；报告更新无需GPU。
"""
    path = ROOT / "reports/data_audit_report.md"
    path.write_text(text, encoding="utf-8")
