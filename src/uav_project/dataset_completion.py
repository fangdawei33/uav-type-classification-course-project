"""Finalize acquired data, without changing v1 or inventing independent splits.

This is an offline data operation: copy selected crops, retain all acquisition
records, check file identities and explain the limits of feasible experiments.
No classifiers, learned features, model evaluation or software tests are run.
"""
from collections import Counter, defaultdict
from pathlib import Path
import shutil
from PIL import Image
from .common import (ROOT, META, OUT, rows, write_rows, read_json, save_json,
                     digest, relative, resolve, now)

VERSION = 'uav_type_v2'
MANIFESTS = ['public_train_pool.csv', 'external_weak_labels.csv', 'supplementary_identity.csv']


def verify_hashes(mapping):
    for name, expected in mapping.items():
        if digest(resolve(name)) != expected:
            raise ValueError('File changed or missing: ' + name)


def group_feasibility(public):
    """Describe existing constraints; an unknown group is never a new flight."""
    result = []
    for label in range(7):
        selected = [r for r in public if int(r['label']) == label]
        grouped = Counter(r['constraint_group'] for r in selected if r['constraint_group'])
        missing = sum(not r['constraint_group'] for r in selected)
        result.append({'label': label, 'class_name': selected[0]['class_name'],
            'images': len(selected), 'observed_constraint_groups': len(grouped),
            'unresolved_group_images': missing,
            'largest_constraint_group': max(grouped.values(), default=0),
            'single_observed_group_contains_whole_class': len(grouped) == 1 and missing == 0,
            'authenticated_independent_source_groups': 0,
            'source_independent_holdout_and_3fold_ready': False})
    return result


def similarity_components(public):
    """Summarize pre-existing near-similarity leads, without re-labelling them."""
    by_id = {r['sample_id']: r for r in public}
    parent = {sid: sid for sid in by_id}
    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid
    links = 0
    for r in rows(META / 'drone_type_set_near_duplicate_candidates.csv'):
        a, b = r['sample_a'], r['sample_b']
        if a in by_id and b in by_id:
            parent[find(a)] = find(b)
            links += 1
    groups = defaultdict(list)
    for sid, r in by_id.items():
        groups[find(sid)].append(r)
    for members in groups.values():
        if len({r['label'] for r in members}) != 1:
            raise ValueError('Similarity component has conflicting labels')
    summary = []
    for label in range(7):
        sizes = [len(v) for v in groups.values() if int(v[0]['label']) == label]
        summary.append({'label': label, 'components': len(sizes),
                        'largest_component': max(sizes), 'images': sum(sizes)})
    return {'existing_candidate_links_with_both_retained_endpoints': links,
            'by_class': summary,
            'meaning': 'Similarity leads, not proven duplicate identities or flights. Transitive components are conservative only.'}


def build_raw_inventory(public, external, supplement):
    """One row per original; YOLO files stay linked to the public original row."""
    quality = {r['sample_id']: r for r in rows(META / 'drone_type_set_quality.csv')}
    kept = {r['sample_id'] for r in public}
    external_kept = {r['sample_id'] for r in external}
    supplementary = {r['sample_id'] for r in supplement}
    inventory = []
    for r in rows(META / 'drone_type_set_images.csv'):
        q = quality[r['sample_id']]
        inventory.append({'sample_id': r['sample_id'], 'dataset': 'Drone-type-Set author subset',
            'data_origin': 'author_public_dataset', 'path': r['path'], 'sha256': q['sha256'],
            'source_url': r['source_url'], 'source_label': r['class_id'], 'class_name': r['class_name'],
            'annotation_path': r['annotation_path'], 'annotation_sha256': q['annotation_sha256'],
            'release_action': 'public_training_pool' if r['sample_id'] in kept else 'excluded_public',
            'is_self_crawled': False, 'is_synthetic': False})
    reviews = {r['sample_id']: r for r in rows(META / 'drone_type_external_review.csv')}
    for name in ('drone_type_external_raw.csv', 'drone_type_article_raw.csv'):
        for r in rows(META / name):
            sid = r['sample_id']
            action = ('external_weak_label_pool' if sid in external_kept else
                      'supplementary_identity' if sid in supplementary else
                      'reference_only' if r.get('sample_scope') == 'reference_only' else 'excluded_external')
            inventory.append({'sample_id': sid, 'dataset': 'self-written public webpage collectors',
                'data_origin': 'self_crawled_public_webpage', 'path': r['path'], 'sha256': r['sha256'],
                'source_url': r['source_url'], 'source_snapshot': r['detail_snapshot'],
                'source_snapshot_sha256': r['detail_sha256'], 'download_url': r['download_url'],
                'source_label': r['candidate_label'], 'class_name': r['candidate_class_name'],
                'annotation_path': '', 'annotation_sha256': '', 'release_action': action,
                'visual_decision': reviews.get(sid, {}).get('visual_decision', 'reference_only'),
                'is_self_crawled': True, 'is_synthetic': False,
                'license': r.get('license', ''), 'license_review': r.get('license_review', '')})
    usable = {r['sample_id']: r['auxiliary_partition'] for name in
              ('cranfield_synthetic_classification_auxiliary_train.csv',
               'cranfield_synthetic_classification_auxiliary_val.csv') for r in rows(META / name)}
    for r in rows(META / 'cranfield_synthetic_classification_images.csv'):
        inventory.append({'sample_id': r['sample_id'], 'dataset': 'Cranfield third-party synthetic four-class',
            'data_origin': 'third_party_3d_rendered_synthetic', 'path': r['path'], 'sha256': r['sha256'],
            'source_revision': r['source_revision'], 'archive_sha256': r['archive_sha256'],
            'source_member': r['source_member'], 'source_label': r['source_label'],
            'class_name': r['source_label'], 'annotation_path': '', 'annotation_sha256': '',
            'release_action': usable.get(r['sample_id'], 'excluded_synthetic_cross_split_duplicate'),
            'is_self_crawled': False, 'is_synthetic': True})
    if len(inventory) != len({r['sample_id'] for r in inventory}):
        raise ValueError('Nonunique original sample ID')
    return inventory


def build_dataset_completion():
    destination = ROOT / 'data/final' / VERSION
    if (destination / 'freeze.json').exists():
        raise FileExistsError('v2 already frozen; use --audit-only, or create a new dataset version')
    previous = ROOT / 'data/final/uav_type_v1'
    old_freeze = read_json(previous / 'freeze.json')
    verify_hashes(old_freeze['file_hashes'])
    verify_hashes(old_freeze['image_hashes'])
    integrity = read_json(OUT / 'drone_type_external_integrity_current.json')
    if not integrity['all_checked_files_and_decisions_match']:
        raise ValueError('Resolve acquired-file/evidence problems first')
    for name, sha in integrity['collection_manifest_sha256'].items():
        if digest(resolve(name)) != sha:
            raise ValueError('Collection changed after actual file audit')
    if digest(META / 'drone_type_external_review.csv') != integrity['review_csv_sha256']:
        raise ValueError('Review changed after actual file audit')
    destination.mkdir(parents=True, exist_ok=True)
    all_images = {}
    pools = {}
    for name in MANIFESTS:
        records = rows(previous / name)
        for r in records:
            source = resolve(r['image_path'])
            local = source.relative_to(previous)
            target = destination / local
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != r['sha256']:
                raise ValueError('Do not overwrite changed destination')
            if not target.exists():
                shutil.copyfile(source, target)
            r['image_path'] = relative(target)
            all_images[r['image_path']] = r['sha256']
        write_rows(destination / name, records)
        pools[name] = records
    # The v1 physical folder also contains one isolated resized derivative.
    # Copying only listed files prevents it from re-entering the v2 image tree.
    exclusions = rows(previous / 'excluded_and_reference.csv')
    known = {r['sample_id'] for r in exclusions}
    reviews = {r['sample_id']: r for r in rows(META / 'drone_type_external_review.csv')}
    public = pools[MANIFESTS[0]]
    external = pools[MANIFESTS[1]]
    supplement = pools[MANIFESTS[2]]
    selected = {r['sample_id'] for r in external + supplement}
    for name in ('drone_type_external_raw.csv', 'drone_type_article_raw.csv'):
        for r in rows(META / name):
            sid = r['sample_id']
            if sid in known or sid in selected:
                continue
            decision = reviews.get(sid, {})
            exclusions.append({'sample_id': sid, 'origin': 'external',
                'candidate_label': r['candidate_label'], 'class_name': r['candidate_class_name'],
                'path': r['path'], 'reason': decision.get('note', 'reference_only'),
                'release_action': 'excluded_from_v2'})
            known.add(sid)
    write_rows(destination / 'excluded_and_reference.csv', exclusions)
    for name in ('duplicate_relations.csv', 'class_map.json'):
        shutil.copyfile(previous / name, destination / name)
    inventory = build_raw_inventory(public, external, supplement)
    write_rows(destination / 'raw_acquisition_inventory.csv', inventory)
    annotations = {r['annotation_path']: r['annotation_sha256'] for r in inventory if r.get('annotation_path')}
    original_hashes = {r['path']: r['sha256'] for r in inventory}
    source_hashes = {r['source_snapshot']: r['source_snapshot_sha256'] for r in inventory if r.get('source_snapshot')}
    # This reads actual acquired files, not a test suite or a model evaluation.
    for i, (name, sha) in enumerate(original_hashes.items(), 1):
        if digest(resolve(name)) != sha:
            raise ValueError('Original changed or missing: ' + name)
        if i % 2000 == 0:
            print('Verified original files:', i, flush=True)
    verify_hashes(annotations)
    verify_hashes(source_hashes)
    for name, sha in all_images.items():
        if digest(resolve(name)) != sha:
            raise ValueError('Copied crop hash mismatch')
        with Image.open(resolve(name)) as im:
            im.load()
            if im.mode != 'RGB' or min(im.size) < 1:
                raise ValueError('Invalid selected crop: ' + name)
    metadata = destination / 'metadata'
    metadata.mkdir(exist_ok=True)
    for p in sorted((previous / 'metadata').glob('*')):
        # Current acquired-source decisions are copied below, preserving the old
        # input snapshots inside v1 itself.
        source = META / p.name
        shutil.copyfile(source if source.exists() else p, metadata / p.name)
    for name in ('drone_type_article_media_inventory.csv', 'drone_type_article_pages.csv',
                 'drone_type_external_quality.csv', 'drone_type_set_near_duplicate_candidates.csv',
                 'cranfield_synthetic_classification_excluded.csv'):
        shutil.copyfile(META / name, metadata / name)
    evidence = destination / 'evidence'
    evidence.mkdir(exist_ok=True)
    evidence_inputs = [OUT / 'drone_type_external_integrity_current.json',
                       OUT / 'drone_type_external_visual_review_completion_v2.json',
                       OUT / 'drone_type_article_collection_summary.json',
                       OUT / 'review_batches/rare_completion_v2/manifest.csv',
                       OUT / 'review_batches/rare_completion_v2/originals_01.jpg',
                       OUT / 'review_batches/rare_completion_v2/originals_02.jpg',
                       OUT / 'review_batches/rare_completion_v2/emax_identity_comparison.jpg',
                       ROOT / 'configs/article_sources.json', ROOT / 'configs/source_discovery.json']
    for p in evidence_inputs:
        shutil.copyfile(p, evidence / p.name)
    feasibility = group_feasibility(public)
    write_rows(destination / 'split_feasibility.csv', feasibility)
    similarity = similarity_components(public)
    save_json(destination / 'similarity_components.json', similarity)
    acquisition_counts = Counter(r['data_origin'] for r in inventory)
    summary = {'version': VERSION, 'created_at': now(),
        'acquisition_round_closed': True, 'data_assembly_complete': True,
        'all_accepted_crops_have_geometry': True, 'unreviewed_acquired_external_candidates': 0,
        'public_train_pool_images': len(public), 'external_weak_label_images': len(external),
        'supplementary_mapping_images': len(supplement), 'selected_crops_materialized': len(all_images),
        'excluded_and_reference_records': len(exclusions),
        'original_image_counts': dict(acquisition_counts), 'original_images_total': len(inventory),
        'raw_annotation_files': len(annotations),
        'synthetic_auxiliary_development': len(rows(META / 'cranfield_synthetic_classification_auxiliary_train.csv')),
        'synthetic_auxiliary_holdout': len(rows(META / 'cranfield_synthetic_classification_auxiliary_val.csv')),
        'new_web_originals_this_round': 8, 'new_primary_images_accepted_this_round': 0,
        'public_class_counts': dict(Counter(r['class_name'] for r in public)),
        'external_class_counts': dict(Counter(r['class_name'] for r in external)),
        'pending_identity_decisions_excluded_from_release': 17,
        'unresolved_Emax_mapping_supplementary_images': len(supplement),
        'seven_class_source_independent_test_ready': False,
        'source_independent_7_to_3_and_grouped_cv_ready': False,
        'split_executed': False, 'local_model_or_software_tests_run': False,
        'input_v1_freeze_sha256': digest(previous / 'freeze.json')}
    save_json(destination / 'dataset_summary.json', summary)
    class_table = '\n'.join('| ' + r['class_name'] + ' | ' + str(r['images']) + ' | ' +
                            str(sum(x['class_name'] == r['class_name'] for x in external)) + ' |'
                            for r in feasibility)
    readme = fr'''# 无人机数据交付 v2

本轮数据整理完成，采集轮次已关闭。已固定图片、标签、原图清单、来源、排除决定和校验哈希。v1和已回传的Colab实验均保留。

## 实际数量

- 公开作者子集：2973张原图、2973份YOLO标注；筛选后训练池{len(public)}张，7类。
- 自编爬虫：179张原图，含173张已查看的实验候选及6张身份参考；网页弱标签池{len(external)}张，6类。
- EZPilot身份补充：{len(supplement)}张，作者Emax类映射未定，类别标签留空。
- 额外第三方合成数据：8001张原图，隔离3张跨split重复后，四类开发/留出各3999张。保持独立任务，不能加到七类真实数据规模。
- 实际固定目标裁剪{len(all_images)}张；已隔离的不同分辨率同图不复制进本版本图片目录。
- 本轮新下载8张，逐图检查后全部排除出主实验：教程叠加图、零件/局部/遥控器、错误机体或来源未核实的宣传产品图。不能把采集成功等同于可用样本增加。

| 类别 | 作者标签训练池 | 网页弱标签池 |
|---|---:|---:|
{class_table}

## 文件入口

- public_train_pool.csv：公开作者标签和实际裁剪图，供监督拟合；尚未执行划分。
- external_weak_labels.csv：135张来源文字支持的弱标签图。不能称为已认证的独立七类实拍测试集。
- supplementary_identity.csv：4张EZPilot身份补充图，不强行赋Emax标签。
- raw_acquisition_inventory.csv：11153张已下载原图的路径、哈希、来源与用途；其中7998张合成辅助可用，其他按排除记录处理。公开行另链YOLO原标注及哈希。
- excluded_and_reference.csv：{len(exclusions)}条主任务排除/参考记录；17张型号待定在本版本统一排除，不继续等候，原始审核历史保留。
- metadata/：生成时的清洗和来源台账快照，含合成辅助两份CSV。
- evidence/：最新179张原图完整性回执、本轮8张观察记录及图片、采集配置。
- split_feasibility.csv、similarity_components.json：真实的划分可行性检查。相似关系是候选，不伪装成已证实重复或飞行ID。
- freeze.json：全部交付文件、11153张原图、2973份标注、目标裁剪和网页证据的哈希。

所有CSV图像路径相对于项目根目录D:/ml_hw；images/是本版本实际目标裁剪，原始图片和YOLO标注仍在data/raw/。不要把整个data/final/递归当成一个训练集：v1和v2属于同批数据的不同版本。按所选版本CSV读取，合成辅助另读metadata/中的四类清单。

## 已完成与仍不成立的结论

文件整理和当前采集轮次已经结束，没有未下载但计入数量的链接，没有暂留图片缺框。作者标签用于拟合，不声称全部型号经过专家鉴定。外部图保留弱标签和未知来源独立性。论文许可与数据再发布许可分别核实；开源仓库默认放代码、来源URL和哈希，不直接上传许可未核实图片。

本次没有产生训练/验证/测试划分或新模型结果。Bebop、Intel、Mambo各只有一个公开相关场景组；若严格遵守已有相关组不跨集合的约束，它们无法同时进入来源隔离的训练和测试。Intel的256张还落入同一保守相似连接分量，不能靠随机打散得到可信的独立7:3。缺少来源证据的Phantom3图片也不能按未知ID当作独立飞行。

因此，数据交付完成不等于满足严格论文的独立七类测试与分组交叉验证。当前网页池缺少已解决映射的Emax，Intel只有一张开箱挑战图。不能宣称当前已经满足老师全部实验要求，也不能将四类合成实验的98%以上成绩换成七类主任务成绩。

数据核查实际读取原图哈希和全部目标裁剪并检查解码、标签、排除表及证据。只运行数据整理；没有本地软件测试、模型推理、学习特征提取、训练或模型评估，没有生成PDF。

## 分步骤源代码

1. 作者数据：scripts/01_import_public_zip.py；联网备用scripts/01_download_public.py。
2. 网页采集：scripts/02_crawl_external.py、02b_crawl_articles.py、02c_discover_source_links.py。
3. 审核：scripts/03b_review_external.py；本轮观察转录scripts/review_records/2026_10_22_complete_rare_sources.py。
4. 裁剪：scripts/04c_export_candidate_crops.py、04d_export_external_candidate_crops.py。
5. 当前收尾：scripts/04f_finalize_dataset.py；对应逻辑src/uav_project/dataset_completion.py。

上述步骤均已有运行产物，不需要再下载一遍。本版本冻结后复核可运行：

    .\.venv\Scripts\python.exe -X utf8 scripts/04f_finalize_dataset.py --audit-only

该命令只核对数据文件，不训练模型。后续若取得新的有效独立来源，另建v3，保留本版本。
'''
    (destination / 'README.md').write_text(readme, encoding='utf-8')
    delivery = {relative(p): digest(p) for p in destination.rglob('*') if p.is_file()}
    save_json(destination / 'freeze.json', {'version': VERSION, 'frozen_at': now(),
        'file_hashes': delivery, 'original_image_hashes': original_hashes,
        'annotation_hashes': annotations, 'source_evidence_hashes': source_hashes,
        'selected_image_hashes': all_images, 'independent_seven_class_test_approved': False})
    save_json(OUT / 'dataset_release_current.json', {'version': VERSION,
        'folder': relative(destination), 'summary': relative(destination / 'dataset_summary.json'),
        'freeze_sha256': digest(destination / 'freeze.json'), 'data_assembly_complete': True,
        'strict_paper_evaluation_ready': False, 'model_run': False})
    return audit_dataset_completion()


def audit_dataset_completion():
    destination = ROOT / 'data/final' / VERSION
    freeze = read_json(destination / 'freeze.json')
    for key in ('file_hashes', 'original_image_hashes', 'annotation_hashes', 'source_evidence_hashes'):
        verify_hashes(freeze[key])
    listed = [r for name in MANIFESTS for r in rows(destination / name)]
    ids = [r['sample_id'] for r in listed]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate selected sample IDs')
    actual = {relative(p) for p in (destination / 'images').rglob('*.png')}
    if actual != {r['image_path'] for r in listed}:
        raise ValueError('Listed and physical image sets differ')
    primary = [r for r in listed if r['label'] != '']
    if len({r['pixel_sha256'] for r in primary}) != len(primary):
        raise ValueError('Exact pixel duplicate in primary pool')
    for name, allowed in ((MANIFESTS[0], set(map(str, range(7)))),
                          (MANIFESTS[1], set(map(str, range(7)))),
                          (MANIFESTS[2], {''})):
        if any(r['label'] not in allowed for r in rows(destination / name)):
            raise ValueError('Invalid class mapping')
    status = {'checked_at': now(), 'version': VERSION, 'all_data_files_match': True,
        'delivery_files_checked': len(freeze['file_hashes']),
        'original_images_checked': len(freeze['original_image_hashes']),
        'annotations_checked': len(freeze['annotation_hashes']),
        'source_snapshots_checked': len(freeze['source_evidence_hashes']),
        'selected_images': len(listed), 'unlisted_images_in_current_version': 0,
        'image_IDs_unique': True, 'primary_exact_pixel_duplicates': 0,
        'model_run': False, 'software_tests_run': False,
        'source_independence_proven': False, 'independent_seven_class_test_ready': False,
        'freeze_sha256': digest(destination / 'freeze.json')}
    save_json(OUT / 'dataset_completion_v2_audit.json', status)
    print('Dataset v2 finalized:', destination, flush=True)
    print('Checked originals:', status['original_images_checked'],
          'YOLO labels:', status['annotations_checked'], 'selected crops:', len(listed), flush=True)
    print('Data assembly complete; source-independent seven-class evaluation is not ready.', flush=True)
    return status
