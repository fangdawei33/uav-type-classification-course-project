"""Acquire and inspect a separate, versioned dataset before defining new labels."""
from collections import Counter, defaultdict
import hashlib
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
from .common import (ROOT, META, OUT, read_json, save_json, rows, write_rows,
                     digest, now, relative, resolve)

SOURCE = 'visiodect_v1'
RECEIPT = OUT / 'expansion' / SOURCE / 'download_receipt.json'
RASTER = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def spec():
    return read_json(ROOT / 'configs/expansion_sources.json')[SOURCE]


def download_visiodect():
    source = spec()
    target = resolve(source['raw_destination'])
    if shutil.disk_usage(ROOT).free < 8_000_000_000:
        raise RuntimeError('At least 8 GB free space is required for this fixed dataset')
    if RECEIPT.exists():
        old = read_json(RECEIPT)
        if old['dataset_handle'] != source['dataset_handle']:
            raise ValueError('Pinned version changed; preserve existing dataset')
        if not target.is_dir():
            raise FileNotFoundError('Recorded download directory missing')
        print('Recorded dataset already acquired:', target, flush=True)
        return old
    # The official public dataset client handles its download protocol. Keep the
    # client cache inside this workspace and never substitute unofficial mirrors.
    os.environ['KAGGLEHUB_CACHE'] = str(ROOT / 'data/downloads/kagglehub')
    import kagglehub
    target.mkdir(parents=True, exist_ok=True)
    started = now()
    print('Downloading fixed author release:', source['dataset_handle'], flush=True)
    result = Path(kagglehub.dataset_download(source['dataset_handle'], output_dir=str(target))).resolve()
    if not result.is_relative_to(target.resolve()):
        raise ValueError('Client returned a directory outside the requested dataset destination')
    receipt = {'source_id': SOURCE, 'dataset_handle': source['dataset_handle'],
        'author_page': source['author_page'], 'started_at': started, 'finished_at': now(),
        'path': relative(result), 'download_client': 'kagglehub.dataset_download',
        'license_as_published': source['license_as_published'],
        'source_config_sha256': digest(ROOT / 'configs/expansion_sources.json'),
        'actual_image_count': 'pending_file_inventory', 'class_mapping_approved': False,
        'added_to_training': False, 'local_models_or_software_tests_run': False}
    save_json(RECEIPT, receipt)
    print('Acquired files:', result, '; run 03g to inspect actual files.', flush=True)
    return receipt


def catalog_visiodect():
    if not RECEIPT.exists():
        raise FileNotFoundError('Run scripts/01e_download_expansion.py first')
    source = spec()
    root = resolve(read_json(RECEIPT)['path'])
    destination = OUT / 'expansion' / SOURCE
    receipt_path = destination / 'inventory_receipt.json'
    if receipt_path.exists():
        raise FileExistsError('Inventory already recorded; preserve the fixed dataset audit')
    records, labels, other, problems = [], [], [], []
    file_entries = []
    files = sorted(p for p in root.rglob('*') if p.is_file())
    class_counts, scenarios = Counter(), Counter()
    image_keys = defaultdict(list)
    annotations = defaultdict(list)
    hashes = defaultdict(list)
    for i, p in enumerate(files, 1):
        name = p.relative_to(root).as_posix()
        parts = p.relative_to(root).parts
        suffix = p.suffix.lower()
        sha = digest(p)
        file_entries.append({'path': relative(p), 'sha256': sha, 'bytes': p.stat().st_size,
                             'extension': suffix})
        if suffix in RASTER:
            try:
                # Author images/<weather>/ filename hierarchy was observed in
                # the public file inventory. Reject unfamiliar structures.
                pos = next(n for n, part in enumerate(parts) if part.lower() == 'images')
                if pos == 0 or len(parts) <= pos + 2:
                    raise ValueError('Unrecognized image folder hierarchy')
                label, scenario = parts[pos - 1], parts[pos + 1].casefold()
                with Image.open(p) as image:
                    image.load()
                    width, height = image.size
                sid = 'visio_path_' + hashlib.sha256(name.encode()).hexdigest()[:20]
                record = {'source_record_id': sid, 'content_id': 'visio_content_' + sha[:20],
                    'sample_id': sid, 'path': relative(p), 'sha256': sha,
                    'source_id': SOURCE, 'source_label': label, 'source_scenario': scenario,
                    'width': width, 'height': height, 'data_origin': source['data_origin'],
                    'source_member': name, 'source_url': source['author_page'],
                    'original_flight_id': '', 'flight_identity_status': 'unknown_not_invented',
                    'label_mapping': 'pending', 'training_approved': False}
                records.append(record)
                class_counts[label] += 1
                scenarios[(label, scenario)] += 1
                image_keys[(label, scenario, p.stem.casefold())].append(record)
                hashes[sha].append(record)
            except Exception as exc:
                problems.append({'path': relative(p), 'problem': repr(exc)})
        elif suffix in {'.txt', '.xml', '.csv'}:
            annotation_container = next((n for n, part in enumerate(parts)
                if part.casefold() in {'labels', 'annotations'}), None)
            label = (parts[annotation_container - 1] if annotation_container is not None and annotation_container > 0
                     else next((r for r in parts if r in source['author_reported_labels']), ''))
            scenario = next((r.casefold() for r in parts if r.casefold() in {'cloudy', 'sunny', 'evening'}), '')
            item = {'path': relative(p), 'sha256': sha, 'source_label': label,
                    'source_scenario': scenario, 'format': suffix, 'bytes': p.stat().st_size}
            if suffix == '.xml':
                try:
                    tree = ET.parse(p)
                    item['object_names'] = '|'.join(n.text or '' for n in tree.findall('.//object/name'))
                    item['object_count'] = len(tree.findall('.//object'))
                    item['xml_width'] = tree.findtext('.//size/width', '')
                    item['xml_height'] = tree.findtext('.//size/height', '')
                except ET.ParseError as exc:
                    problems.append({'path': relative(p), 'problem': repr(exc)})
            labels.append(item)
            annotations[(label, scenario, p.stem.casefold())].append(item)
        else:
            other.append({'path': relative(p), 'sha256': sha, 'bytes': p.stat().st_size})
        if i % 5000 == 0:
            print('Actual files inspected:', i, '/', len(files), flush=True)
    for key, members in image_keys.items():
        matched = annotations.get(key, [])
        for r in members:
            r['annotation_paths'] = ' | '.join(a['path'] for a in matched)
            r['annotation_formats'] = ' | '.join(sorted({a['format'] for a in matched}))
            r['exact_file_duplicate_group_size'] = len(hashes[r['sha256']])
    write_rows(META / 'visiodect_v1_images.csv', records)
    write_rows(META / 'visiodect_v1_annotations.csv', labels)
    write_rows(destination / 'all_acquired_files.csv', file_entries)
    write_rows(destination / 'other_files.csv', other, ['path', 'sha256', 'bytes'])
    write_rows(destination / 'class_scenario_counts.csv',
        [{'source_label': k[0], 'scenario': k[1], 'images': v} for k, v in sorted(scenarios.items())])
    duplicate_rows = [r for group in hashes.values() if len(group) > 1 for r in group]
    write_rows(destination / 'exact_file_duplicate_members.csv', duplicate_rows,
               list(records[0]) if records else ['sample_id'])
    result = {'finished_at': now(), 'source_id': SOURCE, 'actual_files': len(files),
        'actual_decoded_images': len(records), 'annotation_files': len(labels),
        'other_files': len(other), 'source_class_counts': dict(class_counts),
        'author_reported_images': source['author_reported_images'],
        'images_without_matched_annotation': sum(not r['annotation_paths'] for r in records),
        'unique_image_file_hashes': len(hashes),
        'exact_file_duplicate_groups': sum(len(v) > 1 for v in hashes.values()),
        'cross_class_exact_file_groups': sum(len({r['source_label'] for r in v}) > 1 for v in hashes.values()),
        'problems': problems, 'class_mapping_approved': False, 'visual_review_complete': False,
        'added_to_training': False, 'source_independent_split_ready': False,
        'local_model_or_software_tests_run': False,
        'image_manifest_sha256': digest(META / 'visiodect_v1_images.csv'),
        'annotation_manifest_sha256': digest(META / 'visiodect_v1_annotations.csv')}
    result['all_acquired_files_manifest_sha256'] = digest(destination / 'all_acquired_files.csv')
    save_json(receipt_path, result)
    print('Actual image counts:', dict(class_counts), flush=True)
    print('Inventory saved:', receipt_path, '; images are not yet added to training.', flush=True)
    return result
