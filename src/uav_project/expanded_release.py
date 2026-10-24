"""Release actual crops and conservative grouping evidence, without model code."""
from collections import Counter, defaultdict
import csv
import hashlib
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from .common import ROOT, rows, read_json, write_rows, save_json, digest, resolve, relative, now
from .expanded_processing import settings


class Components:
    def __init__(self, ids):
        self.parent = {sid: sid for sid in ids}
        self.size = dict.fromkeys(self.parent, 1)

    def find(self, sid):
        root = sid
        while self.parent[root] != root:
            root = self.parent[root]
        while sid != root:
            previous = self.parent[sid]
            self.parent[sid] = root
            sid = previous
        return root

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def similarity_constraints(records, cfg, target, components):
    """Four 16-bit indexes cover every 64-bit pair within Hamming distance 3."""
    radius = cfg['near_candidate_hamming_distance']
    if radius != 3 or cfg['near_candidate_thumbnail_size'] != 16:
        raise ValueError('This fixed protocol expects radius 3 and 16x16 thumbnails')
    hashes = [int(r['dhash64']) for r in records]
    thumbnails = np.stack([np.frombuffer(bytes.fromhex(r['gray_thumbnail_hex']), dtype=np.uint8)
                           for r in records]).astype(np.int16)
    buckets = defaultdict(list)
    comparisons = edges = cross_class = 0
    with target.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_a', 'sample_b', 'class_a', 'class_b',
            'crop_dhash_hamming', 'gray_mae_16', 'meaning'])
        writer.writeheader()
        for i, value in enumerate(hashes):
            candidates = set()
            keys = [(band, (value >> (band*16)) & 65535) for band in range(4)]
            for key in keys:
                candidates.update(buckets[key])
            for j in sorted(candidates):
                distance = (value ^ hashes[j]).bit_count()
                comparisons += 1
                if distance > radius:
                    continue
                mae = float(np.abs(thumbnails[i] - thumbnails[j]).mean())
                if mae > cfg['near_candidate_gray_mae']:
                    continue
                a, b = records[j], records[i]
                components.join(a['sample_id'], b['sample_id'])
                edges += 1
                cross_class += a['label'] != b['label']
                writer.writerow({'sample_a': a['sample_id'], 'sample_b': b['sample_id'],
                    'class_a': a['class_name'], 'class_b': b['class_name'],
                    'crop_dhash_hamming': distance, 'gray_mae_16': round(mae, 5),
                    'meaning': 'conservative_similarity_constraint_not_duplicate_or_flight_proof'})
            for key in keys:
                buckets[key].append(i)
            if (i+1) % 2000 == 0:
                stream.flush()
                print('Similarity constraint scan:', i+1, '/', len(records), '; links:', edges, flush=True)
    return {'candidate_pairs_compared': comparisons, 'similarity_links': edges,
            'cross_class_similarity_links': cross_class,
            'meaning': 'Small-target shape/background similarity can be a false match; links only constrain splitting, not labels or deletion.'}


def prepare_quality_queue(records, folder):
    """Show flagged cases without using class predictions or deleting hard images."""
    folder.mkdir(parents=True, exist_ok=True)
    flagged = [r for r in records if 'low_contrast' in r['quality_flags'] or 'mostly_dark' in r['quality_flags']]
    write_rows(folder / 'quality_review_queue.csv', flagged, list(records[0]))
    selected = sorted(flagged, key=lambda r: (float(r['gray_std']), r['sample_id']))[:96]
    manifest = []
    for start in range(0, len(selected), 24):
        sheet = Image.new('RGB', (1440, 1440), 'white')
        draw = ImageDraw.Draw(sheet)
        number = start//24 + 1
        path = folder / f'quality_queue_{number:02}.jpg'
        for local, r in enumerate(selected[start:start+24]):
            x, y = local % 4 * 360, local//4 * 240
            with Image.open(resolve(r['image_path'])) as image:
                preview = ImageOps.contain(image, (340, 175))
                sheet.paste(preview, (x+(360-preview.width)//2, y+5+(175-preview.height)//2))
            draw.text((x+8, y+184), r['sample_id'], fill='black')
            draw.text((x+8, y+200), r['class_name']+f" | {r['width']}x{r['height']}", fill='black')
            draw.text((x+8, y+216), 'Prepared for review; no auto exclusion.', fill='black')
            manifest.append({'sample_id': r['sample_id'], 'sheet': relative(path), 'position': local+1,
                             'gray_std': r['gray_std'], 'visually_reviewed': False})
        sheet.save(path, quality=95)
    write_rows(folder / 'quality_queue_preview.csv', manifest, ['sample_id', 'sheet', 'position', 'gray_std', 'visually_reviewed'])
    return len(flagged)


def finalize():
    cfg = settings()
    destination, work = resolve(cfg['destination']), resolve(cfg['work_directory'])
    frozen = destination / 'freeze.json'
    if frozen.exists():
        print('Release already frozen:', frozen, flush=True)
        return read_json(frozen)
    source = work / 'processed_images.csv'
    processing = read_json(work / 'processing_receipt.json')
    if digest(source) != processing['manifest_sha256']:
        raise ValueError('Image processing manifest changed')
    records = rows(source)
    if not records:
        raise ValueError('Complete 04h first')
    pixel_groups = defaultdict(list)
    for r in records:
        pixel_groups[r['pixel_sha256']].append(r)
    observations_path = work / 'quality_visual_observations.csv'
    observations = rows(observations_path)
    observed_by_id = {r['sample_id']: r for r in observations}
    if len(observed_by_id) != len(observations):
        raise ValueError('Repeated visual observations')
    for sid, observation in observed_by_id.items():
        matches = [r for r in records if r['sample_id'] == sid]
        if len(matches) != 1 or observation['crop_sha256'] != matches[0]['sha256']:
            raise ValueError('Visual observation does not match current crop: ' + sid)
    relations, exclusions, visual_exclusions, selected = [], [], [], []
    for members in pixel_groups.values():
        members.sort(key=lambda r: (not bool(r['existing_crop_path']), r['sample_id']))
        conflict = len({r['label'] for r in members}) > 1
        for i, r in enumerate(members):
            observation = observed_by_id.get(r['sample_id'], {})
            if observation.get('decision') == 'exclude_no_identifiable_target':
                visual_exclusions.append({**r, 'reason': 'observed_no_identifiable_target',
                                          'observation': observation['observation']})
            elif conflict or i:
                exclusions.append({**r, 'reason': 'identical_crop_conflicting_labels' if conflict else 'identical_rgb_crop',
                                   'duplicate_of': '' if conflict else members[0]['sample_id']})
            else:
                selected.append(r)
            if len(members) > 1:
                relations.append({'sample_id': r['sample_id'], 'pixel_sha256': r['pixel_sha256'],
                    'group_size': len(members), 'conflicting_labels': conflict,
                    'representative': '' if conflict else members[0]['sample_id']})
    selected.sort(key=lambda r: (int(r['label']), r['sample_id']))
    ids = {r['sample_id'] for r in selected}
    components = Components(ids)
    scene_representatives = {}
    for r in selected:
        # A missing scene identifier does not prove a new independent source.
        key = r['group_constraint'] or 'unresolved_existing_class_' + r['label']
        if key in scene_representatives:
            components.join(r['sample_id'], scene_representatives[key])
        else:
            scene_representatives[key] = r['sample_id']
    old_links = 0
    for r in rows(ROOT / 'data/metadata/drone_type_set_near_duplicate_candidates.csv'):
        if r['sample_a'] in ids and r['sample_b'] in ids:
            components.join(r['sample_a'], r['sample_b'])
            old_links += 1
    links_path = work / 'crop_similarity_constraints.csv'
    similarity = similarity_constraints(selected, cfg, links_path, components)
    groups = defaultdict(list)
    for r in selected:
        groups[components.find(r['sample_id'])].append(r['sample_id'])
    group_ids = {root: 'constraint_' + hashlib.sha256('\n'.join(sorted(members)).encode()).hexdigest()[:20]
                 for root, members in groups.items()}
    clean = []
    for r in selected:
        output = {k: v for k,v in r.items() if k != 'gray_thumbnail_hex'}
        output.update(group_id=group_ids[components.find(r['sample_id'])],
            group_basis='scene_buckets_plus_similarity_constraints_not_verified_flights',
            split='unassigned', source_independent_evaluation_approved=False)
        clean.append(output)
    write_rows(destination / 'clean_manifest.csv', clean)
    write_rows(destination / 'excluded_pixel_duplicates.csv', exclusions, list(records[0])+['reason', 'duplicate_of'])
    write_rows(destination / 'excluded_visual_quality.csv', visual_exclusions,
               list(records[0])+['reason', 'observation'])
    write_rows(destination / 'pixel_duplicate_relations.csv', relations,
               ['sample_id', 'pixel_sha256', 'group_size', 'conflicting_labels', 'representative'])
    write_rows(destination / 'candidate_inventory.csv',
               [{k:v for k,v in r.items() if k != 'gray_thumbnail_hex'} for r in records])
    feasibility, counts = [], []
    for label in sorted({r['label'] for r in records}, key=int):
        members = [r for r in clean if r['label'] == label]
        by_group = Counter(r['group_id'] for r in members)
        name = next(r['class_name'] for r in records if r['label'] == label)
        counts.append({'label': label, 'class_name': name, 'candidate_images': sum(r['label']==label for r in records),
                       'retained_images': len(members), 'excluded_pixel_duplicates': sum(r['label']==label for r in exclusions),
                       'excluded_visual_quality': sum(r['label']==label for r in visual_exclusions)})
        feasibility.append({'label': label, 'class_name': name, 'images': len(members),
            'conservative_constraint_groups': len(by_group), 'largest_group': max(by_group.values(), default=0),
            'at_least_two_constraint_groups': len(by_group)>=2,
            'authenticated_independent_source_groups': 0,
            'source_independent_holdout_supported': False})
    write_rows(destination / 'class_counts.csv', counts)
    write_rows(destination / 'split_feasibility.csv', feasibility)
    flagged = prepare_quality_queue(clean, work / 'remaining_quality')
    class_map = resolve(cfg['candidate_manifest']).parent / 'class_map.json'
    shutil.copyfile(class_map, destination / 'class_map.json')
    blocked = [r['class_name'] for r in feasibility if not r['at_least_two_constraint_groups']]
    summary = {'created_at': now(), 'version': cfg['version'], 'candidate_images': len(records),
        'retained_images': len(clean), 'classes': len(counts), 'pixel_duplicate_exclusions': len(exclusions),
        'visual_quality_exclusions': len(visual_exclusions),
        'quality_visual_observations_sha256': digest(observations_path) if observations else None,
        'cross_label_exact_pixel_groups': sum(len({r['label'] for r in v})>1 for v in pixel_groups.values()),
        'small_targets_short_side_below_24': sum(min(int(r['width']),int(r['height']))<24 for r in clean),
        'quality_flagged_images_for_review': flagged, 'constraint_groups': len(groups),
        'prior_source_similarity_links_used': old_links, 'new_crop_similarity_scan': similarity,
        'classes_with_less_than_two_constraint_groups': blocked,
        'split_executed': False, 'training_approved': False,
        'split_reason': 'Source identity is unresolved; some classes have a single conservative scene group. No leakage-prone random-frame fallback.',
        'all_images_visually_reviewed': False, 'expert_product_authentication_completed': False,
        'cropped_images_use_author_boxes': True, 'raw_data_modified': False,
        'models_or_software_tests_run': False, 'clean_manifest_sha256': digest(destination / 'clean_manifest.csv'),
        'candidate_manifest_sha256': digest(resolve(cfg['candidate_manifest'])),
        'processing_manifest_sha256': digest(source), 'similarity_constraints_sha256': digest(links_path),
        'release_state': 'native_crops_with_automated_checks_and_explicit_review_limits; not a completed experiment'}
    save_json(destination / 'data_summary.json', summary)
    # Image hashes were checked against actual saved pixels in 04h. Bind the
    # complete crop inventory and final metadata, including excluded derivatives.
    files = [{'path': r['image_path'], 'sha256': r['sha256'], 'kind': 'candidate_crop'} for r in records]
    files.extend({'path': relative(p), 'sha256': digest(p), 'kind': 'release_metadata'}
                 for p in sorted(destination.glob('*')) if p.is_file())
    write_rows(destination / 'files_sha256.csv', files)
    save_json(frozen, {'created_at': now(), 'version': cfg['version'],
        'files': len(files), 'inventory_sha256': digest(destination / 'files_sha256.csv'),
        'meaning': 'Data release snapshot, not model-protocol freeze or training approval.'})
    print('Data release saved:', destination, flush=True)
    print('Retained images:', len(clean), '; classes:', len(counts), '; pixel exclusions:', len(exclusions), flush=True)
    print('Single-group classes:', blocked, '; split not executed.', flush=True)
    return summary
