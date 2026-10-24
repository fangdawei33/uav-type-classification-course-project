"""04g. Prepare a reversible 12-class candidate manifest, not a training split."""
from collections import Counter, defaultdict
from _bootstrap import PROJECT_ROOT
from uav_project.common import (ROOT, META, OUT, rows, write_rows, save_json,
                               read_json, digest, relative, now)

NEW_CLASSES = [
    ('Anafi-Extended', 7), ('DJIFPV', 8), ('EFT-E410S', 9),
    ('Mavic_Air', 10), ('Mavic_Enterprise', 11)]


def prepare(sampled_review=False):
    base = OUT / 'expansion/visiodect_v1'
    folder = base / ('candidates_sampled_review' if sampled_review else 'candidates')
    if (folder / 'receipt.json').exists():
        raise FileExistsError('Candidate manifest already recorded; preserve this version')
    source = META / 'visiodect_v1_author_box_audit.csv'
    images = rows(source)
    if not images:
        raise ValueError('Complete 03g and 03i first')
    observations_path = base / 'visual_sample/observations.csv'
    observations = rows(observations_path) if sampled_review else []
    if sampled_review and not observations:
        raise ValueError('Record the actual sampled visual observations first')
    observed_by_id = {r['sample_id']: r for r in observations}
    folder.mkdir(parents=True, exist_ok=True)
    mapping = dict(NEW_CLASSES)
    old = rows(ROOT / 'data/final/uav_type_v2/public_train_pool.csv')
    old_hashes = defaultdict(list)
    for r in old:
        old_hashes[r['raw_sha256']].append(r)
    combined = []
    for r in old:
        combined.append({'sample_id': r['sample_id'], 'label': r['label'],
            'class_name': r['class_name'], 'source_dataset': 'Drone-type-Set author subset',
            'image_path': r['raw_path'], 'image_sha256': r['raw_sha256'],
            'existing_crop_path': r['image_path'], 'existing_crop_sha256': r['sha256'],
            'target_box_xyxy': r['crop_xyxy'], 'label_basis': 'dataset_author_label',
            'group_constraint': r['constraint_group'], 'group_identity_verified': False,
            'visual_review': 'AI_assisted_original_cohort_review',
            'split': 'unassigned', 'training_split_approved': False})
    hashes = defaultdict(list)
    for r in images:
        hashes[r['sha256']].append(r)
    excluded, selected, first_by_hash = [], [], {}
    for r in sorted(images, key=lambda r: r['source_member']):
        reason = ''
        if r['source_label'] not in mapping:
            reason = 'Phantom_family_overlaps_existing_Phantom3_and_Phantom4Pro'
        elif observed_by_id.get(r['sample_id'], {}).get('decision', '').startswith('exclude_'):
            reason = observed_by_id[r['sample_id']]['decision']
        elif r['annotation_check'] != 'valid_numeric_single_author_box':
            reason = r['annotation_check']
        elif len({x['source_label'] for x in hashes[r['sha256']]}) > 1:
            reason = 'exact_file_duplicate_with_conflicting_author_classes'
        elif r['sha256'] in old_hashes:
            reason = 'exact_file_duplicate_with_existing_seven_class_pool'
        elif r['sha256'] in first_by_hash:
            reason = 'exact_file_duplicate_same_class'
        if reason:
            excluded.append({'sample_id': r['sample_id'], 'path': r['path'],
                'source_label': r['source_label'], 'reason': reason,
                'duplicate_of': (old_hashes[r['sha256']][0]['sample_id']
                    if r['sha256'] in old_hashes else first_by_hash.get(r['sha256'], ''))})
            continue
        first_by_hash[r['sha256']] = r['sample_id']
        item = {'sample_id': r['sample_id'], 'label': mapping[r['source_label']],
            'class_name': r['source_label'], 'source_dataset': 'VisioDECT author version 1',
            'image_path': r['path'], 'image_sha256': r['sha256'],
            'existing_crop_path': '', 'existing_crop_sha256': '',
            'target_box_xyxy': r['box_xyxy'], 'label_basis': 'dataset_author_label_not_expert_reauthenticated',
            'source_scenario': r['source_scenario'],
            'group_constraint': 'visiodect_unresolved_' + r['source_label'] + '_' + r['source_scenario'],
            'group_constraint_basis': 'conservative_class_weather_bucket_not_a_flight_ID',
            'group_identity_verified': False,
            'visual_review': ('sampled_candidate_only' if r['sample_id'] in observed_by_id
                              else 'not_individually_visually_reviewed'),
            'target_short_side_pixels': min(float(r['box_width']), float(r['box_height'])),
            'annotation_path': r['selected_annotation_path'],
            'annotation_sha256': r['selected_annotation_sha256'],
            'split': 'unassigned', 'training_split_approved': False}
        selected.append(item)
        combined.append(item)
    old_classes = read_json(ROOT / 'data/final/uav_type_v2/class_map.json')['classes']
    classes = [{'label': c['label'], 'name': c['name'], 'origin': 'existing_author_class'}
               for c in old_classes] + [
                  {'label': label, 'name': name, 'origin': 'VisioDECT_author_folder_label',
                   'exact_product_identity_verification': 'pending; preserve author names'}
                  for name, label in NEW_CLASSES]
    write_rows(folder / 'expanded_12class_candidates.csv', combined)
    write_rows(folder / 'new_5class_candidates.csv', selected)
    write_rows(folder / 'excluded_expansion.csv', excluded)
    save_json(folder / 'class_map.json', {'classes': classes,
        'status': 'candidate_label_space_not_a_frozen_training_protocol',
        'existing_seven_classes_preserved': True, 'synthetic_not_merged': True})
    result = {'prepared_at': now(), 'existing_real_images': len(old),
        'new_source_images_acquired': len(images), 'new_candidate_images': len(selected),
        'combined_candidate_images': len(combined), 'candidate_classes': len(classes),
        'new_candidate_class_counts': dict(Counter(r['class_name'] for r in selected)),
        'excluded_counts': dict(Counter(r['reason'] for r in excluded)),
        'small_target_candidates_short_side_below_24': sum(r['target_short_side_pixels']<24 for r in selected),
        'added_to_training': False, 'split_executed': False,
        'source_independent_test_ready': False, 'all_images_visually_reviewed': False,
        'raw_data_modified': False, 'models_or_software_tests_run': False,
        'input_annotation_audit_sha256': digest(source),
        'sampled_visual_observations_sha256': digest(observations_path) if sampled_review else None,
        'manifest_sha256': digest(folder / 'expanded_12class_candidates.csv'),
        'limitations': 'Candidate pool only. Exact duplicates removed; near frames, crop validity, label authenticity, class imbalance and source-dependent class distribution still require protocol treatment.'}
    save_json(folder / 'receipt.json', result)
    print('Prepared candidate images:', len(combined), 'across', len(classes), 'classes.', flush=True)
    print('New source candidates by class:', result['new_candidate_class_counts'], flush=True)
    print('Candidate pool only; no split, training or full visual approval.', flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sampled-review', action='store_true',
        help='Apply recorded sample exclusions into a separate revision; preserve the first manifest')
    prepare(parser.parse_args().sampled_review)
