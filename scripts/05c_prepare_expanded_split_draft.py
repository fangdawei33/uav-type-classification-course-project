"""05c. Produce an explicit partial-coverage split draft, never a full benchmark claim."""
from collections import Counter, defaultdict
from itertools import combinations
import random
from _bootstrap import PROJECT_ROOT
from uav_project.common import ROOT, rows, read_json, write_rows, save_json, digest, now


def prepare():
    source = ROOT / 'data/final/uav_type_v3/clean_manifest.csv'
    destination = ROOT / 'outputs/expansion/processing_v3/split_draft'
    if (destination / 'receipt.json').exists():
        raise FileExistsError('Preserve the existing split draft')
    frame = rows(source)
    summary = read_json(source.parent / 'data_summary.json')
    if digest(source) != summary['clean_manifest_sha256']:
        raise ValueError('Clean manifest changed')
    grouped = defaultdict(list)
    for r in frame:
        grouped[r['group_id']].append(r)
    if any(len({r['label'] for r in members}) != 1 for members in grouped.values()):
        raise ValueError('This draft chooser requires single-label constraint groups')
    test_groups = set()
    support = []
    rng = random.Random(42)
    for label in sorted({r['label'] for r in frame}, key=int):
        members = [r for r in frame if r['label'] == label]
        groups = sorted({r['group_id'] for r in members})
        rng.shuffle(groups)
        choices = []
        for count in range(1, len(groups)):
            for subset in combinations(groups, count):
                images = sum(len(grouped[g]) for g in subset)
                choices.append((abs(images/len(members)-0.3), subset))
        selected = min(choices, key=lambda item: item[0])[1] if choices else ()
        test_groups.update(selected)
        n_test = sum(len(grouped[g]) for g in selected)
        support.append({'label': label, 'class_name': members[0]['class_name'],
            'images': len(members), 'development': len(members)-n_test, 'holdout': n_test,
            'constraint_groups': len(groups), 'coverage': 'partial_evaluation_candidate' if selected else 'training_only_no_group_isolated_holdout',
            'authenticated_source_independence': False})
    draft = []
    for r in frame:
        draft.append({**r, 'draft_split': 'holdout' if r['group_id'] in test_groups else 'development',
                      'draft_status': 'not_active_training_protocol',
                      'evaluation_scope': '8_class_holdout_support_for_12_label_classifier_only'})
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / 'manifest.csv', draft)
    write_rows(destination / 'class_support.csv', support)
    by_group = defaultdict(set)
    for r in draft:
        by_group[r['group_id']].add(r['draft_split'])
    if any(len(v)>1 for v in by_group.values()):
        raise ValueError('Constraint group crossed partitions')
    actual = Counter(r['draft_split'] for r in draft)
    result = {'created_at': now(), 'seed': 42, 'images': len(frame), 'counts': dict(actual),
        'holdout_fraction': actual['holdout']/len(frame), 'requested_holdout_fraction': 0.3,
        'selection_rule': 'Choose per-class constraint-group subset closest to 30%; labels/counts only, no model scores.',
        'classifier_label_count': len(support), 'holdout_supported_class_count': sum(r['holdout']>0 for r in support),
        'training_only_classes': [r['class_name'] for r in support if r['holdout']==0],
        'group_leakage': False, 'authenticated_source_independence': False,
        'active_training_protocol': False, 'all_12_class_test_ready': False,
        'cross_validation_ready_for_all_12_classes': False,
        'meaning': 'Concrete diagnostic option; retained labels all remain in development. Never report this as full 12-class performance or authenticated source-independent evaluation.',
        'input_manifest_sha256': digest(source), 'draft_manifest_sha256': digest(destination/'manifest.csv'),
        'models_or_software_tests_run': False}
    save_json(destination / 'receipt.json', result)
    print('Partial-coverage split draft:', dict(actual), flush=True)
    print('Holdout support:', result['holdout_supported_class_count'], '/ 12 classes; NOT active.', flush=True)


if __name__ == '__main__':
    prepare()
