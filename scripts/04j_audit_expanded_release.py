"""04j. Verify actual released crop hashes, manifests and partition constraints."""
from collections import Counter
from _bootstrap import PROJECT_ROOT
from uav_project.common import ROOT, rows, read_json, digest, resolve, save_json, now


def audit():
    root = ROOT / 'data/final/uav_type_v3'
    frozen = read_json(root/'freeze.json')
    inventory = root/'files_sha256.csv'
    if digest(inventory) != frozen['inventory_sha256']:
        raise ValueError('Frozen file inventory changed')
    files = rows(inventory)
    issues = []
    for n, r in enumerate(files, 1):
        path = resolve(r['path'])
        if not path.is_file() or digest(path) != r['sha256']:
            issues.append({'path': r['path'], 'issue': 'missing_or_hash_changed'})
        if n % 4000 == 0:
            print('Released files hash-checked:', n, '/', len(files), flush=True)
    clean = rows(root/'clean_manifest.csv')
    candidates = rows(root/'candidate_inventory.csv')
    excluded = rows(root/'excluded_pixel_duplicates.csv')+rows(root/'excluded_visual_quality.csv')
    if len({r['sample_id'] for r in clean}) != len(clean):
        issues.append({'issue': 'repeated_clean_sample_id'})
    if len({r['pixel_sha256'] for r in clean}) != len(clean):
        issues.append({'issue': 'repeated_clean_rgb_pixels'})
    if {r['sample_id'] for r in clean} & {r['sample_id'] for r in excluded}:
        issues.append({'issue': 'clean_excluded_overlap'})
    if {r['sample_id'] for r in clean} | {r['sample_id'] for r in excluded} != {r['sample_id'] for r in candidates}:
        issues.append({'issue': 'candidate_accounting_mismatch'})
    listed = {r['image_path'] for r in candidates}
    actual = {p.relative_to(ROOT).as_posix() for p in (root/'images').rglob('*.png')}
    if actual != listed:
        issues.append({'issue': 'physical_crop_inventory_mismatch'})
    result = {'checked_at': now(), 'files_checked': len(files), 'issues': issues,
        'candidate_crops': len(candidates), 'retained_images': len(clean), 'excluded': len(excluded),
        'class_counts': dict(Counter(r['class_name'] for r in clean)),
        'meaning': 'Actual data integrity audit, not software/model testing or full visual review.',
        'models_or_software_tests_run': False}
    save_json(ROOT/'outputs/expansion/processing_v3/release_audit.json',result)
    print('Data audit:', len(files), 'files;', len(issues), 'issues.', flush=True)
    if issues:
        raise ValueError('Release integrity problems recorded')


if __name__ == '__main__':
    audit()
