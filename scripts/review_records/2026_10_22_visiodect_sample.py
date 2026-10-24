"""Record only the 144 original/crop pairs actually viewed on 2026-10-22."""
import sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from uav_project.common import OUT, rows, write_rows, save_json, digest, now

SHEET_NOTES = {
    1: 'Visible small dark airframes; marked pixelation and similar building background.',
    2: 'Visible airframes against evening sky; several soft silhouettes.',
    3: 'Visible airframes; portrait originals and mixed target distances.',
    4: 'Visible small airframes against cloudy sky; fine details often weak.',
    5: 'Visible airframes against blue sky; structure clearer in some crops.',
    6: 'Visible airframes with substantial blur or pixelation in some crops.',
    7: 'Visible Phantom-family silhouettes; exact subtype not authenticated.',
    8: 'Visible Phantom-family silhouettes; some highly blurred targets.',
    9: 'Visible Phantom-family bodies and landing gear; excluded from fine-grained merge.',
    10: 'Visible larger airframes, long landing gear and tank-like central structure.',
    11: 'Visible large airframes at varying distances; some blurred silhouettes.',
    12: 'Visible airframes; some motion blur and strong perspective changes.',
    13: 'Visible small airframes with guards; source product identity not authenticated.',
    14: 'Mostly visible guarded airframes, some severe pixelation.',
    15: 'Visible guarded airframes; several blurred and distant targets.',
    16: 'Visible dark airframes; product subtype not independently authenticated.',
    17: 'Position 2 shows sky without an identifiable drone; others contain small silhouettes.',
    18: 'Visible dark airframes; several distant blurred targets.',
}
REJECTED = {'visio_path_8dc68e44c7e301142104': 'No identifiable drone in author crop or additionally viewed full original.'}


def record():
    folder = OUT / 'expansion/visiodect_v1/visual_sample'
    if (folder / 'receipt.json').exists():
        raise FileExistsError('Preserve the dated observation record')
    source = OUT / 'expansion/visiodect_v1/boxes/sampled_crops.csv'
    sampled = rows(source)
    observed = []
    for r in sampled:
        page = int(Path(r['sheet']).stem.split('_')[-1])
        if page not in SHEET_NOTES:
            raise ValueError('Sheet was not included in the actually viewed sample')
        sid = r['sample_id']
        observed.append({
            'sample_id': sid, 'path': r['path'], 'sheet': r['sheet'],
            'position': r['position'], 'source_label': r['source_label'],
            'source_scenario': r['source_scenario'],
            'decision': 'exclude_observed_no_identifiable_target' if sid in REJECTED else 'sampled_candidate_only',
            'observation': REJECTED.get(sid, SHEET_NOTES[page]),
            'additional_full_original_viewed': sid in REJECTED,
            'reviewer': 'AI_assisted_visual_observation',
            'expert_product_authentication': False,
            'independent_flight_authentication': False,
        })
    if len(sampled) != 144 or not REJECTED.keys() <= {r['sample_id'] for r in sampled}:
        raise ValueError('The fixed viewed sample changed')
    write_rows(folder / 'observations.csv', observed)
    save_json(folder / 'receipt.json', {
        'recorded_at': now(), 'sheets_actually_viewed': 18,
        'original_crop_pairs_actually_viewed': len(observed),
        'additional_full_originals_viewed': len(REJECTED),
        'decision_counts': dict(Counter(r['decision'] for r in observed)),
        'full_dataset_visually_reviewed': False,
        'source_manifest_sha256': digest(source),
        'observations_sha256': digest(folder / 'observations.csv'),
        'scope': 'Fixed sorted-path sample only; observations do not approve unviewed images or authenticate product labels.',
    })
    print('Recorded actual visual observations:', len(observed), '; exclusions:', len(REJECTED), flush=True)


if __name__ == '__main__':
    record()
