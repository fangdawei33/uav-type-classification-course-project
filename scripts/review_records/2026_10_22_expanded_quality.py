"""Transcribe the ten flagged crops actually viewed with original context."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from uav_project.common import rows, resolve, write_rows, save_json, digest, now
from uav_project.expanded_processing import settings

DECISIONS = {
    'dts_1uiYVlojDRReZW7HSqmw2Om3xkvYo-ZqS': ('retain_difficult_visible_target', 'Small blurred bright airframe remains visible against dark background; darkness alone is not an exclusion.'),
    'visio_path_bec79de19af6ec722f0f': ('retain_difficult_visible_target', 'Very faint blurred horizontal target-like silhouette within the box; retain difficult candidate, product identity not authenticated.'),
    'visio_path_3231aee27a3eff57d686': ('exclude_no_identifiable_target', 'Original/context and crop show sky without an identifiable target.'),
    'visio_path_e825d92cf6af1b1b2871': ('exclude_no_identifiable_target', 'No identifiable target in the source author box; sky/cloud texture only.'),
    'visio_path_d692daa389634da736dd': ('exclude_no_identifiable_target', 'No identifiable airframe in the source author box.'),
    'visio_path_a1a91b586b334dc32c63': ('exclude_no_identifiable_target', 'Crop shows sky without an identifiable airframe.'),
    'visio_path_e6589fdb85bfc0c1b313': ('exclude_no_identifiable_target', 'Nearly uniform sky crop without an identifiable airframe.'),
    'visio_path_9bcc4afbd911e88db30f': ('exclude_no_identifiable_target', 'Sky/cloud crop without an identifiable target.'),
    'visio_path_0a07aad0f06850970ba1': ('exclude_no_identifiable_target', 'Portrait sky crop without an identifiable airframe.'),
    'visio_path_f25aad6e3005d2a67e17': ('exclude_no_identifiable_target', 'Wide sky crop without an identifiable airframe.'),
}


def record():
    work = resolve(settings()['work_directory'])
    target = work / 'quality_visual_observations.csv'
    if target.exists():
        raise FileExistsError('Preserve dated observations')
    queue = rows(work / 'quality_review_queue.csv')
    if {r['sample_id'] for r in queue} != set(DECISIONS):
        raise ValueError('Viewed queue changed')
    observations = []
    for r in queue:
        decision, note = DECISIONS[r['sample_id']]
        observations.append({'sample_id': r['sample_id'], 'crop_path': r['image_path'],
            'crop_sha256': r['sha256'], 'raw_path': r['raw_path'], 'raw_sha256': r['raw_sha256'],
            'decision': decision, 'observation': note, 'reviewer': 'AI_assisted_visual_observation',
            'evidence': 'quality_queue_01.jpg plus quality_context_01.jpg and quality_context_02.jpg',
            'expert_identity_verification': False})
    write_rows(target, observations)
    save_json(work / 'quality_visual_receipt.json', {'recorded_at': now(),
        'observations': len(observations), 'exclusions': 8, 'retained_difficult_cases': 2,
        'observations_sha256': digest(target),
        'evidence_sha256': {name: digest(work / name) for name in
            ['quality_queue_01.jpg', 'quality_context_01.jpg', 'quality_context_02.jpg']},
        'all_dataset_images_visually_reviewed': False})
    print('Recorded quality observations: 10; exclude 8, retain difficult cases 2.', flush=True)


if __name__ == '__main__':
    record()
