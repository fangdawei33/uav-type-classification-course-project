"""Resumable image processing for the acquired 12-class data, without inference."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import shutil
import numpy as np
from PIL import Image
from .common import ROOT, OUT, read_json, rows, write_rows, save_json, digest, resolve, relative, now
from .cleaning import difference_hash


def settings():
    return read_json(ROOT / 'configs/expanded_data_processing.json')


def pixel_hash(image):
    header = f'RGB:{image.width}:{image.height}:'.encode('ascii')
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def prepare_one(item, destination):
    original = resolve(item['image_path'])
    payload = original.read_bytes()
    if hashlib.sha256(payload).hexdigest() != item['image_sha256']:
        raise ValueError('Original hash changed: ' + item['sample_id'])
    with Image.open(io.BytesIO(payload)) as source:
        orientation = source.getexif().get(274, 1)
        if orientation != 1:
            raise ValueError('Unreviewed EXIF orientation: ' + item['sample_id'])
        size = source.size
        values = tuple(float(v) for v in item['target_box_xyxy'].split(','))
        if len(values) != 4 or not all(math.isfinite(v) for v in values):
            raise ValueError('Invalid box: ' + item['sample_id'])
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
            raise ValueError('Out-of-bounds box: ' + item['sample_id'])
        rect = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
        rgb = source.convert('RGB')
        crop = rgb.crop(rect)
    target = destination / 'images' / f"class_{int(item['label']):02}" / (item['sample_id'] + '.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    if item['existing_crop_path']:
        existing = resolve(item['existing_crop_path'])
        if digest(existing) != item['existing_crop_sha256']:
            raise ValueError('Existing crop hash changed: ' + item['sample_id'])
        with Image.open(existing) as image:
            if image.size != crop.size or image.convert('RGB').tobytes() != crop.tobytes():
                raise ValueError('Existing v2 crop does not match candidate box: ' + item['sample_id'])
        if not target.exists():
            shutil.copyfile(existing, target)
    elif not target.exists():
        temporary = target.with_suffix('.tmp.png')
        crop.save(temporary)
        temporary.replace(target)
    with Image.open(target) as saved:
        saved.load()
        if saved.mode != 'RGB' or saved.size != crop.size or pixel_hash(saved) != pixel_hash(crop):
            raise ValueError('Saved crop differs: ' + item['sample_id'])
    gray = crop.convert('L')
    array = np.asarray(gray, dtype=np.float32)
    thumb = np.asarray(gray.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.uint8)
    smooth = np.asarray(gray.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    lap = smooth[:-2, 1:-1] + smooth[2:, 1:-1] + smooth[1:-1, :-2] + smooth[1:-1, 2:] - 4*smooth[1:-1, 1:-1]
    short = min(crop.size)
    flags = []
    if short < 24:
        flags.append('short_side_below_24')
    if float(array.std()) < 3:
        flags.append('low_contrast_requires_review_not_automatic_exclusion')
    if float((array < 16).mean()) > 0.75:
        flags.append('mostly_dark_not_automatic_exclusion')
    return {**item, 'raw_path': item['image_path'], 'raw_sha256': item['image_sha256'],
        'image_path': relative(target), 'sha256': digest(target), 'pixel_sha256': pixel_hash(crop),
        'raw_width': size[0], 'raw_height': size[1], 'width': crop.width, 'height': crop.height,
        'crop_xyxy': ','.join(map(str, rect)), 'coordinate_origin_verified': False if not item['existing_crop_path'] else 'existing_v2_policy',
        'dhash64': str(difference_hash(gray)), 'gray_thumbnail_hex': thumb.tobytes().hex(),
        'gray_mean': round(float(array.mean()), 4), 'gray_std': round(float(array.std()), 4),
        'laplacian_variance_128': round(float(lap.var()), 4), 'quality_flags': '|'.join(flags),
        'all_images_visually_reviewed': False, 'split': 'unassigned'}


def process_images():
    cfg = settings()
    source = resolve(cfg['candidate_manifest'])
    destination = resolve(cfg['destination'])
    work = resolve(cfg['work_directory'])
    work.mkdir(parents=True, exist_ok=True)
    signature = {'candidate_sha256': digest(source),
                 'configuration_sha256': digest(ROOT / 'configs/expanded_data_processing.json'),
                 'processing_code_sha256': digest(__file__)}
    signature_path = work / 'processing_inputs.json'
    if signature_path.exists() and read_json(signature_path) != signature:
        raise ValueError('Processing inputs changed; preserve journal and choose a new version')
    save_json(signature_path, signature)
    receipt = work / 'processing_receipt.json'
    if receipt.exists():
        print('Image processing already completed:', receipt, flush=True)
        return read_json(receipt)
    if (destination / 'freeze.json').exists():
        raise FileExistsError('Do not modify a frozen release')
    candidates = rows(source)
    if len({r['sample_id'] for r in candidates}) != len(candidates):
        raise ValueError('Duplicate candidate IDs')
    journal = work / 'image_processing.jsonl'
    completed = {}
    if journal.exists():
        # Incomplete last JSON is a failed write, never silently accepted as complete.
        for line in journal.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            completed[r['sample_id']] = r
    candidate_by_id = {r['sample_id']: r for r in candidates}
    if not completed.keys() <= candidate_by_id.keys():
        raise ValueError('Journal contains unknown samples')
    for sid, r in completed.items():
        if digest(resolve(r['image_path'])) != r['sha256']:
            raise ValueError('Previously saved crop changed: ' + sid)
    remaining = [r for r in candidates if r['sample_id'] not in completed]
    print('Native RGB crops:', len(candidates), '; already complete:', len(completed), flush=True)
    with journal.open('a', encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(lambda item: prepare_one(item, destination), remaining):
            stream.write(json.dumps(r, ensure_ascii=True) + '\n')
            completed[r['sample_id']] = r
            if len(completed) % 500 == 0:
                stream.flush()
                print('Crops saved and pixel-checked:', len(completed), '/', len(candidates), flush=True)
    result = [completed[r['sample_id']] for r in candidates]
    manifest = work / 'processed_images.csv'
    write_rows(manifest, result)
    info = {'finished_at': now(), 'input_signature': signature, 'images': len(result),
        'class_counts': dict(Counter(r['class_name'] for r in result)),
        'quality_flag_counts': dict(Counter(flag for r in result for flag in r['quality_flags'].split('|') if flag)),
        'native_rgb_crops_saved_and_pixel_compared': True,
        'manifest_sha256': digest(manifest), 'models_or_software_tests_run': False,
        'all_images_visually_reviewed': False, 'split_executed': False}
    save_json(receipt, info)
    print('Image processing completed:', manifest, flush=True)
    return info
