"""03h. Save deterministic, unlabelled review sheets for acquired expansion data."""
from collections import defaultdict
from PIL import Image, ImageOps, ImageDraw
from _bootstrap import PROJECT_ROOT
import hashlib
from uav_project.common import META, OUT, rows, write_rows, resolve, relative, save_json, now, digest


def prepare():
    folder = OUT / 'expansion/visiodect_v1/review'
    if (folder / 'manifest.csv').exists():
        raise FileExistsError('Review preview already exists; preserve the original evidence')
    groups = defaultdict(list)
    records = rows(META / 'visiodect_v1_images.csv')
    if not records:
        # This bounded visual sample can run while the independent full-file
        # inventory is reading all 59,795 files. It does not replace that audit.
        root = resolve('data/raw/expansion/visiodect_v1/VisioDECT Dataset Upload')
        for p in root.glob('*/images/*/*'):
            if p.suffix.casefold() not in {'.jpg', '.jpeg', '.png'}:
                continue
            parts = p.relative_to(root).parts
            name = p.relative_to(root.parent).as_posix()
            sid = 'visio_path_' + hashlib.sha256(name.encode()).hexdigest()[:20]
            records.append({'source_record_id': sid, 'source_member': name,
                'source_label': parts[0], 'source_scenario': parts[2].casefold(),
                'path': relative(p)})
    for r in records:
        groups[(r['source_label'], r['source_scenario'])].append(r)
    if not groups:
        raise ValueError('Run 03g to inventory actual acquired images first')
    folder.mkdir(parents=True, exist_ok=True)
    selected = []
    for number, (key, members) in enumerate(sorted(groups.items()), 1):
        members.sort(key=lambda r: r['source_member'])
        indices = sorted({round(i * (len(members) - 1) / 7) for i in range(8)})
        page = Image.new('RGB', (1600, 1560), 'white')
        draw = ImageDraw.Draw(page)
        draw.text((12, 8), 'Author label: ' + key[0] + '; scenario: ' + key[1], fill='black')
        target = folder / f'group_{number:02}.jpg'
        for j, index in enumerate(indices):
            r = members[index]
            x, y = (j % 2) * 800, 35 + (j // 2) * 380
            with Image.open(resolve(r['path'])) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                thumb = ImageOps.contain(image, (780, 320))
            page.paste(thumb, (x + (800 - thumb.width) // 2, y + (325 - thumb.height) // 2))
            draw.text((x + 8, y + 328), r['source_record_id'], fill='black')
            draw.text((x + 8, y + 345), r['source_member'].split('/')[-1][:100], fill='black')
            draw.text((x + 8, y + 362), 'Original only; no automatic identity approval.', fill='black')
            selected.append({**r, 'sha256': digest(resolve(r['path'])), 'sheet': relative(target), 'position': j + 1,
                             'selection_rule': '8 evenly spaced sorted-path indices per source-label/scenario'})
        page.save(target, quality=95)
    write_rows(folder / 'manifest.csv', selected)
    save_json(folder / 'preparation.json', {'prepared_at': now(), 'sheets': len(groups),
        'sampled_originals': len(selected), 'visually_inspected': False,
        'sampling_is_not_full_dataset_review': True, 'class_mapping_approved': False,
        'no_models_or_tests_run': True})
    print('Prepared review sheets:', len(groups), 'in', folder, flush=True)


if __name__ == '__main__':
    prepare()
