"""03i. Check author bounding boxes and save sampled target crops, without models."""
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from PIL import Image, ImageDraw, ImageOps
from _bootstrap import PROJECT_ROOT
from uav_project.common import (META, OUT, rows, write_rows, resolve, relative,
                               save_json, digest, now)


def audit():
    root = resolve('data/raw/expansion/visiodect_v1/VisioDECT Dataset Upload')
    folder = OUT / 'expansion/visiodect_v1/boxes'
    if (folder / 'receipt.json').exists():
        raise FileExistsError('Annotation audit already recorded; preserve evidence')
    images = rows(META / 'visiodect_v1_images.csv')
    if not images:
        raise ValueError('Run 03g first')
    folder.mkdir(parents=True, exist_ok=True)
    annotation_hashes = {r['path']: r['sha256'] for r in rows(META / 'visiodect_v1_annotations.csv')}
    csv_boxes = defaultdict(list)
    table_problems = []
    for p in root.rglob('*.csv'):
        parts = p.relative_to(root).parts
        if len(parts) < 4 or parts[1].casefold() != 'labels':
            continue
        try:
            with p.open(encoding='utf-8-sig', newline='') as stream:
                for number, values in enumerate(csv.reader(stream), 1):
                    if len(values) != 8:
                        table_problems.append({'path': relative(p), 'row': number,
                                               'issue': 'Expected 8 author fields'});
                        continue
                    label, x, y, w, h, filename, iw, ih = values
                    try:
                        x, y, w, h, iw, ih = map(float, (x, y, w, h, iw, ih))
                        if not all(math.isfinite(v) for v in (x, y, w, h, iw, ih)):
                            raise ValueError('Nonfinite coordinates')
                    except ValueError:
                        table_problems.append({'path': relative(p), 'row': number,
                                               'issue': 'Nonnumeric author box row'})
                        continue
                    csv_boxes[(parts[0], parts[2].casefold(), Path(filename).stem.casefold())].append(
                        {'box': [x, y, x + w, y + h], 'width': iw, 'height': ih,
                         'label_text': label, 'annotation_path': relative(p),
                         'annotation_row': number, 'coordinate_format': 'author_csv_xywh'})
        except Exception as exc:
            table_problems.append({'path': relative(p), 'issue': repr(exc)})
    records = []
    for image_number, r in enumerate(images, 1):
        key = (r['source_label'], r['source_scenario'], Path(r['path']).stem.casefold())
        alternatives = []
        xml_paths = [resolve(x) for x in r['annotation_paths'].split(' | ') if x.endswith('.xml')]
        for p in xml_paths:
            try:
                tree = ET.parse(p)
                width = float(tree.findtext('.//size/width'))
                height = float(tree.findtext('.//size/height'))
                for obj in tree.findall('.//object'):
                    box = [float(obj.findtext('bndbox/' + field))
                           for field in ('xmin', 'ymin', 'xmax', 'ymax')]
                    alternatives.append({'box': box, 'width': width, 'height': height,
                        'label_text': obj.findtext('name', ''), 'annotation_path': relative(p),
                        'annotation_row': '', 'coordinate_format': 'author_voc_xyxy_coordinate_origin_unconfirmed'})
            except Exception as exc:
                table_problems.append({'path': relative(p), 'issue': repr(exc)})
        # CSV is a fallback for missing VOC files (notably all Anafi/cloudy).
        selected = alternatives if alternatives else csv_boxes.get(key, [])
        reason = ''
        if not selected:
            reason = 'no_parsed_author_box'
        elif len(selected) != 1:
            reason = 'multiple_author_boxes_require_review'
        else:
            b = selected[0]
            x1, y1, x2, y2 = b['box']
            if (b['width'], b['height']) != (int(r['width']), int(r['height'])):
                reason = 'annotation_image_dimensions_mismatch'
            elif not all(math.isfinite(v) for v in b['box']):
                reason = 'nonfinite_box'
            elif not (0 <= x1 < x2 <= int(r['width']) and 0 <= y1 < y2 <= int(r['height'])):
                reason = 'box_out_of_bounds_or_empty'
        item = {**r, 'annotation_check': reason or 'valid_numeric_single_author_box',
                'annotation_status_is_not_visual_authentication': True,
                'selected_annotation_path': '', 'annotation_label_text': '',
                'box_xyxy': '', 'box_width': '', 'box_height': '',
                'csv_voc_max_coordinate_difference': ''}
        if len(selected) == 1:
            b = selected[0]
            x1, y1, x2, y2 = b['box']
            item.update(selected_annotation_path=b['annotation_path'],
                selected_annotation_sha256=annotation_hashes[b['annotation_path']],
                annotation_row=b['annotation_row'], annotation_label_text=b['label_text'],
                box_xyxy=','.join(str(v) for v in b['box']),
                box_width=x2-x1, box_height=y2-y1, coordinate_format=b['coordinate_format'])
            csv_alt = csv_boxes.get(key, [])
            if alternatives and len(csv_alt) == 1:
                item['csv_voc_max_coordinate_difference'] = max(abs(a-c) for a,c in zip(b['box'], csv_alt[0]['box']))
        records.append(item)
        if image_number % 5000 == 0:
            print('Annotation records checked:', image_number, '/', len(images), flush=True)
    manifest = META / 'visiodect_v1_author_box_audit.csv'
    write_rows(manifest, records)
    write_rows(folder / 'table_problems.csv', table_problems, ['path', 'row', 'issue'])
    groups = defaultdict(list)
    for r in records:
        if r['annotation_check'] == 'valid_numeric_single_author_box':
            groups[(r['source_label'], r['source_scenario'])].append(r)
    sampled = []
    for group_number, (key, members) in enumerate(sorted(groups.items()), 1):
        members.sort(key=lambda r: r['source_member'])
        indices = sorted({round(i * (len(members)-1) / 7) for i in range(8)})
        sheet = Image.new('RGB', (1600, 1640), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), key[0] + ' / ' + key[1] + ' | Original and author-box crop', fill='black')
        page = folder / f'crops_{group_number:02}.jpg'
        for j, index in enumerate(indices):
            r = members[index]
            with Image.open(resolve(r['path'])) as source:
                image = source.convert('RGB')
                x1,y1,x2,y2 = map(float, r['box_xyxy'].split(','))
                box = (math.floor(x1),math.floor(y1),math.ceil(x2),math.ceil(y2))
                crop = image.crop(box)
                marked = image.copy()
                ImageDraw.Draw(marked).rectangle(box, outline='red', width=max(2,image.width//500))
                original = ImageOps.contain(marked, (760, 330))
                enlarged = ImageOps.contain(crop, (760, 330), method=Image.Resampling.NEAREST)
            y = 35 + j * 200
            left = ImageOps.contain(original,(780,165))
            right = ImageOps.contain(enlarged,(780,165),method=Image.Resampling.NEAREST)
            sheet.paste(left,((800-left.width)//2,y))
            sheet.paste(right,(800+(800-right.width)//2,y))
            draw.text((8,y+167),r['source_record_id']+' | '+str(box),fill='black')
            sampled.append({**r,'sheet':relative(page),'position':j+1,
                'selection_rule':'8 evenly spaced sorted-path indices per valid source-label/scenario'})
        sheet.save(page,quality=95)
    write_rows(folder / 'sampled_crops.csv',sampled)
    valid = [r for r in records if r['annotation_check']=='valid_numeric_single_author_box']
    differences = [r for r in valid if r['csv_voc_max_coordinate_difference']!='' and float(r['csv_voc_max_coordinate_difference'])>2]
    result={'finished_at':now(),'images':len(records),'annotation_check_counts':dict(Counter(r['annotation_check'] for r in records)),
        'valid_numeric_boxes_by_class':dict(Counter(r['source_label'] for r in valid)),
        'target_short_side_below_24_pixels':sum(min(float(r['box_width']),float(r['box_height']))<24 for r in valid),
        'csv_voc_difference_over_2_pixels':len(differences),'table_problem_rows':len(table_problems),
        'sampled_crop_sheets':len(groups),'sampled_images':len(sampled),'sampled_crops_visually_inspected':False,
        'class_mapping_approved':False,'training_approved':False,'models_or_software_tests_run':False,
        'manifest_sha256':digest(manifest),'coordinate_origin_note':'VOC origin convention unconfirmed; sampled previews use direct source xyxy without shifting. No final crop release.'}
    save_json(folder/'receipt.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    audit()
