"""Export visually screened external image crops with source lineage.

This creates candidate derivatives only. It does not approve model labels,
license redistribution, source independence, or train/test membership.
"""
from collections import Counter
from PIL import Image, ImageOps
from .common import META, OUT, rows, resolve, digest, relative, write_rows, save_json, now


def export_external_candidate_crops():
    raw = {r['sample_id']:r for name in ('drone_type_external_raw.csv','drone_type_article_raw.csv')
           for r in rows(META/name) if r.get('sample_scope','external_candidate') != 'reference_only'}
    review = {r['sample_id']:r for r in rows(META/'drone_type_external_review.csv')}
    boxes = {r['sample_id']:r for r in rows(META/'drone_type_external_box_proposals.csv')}
    scopes = {r['sample_id']:r for r in rows(META/'drone_type_external_condition_scope.csv')}
    output = META/'drone_type_external_visual_candidate_crops.csv'
    destination = resolve('data/processed/drone_type_external/visual_candidate_crops')
    records=[]
    missing_geometry=[]
    newly_created=0
    for sid,r in raw.items():
        decision=review.get(sid,{})
        if decision.get('visual_decision') != 'candidate_keep': continue
        box=boxes.get(sid)
        if not box or box.get('status') != 'reviewed_geometry_only':
            missing_geometry.append(sid)
            continue
        if box['image_sha256'] != r['sha256'] or digest(resolve(r['path'])) != r['sha256']:
            raise ValueError('Parent image hash mismatch: '+sid)
        if digest(resolve(r['detail_snapshot'])) != r['detail_sha256']:
            raise ValueError('Source snapshot hash mismatch: '+sid)
        with Image.open(resolve(r['path'])) as original:
            w,h=ImageOps.exif_transpose(original).size
        x1,y1,x2,y2=(int(box[k]) for k in ('x_min','y_min','x_max_exclusive','y_max_exclusive'))
        if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('Box out of bounds: '+sid)
        target=destination/(sid+'.png');target.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(resolve(r['path'])) as source:
            crop=ImageOps.exif_transpose(source).convert('RGB').crop((x1,y1,x2,y2))
        if target.exists():
            with Image.open(target) as existing:
                if existing.convert('RGB').tobytes()!=crop.tobytes() or existing.size!=crop.size:
                    raise ValueError('Existing derivative differs; refusing overwrite: '+sid)
        else:
            tmp=target.with_suffix('.tmp.png');crop.save(tmp);tmp.replace(target)
            newly_created+=1
        scope=scopes.get(sid,{})
        records.append({'sample_id':sid,'label':r['candidate_label'],'class_name':r['candidate_class_name'],
            'crop_path':relative(target),'crop_sha256':digest(target),'crop_width':crop.width,'crop_height':crop.height,
            'parent_path':r['path'],'parent_sha256':r['sha256'],'source_url':r['source_url'],
            'detail_snapshot':r['detail_snapshot'],'detail_sha256':r['detail_sha256'],
            'license':r.get('license',''),'license_status':r.get('license_review','pending'),
            'coordinate_space':box['coordinate_space'],'crop_xyxy':f'{x1},{y1},{x2},{y2}',
            'box_status':'reviewed_geometry_only_not_expert_approved','box_evidence':box['evidence'],
            'box_evidence_sha256':box['evidence_sha256'],'visual_decision':decision['visual_decision'],
            'label_status':decision.get('label_status',''),'condition_scope':scope.get('condition_scope','unresolved'),
            'condition_flags':decision.get('flags',''),'source_group':decision.get('source_group',''),
            'source_group_status':decision.get('source_group_status','unverified'),
            'source_independence_verified':False,'label_authenticated':False,
            'training_approved':False,'test_approved':False})
    if len(records)!=len({r['sample_id'] for r in records}):raise ValueError('Duplicate sample IDs')
    write_rows(output,records)
    receipt={'created_at':now(),'images':len(records),
        'class_counts':dict(Counter(r['class_name'] for r in records)),
        'condition_scope_counts':dict(Counter(r['condition_scope'] for r in records)),
        'manifest_path':relative(output),'manifest_sha256':digest(output),
        'derived_crop_files_created':newly_created,'derived_crop_files_verified':len(records),'raw_files_modified':0,
        'visually_kept_without_geometry':len(missing_geometry),'formal_clean_manifest':False,
        'training_ready':False,'external_test_ready':False,
        'note':'These are source-linked crops of visual candidates. Labels, license redistribution, source independence and evaluation split remain unapproved.'}
    save_json(OUT/'drone_type_external_candidate_crops.json',receipt)
    print('Exported external candidate crops:',len(records),'to',output)
    print('Labels/scopes remain unapproved; no train/test split or model run.')
    return receipt
