"""04k. Export one dataset handoff index, preserving frozen images and evidence."""
from collections import Counter
import shutil
from _bootstrap import PROJECT_ROOT
from uav_project.common import ROOT, OUT, META, rows, read_json, write_rows, save_json, digest, now, relative, resolve


def export():
    root = ROOT/'data/final/uav_type_v3'
    destination = ROOT/'data/delivery/uav_dataset_v3'
    if (destination/'receipt.json').exists():
        raise FileExistsError('Delivery already exists; preserve its inventory')
    audit = read_json(OUT/'expansion/processing_v3/release_audit.json')
    if audit['issues']:
        raise ValueError('Resolve recorded file-integrity issues first')
    old_raw = rows(ROOT/'data/final/uav_type_v2/raw_acquisition_inventory.csv')
    expansion = rows(META/'visiodect_v1_images.csv')
    clean = rows(root/'clean_manifest.csv')
    by_id = {r['sample_id']:r for r in clean}
    excluded = {r['sample_id']:r['reason'] for r in rows(OUT/'expansion/visiodect_v1/candidates_sampled_review/excluded_expansion.csv')}
    excluded.update({r['sample_id']:r['reason'] for r in rows(root/'excluded_visual_quality.csv')})
    excluded.update({r['sample_id']:r['reason'] for r in rows(root/'excluded_pixel_duplicates.csv')})
    inventory = []
    for r in old_raw:
        action = 'primary_clean_crop' if r['sample_id'] in by_id else r['release_action']
        inventory.append({**r, 'current_role':action,
            'clean_path':by_id.get(r['sample_id'],{}).get('image_path',''),
            'current_main_label':by_id.get(r['sample_id'],{}).get('label','')})
    for r in expansion:
        inventory.append({'sample_id':r['sample_id'],'dataset':'VisioDECT author version 1',
            'data_origin':'author_real_video_frames','path':r['path'],'sha256':r['sha256'],
            'source_url':r['source_url'],'source_label':r['source_label'],'class_name':r['source_label'],
            'source_member':r['source_member'],'source_scenario':r['source_scenario'],
            'license':'CC BY-NC-SA 4.0 as published by author',
            'is_self_crawled':False,'is_synthetic':False,
            'current_role':'primary_clean_crop' if r['sample_id'] in by_id else 'excluded_from_primary',
            'current_exclusion_reason':excluded.get(r['sample_id'],''),
            'clean_path':by_id.get(r['sample_id'],{}).get('image_path',''),
            'current_main_label':by_id.get(r['sample_id'],{}).get('label','')})
    if len({r['sample_id'] for r in inventory})!=len(inventory):
        raise ValueError('Raw sample IDs overlap')
    raw_by_id={r['sample_id']:r for r in inventory}
    if any(raw_by_id[r['sample_id']]['sha256']!=r['raw_sha256'] for r in clean):
        raise ValueError('Clean-parent provenance differs')
    if any(r['current_role']=='excluded_from_primary' and not r['current_exclusion_reason'] for r in inventory):
        raise ValueError('Expansion exclusion missing reason')
    destination.mkdir(parents=True,exist_ok=True)
    write_rows(destination/'raw_acquisition_inventory.csv',inventory)
    for name in ['clean_manifest.csv','class_map.json','class_counts.csv','split_feasibility.csv',
                 'excluded_visual_quality.csv','excluded_pixel_duplicates.csv']:
        shutil.copyfile(root/name,destination/name)
    for name in ['external_weak_labels.csv','supplementary_identity.csv','excluded_and_reference.csv']:
        shutil.copyfile(ROOT/'data/final/uav_type_v2'/name,destination/('web_'+name))
    shutil.copyfile(OUT/'expansion/visiodect_v1/candidates_sampled_review/excluded_expansion.csv',destination/'excluded_expansion.csv')
    shutil.copyfile(OUT/'expansion/processing_v3/release_audit.json',destination/'data_integrity_audit.json')
    shutil.copyfile(ROOT/'reports/expanded_data_report.md',destination/'data_processing_report.md')
    save_json(destination/'source_evidence_index.json',{
        'original_seven_class_snapshot':'data/final/uav_type_v2/',
        'original_download_and_crawl_inventory':'data/final/uav_type_v2/raw_acquisition_inventory.csv',
        'crawler_modules':['scripts/02_crawl_external.py','scripts/02b_crawl_articles.py'],
        'crawl_snapshots':'outputs/crawl_pages/',
        'expansion_download_receipt':'outputs/expansion/visiodect_v1/download_receipt.json',
        'expansion_all_original_files':'outputs/expansion/visiodect_v1/all_acquired_files.csv',
        'expansion_annotation_audit':'data/metadata/visiodect_v1_author_box_audit.csv',
        'sampled_visual_observations':'outputs/expansion/visiodect_v1/visual_sample/observations.csv',
        'quality_visual_observations':'outputs/expansion/processing_v3/quality_visual_observations.csv',
        'crop_similarity_constraints':'outputs/expansion/processing_v3/crop_similarity_constraints.csv',
        'self_crawled_originals':179,'public_dataset_download_is_self_crawl':False,
        'dataset_license_review_note':'Author licenses/source evidence retained; no external publication performed.'})
    files=[{'path':relative(p),'sha256':digest(p)} for p in sorted(destination.iterdir()) if p.is_file()]
    write_rows(destination/'files_sha256.csv',files)
    result={'created_at':now(),'raw_originals':len(inventory),'primary_classes':12,
        'primary_clean_images':len(clean),'web_weak_label_images':len(rows(destination/'web_external_weak_labels.csv')),
        'web_supplementary_identity_images':len(rows(destination/'web_supplementary_identity.csv')),
        'raw_source_counts':dict(Counter(r['dataset'] for r in inventory)),
        'raw_manifest_sha256':digest(destination/'raw_acquisition_inventory.csv'),
        'clean_manifest_sha256':digest(destination/'clean_manifest.csv'),
        'clean_manifest_matches_frozen_v3':digest(destination/'clean_manifest.csv')==digest(root/'clean_manifest.csv'),
        'file_inventory_sha256':digest(destination/'files_sha256.csv'),
        'image_paths_are_relative_to_project_root':True,'primary_crop_folder':'data/final/uav_type_v3/images/',
        'active_training_split':False,'all_12_class_independent_test_ready':False,
        'data_preparation_delivery_complete':True,'full_visual_expert_review_complete':False,
        'models_or_software_tests_run':False}
    save_json(destination/'receipt.json',result)
    print('Dataset delivery ready:',destination,flush=True)
    print('Primary:',len(clean),'; labels: 12; raw originals:',len(inventory),'; web weak:',result['web_weak_label_images'],flush=True)


if __name__=='__main__':
    export()
