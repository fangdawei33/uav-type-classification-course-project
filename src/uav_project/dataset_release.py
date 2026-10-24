"""Build one fixed, local dataset version from already collected images.

No collection, model or train/test selection. Author labels, weak webpage
labels and unresolved identities remain explicitly different kinds of evidence.
"""
import csv
import hashlib
import json
import shutil
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image,ImageOps
from .common import ROOT,META,OUT,rows,write_rows,digest,save_json,now,relative,resolve

SLUGS=['bebop','phantom_3','phantom_4_pro','emax','intel_aero_rtf','mambo','yh_19hw']
VERSION='uav_type_v1'

def pixel_hash(path):
    with Image.open(path) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
        return hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()

def build_dataset_release():
    destination=ROOT/'data/final'/VERSION
    if not destination.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Release outside workspace')
    if (destination/'freeze.json').exists():raise FileExistsError('Dataset version already frozen; preserve it and create a new version for future changes')
    destination.mkdir(parents=True,exist_ok=True)
    public=rows(META/'drone_type_set_candidate_crops.csv')
    external=rows(META/'drone_type_external_visual_candidate_crops.csv')
    if len(public)!=2720 or len(external)!=140:raise ValueError('Resolve incomplete crop export before freezing')
    geometry=read_json(OUT/'review_batches/external_remaining_boxes_v1/geometry_record.json')
    if not geometry.get('geometry_review_complete'):raise ValueError('New crops not yet inspected')
    crop_receipt=read_json(OUT/'drone_type_external_candidate_crops.json')
    if crop_receipt['visually_kept_without_geometry']!=0:raise ValueError('Retained images still lack target geometry')
    audit=read_json(OUT/'drone_type_external_integrity_current.json')
    if not audit['all_checked_files_and_decisions_match']:raise ValueError('Resolve file/evidence mismatch')
    raw={r['sample_id']:r for name in ['drone_type_external_raw.csv','drone_type_article_raw.csv'] for r in rows(META/name)}
    review={r['sample_id']:r for r in rows(META/'drone_type_external_review.csv')}
    final=[];quarantine=[];duplicates=[];pixels={};copied=0;hashes={}
    def materialize(source,expected,subset,label,sid):
        nonlocal copied
        source=resolve(source)
        if digest(source)!=expected:raise ValueError('Crop changed: '+sid)
        pix=pixel_hash(source)
        folder=destination/'images'/subset/(SLUGS[int(label)] if str(label)!='' else 'ezpilot_mapping_unresolved')
        folder.mkdir(parents=True,exist_ok=True)
        target=folder/(sid+'.png')
        if target.exists():
            if digest(target)!=expected:raise ValueError('Existing output changed')
        else:shutil.copyfile(source,target);copied+=1
        if digest(target)!=expected:raise ValueError('Copy mismatch')
        hashes[relative(target)]=expected
        return target,pix
    for r in sorted(public,key=lambda r:r['sample_id']):
        sid=r['sample_id'];target,pix=materialize(r['clean_path'],r['clean_sha256'],'public',r['label'],sid)
        prior=pixels.get(pix)
        if prior:
            reason='same_pixels_same_label' if prior['label']==str(r['label']) else 'same_pixels_conflicting_label'
            duplicates.append({'sample_id':sid,'duplicate_of':prior['sample_id'],'relation':reason,'action':'excluded_from_primary_pool','pixel_sha256':pix})
            quarantine.append({'sample_id':sid,'origin':'public','candidate_label':r['label'],'class_name':r['class_name'],'path':relative(target),'reason':reason,'release_action':'isolated_not_primary'})
            if reason=='same_pixels_conflicting_label':raise ValueError('Public label conflict requires explicit resolution')
            continue
        item={'sample_id':sid,'label':r['label'],'class_name':r['class_name'],'image_path':relative(target),'sha256':r['clean_sha256'],'pixel_sha256':pix,'raw_path':r['parent_path'],'raw_sha256':r['parent_sha256'],'crop_xyxy':r['crop_xyxy'],'label_basis':'dataset_author_label','constraint_group':r['conservative_constraint_group'],'source_identity_verified':False,'release_pool':'public_training_pool','split':'unassigned','source_independent_evaluation_approved':False}
        final.append(item);pixels[pix]=item
    # The two files below show the same photographed scene at different encodings/resolutions.
    # This relation was actually inspected in original/crop sheets 4/11 and 2/6.
    batch={int(r['number']):r for r in rows(OUT/'review_batches/external_remaining_boxes_v1/manifest.csv')}
    known_duplicate={batch[88]['sample_id']:batch[30]['sample_id']}
    ext_final=[];supplement=[]
    for r in sorted(external,key=lambda r:r['sample_id']):
        sid=r['sample_id'];mapping_unresolved=int(r['label'])==3
        label='' if mapping_unresolved else r['label']
        target,pix=materialize(r['crop_path'],r['crop_sha256'],'supplementary' if mapping_unresolved else 'external',label,sid)
        record=raw[sid]
        item={'sample_id':sid,'label':label,'class_name':r['class_name'] if not mapping_unresolved else 'EZPilot; Emax class mapping unresolved','image_path':relative(target),'sha256':r['crop_sha256'],'pixel_sha256':pix,'raw_path':r['parent_path'],'raw_sha256':r['parent_sha256'],'crop_xyxy':r['crop_xyxy'],'source_url':record['source_url'],'source_title':record.get('title',''),'source_snapshot':record['detail_snapshot'],'source_snapshot_sha256':record['detail_sha256'],'source_author':record.get('author',''),'license':record.get('license',''),'label_basis':review[sid].get('label_status','source_metadata_label_not_independently_authenticated'),'constraint_group':review[sid].get('source_group',''),'condition_scope':r['condition_scope'],'source_identity_verified':False,'release_pool':'supplementary_identity' if mapping_unresolved else 'external_weak_label_pool','split':'unassigned','source_independent_evaluation_approved':False}
        if mapping_unresolved:
            supplement.append(item)
            quarantine.append({'sample_id':sid,'origin':'external','candidate_label':r['label'],'class_name':r['class_name'],'path':relative(target),'reason':'EZPilot_to_Emax_mapping_unresolved','release_action':'supplementary_only_no_seven_class_label'})
            continue
        prior=pixels.get(pix)
        other=known_duplicate.get(sid) or (prior['sample_id'] if prior else None)
        if other:
            reason='visually_confirmed_resized_same_photo' if sid in known_duplicate else 'exact_rgb_crop_duplicate'
            duplicates.append({'sample_id':sid,'duplicate_of':other,'relation':reason,'action':'excluded_from_primary_pool','pixel_sha256':pix})
            quarantine.append({'sample_id':sid,'origin':'external','candidate_label':r['label'],'class_name':r['class_name'],'path':relative(target),'reason':reason,'release_action':'isolated_not_primary'})
            continue
        ext_final.append(item);pixels[pix]=item
    public_excluded=[r for r in rows(META/'drone_type_set_visual_cohort.csv') if r['visual_decision']!='keep']
    for r in public_excluded:
        quarantine.append({'sample_id':r['sample_id'],'origin':'public','candidate_label':r['label'],'class_name':r['class_name'],'path':r['path'],'reason':r['primary_reason'],'release_action':'excluded_from_primary'})
    for sid,r in raw.items():
        v=review.get(sid)
        if not v:
            quarantine.append({'sample_id':sid,'origin':'external_reference','candidate_label':r.get('candidate_label',''),'class_name':r.get('candidate_class_name',''),'path':r['path'],'reason':'reference_only_not_experiment','release_action':'reference_only'})
        elif v['visual_decision']!='candidate_keep':
            reason='unresolved_type_excluded_from_v1' if v['visual_decision']=='pending' else 'visual_rejection'
            quarantine.append({'sample_id':sid,'origin':'external','candidate_label':r['candidate_label'],'class_name':r['candidate_class_name'],'path':r['path'],'reason':reason+': '+v.get('note',''),'release_action':'excluded_from_v1'})
    write_rows(destination/'public_train_pool.csv',final)
    write_rows(destination/'external_weak_labels.csv',ext_final)
    write_rows(destination/'supplementary_identity.csv',supplement)
    write_rows(destination/'excluded_and_reference.csv',quarantine)
    write_rows(destination/'duplicate_relations.csv',duplicates)
    inputs=['drone_type_set_candidate_crops.csv','drone_type_set_visual_cohort.csv','drone_type_set_reviewed_correlation_constraints.csv','drone_type_external_visual_candidate_crops.csv','drone_type_external_raw.csv','drone_type_article_raw.csv','drone_type_external_review.csv','drone_type_external_box_proposals.csv','drone_type_external_reviewed_groups.csv','drone_type_external_condition_scope.csv','cranfield_synthetic_classification_auxiliary_train.csv','cranfield_synthetic_classification_auxiliary_val.csv']
    metadata=destination/'metadata';metadata.mkdir(exist_ok=True)
    input_hashes={}
    for name in inputs:
        path=META/name;input_hashes[relative(path)]=digest(path);shutil.copyfile(path,metadata/name)
    class_names=read_json(META/'drone_type_set_classes.json')['classes']
    save_json(destination/'class_map.json',{'classes':[{'label':i,'name':n,'folder':SLUGS[i]} for i,n in enumerate(class_names)],'auxiliary_classes_separate':True})
    summary={'version':VERSION,'frozen_at':now(),'data_assembly_complete':True,'pending_geometry':0,'pending_decisions_in_release':0,'public_train_pool_images':len(final),'external_weak_label_images':len(ext_final),'supplementary_mapping_images':len(supplement),'all_materialized_target_crops':len(hashes),'excluded_and_reference_records':len(quarantine),'duplicate_relations':len(duplicates),'public_class_counts':dict(Counter(r['class_name'] for r in final)),'external_weak_label_class_counts':dict(Counter(r['class_name'] for r in ext_final)),'raw_files_modified':0,'model_run':False,'software_tests_run':False,'split_executed':False,'training_pool_policy':'Author labels accepted for supervised fitting; source authentication is not asserted. No source-independent validation approved by this release.','external_policy':'Web metadata weak labels. Do not describe this pool as authenticated, independent seven-class final test.','synthetic_policy':'7998 synthetic samples remain separate; original train/val manifests copied, existing images not duplicated.','image_manifest_paths':['public_train_pool.csv','external_weak_labels.csv','supplementary_identity.csv'],'input_hashes':input_hashes}
    save_json(destination/'dataset_summary.json',summary)
    public_table='\n'.join('| '+class_names[i]+' | '+str(sum(int(r['label'])==i for r in final))+' | '+str(sum(str(r['label'])==str(i) for r in ext_final))+' |' for i in range(7))
    readme=f'''# 无人机图像数据集 v1（已固定的数据整理版本）

固定日期：2026-10-22。原始数据不覆盖；清洗、裁剪、标签、来源和排除记录均可追溯。当前目录是真实生成的图片与CSV，不是仅有框架。**本版本的数据整理已结束**，未启动下一步模型训练或划分。

## 交付规模

- 作者标签训练池：{len(final)}张，七类；路径public_train_pool.csv及images/public/。
- 网页弱标签外部池：{len(ext_final)}张；路径external_weak_labels.csv及images/external/。不作为已认证的来源独立七类测试集。
- EZPilot补充身份图：{len(supplement)}张；路径supplementary_identity.csv及images/supplementary/，不强行赋Emax标签。
- 实际固定目标裁剪：{len(hashes)}张，包含明确隔离的重复派生图；原有2720公开裁剪和140外部裁剪的像素/文件身份均已核对。
- 排除/参考清单：{len(quarantine)}条，见excluded_and_reference.csv；含17张未确定型号、8张外部视觉排除和6张身份参考，以及公开排除、映射未定和重复关系。排除不删除原图。
- 额外合成辅助：7998张，四类开发/留出各3999；清单在metadata/，原图仍在data/raw/auxiliary/，不混入七类数量。

| 类别 | 作者标签训练池 | 网页弱标签外部池 |
|---|---:|---:|
{public_table}

## 目录和字段

images/按来源池和类别分目录。CSV的image_path相对整个D:/ml_hw项目根目录，标签与class_map.json一致；保留sha256、RGB像素哈希、原图路径/哈希、目标框及来源相关性字段。外部CSV另外保留来源网页、作者、页面快照哈希和标签依据。metadata/保存生成时的输入台账快照；freeze.json固定交付文件及图片清单，不回写历史记录。

## 标签和来源的实际边界

作者训练池采用原数据集作者标签，经过图像内容筛选，但没有声称逐台设备身份得到专家认证。它可以作为下一步监督学习拟合的数据池；本次并未执行7:3划分。原清洗协议的严格来源认证门槛继续约束“来源独立评测”的说法，不再阻止训练用图像整理交付。该使用范围调整在本文件明确记录，未将unknown改成confirmed。

网页外部池采用来源页面类别/文字支持的弱标签，几何审核只确认裁剪对象。Emax映射未解决的4张已单独放置；Intel仅有一张开箱挑战图，缺少可宣称完整七类独立测试的证据。17张型号未明样本在本版本明确排除，不需要继续等待它们才能结束v1整理。后续修订采用新版本，不回改此版本。

训练/验证/最终测试划分属于下一步实验协议。源内相关帧和同一文章照片必须一起处理，未知来源不能随机伪装成独立飞行。不得把约1:1的四类合成实验称为七类7:3主实验；也不得将网页弱标签池成绩直接称为已认证真实泛化。当前没有通过重新命名文件改变这些事实。

## 已完成操作

剩余109张外部原图和对应裁剪均实际查看，外部140张暂留图全部完成框几何，缺框0。已检查全部交付图片解码/像素、复制文件SHA256和标签范围，生成排除/重复关系及固定版本回执。这里只做数据处理，不运行本地软件测试、模型推理、训练或评价；未生成PDF。

可复现的分步骤入口为scripts/03f_review_external_boxes.py（本轮固定批次证据与坐标）、scripts/04d_export_external_candidate_crops.py（原图派生裁剪）、scripts/04e_build_dataset_release.py（固定数据目录与清单）。已完成的带日期批次不要重复覆盖。
'''
    (destination/'README.md').write_text(readme,encoding='utf-8')
    frozen_files={relative(p):digest(p) for p in destination.rglob('*') if p.is_file() and 'images' not in p.relative_to(destination).parts}
    save_json(destination/'freeze.json',{'version':VERSION,'frozen_at':now(),'file_hashes':frozen_files,'image_hashes':hashes,'data_assembly_complete':True,'independent_final_test_approved':False,'labels_and_source_limitations_preserved':True})
    save_json(OUT/'dataset_release_current.json',{'version':VERSION,'folder':relative(destination),'summary':relative(destination/'dataset_summary.json'),'freeze_sha256':digest(destination/'freeze.json'),'data_assembly_complete':True,'model_run':False})
    print('Dataset version frozen:',destination)
    print('Public train pool:',len(final),'External weak-label pool:',len(ext_final),'Supplementary:',len(supplement),'Excluded/reference:',len(quarantine),'Duplicates:',len(duplicates))
    return summary


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
