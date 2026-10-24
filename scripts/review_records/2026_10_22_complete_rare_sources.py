"""Record the eight actually inspected final source candidates; retain raw media."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from uav_project.common import *

OBSERVATIONS={
1:('postproduction_arrows_and_caption','Full Intel airframe photograph contains added instructional arrows; reference only under the existing no-overlay primary-cohort policy.'),
2:('partial_airframe|composite_instruction','Two annotated rear assembly views, not a complete single airframe instance.'),
3:('parts_only|composite_instruction','Propeller and motor close-up montage; no complete drone instance.'),
4:('partial_airframe|foreground_cables','Cable setup close-up with cropped airframe; no complete machine observation.'),
5:('wrong_airframe_for_candidate_label','White closed-shell quadcopter on a charging pad does not match the carbon-frame Intel Aero RTF. Page topic is not image ground truth.'),
6:('product_promotion_reference','Isolated first-generation Bebop product illustration/photo on white; original capture/render origin unverified; excluded from real-world external evaluation.'),
7:('controller_only','Skycontroller remote only; no drone target present despite image alt text.'),
8:('product_promotion_reference','Isolated indoor-hull Bebop product illustration/photo; original capture/render origin unverified; excluded from real-world external evaluation.')}

if __name__=='__main__':
    folder=OUT/'review_batches/rare_completion_v2'
    output=OUT/'drone_type_external_visual_review_completion_v2.json'
    if output.exists():raise FileExistsError('Review already recorded; preserve history')
    manifest=rows(folder/'manifest.csv')
    if {int(r['number']) for r in manifest}!=set(OBSERVATIONS):raise ValueError('Fixed batch changed')
    reviews=rows(META/'drone_type_external_review.csv');index={r['sample_id']:r for r in reviews};records=[]
    write_rows(folder/'before_apply_review.csv',reviews)
    for r in manifest:
        n=int(r['number']);flags,note=OBSERVATIONS[n]
        evidence=folder/f'originals_{(n-1)//4+1:02}.jpg'
        sid=r['sample_id']
        if digest(resolve(r['path']))!=r['sha256']:raise ValueError('Raw image changed')
        if index[sid]['visual_decision']!='pending':raise ValueError('Refuse overwrite of existing non-pending decision')
        item={'sample_id':sid,'visual_decision':'reject','candidate_label':r['candidate_label'],'label_status':'not_a_primary_experiment_sample','box_status':'not_required_excluded','source_group':'','source_group_status':'unverified','flags':flags,'note':note,'reviewer':'Codex_AI_assisted_visual_review','evidence':relative(evidence)}
        index[sid].update(item)
        records.append({**item,'image_sha256':r['sha256'],'evidence_sha256':digest(evidence),'sheet':relative(evidence),'position':(n-1)%4+1,'original_image_inspected':True})
    write_rows(META/'drone_type_external_review.csv',reviews)
    save_json(output,{'recorded_at':now(),'records':records,'new_original_images_actually_acquired':8,'new_primary_images_accepted':0,'raw_images_deleted_or_modified':0,'models_or_software_tests_run':False})
    print('Eight acquired candidates inspected and excluded from primary experiment; raw images retained.')
