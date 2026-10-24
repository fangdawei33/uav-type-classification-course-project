"""Record context for actually inspected retained originals and target crops."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from uav_project.common import META,OUT,rows,write_rows,save_json,now,relative,digest

if __name__=='__main__':
    folder=OUT/'review_batches/external_remaining_boxes_v1'
    backup=folder/'before_condition_scope.csv'
    if backup.exists():raise FileExistsError('Scope completion already recorded')
    scopes=rows(META/'drone_type_external_condition_scope.csv');oldids={r['sample_id'] for r in scopes}
    reviews=rows(META/'drone_type_external_review.csv');boxes={r['sample_id']:r for r in rows(META/'drone_type_external_box_proposals.csv')}
    batch={r['sample_id']:int(r['number']) for r in rows(folder/'manifest.csv')}
    layout={11,15,16,17,18,21}
    added=[]
    for r in reviews:
        sid=r['sample_id']
        if r['visual_decision']!='candidate_keep' or sid in oldids:continue
        b=boxes[sid];challenge=batch.get(sid) in layout
        added.append({'sample_id':sid,'condition_scope':'challenge_only' if challenge else 'standard_candidate','label_status':r['label_status'],'flags':r['flags']+('|equipment_layout' if challenge else ''),'training_approved':False,'test_approved':False,'reason':'Actually inspected target and crop; equipment-layout challenge.' if challenge else 'Actually inspected drone target and crop. Standard candidate means scene scope only, not authenticated type or independence.','evidence':b['evidence'],'evidence_sha256':b['evidence_sha256']})
    write_rows(backup,scopes);write_rows(META/'drone_type_external_condition_scope.csv',scopes+added)
    save_json(folder/'condition_completion.json',{'recorded_at':now(),'added':len(added),'records':added,'label_authentication_changed':False,'model_run':False})
    print('Completed context records:',len(added))
