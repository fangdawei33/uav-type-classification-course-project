"""Supplement conservative source/correlation links visible in reviewed metadata."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from uav_project.common import META,OUT,rows,write_rows,resolve,digest,save_json,relative,now,protect_frozen
SNAPSHOT=OUT/'drone_type_external_visual_review_group_resolution_2026-10-22.json'
LINKS={
'dte_c70a49208cf8b52938f7':('ext_bebop_weiwei_20141215','同一作者同日Bebop工作台照片序列；非独立飞行。'),
'dte_9fee02965c88b38dd66d':('ext_bebop_weiwei_20141215','同一作者同日Bebop工作台照片序列；非独立飞行。'),
'dte_693feff92340becdbc20':('ext_bebop_weiwei_20141215','同一作者同日Bebop工作台照片序列；非独立飞行。'),
'dte_3f7973b715f67ffb04cc':('ext_p3_sheba_20180801','同一作者同日Flickr无人机系列；不得跨划分。'),
'dte_2295c047de53d7b2d867':('ext_p4_cpplus_20170225','同一CP+2017展会摄影序列；不得跨划分。'),
'dte_7fa0e1d1d9bf42722228':('ext_p4_mavic_comparison_20241019','同一Phantom/Mavic产品对比文章四图；不得跨划分。'),
'dte_dad7c77efc3a6076ac10':('ext_p4_mavic_comparison_20241019','同一Phantom/Mavic产品对比文章四图；不得跨划分。'),
'dte_0d9263b4a3e4c472a9c7':('ext_p4_mavic_comparison_20241019','同一Phantom/Mavic产品对比文章四图；不得跨划分。'),
'dte_8fbeefe0bc790103bc21':('ext_p4_mavic_comparison_20241019','同一Phantom/Mavic产品对比文章四图；不得跨划分。'),
'dte_788242f54dca5f24e8be':('ext_p4_zlea_20250227','同一作者同日Phantom4/Pro对比组；不得跨划分。'),
'dte_b96708a837c166ecbd17':('ext_p4_zlea_20250227','同一作者同日Phantom4/Pro对比组，且有提取子图；不得跨划分。'),
'dte_39e91973f52200d71f28':('ext_bebop_pesce_20150106','同一作者同日Bebop照片组；不得跨划分。'),
'dte_5043f574e0181fdedfb2':('ext_bebop_ifa_20150904','同一IFA2015展会拍摄序列；不得跨划分。')}
def record():
 protect_frozen()
 if SNAPSHOT.exists():raise FileExistsError(SNAPSHOT)
 raw={r['sample_id']:r for name in ('drone_type_external_raw.csv','drone_type_article_raw.csv') for r in rows(META/name)}
 review=rows(META/'drone_type_external_review.csv');before_review=[dict(r) for r in review];byid={r['sample_id']:r for r in review}
 groups=rows(META/'drone_type_external_reviewed_groups.csv');grouped={r['sample_id'] for r in groups}
 observations=[]
 for sid,(group,basis) in LINKS.items():
  r=raw[sid];u=byid[sid]
  if u.get('source_group') and u['source_group']!=group:raise ValueError('Conflicting group: '+sid)
  u.update({'source_group':group,'source_group_status':'reviewed_correlation_constraint'})
  if sid not in grouped:
   groups.append({'sample_id':sid,'source_group':group,'status':'reviewed_correlation_constraint','basis':basis,
      'evidence':u['evidence'].split(';')[0],'source_url':r['source_url'],'detail_snapshot':r['detail_snapshot'],
      'independent_from_public':'unverified','original_flight_id':''})
   grouped.add(sid)
  ev=u['evidence'].split(';')[0]
  observations.append({**u,'sheet':ev,'evidence_sha256':digest(resolve(ev)),'image_sha256':r['sha256'],
      'detail_snapshot':r['detail_snapshot'],'detail_sha256':r['detail_sha256'],
      'correlation_only':True,'source_independence_confirmed':False})
 for name,data in [('review',before_review),('groups',rows(META/'drone_type_external_reviewed_groups.csv'))]:
  dest=OUT/f'review_batches/external_resolution_2026-10-22/before_group_resolution_{name}.csv'
  if dest.exists():raise FileExistsError(dest)
  write_rows(dest,data)
 write_rows(META/'drone_type_external_review.csv',[byid[r['sample_id']] for r in review])
 write_rows(META/'drone_type_external_reviewed_groups.csv',groups)
 save_json(SNAPSHOT,{'recorded_at':now(),'records':observations,'review_type':'metadata-linked correlation constraints; not source authentication',
   'source_independence_confirmed':False,'original_flight_ids_invented':False})
 print('Linked',len(LINKS),'images into existing conservative source cohorts; independence unverified.')
if __name__=='__main__':record()
