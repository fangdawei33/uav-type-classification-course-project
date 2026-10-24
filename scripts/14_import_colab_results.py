"""14. Import returned Colab report artifacts and check file/table consistency.

Does not deserialize models, run inference, fit models or software tests.
Raw ZIP and returned outputs remain immutable in a content-addressed folder.
"""
import argparse, csv, hashlib, io, json, math, stat, zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from _bootstrap import PROJECT_ROOT
from uav_project.common import digest, now, rows, write_rows, save_json, relative

BASE='outputs/representation_v1/auxiliary/'

def run(archive):
 archive=Path(archive).resolve()
 if not archive.is_relative_to(PROJECT_ROOT):raise ValueError('Archive must be in this project')
 archive_hash=digest(archive)
 destination=PROJECT_ROOT/'outputs/imported_results'/('auxiliary_'+archive_hash[:16])
 destination.mkdir(parents=True,exist_ok=True)
 checks=[];problems=[]
 def check(condition,name,detail=None):
  checks.append({'check':name,'passed':bool(condition),'detail':detail})
  if not condition:problems.append(name)
 with zipfile.ZipFile(archive) as z:
  infos=z.infolist();names=[i.filename for i in infos]
  if len(names)!=len(set(names)):raise ValueError('Duplicate archive member')
  if sum(i.file_size for i in infos)>100_000_000:raise ValueError('Unexpected report archive size')
  for i in infos:
   p=PurePosixPath(i.filename)
   if p.is_absolute() or '..' in p.parts or '\\' in i.filename or ':' in i.filename:
    raise ValueError('Unsafe archive path: '+i.filename)
   if stat.S_ISLNK(i.external_attr>>16):raise ValueError('Symlink not accepted')
   if i.is_dir():continue
   target=(destination/Path(*p.parts)).resolve()
   if not target.is_relative_to(destination):raise ValueError('Archive escaped destination')
   data=z.read(i.filename)
   if target.exists() and target.read_bytes()!=data:raise ValueError('Refusing to overwrite imported history')
   if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  contents={name:z.read(name) for name in names if not name.endswith('/')}
 def js(name):return json.loads(contents[name].decode('utf-8-sig'))
 def table(name):return list(csv.DictReader(io.StringIO(contents[name].decode('utf-8-sig'))))
 sha=lambda data:hashlib.sha256(data).hexdigest()
 frozen=js(BASE+'freeze.json'); measured=js(BASE+'evaluation/measured_results.json');protocol=js(BASE+'protocol.json')
 check(measured['freeze_hash']==sha(contents[BASE+'freeze.json']),'evaluation_links_to_returned_freeze')
 check(measured['primary_model']==frozen['primary_model'],'frozen_primary_model_unchanged')
 check(not measured['main_seven_class_result'] and not measured['external_real_test_evaluated'],'auxiliary_scope_retained')
 verified=[];unreturned=[];local_references=[]
 for name,expected in frozen['file_hashes'].items():
  if name in contents:
   check(sha(contents[name])==expected,'frozen_member_hash:'+name);verified.append(name)
  else:
   unreturned.append(name)
   local=PROJECT_ROOT/name
   if local.exists():
    match=digest(local)==expected
    local_references.append({'path':name,'matches_frozen_hash':match})
    check(match,'local_reference_matches_frozen:'+name)
 # This is not a complete audit of large unreturned feature/model binaries.
 check(datetime.fromisoformat(frozen['frozen_at'])<datetime.fromisoformat(measured['evaluated_at']),'freeze_precedes_evaluation')
 local_manifest=rows(PROJECT_ROOT/'outputs/representation_v1/auxiliary/manifest.csv')
 dev={r['sample_id']:r for r in local_manifest if r['split']=='development'}
 hold={r['sample_id']:r for r in local_manifest if r['split']=='holdout'}
 check(len(dev)==3999 and len(hold)==3999 and not set(dev)&set(hold),'local_protocol_development_holdout_ids')
 cv=js(BASE+'cv_folds.json');check(set(cv['development_sample_ids'])==set(dev),'cv_uses_only_development_ids')
 universe=set(range(len(dev)));validation=[]
 for idx,f in enumerate(cv['folds']):
  tr,va=set(f['train']),set(f['validation']);validation+=f['validation']
  check(not tr&va and tr|va==universe,'fold_disjoint_and_complete:'+str(idx))
 check(Counter(validation)==Counter({i:1 for i in universe}),'validation_coverage_once')
 check(len(cv['folds'])==3,'three_cv_folds')
 byresult={(r['representation'],r['classifier']):r for r in measured['results']}
 comparison=table(BASE+'evaluation/model_comparison.csv');derived=[]
 for row in comparison:
  rep,model=row['representation'],row['classifier'];r=byresult[(rep,model)]
  pred_path=BASE+f'evaluation/{rep}_{model}_predictions.csv';pred=table(pred_path)
  expected=measured['prediction_hashes'].get(pred_path)
  if expected is None:expected=measured['prediction_hashes'].get(f'{rep}_{model}')
  if expected is None:expected=measured['prediction_hashes'].get(pred_path.split('/')[-1])
  check(expected==sha(contents[pred_path]),'prediction_hash:'+rep+'_'+model)
  ids=[p['sample_id'] for p in pred]
  check(len(ids)==len(set(ids))==3999 and set(ids)==set(hold),'prediction_id_coverage:'+rep+'_'+model)
  cm=[[0]*4 for _ in range(4)]
  for p in pred:
   sid=p['sample_id'];y=int(p['label']);v=int(p['predicted_label'])
   if not (0<=y<4 and 0<=v<4):raise ValueError('Invalid returned label')
   if sid not in hold or y!=int(hold[sid]['label']):raise ValueError('Prediction truth differs from local manifest')
   if p['class_name']!=protocol['classes'][y] or p['predicted_class']!=protocol['classes'][v]:raise ValueError('Prediction names differ from class index')
   cm[y][v]+=1
  check(cm==r['holdout']['confusion_matrix'],'prediction_counts_equal_reported_confusion:'+rep+'_'+model)
  support=[sum(rr) for rr in cm];n=sum(support)
  recalls=[cm[i][i]/support[i] for i in range(4)]
  f1=[2*cm[i][i]/(support[i]+sum(cm[j][i] for j in range(4))) for i in range(4)]
  arithmetic={'accuracy':sum(cm[i][i] for i in range(4))/n,'balanced_accuracy':sum(recalls)/4,
              'macro_f1':sum(f1)/4,'weighted_f1':sum(f*s for f,s in zip(f1,support))/n}
  check(all(math.isclose(v,r['holdout'][k],abs_tol=1e-12) and math.isclose(v,float(row[k]),abs_tol=1e-12) for k,v in arithmetic.items()),'returned_table_arithmetic_consistent:'+rep+'_'+model)
  receipt=js(BASE+f'models/{rep}/{model}.json')
  check(math.isclose(float(row['cv_macro_f1']),receipt['cv_macro_f1_mean'],abs_tol=1e-12),'cv_receipt_matches_table:'+rep+'_'+model)
  derived.append({'representation':rep,'classifier':model,'errors':n-sum(cm[i][i] for i in range(4)),
   'cv_minus_holdout_macro_f1_pp':100*(r['cv_macro_f1']-r['holdout']['macro_f1']),
   'training_minus_cv_macro_f1_pp':100*(r['training_macro_f1']-r['cv_macro_f1'])})
 for rep in ['handcrafted','siglip2']:
  de=js(BASE+f'features/{rep}/development.json');ho=js(BASE+f'features/{rep}/holdout.json')
  check(set(de['signature']['sample_ids'])==set(dev) and set(ho['signature']['sample_ids'])==set(hold),'feature_ids_match_protocol:'+rep)
  check(datetime.fromisoformat(frozen['frozen_at'])<datetime.fromisoformat(ho['finished_at']),'holdout_features_finished_after_freeze:'+rep)
  if rep=='siglip2':
   fields=['inference_dtype','revision','torch','transformers','model_id']
   check(all(de['signature'][k]==ho['signature'][k] for k in fields),'encoder_version_and_precision_consistent')
 check(len(comparison)==10 and len(byresult)==10,'ten_declared_result_rows')
 receipt={'imported_at':now(),'archive':relative(archive),'archive_sha256':archive_hash,
  'destination':relative(destination),'archive_file_count':len(contents),'checks':checks,'problems':problems,
  'returned_frozen_files_verified':len(verified),'frozen_files_not_in_report_package':unreturned,
  'local_frozen_references':local_references,'all_available_checks_passed':not problems,
  'not_checked':'Unreturned joblib and npz files, actual GPU execution and unknown render-parent independence cannot be independently authenticated from this report ZIP.',
  'local_model_inference_run':False,'local_model_training_run':False,'software_tests_run':False,
  'only_returned_files_and_numeric_tables_inspected':True}
 save_json(destination/'import_receipt.json',receipt)
 write_rows(destination/'comparison_diagnostics.csv',derived)
 save_json(PROJECT_ROOT/'outputs/auxiliary_results_current.json',{
  'updated_at':now(),'import_folder':relative(destination),'receipt':relative(destination/'import_receipt.json'),
  'report_archive_sha256':archive_hash,'all_available_checks_passed':not problems,
  'gpu':js('outputs/colab_environment.json')['gpu'],'result_count':len(comparison),
  'main_seven_class_result':False,'external_real_test_evaluated':False})
 print('Imported',len(contents),'files:',destination)
 print('Consistency checks:',len(checks),'passed:',len(checks)-len(problems),'issues:',len(problems))
 print('Returned frozen files verified:',len(verified),'unreturned frozen references:',len(unreturned))
 if problems:raise ValueError('Returned result consistency findings: '+str(problems))
 return destination
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('archive',type=Path)
 run(parser.parse_args().archive)
