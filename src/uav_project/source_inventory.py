"""Inventory configured Commons category pages without downloading new images."""
from collections import deque, Counter
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlsplit, parse_qs, unquote
from bs4 import BeautifulSoup
from .common import ROOT,META,OUT,config,read_json,rows,write_rows,save_json,now,append_log,protect_frozen
from .http_client import Web
from .crawler import save_page,canonical_file_url,RASTER_SUFFIXES


def _category(url):
    p=urlsplit(url)
    if p.path.startswith('/wiki/'):
        return unquote(p.path.removeprefix('/wiki/')).replace(' ','_')
    return parse_qs(p.query).get('title',[''])[0].replace(' ','_')


def inventory_commons(class_id=None,max_pages_per_seed=10):
    protect_frozen()
    spec=read_json(ROOT/'configs/crawl_sources.json')
    client=Web({**config(),'request_delay_seconds':spec['request_delay_seconds']})
    selected=[r for r in spec['classes'] if class_id is None or r['class_id']==class_id]
    already={r['source_url'] for r in rows(META/'drone_type_external_raw.csv')}
    by_id={r['sample_id']:r for r in rows(META/'drone_type_external_raw.csv')}
    folder=OUT/'source_inventory';folder.mkdir(parents=True,exist_ok=True)
    all_links=[]; class_reports=[]
    for entry in selected:
        found={}; seeds=[]
        for seed in entry['category_urls']:
            queue=deque([seed]);seen=set();failures=[];pages=[]
            while queue and len(seen)<max_pages_per_seed:
                url=queue.popleft()
                if url in seen:continue
                seen.add(url)
                try:
                    payload=client.get(url,spec['max_html_bytes'])
                    snapshot,sha=save_page(payload)
                    soup=BeautifulSoup(payload,'html.parser')
                    links=[]
                    for a in soup.select('.gallerytext a[href]'):
                        target=urljoin(url,a['href'])
                        if urlsplit(target).hostname!='commons.wikimedia.org' or not unquote(urlsplit(target).path).startswith('/wiki/File:'):
                            continue
                        file=canonical_file_url(target)
                        if Path(unquote(urlsplit(file).path)).suffix.lower() not in RASTER_SUFFIXES:continue
                        links.append(file)
                        sid='dte_'+hashlib.sha256(file.encode('utf-8')).hexdigest()[:20]
                        found.setdefault(file,{'candidate_class_id':entry['class_id'],'class_name':entry['class_name'],
                            'sample_id':sid,'file_page_url':file,'first_observed_category':url,'category_snapshot':snapshot,
                            'category_snapshot_sha256':sha,'already_saved':sid in by_id or file in already,
                            'status':'file_link_only_not_reviewed_target_or_independent_source','observed_at':now()})
                    for a in soup.select('#mw-category-media a[href], a.mw-nextlink[href]'):
                        target=urljoin(url,a['href']);p=urlsplit(target);q=parse_qs(p.query)
                        if p.hostname=='commons.wikimedia.org' and _category(target)==_category(seed) and any(k in q for k in ('filefrom','fileuntil')) and target not in seen and target not in queue:
                            queue.append(target)
                    pages.append({'url':url,'snapshot':snapshot,'sha256':sha,'raster_file_links':len(set(links))})
                    print(f"class {entry['class_id']}: {len(set(links))} file links on {url}",flush=True)
                except Exception as exc:
                    failure={'url':url,'error':repr(exc),'time':now()};failures.append(failure)
                    append_log('commons_inventory',{'class_id':entry['class_id'],'kind':'page_failed',**failure})
                    print(f"class {entry['class_id']}: inventory unavailable: {repr(exc)}",flush=True)
            seeds.append({'seed':seed,'pages':pages,'failures':failures,'pending_page_urls':list(queue),
                'configured_seed_traversal_complete':not queue and not failures})
        report={'observed_at':now(),'class_id':entry['class_id'],'class_name':entry['class_name'],
            'configured_category_seeds':len(seeds),'unique_raster_file_links':len(found),
            'already_saved_links':sum(r['already_saved'] for r in found.values()),
            'not_yet_saved_links':sum(not r['already_saved'] for r in found.values()),'seeds':seeds,
            'usable_target_images':'unknown_until_file_metadata_and_visual_review',
            'independent_capture_sources':'not_established_by_category_inventory','images_downloaded_by_inventory':0}
        save_json(folder/f"commons_class_{entry['class_id']}.json",report)
        all_links.extend(found.values());class_reports.append(report)
    selected_ids={r['class_id'] for r in selected}
    retained=[r for r in rows(META/'drone_type_commons_available_links.csv') if int(r['candidate_class_id']) not in selected_ids]
    write_rows(META/'drone_type_commons_available_links.csv',retained+all_links)
    timestamp=now()
    summary={'observed_at':timestamp,'class_filter':class_id,'class_reports':class_reports,
        'known_raster_file_links_in_selected_classes':len(all_links),'new_images_downloaded':0,
        'software_tests_run':False,'model_training_or_evaluation_run':False,
        'caution':'Category/file link counts are not qualified target images, model labels or independent sources.'}
    archive=folder/('commons_run_'+timestamp.replace(':','').replace('+','_')+'.json')
    save_json(archive,summary)
    save_json(folder/'commons_inventory_current.json',{**summary,'archived_run':str(archive.relative_to(ROOT)).replace('\\','/')})
    print('Commons inventory complete: links only; no image acquisition or model execution.',flush=True)
    return summary
