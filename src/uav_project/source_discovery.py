"""Bounded public link discovery with robots, snapshots and no automatic media download."""
from collections import Counter
from urllib.parse import urljoin, urlsplit, urldefrag
import xml.etree.ElementTree as ET
import traceback
from bs4 import BeautifulSoup
from .common import ROOT,OUT,config,read_json,rows,write_rows,save_json,append_log,now,digest
from .http_client import Web
from .crawler import save_page


def discover_source_links(source_id=None, group=None):
    spec_path=ROOT/'configs/source_discovery.json'
    spec=read_json(spec_path)
    chosen=[s for s in spec['sources'] if (source_id is None or s['source_id']==source_id) and (group is None or s.get('group')==group)]
    if not chosen:raise ValueError('Unknown discovery source: '+str(source_id))
    destination=OUT/'source_discovery'
    destination.mkdir(parents=True,exist_ok=True)
    events=[]
    for source in chosen:
        client=Web({**config(),'request_delay_seconds':spec['request_delay_seconds']},allowed_hosts=source['allowed_hosts'])
        try:
            payload=client.get(source['url'],spec['max_html_bytes'])
            snapshot,sha=save_page(payload)
            title=''
            candidates=[]
            if payload.lstrip().startswith(b'<?xml') or urlsplit(source['url']).path.endswith('.xml'):
                tree=ET.fromstring(payload)
                candidates=[(n.text.strip(),'sitemap_loc') for n in tree.iter() if n.tag.rsplit('}',1)[-1]=='loc' and n.text]
            else:
                soup=BeautifulSoup(payload,'html.parser')
                title=soup.title.get_text(' ',strip=True) if soup.title else ''
                for node in soup.select('a[href], iframe[src], img[src]'):
                    candidates.append((node.get('href',node.get('src','')),node.get_text(' ',strip=True)))
            links={}
            for target,label in candidates:
                url=urldefrag(urljoin(source['url'],target))[0]
                if urlsplit(url).scheme not in {'https','http'}:continue
                matches=[term for term in source['match_terms'] if term.casefold() in (url+' '+label).casefold()]
                if not matches:continue
                links[url]={'source_id':source['source_id'],'page_url':source['url'],'page_title':title,
                    'observed_url':url,'link_text':label,'matched_terms':'|'.join(matches),
                    'snapshot':snapshot,'snapshot_sha256':sha,'observed_at':now(),
                    'status':'source_lead_only_not_media_or_verified_label'}
            write_rows(destination/(source['source_id']+'_links.csv'),links.values(),
                       ['source_id','page_url','page_title','observed_url','link_text','matched_terms','snapshot','snapshot_sha256','observed_at','status'])
            save_json(destination/(source['source_id']+'_page.json'),{'source':source,'snapshot':snapshot,'snapshot_sha256':sha,
                'title':title,'matching_links':len(links),'recorded_at':now()})
            events.append({'source_id':source['source_id'],'status':'discovered','matching_links':len(links),'snapshot':snapshot})
            print(source['source_id'], ': observed links',len(links),flush=True)
        except Exception as exc:
            event={'source_id':source['source_id'],'status':'failed','url':source['url'],'error':repr(exc),'traceback':traceback.format_exc(),'recorded_at':now()}
            events.append(event);append_log('source_discovery',event)
            print(source['source_id'],': unavailable',repr(exc),flush=True)
    run_time=now()
    result={'recorded_at':run_time,'config_sha256':digest(spec_path),'source_filter':source_id,'group_filter':group,'events':events,
            'downloads_images':False,'verifies_source_identity':False}
    save_json(destination/('run_'+run_time.replace(':','').replace('+','_')+'.json'),result)
    save_json(destination/'latest_run.json',result)
    if not any(e['status']=='discovered' for e in events):raise RuntimeError('No configured source succeeded; failures saved.')
    return events
