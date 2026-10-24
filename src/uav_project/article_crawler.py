"""Collect explicitly listed article HTML and manually selected media, with provenance.

Discovery never treats an article title as image ground truth. Image collection
requires an explicit decision in the discovery inventory, and retains copyrights
as unresolved unless the source provides verifiable terms.
"""
from collections import Counter
import hashlib
import json
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup
from PIL import Image

from .common import (ROOT, META, OUT, config, setup, protect_frozen, rows,
                     write_rows, relative, resolve, read_json, save_json,
                     append_log, digest, now)
from .crawler import save_page
from .http_client import Web

INVENTORY = META / 'drone_type_article_media_inventory.csv'
RAW = META / 'drone_type_article_raw.csv'
PAGES = META / 'drone_type_article_pages.csv'
RASTER = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff'}


def image_url_allowed(url, source):
    parts = urlsplit(url)
    return (parts.scheme == 'https' and parts.hostname in source['allowed_hosts']
            and Path(parts.path).suffix.lower() in RASTER)


def discover_article(client, source, payload=None):
    if payload is None:
        payload = client.get(source['url'], source['max_html_bytes'])
    snapshot, snapshot_hash = save_page(payload)
    soup = BeautifulSoup(payload, 'html.parser')
    if source.get('content_format') == 'jsonld_howto':
        # Read structured data already published inside this exact public HTML.
        # Do not call private app endpoints or execute the site's JavaScript.
        documents = [json.loads(node.get_text()) for node in soup.select('script[type="application/ld+json"]')]
        howto = next((d for d in documents if isinstance(d, dict) and d.get('@type') == 'HowTo'), None)
        if howto is None:
            raise ValueError('No public HowTo structured data; snapshot: ' + snapshot)
        title = howto.get('name', '')
        content = soup.new_tag('section')
        for item in [howto] + howto.get('step', []):
            paragraph = soup.new_tag('p')
            paragraph.string = item.get('name', '') + ': ' + item.get('text', item.get('description', ''))
            content.append(paragraph)
            image = item.get('image', '')
            image = image.get('url', '') if isinstance(image, dict) else image
            if isinstance(image, str) and image:
                figure = soup.new_tag('figure')
                figure.append(soup.new_tag('img', src=image, alt=item.get('name', '')))
                caption = soup.new_tag('figcaption')
                caption.string = paragraph.get_text()
                figure.append(caption)
                content.append(figure)
        containers = [content]
    else:
        containers = soup.select(source['content_selector'])
        if not containers:
            raise ValueError('Configured article-content selector found no content; preserve HTML: ' + snapshot)
        title_node = soup.select_one(source.get('title_selector', 'h1'))
        title = title_node.get_text(' ', strip=True) if title_node else ''
    missing = [term for term in source['required_title_terms'] if term.casefold() not in title.casefold()]
    if missing:
        raise ValueError('Article identity mismatch; title=' + repr(title) + '; snapshot=' + snapshot)
    text_path = OUT / 'crawl_articles' / (snapshot_hash + '.txt')
    text_path.parent.mkdir(parents=True, exist_ok=True)
    article_text = '\n\n'.join(c.get_text(' ', strip=True) for c in containers)
    text_path.write_text(article_text, encoding='utf-8')
    candidates = {}
    for content in containers:
        for image in content.select('img'):
            urls = []
            if source.get('prefer_same_image_full_resolution_links'):
                # A few original documentation pages place the high-resolution
                # link beside the thumbnail, not around it. Select only a link
                # actually published in the same paragraph with a matching stem.
                group = image.find_parent(['figure', 'p'])
                original_stem = Path(urlsplit(image.get('src', '')).path).stem.removesuffix('_small')
                if group and original_stem:
                    for anchor in group.select('a[href]'):
                        observed = urljoin(source['url'], anchor['href'])
                        if Path(urlsplit(observed).path).stem == original_stem:
                            urls.append(observed)
            # Only URLs actually present in source HTML; never guess original URLs.
            parent = image.find_parent('a', href=True)
            if parent:
                urls.append(parent['href'])
            for attr in ['data-orig-file', 'data-large-file', 'data-src', 'src']:
                if image.get(attr):
                    urls.append(image[attr])
            for attr in ['data-srcset', 'srcset']:
                for entry in image.get(attr, '').split(','):
                    if entry.strip():
                        urls.append(entry.strip().split()[0])
            allowed = [urljoin(source['url'], u) for u in urls
                       if image_url_allowed(urljoin(source['url'], u), source)]
            if not allowed:
                continue
            # Original anchor/data-orig-file precedes srcset renditions. Keep all
            # observed alternatives for provenance but collect only the chosen URL.
            media_url = allowed[0]
            figure = image.find_parent(['figure', 'p'])
            caption = figure.get_text(' ', strip=True) if figure else ''
            candidates.setdefault(media_url, {
                'source_id': source['source_id'], 'candidate_label': source['class_id'],
                'candidate_class_name': source['class_name'], 'source_url': source['url'],
                'article_title': title, 'media_url': media_url,
                'alt': image.get('alt', ''), 'caption': caption,
                'observed_alternatives': ' | '.join(dict.fromkeys(allowed)),
                'detail_snapshot': snapshot, 'detail_sha256': snapshot_hash,
                'collection_decision': 'pending', 'decision_note': '',
                'source_role': source['source_role'], 'license_review': 'pending'
            })
    return {'source_id': source['source_id'], 'candidate_label': source['class_id'],
            'source_url': source['url'], 'title': title, 'detail_snapshot': snapshot,
            'detail_sha256': snapshot_hash, 'text_path': relative(text_path),
            'text_sha256': digest(text_path), 'media_candidates': len(candidates),
            'status': 'discovered', 'collected_at': now()}, list(candidates.values())


def collect_articles(download_approved=False, source_id=None, use_saved_page=False, group=None):
    protect_frozen()
    setup()
    spec_path = ROOT / 'configs' / 'article_sources.json'
    spec = read_json(spec_path)
    inventory = {(r['source_id'], r['media_url']): r for r in rows(INVENTORY)}
    pages = {r['source_id']: r for r in rows(PAGES)}
    records = {r['sample_id']: r for r in rows(RAW)}
    failures, events = [], []
    selected_sources = [s for s in spec['sources'] if s.get('enabled', True)
                        and (source_id is None or s['source_id'] == source_id)
                        and (group is None or s.get('group') == group)]
    if not selected_sources:
        raise ValueError('Unknown configured source_id: ' + str(source_id))
    for original in selected_sources:
        source = {**original, 'max_html_bytes': spec['max_html_bytes']}
        client = Web({**config(), 'request_delay_seconds': spec['request_delay_seconds']},
                     allowed_hosts=source['allowed_hosts'])
        try:
            if not download_approved:
                saved = None
                if use_saved_page:
                    cached = resolve(source['saved_html_snapshot'])
                    if digest(cached) != source['saved_html_sha256']:
                        raise ValueError('Saved source HTML hash changed')
                    saved = cached.read_bytes()
                page, discovered = discover_article(client, source, saved)
                page['parsing_mode'] = 'saved_public_html' if use_saved_page else 'live_public_html'
                pages[source['source_id']] = page
                for r in discovered:
                    key = (r['source_id'], r['media_url'])
                    previous = inventory.get(key)
                    if previous:
                        # A changed evidence page invalidates the old media selection.
                        if previous['detail_sha256'] == r['detail_sha256']:
                            r['collection_decision'] = previous['collection_decision']
                            r['decision_note'] = previous['decision_note']
                    inventory[key] = r
                write_rows(PAGES, pages.values())
                write_rows(INVENTORY, inventory.values())
                print(source['source_id'], ': article saved, media leads =', len(discovered), flush=True)
                events.append({'source_id': source['source_id'], 'kind': 'article_discovered',
                               'candidates': len(discovered)})
                continue
            chosen = [r for r in inventory.values() if r['source_id'] == source['source_id']
                      and r['collection_decision'] == 'selected']
            for r in chosen:
                if any(term.casefold() not in r['article_title'].casefold() for term in source['required_title_terms']):
                    raise ValueError('Selected media has an unrelated article title')
                if r['source_url'] != source['url'] or not image_url_allowed(r['media_url'], source):
                    raise ValueError('Selected URL outside configured source scope')
                if digest(resolve(r['detail_snapshot'])) != r['detail_sha256']:
                    raise ValueError('Article evidence hash changed')
                sid = 'dta_' + hashlib.sha256(r['media_url'].encode()).hexdigest()[:20]
                if sid in records:
                    if int(records[sid]['candidate_label']) != source['class_id']:
                        raise ValueError('Same media selected under different candidate classes')
                    if digest(resolve(records[sid]['path'])) != records[sid]['sha256']:
                        raise ValueError('Previously collected image hash changed')
                    continue
                try:
                    payload = client.get(r['media_url'], spec['max_image_bytes'])
                    with Image.open(BytesIO(payload)) as im:
                        suffix = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'TIFF': '.tiff'}.get(im.format)
                        if suffix is None or getattr(im, 'n_frames', 1) != 1:
                            raise ValueError('Not an allowed single-frame raster image')
                        width, height = im.size
                        im.verify()
                    destination = ROOT / 'data' / 'raw' / 'drone_type_articles' / (sid + suffix)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    sha = hashlib.sha256(payload).hexdigest()
                    if destination.exists() and digest(destination) != sha:
                        raise ValueError('Existing raw file differs; do not overwrite')
                    if not destination.exists():
                        temporary = destination.with_suffix(suffix + '.part')
                        temporary.write_bytes(payload)
                        temporary.replace(destination)
                    records[sid] = {'sample_id': sid, 'path': relative(destination),
                        'domain': 'external_article_candidate', 'candidate_label': source['class_id'],
                        'candidate_class_name': source['class_name'], 'source_id': source['source_id'],
                        'source_url': r['source_url'], 'download_url': r['media_url'],
                        'collected_at': now(), 'sha256': sha, 'width': width, 'height': height,
                        'title': r['article_title'], 'description': r['alt'], 'caption': r['caption'],
                        'detail_snapshot': r['detail_snapshot'], 'detail_sha256': r['detail_sha256'],
                        'review_status': 'pending', 'target_box_status': 'pending',
                        'group_id': '', 'group_status': 'unverified',
                        'license': 'not_verified_do_not_redistribute', 'license_review': 'pending',
                        'label_status': 'unverified_source_label', 'source_role': source['source_role'],
                        'sample_scope': source.get('sample_scope', 'external_candidate')}
                    write_rows(RAW, records.values())
                    events.append({'source_id': source['source_id'], 'kind': 'image_saved', 'sample_id': sid})
                    print(source['source_id'], ': saved', sid, f'({width}x{height})', flush=True)
                except Exception as exc:
                    failure = {'source_id': source['source_id'], 'url': r['media_url'], 'error': repr(exc)}
                    failures.append(failure)
                    append_log('article_crawl', {'kind': 'image_failed', **failure})
                    print('Image unavailable:', failure, flush=True)
        except Exception as exc:
            failure = {'source_id': source['source_id'], 'url': source['url'], 'error': repr(exc)}
            failures.append(failure)
            append_log('article_crawl', {'kind': 'source_failed', **failure})
            print('Source unavailable:', failure, flush=True)
    for event in events:
        append_log('article_crawl', event)
    summary = {'finished_at': now(), 'mode': 'download_selected' if download_approved else ('discover_saved_html' if use_saved_page else 'discover'),
        'source_filter': source_id, 'group_filter': group, 'source_config_sha256': digest(spec_path),
        'event_counts': dict(Counter(e['kind'] for e in events)), 'failures': failures,
        'total_article_pages': len(pages), 'total_media_leads': len(inventory),
        'total_saved_article_images': len(records),
        'total_downloaded_candidates': sum(r.get('sample_scope', 'external_candidate') != 'reference_only' for r in records.values()),
        'reference_only_images': sum(r.get('sample_scope') == 'reference_only' for r in records.values()),
        'candidate_images_by_class': dict(Counter(r['candidate_class_name'] for r in records.values() if r.get('sample_scope', 'external_candidate') != 'reference_only')),
        'labels_verified': False, 'source_independence_verified': False, 'test_ready': False}
    save_json(OUT / ('drone_type_article_collection_summary.json' if download_approved else
                     'drone_type_article_discovery_summary.json'), summary)
    print('Article stage complete:', summary['event_counts'], '; candidate images:', summary['total_downloaded_candidates'], '; reference images:', summary['reference_only_images'], flush=True)
    if failures and not events:
        raise RuntimeError('No new article operation succeeded; inspect the saved source-failure summary.')
