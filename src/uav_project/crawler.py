"""Crawl explicitly configured Commons category/file HTML pages.

Seven-class candidates use separate paths from the retired binary dataset.
A category is a search lead, never a ground-truth label or a flight identifier.
"""
from collections import Counter, deque
import hashlib
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlsplit, unquote, quote

from bs4 import BeautifulSoup
from PIL import Image

from .common import (ROOT, META, OUT, config, setup, protect_frozen, rows, write_rows,
                     resolve, relative, read_json, save_json, append_log, digest, now)
from .http_client import Web

MANIFEST = META / "drone_type_external_raw.csv"
SUMMARY = OUT / "drone_type_external_crawl_summary.json"
RASTER_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def canonical_file_url(url):
    parts = urlsplit(url)
    path = unquote(parts.path).replace(" ", "_")
    if parts.scheme != "https" or parts.hostname != "commons.wikimedia.org" or not path.startswith("/wiki/File:"):
        raise ValueError("不是配置允许的Commons文件说明页")
    return "https://commons.wikimedia.org" + quote(path, safe="/():,_-.")


def file_metadata(soup, page_url):
    def text(selector):
        node = soup.select_one(selector)
        return node.get_text(" ", strip=True) if node else ""

    def table_value(marker):
        node = soup.select_one(marker)
        row = node.find_parent("tr") if node else None
        cells = row.find_all("td", recursive=False) if row else []
        return cells[-1].get_text(" ", strip=True) if cells else ""

    license_urls = sorted({urljoin(page_url, a["href"]) for a in soup.select(".licensetpl a[href]")
                           if "creativecommons.org/" in a["href"] or "gnu.org/licenses/" in a["href"]})
    return {"title": text("h1"), "description": text(".description"),
        "author": table_value("#fileinfotpl_aut"), "original_source": table_value("#fileinfotpl_src"),
        "source_date_text": table_value("#fileinfotpl_date"),
        "license": text(".licensetpl_short") or text(".licensetpl"),
        "license_urls": " | ".join(license_urls), "license_review": "pending"}


def save_page(payload):
    # Content-addressed HTML snapshots preserve the evidence used for each record.
    key = hashlib.sha256(payload).hexdigest()
    path = OUT / "crawl_pages" / (key + ".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(payload)
    return relative(path), key


def persist_records(records):
    if records:
        temporary = MANIFEST.with_suffix(".csv.tmp")
        write_rows(temporary, records.values())
        temporary.replace(MANIFEST)


def crawl_external(class_id=None, max_per_class=None):
    protect_frozen()
    setup()
    spec = read_json(ROOT / "configs" / "crawl_sources.json")
    classes = read_json(META / "drone_type_set_classes.json")["classes"]
    configured = spec["classes"]
    if [c["class_id"] for c in configured] != list(range(len(classes))) or [c["class_name"] for c in configured] != classes:
        raise ValueError("爬虫类别映射与已导入的七类清单不一致")
    limit = max_per_class if max_per_class is not None else spec["target_per_class"]
    if limit < 1:
        raise ValueError("每类候选上限必须为正数")
    cfg = {**config(), "request_delay_seconds": spec["request_delay_seconds"]}
    client = Web(cfg)
    existing = rows(MANIFEST)
    records = {r["sample_id"]: r for r in existing}
    if len(records) != len(existing):
        raise ValueError("候选清单有重复ID，需先核查")
    # Verify existing receipt hashes before treating files as resumable successes.
    valid = set()
    for sid, r in records.items():
        path = resolve(r["path"])
        if path.exists():
            if digest(path) != r["sha256"]:
                raise ValueError("已有候选文件变化，停止覆盖：" + r["path"])
            valid.add(sid)
    destination = ROOT / "data" / "raw" / "drone_type_external"
    destination.mkdir(parents=True, exist_ok=True)
    events, conflicts, attempts = [], [], set()
    started = now()

    def count(label):
        return sum(sid in valid and int(r["candidate_label"]) == label for sid, r in records.items())

    def event(kind, url, label, **extra):
        record = {"time": now(), "kind": kind, "url": url, "class_id": label, **extra}
        events.append(record)
        append_log("drone_type_crawl", record)

    for entry in configured:
        label = entry["class_id"]
        if class_id is not None and label != class_id:
            continue
        if not entry["category_urls"]:
            print(f"[{label}] {entry['class_name']}：尚无已核实采集页，记录缺口。", flush=True)
            continue
        for seed in entry["category_urls"]:
            queue, visited = deque([seed]), set()
            while queue and len(visited) < spec["max_pages_per_seed"] and count(label) < limit:
                category_url = queue.popleft()
                if category_url in visited:
                    continue
                visited.add(category_url)
                try:
                    payload = client.get(category_url, spec["max_html_bytes"])
                    snapshot, snapshot_hash = save_page(payload)
                    soup = BeautifulSoup(payload, "html.parser")
                    urls = []
                    for a in soup.select(".gallerytext a[href]"):
                        url = urljoin(category_url, a["href"])
                        if unquote(urlsplit(url).path).startswith("/wiki/File:"):
                            file_url = canonical_file_url(url)
                            if Path(unquote(urlsplit(file_url).path)).suffix.lower() in RASTER_SUFFIXES and file_url not in urls:
                                urls.append(file_url)
                    # Follow pagination of this exact category only. No child categories:
                    # 'Taken with ...' pages mostly show landscapes, not the drone itself.
                    for a in soup.select("a.mw-nextlink[href], #mw-category-media a[href]"):
                        target = urljoin(category_url, a["href"])
                        parsed = urlsplit(target)
                        if (parsed.scheme == "https" and parsed.hostname == "commons.wikimedia.org"
                                and unquote(parsed.path) == unquote(urlsplit(seed).path)
                                and parsed.query and target not in visited and target not in queue):
                            queue.append(target)
                    event("category", category_url, label, file_links=len(urls), page_snapshot=snapshot,
                          page_sha256=snapshot_hash)
                    if not urls:
                        event("no_raster_links", category_url, label,
                              note="可能页面无图片或解析结构变化；不自动改抓搜索页/API")
                    print(f"[{label}] {entry['class_name']}：发现{len(urls)}条候选链接；已有{count(label)}/{limit}", flush=True)
                except Exception as exc:
                    event("category_failed", category_url, label, error=repr(exc))
                    print(f"来源暂不可用：{category_url}；{exc}", flush=True)
                    continue

                for file_url in urls:
                    if count(label) >= limit:
                        break
                    sid = "dte_" + hashlib.sha256(file_url.encode("utf-8")).hexdigest()[:20]
                    if sid in records and int(records[sid]["candidate_label"]) != label:
                        records[sid]["review_status"] = "candidate_class_conflict"
                        conflict = {"sample_id": sid, "first_candidate_label": records[sid]["candidate_label"],
                                    "other_candidate_label": label, "source_url": file_url}
                        conflicts.append(conflict)
                        append_log("drone_type_candidate_conflicts", conflict)
                        persist_records(records)
                        continue
                    if sid in valid or file_url in attempts:
                        continue
                    attempts.add(file_url)
                    try:
                        detail_payload = client.get(file_url, spec["max_html_bytes"])
                        detail_snapshot, detail_hash = save_page(detail_payload)
                        detail = BeautifulSoup(detail_payload, "html.parser")
                        original = detail.select_one(".fullMedia a.internal[href]")
                        if original is None:
                            raise ValueError("没有找到原图链接；不把缩略图或视频封面当原图")
                        image_url = urljoin(file_url, original["href"])
                        image_payload = client.get(image_url, spec["max_image_bytes"])
                        with Image.open(BytesIO(image_payload)) as image:
                            extension = {"JPEG":".jpg", "PNG":".png", "WEBP":".webp", "TIFF":".tiff"}.get(image.format)
                            if extension is None or getattr(image, "n_frames", 1) != 1:
                                raise ValueError("只接收单帧静态栅格图片")
                            width, height = image.size
                            image.verify()
                        target = destination / (sid + extension)
                        payload_hash = hashlib.sha256(image_payload).hexdigest()
                        if target.exists() and digest(target) != payload_hash:
                            raise ValueError("同名原文件内容不同，保留原文件并停止该样本")
                        if not target.exists():
                            temporary = target.with_suffix(target.suffix + ".part")
                            temporary.write_bytes(image_payload)
                            temporary.replace(target)
                        records[sid] = {"sample_id":sid, "path":relative(target), "domain":"external_candidate",
                            "candidate_label":label, "candidate_class_name":entry["class_name"],
                            "review_status":"pending", "group_id":"", "target_box_status":"pending",
                            "source_url":file_url, "category_url":category_url, "download_url":image_url,
                            "collected_at":now(), "sha256":payload_hash, "width":width, "height":height,
                            "category_snapshot":snapshot, "detail_snapshot":detail_snapshot,
                            "detail_sha256":detail_hash, **file_metadata(detail, file_url)}
                        valid.add(sid)
                        persist_records(records)
                        event("image_saved", file_url, label, sample_id=sid, sha256=payload_hash)
                        print(f"[{label}] 已保存 {count(label)}/{limit}：{sid}", flush=True)
                    except Exception as exc:
                        event("image_failed", file_url, label, error=repr(exc))
                        print(f"候选跳过：{file_url}；{exc}", flush=True)
    coverage = []
    for entry in configured:
        n = count(entry["class_id"])
        coverage.append({"class_id":entry["class_id"], "class_name":entry["class_name"],
            "candidate_images":n, "source_count":len(entry["category_urls"]),
            "source_status":entry["status"], "target_per_class":limit,
            "reviewed_test_images":0, "note":entry["note"]})
    write_rows(META / "drone_type_external_coverage.csv", coverage)
    save_json(SUMMARY, {"started_at":started, "finished_at":now(),
        "status":"candidate_collection_only", "class_filter":class_id,
        "valid_existing_and_new_images":len(valid), "attempted_new_file_pages":len(attempts),
        "event_counts":dict(Counter(e["kind"] for e in events)), "coverage":coverage,
        "class_conflicts_this_run":conflicts, "all_labels_review_required":True,
        "full_seven_class_external_test_ready":False, "public_external_dedup_complete":False,
        "source_config_sha256":digest(ROOT / "configs" / "crawl_sources.json")})
    print("候选清单：", MANIFEST, flush=True)
    print("采集摘要：", SUMMARY, flush=True)
    print("这是待核实候选，尚未成为独立测试集。未配置来源的类别在覆盖表中列明。", flush=True)
    if not valid:
        raise RuntimeError("本轮没有可用候选原图。查看采集摘要与outputs/logs/drone_type_crawl.jsonl；不要关闭robots检查。")
