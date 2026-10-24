"""Record public catalog metadata and actual access outcomes, without credentials.

Raw responses are archived with SHA256. Search results and repository file counts
are discovery evidence only, never verified image labels or independent samples.
"""
import json
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from .common import OUT, ROOT, config, now, save_json, write_rows, read_json, digest, relative
from .http_client import Web


def inventory_platforms(platform=None, source_ids=None):
    spec = read_json(ROOT / "configs/platform_inventory.json")
    stamp = now().replace(":", "").replace("+", "_")
    folder = OUT / "platform_inventory" / stamp
    folder.mkdir(parents=True, exist_ok=True)
    events, catalogs = [], []
    for name, sources in spec["platforms"].items():
        if platform and name != platform:
            continue
        hosts = {s["host"] for s in sources}
        client = Web({**config(), "request_delay_seconds": 2.0}, allowed_hosts=hosts)
        for source in sources:
            if source_ids and source["id"] not in source_ids:
                continue
            event = {"platform": name, "source_id": source["id"], "url": source["url"], "observed_at": now()}
            try:
                payload = client.get(source["url"], max_bytes=8_000_000)
                try:
                    data = json.loads(payload)
                except (ValueError, UnicodeDecodeError):
                    data = None
                path = folder / (source["id"] + (".json" if data is not None else ".html"))
                path.write_bytes(payload)
                event.update(snapshot=relative(path), sha256=digest(path), bytes=len(payload))
                if data is not None:
                    event["status"] = "public_json_observed"
                    event["top_level"] = list(data)[:30] if isinstance(data, dict) else "list"
                    items = data if isinstance(data, list) else data.get("datasetFiles", data.get("files", []))
                    if isinstance(items, list):
                        event["returned_items"] = len(items)
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            catalogs.append({"platform": name, "source_id": source["id"],
                                "item_id": item.get("id", item.get("ref", item.get("path", item.get("name", "")))),
                                "title": item.get("title", ""),
                                "path": item.get("path", item.get("name", "")),
                                "bytes": item.get("size", item.get("totalBytes", item.get("total_bytes", ""))),
                                "item_type": item.get("type", ""), "snapshot": relative(path),
                                "status": "catalog_metadata_only_not_verified_labels_or_images"})
                else:
                    soup = BeautifulSoup(payload, "html.parser")
                    event["status"] = "public_html_observed_not_verified_dataset"
                    event["title"] = soup.title.get_text(" ", strip=True) if soup.title else ""
                    event["structured_dataset_metadata"] = [json.loads(tag.get_text()) for tag in soup.select('script[type="application/ld+json"]') if tag.get_text().lstrip().startswith("{")]
                    event["dataset_links"] = list(dict.fromkeys(urljoin(source["url"], a["href"])
                        for a in soup.select("a[href]") if any(term in (a["href"]+" "+a.get_text(" ",strip=True)).casefold()
                            for term in ("dataset", "数据集", "无人机", "drone"))))[:100]
                print(name, source["id"], event["status"], flush=True)
            except Exception as exc:
                event.update(status="access_failed_or_robots_unconfirmed", error=repr(exc))
                print(name, source["id"], event["status"], repr(exc), flush=True)
            events.append(event)
            save_json(folder / "progress.json", events)
    write_rows(folder / "catalog_items.csv", catalogs,
        ["platform", "source_id", "item_id", "title", "path", "bytes", "item_type", "snapshot", "status"])
    result = {"observed_at": now(), "platform_filter": platform, "source_filter": source_ids, "events": events,
        "catalog_items": len(catalogs), "downloaded_dataset_media": False,
        "class_scope_changed": False, "ran_models_or_software_tests": False,
        "searches_are_bounded_not_exhaustive": True, "folder": relative(folder)}
    save_json(folder / "summary.json", result)
    save_json(OUT / "platform_inventory" / ((platform or "all") + ("_followup" if source_ids else "") + "_current.json"), result)
    return result
