"""Prepare external visual-review evidence and similarity leads; never approve labels."""
from collections import defaultdict, Counter
import hashlib
from io import BytesIO
import html
import re
from PIL import Image, ImageDraw, ImageOps
from .common import META, OUT, rows, write_rows, resolve, relative, digest, save_json, protect_frozen, setup, read_json, now
from .cleaning import difference_hash

RAW = META / "drone_type_external_raw.csv"
REVIEW = META / "drone_type_external_review.csv"


def collected_external_candidates():
    """Keep model-reference acquisitions outside the experiment candidate pool."""
    return [r for r in rows(RAW) + rows(META / "drone_type_article_raw.csv")
            if r.get("sample_scope", "external_candidate") != "reference_only"]


def prepare_external_review():
    protect_frozen()
    setup()
    images = collected_external_candidates()
    if len({r["sample_id"] for r in images}) != len(images):
        raise ValueError("Duplicate candidate IDs across collection manifests")
    if not images:
        raise ValueError("先执行02采集外部候选")
    folder = OUT / "review_external"
    folder.mkdir(parents=True, exist_ok=True)
    previous = {r["sample_id"]: r for r in rows(REVIEW)}
    quality, review, by_class, source_leads = [], [], defaultdict(list), []
    for r in images:
        p = resolve(r["path"])
        if digest(p) != r["sha256"] or digest(resolve(r["detail_snapshot"])) != r["detail_sha256"]:
            raise ValueError("原图或来源快照哈希变化：" + r["sample_id"])
        with Image.open(p) as im:
            orientation = im.getexif().get(274, 1)
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            q = {"sample_id": r["sample_id"], "path": r["path"], "candidate_label": r["candidate_label"],
                 "sha256": r["sha256"], "display_width": rgb.width, "display_height": rgb.height,
                 "original_exif_orientation": orientation,
                 "pixel_sha256": hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest(),
                 "dhash64": str(difference_hash(rgb.convert("L")))}
            thumb = ImageOps.contain(rgb, (620, 300))
            tile = Image.new("RGB", (640, 350), "white")
            tile.paste(thumb, ((640-thumb.width)//2, (300-thumb.height)//2))
            n = len(by_class[int(r["candidate_label"])])+1
            drawing = ImageDraw.Draw(tile)
            drawing.text((6, 305), f"class={r['candidate_label']} index={n:02d} | {r['sample_id']}", fill="black")
            drawing.text((6, 322), f"{rgb.width} x {rgb.height} | candidate only; no target box yet", fill="black")
            tile.save(folder / (r["sample_id"]+".jpg"), quality=93)
        quality.append(q)
        by_class[int(r["candidate_label"])].append({**r, "index": n})
        review.append(previous.get(r["sample_id"], {"sample_id":r["sample_id"], "visual_decision":"pending",
            "candidate_label":r["candidate_label"], "label_status":"unverified", "box_status":"pending",
            "source_group":"", "source_group_status":"unverified", "flags":"", "note":"",
            "reviewer":"", "evidence":""}))
        # This is a lead only: author/date strings may be incomplete or contain aliases.
        author = re.sub(r"\s+", " ", r.get("author", "").strip()).casefold()
        day = re.sub(r",?\s*\d{1,2}:\d{2}(:\d{2})?.*$", "", r.get("source_date_text", "")).strip()
        # Missing author/day must not collapse unrelated article sources into one group.
        identity = author+"|"+day if author and day else "unknown|"+r["source_url"]
        key = hashlib.sha256(identity.encode()).hexdigest()[:16]
        source_leads.append({"sample_id":r["sample_id"], "author":r.get("author", ""), "date_text":r.get("source_date_text", ""),
            "candidate_author_day_group":"author_day_"+key, "status":"metadata_lead_only",
            "source_url":r["source_url"], "detail_snapshot":r["detail_snapshot"]})
    sheet_manifest, links = [], []
    for label, members in sorted(by_class.items()):
        for offset in range(0, len(members), 10):
            sheet = Image.new("RGB", (1280, 1750), "white")
            page = folder / f"class_{label}_page_{offset//10+1:02d}.jpg"
            current_members = members[offset:offset+10]
            for i, r in enumerate(current_members):
                with Image.open(folder/(r["sample_id"]+".jpg")) as tile:
                    sheet.paste(tile, ((i%2)*640, (i//2)*350))
            # Existing review observations refer to immutable page hashes. An
            # expanded partial page must get a new name instead of replacing it.
            buffer = BytesIO()
            sheet.save(buffer, format="JPEG", quality=95)
            content = buffer.getvalue()
            content_hash = hashlib.sha256(content).hexdigest()
            if page.exists() and digest(page) != content_hash:
                page = page.with_name(page.stem + "_" + content_hash[:16] + page.suffix)
            if not page.exists():
                page.write_bytes(content)
            elif digest(page) != content_hash:
                raise ValueError("Review page content collision: " + str(page))
            for i, r in enumerate(current_members):
                sheet_manifest.append({"sample_id":r["sample_id"], "candidate_label":label,
                    "class_index":r["index"], "sheet":relative(page), "position":i+1,
                    "path":r["path"], "title":r["title"], "description":r["description"],
                    "source_url":r["source_url"], "sha256":r["sha256"]})
            links.append(f"<h2>class {label}, page {offset//10+1}</h2><img loading='lazy' src='{page.name}'>")
    public = rows(META / "drone_type_set_quality.csv")
    pair_path = META / "drone_type_external_similarity_candidates.csv"
    previous_pairs = {(r["external_sample"], r["other_sample"], r["other_domain"]): r
                      for r in rows(pair_path)}
    pairs = []
    for i, r in enumerate(quality):
        # Full-frame similarity can miss crop/overlay variants. Do not infer independence.
        for other, domain in [(x, "external") for x in quality[:i]] + [(x, "public") for x in public]:
            distance = (int(r["dhash64"]) ^ int(other["dhash64"])).bit_count()
            exact = r["pixel_sha256"] == other["pixel_sha256"]
            if distance <= 6 or exact:
                pair = {"external_sample":r["sample_id"], "other_sample":other["sample_id"],
                    "other_domain":domain, "dhash_distance":distance, "pixel_identical":exact,
                    "external_sha256":r["sha256"], "other_sha256":other["sha256"],
                    "status":"pending_visual_verification", "note":"", "evidence":""}
                previous_pair = previous_pairs.get((r["sample_id"], other["sample_id"], domain))
                if previous_pair and all(previous_pair.get(k) == pair[k]
                                         for k in ("external_sha256", "other_sha256")):
                    for field in ("status", "note", "evidence"):
                        pair[field] = previous_pair.get(field, pair[field])
                pairs.append(pair)
    write_rows(META/"drone_type_external_quality.csv", quality)
    write_rows(REVIEW, review)
    write_rows(META/"drone_type_external_source_leads.csv", source_leads)
    write_rows(pair_path, pairs,
               ["external_sample","other_sample","other_domain","dhash_distance","pixel_identical",
                "external_sha256","other_sha256","status","note","evidence"])
    write_rows(folder/"manifest.csv", sheet_manifest)
    (folder/"index.html").write_text("<!doctype html><meta charset='utf-8'><title>外部候选审核</title>"
        "<style>body{font-family:system-ui}img{max-width:100%}</style><h1>外部候选原图</h1>"
        "<p>网页类别不是标签真值，当前未标目标框。生成预览不等于完成审核。</p>"+"".join(links), encoding="utf-8")
    save_json(OUT/"drone_type_external_review_preparation.json", {"image_count":len(images),
        "manifest_sha256":digest(RAW),
        "collection_manifests":{relative(p):digest(p) for p in (RAW, META/"drone_type_article_raw.csv") if p.exists()},
        "images_by_class":dict(Counter(r["candidate_class_name"] for r in images)),
        "visual_decision_counts":dict(Counter(r["visual_decision"] for r in review)),
        "sheet_count":len(links),
        "candidate_pair_counts":dict(Counter(p["other_domain"] for p in pairs)),
        "pixel_identical_pairs":sum(p["pixel_identical"] for p in pairs),
        "source_metadata_groups":len({r["candidate_author_day_group"] for r in source_leads}),
        "caution":"Metadata groups are not verified independent sources; dHash is only full-frame screening.",
        "review_complete":False, "test_set_ready":False})
    print("外部候选审核准备完成：",folder/"index.html")
    print("相似候选对：",len(pairs),"；没有自动删除、批准标签或划分数据。")


def summarize_external_review():
    """Summarize current decisions without regenerating visual evidence."""
    setup()
    images = collected_external_candidates()
    decisions = rows(REVIEW)
    if len({r["sample_id"] for r in images}) != len(images):
        raise ValueError("Duplicate collected sample ID")
    if len({r["sample_id"] for r in decisions}) != len(decisions):
        raise ValueError("Duplicate review sample ID")
    reviewed = {r["sample_id"]:r for r in decisions}
    groups = rows(META / "drone_type_external_reviewed_groups.csv")
    classes = read_json(META / "drone_type_set_classes.json")["classes"]
    coverage = []
    for label, name in enumerate(classes):
        members = [r for r in images if int(r["candidate_label"]) == label]
        counts = Counter(reviewed.get(r["sample_id"], {}).get("visual_decision", "pending") for r in members)
        coverage.append({"class_id":label, "class_name":name, "collected_candidates":len(members),
            "candidate_keep":counts["candidate_keep"], "pending":counts["pending"], "reject":counts["reject"],
            "source_pages":len({r["source_url"] for r in members}),
            "verified_independent_test_images":0})
    status = {"updated_at":now(), "total_collected_candidates":len(images),
        "total_saved_images":len(rows(RAW)) + len(rows(META / "drone_type_article_raw.csv")),
        "commons_candidates":len(rows(RAW)),
        "article_candidates":sum(r.get("sample_scope", "external_candidate") != "reference_only" for r in rows(META / "drone_type_article_raw.csv")),
        "reference_only_images":sum(r.get("sample_scope") == "reference_only" for r in rows(META / "drone_type_article_raw.csv")),
        "visual_decision_counts":dict(Counter(reviewed.get(r["sample_id"], {}).get("visual_decision", "pending")
                                              for r in images)),
        "coverage":coverage,
        "uncollected_classes":[r["class_name"] for r in coverage if not r["collected_candidates"]],
        "reviewed_correlation_groups":len({r["source_group"] for r in groups}),
        "images_with_correlation_constraints":len({r["sample_id"] for r in groups}),
        "source_independence_verified":False, "final_labels_and_boxes_verified":False,
        "test_set_ready":False,
        "note":"Candidate keep is a visual decision. Correlation groups are not independent flights; image count is not source count.",
        "collection_manifest_sha256":{relative(p):digest(p) for p in (RAW,META/"drone_type_article_raw.csv") if p.exists()},
        "review_sha256":digest(REVIEW),
        "review_group_sha256":digest(META/"drone_type_external_reviewed_groups.csv") if groups else None}
    write_rows(META / "drone_type_external_combined_coverage.csv", coverage)
    save_json(OUT / "drone_type_external_current_status.json", status)
    print("External current status:", status["total_collected_candidates"], status["visual_decision_counts"])
    return status
