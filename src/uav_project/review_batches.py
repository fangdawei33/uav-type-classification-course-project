"""Generate traceable contact sheets for actual data review, without approving samples."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from .common import META, OUT, digest, protect_frozen, relative, resolve, rows, save_json, write_rows
from .review import image_number
from .cleaning import boxes_for, pixel_box


def make_batches(class_id=None, flag=None, start=None, end=None):
    protect_frozen()
    source = META / "drone_type_set_quality.csv"
    if not source.exists():
        raise ValueError("先运行04_clean_data.py生成质量表")
    selected = rows(source)
    if class_id is not None:
        selected = [r for r in selected if int(r["class_id"]) == class_id]
    if flag:
        selected = [r for r in selected if flag in r["flags"].split("|")]
    if start is not None:
        selected = [r for r in selected if image_number(r) >= start]
    if end is not None:
        selected = [r for r in selected if image_number(r) <= end]
    selected.sort(key=lambda r: (int(r["class_id"]), image_number(r)))
    if not selected:
        raise ValueError("没有符合条件的图片")
    # Stable folder names keep evidence references reproducible across reruns.
    name = f"class_{class_id if class_id is not None else 'all'}"
    if flag:
        if not flag.replace("_", "").isalnum():
            raise ValueError("flag只允许字母、数字、下划线")
        name += "_" + flag
    if start is not None or end is not None:
        name += f"_{start}_{end}"
    folder = OUT / "review_batches" / name
    folder.mkdir(parents=True, exist_ok=True)
    evidence = []
    links = []
    for offset in range(0, len(selected), 12):
        members = selected[offset:offset+12]
        sheet = Image.new("RGB", (1440, 1860), "white")
        for i, row in enumerate(members):
            # Reuse the original/crop evidence card; never use this sheet for training.
            with Image.open(resolve(row["preview_path"])) as source_card:
                card = source_card.convert("RGB")
            if int(row["box_count"]) > 1:
                # Show every candidate explicitly; this never selects the largest box.
                with Image.open(resolve(row["path"])) as source_image:
                    original = source_image.convert("RGB")
                boxes = boxes_for(row)
                marked = original.copy()
                painter = ImageDraw.Draw(marked)
                for box_index, box in enumerate(boxes):
                    rect = pixel_box(box, *original.size)
                    painter.rectangle(rect, outline="red", width=3)
                    painter.text((rect[0], rect[1]), str(box_index), fill="yellow", stroke_width=1, stroke_fill="black")
                card = Image.new("RGB", (720, 310), "white")
                card.paste(ImageOps.contain(marked, (420, 250)), (0, 0))
                drawing = ImageDraw.Draw(card)
                height = 250 // len(boxes)
                for box_index, box in enumerate(boxes):
                    crop = original.crop(pixel_box(box, *original.size))
                    card.paste(ImageOps.contain(crop, (280, max(1, height-18))), (430, box_index*height+18))
                    drawing.text((430, box_index*height), f"annotation index={box_index}", fill="black")
                drawing.text((5, 254), f"class={row['class_id']} | {Path(row['path']).name}", fill="black")
                drawing.text((5, 272), "All source boxes; no automatic selection. REVIEW ONLY.", fill="black")
            sheet.paste(card, ((i % 2)*720, (i // 2)*310))
            evidence.append({"sample_id": row["sample_id"], "path": row["path"],
                "class_id": row["class_id"], "sheet": relative(folder / f"page_{offset//12+1:03d}.jpg"),
                "position": i+1, "sha256": row["sha256"],
                "annotation_sha256": row["annotation_sha256"], "flags": row["flags"]})
        page = folder / f"page_{offset//12+1:03d}.jpg"
        sheet.save(page, quality=93)
        links.append(f"<p><a href='{page.name}'>{page.name}</a></p><img loading='lazy' src='{page.name}'>")
    write_rows(folder / "manifest.csv", evidence)
    save_json(folder / "selection.json", {"class_id": class_id, "flag": flag, "start": start,
        "end": end, "images": len(selected), "quality_sha256": digest(source),
        "purpose": "visual review only; generation does not imply inspection or approval"})
    (folder / "index.html").write_text("<!doctype html><meta charset='utf-8'>"
        "<title>分批审核</title><style>body{font-family:system-ui}img{max-width:100%}</style>"
        "<h1>原图标注框 / 候选裁剪</h1><p>仅供审核，不是模型输入。生成预览不等于完成审核。</p>"
        + "".join(links), encoding="utf-8")
    print(f"审核批次：{folder}；图片{len(selected)}张；页数{len(links)}")
