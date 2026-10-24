"""Shared image provenance records, independent of dataset source."""
from PIL import Image, ImageOps
from .common import relative, now, digest

def image_record(path, domain, label, source_url, sample_id, extra=None):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB"); im.load()
        width, height = im.size
    return {"sample_id": sample_id, "path": relative(path), "domain": domain,
            "candidate_label": label, "source_url": source_url, "collected_at": now(),
            "sha256": digest(path), "width": width, "height": height,
            "group_hint": sample_id, **(extra or {})}
