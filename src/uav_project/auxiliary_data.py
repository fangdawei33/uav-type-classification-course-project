"""Auxiliary data acquisition stays separate from the seven-class raw manifests."""
from pathlib import Path
from importlib.metadata import version
from .common import ROOT, OUT, read_json, save_json, digest, relative, now, protect_frozen


def auxiliary_spec(source_id):
    sources = read_json(ROOT / "configs/auxiliary_sources.json")["sources"]
    if source_id not in sources:
        raise ValueError("Unknown auxiliary source: " + source_id)
    return sources[source_id]


def download_auxiliary(source_id):
    protect_frozen()
    spec = auxiliary_spec(source_id)
    folder = ROOT / "data/downloads/auxiliary" / source_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / spec["filename"]
    started = now()
    log = OUT / "auxiliary" / source_id
    save_json(log / "download_current.json", {"status": "started", "started_at": started,
        "source_id": source_id, "revision": spec["revision"], "media_expected_bytes": spec["expected_bytes"],
        "seven_class_scope_changed": False, "media_path": relative(path)})
    if not path.exists():
        from huggingface_hub import hf_hub_download
        print("Downloading one public auxiliary archive:", spec["repo_id"], flush=True)
        print("Pinned revision:", spec["revision"], "expected bytes:", spec["expected_bytes"], flush=True)
        # Explicit anonymous access: do not read or send a saved account token.
        received = hf_hub_download(repo_id=spec["repo_id"], repo_type="dataset",
            revision=spec["revision"], filename=spec["filename"], local_dir=folder,
            token=False)
        if Path(received).resolve() != path.resolve():
            raise ValueError("Downloader returned an unexpected destination")
    actual = digest(path)
    if path.stat().st_size != spec["expected_bytes"] or actual != spec["expected_sha256"]:
        raise ValueError("Archive size/SHA256 differs from pinned public metadata; retained for inspection, not imported")
    # Keep the data card next to acquisition metadata; the media archive is immutable.
    from huggingface_hub import hf_hub_download
    card = hf_hub_download(repo_id=spec["repo_id"], repo_type="dataset", revision=spec["revision"],
        filename="README.md", local_dir=folder, token=False)
    status = {"status": "downloaded_and_sha256_verified_not_imported", "source_id": source_id,
        "started_at": started, "finished_at": now(), "repository": spec["repo_id"], "revision": spec["revision"],
        "archive": relative(path), "bytes": path.stat().st_size, "sha256": actual,
        "card": relative(card), "card_sha256": digest(card), "huggingface_hub_version": version("huggingface_hub"),
        "origin": spec["origin"], "role": spec["intended_role"],
        "current_seven_class_mapping": spec["current_seven_class_mapping"],
        "included_in_training": False, "dataset_extracted": False, "gpu_used": False}
    save_json(log / "download_current.json", status)
    save_json(log / ("download_" + status["finished_at"].replace(":", "").replace("+", "_") + ".json"), status)
    print("Auxiliary archive SHA256 verified:", path, flush=True)
    return status
