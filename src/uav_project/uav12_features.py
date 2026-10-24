"""Leakage-safe feature extraction/cache for the fixed v3 12-label protocol."""
import importlib.metadata

import numpy as np
import pandas as pd

from .common import ROOT, digest, read_json, save_json, now, resolve, relative
from .uav12_protocol import CONFIG, load_protocol, run_dir, settings


def _partition(frame, name):
    part = frame[frame.split.eq(name)].sort_values(["label", "sample_id"]).reset_index(drop=True)
    if part.empty:
        raise ValueError(f"Empty partition: {name}")
    return part


def _atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _verify_images(part):
    for index, row in enumerate(part.itertuples(), start=1):
        path = resolve(row.clean_path)
        if not path.is_file() or digest(path) != row.clean_sha256:
            raise ValueError(f"Image hash mismatch: {row.sample_id}")
        if index % 2000 == 0:
            print(f"Image identity checked: {index}/{len(part)}", flush=True)


def _load_siglip(device_batch_size):
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModel, AutoProcessor

    cfg = settings()["siglip2"]
    folder = run_dir() / "features" / "siglip2"
    folder.mkdir(parents=True, exist_ok=True)
    pin_path = folder / "model_pin.json"
    if pin_path.exists():
        pin = read_json(pin_path)
        if pin["model_id"] != cfg["model_id"]:
            raise ValueError("SigLIP2 pin differs from the declared model")
    else:
        info = HfApi(token=False).model_info(cfg["model_id"], revision="main")
        pin = {"model_id": info.id, "revision": info.sha, "pinned_at": now()}
        save_json(pin_path, pin)
    if not torch.cuda.is_available():
        raise RuntimeError("SigLIP2 extraction needs a CUDA GPU; run this Colab stage on T4")
    dtype = torch.float16
    processor = AutoProcessor.from_pretrained(pin["model_id"], revision=pin["revision"], token=False,
                                              trust_remote_code=False)
    model = AutoModel.from_pretrained(pin["model_id"], revision=pin["revision"], token=False,
                                      trust_remote_code=False, use_safetensors=True,
                                      torch_dtype=dtype).to("cuda").eval()
    model.requires_grad_(False)
    return model, processor, torch, pin, dtype, device_batch_size


def extract_uav12_features(representation, partition="development", batch_size=None):
    if representation not in {"handcrafted", "siglip2"}:
        raise ValueError("representation must be handcrafted or siglip2")
    folder, protocol, all_rows = load_protocol()
    if partition not in {"development", "holdout"}:
        raise ValueError("partition must be development or holdout")
    if partition == "holdout":
        from .uav12_models import verify_uav12_freeze
        verify_uav12_freeze()
    part = _partition(all_rows, partition)
    destination = folder / "features" / representation
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"{partition}.npz"
    receipt_path = destination / f"{partition}.json"
    code_paths = [ROOT / "src/uav_project/uav12_features.py"]
    if representation == "handcrafted":
        code_paths.append(ROOT / "src/uav_project/features.py")
    base_signature = {
        "protocol_hash": protocol["manifest_sha256"],
        "config_hash": digest(CONFIG),
        "partition": partition,
        "representation": representation,
        "sample_ids": part.sample_id.tolist(),
        "image_hashes": part.clean_sha256.tolist(),
        "feature_code_hashes": {relative(path): digest(path) for path in code_paths},
        "numpy": importlib.metadata.version("numpy"),
    }
    if receipt_path.exists():
        existing = read_json(receipt_path)
        old_signature = existing.get("signature", {})
        stable_keys = ["protocol_hash", "config_hash", "partition", "representation",
                       "sample_ids", "image_hashes", "feature_code_hashes"]
        if any(old_signature.get(key) != base_signature.get(key) for key in stable_keys):
            raise RuntimeError("Handcrafted cache signature changed; start a new experiment version")
        if existing.get("status") == "completed" and existing.get("signature", {}).get("partition") == partition:
            if representation == "siglip2":
                pin_path = destination / "model_pin.json"
                if not pin_path.is_file():
                    raise ValueError("SigLIP2 cache has no retained model revision pin")
                pin = read_json(pin_path)
                if old_signature.get("model_id") != pin.get("model_id") or old_signature.get("revision") != pin.get("revision"):
                    raise ValueError("SigLIP2 feature cache model revision differs from the recorded pin")
            if output_path.exists() and digest(output_path) == existing.get("feature_sha256"):
                print("Feature cache already complete:", output_path, flush=True)
                return
            raise ValueError("Completed feature cache missing or changed")

    _verify_images(part)
    if representation == "handcrafted":
        from .features import features
        matrix, feature_names = features(part)
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape[0] != len(part) or not np.isfinite(matrix).all():
            raise ValueError("Handcrafted feature matrix is incomplete or nonfinite")
        signature = base_signature
        _atomic_npz(output_path, X=matrix, names=np.asarray(feature_names, dtype=str),
                    sample_ids=part.sample_id.to_numpy(dtype=str),
                    image_hashes=part.clean_sha256.to_numpy(dtype=str))
        extra = {"feature_dimensions": int(matrix.shape[1]), "fitted_dataset_statistics": False}
    else:
        import torch
        cfg = settings()["siglip2"]
        model, processor, torch, pin, dtype, actual_batch = _load_siglip(batch_size or cfg["batch_size"])
        if actual_batch < 1:
            raise ValueError("Inference batch size must be positive")
        signature = {**base_signature, "model_id": pin["model_id"], "revision": pin["revision"],
                     "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                     "inference_dtype": str(dtype), "batch_size": actual_batch}
        shard_dir = destination / f"{partition}_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_signature_path = shard_dir / "signature.json"
        shard_signature = {key: value for key, value in signature.items() if key != "batch_size"}
        if shard_signature_path.exists() and read_json(shard_signature_path) != shard_signature:
            raise ValueError("Existing SigLIP2 shards use different inputs, model, code or runtime")
        save_json(shard_signature_path, shard_signature)
        feature_blocks, feature_names = [], None
        from PIL import Image, ImageOps
        shard_size = int(cfg["shard_size"])
        for start in range(0, len(part), shard_size):
            piece = part.iloc[start:start + shard_size]
            shard_path = shard_dir / f"{start:06d}.npz"
            shard_hashes = piece.clean_sha256.to_numpy(dtype=str)
            if shard_path.exists():
                with np.load(shard_path, allow_pickle=False) as saved:
                    if saved["sample_ids"].tolist() != piece.sample_id.tolist() or not np.array_equal(saved["image_hashes"], shard_hashes):
                        raise ValueError(f"SigLIP2 shard belongs to different samples: {shard_path}")
                    block, feature_names = saved["X"], saved["names"].tolist()
            else:
                _verify_images(piece)
                chunks = []
                for offset in range(0, len(piece), actual_batch):
                    images = []
                    for row in piece.iloc[offset:offset + actual_batch].itertuples():
                        with Image.open(resolve(row.clean_path)) as image:
                            images.append(ImageOps.exif_transpose(image).convert("RGB"))
                    inputs = processor(images=images, return_tensors="pt").to("cuda")
                    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                        vector = model.get_image_features(**inputs)
                        if not torch.is_tensor(vector):
                            vector = vector.pooler_output
                    if vector.ndim != 2:
                        raise ValueError("Expected pooled SigLIP2 image features")
                    vector = torch.nn.functional.normalize(vector.float(), dim=-1)
                    chunks.append(vector.cpu().numpy())
                block = np.concatenate(chunks).astype(np.float32)
                feature_names = [f"siglip2_embedding_{index}" for index in range(block.shape[1])]
                _atomic_npz(shard_path, X=block, names=np.asarray(feature_names, dtype=str),
                            sample_ids=piece.sample_id.to_numpy(dtype=str), image_hashes=shard_hashes)
            if not np.isfinite(block).all():
                raise ValueError(f"Nonfinite features in {shard_path}")
            feature_blocks.append(block)
            print(f"SigLIP2 {partition}: {min(start+len(piece),len(part))}/{len(part)}", flush=True)
        matrix = np.concatenate(feature_blocks).astype(np.float32)
        _atomic_npz(output_path, X=matrix, names=np.asarray(feature_names, dtype=str),
                    sample_ids=part.sample_id.to_numpy(dtype=str),
                    image_hashes=part.clean_sha256.to_numpy(dtype=str))
        extra = {"feature_dimensions": int(matrix.shape[1]), "fitted_dataset_statistics": False,
                 "model_id": pin["model_id"], "revision": pin["revision"], "inference_dtype": str(dtype)}
        del model, processor
        torch.cuda.empty_cache()
    save_json(receipt_path, {"status": "completed", "signature": signature,
                            "feature_sha256": digest(output_path), "samples": len(part),
                            "finished_at": now(), **extra,
                            "holdout_read_after_global_freeze": partition != "holdout" or (folder / "freeze.json").exists()})
    print("Saved feature matrix:", output_path, matrix.shape, flush=True)


def load_uav12_features(representation, partition):
    folder, protocol, frame = load_protocol()
    part = _partition(frame, partition)
    path = folder / "features" / representation / f"{partition}.npz"
    receipt_path = path.with_suffix(".json")
    if not path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"Feature cache missing: run 06d for {representation}/{partition}")
    receipt = read_json(receipt_path)
    if receipt.get("status") != "completed" or digest(path) != receipt.get("feature_sha256"):
        raise ValueError("Feature cache incomplete or hash mismatch")
    signature = receipt["signature"]
    if (signature["protocol_hash"] != protocol["manifest_sha256"] or
            signature.get("config_hash") != digest(CONFIG) or
            signature["sample_ids"] != part.sample_id.tolist()):
        raise ValueError("Feature cache does not align with the fixed protocol")
    if signature["image_hashes"] != part.clean_sha256.tolist():
        raise ValueError("Feature cache image identities changed")
    for code_path, expected in signature["feature_code_hashes"].items():
        if digest(resolve(code_path)) != expected:
            raise ValueError("Feature extractor changed: " + code_path)
    with np.load(path, allow_pickle=False) as cached:
        matrix, names = cached["X"], cached["names"].tolist()
        if cached["sample_ids"].tolist() != part.sample_id.tolist():
            raise ValueError("Feature sample order changed")
    if matrix.shape[0] != len(part) or not np.isfinite(matrix).all():
        raise ValueError("Feature matrix dimensions/values invalid")
    return matrix, names, part
