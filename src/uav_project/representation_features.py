"""Frozen SigLIP2 embeddings or fixed HOG/HSV/LBP features, with resumable shards.

No statistics, PCA, scaling, classifier or prompts are fitted here. Holdout image
features are deliberately unavailable until model artifacts have been frozen.
"""
import importlib.metadata
import numpy as np
import pandas as pd
from .common import ROOT, META, OUT, resolve, digest, read_json, save_json, now
from .representation_protocol import SETTINGS, load_protocol


def _atomic_npz(path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def extract_representation(dataset, representation, partition="development", batch_size=None):
    cfg = read_json(SETTINGS)
    if dataset == "candidates":
        if partition != "development":
            raise ValueError("Candidate crops have no approved holdout partition")
        manifest = META / "drone_type_set_candidate_crops.csv"
        frame = pd.read_csv(manifest).fillna("").sort_values("sample_id").reset_index(drop=True)
        if frame.training_approved.astype(str).str.lower().eq("true").any():
            raise ValueError("Candidate status unexpectedly changed")
        folder = OUT / "representation_v1/candidates"
        manifest_hash = digest(manifest)
    else:
        folder, protocol, all_rows = load_protocol(dataset)
        if partition == "holdout":
            from .representation_training import verify_freeze
            verify_freeze(dataset)
        frame = all_rows[all_rows.split.eq(partition)].reset_index(drop=True)
        manifest_hash = protocol["manifest_hash"]
    if frame.empty:
        raise ValueError("No selected images")
    if representation not in cfg["representations"]:
        raise ValueError("Unknown representation")
    feature_dir = folder / "features" / representation
    feature_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = feature_dir / (partition + ".json")
    output_path = feature_dir / (partition + ".npz")
    signature = {"manifest_hash": manifest_hash, "representation": representation,
                 "config_hash": digest(SETTINGS), "partition": partition,
                 "extractor_code": digest(ROOT / "src/uav_project/representation_features.py"),
                 "handcrafted_code": digest(ROOT / "src/uav_project/features.py"),
                 "sample_ids": frame.sample_id.tolist(), "image_hashes": frame.clean_sha256.tolist()}
    model = processor = torch = inference_dtype = None
    if representation == "siglip2":
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModel, AutoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError("SigLIP2提取留给T4/GPU执行，本机不运行。")
        pin_path = feature_dir / "model_pin.json"
        if pin_path.exists():
            pin = read_json(pin_path)
            if pin["model_id"] != cfg["foundation"]["model_id"]:
                raise ValueError("Model pin differs from configuration")
        else:
            info = HfApi(token=False).model_info(cfg["foundation"]["model_id"], revision="main")
            pin = {"model_id": info.id, "revision": info.sha, "pinned_at": now()}
            save_json(pin_path, pin)
        native_bf16 = torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()
        inference_dtype = torch.bfloat16 if native_bf16 else torch.float16
        signature["inference_dtype"] = str(inference_dtype)
        signature.update(model_id=pin["model_id"], revision=pin["revision"],
                         transformers=importlib.metadata.version("transformers"), torch=torch.__version__)
    else:
        signature["scikit_image"] = importlib.metadata.version("scikit-image")
    if receipt_path.exists():
        old = read_json(receipt_path)
        if old["signature"] != signature:
            raise RuntimeError("Cached extraction uses different inputs or code; start a new experiment version")
        if old.get("status") == "completed" and output_path.exists():
            if digest(output_path) != old["feature_sha256"]:
                raise ValueError("Feature cache hash changed")
            print("Feature cache already complete:", output_path)
            return
    save_json(receipt_path, {"signature": signature, "status": "started", "started_at": now()})
    if representation == "siglip2":
        model = AutoModel.from_pretrained(pin["model_id"], revision=pin["revision"], token=False,
                                          trust_remote_code=False, use_safetensors=True).to("cuda").eval()
        model.requires_grad_(False)
        processor = AutoProcessor.from_pretrained(pin["model_id"], revision=pin["revision"], token=False, trust_remote_code=False)
    batch_size = batch_size or cfg["foundation"]["batch_size"]
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    shard_dir = feature_dir / (partition + "_shards")
    shard_dir.mkdir(exist_ok=True)
    matrices = []
    names = None
    # Fixed shard boundaries permit retries with a smaller inference batch.
    for start in range(0, len(frame), 128):
        part = frame.iloc[start:start + 128]
        shard = shard_dir / f"{start:06d}.npz"
        # Verify original image identity even when resuming a cached shard.
        for row in part.itertuples():
            if digest(resolve(row.clean_path)) != row.clean_sha256:
                raise ValueError("Image changed: " + row.sample_id)
        if shard.exists():
            with np.load(shard, allow_pickle=False) as saved:
                if saved["sample_ids"].tolist() != part.sample_id.tolist():
                    raise ValueError("Shard sample ordering differs")
                x, names = saved["X"], saved["names"].tolist()
        elif representation == "handcrafted":
            from .features import features
            x, names = features(part)
            _atomic_npz(shard, X=x, names=np.asarray(names), sample_ids=part.sample_id.to_numpy(dtype=str))
        else:
            from PIL import Image, ImageOps
            batches = []
            for offset in range(0, len(part), batch_size):
                images = []
                for row in part.iloc[offset:offset + batch_size].itertuples():
                    with Image.open(resolve(row.clean_path)) as image:
                        images.append(ImageOps.exif_transpose(image).convert("RGB"))
                inputs = processor(images=images, return_tensors="pt").to("cuda")
                with torch.inference_mode(), torch.autocast("cuda", dtype=inference_dtype):
                    vector = model.get_image_features(**inputs)
                    if not torch.is_tensor(vector):
                        vector = vector.pooler_output
                if vector.ndim != 2:
                    raise ValueError("Expected pooled image feature matrix")
                vector = torch.nn.functional.normalize(vector.float(), dim=-1)
                batches.append(vector.cpu().numpy())
            x = np.concatenate(batches)
            names = [f"siglip2_embedding_{i}" for i in range(x.shape[1])]
            _atomic_npz(shard, X=x, names=np.asarray(names), sample_ids=part.sample_id.to_numpy(dtype=str))
        if not np.isfinite(x).all() or len(x) != len(part):
            raise ValueError("Nonfinite or incomplete feature shard")
        matrices.append(x)
        print(f"{dataset}/{representation}/{partition}: {start + len(part)}/{len(frame)}", flush=True)
    X = np.concatenate(matrices).astype(np.float32)
    _atomic_npz(output_path, X=X, names=np.asarray(names), sample_ids=frame.sample_id.to_numpy(dtype=str),
                image_hashes=frame.clean_sha256.to_numpy(dtype=str))
    save_json(receipt_path, {"signature": signature, "status": "completed", "finished_at": now(),
                           "shape": list(X.shape), "feature_sha256": digest(output_path),
                           "fitted_dataset_statistics": False, "classifier_trained": False,
                           "training_approved": dataset != "candidates"})
    print("Saved features:", output_path)


def load_features(dataset, representation, partition):
    folder, protocol, frame = load_protocol(dataset)
    part = frame[frame.split.eq(partition)].reset_index(drop=True)
    path = folder / "features" / representation / (partition + ".npz")
    receipt = read_json(path.with_suffix(".json"))
    if receipt["status"] != "completed" or digest(path) != receipt["feature_sha256"]:
        raise ValueError("Feature extraction incomplete or cache changed")
    signature = receipt["signature"]
    if signature["manifest_hash"] != protocol["manifest_hash"] or signature["config_hash"] != digest(SETTINGS):
        raise ValueError("Features do not match protocol")
    if signature["extractor_code"] != digest(ROOT / "src/uav_project/representation_features.py") or signature["handcrafted_code"] != digest(ROOT / "src/uav_project/features.py"):
        raise ValueError("Feature implementation changed")
    with np.load(path, allow_pickle=False) as saved:
        if saved["sample_ids"].tolist() != part.sample_id.tolist() or saved["image_hashes"].tolist() != part.clean_sha256.tolist():
            raise ValueError("Feature/sample alignment differs")
        X, names = saved["X"], saved["names"].tolist()
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite features")
    return X, names, part


