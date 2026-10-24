"""QLoRA SFT and sealed-holdout inference for Qwen3-VL on crop images only."""
import importlib.metadata
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, digest, read_json, save_json, now, relative, resolve
from .uav12_protocol import CONFIG, load_protocol, run_dir, settings


def _resolve_model_pin():
    from huggingface_hub import HfApi

    cfg = settings()["qwen3_vl"]
    destination = run_dir() / "models" / "qwen3_vl"
    destination.mkdir(parents=True, exist_ok=True)
    pin_path = destination / "model_pin.json"
    if pin_path.exists():
        pin = read_json(pin_path)
        if pin["model_id"] != cfg["model_id"]:
            raise ValueError("Qwen base-model pin differs from the frozen configuration")
        return pin, pin_path
    info = HfApi(token=False).model_info(cfg["model_id"], revision="main")
    pin = {"model_id": info.id, "revision": info.sha, "pinned_at": now(),
           "model_card": "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct"}
    save_json(pin_path, pin)
    return pin, pin_path


def _load_processor(pin):
    from transformers import AutoProcessor

    cfg = settings()["qwen3_vl"]
    processor = AutoProcessor.from_pretrained(pin["model_id"], revision=pin["revision"], token=False,
        trust_remote_code=False, min_pixels=int(cfg["min_image_pixels"]),
        max_pixels=int(cfg["max_image_pixels"]))
    processor.tokenizer.padding_side = "right"
    return processor


def _user_messages(image, class_names):
    choices = "\n".join(f"LABEL_{index:02d}: {name}" for index, name in enumerate(class_names))
    prompt = ("Identify the UAV type visible in the target crop. Use visual evidence only. "
              "Return exactly one label token from the list; do not explain.\n" + choices +
              "\nAnswer format: LABEL_00")
    return [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]


class _CropSFTDataset:
    def __init__(self, frame, processor, class_names, variant, max_length):
        self.frame = frame.reset_index(drop=True)
        self.processor = processor
        self.class_names = class_names
        self.variant = variant
        self.max_length = int(max_length)
        counts = np.bincount(self.frame.label.to_numpy(dtype=int), minlength=len(class_names))
        if (counts == 0).any():
            raise ValueError("Every label must be present in the development SFT set")
        raw = np.ones(len(counts), dtype=np.float64)
        if variant == "sqrt_class_balanced":
            raw = np.sqrt(counts.sum() / (len(counts) * counts))
            raw /= np.average(raw, weights=counts)
        self.sample_weights = raw

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        from PIL import Image, ImageOps
        import torch

        row = self.frame.iloc[index]
        with Image.open(resolve(row.clean_path)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        user = _user_messages(image, self.class_names)
        answer = f"LABEL_{int(row.label):02d}"
        # Qwen3-VL's processor expects every message content to be a
        # multimodal content list, including the supervised assistant answer.
        full = user + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        encoded = self.processor.apply_chat_template(full, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt", enable_thinking=False)
        prefix = self.processor.apply_chat_template(user, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False)
        input_ids = encoded["input_ids"].squeeze(0)
        prefix_ids = prefix["input_ids"].squeeze(0)
        if len(input_ids) > self.max_length:
            raise ValueError(f"Qwen prompt exceeds configured token budget ({len(input_ids)}>{self.max_length})")
        if len(prefix_ids) >= len(input_ids) or not torch.equal(input_ids[:len(prefix_ids)], prefix_ids):
            raise ValueError("Assistant prompt prefix differs from tokenized supervised example")
        labels = input_ids.clone()
        labels[:len(prefix_ids)] = -100
        if not labels.ne(-100).any():
            raise ValueError("No assistant target tokens remain after prompt masking")
        result = {}
        for key, value in encoded.items():
            if torch.is_tensor(value):
                if key in {"input_ids", "attention_mask", "position_ids", "token_type_ids"}:
                    continue
                if key in {"pixel_values", "image_grid_thw", "pixel_attention_mask"}:
                    result[key] = value
                else:
                    result[key] = value.squeeze(0) if value.ndim >= 2 and value.shape[0] == 1 else value
        result["input_ids"] = input_ids
        result["labels"] = labels
        result["sample_weights"] = torch.tensor(self.sample_weights[int(row.label)], dtype=torch.float32)
        result["sample_id"] = str(row.sample_id)
        return result


class _CropCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch

        sequences = [item["input_ids"] for item in features]
        labels = [item["labels"] for item in features]
        max_length = max(len(value) for value in sequences)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer has no pad token")
        batch = {
            "input_ids": torch.stack([torch.nn.functional.pad(value, (0, max_length-len(value)), value=pad_id)
                                      for value in sequences]),
            "labels": torch.stack([torch.nn.functional.pad(value, (0, max_length-len(value)), value=-100)
                                   for value in labels]),
            "attention_mask": torch.stack([torch.nn.functional.pad(torch.ones_like(value),
                (0, max_length-len(value)), value=0) for value in sequences]),
            "sample_weights": torch.stack([item["sample_weights"] for item in features]),
            "sample_ids": [item["sample_id"] for item in features],
        }
        keys = set().union(*(item.keys() for item in features))
        keys -= {"input_ids", "labels", "sample_weights", "sample_id"}
        for key in keys:
            values = [item[key] for item in features if key in item]
            if len(values) != len(features):
                raise ValueError(f"Missing multimodal field in batch: {key}")
            if key in {"mm_token_type_ids", "token_type_ids"}:
                batch[key] = torch.stack([torch.nn.functional.pad(value.reshape(-1),
                    (0, max_length-value.numel()), value=0) for value in values])
            elif key in {"pixel_values", "image_grid_thw", "pixel_attention_mask"}:
                batch[key] = torch.cat(values, dim=0)
            elif all(value.shape == values[0].shape for value in values):
                batch[key] = torch.stack(values)
            else:
                raise ValueError(f"No safe collate rule for Qwen input field: {key}")
        return batch


def _load_quantized_model(pin, training):
    import torch
    from transformers import BitsAndBytesConfig
    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelClass
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass

    quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = ModelClass.from_pretrained(pin["model_id"], revision=pin["revision"],
        token=False, trust_remote_code=False, quantization_config=quant_cfg,
        torch_dtype=torch.float16, device_map={"": 0}, low_cpu_mem_usage=True,
        attn_implementation="sdpa")
    model.config.use_cache = not training
    if training:
        model.config.use_cache = False
    return model


def _trainer_class():
    import torch
    import torch.nn.functional as F
    from transformers import Trainer

    class WeightedCausalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weights = inputs.pop("sample_weights")
            inputs.pop("sample_ids", None)
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            target = labels[:, 1:].contiguous()
            valid = target.ne(-100)
            # Only convert supervised answer-token logits to FP32. Prompt and
            # image positions have no target loss and need no large FP32 copy.
            token_losses = F.cross_entropy(outputs.logits[:, :-1, :][valid].float(),
                                            target[valid], reduction="none")
            per_sample = []
            offset = 0
            for count in valid.sum(dim=1).tolist():
                if count == 0:
                    raise ValueError("An SFT sample has no answer tokens")
                per_sample.append(token_losses[offset:offset+count].mean())
                offset += count
            per_sample = torch.stack(per_sample)
            weights = weights.to(per_sample.device)
            # Weights have global sample mean 1. Normalizing within a batch of
            # one would cancel class weighting completely.
            loss = (per_sample * weights).mean()
            return (loss, outputs) if return_outputs else loss

    return WeightedCausalTrainer


def train_qwen_lora(variant):
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    cfg = settings()
    qcfg = cfg["qwen3_vl"]
    if variant not in qcfg["variants"]:
        raise ValueError(f"Unknown Qwen variant: {variant}")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL QLoRA requires the Colab T4 GPU")
    folder, protocol, manifest = load_protocol()
    if (folder / "freeze.json").exists():
        raise RuntimeError("Global protocol is frozen; Qwen training cannot change")
    set_seed(int(qcfg["seed"]))
    development = manifest[manifest.split.eq("development")].reset_index(drop=True)
    if set(development.label.astype(int)) != set(range(12)):
        raise ValueError("Qwen must be trained on all 12 development labels")
    for row in development.itertuples():
        if digest(resolve(row.clean_path)) != row.clean_sha256:
            raise ValueError("Development crop changed: " + str(row.sample_id))
    pin, pin_path = _resolve_model_pin()
    model_dir = folder / "models" / "qwen3_vl" / variant
    adapter_dir = model_dir / "adapter"
    receipt_path = model_dir / "training_receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (receipt["protocol_hash"] != protocol["manifest_sha256"] or
                receipt["model_revision"] != pin["revision"] or
                receipt["training_code_hash"] != digest(ROOT / "src/uav_project/uav12_qwen.py")):
            raise ValueError("Existing Qwen adapter does not match current protocol/model/code")
        for item in receipt["adapter_files"]:
            path = resolve(item["path"])
            if not path.is_file() or digest(path) != item["sha256"]:
                raise ValueError("Existing Qwen adapter file is missing or changed: " + str(path))
        print("Qwen adapter already completed:", adapter_dir, flush=True)
        return
    processor = _load_processor(pin)
    model = _load_quantized_model(pin, training=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    targets = list(qcfg["target_modules"])
    matching = [name for name, _ in model.named_modules()
                if name.endswith(tuple("." + target for target in targets)) and "visual" not in name.lower()]
    if not matching:
        raise RuntimeError("Configured language-model LoRA modules were not found in this Qwen revision")
    adapter_config = LoraConfig(task_type="CAUSAL_LM", r=int(qcfg["lora_rank"]),
        lora_alpha=int(qcfg["lora_alpha"]), lora_dropout=float(qcfg["lora_dropout"]),
        target_modules=matching, bias="none")
    model = get_peft_model(model, adapter_config)
    model.print_trainable_parameters()
    dataset = _CropSFTDataset(development, processor, protocol["classes"], variant,
                              int(qcfg["max_length"]))
    collator = _CropCollator(processor.tokenizer)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(os.environ.get("UAV12_CHECKPOINT_ROOT",
        str(folder / "models" / "qwen3_vl" / "checkpoints"))).expanduser()
    checkpoint_dir = checkpoint_root / variant
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_config = {
        "variant": variant, "model_id": pin["model_id"], "model_revision": pin["revision"],
        "protocol_hash": protocol["manifest_sha256"], "config_hash": digest(CONFIG),
        "development_manifest_hash": digest(folder / "manifest.csv"),
        "development_samples": len(development), "development_class_counts":
            {protocol["classes"][int(key)]: int(value) for key, value in development.label.value_counts().items()},
        "epochs": int(qcfg["epochs"]), "learning_rate": float(qcfg["learning_rate"]),
        "weight_decay": float(qcfg["weight_decay"]), "per_device_train_batch_size": int(qcfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(qcfg["gradient_accumulation_steps"]),
        "target_modules": targets, "matching_lora_module_count": len(matching),
        "image_pixels": [int(qcfg["min_image_pixels"]), int(qcfg["max_image_pixels"])],
        "balance_rule": variant, "training_data_split": "development only",
        "validation": "none; fixed schedule and both predeclared variants",
        "holdout_used": False, "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "peft": importlib.metadata.version("peft"), "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "code_hashes": {relative(ROOT / "src/uav_project/uav12_qwen.py"): digest(ROOT / "src/uav_project/uav12_qwen.py"),
                        relative(ROOT / "src/uav_project/uav12_protocol.py"): digest(ROOT / "src/uav_project/uav12_protocol.py")}}
    config_path = model_dir / "training_config.json"
    if config_path.exists() and read_json(config_path) != train_config:
        # A failed first batch can leave only a provisional training_config.json.
        # It is safe to replace that incomplete receipt when no adapter receipt
        # exists; completed adapters remain immutable.
        if receipt_path.exists():
            raise ValueError("Saved Qwen training configuration differs; use a new experiment version")
        print("Replacing incomplete Qwen training configuration after a failed run.", flush=True)
    save_json(config_path, train_config)
    arguments = TrainingArguments(
        output_dir=str(checkpoint_dir), num_train_epochs=int(qcfg["epochs"]),
        per_device_train_batch_size=int(qcfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(qcfg["gradient_accumulation_steps"]),
        learning_rate=float(qcfg["learning_rate"]), weight_decay=float(qcfg["weight_decay"]),
        warmup_ratio=float(qcfg["warmup_ratio"]), lr_scheduler_type="cosine",
        optim="paged_adamw_8bit", max_grad_norm=1.0, fp16=True, bf16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=25, save_strategy="steps", save_steps=250, save_total_limit=2,
        save_safetensors=True, report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0, seed=int(qcfg["seed"]), data_seed=int(qcfg["seed"]),
        ddp_find_unused_parameters=False)
    trainer = _trainer_class()(model=model, args=arguments, train_dataset=dataset, data_collator=collator)
    trainer.model_accepts_loss_kwargs = False
    checkpoint = get_last_checkpoint(str(checkpoint_dir))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=checkpoint) if checkpoint else trainer.train()
    training_seconds = time.perf_counter() - started
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(model_dir / "processor")
    history_path = model_dir / "training_history.csv"
    pd.DataFrame(trainer.state.log_history).to_csv(history_path, index=False, encoding="utf-8-sig")
    adapter_files = []
    for path in sorted(adapter_dir.rglob("*")):
        if path.is_file():
            adapter_files.append({"path": relative(path), "sha256": digest(path), "bytes": path.stat().st_size})
    if not adapter_files:
        raise RuntimeError("Qwen adapter export did not produce files")
    receipt = {"trained_at": now(), "variant": variant,
        "model_id": pin["model_id"], "model_revision": pin["revision"],
        "protocol_hash": protocol["manifest_sha256"], "training_config_hash": digest(config_path),
        "adapter_directory": relative(adapter_dir), "adapter_files": adapter_files,
        "adapter_file_count": len(adapter_files), "development_samples": len(development),
        "holdout_used": False, "training_examples_contain_answer_label": True,
        "training_objective": "assistant-only causal cross entropy; class-weighted per-example mean" if variant == "sqrt_class_balanced" else "assistant-only causal cross entropy; unweighted per-example mean",
        "train_loss": float(result.training_loss), "global_steps": int(trainer.state.global_step),
        "session_training_seconds": training_seconds,
        "session_peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "session_peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cost_caveat": "Session timing excludes model download; resumed runs need all session logs for total cost.",
        "history_sha256": digest(history_path), "model_pin_sha256": digest(pin_path),
        "resumed_from_checkpoint": checkpoint, "environment": train_config,
        "training_code_hash": digest(ROOT / "src/uav_project/uav12_qwen.py")}
    save_json(model_dir / "training_receipt.json", receipt)
    print(f"Qwen LoRA {variant} finished: {len(adapter_files)} adapter files; holdout remains unopened.", flush=True)
    del trainer, model
    torch.cuda.empty_cache()


def _parse_label(text):
    match = re.fullmatch(r"LABEL_(\d{2})", text.strip())
    return int(match.group(1)) if match and int(match.group(1)) < 12 else -1


def predict_qwen_holdout():
    import torch
    from peft import PeftModel
    from PIL import Image, ImageOps

    from .uav12_models import verify_uav12_freeze
    verify_uav12_freeze()
    folder, protocol, manifest = load_protocol()
    holdout = manifest[manifest.split.eq("holdout")].reset_index(drop=True)
    cfg = settings()["qwen3_vl"]
    qwen_root = folder / "models" / "qwen3_vl"
    pin = read_json(qwen_root / "model_pin.json")
    variants = cfg["variants"]
    keys = ["qwen3_vl_zero_shot"] + [f"qwen3_vl_lora_{variant}" for variant in variants]
    evaluation = folder / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    progress_root = Path(os.environ.get("UAV12_PROGRESS_ROOT", str(evaluation / "qwen_resume")))
    progress_root.mkdir(parents=True, exist_ok=True)
    progress_path = progress_root / "qwen_progress.json"
    progress = read_json(progress_path) if progress_path.exists() else None
    results = {key: [] for key in keys}
    prediction_texts = {key: [] for key in keys}
    start = 0
    freeze_hash = digest(folder / "freeze.json")
    if progress:
        if progress["freeze_hash"] != freeze_hash:
            raise ValueError("Qwen partial predictions belong to a different global freeze")
        start = int(progress["completed_samples"])
        if start < 0 or start > len(holdout):
            raise ValueError("Invalid Qwen resume offset")
        for key in keys:
            path = evaluation / progress["files"][key]["filename"]
            if digest(path) != progress["files"][key]["sha256"]:
                raise ValueError("Qwen partial predictions changed")
            part = pd.read_csv(path, keep_default_na=False)
            if part.sample_id.tolist() != holdout.sample_id.iloc[:start].tolist():
                raise ValueError("Qwen partial prediction samples differ from the holdout prefix")
            results[key] = part.predicted_label.astype(int).tolist()
            prediction_texts[key] = part.prediction_text.tolist()
        print(f"Resuming Qwen predictions at sample {start}/{len(holdout)}", flush=True)
    if start == len(holdout):
        return {key: _qwen_prediction_frame(holdout, results[key], prediction_texts[key]) for key in keys}
    processor = _load_processor(pin)
    model = _load_quantized_model(pin, training=False)
    model = PeftModel.from_pretrained(model, str(qwen_root / variants[0] / "adapter"),
                                      adapter_name=variants[0], is_trainable=False)
    for variant in variants[1:]:
        model.load_adapter(str(qwen_root / variant / "adapter"), adapter_name=variant, is_trainable=False)
    model.eval()
    elapsed_prior = float(progress.get("session_inference_seconds", 0)) if progress else 0.0
    started = time.perf_counter()
    for index, row in enumerate(holdout.iloc[start:].itertuples(), start=start + 1):
        if digest(resolve(row.clean_path)) != row.clean_sha256:
            raise ValueError("Holdout crop changed: " + str(row.sample_id))
        with Image.open(resolve(row.clean_path)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        messages = _user_messages(image, protocol["classes"])
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False)
        # Transformers' Qwen3-VL example drops tokenizer-only type ids before
        # forwarding the multimodal batch to generate().
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(model.device)
        input_length = inputs["input_ids"].shape[1]
        for key, adapter in [("qwen3_vl_zero_shot", None)] + [
                (f"qwen3_vl_lora_{variant}", variant) for variant in variants]:
            if adapter is None:
                context = model.disable_adapter()
            else:
                model.set_adapter(adapter)
                context = torch.no_grad()
            with context, torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=int(cfg["max_new_tokens"]),
                                        do_sample=False, use_cache=True)
            generated = processor.batch_decode(output[:, input_length:], skip_special_tokens=True)[0].strip()
            prediction_texts[key].append(generated)
            results[key].append(_parse_label(generated))
        if index % 250 == 0 or index == len(holdout):
            print(f"Qwen frozen holdout predictions: {index}/{len(holdout)}", flush=True)
            files = {}
            for key, predicted in results.items():
                partial = pd.DataFrame({"sample_id": holdout.sample_id.iloc[:index],
                    "label": holdout.label.iloc[:index].to_numpy(), "predicted_label": predicted,
                    "prediction_text": prediction_texts[key]})
                path = progress_root / f"{key}_predictions_{index:06d}.csv"
                partial.to_csv(path, index=False, encoding="utf-8-sig")
                files[key] = {"filename": path.name, "sha256": digest(path)}
            save_json(progress_path, {"freeze_hash": freeze_hash, "completed_samples": index,
                "files": files, "saved_at": now(),
                "session_inference_seconds": elapsed_prior + time.perf_counter()-started})
            for old in progress_root.glob("*_predictions_*.csv"):
                if all(old.name != record["filename"] for record in files.values()):
                    old.unlink()
    records = {}
    for key in results:
        records[key] = _qwen_prediction_frame(holdout, results[key], prediction_texts[key])
    del model
    torch.cuda.empty_cache()
    return records


def _qwen_prediction_frame(holdout, predictions, texts):
    frame = holdout[["sample_id", "label", "class_name", "group_id"]].copy()
    frame["predicted_label"] = predictions
    frame["prediction_text"] = texts
    return frame
