"""Restore verified prediction CSVs at the path expected by a frozen evaluator.

This file repair uses only the standard library and never starts GPU inference.
It deliberately leaves the frozen training/evaluation source and receipts intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def restore_prediction_files(root, progress_root):
    root, progress_root = Path(root).resolve(), Path(progress_root).resolve()
    config = json.loads((root / "configs/uav12_experiment.json").read_text(encoding="utf-8-sig"))
    folder = (root / config["run_directory"]).resolve()
    progress = json.loads((progress_root / "qwen_progress.json").read_text(encoding="utf-8-sig"))
    if sha256(folder / "freeze.json") != progress["freeze_hash"]:
        raise ValueError("Prediction progress belongs to a different experiment freeze")
    expected = {"qwen3_vl_zero_shot"} | {
        "qwen3_vl_lora_" + name for name in config["qwen3_vl"]["variants"]}
    if set(progress["files"]) != expected:
        raise ValueError("Unexpected prediction variant set")
    completed = int(progress["completed_samples"])
    if completed <= 0:
        raise ValueError("No saved prediction prefix to restore")
    evaluation = folder / "evaluation"
    planned = []
    # Validate every source before copying any file. The evaluator additionally
    # checks the CSV sample IDs against the frozen holdout prefix on resume.
    for key, record in progress["files"].items():
        filename = record["filename"]
        if filename != f"{key}_predictions_{completed:06d}.csv":
            raise ValueError("Unexpected prediction filename")
        source, destination = progress_root / filename, evaluation / filename
        if sha256(source) != record["sha256"]:
            raise ValueError("Saved prediction hash mismatch: " + filename)
        if destination.exists() and sha256(destination) != record["sha256"]:
            raise ValueError("Conflicting local prediction; preserving it: " + filename)
        planned.append((source, destination, record["sha256"]))
    evaluation.mkdir(parents=True, exist_ok=True)
    for source, destination, expected_hash in planned:
        if not destination.exists():
            shutil.copy2(source, destination)
        if sha256(destination) != expected_hash:
            raise ValueError("Restored prediction hash mismatch: " + destination.name)
        print("Verified:", destination.name, flush=True)
    print(f"Ready to resume after {completed} saved samples. Run notebook cell 14 only.", flush=True)
    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ.get("ML_PROJECT_ROOT", Path(__file__).resolve().parents[1])))
    parser.add_argument("--progress-root", type=Path,
                        default=os.environ.get("UAV12_PROGRESS_ROOT"))
    args = parser.parse_args()
    if args.progress_root is None:
        parser.error("Set UAV12_PROGRESS_ROOT or pass --progress-root")
    restore_prediction_files(args.root, args.progress_root)
