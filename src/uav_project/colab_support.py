"""Colab-only helpers: verified image unpacking and per-step Drive persistence.

These utilities do not contain feature/model implementations. Every actual stage
continues to run its independent scripts/NN*.py entrypoint in a subprocess.
"""
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def extract_checked(archive_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            target = (destination / entry.filename).resolve()
            mode = (entry.external_attr >> 16) & 0o170000
            if not target.is_relative_to(destination) or mode == 0o120000 or "\\" in entry.filename:
                raise ValueError("Unsafe archive entry: " + entry.filename)
        archive.extractall(destination)


def unpack_image_transfer(transfer_dir, project_root, receipt):
    transfer_dir, project_root = Path(transfer_dir), Path(project_root)
    spec = receipt["archives"]["auxiliary_images.zip"]
    cache = Path("/content/ml_hw_transfer_cache")
    cache.mkdir(parents=True, exist_ok=True)
    copied = cache / "auxiliary_images.zip"
    if not copied.exists() or file_hash(copied) != spec["sha256"]:
        print("Copying one image archive from Drive to runtime SSD...", flush=True)
        shutil.copyfile(transfer_dir / "auxiliary_images.zip", copied)
    if file_hash(copied) != spec["sha256"]:
        raise ValueError("Image archive hash mismatch")
    marker = project_root / "outputs/colab_unpack_receipt.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8"))["archive_sha256"] == spec["sha256"]:
        print("Image archive was already unpacked in this runtime.", flush=True)
        return
    print("Unpacking images to runtime SSD (not Drive)...", flush=True)
    extract_checked(copied, project_root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"archive_sha256": spec["sha256"], "unpacked_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")


class ColabRun:
    def __init__(self, project_root, transfer_dir, transfer_id):
        self.root = Path(project_root)
        self.remote = Path(transfer_dir) / "results" / transfer_id
        self.remote.mkdir(parents=True, exist_ok=True)
        self.targets = ["outputs/representation_v1/auxiliary", "outputs/logs", "outputs/colab_environment.json"]
        self.batch_size = None

    def restore(self):
        for name in self.targets:
            source = self.remote / name
            target = self.root / name
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        print("Restored prior step artifacts, if any:", self.remote, flush=True)

    def sync(self):
        for name in self.targets:
            source = self.root / name
            target = self.remote / name
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("*.tmp", "*.tmp.npz", "__pycache__"))
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        print("Saved completed/partial stage artifacts to:", self.remote, flush=True)

    def record_environment(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Colab未分配GPU：运行时→更改运行时类型→GPU，然后重新连接。")
        gpu = torch.cuda.get_device_properties(0)
        total_gib = gpu.total_memory / 1024 ** 3
        # Batch size affects inference throughput; the actual dtype is recorded
        # and development/holdout dtype consistency is checked by evaluation.
        self.batch_size = 64 if total_gib >= 30 else 16
        packages = {}
        for name in ["torch", "transformers", "huggingface-hub", "numpy", "pandas", "scikit-learn", "scikit-image", "Pillow", "joblib"]:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "MISSING"
        info = {"recorded_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "platform": platform.platform(), "gpu": gpu.name, "gpu_memory_gib": total_gib,
                "cuda_runtime": torch.version.cuda, "batch_size": self.batch_size,
                "inference_dtype": "bfloat16" if (torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()) else "float16",
                "packages": packages, "environment_only_no_model_run": True}
        path = self.root / "outputs/colab_environment.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        self.sync()
        return info

    def run(self, script, *arguments):
        allowed = {"06b_extract_representation.py", "07b_train_representation.py", "09b_freeze_representation.py",
                   "10b_evaluate_representation.py", "11b_export_representation.py"}
        if script not in allowed:
            raise ValueError("Unexpected Colab stage: " + script)
        if script == "07b_train_representation.py" and (self.root / "outputs/representation_v1/auxiliary/freeze.json").exists():
            from .representation_training import verify_freeze
            verify_freeze("auxiliary")
            print("模型已经冻结并核对一致，跳过已完成的训练步骤。", flush=True)
            return
        command = [sys.executable, "-u", str(self.root / "scripts" / script), *map(str, arguments)]
        print("Running:", " ".join(command), flush=True)
        stage_failed = False
        try:
            subprocess.run(command, cwd=self.root, check=True)
        except BaseException:
            stage_failed = True
            raise
        finally:
            # Preserve the original error if Drive becomes unavailable as well.
            try:
                self.sync()
            except Exception as exc:
                print("Drive sync failed; keep this runtime alive and retry session.sync():", repr(exc), flush=True)
                if not stage_failed:
                    raise


def export_report_bundle(session):
    folder = session.root / "outputs/representation_v1/auxiliary"
    if not (folder / "evaluation/measured_results.json").exists():
        raise RuntimeError("No measured evaluation exists; do not export an empty results package")
    selected = [folder / name for name in ["protocol.json", "cv_folds.json", "freeze.json", "results.md"]]
    selected += list((folder / "evaluation").glob("*")) + list((folder / "figures").glob("*.png"))
    selected += list((folder / "models").glob("*/*.json"))
    selected += list((folder / "models").glob("*/*.csv"))
    selected += list((folder / "features").glob("*/*.json"))
    selected += [session.root / "outputs/colab_environment.json"]
    target = session.root / "outputs/auxiliary_results_for_report.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(selected)):
            if path.is_file():
                archive.write(path, path.relative_to(session.root).as_posix())
    session.sync()
    shutil.copy2(target, session.remote / target.name)
    print("Report evidence package:", target, "\nDrive copy:", session.remote / target.name, flush=True)
    return target


