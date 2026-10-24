"""Transport verification and per-stage Drive persistence for the T4 notebook."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from .common import digest, read_json, save_json, now


def _unpack_checked(archive_path, root):
    root = Path(root).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root) or "\\" in item.filename or ":" in item.filename:
                raise ValueError("Unsafe archive path: " + item.filename)
            if ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("Archive symlinks are not allowed")
        archive.extractall(root)


def unpack_images(transfer, root, receipt):
    transfer, root = Path(transfer), Path(root)
    archive_record = receipt["archives"]["uav12_images.zip"]
    marker = root / ".uav12_images_unpacked.json"
    if marker.exists() and read_json(marker).get("archive_sha256") == archive_record["sha256"]:
        print("Images already unpacked; per-stage image hashes will still be checked.", flush=True)
        return
    archive_path = transfer / "uav12_images.zip"
    if not archive_path.is_file() or digest(archive_path) != archive_record["sha256"]:
        raise ValueError("Image archive missing or upload is incomplete")
    cache_dir = Path("/content/uav12_transfer_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_archive = cache_dir / "uav12_images.zip"
    shutil.copyfile(archive_path, local_archive)
    if digest(local_archive) != archive_record["sha256"]:
        raise ValueError("Copied image archive hash mismatch")
    _unpack_checked(local_archive, root)
    save_json(marker, {"archive_sha256": archive_record["sha256"], "unpacked_at": now(),
                       "image_count": archive_record["files"]})
    print("Image archive unpacked:", archive_record["files"], flush=True)


class ColabUAV12:
    def __init__(self, root, transfer, transfer_id, experiment_path):
        self.root = Path(root)
        self.transfer = Path(transfer)
        self.run_relative = Path(experiment_path)
        self.local = self.root / self.run_relative
        self.remote = self.transfer / "results" / transfer_id
        self.remote.mkdir(parents=True, exist_ok=True)
        self.checkpoints = self.remote / "qwen_checkpoints"
        self.checkpoints.mkdir(parents=True, exist_ok=True)

    def restore(self):
        prior = self.remote / "experiment"
        if prior.is_dir():
            shutil.copytree(prior, self.local, dirs_exist_ok=True)
            print("Restored completed artifacts from Drive:", prior, flush=True)
        os.environ["ML_PROJECT_ROOT"] = str(self.root)
        os.environ["UAV12_CHECKPOINT_ROOT"] = str(self.checkpoints)
        os.environ["UAV12_PROGRESS_ROOT"] = str(self.remote / "qwen_progress")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["USE_TF"] = "0"
        os.environ["USE_FLAX"] = "0"

    def record_environment(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime in Colab before continuing")
        packages = {}
        for package in ["torch", "torchvision", "transformers", "peft", "accelerate", "bitsandbytes",
                        "numpy", "pandas", "scikit-learn", "scikit-image", "Pillow"]:
            packages[package] = importlib.metadata.version(package)
        environment = {"recorded_at": now(), "python": sys.version, "packages": packages,
            "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda": torch.version.cuda, "compute_capability": list(torch.cuda.get_device_capability(0))}
        old_path = self.local / "colab_environment.json"
        if old_path.exists():
            previous = read_json(old_path)
            if previous["packages"] != packages:
                raise RuntimeError("Package versions changed since this experiment began; restore the prior environment")
        save_json(old_path, environment)
        freeze_text = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (self.local / "colab_pip_freeze.txt").write_text(freeze_text, encoding="utf-8")
        self.sync()
        print(json.dumps({"gpu": environment["gpu"], "cuda": environment["cuda"],
                          "memory_GiB": round(environment["gpu_total_bytes"]/1024**3, 2)}, indent=2), flush=True)
        return environment

    def sync(self):
        destination = self.remote / "experiment"
        destination.mkdir(parents=True, exist_ok=True)
        # Artifacts are copied only between stages; Qwen optimizer checkpoints
        # are written directly to Drive by Trainer during the long SFT stage.
        for source in self.local.rglob("*"):
            relative_path = source.relative_to(self.local)
            parts = relative_path.parts
            if not source.is_file() or any(piece == "checkpoints" for piece in parts):
                continue
            if (len(parts) >= 4 and parts[:4] == ("data", "final", "uav_type_v3", "images")):
                continue
            if len(parts) >= 2 and parts[:2] == ("data", "raw"):
                continue
            if source.suffix in {".tmp", ".partial"}:
                continue
            target = destination / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == source.stat().st_size:
                if digest(target) == digest(source):
                    continue
            shutil.copy2(source, target)

    def run(self, script, *arguments):
        if Path(script).name != script or not script.endswith(".py"):
            raise ValueError("Use a Python entrypoint filename from scripts/")
        command = [sys.executable, "-u", str(self.root / "scripts" / script), *map(str, arguments)]
        print("Running:", " ".join(command), flush=True)
        try:
            subprocess.run(command, cwd=self.root, env=os.environ.copy(), check=True)
        finally:
            self.sync()
            print("Artifacts saved to:", self.remote, flush=True)

    def export_results(self):
        self.sync()
        output = self.root / "uav12_results_for_report.zip"
        selected = []
        for path in self.local.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in {".csv", ".json", ".jsonl", ".txt", ".md", ".png", ".jpg"}:
                continue
            parts = path.relative_to(self.local).parts
            if any(piece in {"features", "checkpoints", "processor", "images", "raw"} for piece in parts):
                continue
            selected.append(path)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(selected):
                archive.write(path, path.relative_to(self.root).as_posix())
        print("Report results package:", output, flush=True)
        print("Adapter weights, feature matrices and resumable checkpoints remain on Drive.", flush=True)
        return output
