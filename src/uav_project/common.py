from __future__ import annotations
import csv, hashlib, json, os, platform, subprocess, sys, time, traceback, zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("ML_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT = ROOT / "outputs"
META = ROOT / "data" / "metadata"
CFG = ROOT / "configs" / "experiment.json"


def setup():
    for p in [OUT, META, OUT / "exports", OUT / "logs", OUT / "models", OUT / "figures"]:
        p.mkdir(parents=True, exist_ok=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def config():
    return json.loads(CFG.read_text(encoding="utf-8-sig"))


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    if not Path(path).exists(): return []
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, data, fields=None):
    data = list(data); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in data for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(data)


def append_log(name, record):
    setup()
    with (OUT / "logs" / (name + ".jsonl")).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now(), **record}, ensure_ascii=False, default=str) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""): h.update(b)
    return h.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def resolve(path):
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT): raise ValueError("Path outside project: " + str(path))
    return p


def protect_frozen():
    if (OUT / "protocol_lock.json").exists():
        raise RuntimeError("实验方案已冻结。不要覆盖数据或训练结果；先完成最终评估。另开研究须使用新项目副本。")


def bundle(stage):
    setup()
    dest = OUT / "exports" / (stage + "_handoff.zip")
    paths = [CFG, ROOT / "docs" / "RUNNING.md"]
    paths += list(META.glob("*.csv")) + list(META.glob("*.json"))
    paths += list((OUT / "logs").glob("*"))
    paths += list(OUT.glob("*.json")) + list(OUT.glob("*.csv")) + list(OUT.glob("*.md")) + list(OUT.glob("*.txt"))
    for folder in ["review", "figures", "predictions"]:
        paths += list((OUT / folder).rglob("*")) if (OUT / folder).exists() else []
    paths += list((OUT / "models").glob("*/history.csv"))
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(paths)):
            if p.is_file(): z.write(p, relative(p))
    print("回传文件：", dest, flush=True)
    return dest


def environment():
    setup()
    import importlib.metadata as m
    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "torch", "torchvision", "Pillow", "requests", "gdown", "scikit-image"]:
        try: packages[name] = m.version(name)
        except m.PackageNotFoundError: packages[name] = "MISSING"
    info = {"time": now(), "python": sys.version, "platform": platform.platform(), "packages": packages}
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except ImportError:
        info["cuda"] = False
        info["gpu"] = "PyTorch not installed (not required for collection)"
    save_json(OUT / "environment.json", info)
    (OUT / "environment_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))



def run_stage(name, action):
    """Record a stage outcome; never start the next stage automatically."""
    setup()
    allowed = config().get("enabled_stages")
    if allowed is not None and name not in allowed:
        raise RuntimeError(
            "该步骤尚未适配当前七类机型任务，已停止执行。"
            "本阶段开放数据获取、审核和清洗准备，尚未开放划分与训练，见docs/RUNNING.md。"
        )
    append_log("stages", {"stage": name, "status": "started"})
    try:
        result = action()
    except Exception as exc:
        append_log("errors", {"stage": name, "error": repr(exc), "traceback": traceback.format_exc()})
        print("本步骤未完成。日志：", OUT / "logs" / "errors.jsonl", flush=True)
        raise
    append_log("stages", {"stage": name, "status": "completed"})
    return result



