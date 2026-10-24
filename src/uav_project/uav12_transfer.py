"""Export modular source and crop transport archives without executing models."""
import json
from pathlib import Path
import zipfile

from .common import ROOT, digest, read_json, save_json, now, relative, resolve, write_rows
from .uav12_protocol import CONFIG, load_protocol


def _notebook(transfer_id, experiment_path):
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text + "\n"})

    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": text + "\n"})

    md("# 十二类主任务：T4 分步骤运行\n\n每格调用独立的 `.py` 文件。数据已准备，无需重采集。开发14106张、留出5904张；分组CV评分覆盖6类，留出覆盖8类，不能宣称完整十二类独立评测。\n\n将 `uav12_code.zip`、`uav12_images.zip`、`transfer_receipt.json` 放入 Drive 的 `MyDrive/ml_hw_uav12/`。选择GPU运行时，按顺序执行。压缩包只用于传输，内部保留模块目录。")
    md("## 1 挂载 Drive")
    code("from google.colab import drive\ndrive.mount('/content/drive')\nfrom pathlib import Path\nTRANSFER = Path('/content/drive/MyDrive/ml_hw_uav12')\nROOT = Path('/content/ml_hw_uav12')")
    md("## 2 核对并展开代码")
    code("import hashlib, json, os, shutil, sys, zipfile\nreceipt = json.loads((TRANSFER/'transfer_receipt.json').read_text(encoding='utf-8-sig'))\ncode_zip = TRANSFER/'uav12_code.zip'\ndef file_hash(path):\n    digest = hashlib.sha256()\n    with path.open('rb') as stream:\n        for chunk in iter(lambda: stream.read(1024*1024), b''):\n            digest.update(chunk)\n    return digest.hexdigest()\nif file_hash(code_zip) != receipt['archives']['uav12_code.zip']['sha256']:\n    raise ValueError('代码包上传不完整或版本不一致')\nROOT.mkdir(parents=True, exist_ok=True)\nwith zipfile.ZipFile(code_zip) as archive:\n    for item in archive.infolist():\n        target = (ROOT/item.filename).resolve()\n        if not target.is_relative_to(ROOT.resolve()) or '\\\\' in item.filename or ':' in item.filename or ((item.external_attr >> 16)&0o170000)==0o120000:\n            raise ValueError('不安全的压缩包路径')\n    archive.extractall(ROOT)\nos.environ['ML_PROJECT_ROOT'] = str(ROOT)\nos.environ['USE_TF'] = '0'\nos.environ['USE_FLAX'] = '0'\nsys.path.insert(0, str(ROOT/'src'))\nprint('代码已展开：', ROOT)")
    md("## 3 安装同一套依赖\n\n在模型步骤之前执行一次；不要在开发和留出之间升级包。保留 Colab 已有的兼容 CUDA torch/torchvision。")
    code("import subprocess\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT/'requirements-uav12-qwen.txt')], check=True)")
    md("## 4 展开图片、恢复结果、核对实际GPU\n\nT4默认FP16、单图训练、梯度累积8步。Qwen每250个优化步直接保存可恢复检查点到Drive，其余产物每一步结束后同步。断线后重新执行1至4，再运行尚未完成的步骤。")
    code("from uav_project.uav12_colab import unpack_images, ColabUAV12\nunpack_images(TRANSFER, ROOT, receipt)\nsession = ColabUAV12(ROOT, TRANSFER, receipt['transfer_id'], receipt['experiment_path'])\nsession.restore()\nsession.record_environment()")
    stages = [
        ("5 开发集手工特征", "06d_extract_uav12_features.py", ["--representation", "handcrafted"], "CPU步骤，只处理开发集。"),
        ("6 开发集 SigLIP2 表征", "06d_extract_uav12_features.py", ["--representation", "siglip2", "--batch-size", "16"], "冻结编码器、固定模型commit；GPU只提取特征。显存不足可将16改8重跑本格。"),
        ("7 手工特征四分类器和网格搜索", "07d_train_uav12_baselines.py", ["--representation", "handcrafted", "--jobs", "2"], "开发集两折分组CV，6类评分、12类训练；标准化和筛选在训练折内拟合。"),
        ("8 SigLIP2表征四分类器和网格搜索", "07d_train_uav12_baselines.py", ["--representation", "siglip2", "--jobs", "2"], "与上一格使用相同分组折和评分范围。CPU运行是正常现象。"),
        ("9 Qwen普通监督微调", "07q_train_qwen3vl.py", ["--variant", "standard"], "QLoRA只适配语言模块；视觉骨干冻结；固定一轮开发集监督学习，不使用留出集挑检查点。首次下载基础权重。"),
        ("10 Qwen类别加权监督微调", "07q_train_qwen3vl.py", ["--variant", "sqrt_class_balanced"], "与普通组从同一基础模型开始，使用同一样本和步数，仅改变归一化的平方根逆频率损失权重。不是强化学习。"),
        ("11 全局冻结", "09d_freeze_uav12.py", [], "全部已声明模型完成才允许冻结。冻结后不能更改配置、提示词、训练参数或权重。"),
        ("12 冻结后的留出手工特征", "06d_extract_uav12_features.py", ["--representation", "handcrafted", "--partition", "holdout"], "未冻结时程序拒绝读取。"),
        ("13 冻结后的留出SigLIP2表征", "06d_extract_uav12_features.py", ["--representation", "siglip2", "--partition", "holdout", "--batch-size", "16"], "模型commit、代码和数据身份与开发阶段对应。"),
        ("14 一次性冻结评估", "10d_evaluate_uav12.py", [], "报告全部传统模型、原始Qwen和两个LoRA版本；非法输出按错误计入。每250张保存Qwen中间预测以支持续跑。"),
        ("15 结果表与图", "11d_export_uav12_report.py", [], "从实际输出生成CSV、混淆矩阵、特征重要性及Markdown，不生成PDF。"),
    ]
    for title, script, arguments, description in stages:
        md("## " + title + "\n\n" + description)
        code("session.run(" + ", ".join(repr(value) for value in [script, *arguments]) + ")")
    md("## 16 下载结果\n\n把结果ZIP交回来，用于继续写论文。模型适配器、特征矩阵和训练检查点继续保留在Drive，不必打包下载。")
    code("from google.colab import files\nresult_zip = session.export_results()\nfiles.download(str(result_zip))")
    md("本Notebook只生成，未在本地执行训练、模型推理或软件测试。首次GPU运行仍需确认依赖与显存；出现报错停在对应格，发完整错误，修复后按检查点续跑。当前是单种子、固定训练预算的课程研究实验，不能提前声称性能提升或论文创新。")
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "colab": {"name": "UAV12_T4.ipynb", "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}, "accelerator": "GPU"}, "cells": cells}


def export_uav12_transfer():
    folder, protocol, frame = load_protocol()
    destination = ROOT / "outputs/colab_uav12"
    destination.mkdir(parents=True, exist_ok=True)
    source_files = set()
    for path in sorted((ROOT / "src/uav_project").glob("uav12_*.py")):
        source_files.add(path)
    for name in ["__init__.py", "common.py", "features.py", "splitting.py"]:
        source_files.add(ROOT / "src/uav_project" / name)
    for name in ["_bootstrap.py", "05d_prepare_uav12_protocol.py", "06d_extract_uav12_features.py",
                 "07d_train_uav12_baselines.py", "07q_train_qwen3vl.py", "09d_freeze_uav12.py",
                 "10d_evaluate_uav12.py", "11d_export_uav12_report.py", "12d_export_uav12_transfer.py"]:
        source_files.add(ROOT / "scripts" / name)
    for path in [CONFIG, ROOT / "requirements-uav12.txt", ROOT / "requirements-uav12-qwen.txt",
                 ROOT / "docs/UAV12_EXPERIMENT.md", ROOT / "docs/UAV12_COLAB.md"]:
        source_files.add(path)
    for path in protocol["input_hashes"]:
        source_files.add(resolve(path))
    for name in ["manifest.csv", "protocol.json", "cv_folds.json", "class_split_counts.csv"]:
        source_files.add(folder / name)
    image_files = {resolve(row.clean_path): row.clean_sha256 for row in frame.itertuples()}
    archives = {}
    for filename, paths, images in [("uav12_code.zip", sorted(source_files, key=lambda path: str(path)), False),
                                    ("uav12_images.zip", sorted(image_files, key=lambda path: str(path)), True)]:
        archive_path = destination / filename
        members_path = destination / (filename + ".members.json")
        members = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            expected = image_files[path] if images else None
            actual_hash = digest(path)
            if expected and actual_hash != expected:
                raise ValueError("Image changed since data release: " + relative(path))
            members.append({"path": relative(path), "sha256": actual_hash, "bytes": path.stat().st_size})
        reuse = (members_path.exists() and archive_path.exists() and
                 read_json(members_path).get("members") == members and
                 read_json(members_path).get("archive_sha256") == digest(archive_path))
        if not reuse:
            temporary = archive_path.with_suffix(".partial")
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED if images else zipfile.ZIP_DEFLATED) as archive:
                for index, path in enumerate(paths, start=1):
                    archive.write(path, relative(path))
                    if images and index % 4000 == 0:
                        print(f"Image transport files: {index}/{len(paths)}", flush=True)
            temporary.replace(archive_path)
            save_json(members_path, {"members": members, "archive_sha256": digest(archive_path)})
        archives[filename] = {"sha256": digest(archive_path), "bytes": archive_path.stat().st_size,
                              "files": len(paths)}
    transfer_id = "uav12_" + archives["uav12_code.zip"]["sha256"][:12]
    receipt = {"created_at": now(), "transfer_id": transfer_id, "archives": archives,
        "experiment_path": relative(folder), "images": len(frame), "class_count": 12,
        "development_samples": protocol["counts"]["development"],
        "holdout_samples": protocol["counts"]["holdout"],
        "cv_scored_class_count": len(protocol["development_cv_labels"]),
        "holdout_class_count": len(protocol["holdout_labels"]),
        "all_12_class_independent_evaluation": False,
        "models_or_software_tests_run": False, "uploaded": False,
        "purpose": "transport only; source remains modular; execute the independent scripts in order"}
    notebook = _notebook(transfer_id, relative(folder))
    notebook_text = json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"
    notebook_path = ROOT / "notebooks/UAV12_T4.ipynb"
    notebook_path.write_text(notebook_text, encoding="utf-8")
    packaged_notebook = destination / "UAV12_T4.ipynb"
    packaged_notebook.write_text(notebook_text, encoding="utf-8")
    receipt["notebook"] = {"path": "UAV12_T4.ipynb", "sha256": digest(packaged_notebook),
                           "bytes": packaged_notebook.stat().st_size}
    save_json(destination / "transfer_receipt.json", receipt)
    print("Prepared transport:", destination, flush=True)
    print("Notebook:", notebook_path, flush=True)
    print("Upload 4 files (2 archives, receipt, notebook) to Drive/MyDrive/ml_hw_uav12; open UAV12_T4.ipynb in Colab.", flush=True)
    print("Local model execution: none.", flush=True)

