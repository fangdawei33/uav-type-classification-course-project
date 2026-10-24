"""Export two transport archives and a per-stage Colab notebook.

The archives retain normal Python modules and CSVs. Nothing is trained, tested,
installed or downloaded by this local packaging step.
"""
import json
import shutil
import textwrap
import zipfile
from pathlib import Path
from .common import ROOT, OUT, META, rows, read_json, write_rows, save_json, digest, relative, now
from .gpu_handoff import prepare_gpu_handoff


def _notebook():
    cells = []

    def markdown(text):
        cells.append({"cell_type": "markdown", "id": f"cell-{len(cells):02d}", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"})

    def code(text):
        cells.append({"cell_type": "code", "id": f"cell-{len(cells):02d}", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": textwrap.dedent(text).strip() + "\n"})

    markdown('''
    # Colab 分步骤入口：四类辅助表征实验

    **这个Notebook不是七类主实验最终结果。** 它只运行已经准备好的Cranfield四类辅助数据：开发3999张、留出3999张。渲染父图未知，来源独立性不成立；不将辅助标签强行映射成七类。

    截图会话显示A100 40GB；代码按当前会话实际检测设备，显存≥30GiB用推理批量64，否则用16。现有GPU无需更换。每格调用单独的`.py`模块入口，按顺序运行。

    先把`project_code.zip`、`auxiliary_images.zip`和`transfer_receipt.json`三个文件上传至Google Drive的`MyDrive/ml_hw_colab/`，然后运行下面各格。代码包保留模块目录，图片包只用于传输，不合成单文件程序。
    ''')
    markdown('''
    ## 1 挂载Google Drive
    这一步需要你在Colab界面完成Google账号授权。项目计算在`/content/ml_hw`本地SSD进行；每个步骤结束或报错后，结果同步回Drive。
    ''')
    code('''
    from google.colab import drive
    drive.mount('/content/drive')
    from pathlib import Path
    TRANSFER = Path('/content/drive/MyDrive/ml_hw_colab')
    ROOT = Path('/content/ml_hw')
    ''')
    markdown('''
    ## 2 校验并展开代码包
    如果提示缺少文件，检查三个传输文件是否放到了上面指定的同一Drive目录。图片下一步再展开，不需要重新采集。
    ''')
    code('''
    import hashlib, json, os, shutil, sys, zipfile
    receipt = json.loads((TRANSFER / 'transfer_receipt.json').read_text(encoding='utf-8-sig'))
    local_cache = Path('/content/ml_hw_transfer_cache')
    local_cache.mkdir(parents=True, exist_ok=True)
    local_code = local_cache / 'project_code.zip'
    shutil.copyfile(TRANSFER / 'project_code.zip', local_code)
    if hashlib.sha256(local_code.read_bytes()).hexdigest() != receipt['archives']['project_code.zip']['sha256']:
        raise ValueError('代码包SHA256不一致：请检查上传是否完成')
    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_code) as archive:
        for item in archive.infolist():
            target = (ROOT / item.filename).resolve()
            if not target.is_relative_to(ROOT.resolve()) or ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('不安全的归档路径')
        archive.extractall(ROOT)
    sys.path.insert(0, str(ROOT / 'src'))
    os.environ['ML_PROJECT_ROOT'] = str(ROOT)
    print('代码目录：', ROOT, '传输版本：', receipt['transfer_id'])
    ''')
    markdown('''
    ## 3 安装建模依赖
    使用当前Colab的CUDA版PyTorch；已满足版本要求的安装会保留。不要安装本地采集环境的`requirements-auxiliary.txt`。
    ''')
    code('''
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-representation.txt')], check=True)
    ''')
    markdown('''
    ## 4 展开图片、恢复已保存结果并记录真实GPU
    首次复制约0.87GB图片包并在SSD展开，断线重连后再次运行1—4。推理精度和包版本会写入环境记录。若开发与留出阶段的精度/版本不同，程序会停止，避免混用实验结果。
    ''')
    code('''
    from uav_project.colab_support import unpack_image_transfer, ColabRun
    unpack_image_transfer(TRANSFER, ROOT, receipt)
    session = ColabRun(ROOT, TRANSFER, receipt['transfer_id'])
    session.restore()
    environment = session.record_environment()
    print('后续每个步骤的结果保存至：', session.remote)
    ''')
    markdown('''
    ## 5 手工特征：开发集
    入口06b。HOG、HSV、LBP等固定特征；这里只处理开发集。该步骤使用CPU。
    ''')
    code("session.run('06b_extract_representation.py', '--dataset', 'auxiliary', '--representation', 'handcrafted')")
    markdown('''
    ## 6 视觉语言预训练图像编码器：开发集
    入口06b。首次需要下载模型；模型commit被固定并记录。编码器冻结，不训练其参数。若显存不足，将下面`session.batch_size`改为`8`后只重跑本格；已有完整分片可以续用。
    ''')
    code("session.run('06b_extract_representation.py', '--dataset', 'auxiliary', '--representation', 'siglip2', '--batch-size', session.batch_size)")
    markdown('''
    ## 7 手工特征的四个分类器与网格搜索
    入口07b。逻辑回归、KNN、RBF-SVM、随机森林分别训练；折内填补/缩放/方差筛选。根据开发CV Macro-F1选出家族做网格搜索。这个步骤主要使用CPU，GPU使用率低是正常现象。
    ''')
    code("session.run('07b_train_representation.py', '--dataset', 'auxiliary', '--representation', 'handcrafted', '--jobs', '2')")
    markdown('''
    ## 8 预训练表征的同样四个分类器与网格搜索
    入口07b。它是与上一格相同分类器条件下的表征对照，不把四个模型融合。
    ''')
    code("session.run('07b_train_representation.py', '--dataset', 'auxiliary', '--representation', 'siglip2', '--jobs', '2')")
    markdown('''
    ## 9 冻结模型和方案
    入口09b。此时才允许打开留出集；后续不能根据留出成绩重选提示词、参数或模型。
    ''')
    code("session.run('09b_freeze_representation.py', '--dataset', 'auxiliary')")
    markdown('''
    ## 10 手工特征：冻结后的留出集
    入口06b。未冻结时程序会拒绝执行。
    ''')
    code("session.run('06b_extract_representation.py', '--dataset', 'auxiliary', '--representation', 'handcrafted', '--partition', 'holdout')")
    markdown('''
    ## 11 同一个预训练编码器：冻结后的留出集
    入口06b。使用开发阶段固定的模型commit；开发和留出提取器签名必须一致。
    ''')
    code("session.run('06b_extract_representation.py', '--dataset', 'auxiliary', '--representation', 'siglip2', '--partition', 'holdout', '--batch-size', session.batch_size)")
    markdown('''
    ## 12 最终留出评估
    入口10b。报告全部预先声明的模型，不按留出结果挑选最好看的模型。辅助集来源未知，不给出伪装为来源独立的置信区间。
    ''')
    code("session.run('10b_evaluate_representation.py', '--dataset', 'auxiliary')")
    markdown('''
    ## 13 生成实际结果表与图
    入口11b。只有存在实测结果才导出CSV、混淆矩阵、特征重要性和Markdown；不生成PDF。
    ''')
    code("session.run('11b_export_representation.py', '--dataset', 'auxiliary')")
    markdown('''
    ## 14 下载给我的结果包
    这里只打包小体积结果证据，不包含图片、特征大矩阵或模型权重；这些会继续保留在Drive结果目录。下载后发回来用于写报告，不能把本辅助四分类结果当作七类真实测试结果。
    ''')
    code('''
    from uav_project.colab_support import export_report_bundle
    from google.colab import files
    evidence_zip = export_report_bundle(session)
    files.download(str(evidence_zip))
    ''')
    markdown('''
    出错时停在出错的那一格，发完整报错。对应`outputs/logs/errors.jsonl`会尽可能同步至Drive。修复后重跑对应步骤即可，不必重采集。断线会丢失当前尚未同步的步骤进度，完整步骤产物和已保存的分片可恢复；不承诺断电瞬间的全部进度都能保留。

    本Notebook在本机仅生成，未执行其中的依赖安装、模型推理、训练、测试或评估。第一次Colab实际运行仍需要验证运行环境/API兼容性。
    ''')
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "colab": {"name": "COLAB_RUN.ipynb", "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}, "accelerator": "GPU"}, "cells": cells}


def export_colab_transfer():
    prepare_gpu_handoff(include_candidates=False)
    manifest_path = OUT / "gpu_handoff/t4_auxiliary_only_v1/transfer_manifest.csv"
    records = rows(manifest_path)
    destination = OUT / "colab_transfer"
    destination.mkdir(parents=True, exist_ok=True)
    archives = {}
    for filename, images in [("project_code.zip", False), ("auxiliary_images.zip", True)]:
        selected = [r for r in records if r["role"].startswith("auxiliary_") and r["role"].endswith("_image")] if images else [r for r in records if not (r["role"].startswith("auxiliary_") and r["role"].endswith("_image"))]
        target = destination / filename
        membership = [{"path": r["path"], "sha256": r["sha256"], "bytes": int(r["bytes"])} for r in selected]
        previous_path = destination / (filename + ".members.json")
        # Resume packaging only if the exact prior membership and archive hash match.
        reusable = False
        if target.exists() and previous_path.exists():
            previous = read_json(previous_path)
            reusable = previous["members"] == membership and previous["archive_sha256"] == digest(target)
        if not reusable:
            temporary = target.with_suffix(".partial")
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED if images else zipfile.ZIP_DEFLATED) as archive:
                for i, record in enumerate(selected, 1):
                    path = ROOT / record["path"]
                    if path.stat().st_size != int(record["bytes"]):
                        raise ValueError("File size changed while packaging: " + record["path"])
                    archive.write(path, record["path"])
                    if images and i % 2000 == 0:
                        print(f"Image transport archive {i}/{len(selected)}", flush=True)
            temporary.replace(target)
            save_json(previous_path, {"members": membership, "archive_sha256": digest(target)})
        archives[filename] = {"sha256": digest(target), "bytes": target.stat().st_size, "members": len(selected)}
        print("Prepared:", target.name, archives[filename], flush=True)
    transfer_id = "auxiliary_" + archives["project_code.zip"]["sha256"][:12]
    save_json(destination / "transfer_receipt.json", {"created_at": now(), "transfer_id": transfer_id,
        "archives": archives, "scope": "four-class auxiliary only; not seven-class final results",
        "raw_images": 7998, "raw_file_hashing_repeated": False,
        "model_execution_performed": False, "software_tests_performed": False,
        "main_data_protocol_changed": False})
    notebook = _notebook()
    notebook_path = ROOT / "notebooks/COLAB_RUN.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"
    notebook_path.write_text(text, encoding="utf-8")
    (destination / "COLAB_RUN.ipynb").write_text(text, encoding="utf-8")
    instructions = ["Colab upload directory: MyDrive/ml_hw_colab/", "",
        "Upload only these three files to that directory:", "project_code.zip", "auxiliary_images.zip", "transfer_receipt.json", "",
        "Open COLAB_RUN.ipynb in Colab and execute cells 1 through 14 in order.",
        "GPU is auto-detected; current screenshot shows A100 40GB. T4 is also supported.",
        "Code remains modular. The notebook invokes separate scripts; it is not a monolithic training file.",
        "No local model run or software test was performed. These are unexecuted runtime instructions.",
        "This run is auxiliary four-class diagnostics, not the source-blocked seven-class main experiment."]
    (destination / "UPLOAD_README.txt").write_text("\n".join(instructions) + "\n", encoding="utf-8")
    print("Open notebook:", notebook_path, "\nUpload files:", destination, flush=True)
    return archives


