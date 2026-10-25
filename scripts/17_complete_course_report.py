"""Rebuild the course report from saved results; never load or run a model.

Requires the returned experiment package and local delivery CSVs. Word output
uses python-docx; pass --markdown-only when that optional dependency is absent.
"""
import argparse
import json
from pathlib import Path
import sys
import textwrap

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REPO = "https://github.com/fangdawei33/uav-type-classification-course-project"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def table(headers, rows):
    def cell(x):
        return str(x).replace("|", "/").replace("\n", " ")
    return "\n".join("| " + " | ".join(map(cell, row)) + " |"
                     for row in [headers, ["---"] * len(headers), *rows])


def excerpt(file, first, last):
    """Quote a contiguous, real source excerpt, excluding the end marker."""
    lines = (ROOT / "src/uav_project" / file).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if first in line)
    stop = next(i for i in range(start + 1, len(lines)) if last in lines[i])
    return "```python\n" + textwrap.dedent("\n".join(lines[start:stop])).strip() + "\n```"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=ROOT / "outputs/imported_results/uav12_a3479ced386e1dea/outputs/experiments/uav_type_v3_v1")
    parser.add_argument("--markdown-only", action="store_true")
    parser.add_argument("--refresh-figures", action="store_true", help="Redraw publication figures from saved CSV/JSON only")
    args = parser.parse_args()
    run = args.result_root
    if args.refresh_figures:
        from uav_project.paper_figures import build_figures
        build_figures(ROOT, run)
    if not (ROOT / "reports/paper_figures/figure_receipt.json").is_file():
        raise FileNotFoundError("Run this report command with --refresh-figures once to generate the required figures")
    measured = read(run / "evaluation/measured_results.json")
    protocol = read(run / "protocol.json")
    manifest = pd.read_csv(run / "manifest.csv", low_memory=False)
    clean = pd.read_csv(ROOT / "data/delivery/uav_dataset_v3/clean_manifest.csv", low_memory=False)
    delivery = read(ROOT / "data/delivery/uav_dataset_v3/receipt.json")
    results = {r["model"]: r for r in measured["results"]}
    summaries = {name: read(run / "models" / name / "summary.json") for name in ["handcrafted", "siglip2"]}
    train = manifest[manifest.split.eq("development")]
    test = manifest[manifest.split.eq("holdout")]
    assert len(clean) == 20010 and len(train) == 14106 and len(test) == 5904
    assert set(train.group_id).isdisjoint(set(test.group_id))
    assert set(clean.sample_id) == set(manifest.sample_id)
    assert len(results) == 13
    quality_cols = ["width", "height", "gray_mean", "gray_std", "laplacian_variance_128"]
    missing = {name: int(train[name].isna().sum()) for name in quality_cols}
    short = clean[["width", "height"]].min(axis=1)
    size_counts = {str(n): int((short < n).sum()) for n in [12, 16, 24, 32, 48, 64]}
    class_rows = []
    for label, group in manifest.groupby("label", sort=True):
        class_rows.append([label, group.class_name.iloc[0], len(group),
                           int(group.split.eq("development").sum()),
                           int(group.split.eq("holdout").sum()), group.group_id.nunique()])
    class_rows.append(["合计", "12 个类别", len(manifest), len(train), len(test), manifest.group_id.nunique()])
    names = {"handcrafted": "手工", "siglip2": "SigLIP2"}
    classifier_names = {"logistic": "逻辑回归", "knn": "KNN", "svm": "RBF-SVM", "forest": "随机森林", "tuned": "调优 KNN"}
    labels = {}
    for rep in names:
        for classifier, name in classifier_names.items():
            labels[f"{rep}_{classifier}"] = names[rep] + "＋" + name
    labels.update({"qwen3_vl_zero_shot": "Qwen 零样本", "qwen3_vl_lora_standard": "Qwen 标准 LoRA",
                   "qwen3_vl_lora_sqrt_class_balanced": "Qwen 加权 LoRA"})
    metrics = lambda model: results[model]["metrics"]
    order = sorted(results, key=lambda model: metrics(model)["macro_f1_supported_labels"], reverse=True)
    result_rows = []
    for model in order:
        m = metrics(model)
        cv = results[model].get("cv_macro_f1_supported_labels")
        result_rows.append([labels[model], f"{m['accuracy']:.4f}", f"{m['balanced_accuracy_present_labels']:.4f}",
                            f"{m['macro_f1_supported_labels']:.4f}", f"{m['weighted_f1']:.4f}",
                            "—" if cv is None else f"{cv:.4f}"])
    tuning_rows, tuning_delta = [], []
    for rep in names:
        base = next(x for x in summaries[rep]["baselines"] if x["model"] == "knn")
        tuned = summaries[rep]["tuned"]
        for stage, receipt, model in [("调优前", base, rep + "_knn"), ("调优后", tuned, rep + "_tuned")]:
            m = metrics(model)
            tuning_rows.append([names[rep] + stage,
                                "5／uniform" if stage == "调优前" else "3／distance",
                                f"{receipt['cv_macro_f1_mean']:.4f} ± {receipt['cv_macro_f1_std']:.4f}",
                                f"{m['accuracy']:.4f}", f"{m['macro_f1_supported_labels']:.4f}"])
        old, new = metrics(rep + "_knn"), metrics(rep + "_tuned")
        tuning_delta.append([names[rep], f"{(tuned['cv_macro_f1_mean']-base['cv_macro_f1_mean'])*100:+.3f}",
                             f"{(new['accuracy']-old['accuracy'])*100:+.3f}",
                             f"{(new['macro_f1_supported_labels']-old['macro_f1_supported_labels'])*100:+.3f}"])
    examples = []
    for row in [clean.iloc[0], clean.iloc[1], clean.loc[short.lt(16)].iloc[0]]:
        examples.append([row["class_name"], f"{row.raw_width}×{row.raw_height}", row.crop_xyxy,
                         f"{row.width}×{row.height}", "短边小于 24，保留质量标记" if min(row.width, row.height) < 24 else "RGB 裁剪，像素核验"])
    importance = pd.read_csv(run / "models/handcrafted/forest_importance.csv").head(6)
    importance_rows = [[r.feature, f"{r.impurity_importance:.6f}"] for r in importance.itertuples()]
    truth_pred = pd.read_csv(run / "evaluation/qwen3_vl_lora_standard_predictions.csv")
    prediction_examples = []
    # Fixed display rule: first correct and first incorrect in saved order, per class.
    for label in [2, 7, 8, 9]:
        subset = truth_pred[truth_pred.label.eq(label)]
        for correct in [True, False]:
            selected = subset[subset.predicted_label.eq(subset.label).eq(correct)]
            if not selected.empty:
                r = selected.iloc[0]
                prediction_examples.append([r.sample_id, r.class_name,
                    protocol["classes"][int(r.predicted_label)] if int(r.predicted_label) >= 0 else "无效输出", "正确" if correct else "错误"])
    train_rows = []
    for rep in names:
        for clf in ["logistic", "knn", "svm", "forest", "tuned"]:
            receipt = read(run / "models" / rep / (clf + ".json"))
            train_rows.append([labels[rep + "_" + clf], f"{receipt['training_report']['accuracy']:.4f}",
                               f"{metrics(rep+'_'+clf)['accuracy']:.4f}"])
    folds = [[f["fold"] + 1, f["train_samples"], f["validation_samples"], "6 类计分／12 类训练"] for f in protocol["folds"]]
    corr = train[quality_cols].corr()
    correlation_rows = [["宽度与高度", f"{corr.loc['width','height']:.4f}"],
                        ["宽度与拉普拉斯方差", f"{corr.loc['width','laplacian_variance_128']:.4f}"],
                        ["高度与拉普拉斯方差", f"{corr.loc['height','laplacian_variance_128']:.4f}"]]
    parts = []
    def add(value): parts.append(textwrap.dedent(value).strip())
    add("""
    # 基于真实公开数据的无人机机型图像分类与多模型比较研究

    ## 摘要

    面向无人机细粒度机型识别，本文建立公开数据获取、自编网页爬虫、目标框审核、图像裁剪、特征工程与多模型评估流程。原始采集总表记录 32,077 张原图；主实验形成 20,010 张真实目标裁剪，涵盖 12 个作者定义类别。按保守场景与相似性约束组划分训练集 14,106 张、测试集 5,904 张，比例约为 7∶3。传统模型在训练集内部执行两折分组交叉验证和网格搜索，Qwen3-VL 则按预设参数训练一轮，未另设验证集。实验比较手工视觉特征与冻结 SigLIP2 特征上的逻辑回归、KNN、RBF-SVM、随机森林，以及 Qwen3-VL-2B 的零样本和两种 QLoRA 监督微调，共 13 组结果。手工特征调优 KNN 在测试集取得 0.7307 的准确率和 0.6108 的 Macro-F1；标准 QLoRA 的相应指标为 0.4946 和 0.3668，优于零样本，但未超过传统基线。测试集只覆盖 8 类，交叉验证仅对可分组的 6 类计分；1,262 张裁剪短边小于 24 像素，尚未开展尺度筛选后重训。因此，本研究提供的是当前分组协议下的诊断结果，不能据此宣称完整十二类独立来源泛化能力。报告保留关键代码、数据样例、参数、调优对比、失败记录和可复现性说明。

    **关键词：** 无人机机型分类；网页爬虫；特征工程；分组交叉验证；视觉语言模型；QLoRA
    """)
    add("项目代码仓库：" + REPO)
    add("""
    ## 1 实验概述

    ### 1.1 实验目的与任务

    本实验对应课程任务背景 2 的无人机机型分类分支，目标是完成真实数据采集、预处理、特征工程、数据划分、至少四种不同原理模型、指标比较、网格搜索和结果分析的完整流程。Drone-type-Set 提供按机型研究检测与跟踪的数据基础[1]；本研究输入是公开作者目标框裁剪后的无人机图像，输出为 0 至 11 的机型标签。分类以外的端到端目标检测不在本次实现范围内。

    研究问题 RQ1 关注公开采集样本如何转化为可追溯的监督数据，并判断类别与来源组能支持怎样的评价范围；RQ2 比较手工特征与冻结预训练视觉表示在相同协议下的识别表现；RQ3 检查有限预算的视觉语言模型监督适配能否改善零样本，以及类别加权是否带来进一步收益。第3章回答数据与方法设计，第4章展示对应证据，第5章讨论误差与结论边界。

    本研究的工作包括可复查的数据获取与处理流程、相同测试协议下的多模型比较，以及基于真实失败结果的细粒度识别诊断。HOG、KNN、SigLIP2 和 QLoRA 均采用既有方法，不将流程组合表述为已验证的新算法。网页标题、图注等文本用于采集溯源和人工核对，没有与图像逐条配对的飞行日志或传感器数据，因此本实验属于图像分类与视觉语言模型适配，不声称完成图文传感器多模态融合。

    图1概括三条独立比较路径。传统分支用于建立有解释力的基线，Qwen 分支检验监督适配；各分支共享数据协议而非彼此融合。
    """)
    add("![研究框架](paper_figures/fig01_framework.png)\n\n图1 研究框架与独立比较路径。传统模型在训练内部交叉验证，Qwen 按预设参数训练；统一冻结后评价同一测试集。")
    add("""

    ### 1.2 软硬件环境

    本地 Windows 环境负责数据采集、裁剪、清单整理与报告生成。主模型实验在 Google Colab 的 NVIDIA A100-SXM4-40GB 上完成，随机种子设为 42；早期四类合成辅助实验使用 T4，两套实验单独登记，不混入主结果。主实验返回环境记录为 Python 3.13.15、PyTorch 2.11.0+cu130、CUDA 13.0、Transformers 4.57.1、PEFT 0.17.1、Accelerate 1.15.0、bitsandbytes 0.48.2、NumPy 2.1.3、pandas 2.2.3、scikit-learn 1.6.1、scikit-image 0.25.2、Pillow 11.3.0。采集和可视化代码使用 requests、BeautifulSoup、matplotlib 等，完整依赖由 requirements 文件及运行回执记录。

    ## 2 相关原理说明

    ### 2.1 爬虫与数据预处理

    HTTP 爬虫获取公开页面，用 HTML 解析器按配置选择正文、图片链接与图注，再下载允许采集的媒体。页面结构、网络状态、限流、robots 规则和内容更新都会影响采集结果。URL、时间、HTML 快照和 SHA-256 哈希构成追溯链；网页中的机型词只能提供候选标签，不能代替对目标身份和框位置的核查。

    预处理包含缺失统计、无效样本处理、文本解析、图像解码、颜色空间统一和重复检测。标签或目标框属于监督必需字段，缺失时应阻止准入或待复核，不能用均值补造；数值特征缺失可在训练折内用中位数填补。对异常框执行边界与面积检查，对低清晰度、暗图和小目标保留质量标记，不能把罕见但有效的困难样本直接当作统计异常值删除。

    ### 2.2 特征与四类传统模型

    HOG 统计局部梯度方向，描述机身与旋翼轮廓；HSV 统计量和直方图描述颜色分布；LBP 描述局部纹理。特征标准化采用 z=(x−μ训练)/σ训练，均值和标准差只从当前训练折估计。方差筛选删除当前训练折的常量维度。SigLIP2 提供冻结预训练图像编码器输出的 768 维表示，提取表示后仍需用本任务训练集拟合分类器。

    逻辑回归通过线性得分和概率损失学习类别边界，计算稳定但不能直接表达复杂非线性结构。KNN 根据特征距离选取邻居并投票，几乎无需参数化训练，但高维距离与预测开销可能成为限制。RBF-SVM 使用核函数形成非线性间隔边界，C 控制拟合与正则化的权衡，gamma 控制局部影响范围。随机森林对多个随机化决策树的预测集成，能表达非线性和特征交互，并提供基于不纯度下降的重要性。四类模型分别代表线性、邻域、核方法和树集成原理，不通过任意融合来凑足模型数量。

    ### 2.3 视觉语言模型监督适配

    Qwen3-VL-2B-Instruct[5] 接收目标裁剪与候选标签说明，输出 LABEL_00 至 LABEL_11 中的一个标签字符串。标签字符串可能包含多个 tokenizer token，因此不把其等同于单个词元。QLoRA[4] 以 4-bit NF4 存储基座权重，在选定语言侧线性层加入 LoRA[3] 低秩可训练矩阵；本次冻结视觉主干，训练标准损失和平方根逆类别频率加权损失两个版本。类别权重在全训练集归一化，不用单样本批内的归一化方式抵消加权效果。本实验未实现强化学习。

    ### 2.4 评价指标与交叉验证

    对类别 c，有 Precision_c=TP_c/(TP_c+FP_c)，Recall_c=TP_c/(TP_c+FN_c)，F1_c=2×Precision_c×Recall_c/(Precision_c+Recall_c)。Accuracy 为全部样本预测正确的比例；Balanced Accuracy 为实际出现类别的召回率均值；Macro-F1 为各计分类别 F1 等权均值；Weighted-F1 按真实类别样本数加权。分母为零时按代码设置取 0。

    交叉验证在训练数据内部反复形成训练折和验证折，以验证分数比较候选模型。网格搜索枚举超参数组合，对每组计算同一交叉验证协议下的平均分，再以最佳参数在完整训练集重新拟合。测试集承担方法冻结后的最终评价；用测试分数反复调参会将测试信息反馈给方法设计。相关帧必须按约束组隔离，不能只依靠随机种子保证独立性。

    ## 3 实验设计与实现过程

    ### 3.1 数据获取

    采集分为作者公开数据导入和自编网页爬虫两条路径。Drone-type-Set 公开目录实际取得 2,973 张原图及同名标注；VisioDECT 作者版本编目 20,924 张图像。程序只登记实际可取得的文件，不将论文中更大的名义数量作为已下载数量。第三方四类合成原图 8,001 张作为独立辅助材料。网页爬虫最终取得 179 张原图；其中清洗后的 135 张弱标签目标裁剪与 4 张身份参考单独交付，没有进入本次主监督训练和测试。早期日志中的 171 是采集过程快照，最终数量以交付回执为准。
    """)
    add("表1 原始数据获取与使用角色\n\n" + table(["来源", "原图数量", "本实验角色"],
        [[name, number, role] for name, number, role in [
            ["Drone-type-Set 作者目录", 2973, "审核后七类 2,720 张进入主数据"],
            ["VisioDECT 作者版本", 20924, "审核后五类 17,290 张进入主数据"],
            ["自编公开网页爬虫", 179, "弱标签材料与身份参考，未参与主评估"],
            ["第三方四类合成数据", 8001, "独立辅助实验，未合并主结果"],
            ["原始总表", 32077, "包含不同角色，不能等同训练集规模"]]]))
    add("""
    文章爬虫读取配置中的来源 URL、允许主机、正文选择器、标题关键词和候选类别。先检查 robots，再按请求间隔访问正文；保存 HTML 快照后提取 img 的 src、srcset、alt 和上下文图注，保留页面实际出现的链接。下载阶段只处理明确选中的媒体，记录原始字节哈希、时间和许可状态。Commons 采集器同时登记文件页、作者与许可信息。访问失败、标题不符、无效图片和未知许可分别记录，不依靠破解反爬获取数据。

    原始 CSV 包含 sample_id、source_url、download_url、collected_at、sha256、candidate_label、title、description、caption、width、height、group_id、license_review 等字段。其中 ID 与哈希用于溯源，尺寸用于质量诊断，标签与分组需审核，标题和描述属于混杂文本，不能直接当数值送入模型。

    表2 自采集原始记录示例
    """)
    add(table(["字段", "文章样例", "Commons 样例"], [
        ["sample_id", "dta_4feae19ca5e40e4d044f", "dte_e979f5fddecf1e4d92c4"],
        ["候选类别／原图尺寸", "Mambo／1024×580", "Bebop／816×538"],
        ["网页标题", "Ryze Tello vs Parrot Mambo: Which is the Best Micro Drone?", "File: Bebop Drone authorized flight over Paris Air Show.jpg"],
        ["图像文本线索", "Top angle view of Parrot Mambo", "Parrot drone over Le Bourget Air Show"],
        ["原始审核与许可状态", "候选待审／许可未核验", "候选待审／文件页记载 CC BY-SA 4.0"],
    ]))
    add("来源：文章样例来自 https://www.halfchrome.com/tello-vs-mambo/；原始 CSV 为 data/metadata/drone_type_article_raw.csv 和 drone_type_external_raw.csv。此处展示采集时记录，后续审核状态保存在独立决策表，不伪装为原始字段已自动确认。")
    add("代码1 robots 检查后的请求入口与文章正文解析节选")
    add("```python\n# src/uav_project/http_client.py 的 Web.get\ndef get(self, url, max_bytes=25_000_000):\n    self.allowed(url)\n    return self.raw(url, max_bytes)\n```")
    add(excerpt("article_crawler.py", "        containers = soup.select(source['content_selector'])", "    missing = [term"))
    add("allowed 方法核验 HTTPS、来源白名单与 robots；raw 方法执行延时、重定向复核、流式大小上限和最多三次请求尝试。文章配置请求间隔为 2 秒，若站点给出更长 crawl-delay 则采用更长间隔。403 或无法确认 robots 时停止对应来源；原始下载重试和网页解析代码均保留在仓库。")
    add("""
    ### 3.2 数据预处理

    缺失值按字段职责处理。缺失目标标签、缺失目标框或解码失败不能自动生成监督样本；保留为待审核或排除记录。group_id 不完整时不将未知图片随机分配到不同集合。标题、描述或许可字段为空时保持来源不确定性，禁止把空许可视为允许再分发。主实验清洗表的 width、height、gray_mean、gray_std、laplacian_variance_128 在训练集中的缺失数均为 0。target_short_side_pixels 属于早期来源的可选字段，并非所有来源填写；统一从实际裁剪宽高重新计算短边。
    """)
    add("表3 训练集质量字段缺失统计\n\n" + table(["数值字段", "样本数", "缺失数"], [[name, len(train), missing[name]] for name in quality_cols]))
    add("""
    清洗先核验原图字节哈希，将目标框的逗号分隔文本解析为四个有限浮点数，检查坐标方向、边界和面积，再按左上取 floor、右下取 ceil 生成 RGB 原生尺寸裁剪；保存后检查尺寸与像素一致性。对已存在的早期裁剪同时核对哈希和 RGB 像素，防止读取到不同版本。EXIF 方向不为 1 时停止该记录并要求核对坐标，而非盲目旋转后沿用旧框。

    原七类排除 253 项，记录的原因包括烧录检测框与置信度、标题卡、字幕、画中画、错框、转场叠影及机身出画。扩展候选共 20,018 张，另排除 8 张无法辨认目标的裁剪，形成 20,010 张主清单。两次排除处于不同阶段，不应重复计数。文件夹仍可能保留历史排除文件，训练必须按清单读取。

    文本处理采用 BeautifulSoup 提取正文与去除 HTML 标签，统一空白；类名通过固定映射转换为整数 y，框坐标解析为数值。标题、文件名、来源 ID 与网页描述不进入 X，防止模型从元数据猜标签。没有匹配的飞行日志，不编造飞行速度或传感器数值填充表格。
    """)
    add("表4 原始图像与清洗裁剪对比\n\n" + table(["类别", "原始尺寸", "目标框 xyxy", "清洗尺寸", "处理结果"], examples))
    add("代码2 目标框文本解析与裁剪核心代码")
    add(excerpt("expanded_processing.py", "        values = tuple(float(v)", "    target = destination"))
    add("""
    对短边小于 24 像素的样本，当前代码添加 short_side_below_24 标记，没有自动删除。低对比度和大面积暗区也仅触发质量提示。质量标记不代表已完成专家逐图审核；本实验未执行全量可辨认性复核或按尺度筛选后重新训练。

    ### 3.3 特征工程

    手工路径将裁剪转换到 128×128 计算特征，同时保留原生宽高比。HSV 三通道各含均值、标准差和 16 桶直方图，共 54 维；HOG 采用 9 个方向、16×16 像素单元、2×2 单元块，共 1,764 维；LBP 采用 8 邻点、半径 1、uniform 模式，共 10 维；灰度均值、灰度标准差和宽高比共 3 维。合计 1,831 维。

    SigLIP2[2,6] 使用 google/siglip2-base-patch16-224 的冻结图像编码器，输出 768 维向量并进行 L2 归一化，模型 revision、处理器和推理精度在特征回执中固定。原生低分辨率图像即使被放大，也无法恢复真实机身细节。

    标签采用固定的 0 至 11 整数编码，标签数值不表示大小或距离。输入不包含类别型表格特征，因此没有必要额外对机型标签做独热编码作为 X；监督目标 y 与输入 X 分开保存。文件名、URL、哈希、来源组与人工审核字段仅用于数据管理。训练集全量 X 的形状分别为 14,106×1,831 和 14,106×768，经过折内常量特征筛选后维数以相应模型的支持掩码为准。
    """)
    add("代码3 折内填补 方差筛选与标准化流水线")
    add(excerpt("uav12_models.py", "    def pipe(model, scale=True):", "    return {"))
    add("""
    该 Pipeline 在交叉验证时由每个训练折独立拟合；随机森林设置 scale=False，其他三类模型执行 StandardScaler。VarianceThreshold 使用默认阈值 0，只排除训练折内常量维度，不声称已通过相关性阈值删除全部冗余特征。

    报告补充对已保存训练清单中质量字段的 Pearson 相关性分析：宽度与高度相关较高，宽高也与统一尺度下的拉普拉斯方差相关，提示目标尺度、构图和清晰度统计并非完全独立。此项为结果解释，不参与本次模型选择，也不把质量字段之间的相关性等同于 1,831 维全部特征的相关性。
    """)
    add("表5 训练集部分质量字段相关系数\n\n" + table(["字段组合", "Pearson r"], correlation_rows))
    add("""
    ### 3.4 数据集划分与验证集用途

    常见三类数据角色是：训练集拟合模型，验证集选择参数或检查点，测试集评价最终固定方案。本实验采用训练集内部交叉验证，因此不要求另存一份固定验证集。原代码把可用于训练和调参的 14,106 张称为 development，把保留到最后评价的 5,904 张称为 holdout。本报告统一称为训练集与测试集；为保持历史清单和哈希不变，CSV 与源码中的原英文取值保留，对应关系在附录说明。

    按预先构造的场景与相似性约束组整体划分，以接近 30% 测试量为目标，最终训练／测试占比为 70.495%／29.505%，种子为 42。整组不可拆分，所以不是严格逐样本随机 7∶3。训练与测试的 group_id 集合无交集，但这些约束组尚未认证为真实独立飞行或独立拍摄来源，不能把组不重叠写成已证明外部独立。
    """)
    add("表6 训练集内部的两折分组验证\n\n" + table(["验证轮次", "训练折样本", "验证折样本", "类别覆盖规则"], folds))
    add("""
    验证折只对标签 2、7、8、9、10、11 共 6 类计算 Macro-F1，其余 6 类因训练集内只有一个约束组，始终留在每轮训练折中；各训练折仍包含全部 12 个标签。因此这是带有始终训练样本的两折分组验证，不能说每张样本都轮流做过验证，也不能把验证分数称为完整十二类交叉验证成绩。传统模型选参结束后在全部 14,106 张训练样本上重新拟合。

    Qwen 两个 LoRA 版本都在全部训练集上按预设参数训练一轮，没有另设固定验证集，没有验证损失选检查点或验证早停。测试集在模型、参数、提示词与协议冻结后评价，未用于这次选参。没有独立验证集是本次大模型实验的局限，不能在报告阶段补写成做过三集划分。未来如需早停或比较更多 LoRA 方案，应先在训练组内设定验证组，再用新的独立测试版本评价。
    """)
    add("![数据与评价协议](paper_figures/fig02_data_protocol.png)\n\n图2 数据构建与评价协议。自采弱标签与合成辅助材料独立登记；主清单按约束组划分训练和测试。图中原图与目标裁剪是不同计数单位。")
    add("代码4 按组构造训练折与验证折节选")
    add(excerpt("uav12_models.py", "    for fold in protocol[\"folds\"]:", "        expected = set("))
    add("""
    泄露防护包括：相关样本整体分组，训练折内拟合预处理器，测试特征只在最终评价阶段生成；超参数依据训练内部验证分数；固定数据清单、代码哈希、模型与提示词；续跑时核验同一冻结版本和预测前缀。中断后恢复已冻结测试推理不等于重新调参。测试成绩已经公开后，再根据错误案例优化方法，应视为新实验并使用新独立测试数据。

    ### 3.5 多模型训练与参数

    两种表示各训练四个基础分类器，分别构成八组传统基线；每种表示再对验证最优的模型族执行一次网格搜索，形成两组调优结果。Qwen 零样本、标准 LoRA 与加权 LoRA 共三组，合计十三组。模型使用相同主数据版本和测试样本，不将早期合成集的高准确率放到本表比较。

    表7 传统模型基础参数
    """)
    add(table(["模型", "关键参数", "预处理与用途"], [
        ["逻辑回归", "max_iter=3000；class_weight=balanced；seed=42", "中位数填补＋去常量＋标准化"],
        ["KNN", "n_neighbors=5；weights=uniform；p=2", "中位数填补＋去常量＋标准化"],
        ["RBF-SVM", "C=1；gamma=scale；class_weight=balanced；ovr", "中位数填补＋去常量＋标准化"],
        ["随机森林", "300 棵树；class_weight=balanced；n_jobs=1；seed=42", "中位数填补＋去常量；不标准化"],
    ]))
    add("代码5 基础模型的交叉验证与最终训练节选")
    add(excerpt("uav12_models.py", "            scores = cross_val_score(", "            receipt = {\"model\": name"))
    add("""
    Qwen 采用 4-bit NF4、float16，图像像素范围 50,176 至 150,528，输入最大长度 1,024，最多生成 8 个新 token；LoRA r=8、alpha=16、dropout=0.05，学习率 1×10⁻⁵，weight_decay=0.01，warmup_ratio=0.03。每设备 batch=1，梯度累积 8，启用梯度检查点；每个版本训练 1 epoch，回执记录各 1,764 个优化步。适配目标为语言侧 q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj，视觉模块不训练。两个版本只在预先声明的样本损失权重上不同，并非训练后依测试集选择加权方案。

    ### 3.6 评价实现与网格搜索

    先在相同训练内分组折上按支持六类的平均 Macro-F1 选择模型族。两种表示实际选中 KNN，随后只搜索 n_neighbors∈{3,5,9,15}、weights∈{uniform,distance}，共 8 组参数，每种表示执行 16 次验证拟合，再以最佳参数全训练集 refit。两者均选中 n_neighbors=3、weights=distance。源码为其他分类器保留的参数网格不代表本次也全部执行过。
    """)
    add("代码6 网格搜索实现节选")
    add(excerpt("uav12_models.py", "        grid = GridSearchCV(", "        tuned_model_path ="))
    add("""
    scorer 使用 f1_score 的 labels=[2,7,8,9,10,11]、average='macro' 和 zero_division=0。模型族选择和调参共用同一组验证折，没有嵌套交叉验证，最优验证分数可能偏乐观。预先按交叉验证选中的传统模型是 SigLIP2＋调优 KNN；最终测试中手工特征更好属于观察结果，不能倒过来声称它在测试前已被选为最优。主研究方法 Qwen 标准 LoRA 是预先声明的方法，不因测试低分改名替换。

    最终测试有真实样本的类别为 2、3、6、7、8、9、10、11 共 8 类，正文 Macro-F1 明确对这八类计分，另保存全十二标签口径。预测成测试中没有真实样本的类别仍会造成真类漏判，未从准确率中删除。缺失的四类测试支持数为 0，不能用把它们的 F1 设零来证明这些类别的真实泛化能力。
    """)
    add("代码7 测试指标的实际实现节选")
    add(excerpt("uav12_evaluation.py", "    supported_macro = float(", "def _save_predictions("))
    add("""
    ## 4 实验结果

    ### 4.1 数据集统计

    20,010 张主样本由原七类的 2,720 张和新增五类的 17,290 张构成。训练集覆盖 12 类，测试集覆盖 8 类；类别数量不平衡，最少类只有 67 张。下表同时报告每类的训练、测试与约束组数量，避免只用总样本量掩盖独立组不足。
    """)
    add("表8 类别与训练测试分布\n\n" + table(["标签", "作者类别名", "总数", "训练集", "测试集", "约束组"], class_rows))
    add("![数据分布](paper_figures/fig03_dataset_profile.png)\n\n图3 数据分布。（a）各类别训练与测试样本数，C00至C11对应表8；（b）原生裁剪短边的互斥区间分布，反映进入主清单的尺度差异。")
    add("""
    ### 4.2 多模型测试结果

    各模型均评价同一批 5,904 张冻结测试样本。表中 CV-F1 对训练内部可验证六类计分，Test-F1 对测试出现八类计分，二者口径不同，不能直接将差值解释为模型过拟合程度。结果按测试 Macro-F1 降序展示，排序不改变实验前的方法声明。
    """)
    add("表9 十三组模型评价指标\n\n" + table(["表示与模型", "Accuracy", "平衡准确率", "Test-F1", "加权 F1", "CV-F1"], result_rows))
    add("![模型性能](paper_figures/fig04_model_comparison.png)\n\n图4 总体性能对比。（a）测试准确率；（b）测试八类 Macro-F1。蓝、绿、橙分别表示手工特征、SigLIP2 和 Qwen；两图采用相同排序，排序不参与模型选择。")
    add("""
    ### 4.3 调优前后对比

    同一表示下以基础 KNN 对应调优 KNN，除参数外沿用相同数据和预处理协议。报告均值与两折标准差，另以百分点表示指标增减，防止把小幅变化夸大为稳定显著提升。
    """)
    add("表10 KNN 调优前后指标\n\n" + table(["表示与阶段", "邻居数／权重", "CV-F1 均值±标准差", "测试准确率", "测试 Macro-F1"], tuning_rows))
    add("表11 调优后的指标变化 单位为百分点\n\n" + table(["表示", "CV-F1 变化", "测试准确率变化", "测试 Macro-F1 变化"], tuning_delta))
    add("![参数搜索](paper_figures/fig05_grid_search.png)\n\n图5 KNN 参数搜索。（a）手工表示；（b）SigLIP2 表示。统一色标显示支持六类的两折平均验证 F1，橙框标记最优组合；未使用测试分数选择网格参数。")
    add("手工 KNN 调优后测试准确率与 Macro-F1 小幅提高；SigLIP2 KNN 的验证分数提高而测试分数下降。这说明在有限来源组上的参数最优并不保证测试收益。只有一次固定分组与两折验证，未进行多随机种子重复或显著性检验，不能宣称调优收益具有统计显著性。")
    add("""
    ### 4.4 微调策略对照与训练过程

    为回答 RQ3，固定 Qwen 基座、图像输入和测试协议，对比零样本、标准 LoRA 与平方根逆频率加权 LoRA。标准微调相对零样本提高了准确率与 Macro-F1；类别加权对整体 Macro-F1 的变化很小，同时准确率下降，未取得一致收益。此对照检验的是当前适配设置，不是视觉模块、训练轮数或模型尺寸的全面消融。

    图6右侧来自 training_history.csv 中实际保存的逐步训练损失，未补绘验证曲线，也未将最终汇总损失当作额外训练点。两种损失目标及样本权重不同，曲线绝对大小不宜直接当作优劣排名。训练损失下降说明优化目标被拟合，不能独立证明测试泛化提升。
    """)
    add("![适配策略与训练过程](paper_figures/fig06_adaptation.png)\n\n图6 Qwen 适配对照。（a）相同测试集上的三种设置；（b）标准与加权 LoRA 的原始训练损失日志，各训练一轮，未设固定验证集。")
    add("""
    ### 4.5 类别错误与预测样例

    本任务标签是离散机型，无数值距离含义。以混淆矩阵、逐类 F1 和真实／预测标签样例代替回归的真实值与预测值散点图。下面展示标准 LoRA 的预测，样例按保存顺序、对指定类分别取首个正确与首个错误，避免只展示成功案例；完整 sample_id 由预测 CSV 保留。
    """)
    # Long source IDs are kept in the public evidence CSV; table shows deterministic row IDs.
    display_predictions = [[f"P{i+1:02}", *r[1:]] for i, r in enumerate(prediction_examples)]
    add("![混淆矩阵对比](paper_figures/fig07_confusion_comparison.png)\n\n图7 手工调优 KNN 与标准 LoRA 的混淆矩阵。仅展示有测试支持的真实类别行，保留全部十二个预测类别及无效输出列；按真实类别归一化为百分比。零项不标数字，非零但不足1%的项标为＜1，类别名称见表8。")
    add("表12 标准 LoRA 真实与预测标签样例\n\n" + table(["样例编号", "真实类别", "预测类别", "判定"], display_predictions))
    add("![逐类F1](paper_figures/fig08_class_f1.png)\n\n图8 三种代表模型的逐类 F1，统一色标为0至100%。仅显示测试有支持的八类，不能由该图推断其他四类性能。")
    add("""
    ## 5 实验分析与讨论

    ### 5.1 不同模型的表现差异

    手工调优 KNN 的测试准确率 0.7307、Macro-F1 0.6108 是本次观察的最佳结果。手工 KNN 高于线性模型，表明固定表示中的局部邻域对当前标签有用；RBF-SVM 提供非线性判别，但其基础参数没有获得相同测试优势。随机森林虽有集成机制，却在高维视觉描述子上取得较低测试成绩，不能仅凭“集成”名称保证更好效果。结果受表示、参数和来源分布共同影响，不能推导任一模型普遍优劣。

    SigLIP2 的基础 KNN 训练内部 CV-F1 为 0.8017，而测试八类 F1 为 0.5749。口径统一到测试中的六个共同标签子集后，该 KNN 的 F1 为 0.5880，仍提示训练内部验证无法充分代表当前测试分布，但这是事后诊断且样本分布仍不同，不能精确分解数据域偏移和过拟合各自造成的损失。

    Qwen 标准 LoRA 的准确率为 0.4946，零样本为 0.0159。零样本 5,809／5,904 张预测集中到标签 0，表现出明显输出偏置；这不等同于均匀随机猜测，不能仅由此认定全部数据标签错误。监督适配改善了标签格式与部分决策边界，但标准 LoRA 在 Anafi→DJIFPV、DJIFPV→Mavic_Enterprise 等类别上仍有大量混淆。其 EFT-E410S 召回率为 0.9340、精确率为 0.9522，二者不应混写。加权 LoRA 的 Macro-F1 约 0.3670，准确率低于标准版本，没有形成整体改善。

    可能影响 Qwen 表现的因素包括冻结视觉主干、语言侧低秩适配、单轮预算、闭集标签字符串学习、小目标放大与来源偏差。这些是与配置一致的待验证解释，并非已通过消融证实的因果结论。模型参数多并不保证细粒度、小尺度分类一定占优。

    ### 5.2 特征重要性与误差来源

    随机森林中 lbp_0、lbp_9、saturation_mean、hue_mean 等特征排名靠前，说明该模型利用了纹理、色彩与饱和度统计。这既可能包含机体外观信息，也可能编码背景与成像条件。相关特征会分摊重要性，基于不纯度的重要性不提供因果解释；不能将其写成“色彩决定机型”。应通过独立背景、更严格来源隔离和适当的解释方法继续核对。

    误差还可能来自作者标签的商品子型未独立核实、相近型号形态相似、连续帧相关、部分框的目标比例低、图像压缩或遮挡，以及未认证来源组。爬虫弱标签没有混入主训练，避免引入未经确认的噪声，但也意味着本次没有验证“公开训练、自采外部测试”的研究设想。
    """)
    add("表13 手工特征随机森林的前六项重要性\n\n" + table(["特征名", "不纯度重要性"], importance_rows))
    add("![森林特征重要性](paper_figures/fig09_forest_importance.png)\n\n图9 手工特征随机森林的重要性排序，仅显示前十二项。该解释针对森林模型，不能作为 KNN、SVM 或 Qwen 的重要性，也不代表因果关系。")
    add("""

    ### 5.3 小目标与数据质量局限

    依据清洗清单的原生裁剪宽高计算，短边中位数为 51 像素，最小为 8、最大为 720。以下阈值是累计统计，行之间不可相加。
    """)
    add("表14 原生目标裁剪短边统计\n\n" + table(["短边条件", "样本数", "占 20,010 张比例"],
        [[f"小于 {n} 像素", size_counts[str(n)], f"{size_counts[str(n)]/len(clean):.2%}"] for n in [12,16,24,32,48,64]]))
    add("""
    短边小于 24 的 1,262 张占 6.31%，小于 32 的 3,448 张占 17.23%。这回应了部分目标很小、肉眼难辨的问题，说明“清洗后”不能等同于“每张均适合细粒度机型鉴别”。尺寸只是审核线索，不存在对所有机型通用的自动可辨阈值。当前结果保留这些样本，没有执行去除后重训，也没有通过尺度消融证明低分主要由小图造成。后续需在训练阶段预定可辨认性规则，保留困难测试样本单独报告；不能在看到测试错误后删掉难例，再把新高分与旧协议直接比较。

    进一步将已保存的逐图预测按原生短边分为＜24、24至31、32至47、48至63、64至95、≥96像素的互斥区间。图10显示各区间准确率与支持数，明细保存于 paper_figures/scale_diagnostics.csv。标准 LoRA 在短边＜24像素的332张测试图上准确率为8.43%，在48至63像素的1,385张上为75.31%，但在≥96像素的598张上又降至33.61%，并不呈单调关系。此分析没有重新推理、筛除测试图或拟合新参数。各区间的类别和来源组成不同，曲线差异不能解释为单纯提高分辨率即可获得的性能增益，也不是控制变量的尺度消融。
    """)
    add("![尺度分层诊断](paper_figures/fig10_scale_diagnostics.png)\n\n图10 目标尺度分层诊断。（a）三种代表模型在各原生短边区间的测试准确率；（b）相应测试样本数。该图为固定结果的事后描述，不提供尺度因素的因果结论。")
    add("""

    ### 5.4 过拟合与欠拟合
    """)
    add("表15 传统模型训练与测试准确率\n\n" + table(["表示与模型", "训练准确率", "测试准确率"], train_rows))
    add("""
    多个模型训练准确率接近 1，而测试明显较低，提示记忆训练样本、相关帧、来源分布差异或模型容量等问题需要排查。距离加权 KNN 在本训练点上预测会包含该点自身，训练准确率为 1 并不单独证明数据泄露；应结合组间验证与独立测试判断。Qwen 本次没有逐轮验证曲线，不能仅凭训练损失下降判断泛化改善，也不能确定单轮结果属于纯粹欠拟合。

    可执行的后续方案包括收集每类多个可核实来源、在训练组内固定验证方案、使用正则化和可验证的早停、增加真实而有身份依据的训练样本，及开展目标尺度与视觉适配消融。本报告只提出这些方案，不将它们列为已实施的性能提升。

    ## 6 实验问题与解决方法

    公开目录数量与论文名义数量不一致。七类目录实际取得 2,973 对，而非预设每类 1,000 对；初版程序因此停止。之后将实际清单与数量差异登记，按可访问子集处理，没有虚构缺少图片或把部分目录写成完整论文数据。

    小文件批量下载缓慢，gdown 出现 Cannot retrieve the public link 报错。后续由浏览器下载公开目录压缩包，使用 ZIP 导入脚本核对文件和标注；这解决取得公开原始文件的路径问题，独立网页爬虫仍按来源规则执行，不用手动下载冒充自编爬虫。

    部分网页返回 403 或无法取得 robots。采集器停止对应来源，记录请求、错误与 HTML 快照，再处理允许访问的其他来源；未绕过访问控制。公开网页图片的许可和类别身份并不随下载成功自动确认，相关候选保持弱标签角色。

    结果整理中出现清单字段名 partition 与真实 split 不一致的问题，导致字段访问异常。按真实表头修正读取逻辑，并固定后续数据格式；缺失目标框或候选字段不匹配时回到原记录核对，禁止补造目标框以使代码运行。

    Colab 测试推理续跑时，进度 JSON 声明已完成 3,250 或 4,500 张，但本地 evaluation 目录缺少对应预测前缀 CSV，触发 FileNotFoundError。恢复步骤依据 Drive 中已保存的最新前缀与 SHA-256，将 CSV 放回期望目录后继续同一冻结版本；最终返回三组 Qwen 各 5,904 条预测。恢复进度不应仅信任 completed_samples 数值，必须同时核验文件存在、哈希、样本顺序与冻结版本。

    零样本大量输出同一标签。报告保留该失败结果与微调对比，未用规则改写预测来提高分数。小尺寸目标和缺少四类独立测试样本仍未解决，明确写入限制。报告复核还纠正了基础 KNN 与调优 KNN 的 CV 分数混用，以及将 EFT 类精确率误写成召回率的问题；以原始 JSON 与逐图预测为准。

    ## 7 思考题作答

    ### 7.1 真实爬虫数据有哪些噪声及不预处理的后果

    噪声包括缺失文本、标题与图片不符、重复或相邻帧、损坏图、格式差异、错误框、目标出画、字幕、检测框烧录与来源标签错误。网页还可能含宣传图、渲染图或身份未确认的相似产品。跳过预处理会造成读取异常、错误监督、背景捷径、尺度主导及训练测试重复，既可能降低实际识别能力，也可能虚高离线分数。本实验用原图与裁剪哈希、框校验、审核台账和分组协议缓解这些问题，但仍保留小目标与来源认证不足的限制。

    ### 7.2 为什么 KNN 需要标准化而树通常不需要

    KNN 的距离会被数值范围较大的维度主导。手工特征同时包含概率直方图、梯度统计和宽高比，不统一尺度会使“距离近”偏离预期。标准化参数应仅从当前训练折估计，不能在全数据上预先拟合。树模型逐维按阈值和排序切分，严格单调缩放通常只改变阈值数值而不改变候选分割顺序，所以随机森林通常无需标准化；这并不免除缺失、错误标签和泄露处理。

    ### 7.3 训练好而测试差属于什么现象及缓解方法

    通常首先考虑过拟合，同时排查来源域偏移、标签问题、预处理不一致与评估口径差异。本实验传统模型训练接近满分而测试下降，不能只归因于模型复杂；相关帧与未认证来源同样需要检查。可在训练组内验证、加强正则化、限制容量、对深度模型验证早停、增加经过身份核实的独立数据并预定质量规则。不能用测试结果反复筛图、选参后仍把同一测试集当完全未见数据。

    ### 7.4 为什么网格搜索应配合交叉验证

    单次验证容易受样本和来源分配影响，交叉验证让各参数在相同多轮分组划分上比较，能观察平均表现与波动。本实验每种表示搜索 8 组 KNN 参数，采用两折而非测试集调参。直接利用测试集选参数会把测试信息引入训练决策，导致偏乐观估计；即使未对测试反向传播，也属于方法选择层面的泄露。交叉验证不能消除错误分组或少量来源带来的偏差，使用同一验证折选模型族与网格参数也有选择偏差。

    ## 8 实验总结与心得体会

    本实验建立了公开数据导入与网页采集、质量审核、目标裁剪、特征提取、分组验证、多模型训练、调优和冻结测试的可追溯流程。课程要求的四种传统模型均有实测结果，进一步比较了冻结视觉表示与视觉语言模型适配。主数据共 20,010 张真实目标裁剪，十三组模型结果与原始预测相对应，手工调优 KNN 在当前测试集获得最高表现，Qwen 微调改善零样本但未超越该基线。

    实验所体现的方法认识是：模型规模、样本总量与数据质量不能互相替代；类别覆盖、来源独立性和评价口径应与性能分数同时报告。数据处理和断点恢复属于研究过程的一部分，只有保存清单、参数和失败记录，才能区分真实方法效果与流程问题。

    当前需要加强的知识环节包括来源鉴别与标签核验、小目标可辨认性评估、类不平衡下的分组划分、视觉骨干适配与预算控制，以及选择偏差和不确定性分析。下一阶段应优先补齐每类独立来源与固定验证协议，再设计尺度分层、视觉侧适配和多次重复实验。对已打开的测试集做后续优化时，应明确更换实验版本，并用新独立数据检验改进。

    ## 附录 A 可复现性与开源材料
    """)
    add("项目代码仓库：" + REPO)
    add("""
    scripts 保存分阶段入口，src/uav_project 保存实现，configs 保存来源与参数，data/delivery 保存数据来源索引，reports 保存正文、图表和证据。源码与历史清单仍使用 development／holdout，分别对应本文训练集／测试集；validation 为训练内部临时验证折，不单独增加到 20,010 张之外。修改报告术语不修改冻结清单与模型结果。

    主要运行入口如下，先依 README、docs/UAV12_EXPERIMENT.md 与 docs/UAV12_COLAB.md 准备数据和依赖，再执行所需阶段。已有冻结结果时不要为更新报告重新训练。

    ```text
    01_import_public_zip.py          公开压缩包导入
    02_crawl_external.py             Commons 公开网页采集
    02b_crawl_articles.py            配置文章及媒体采集
    03_review_data.py                审核记录与完整性核对
    04h_process_expanded_images.py   扩展原生目标裁剪
    04i_finalize_expanded_data.py    固定清洗数据版本
    05d_prepare_uav12_protocol.py   生成训练测试与分组验证协议
    06d_extract_uav12_features.py   手工或 SigLIP2 特征
    07d_train_uav12_baselines.py    传统模型与网格搜索
    07q_train_qwen3vl.py            Qwen QLoRA 训练
    09d_freeze_uav12.py             冻结协议与工件
    10d_evaluate_uav12.py           冻结测试评价
    11d_export_uav12_report.py      结果导出
    17_complete_course_report.py   更新报告，--refresh-figures 重画图表
    ```

    代码采用 MIT 许可证；第三方图片、模型与数据许可单独适用。公开仓库提供代码、配置、可发布索引、汇总图表与报告，不直接再分发第三方原图、压缩包、基座权重和大体积特征。仅克隆仓库而没有合法取得的数据与模型缓存，不能直接复现全部数值。报告生成脚本需已经解包的结果 JSON／CSV 和交付清单，使用 pandas、python-docx 与 matplotlib，不调用模型、不启动训练。paper_figures 保存十组320 dpi PNG、SVG矢量源图、尺度诊断表与绘图回执，确保图表和固定结果对应。
    """)
    add("结果包 SHA-256：a3479ced386e1dea40528d7dd7e50a47577ccf58d17b1599272f93185dcf3eea。最终评估时间：" + measured["evaluated_at"] + "。模型与数据冻结哈希：" + measured["freeze_hash"] + "。完整环境、训练配置、日志、模型版本和预测校验记录随 uav12_results_for_report.zip 保存。")
    add("""
    ## 附录 B 课程要求与证据对应

    数据获取对应 3.1 的字段、原始记录和代码1；预处理对应 3.2 的缺失统计、前后对比与代码2；特征工程对应 3.3 的编码、筛选、标准化与相关性；划分对应 3.4 的比例、种子、验证折与防泄露；四类模型对应 3.5；评价和调参对应 3.6 与表9至表11；图表和误差分析对应第4至第5章；真实问题对应第6章；四道思考题对应第7章；总结对应第8章。

    仍需区分报告覆盖和实验证据边界：自编爬虫已执行，但自采弱标签未进入主监督测试；本次没有完成图文传感器的真实配对融合；训练标签为十二类，测试仅覆盖八类，内部验证计分仅六类；Qwen 没有固定验证集；全量专家身份审核与小目标筛选后重训尚未完成。不能仅通过补写报告把这些限制改成已解决。

    ## 参考文献

    [1] AlDosari K, et al. Drone-type-Set: Drone types detection benchmark for drone detection and tracking. arXiv:2405.10398, 2024. https://arxiv.org/abs/2405.10398.

    [2] Tschannen M, et al. SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. arXiv:2502.14786, 2025. https://arxiv.org/abs/2502.14786.

    [3] Hu E J, et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.

    [4] Dettmers T, et al. QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS, 2023.

    [5] Qwen Team. Qwen3-VL-2B-Instruct model card. https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct.

    [6] Google. SigLIP2 base patch16 224 model card. https://huggingface.co/google/siglip2-base-patch16-224.

    [7] 项目数据来源登记与作者版本索引. data/delivery/uav_dataset_v3/README.md 及 raw_acquisition_inventory.csv.
    """)
    report_dir = ROOT / "reports"
    body = "\n\n".join(parts) + "\n"
    (report_dir / "final_report.md").write_text(body, encoding="utf-8")
    evidence = {"report_update_only": True, "models_run": False, "sample_count": len(clean),
                "train_count": len(train), "test_count": len(test), "validation_folds": folds,
                "train_quality_missing": missing, "crop_short_side_below": size_counts,
                "crop_short_side_median": float(short.median()), "tuning_deltas_percentage_points": tuning_delta,
                "evaluation_timestamp": measured["evaluated_at"], "freeze_hash": measured["freeze_hash"]}
    (report_dir / "course_report_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    pd.DataFrame(prediction_examples, columns=["sample_id", "truth", "prediction", "outcome"]).to_csv(report_dir / "prediction_examples.csv", index_label="example_index", encoding="utf-8-sig")
    add_summary = f"""# 最终实验结果摘要

项目代码仓库：{REPO}

完整正文为 [final_report.md](final_report.md)，Word 为 [final_report_paper.docx](final_report_paper.docx)。正文按老师要求设置八章，并按研究问题组织方法和实验证据，附录保存复现材料和要求映射。

真实裁剪共 20,010 张，训练集 14,106 张，测试集 5,904 张，约 7∶3。development 对应训练集，holdout 对应测试集。传统模型在训练内部做两折分组验证；Qwen 未设固定验证集，未实施验证早停。

{table(["方法", "测试准确率", "测试八类 Macro-F1"], [[labels[m], f"{metrics(m)['accuracy']:.4f}", f"{metrics(m)['macro_f1_supported_labels']:.4f}"] for m in ['handcrafted_tuned','siglip2_tuned','qwen3_vl_zero_shot','qwen3_vl_lora_standard','qwen3_vl_lora_sqrt_class_balanced']])}

限制：训练十二类、测试八类、验证计分六类；来源组未认证为独立飞行。自采原图最终 179 张，弱标签材料没有并入主监督评价。1,262 张目标短边小于 24 像素，未开展尺度筛选后重训。不能宣称完整十二类外部泛化或真实图文传感器融合。

本次只补正文、纠正分数引用、整理保存结果与数据质量统计，没有重新训练或修改预测。
"""
    (report_dir / "final_result_summary.md").write_text(add_summary, encoding="utf-8")
    mapping = """# 课程要求与现有证据

更新：2026-10-25。以二万张真实裁剪的主实验为准，四类合成实验单独保存。

""" + table(["要求", "报告位置与证据", "完成边界"], [
        ["公开数据与自编爬虫", "3.1；代码1；来源清单；原始 CSV", "实际采集179张；弱标签未用于主监督测试"],
        ["缺失异常与文本清洗", "3.2；表3、表4；代码2", "数值缺失统计、框解析、审核排除；小目标只标记"],
        ["编码缩放与特征筛选", "3.3；表5；代码3", "1831/768维；折内填补、去常量、缩放；事后质量相关性"],
        ["7:3划分与防泄露", "3.4；表6；冻结清单", "14106/5904；验证六类、测试八类；组身份未认证"],
        ["四类监督模型", "3.5；表7；代码5", "逻辑回归、KNN、SVM、森林，另比较Qwen"],
        ["评价与网格搜索", "3.6、4.2、4.3；表9至表11；代码6、7", "每表示8组KNN参数×2折；无嵌套CV"],
        ["可视化与讨论", "4.4、4.5、第5章；图1至图10", "框架、协议、分布、对比、网格、适配、错误与重要性"],
        ["问题记录", "第6章", "真实下载、字段、断点报错和未解决问题"],
        ["思考题与总结", "第7、8章", "四题完整作答；后续改进与已完成实验分开"],
        ["复现与开源", "附录A、B；GitHub", "不再分发第三方原图与大权重；用户自行制作PPT"]]) + "\n"
    (report_dir / "scoring_map_v2.md").write_text(mapping, encoding="utf-8")
    if not args.markdown_only:
        from uav_project.paper_document import render_paper
        render_paper(report_dir / "final_report.md", report_dir / "final_report_paper.docx")
    print(f"Report updated from saved results: {len(body):,} characters, {len(results)} result rows; no models run.")


if __name__ == "__main__":
    main()
