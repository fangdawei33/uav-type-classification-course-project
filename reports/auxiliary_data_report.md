# Cranfield合成辅助数据：获取、核查与准备记录

数据阶段生成时间：2026-10-22T09:43:07.042223+00:00。本报告保留数据获取/准备记录，不包含训练指标。后续Colab四类辅助实验已完成并回传，实际Tesla T4；实测指标另见[辅助实验结果与分析](auxiliary_experiment_results.md)，无需重复运行。

## 数据来源和用途

来源为Hugging Face `mazqtpopx/cranfield-synthetic-drone-classification`，固定revision `61f3f1858aa16eb6fbb763b0814d96dd58c0716a`。ZIP 855,922,191字节，SHA256为`bb3a3b1a19ef7d79da1c1aa8ca3a35e2467d23095a6585f7aee6c17425adb322`，与公开LFS元数据一致。对应论文DOI为10.3390/jimaging8080218；原始数据DOI为10.17862/cranfield.rd.19423925。当前取得的是镜像包，没有把它当作第二个独立采集来源。

来源数据卡声明CC BY 4.0，已保存同版本README。这是第三方三维渲染合成域数据，不是本项目用大模型生成的图片，也不是老师要求的真实网页爬虫样本。

标签为三个DJI机型系列和No Drone。Phantom系列不能无证据映射为Phantom 3或Phantom 4 Pro，No Drone不属于既有给定区域七分类。全部辅助清单的seven_class_label保持空值，单独保留四类预训练标签。后续已用于独立的辅助四分类实验，未混入主七类训练或改写主任务类别。

## 实际规模与隔离处理

已读取ZIP并完成安全路径检查、CRC读取、全量图像解码、文件及RGB像素哈希。8001张全部解码成功，解码/提取失败0项；全部原图保留。

| 来源类别 | 原始train | 原始val | 隔离后辅助train候选 | 隔离后辅助val候选 |
|---|---:|---:|---:|---:|
| dji_inspire | 1000 | 1000 | 1000 | 1000 |
| dji_mavic | 1000 | 1000 | 1000 | 1000 |
| dji_phantom | 1000 | 1000 | 1000 | 1000 |
| no_drone | 1001 | 1000 | 999 | 999 |
| 合计 | 4001 | 4000 | 3999 | 3999 |

发现一组3张No Drone图片RGB像素完全相同，分布为train 2张、val 1张。文件编码不同，单查文件SHA256会漏掉。三张全部从辅助候选清单隔离，记录在`data/metadata/cranfield_synthetic_classification_excluded.csv`，原图和原始split不改。共保留7998张辅助候选，不把8001张都写成训练图，也不把原val称为真实外部测试。

该包有8001个文件哈希、7999种不同RGB像素内容。与2973张主公开图像的文件哈希精确匹配为0项；这不等于跨数据集近重复或来源独立核实。尚未完成裁剪变体、相近渲染父样本和背景复用检查。

## 尺寸与实际抽查

保留候选中最短边不足64像素的有884张。本轮不因分辨率低就按模型效果删图，后续需要报告尺度分布与低清条件。放大不能恢复不存在的机体细节。

![图1 16张确定性抽查预览](../outputs/auxiliary/cranfield_synthetic_classification/sample_preview.jpg)

图1：每类各选train两张、val两张，共16张。预览及清单哈希与实际查看记录匹配：True。可见不同背景、视角和模糊程度；这是接触表抽查，不是8001张的逐图视觉审核，也不证明具体机型代际或原始渲染父样本独立。

## 可复现入口

1. `scripts/01d_download_auxiliary.py`：仅下载固定版本ZIP及数据卡、核对大小与SHA256。下载客户端版本固定在`requirements-auxiliary.txt`。
2. `scripts/03e_catalog_auxiliary.py`：解码、保留原图、生成原始清单和重复关系，不训练模型。
3. `scripts/04b_prepare_auxiliary.py`：按当前精确重复规则隔离3张，输出辅助train/val清单及本报告；不触碰主七类正式清洗准入。

本轮已执行上述数据步骤，不需要用户重复运行。关键输出：`data/metadata/cranfield_synthetic_classification_images.csv`；`data/metadata/cranfield_synthetic_classification_auxiliary_train.csv`；`data/metadata/cranfield_synthetic_classification_auxiliary_val.csv`；`outputs/auxiliary/cranfield_synthetic_classification/preparation_current.json`。

原始生成参数、父渲染ID及背景ID未知，记录unknown。辅助train/val保留来源划分，只能作为辅助开发划分，不能声称来源完全独立。已有数据的合成预训练与未来结构保持背景生成增强是两种不同实验，应分开比较和报告计算量。

上述数据获取/准备步骤没有运行软件测试、模型训练、模型评估或图片生成。后续辅助表征提取与模型实验在Colab运行，回传指标另见结果报告；本地未运行模型或软件测试，未生成PDF。
