# 无人机机型图像分类与多模型比较

本项目是机器学习课程综合设计实验，研究任务为：给定公开数据中的无人机目标框裁剪，预测作者定义的无人机机型类别。

## 最终实验结果

主实验使用 20,010 张真实目标裁剪和 12 个作者定义类别。训练集 14,106 张，用于拟合、内部两折分组交叉验证和网格搜索；测试集 5,904 张，用于冻结后的最终评估。测试集实际覆盖 8 个类别，内部验证计分仅覆盖 6 类，因此不是完整 12 类独立来源测试。历史代码和清单中的 development／holdout 分别对应本文训练集／测试集；Qwen 按预设一轮训练，没有单独验证集或验证早停。

| 模型 | Accuracy | Macro-F1 |
|---|---:|---:|
| 手工特征调优 KNN | 0.7307 | 0.6108 |
| 手工特征 KNN | 0.7276 | 0.6029 |
| Qwen3-VL 标准 LoRA | 0.4946 | 0.3668 |
| Qwen3-VL 类别加权 LoRA | 0.4804 | 0.3670 |
| Qwen3-VL 零样本 | 0.0159 | 0.0794 |

Qwen 微调相对于零样本有明显改善，但在本数据、目标裁剪输入、单轮 QLoRA 和冻结视觉主干设置下没有超过传统基线。项目如实保留这一结果，不把大模型名称当作性能保证。

完整实验报告：[`reports/final_report.md`](reports/final_report.md)；按课程八章要求补充后的 Word 版本：[`reports/final_report_revised.docx`](reports/final_report_revised.docx)。原 `final_report.docx` 为历史版本，提交时使用补充版。课程要求与证据对应见 [`reports/scoring_map_v2.md`](reports/scoring_map_v2.md)。

## 目录

- `scripts/`：按阶段执行的数据导入、网页采集、审核、清洗、特征提取、训练、评估和结果导出入口。
- `src/uav_project/`：模块化实现。
- `configs/`：数据源、处理和实验配置。
- `docs/`：实验协议、数据交付说明、Colab运行顺序和复现限制。
- `reports/`：论文正文、数据处理记录、结果摘要和思考题。
- `data/delivery/`：可公开的数据索引和来源说明，不包含第三方原始图片。

## 复现边界

代码仓库不再分发第三方原始图片、作者数据压缩包、Qwen/SigLIP2 权重、特征矩阵和 Colab 检查点。使用者应按来源登记和许可条件自行获取数据，再按脚本生成本地数据目录。主实验输入是作者目标框裁剪分类，不是端到端目标检测，也没有将网页文本或飞行日志随机拼接成伪多模态样本。

数据中含公开作者数据和 CC BY-NC-SA 来源，代码采用 MIT License，但代码许可证不会改变原始数据的许可条件。发布前请分别阅读数据集作者的许可和使用要求。

## 运行顺序

```text
01_import_public_zip.py
02_crawl_external.py / 02b_crawl_articles.py
03_review_data.py
04h_process_expanded_images.py
04i_finalize_expanded_data.py
05d_prepare_uav12_protocol.py
06d_extract_uav12_features.py
07d_train_uav12_baselines.py
07q_train_qwen3vl.py
09d_freeze_uav12.py
10d_evaluate_uav12.py
11d_export_uav12_report.py
```

长时间 GPU 训练和留出推理建议在 Google Colab 执行。Notebook 会把 Qwen 检查点和每 250 张的预测进度保存到 Drive。运行前请先阅读 [`docs/UAV12_COLAB.md`](docs/UAV12_COLAB.md) 和 [`docs/UAV12_EXPERIMENT.md`](docs/UAV12_EXPERIMENT.md)。

## 可复现性

随机种子为 42。最终实验运行环境、依赖版本、模型 revision、训练配置、代码哈希、数据清单哈希和错误日志都在用户侧结果包中保存。公开仓库中的脚本和协议可复查数据处理和实验设计；没有原始图片和权重时，不能声称仅克隆仓库即可复现全部数值。

## 引用

如果使用本项目的代码或实验协议，请引用 `reports/final_report.md`，并同时引用实际使用的数据集及其原始论文。


公开仓库地址：<https://github.com/fangdawei33/uav-type-classification-course-project>
