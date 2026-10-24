# 下一步：在 Colab T4 运行十二类实验

数据集、固定分组协议和分步骤运行代码均已准备好。现在只需把交付文件上传并运行 Notebook；无需重新下载、清洗或划分数据。

## 现在操作

1. 打开 `D:\ml_hw\outputs\colab_uav12\`，将 `uav12_code.zip`、`uav12_images.zip`、`transfer_receipt.json` 和 `UAV12_T4.ipynb` 四个文件上传到 Google Drive 的 `MyDrive/ml_hw_uav12/`。
2. 在 Colab 通过“文件 → 在云端硬盘中打开笔记本”打开 `MyDrive/ml_hw_uav12/UAV12_T4.ipynb`，连接 GPU 运行时，按顺序运行所有单元格。环境记录单元会显示实际 GPU；继续前确认是 Tesla T4。
3. 全部单元格成功后，下载 `uav12_results_for_report.zip` 并发回来。我会依据实测结果完成结果分析、图表、论文正文和课程报告。

交付包约 15 MB 代码、506 MB 图片，另含很小的回执和 Notebook。图片包只用于上传传输，Notebook 会验证 SHA-256 后展开。

## Notebook 的运行阶段

| 顺序 | 运行脚本 | 内容 |
|---:|---|---|
| 1 | `06d_extract_uav12_features.py --representation handcrafted` | 开发集 HOG、HSV、LBP 手工特征 |
| 2 | `06d_extract_uav12_features.py --representation siglip2 --batch-size 16` | T4 冻结 SigLIP2 视觉表征；显存不足时改为 8 并重跑本格 |
| 3 | `07d_train_uav12_baselines.py --representation handcrafted --jobs 2` | 四个传统分类器的分组交叉验证和调参 |
| 4 | `07d_train_uav12_baselines.py --representation siglip2 --jobs 2` | SigLIP2 特征上的相同四模型对照 |
| 5 | `07q_train_qwen3vl.py --variant standard` | Qwen3-VL 普通 LoRA 监督微调 |
| 6 | `07q_train_qwen3vl.py --variant sqrt_class_balanced` | 预先声明的类别加权 LoRA 对照 |
| 7 | `09d_freeze_uav12.py` | 冻结协议、代码和已训练模型 |
| 8 | `06d_extract_uav12_features.py` 两种表征均加 `--partition holdout` | 冻结后才处理留出集 |
| 9 | `10d_evaluate_uav12.py` | 一次性留出评估 |
| 10 | `11d_export_uav12_report.py` | 导出实测表格、图和 Markdown 结果 |

## 运行注意

Colab 先安装一次 Notebook 中锁定的依赖；不要在实验中途升级库。传统模型主要使用 CPU。Qwen 训练检查点和阶段产物会保存到 Drive，断线后按 Notebook 的恢复单元恢复，再继续未完成阶段。某单元报错时停在该格，把完整错误信息发回来，不要清空检查点或自行改训练参数。

留出仅覆盖 8 类，开发交叉验证评分覆盖 6 类。报告会明确写出覆盖范围，不将结果表述为完整 12 类独立来源泛化。
