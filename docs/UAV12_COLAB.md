# Colab T4 上传与运行清单

本文件是 `notebooks/UAV12_T4.ipynb` 的文字版。上传位置固定为：

```text
MyDrive/ml_hw_uav12/
```

上传四个文件（Notebook 也已复制到这个交付目录）：

```text
uav12_code.zip
uav12_images.zip
transfer_receipt.json
UAV12_T4.ipynb
```

上传后从 Google Drive 打开 `UAV12_T4.ipynb`，连接 GPU 运行时并确认环境记录显示 Tesla T4，再按顺序运行每一个单元格。Notebook 会自动：

- 校验两个压缩包的 SHA-256；
- 拒绝目录穿越和符号链接；
- 将图片展开到 Colab 本地 SSD；
- 将分阶段结果同步到 Drive；
- 记录实际 GPU、CUDA、Python 与主要依赖版本；
- 将 Qwen 训练检查点和最终评测的逐250张预测进度写入 Drive，以便断线恢复。

T4 先按 SigLIP2 batch size 16运行。显存不足时，只把该单元中的 `--batch-size 16` 改为 `8`，重跑同一单元，已完成的 128 张特征分片会复用。Qwen 训练使用单图 batch、梯度累积 8 步、FP16、4-bit 基座和一轮固定训练；不要在两组 LoRA 之间升级依赖。

传统模型训练主要使用 CPU，这是正常现象。它们在开发集上完成分组交叉验证和网格搜索，不读取留出图。只有 `09d_freeze_uav12.py` 成功后，才允许提取留出特征和运行 `10d_evaluate_uav12.py`。

下载给我的文件是 Notebook 最后生成的 `uav12_results_for_report.zip`。它只包含结果表、JSON、日志和图，不包含大体积特征矩阵、模型适配器或检查点；后者继续保留在 Drive。

## 2026-10-23：已冻结评估的预测续跑路径修复

确认故障：`predict_qwen_holdout` 把预测 CSV 写在 `UAV12_PROGRESS_ROOT`，
续跑时却从实验目录 `evaluation/` 读取同名文件，导致 `FileNotFoundError`。
这不是 LoRA 训练失败；3250 张预测及 SHA-256 已保存到 Drive。

对于已经冻结且正在运行的实验，不修改被哈希记录的源码或 freeze.json。
新增的 `scripts/10r_restore_uav12_predictions.py` 只使用标准库，将经过
进度记录、冻结摘要、三个 CSV 摘要校验的预测文件复制到旧评估器读取的位置。
已有本地同名文件内容不同时停止，保留原件。它不启动模型，也不重新训练。

在 Colab 中可将此脚本上传到项目 scripts 目录后运行；已初始化 session 时：

```python
session.run('10r_restore_uav12_predictions.py')
```

然后只重新运行第14格。原评估器会再次核对 sample_id 前缀，并跳过已经保存的
Qwen 样本；传统分类器的预测和指标可能重新计算，但不重新拟合模型。
本修复单元需在每次断线恢复、重新运行第14格之前执行，以恢复最新 CSV。
第14格成功后依次执行第15、16格。不要重新运行训练或全局冻结。

本地只编辑和检查修复源码，不运行模型或本地测试。此脚本尚未包含于旧上传包；
当前会话也可直接使用聊天中等价的文件校验/复制单元，无需替换旧代码包。

后续续跑优先使用合并入口：`scripts/10e_resume_uav12_evaluation.py`（需一并上传10r脚本）。
对于当前旧上传包，直接将 `outputs/colab_uav12/resume_evaluation_cell.py` 的全部内容
粘贴到 Colab 替换第14格。该单元每次运行都会先读取并校验最新进度、恢复三个CSV，
再启动原冻结评估入口；异常时同时显示最新内部日志。运行时重置后先恢复第1～4格。
不需要修改旧代码包、transfer_id、freeze.json 或模型源码。
