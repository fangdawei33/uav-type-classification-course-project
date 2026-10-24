# 当前运行状态与入口

数据采集、清洗和交付已完成；十二类分组实验协议已固定；模型尚未训练，且没有在本地执行任何模型推理或软件测试。后续模型计算全部在 Google Colab T4 进行。

| 项目阶段 | 状态 | 入口 |
|---|---|---|
| 当前冻结数据 | 已完成 | `data/delivery/uav_dataset_v3/clean_manifest.csv` |
| 当前开发/留出协议 | 已生成 | `outputs/experiments/uav_type_v3_v1/protocol.json` |
| 代码和图像传输 | 已打包 | `outputs/colab_uav12/` |
| Colab 分步运行 | 待用户上传并运行 | `notebooks/UAV12_T4.ipynb` |
| 结果解释与报告 | 等待实测结果 ZIP | `uav12_results_for_report.zip` |

## 本地准备命令

下面的协议与传输步骤已经执行完成，一般无需重跑：

```powershell
cd D:\ml_hw
.\.venv\Scripts\python.exe -X utf8 scripts/05d_prepare_uav12_protocol.py
.\.venv\Scripts\python.exe -X utf8 scripts/12d_export_uav12_transfer.py
```

若在任何模型/特征生成前确需重建协议，只有在当前开发清单和约束组没有变化时才运行 `05d_prepare_uav12_protocol.py --refresh-before-training`；该操作会被特征和模型产物保护。当前不要运行旧 `05b`、旧 `06b/07b/09b/10b/11b` 或旧二分类入口。

## Colab 命令对应关系

具体逐格操作见 [NEXT_STEPS.md](NEXT_STEPS.md)。活动入口为：

```text
06d_extract_uav12_features.py
07d_train_uav12_baselines.py
07q_train_qwen3vl.py
09d_freeze_uav12.py
10d_evaluate_uav12.py
11d_export_uav12_report.py
```

四个传统分类器为独立对照；Qwen 为主模型及类别均衡消融。留出集只在所有声明的模型训练完成并通过全局冻结后开放。Colab 实际包版本、T4 设备和完整 pip freeze 会写入实验运行目录。
