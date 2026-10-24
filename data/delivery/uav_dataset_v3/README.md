# 数据集交付入口

本轮采集、目标裁剪、质量统计、排除记录、像素去重、相似关系和实际文件核对已完成。主任务保留20010张真实目标裁剪、12个作者定义类别，原七类2720张全部保留，新增五类17290张。不是20010张均已专家逐图鉴定。

## 直接读取

清洗数据入口为本目录clean_manifest.csv，图片实际位于data/final/uav_type_v3/images/，路径均相对项目根目录D:/ml_hw。按清单加载，不递归载入图片目录：图目录还保留8张已排除裁剪，来源与历史版本也不能重复当新样本。

```python
from pathlib import Path
import pandas as pd
from PIL import Image

project = Path('D:/ml_hw')
manifest = pd.read_csv(project / 'data/delivery/uav_dataset_v3/clean_manifest.csv')
row = manifest.iloc[0]
image = Image.open(project / row.image_path).convert('RGB')
label = int(row.label)
```

这是读取示例，本轮未运行模型或训练测试。

## 文件对应

| 文件 | 内容 |
|---|---|
| raw_acquisition_inventory.csv | 32077张已取得原图的合并索引，公开真实、自采网页、第三方合成分别标记 |
| clean_manifest.csv | 主任务20010张裁剪，每行有标签、原图/裁剪哈希、目标框、质量信息和保守组 |
| class_counts.csv、class_map.json | 12类数量及标签0到11映射 |
| excluded_expansion.csv | 扩类初筛排除原因，含通用Phantom隔离 |
| excluded_visual_quality.csv | 本轮额外查看后排除的8张无可辨目标裁剪 |
| excluded_pixel_duplicates.csv | 本轮裁剪像素完全重复排除记录，实际为空表 |
| web_external_weak_labels.csv | 自采网页池135张弱标签裁剪，与主池分开 |
| web_supplementary_identity.csv | 4张身份研究补充，没有强行赋予主标签 |
| web_excluded_and_reference.csv | 旧池排除与参考记录，不全是网页样本 |
| source_evidence_index.json | 下载回执、自编爬虫、原网页快照和标注核查的入口 |
| data_integrity_audit.json | 20027项实际文件哈希核对，0问题 |
| files_sha256.csv、receipt.json | 本交付索引的文件身份和数量 |

图像数据本身保持原位，本入口通过CSV引用，避免复制出另一份同图训练数据。原始CSV对应真实取得的图像，不把公开下载冒充自编爬虫。8001张第三方合成数据只进入原始总表，既有四类辅助任务独立保存，不混入十二类主池。

## 当前边界

图像输入是作者目标框裁剪，任务是给定目标区域分类，尚未做端到端检测。新增类别沿用作者命名，Mavic具体商品子型未独立鉴定。保留1262张目标短边小于24像素的困难图，不通过放大假造细节。

清洗表split仍为unassigned；这是数据已整理、实验划分尚未正式启用。Bebop、Phantom 3、Intel Aero、Mambo各只有一个保守场景桶，不能将其随机拆散后声称严格来源隔离。outputs/expansion/processing_v3/split_draft/已有开发14106/留出5904的诊断草案，但留出只覆盖8类，非完整12类最终协议，也不需要为数据交付重跑它。

完整处理报告见项目根reports/expanded_data_report.md，原位报告中的图片链接可正常打开。当前目录内data_processing_report.md是正文快照，其图片以原位报告为入口。没有本地训练、模型推理、软件测试或PDF生成。图片再分发许可与代码开源分别处理。
