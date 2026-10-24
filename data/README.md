# 当前七类任务的数据目录

原始数据和审核决定分开保存；以下是当前活动数据，不沿用旧鸟类/无人机二分类路径。

| 路径 | 用途 |
|---|---|
| downloads/ | 作者网页下载ZIP，保留 |
| raw/drone_type_set/ | 2973张作者图片与对应标注，保留不覆盖 |
| raw/drone_type_external/ | 自编爬虫保存的外部候选原图 |
| metadata/drone_type_set_images.csv | 作者图像原始清单与URL |
| metadata/drone_type_set_review.csv | 当前视觉决定：2720保留、253排除、0待审 |
| metadata/drone_type_set_visual_cohort.csv | 全量筛选台账，包含保留和排除；非正式清洗清单 |
| metadata/drone_type_set_quality.csv | 质量准备时快照；review_status不是当前决定 |
| metadata/drone_type_set_box_selections.csv | 原标注行选择与证据，不默认最大框 |
| metadata/drone_type_set_reviewed_correlation_constraints.csv | 2807张/12组保守相关性约束，非已核实飞行 |
| metadata/drone_type_set_source_groups.csv | 正式来源待核实，confirmed仍为0组 |
| metadata/drone_type_external_review.csv | 外部候选当前决定：66暂留、20待定、7排除 |

正式processed/drone_type_set/target_crop清洗集及训练/测试划分尚未生成。旧public/、external/和通用二分类metadata如仍存在，不能当作当前七类实验输入。原图不默认纳入开源仓库；出处和许可分别核实。

完整说明见../docs/DATA_STATUS.md与../docs/RUNNING.md。当前不需要重新下载，全部公开图片已经本地导入。
