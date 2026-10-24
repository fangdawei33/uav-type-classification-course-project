# 当前数据状态

截至 2026-10-23，主数据版本为 [uav_dataset_v3](../data/delivery/uav_dataset_v3/README.md)：20,010 张目标裁剪、12 个作者定义类别。图片与 `clean_manifest.csv` 中的路径相对项目根目录；按清单读取，不递归加载历史图片目录。

## 数据与实验状态

| 阶段 | 当前状态 |
|---|---|
| 原始数据获取、网页采集、整理 | 已完成；原始文件、采集记录和证据保留 |
| 12 类主清单 | 20,010 张真实目标裁剪，哈希绑定到 v3 冻结版本 |
| 自采网页图像 | 179 张原始图；135 张弱标签裁剪另列，4 张身份补充另列；不进入本次监督训练/测试 |
| 第三方合成数据 | 8,001 张原图及已完成的四类辅助实验独立保存；不混入十二类主池 |
| 分组实验协议 | 已生成；开发 14,106 张、留出 5,904 张，比例约 70.5:29.5 |
| 开发分组交叉验证 | 2 折，实际支持 6 类评分；其他类别仍进入各折训练 |
| 留出评估覆盖 | 8/12 类；不是完整十二类测试，也未认证为独立视频/飞行来源 |
| 模型训练与主任务测试 | 尚未运行；Colab T4 分步骤代码与上传包已准备 |

## 固定类别数

主集包括 Bebop 447、DJI phantom 3 67、DJI phantom 4 pro 572、Emax 329、Intel Aero RTF 256、Mambo 578、YH 19Hw 471、Anafi-Extended 3371、DJIFPV 3439、EFT-E410S 3552、Mavic_Air 3519、Mavic_Enterprise 3409，共 20,010 张。

## 数据边界

四个单场景类没有可隔离的留出组，Emax 在开发侧也只有一个约束组；因此开发 CV 只给六类打分。保守约束组表达不能跨集合的相似场景，不等于经验证的独立飞行/视频 ID。新增五类沿用作者文件夹标签，具体产品身份没有全部独立核实。

任务输入是作者目标框裁剪，报告应称为给定目标区域分类。当前没有配对的每次飞行文本和遥测记录，不声称三模态融合。完整限制见[十二类实验方案](UAV12_EXPERIMENT.md)。

## 关键文件

- 数据交付：[data/delivery/uav_dataset_v3](../data/delivery/uav_dataset_v3/README.md)
- 固定协议：`outputs/experiments/uav_type_v3_v1/protocol.json`
- 分组折清单：`outputs/experiments/uav_type_v3_v1/cv_folds.json`
- 当前 T4 上传包：`outputs/colab_uav12/`
- 分步运行清单：[NEXT_STEPS.md](NEXT_STEPS.md)
