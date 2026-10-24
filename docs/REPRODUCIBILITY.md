# 复现与公开材料说明

## 1 数据获取

主数据来源、版本、文件数量、标签限制和许可状态见 `configs/expansion_sources.json`、`docs/research/data_source_registry.json`、`reports/data_collection_record.md` 和 `data/delivery/uav_dataset_v3/README.md`。公开网页候选由 `scripts/02_crawl_external.py` 和 `scripts/02b_crawl_articles.py` 采集。爬虫执行 robots 检查、来源白名单、访问延时、有限重试和请求日志，不绕过反爬限制。

## 2 数据处理

主数据形成 20,010 张目标裁剪。处理过程保留原始下载索引、目标框、SHA-256、像素哈希、排除原因和相似性约束。使用者应按清单加载，不能递归加载历史目录，以免把排除图、重复版本或弱标签网页池混入主任务。

## 3 实验

`configs/uav12_experiment.json` 固定随机种子、表示、分类器、Qwen训练参数和留出规则。传统模型按 `05d` 至 `07d` 脚本执行；Qwen 使用 `07q_train_qwen3vl.py`，需要支持 CUDA 的 Colab GPU。`09d_freeze_uav12.py` 成功后才允许读取留出集，随后执行 `10d_evaluate_uav12.py` 和 `11d_export_uav12_report.py`。

开发集 14,106 张，留出集 5,904 张。开发集用于训练和分组交叉验证，留出集只用于最终评价。留出集覆盖 8/12 类，且约束组尚未被认证为独立飞行或视频来源；报告中的指标必须带这一限定。

## 4 不随仓库分发的材料

以下材料不直接提交到公开仓库：第三方原始图片、作者压缩包、模型权重、LoRA适配器、SigLIP2特征矩阵、Qwen检查点和含大体积图片的传输包。原因是原始数据许可与代码许可不同，且模型权重有各自的上游条件。

## 5 结果核对

`reports/final_result_summary.json` 和 `reports/per_class_metrics.csv` 是从最终结果包重新计算并核对的摘要。最终结果包 SHA-256 记录在论文附录中。公开仓库中的报告数字不由手工填写，而是依据保存的预测文件和评估 JSON 生成。


公开仓库地址：<https://github.com/fangdawei33/uav-type-classification-course-project>
