# 课程要求、证据和当前完成状态

更新：2026-10-22。七类真实目标区域分类为主任务；已返回的四类合成结果是辅助实验。两者分开记录，不自行猜测分值，不将辅助结果替代主任务。

| 课程环节 | 当前证据 | 完成边界 |
|---|---|---|
| 自编爬虫 | scripts/02_crawl_external.py、02b_crawl_articles.py及实现模块、页面快照、robots/延时、CSV与日志 | 真实候选165张：140暂留/17待定/8排除；另6张参考。标签/框/来源独立性仍需确认，合成下载不计作自采集 |
| 数据预处理 | 2973张逐图视觉台账、错框/叠加/重复记录；辅助全图RGB像素去重 | 主数据2720视觉候选/253排除，尚未正式来源准入；辅助8001原图隔离3张，7998张入独立实验 |
| 特征工程 | 06b、features.py、representation_features.py、返回特征回执 | Colab已实际提取1831维HOG/HSV/LBP等和768维冻结SigLIP2；折内Pipeline拟合填补、方差筛选、缩放 |
| 7:3划分与复现 | 来源约束、种子42、辅助固定三折/manifest与freeze | 辅助按原train/val各3999，约1:1，**未满足主任务7:3**；七类来源条件仍不足，不能伪造隔离 |
| 至少四种不同原理模型 | 07b及返回模型JSON、逐图预测 | 辅助实际比较逻辑回归、KNN、RBF-SVM、随机森林，两种表征共8组基础结果；没有硬融合 |
| 评估与调参 | 原始指标JSON/CSV、grid_search.csv、冻结记录 | 辅助SVM分别搜索6组C/gamma、3折CV，2组调参结果；Accuracy、Balanced Accuracy、Macro/Weighted-F1和逐类指标齐全；七类与真实外部集未评估 |
| 可视化 | 数据筛选图、模型对比图、混淆矩阵、固定规则错误案例、森林重要性 | 辅助图已实际导出查看；分类任务不用机型编号冒充连续回归散点；森林重要性不称为SVM解释 |
| 过程与报告 | data_collection_record.md、data_audit_report.md、auxiliary_experiment_results.md、版本/哈希回执 | 数据阶段与辅助结果章已完成；主任务完整结果论文及PPT未完成，按用户要求不生成PDF |
| 思考题 | thinking_questions.md | 四题已答，可与实测预处理、CV和泛化差距案例合并 |

实测辅助结果：手工＋调参SVM留出Macro-F1为98.925%，冻结SigLIP2＋调参SVM为98.300%。后者是开发CV预选模型，冻结记录保持不变。完整分析见[auxiliary_experiment_results.md](auxiliary_experiment_results.md)。不以大模型名称或高合成分数宣称创新性、真实泛化或发表分区。

本地只执行数据处理、结果导入、返回文件/表格一致性核对与绘图；未运行软件测试、模型推理、训练或评估。辅助模型实验在Google Colab的Tesla T4环境运行，返回包中的98项可核查一致性检查通过；未返回的模型/特征二进制不能宣称已独立认证。
