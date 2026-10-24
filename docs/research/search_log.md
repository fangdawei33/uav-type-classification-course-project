# 检索与证据记录

日期：2026-10-22。仅文献检索、网页阅读和研究文档编写；未运行数据/模型测试或训练，未下载整套数据或保存论文PDF。

## 问题与检索式

1. 数据：`Drone-type-Set publication DOI`；`MMAUD multi modal anti UAV dataset official ground truth drone types`；`AU-AIR dataset flight metadata object detection`。
2. 视觉RL：`Visual-RFT`；`fine-grained visual reinforcement learning classification`；`Specificity-aware reinforcement learning`。
3. Agent：`DeepEyes Incentivizing`；`AgentIAD`；`ViperGPT`；`V* Guided Visual Search`。
4. 正反对照：`SFT Memorizes RL Generalizes`；`Does Reinforcement Learning Really Incentivize Reasoning`；`RL Fine-Tuning Heals OOD Forgetting`。
5. 模块分工：`Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions`；`Eyes Wide Shut`。

## 纳入规则

- 用论文原文、作者仓库、正式出版页面支持论点；不以博客或二次AI摘要作为结论依据。
- 纳入能回答数据对应、主模型职责、奖励设计、公平对照的问题的论文。
- 区分预印本提交年、正式出版年和访问日期；只使用截至调研日期可核查的内容。
- 没有查全引用网络，不声称“首个”“无人做过”或保证某期刊分区。

## 已知访问与版本限制

- CVF若干页面/全文直开返回403，但搜索返回的官方出版摘要可读。部分方法改读作者arXiv公开稿；不是绕过付费墙或下载受限材料。
- Visual-RFT会议出版摘要与arXiv v1某些检测设置/数值有变化，报告不混用数字。
- SpeciaRL检索到CVPR2026补充材料，但完整正式书目信息未完成核对，BibTeX先引用arXiv。
- DeepEyes检索到标示ICLR2026的PDF记录，OpenReview会话页面要求浏览器验证；本报告按已打开的arXiv v1说明方法，不凭搜索片段填写会议书目。
- AgentIAD抓取的摘要标题与正文标题不一致；矩阵标记“只使用摘要、正式引用前锁版本”，不把变化后的方法细节混写。
- TaxonRL检索到WACV2026条目，但本轮全文打开失败；仅列为后续阅读候选，不计入19篇主文献。
- 曾拼写错误的V*页面地址未作为证据；正确出版记录由官方搜索结果与arXiv 2312.14135交叉核对。
- 旧repo脚本、旧README还未全面同步本次研究方案。当前调研不代表这些旧代码已实现多类大模型或RL。

## 核心事实与证据定位

| 事实 | 原始证据定位 | 推论限制 |
|---|---|---|
| Visual-RFT研究视觉分类RL奖励 | arXiv 2503.01785v1 §3.2.1 CLS Reward、官方ICCV2025条目 | 不外推无人机性能 |
| GRPO出处 | arXiv 2402.03300 摘要及算法名称 | 非视觉效果证明 |
| DeepEyes存在真实工具反馈循环 | arXiv 2505.14362v1 §3.1–3.2 | 固定程序不是学习策略 |
| SpeciaRL是开放世界分类 | arXiv 2603.03197v1 §3.1、§3.4 | 与固定七类标签不同 |
| MMAUD有同步数据与机型 | arXiv 2402.03706v1 §III–V，官方仓库下载表 | 没核实成套机载日志/工况文本 |
| AU-AIR遥测来自拍摄平台 | arXiv 2001.11737 数据定义 | 不是目标无人机的日志 |
| LLM可不承担最终视觉分类 | arXiv 2401.02460v2、CVPR2024对应条目 | 不能假定生成语义必有益 |
| 模块化视觉推理有正式研究 | arXiv 2303.08128、ICCV2023条目 | 其系统不等于经过RL |

## 未完成事项

- 无人机七机型图像、真实工况文本、飞行日志的一致对应关系。
- 候选数据集实际下载、许可细节、标注格式与来源分组核查。
- 实际硬件资源及RL训练配置是否可行。
- 架构A与B1的实证优劣；本轮只有文献与研究设计，没有模型结果。
- 投稿前更新所有书目最终版本及完整作者，不把首作者+others的初稿当最终格式。

## 数据来源后续核查（2026-10-22）

已通过交互浏览器实际打开Drone-type-Set作者目录，核实七类文件夹、Annotations.txt和Bebop/Emax两组jpg/txt样例。更正此前“实际目录尚未读取”的状态；完整下载和逐文件标签质量仍未验证。
详见 `data_feasibility_2026-10-22.md` 与 `evidence/drone_type_set_browser_observations.md`。

补充核查到原厂技术文本入口和PX4官方公开日志来源，但没有证明它们与Drone-type-Set图片逐样本对应。数据/模型训练未运行。
