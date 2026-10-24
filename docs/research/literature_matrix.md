# 文献矩阵（截至2026-10-22）

共19篇论文：15篇原始调研条目 + 4篇针对“大模型与专用模块分工”的补充条目。部分只核对摘要和出版信息，不能称为全部精读。会议/期刊信息仅在本轮原始来源支持时填写；否则按arXiv版本引用，不猜分区。

## 阅读与引用索引

| ID | 论文 | 年份/本轮引用版本 | 阅读深度 | 在本项目中的用途与限制 |
|---|---|---|---|---|
| D01 | Drone-type-Set: Drone types detection benchmark for drone detection and tracking | arXiv 2024 | 全文数据与配置部分 | 机型图像候选；类别和框不代表配套真实文本/机载日志；来源含视频，须分组 |
| D02 | MMAUD: A Comprehensive Multi-Modal Anti-UAV Dataset for Modern Miniature Drone Threats | ICRA 2024，官方仓库提供DOI | 全文传感器、格式、局限及官方仓库 | 同步多传感与机型参考；Leica轨迹是外部测量，不等于机载日志；不擅自替换用户选题数据 |
| D03 | AU-AIR: A Multi-modal Unmanned Aerial Vehicle Dataset for Low Altitude Traffic Surveillance | 引用arXiv 2020 | 全文数据定义部分 | 图像+平台遥测关联案例；识别的是交通目标，不是被拍无人机机型 |
| M01 | Qwen3-VL Technical Report | arXiv技术报告 2025 | 摘要、模型族说明 | 图文基座候选；不证明无人机领域效果和单卡RL预算 |
| M02 | Qwen2.5-VL Technical Report | arXiv技术报告 2025 | 摘要与功能说明 | 多篇相关RL工作的基座背景；不把报告能力宣传作为本项目实测 |
| R01 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | arXiv 2024 v3 | 摘要、算法出处核对 | GRPO原始出处；数学结果不等于视觉分类结果 |
| R02 | Visual-RFT: Visual Reinforcement Fine-Tuning | ICCV 2025；方法细读使用arXiv v1 | 方法、分类奖励、实验设置及官方仓库 | 最贴近“VLM+RL+细粒度分类”的参照；出版版与早期版数值有变化，不混抄性能数字 |
| R03 | Specificity-aware reinforcement learning for fine-grained open-world classification（SpeciaRL） | 本轮按arXiv 2026 v1引用 | 方法、开放标签定义、附录及作者仓库 | 开放世界细粒度预测与正确性/具体性权衡；不能原样当作七类闭集奖励 |
| R04 | VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning | 引用arXiv 2025 v3；作者仓库标注NeurIPS25 | 摘要与官方仓库 | 组内优势消失、样本重放；推理任务结论不等于机型识别 |
| R05 | DeepEyes: Incentivizing “Thinking with Images” via Reinforcement Learning | 本轮按arXiv 2025 v1引用 | 方法、工具交互、奖励、训练设置 | 真正工具型Agent RL的参照；不把单轮文本回答冒充智能体 |
| R06 | AgentIAD: Tool-Augmented Single-Agent for Industrial Anomaly Detection | 引用arXiv 2025摘要记录 | 摘要；抓取正文标题与摘要有版本差异 | 细节观察/参考检索与SFT→RL；非无人机，引用前须锁定具体PDF版本 |
| R07 | SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training | 引用arXiv 2025 v1 | 全文任务、结论/局限及官方仓库 | SFT/RL公平比较与视觉变体泛化动机；结论仅限研究设置 |
| R08 | Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? | 引用arXiv 2025 v1 | 全文视觉推理与讨论部分 | pass@k/采样分布角度的反面证据；非无人机实验，也不是“RL都无效” |
| R09 | RL Fine-Tuning Heals OOD Forgetting in SFT | arXiv 2025 | 摘要与正文研究设置 | 解释SFT检查点和OOD恢复；提醒不要挑过拟合SFT当弱对照 |
| R10 | Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability | arXiv 2026 | 摘要核对 | 补充SFT泛化条件与训练预算控制；未精读，不引用具体数值 |
| H01 | ViperGPT: Visual Inference via Python Execution for Reasoning | ICCV 2023 | 会议摘要与arXiv摘要 | 大模型编排专用模块的已发表依据；不需要RL的系统也能称工具型推理，但不能称强化学习训练 |
| H02 | V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs | CVPR 2024（预印本2023） | 全文引言与架构部分 | 搜索视觉细节与模块分工；其最终问答模型仍可为MLLM，不是“所有分类都应交给小模型”的证据 |
| H03 | Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions | CVPR 2024 / arXiv v2 | 会议摘要、arXiv摘要及作者代码索引 | LLM产生语义描述、识别由视觉语言模型完成的直接参照；论文也指出单纯测试时增加描述不必然有益 |
| H04 | Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs | CVPR 2024 | 会议摘要及arXiv摘要 | 专用视觉表示和细节辨识仍有价值；旧模型视觉短板不能直接归结为所有新模型缺陷 |

## 每条文献可以支持与不能支持的说法

### D01 — `aldosari2024dronetype`

可支持：机型图像数据与目标框可用于构造明确的识别任务。作者报告7类7000图，视频来源需要特别关注近重复与来源隔离。
不可支持：现成拥有三模态配对数据；把GT框裁剪分类称为端到端检测。
原始来源：`https://arxiv.org/abs/2405.10398`

### D02 — `yuan2024mmaud`

可支持：同步多传感器和目标无人机型号/轨迹的数据方案具有公开研究依据。论文说明Leica位置真值及其缺失/频率限制。
不可支持：将轨迹真值改名为机载电量/姿态日志；给任意网页图片配上同型号的一条轨迹。
论文：`https://arxiv.org/abs/2402.03706`
作者项目：`https://github.com/ntu-aris/MMAUD`

### D03 — `bozcan2020auair`

可支持：RGB与飞行平台时间、位置、IMU、速度的配对与处理。
不可支持：直接把交通目标标签当作无人机型号；把拍摄平台日志当作被拍目标日志。
来源：`https://arxiv.org/abs/2001.11737`

### M01 / M02 — `qwen2025qwen3vl` / `bai2025qwen25vl`

可支持：使用公开视觉语言基座开展图文处理与后训练。
不可支持：采用了这些基座就自动具备可靠的三模态融合；数学/通用视觉分数可换算为无人机准确率。
来源：`https://arxiv.org/abs/2511.21631`；`https://arxiv.org/abs/2502.13923`

### R01 — `shao2024deepseekmath`

可支持：GRPO算法来源。我们需要说明组采样、奖励验证和策略更新。
不可支持：无人机任务上RL一定优于SFT；把LoRA或提示词修改称为GRPO。
来源：`https://arxiv.org/abs/2402.03300`

### R02 — `liu2025visualrft`

可支持：视觉感知和细粒度分类可以构造可验证奖励进行RL后训练。闭集标签核验是合理起点。
不可支持：把其少样本实验提升幅度移植为我们的预期结果；仅有框而无类别核验也可随意训练。
出版记录：`https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Visual-RFT_Visual_Reinforcement_Fine-Tuning_ICCV_2025_paper.html`
方法版本：`https://arxiv.org/html/2503.01785v1`
官方代码：`https://github.com/Liuziyu77/Visual-RFT`

### R03 — `angheben2026special`

可支持：细粒度识别中，“说得更具体”与“说得正确”需要分别评价。
不可支持：开放世界LLM验证器无需改造即可用作我们固定标签奖励；更长解释一定更可信。
来源：`https://arxiv.org/html/2603.03197v1`；`https://github.com/s-angheben/SpeciaRL`

### R04 — `wang2025vlrethinker`

可支持：RL后训练需要关注组内有效优势与采样质量。
不可支持：加入强制反思就必然改善机型视觉细节辨识。
来源：`https://arxiv.org/abs/2504.08837`；`https://github.com/TIGER-AI-Lab/VL-Rethinker`

### R05 — `zheng2025deepeyes`

可支持：大模型自主调用图像工具、消费返回图像，再生成结果，可以作为Agent RL的具体参考。
不可支持：规则程序代替模型选择动作却声称策略经过RL；将GT框作为测试时工具输入。
方法版本：`https://arxiv.org/html/2505.14362v1`
代码：`https://github.com/Visual-Agent/DeepEyes`
注意：目前该仓库也涉及V2，复现时不能把V1论文与V2配置混用。

### R06 — `miao2025agentiad`

可支持：工业视觉任务中用观察/检索工具、分阶段训练的研究动机。
不可支持：直接把工业异常识别结果当无人机效果。抓取摘要与正文呈现不同标题，本文只按已核对的摘要讨论，正式引用需再次锁版本。
来源：`https://arxiv.org/abs/2512.13671`

### R07 — `chu2025sftvsrl`

可支持：特定规则和视觉变体任务中，比较SFT与RL的泛化；输出格式适配与后续RL之间有联系。
不可支持：把“SFT只记忆、RL必泛化”当定律。官方复现实验的多卡需求不能被误写为单卡最低需求。
来源：`https://arxiv.org/html/2501.17161v1`；`https://github.com/LeslieTrue/SFTvsRL`

### R08 — `yue2025rlcapacity`

可支持：区分单次回答表现与高采样覆盖能力；讨论RL可能主要重分配已有输出概率。
不可支持：否定一切RL应用，或据数学视觉实验断言无人机任务没有收益。
来源：`https://arxiv.org/html/2504.13837v1`

### R09 / R10 — `jin2025rlheals` / `ren2026rethinkingsft`

可支持：SFT检查点、优化程度与数据质量会影响对SFT/RL差异的判断。
不可支持：直接采用这些预印本结果作为无人机领域定论。
来源：`https://arxiv.org/abs/2509.12235`；`https://arxiv.org/abs/2604.06628`

### H01 — `suris2023vipergpt`

可支持：大模型可以负责组合视觉能力，数值计算和具体感知可以由专用模块承担。
不可支持：任何调用链都经过RL；不经验证即可让自由生成代码读取全项目或测试标签。
来源：`https://arxiv.org/abs/2303.08128`
出版：ICCV 2023，11888–11898。

### H02 — `wu2024vstar`

可支持：将视觉搜索与后续理解协作，检查小细节是合理研究方向。
不可支持：把局部放大生成的图当作新增独立数据；直接沿用其评价分数。
来源：`https://arxiv.org/abs/2312.14135`
出版：CVPR 2024，13084–13094；CVF页面标题的星号有编码差异，BibTeX保留V*。

### H03 — `saha2024adaptvlm`

可支持：LLM用于类别语义构建，最终识别由视觉语言模型完成；必须验证描述如何用于训练。
不可支持：类级描述就是实采工况文本，或仅把生成描述附到每张图片就一定提高效果。
来源：`https://arxiv.org/abs/2401.02460`
官方代码：`https://github.com/cvl-umass/AdaptCLIPZS`
出版：CVPR 2024，17542–17552。

### H04 — `tong2024eyes`

可支持：视觉细节能力需要单独评估，不能只凭语言解释流畅度判断识别可信度。
不可支持：其2024年基座的结论可无差别外推到2026年的所有模型。
来源：`https://arxiv.org/abs/2401.06209`
出版：CVPR 2024，9568–9578。

## 优先阅读顺序

先读：R02（直接RL分类依据）、H03（分类器与LLM分工依据）、D01/D02（任务数据）。
再读：R05/H01/H02（工具与视觉搜索）、R07/R08/R09（公平对照与边界）。
基座报告M01/M02用于模型描述；R10只做补充讨论，正式写作前需精读全文。

## 引用规范

- 正文按实际用到的论点引用，不为了凑数把19篇全部塞进引言。
- `references.bib`包含19条初始条目。大型团队作者暂用首作者+others，不捏造缺失作者；投稿前从原出版页面重新导出最终书目。
- arXiv年份不等于正式会议年份；正式会议未独立核实时保留预印本身份。
- 本轮没有下载保存这些论文PDF，文件夹里是调研笔记、书目与检索记录。
