# 十二类无人机机型分类实验方案

更新时间：2026-10-23。

## 研究对象

主任务是根据作者标注框得到的无人机目标裁剪，识别 12 个数据集定义类别。输入是 `target_crop`，因此论文中写作“给定目标区域的机型分类”，不写成已经完成端到端目标检测。

当前主池为 20010 张真实目标裁剪，按 `data/delivery/uav_dataset_v3/clean_manifest.csv` 读取。第三方合成数据和网页弱标签数据不混入本实验。自采网页数据保留为采集与清洗过程证据，不拿弱标签冒充独立测试集。

## 固定协议

`scripts/05d_prepare_uav12_protocol.py` 已经生成版本化协议：开发集 14106 张，留出集 5904 张，比例 70.50%/29.50%。划分沿用数据阶段已确定的约束组，不使用模型分数挑选。

这不是完整 12 类独立来源测试。留出集覆盖 8 类：DJI phantom 4 pro、Emax、YH 19Hw、Anafi-Extended、DJIFPV、EFT-E410S、Mavic_Air、Mavic_Enterprise。Bebop、DJI phantom 3、Intel Aero RTF、Mambo没有留出组。开发侧可进行分组折评分的类别实际为 6 类；其余类别仍进入每个模型的训练折，但不计入开发折 Macro-F1。相关场景组还没有被认证为独立飞行或独立视频来源。

因此报告必须同时写清：

- 传统模型和 Qwen 都训练 12 类；
- 开发折 Macro-F1 只在每个验证折实际出现且预先声明的 6 类上计算；
- 最终留出 Macro-F1 只在留出中实际出现的 8 类上作为主要指标；
- 12 类 Macro-F1 只能作为标签空间参考，不是 12 类完整测试成绩。

## 模型对照

每种视觉表示分别运行四个独立分类器：逻辑回归、KNN、RBF-SVM、随机森林。预处理放在 Pipeline 内：中位数填补、零方差筛选；逻辑回归、KNN 和 SVM 标准化，随机森林不标准化。先用开发分组折选出模型家族，再只对选中的家族网格搜索，最后用全部开发数据拟合。留出集在全局冻结前不能读取。

第二种表示是冻结的 `google/siglip2-base-patch16-224` 图像编码器向量。编码器只做固定视觉表示，不训练语言分支，也不声称已经完成图像、工况文本和飞行日志的真实配对融合。

主方法是 Qwen3-VL-2B-Instruct 的 4-bit QLoRA 监督微调。大模型输入目标裁剪和固定的 12 类标签列表，输出一个 `LABEL_00` 到 `LABEL_11`。视觉骨干不更新，只对语言侧 LoRA 模块训练。预先声明两个组：普通交叉熵和平方根逆频率的类别加权交叉熵；两组使用同一数据、随机种子、轮数和梯度预算。它们是监督微调对照，不是强化学习。当前不把 GRPO 或工具型智能体塞进主实验，因为本项目没有自主动作、环境反馈和多轮工具轨迹。

## 运行顺序

本地的数据协议与传输包已经准备好；不需要再运行本地模型、特征提取或测试。当前交付目录为 `outputs/colab_uav12/`。将其中的 `uav12_code.zip`、`uav12_images.zip`、`transfer_receipt.json` 和 `UAV12_T4.ipynb` 四个文件上传到 Google Drive 的 `MyDrive/ml_hw_uav12/`，从该目录打开 Notebook，按单元格顺序运行。Notebook 每一格都调用独立入口：

1. `06d_extract_uav12_features.py --representation handcrafted`
2. `06d_extract_uav12_features.py --representation siglip2 --batch-size 16`
3. `07d_train_uav12_baselines.py --representation handcrafted --jobs 2`
4. `07d_train_uav12_baselines.py --representation siglip2 --jobs 2`
5. `07q_train_qwen3vl.py --variant standard`
6. `07q_train_qwen3vl.py --variant sqrt_class_balanced`
7. `09d_freeze_uav12.py`
8. 两次 `06d_extract_uav12_features.py` 读取冻结后的 holdout
9. `10d_evaluate_uav12.py`
10. `11d_export_uav12_report.py`

出现错误时停在当前单元格；恢复 Colab 后重新执行挂载、展开、恢复和环境记录单元，再从未完成的脚本继续。Qwen 检查点写入 Drive，留出预测每 250 张写入进度文件。

## 论文结果口径

传统模型结果用于满足课程的多模型训练、指标比较、网格搜索和特征重要性要求。Qwen 零样本、普通 LoRA 和类别加权 LoRA 用于前沿方法对照。主指标为 Macro-F1、Balanced Accuracy、Accuracy、逐类召回和混淆矩阵；Qwen 的非法格式输出单独计数并按错误处理。

最终报告中区分“数据集整理完成”“实验协议完成”“模型实际运行完成”和“独立来源泛化成立”。不能在运行前填写性能数值，也不能把辅助四分类实验数字改写为十二类结果。
