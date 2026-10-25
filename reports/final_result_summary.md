# 最终实验结果摘要

项目代码仓库：https://github.com/fangdawei33/uav-type-classification-course-project

完整正文为 [final_report.md](final_report.md)，Word 为 [final_report_revised.docx](final_report_revised.docx)。正文按老师要求设置八章，附录保存复现材料和要求映射。

真实裁剪共 20,010 张，训练集 14,106 张，测试集 5,904 张，约 7∶3。development 对应训练集，holdout 对应测试集。传统模型在训练内部做两折分组验证；Qwen 未设固定验证集，未实施验证早停。

| 方法 | 测试准确率 | 测试八类 Macro-F1 |
| --- | --- | --- |
| 手工＋调优 KNN | 0.7307 | 0.6108 |
| SigLIP2＋调优 KNN | 0.5960 | 0.5739 |
| Qwen 零样本 | 0.0159 | 0.0794 |
| Qwen 标准 LoRA | 0.4946 | 0.3668 |
| Qwen 加权 LoRA | 0.4804 | 0.3670 |

限制：训练十二类、测试八类、验证计分六类；来源组未认证为独立飞行。自采原图最终 179 张，弱标签材料没有并入主监督评价。1,262 张目标短边小于 24 像素，未开展尺度筛选后重训。不能宣称完整十二类外部泛化或真实图文传感器融合。

本次只补正文、纠正分数引用、整理保存结果与数据质量统计，没有重新训练或修改预测。
