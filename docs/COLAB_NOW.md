# Colab本轮已完成：操作留作复现参考

更新：2026-10-22。auxiliary_results_for_report.zip已回传并核对，见[结果报告](../reports/auxiliary_experiment_results.md)。实际环境回执为Tesla T4、14.563 GiB、batch=16、float16；早期A100截图判断不用于本轮实验硬件描述。**无需重新上传或运行本轮Notebook。**

本轮为四类合成辅助实验，3999张开发、3999张留出，不能替代七类真实主任务。返回包与导入回执在outputs/imported_results/auxiliary_d27241ce809956a7/。本地未运行软件测试、模型推理、训练或评估。以下为原分步骤复现说明；Drive中模型和特征缓存仍需保留。

## 原运行步骤（已完成）

1. 打开本地`D:\ml_hw\outputs\colab_transfer\`。把其中以下三个文件上传到Google Drive同一个目录`我的云端硬盘/ml_hw_colab/`：
   - `project_code.zip`：模块化源码、配置和固定运行协议。
   - `auxiliary_images.zip`：7998张已准备好的辅助原图，仅用于传输。
   - `transfer_receipt.json`：归档SHA256与版本号。
2. 在Colab选择“文件→上传笔记本”，打开`D:\ml_hw\notebooks\COLAB_RUN.ipynb`。保留截图中已连接的GPU会话。若网页显示不同中文菜单，以“上传笔记本”功能为准。
3. 从第1格到第14格依次运行。每格只调用一个已有独立`.py`步骤。第1格挂载Drive需要你完成Google账号授权。报错时停在对应格，发完整报错。

不用重新下载数据，不上传`.venv`，不把8000张小图逐个上传。传输包里的源代码仍然是独立的`scripts/`与`src/uav_project/`文件，并未合成单文件训练程序。这里只是Colab运行所需代码子集，最终给老师的完整采集源码仍保留在本地项目。

## Notebook每步做什么

| 格 | 作用 | 实际调用 |
|---|---|---|
| 1 | 挂载Drive | Colab账号授权 |
| 2 | 校验、解压代码 | 校验传输回执SHA256 |
| 3 | 安装建模依赖 | requirements-representation.txt |
| 4 | 解压图片、恢复结果、识别GPU | colab_support.py |
| 5 | 开发集手工特征 | 06b_extract_representation.py |
| 6 | 开发集冻结SigLIP2图像表征 | 06b_extract_representation.py |
| 7 | 手工特征四模型与网格搜索 | 07b_train_representation.py |
| 8 | 预训练表征四模型与网格搜索 | 07b_train_representation.py |
| 9 | 冻结模型、数据与代码 | 09b_freeze_representation.py |
| 10 | 冻结后的留出集手工特征 | 06b_extract_representation.py |
| 11 | 冻结后的留出集预训练表征 | 06b_extract_representation.py |
| 12 | 全部声明模型留出评估 | 10b_evaluate_representation.py |
| 13 | CSV、图表与Markdown | 11b_export_representation.py |
| 14 | 下载给我写报告的结果包 | auxiliary_results_for_report.zip |

分类器是逻辑回归、KNN、RBF-SVM和随机森林，不生硬融合。每种表征用相同分类器作比较，开发集三折CV和折内Pipeline防止缩放泄露。依据开发Macro-F1选模型和参数；冻结之前不处理留出图像。辅助数据原val作为留出，比例约1:1，保留来源划分；不是课程七类主任务的7:3划分结果。

## 断线和恢复

图片解压在Colab本地SSD，不直接从Drive逐张读取。每个步骤结束或出错后，把运行产物和日志同步到`MyDrive/ml_hw_colab/results/<传输版本>/`。断线后重新运行1—4，恢复之前已保存产物；再从未完成的步骤继续。已完成特征按签名和分片复用，已冻结训练步骤核对后跳过。

不保证突发断线前尚未同步的进度能够保留；尤其不要在当前步骤还未保存时主动删除运行时。若提示Drive同步失败，保留当前会话，恢复Drive连接后运行`session.sync()`。

精度和环境版本参与缓存签名。如果开发和留出推理时换了GPU导致FP16/BF16改变，程序会拒绝混用，优先继续使用同类会话。不要删除失败日志伪装已成功。

## 运行完发什么

第14格下载`auxiliary_results_for_report.zip`后发回来。里面有真实指标、预测CSV、逐类结果、混淆矩阵、图表、网格搜索和环境记录。大特征矩阵与模型文件保留在Drive，不必重复下载。没有实际结果时程序不会造一个报告成绩包，不生成PDF。

## 当前联网补采状态

本轮成功保存六个公开发现页，解析四篇新文章/文档，共29条新媒体线索；选择八张用于下一轮实际观察。随后图片下载命令被本机自动审批服务403错误拦截，**本轮新增实际图片为0**。这是执行审批错误，不是来源网站返回403，不能算爬虫拿到了八张图。相关记录在`outputs/rare_round2_2026-10-22/`。七类主任务的来源补足仍待继续，不把辅助结果伪装成主任务完成。
