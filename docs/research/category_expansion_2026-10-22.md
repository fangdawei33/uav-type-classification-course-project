# 真实机型类别扩充核查

记录日期：2026-10-22。本轮用户授权寻找更多类别以扩大训练数据。现有七类固定数据v2和四类辅助实验不覆盖；新数据单独获取，尚未改变训练标签。

## 重点候选：VisioDECT作者发布版本1

- 作者Kaggle标识：simeonajakwe/visiodect-for-scenario-based-multi-drone-detection。
- 固定下载标识：simeonajakwe/visiodect-for-scenario-based-multi-drone-detection/versions/1。
- 作者元数据报告20924张图像，6类：Anafi-Extended、DJI FPV、DJI Phantom、EFT-E410S、Mavic2-Air、Mavic2-Enterprise。
- 元数据说明：真实视频抽帧，852×480；Cloudy、Sunny、Evening条件；提供TXT、XML、CSV标注。12个地点、约20个月是作者描述，不是已经核实的独立序列数量。
- 作者页面结构化下载信息：2100225748字节，约2.10GB；API列出的2251932239字节与下载包体积口径不同，原值均保留。
- 页面许可：CC BY-NC-SA 4.0。代码开源和数据再分发分别处理，保留作者署名与许可。
- 作者下载入口公开可用，官方kagglehub固定版本下载已完成。实际文件数以outputs/expansion/visiodect_v1/中的回执为准。

## 已取得的来源证据

1. outputs/platform_inventory/2026-10-22T135352.006267_0000/visiodect_search.json：四条搜索返回记录，选作者账号，不选许可未知的转载副本。
2. 同目录visiodect_author_dataset_version.html：作者页面标题、结构化Dataset说明、版本和许可。
3. outputs/platform_inventory/2026-10-22T135547.251606_0000/visiodect_author_metadata.json：当前版本1，创建日期2025-09-24。
4. 同目录visiodect_author_files.json：实际读取第一页200条文件，存在nextPageToken，不能把第一页称为完整清单。已观察Anafi-Extended/images/Cloudy/路径和JPEG文件。

以上是本机实际保存的HTTP响应。正文引用论文时另行核实原文书目信息，当前不填未经确认的DOI或卷期。

## 类别映射原则

目标是保留原七类，优先核实五个不重叠的新标签，而不是为了凑11类拆分真实/合成来源：

| 作者标签 | 当前处理 |
|---|---|
| Anafi-Extended | 新类别候选，需核查实际目录、整机图像与标注 |
| DJI FPV | 新类别候选，不能与所有FPV品牌机型合并 |
| EFT-E410S | 新类别候选，需核实命名及图像 |
| Mavic2-Air | 保留作者原名；未核实前不擅改成Mavic Air 2或其他型号 |
| Mavic2-Enterprise | 保留作者原名；与其他Mavic类别检查区别 |
| DJI Phantom | 系列级标签，与现有Phantom3/4Pro重叠，暂不加入统一细分类训练 |

因此可能扩至12个作者定义类别，但这不是已经完成的12分类数据集。实际合格数量必须在解码、标注、去重和视觉核查后确定；不能先将20924全加到2720并称为可训练数据。

## 获取与检查入口

1. scripts/01e_download_expansion.py：只下载作者固定版本，存放data/raw/expansion/visiodect_v1/。
2. scripts/03g_catalog_expansion.py：实际图像解码、原文件SHA256、类别/天气计数、标注配对、完全重复核查；不运行模型或软件测试。
3. scripts/03h_preview_expansion.py：每个作者类别/天气按固定规则抽8张生成原图接触表，准备不等于已完成审核。

逻辑在src/uav_project/expansion_data.py，配置在configs/expansion_sources.json，额外下载依赖在requirements-expansion.txt。

新的来源、场景和视频相邻帧必须保留；天气目录不是flight_id。类别增加不能消除原七类的来源不足问题。旧来源不足的类别仍需明确验证范围；暂不进行新划分、GPU推理或训练。

## 下载完成后的直接观察

作者版本1已于本机完成下载、解压，回执在outputs/expansion/visiodect_v1/download_receipt.json。实际目录为Anafi-Extended、DJIFPV、DJIPhantom、EFT-E410S、Mavic_Air、Mavic_Enterprise。

按文件目录实际计数20924张：Anafi-Extended3471、DJIFPV3600、DJIPhantom3300、EFT-E410S3600、Mavic_Air3521、Mavic_Enterprise3432。五个非Phantom目录共17624张，尚未全量审核，不是已批准训练数量。

与元数据不同的实际观察：

- 图像分辨率不止852×480，原图样例包括1920×1080、1080×1920、2400×1080、720×1280等；不统一假设或直接缩放标注坐标。
- Anafi/cloudy当前只有汇总csv.csv，没有逐图TXT/XML，编目需支持从作者CSV读取对应文件名和xywh。
- DJI FPV/cloudy的TXT目录名为txt1，不能硬编码只找txt/。
- Phantom/sunny的汇总表是csv.xlsx；其他目录多为csv.csv。保留原表，优先读取已有逐图XML。
- Mavic Enterprise的某张Evening图对应标注名称包含Cloudy文字；类别、天气与文件名分别保存，不凭单个name字段覆盖来源信息。
- 所有上述观察仅为文件例子，不等同于全量数值坐标或视觉审核已经通过。

补充入口scripts/03i_audit_expansion_boxes.py检查单目标框数值/尺寸、CSV回退、XML/CSV差异和小目标数量，导出固定样本原图/裁剪图用于实际查看，不调用模型。

完成标注核查后，scripts/04g_prepare_expanded_candidates.py生成可追溯的12类候选CSV：原七类全部保留，新增五个作者目录类别；通用Phantom、数值非法框及完全重复按规则隔离。这个文件不执行划分，不将候选状态改成全部视觉通过，不覆盖v2。

潜在混杂：旧类主要来自Drone-type-Set，新类主要来自VisioDECT，因此“机型标签”和“来源域”可能相关。后续必须区分来源域、目标尺寸和真实/合成，不能仅通过混合后随机划分取得高分就声称独立真实泛化。类数扩大与有效来源数量分别统计。

## 次级候选：FRED

此前已取得FRED官方项目、数据卡和目录元数据；作者报告5类、同步RGB/事件数据，每个ZIP对应序列，原始全模态文件约205GB。它可能更适合序列分组，但完整模型类别映射尚未核实，本轮不与VisioDECT同时启动大体积下载。相关历史证据见platform_datasets_and_synthetic_2026-10-22.md。

## 已完成的全量文件与目标框核对

全量编目于2026-10-22完成，实际59795文件：20924张图像、38869个TXT/XML/CSV文件、2个其他文件。20924张全部成功解码，无解码或XML解析问题记录。文件哈希发现14组完全重复、0组跨作者类别完全重复。全部原文件清单与哈希保存在outputs/expansion/visiodect_v1/all_acquired_files.csv。

按同名逐图标注匹配时1507张缺配，但这不等于1507张无法使用：后续已读取作者汇总CSV作为回退。全量目标框检查结果为20612张单目标框数值有效、307张无可解析框、5张多框需处理；尺寸及边界检查已执行，CSV与VOC比较没有超过2像素的差异。数值有效不是图像内容通过。

实际查看了boxes/crops_01.jpg至crops_18.jpg全部18页，共144对原图和目标裁剪，每个作者类别/天气8张，选择规则为按文件路径排序的等间隔索引。不是随机代表性抽样，也不是全量逐图审核。另单独打开Mavic_Enterprise/Evening/MAVIC2_Enterprise_Dual_Evening (117).jpg原图，确认无法辨认无人机，作者框内只有天空。该样本已从新版候选清单排除。其他抽样存在自然模糊、低分辨率和相似背景，不据此自动把整批批准为合格。

实际观察转录：outputs/expansion/visiodect_v1/visual_sample/observations.csv，生成入口scripts/review_records/2026_10_22_visiodect_sample.py。记录仅针对实际查看的144项，非专家产品身份鉴定。

## 本轮候选交付数量

保留原七类全部2720张，新增五类实际保留17298张，合计20018张、12类：

| 类别 | 候选数 | 来源 |
|---|---:|---|
| Bebop | 447 | 原七类固定v2 |
| DJI phantom 3 | 67 | 原七类固定v2 |
| DJI phantom 4 pro | 572 | 原七类固定v2 |
| Emax | 329 | 原七类固定v2 |
| Intel Aero RTF | 256 | 原七类固定v2 |
| Mambo | 578 | 原七类固定v2 |
| YH 19Hw | 471 | 原七类固定v2 |
| Anafi-Extended | 3371 | VisioDECT作者v1 |
| DJIFPV | 3439 | VisioDECT作者v1 |
| EFT-E410S | 3552 | VisioDECT作者v1 |
| Mavic_Air | 3519 | VisioDECT作者v1，沿用目录名称 |
| Mavic_Enterprise | 3417 | VisioDECT作者v1，沿用目录名称 |
| 合计 | 20018 | 未加入合成或网页弱标签池 |

新增20924张的去向：通用Phantom3300张隔离，307张无可解析目标框、4张非Phantom多框、14张同类完全重复、1张实际观察无可辨目标排除，剩余17298张。第五张多框属于已隔离的Phantom，不重复计数。新增候选与旧七类文件哈希交叉比较，未发现完全重复。

当前入口outputs/expansion/visiodect_v1/candidates_sampled_review/expanded_12class_candidates.csv，旁有new_5class_candidates.csv、excluded_expansion.csv、class_map.json、receipt.json。初版candidates/20019张清单保留为观察前记录，当前采用剔除纯天空样本后的20018张版本。原始图像与既有v2均未覆盖。

全部20018行仍为split=unassigned、training_split_approved=False；这两个字段反映处理状态，不表示需要用户额外批准才能继续。新增裁剪尚未批量生成，750个新增候选的目标短边小于24像素；完整内容筛查、近重复帧约束、裁剪派生和实验划分是下个数据处理步骤。不会为追求数量把天气目录编造成独立飞行编号。类别数和来源独立性分别报告，公开/网页和真实/合成分别留证。

已执行顺序：01e下载、03g编目、03i目标框检查、04g初版候选、review_records/2026_10_22_visiodect_sample.py实际观察转录、04g --sampled-review修订候选。当前无需用户重跑；未运行模型、特征提取、训练、软件测试或生成PDF。
