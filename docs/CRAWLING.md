# 自行编写爬虫与来源记录

更新：2026-10-22。本地实际采集已保存171张原图：165张实验候选＋6张Tinyhawk身份参考图；七类都有候选。当前125视觉暂留、32待定、8排除；新增63张完成接触表筛选，其中14张另看原图，尚非独立外部测试集。无需GPU，已执行内容无需用户重跑。

## 模块职责

| 入口 | 实现 | 配置 | 输出与用途 |
|---|---|---|---|
| 02_crawl_external.py | src/uav_project/crawler.py | configs/crawl_sources.json | Commons分类页→文件说明页→原图，当前146张 |
| 02b_crawl_articles.py | src/uav_project/article_crawler.py | configs/article_sources.json | 原始文章HTML/正文→媒体线索→逐项选择→原图，19候选＋6参考 |
| 02c_discover_source_links.py | src/uav_project/source_discovery.py | configs/source_discovery.json | 公开页面/站点地图实际链接和快照；不自动下载图片/视频 |
| 02e_inventory_platforms.py | src/uav_project/platform_inventory.py | configs/platform_inventory.json | Kaggle/HF公开API、头歌首页及快照，仅目录核查，不自动下载媒体 |
| 03b_review_external.py | external_review.py、external_integrity.py | 现有清单 | 外部证据准备、当前汇总、实际文件核对 |

原始ZIP下载不替代自编爬虫；作者集与自采集文件分开保存。02的category_urls只适用于Commons，文章采集另用article_sources.json，不因补采成功而编造Commons分类页。

## 当前采集范围

Commons：当前Bebop 44、Phantom 3 63、Phantom 4 Pro 39张，共146张。此前首批各30张为试采历史。首批尝试100个文件页，跳过10项；原始311.62 MiB记录保留。仅检查明确分类页，不自动递归Taken with等目录，避免把航拍风景当作无人机目标。

原始文章：

| 来源ID | 内容 | 已保存 | 当前范围 |
|---|---|---:|---|
| halfchrome_tello_vs_mambo | Tello/Mambo对比中的Mambo实拍 | 3 | 候选，均视觉暂留 |
| halfchrome_yh19hw_spark_clone | 19HW实拍测评 | 4 | 候选，均视觉暂留 |
| halfchrome_ezpilot_original | 原版EZ Pilot实拍 | 4 | 3整机暂留、1拆机待定；作者Emax对应未确证 |
| intel_official_about_rtf | Intel官方RTF开箱图 | 1 | 候选待定，未装桨/包装状态 |
| trustedreviews_original_mambo | 第二篇Mambo实拍测评 | 7 | 6视觉暂留，1人物手机图无可辨目标而排除 |
| quadifyrc_tinyhawk_setup_review | Tinyhawk实拍参考 | 6 | sample_scope=reference_only，排除出实验候选 |

原图、URL、时间、来源HTML、正文、图注、尺寸与SHA-256保存在drone_type_article_raw.csv等清单。单次采集摘要可能只反映一个来源/失败尝试；当前数量统一读取outputs/drone_type_external_current_status.json。

## 文章解析和逐项选择

1. 从配置中明确列出的公开文章获取HTML，保存内容哈希快照与正文；校验标题中的机型词，拒绝无关重定向。
2. 只使用正文实际出现的图片地址。Instructables读取公开HTML中的JSON-LD HowTo；GitHub限定#wiki-body。
3. 先生成drone_type_article_media_inventory.csv，人工/AI辅助逐项记录selected或reject及依据，再下载已选条目。商品推广、参数图、比较中的其他机型和遥控器单图不自动赋成目标类。
4. 如官方正文同时提供同一图片的缩略图和同名原图锚点，可以选择实际存在的原图链接；不凭文件名猜下载地址。
5. 每个采集器使用精确域名白名单，重定向逐跳检查。下载使用至少2秒延时、大小上限、栅格图片解码与格式检查，保留失败日志。
6. 身份参考图通过sample_scope与实验候选区分。文章图注可辅助标注，最终测试图对应文章不得用于分类器输入、提示调优或检索答案。

## robots与真实失败处理

每个来源访问前读取robots.txt；403、429、5xx、网络错误或规则禁止时停止/按限定次数重试。Intel教程图片站content.instructables.com的robots返回403，该路线停止，没有改域名/禁用检查绕过。

官方Intel图片的已观察GitHub链接会重定向到raw.githubusercontent.com。该主机robots明确HTTP 404；按RFC 9309第2.3.1.3节处理为未提供排除文件。此分支只接收明确404，不把403/限流/网络失败当允许。请求、状态JSON和生成的解析策略单独保存，并明确策略文本不是下载到的robots内容。

旧YH和Mambo地址404、错误Intel论坛链接跳转无关Skywalker帖子、旧PX4页面404的记录仍保留。找到可用的新来源后另设source_id，不覆盖旧失败；不把“selected但下载失败”写成已取得图片。

EMAX官方产品页现已实际保存，发现6条嵌入视频链接；其中包括Ready Set Drone原版EZ Pilot测评。它们是来源核对线索，未下载/逐帧匹配，未增加confirmed来源组。

## 文件与当前核对

- data/raw/drone_type_external/：Commons原图。
- data/raw/drone_type_articles/：文章候选与参考原图，范围由清单区分。
- data/metadata/drone_type_external_raw.csv、drone_type_article_raw.csv：原始采集记录，不改写采集时pending状态。
- data/metadata/drone_type_external_review.csv：当前审核决定。
- data/metadata/drone_type_external_box_proposals.csv：16项未最终批准的框提案。
- data/metadata/drone_type_source_video_leads.csv：实际嵌入视频线索。
- outputs/crawl_pages/、crawl_articles/：原始网页/正文证据。
- outputs/source_discovery/：来源发现结果。
- outputs/logs/：访问、robots、成功/失败过程。
- outputs/drone_type_external_integrity_current.json：171张实际解码及来源快照核对；165条观察和16项框提案匹配，0项不一致；新增63张的观察层次单独记录；不是标签鉴定或独立性证明。

03b --summary-only刷新当前数量；03b --audit-integrity实际读取文件核对；03b默认重新准备审核图并保留已有意见；内容变化的图页使用新哈希文件名，保留历史证据。已看/未看必须另行记录，不能把生成预览当作已审。

许可仍需核实；网页公开可见不自动授予图片再分发许可。原始文件SHA-256无同文件命中，不排除裁剪、转码或共同来源。外部集尚未冻结，不能据其分数进行方案选择；当前也没有任何模型分数。

公开合成数据的下载不是网页爬虫成果。已下载8001张Cranfield合成原图，单独编目、隔离重复，见../reports/auxiliary_data_report.md；真实网页候选数仍为165。
