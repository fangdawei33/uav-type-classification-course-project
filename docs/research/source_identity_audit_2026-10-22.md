# 补采、机型身份与来源核对记录

> 后续执行更新（2026-10-22）：新增63张爬虫候选已完成接触表初筛，14张补看原图；当前125暂留/32待定/8排除，171张文件与165条观察核对通过。Cranfield镜像8001张已下载解码，隔离3张后辅助train/val各3999张。下文保留此前调查/轮次状态，当前以docs/DATA_STATUS.md及reports/auxiliary_data_report.md为准。

日期：2026-10-22。此前93张记录为历史；当前102张实验候选、6张身份参考图、共108张自采集原图。新增9张候选全部实际查看，新增7项框提案的源图/裁剪卡已查看；高分辨率原图在查看工具中可能缩放，不声称所有图片都做了逐像素原生比例检查。

## YH 19HW：四张真实候选

实际来源：https://www.halfchrome.com/19hw-good-looking-spark-clone/ ，标题19HW: A Good Looking Spark Clone。旧的/yh-19hw/地址404保留为失效记录。

从10条媒体线索中选择4张实拍，排除遥控器单图、对照中的其他机型、推广/重复线索。四张分别为俯视、与遥控器同图、正面近景、亮灯斜视。正面近景有自然截断，仍可见中央机体；与遥控器同图的框仅取无人机。

四张均视觉暂留，加入同文章/桌面相关约束ext_halfchrome_yh19hw_tabletop_article，不是四次独立飞行。四项框均为proposed_not_final。斜视图初始框漏裁左桨，已查看后修正为(65,44,962,577)，旧卡、修正卡及JSON均保留。

来源快照：outputs/crawl_pages/41431d9f9dc47c6c11b99d51e90c879355d22ff3d68df35034c5be8ba3c18e66.html。媒体选择回执：outputs/yh19hw_collection_selection_2026-10-22.csv。

## Emax：先区分产品，再追溯作者类别

作者类名是Emax，品牌本身不足以确认具体机型。实际查看Emax (44)、(206)、(284)等原图，观察到白色桨罩、红色桨、白色遥控器/眼镜；206可见READY SET DRONE上衣，284是遥控器旁的白红机体及三角图形。这些观察不能自动证明333张全部同一商品型号。

首先采集了：https://www.quadifyrc.com/reviews/emax-tinyhawk-setup-and-review ，六张实拍实际查看，含整机、包装/配件和底面。由于与作者Emax类别对应未确认，六张统一标sample_scope=reference_only，不计入实验候选。没有把所有EMAX品牌产品并成作者类别。

随后实际来源：https://www.halfchrome.com/emax-ez-pilot-fpv/ ，标题Emax EZ Pilot – Another great intro to FPV。9条媒体线索选4张：3张白红完整机体视觉暂留，1张上壳打开的拆机照片待定。前景目标与后景其他机型分开，三项框提案已查看。同文章/桌面相关组ext_halfchrome_ezpilot_tabletop_article约束全部四张。

来源正文讨论原版EZ Pilot；图片alt中某项同时出现Tinyhawk，原始歧义保留，不仅凭alt赋真值。现阶段“更可能是原版EZ Pilot”是身份假设，不改写作者Emax标签。

### 新取得的官方证据

实际访问并保存EMAX官方产品页：https://shop.emax-usa.com/products/ez-pilot-drone-only-replacement 。HTML标题为EMAX EZ Pilot Replacement Drone Only (No Battery) | EMAX USA | Emax USA。

产品页实际嵌入6条YouTube链接，其中iframe发布标题包括：

- Finally a Beginner FPV Drone - EMAX EZ Pilot - CAPTAIN DRONE。
- EMAX EZ Pilot Beginner FPV Drone Kit - Ready Set Drone，链接https://www.youtube.com/embed/XUhDe4JYhXg 。
- Learn FPV in 5 Mins - EMAX EZ Pilot - Drone Camps RC。
- EMAX EZ Pilot FPV Drone Review - Giant Ant Cowboy。

这为206图的衣服文字提供具体可追溯的视频候选；视频尚未取得并匹配具体帧，不能填confirmed、flight_id或声称已找到全部原始视频。六条实际链接及官方页哈希见drone_type_source_video_leads.csv和outputs/drone_type_emax_identity_audit_2026-10-22.json。

官方页快照：outputs/crawl_pages/b99618d6706b2213d9c03eec34d85d64f3947de90019414b33bf6f1263e8b890.html。原版EZ Pilot媒体选择回执：outputs/ezpilot_intel_collection_selection_2026-10-22.csv。

## Intel Aero：官方原图已取得，运行场景仍不足

实际官方来源：https://github.com/intel-aero/meta-intel-aero/wiki/01-About-Intel-Aero 。在#wiki-body中发现同段缩略图和完整图片链接，使用页面实际发布的原图：https://github.com/intel-aero/Documents/raw/master/doc_photos/unboxing_Intel_Aero_RTK_kit.jpg 。GitHub重定向官方raw主机，未猜造媒体地址。

原图4032×2968，已查看；泡沫包装中可见中央机体/碳臂及遥控器，桨未安装，包装有Intel Aero Ready to Fly Drone文字。该图对身份核对有价值，但不是已批准的正常运行/飞行场景测试样本，仍pending，不强行标最终框。同组ext_intel_official_rtf_unboxing只是保守相关来源约束。

此前Instructables内容站robots403路线保持停止；旧PX4文档404、无关论坛跳转失败保持原记录。官方原图成功不反向改写这些失败。

raw.githubusercontent.com/robots.txt实际明确404，按RFC 9309第2.3.1.3处理缺失排除文件，仅404分支允许，关闭HTTP错误响应并保存状态/策略记录。403、429、5xx和网络错误不进入该分支。依据：https://www.rfc-editor.org/rfc/rfc9309.html#section-2.3.1.3 。

## 审核、完整性及实际问题

当前102候选为73视觉暂留、22待定、7排除；6参考图另列。新增9项观察、6张参考图检查及来源页哈希保存在outputs/drone_type_external_visual_review_supplement_2026-10-22.json。审核前旧CSV与分组/框表已另存，不覆盖历史意见。

当前核查实际完整读取108张原图，检查102项候选视觉证据、6项参考记录、10项框提案、61项相关约束，最终0项记录不一致。该核查不证明类别真值或来源独立，未运行软件测试/模型训练/评估。

实际遇到并解决的问题：

1. 初始YH框漏裁左桨：扩大左边界并查看修正卡；保留旧证据。
2. 两张Commons照片有EXIF方向6/8，采集表按存储尺寸，审核框按旋转后显示坐标。核查代码最初错误比较两者，已分别验证；原图不旋转重存。
3. 旧Mambo审核使用decision字段和独立article_visual_review快照；新核查最初仅加载external_visual_review与visual_decision，导致三项记录未找到。已兼容实际历史结构，不伪造新审核。
4. 官方产品页在受限执行环境最初无法确认robots，随后使用已授权联网执行取得公开页，保留前次失败与后续成功。未把网络访问失败误写为站方403。
5. 读取官方页正文到终端时GBK不能编码©，出现UnicodeEncodeError。页面快照已保存；随后按结构解析并保存UTF-8 JSON/CSV，未为这个显示错误重采原图。
6. 旧报告导出硬编码93张和三类缺失，已改为读取当前覆盖表/状态；当前只生成PNG、CSV、JSON和Markdown，不再生成PDF。

关键结论：七类都有候选，仍缺足够且已验证的独立来源、最终类别/框。Bebop、Intel、Mambo公开部分各只有一个观察相关组，当前数据不能支撑声称来源独立的七类内部7:3与分组K折。模型方法和GPU训练须建立在真实可行协议之上。
