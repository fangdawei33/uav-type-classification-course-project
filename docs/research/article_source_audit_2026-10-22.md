# 原始测评与教程采集核查

> 这是本日前次93张阶段的历史记录。后续已取得新增候选；当前102候选＋6参考图的证据见[source_identity_audit_2026-10-22.md](source_identity_audit_2026-10-22.md)及统一状态JSON。

日期：2026-10-22。以下区分实际HTTP结果、已取得的数据和未取得的数据；没有运行软件测试、模型训练或评估。

## 已确认的Mambo来源

- 站内检索：https://www.halfchrome.com/?s=mambo
- 实际观察到并访问的原文：https://www.halfchrome.com/tello-vs-mambo/
- 标题：Ryze Tello vs Parrot Mambo: Which is the Best Micro Drone?
- 原文HTML、正文文本、来源清单均已保存；正文有14条图片线索。
- 选中Parrot-Mambo、Mambo-Front-View、Mambo-Bottom-View三张照片，其余为Tello/其他机型、参数图和文章宣传图，未下载。
- 3张照片均1024×580；原始字节、SHA-256、图注、来源页和网页快照保留。
- 三张原图均已逐张查看，全部标为candidate_keep。正面近景和底部图存在机体接近/到达边界的情况，记录标记，不据此自动剔除。
- 图注可作为机型标签证据，但最终类别映射、来源独立性和框仍待审核。三图来自同一文章/相似桌面布景，纳入一个相关性组，不能算三次独立飞行。
- 网页声明不足以自动确认图片再分发许可，保留pending状态。

## Intel Aero：取得页面，未取得图片

- 官方原始资料目录：https://github.com/intel-aero/meta-intel-aero/wiki/99-External-Links
- 目录明确链接作者教程：https://www.instructables.com/Intel-Aero-Drone-Adding-Propeller-Guards/
- GitHub目录HTML已保存，链接来自实际页面，不是猜出的图片地址。
- 教程页面及robots首次访问均返回200；标题为Intel Aero Drone - Adding Propeller Guards。
- 初始正文article选择器为空。检查已保存HTML发现页面把公开内容放在application/ld+json中，随后实现jsonld_howto解析，直接读取同一份已取得的HTML。
- HowTo中解析出6条媒体线索；选择首图与末步图作为待查看候选，其余为零件/安装细节，不选为完整机体候选。
- 下载两图前，对content.instructables.com/robots.txt的请求均返回403；停止该来源，没有下载图片，没有改用缩略图、换域名或绕过检查。
- Intel Aero实际图片计数仍是0。官方说明页只能作为参考来源，不能报告为已有独立外部测试观测。

## 失效与不相关来源

| 地址 | 实际观察 | 处置 |
|---|---|---|
| https://www.halfchrome.com/parrot-mambo-fpv/ | HTTP 404 | 不作为可用来源，改查站内公开检索结果 |
| https://www.halfchrome.com/yh-19hw/ | HTTP 404 | 不采集；YH-19HW保持缺口 |
| https://discuss.ardupilot.org/t/intel-aero-rtf-testing-setup-instructions/15706 | 301后实际标题Skywalker x8 waveing during flight in AUTO mode | 拒绝无关来源，唯一媒体线索为日志截图，未下载 |

站内对yh、19hw、yh 19hw的本轮查询没有发现匹配链接；这不代表全网无数据。Emax仅凭品牌名称无法确认具体产品，不能拿其他EMAX机型照片直接赋成同一类别。

无关论坛页面与线索保留为rejected_unrelated_redirect_target/reject，便于解释问题；没有删除历史记录或把错误页面混入图片集。爬虫新增文章标题身份校验，阻止后续同类误收集。

## 当前数据与脚本

- Commons：90张，3类；原始测评：3张Mambo；合计93张，4类。
- 首轮视觉决定：66张候选暂留、20张待定、7张排除。
- 相关性约束：13组、52张；不是已确认的独立飞行计数。
- scripts/02b_crawl_articles.py：明确文章发现、选定媒体下载、离线快照解析。
- scripts/03b_review_external.py：两个采集清单统一生成预览；--summary-only更新现有决定汇总。
- 已修正相似候选表重生成时覆盖已有复核意见的问题：现在将意见绑定到两张图的SHA-256，只有文件未变化才保留。
- 93张候选的质量准备和文件/快照哈希核对实际执行；未运行软件测试或模型实验。

文章正文和图注仅作溯源/标注证据。这些候选进入外部评估后，对应文章内容不得用作分类器输入、测试提示或调参材料。

## 证据文件

- data/metadata/drone_type_article_pages.csv：页面身份、快照、文本和失败来源隔离。
- data/metadata/drone_type_article_media_inventory.csv：逐项选择理由。
- data/metadata/drone_type_article_raw.csv：实际下载的3张照片。
- outputs/drone_type_article_visual_review_2026-10-22.json：原图审核及框提案证据。
- outputs/drone_type_source_link_probe_2026-10-22.json、第二轮round2文件：公开页面链接检索证据。
- outputs/logs/requests.jsonl、article_crawl.jsonl、source_link_probe.jsonl、errors.jsonl：实际请求与失败过程。
- outputs/crawl_pages/：按内容哈希保存的原始HTML。
- outputs/drone_type_external_current_status.json：当前统一数量与边界。

本轮最后一次Intel图片下载摘要记录的是失败尝试；不能仅凭最新单次运行摘要否定此前Mambo采集成功，也不能把失败中的selected媒体计为已下载。
