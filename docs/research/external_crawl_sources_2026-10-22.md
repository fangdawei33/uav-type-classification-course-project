# 外部采集来源核查记录

日期：2026-10-22。核查方式：搜索并浏览Wikimedia Commons原始分类/文件说明页；随后已在本机实际运行爬虫，首批获得90张待审候选。

已核查：

- https://commons.wikimedia.org/wiki/Category:Parrot_Bebop
- https://commons.wikimedia.org/wiki/Category:DJI_Phantom_3
- https://commons.wikimedia.org/wiki/Category:DJI_Phantom_3_Standard
- https://commons.wikimedia.org/wiki/Category:DJI_Phantom_3_Professional
- https://commons.wikimedia.org/wiki/Category:DJI_Phantom_3_Advanced
- https://commons.wikimedia.org/wiki/Category:DJI_Phantom_3_4K
- https://commons.wikimedia.org/wiki/Category:DJI_Phantom_4_Pro
- https://commons.wikimedia.org/wiki/File:Dji_phantom_4_pro.jpg
- https://commons.wikimedia.org/wiki/File:DJI_Phantom_4_Pro_-_04.jpg
- https://commons.wikimedia.org/robots.txt
- https://upload.wikimedia.org/robots.txt

分类页既有机型子类别，也有Taken with航拍照片子类别；只允许配置中的具体分类及其同页分页，不自动爬下级目录。类别名称并不是文件内目标标签的最终证据。

Phantom 4 Pro目录含机型对比图，部分文件是其他文件的裁剪版本；需做目标实例审核和同源查重，不能按不同文件名统计独立测试样本。

搜索Commons中的Parrot Mambo、Intel Aero、Emax和YH-19/YH19等词，本轮未核实可用来源。该结论只指本次检索，不能据此声称全网没有数据。Emax还需先解决具体产品型号。

配置位置：configs/crawl_sources.json。运行时每个URL仍需重新按robots规则检查；网页浏览核查不能替代程序实跑。


## 后续来源复查

已通过HalfChrome实际站内检索找到Tello/Mambo原始对比测评，并采集3张Mambo候选；外部清单总计93张。旧测评URL404、Intel论坛无关跳转、Intel教程图片域robots403的实际核查记录见[原始文章来源核查](article_source_audit_2026-10-22.md)。先前“本轮未核实”只描述首批Commons检索，不代表后续没有新增来源。
