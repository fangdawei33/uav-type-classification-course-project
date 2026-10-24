"""Build a dated PDF of actual data audit findings; no model results are implied."""
from pathlib import Path
from xml.sax.saxutils import escape
from .common import META, OUT, ROOT, now, read_json, rows


def export_data_report_pdf():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether

    pdfmetrics.registerFont(TTFont("AuditChinese", "C:/Windows/Fonts/simhei.ttf"))
    # Font path is the Windows export environment, not an experiment dependency.
    font = "AuditChinese"
    styles = {
        "title": ParagraphStyle("Title", fontName=font, fontSize=22, leading=30, textColor=colors.HexColor("#173c54"), spaceAfter=14),
        "h1": ParagraphStyle("H1", fontName=font, fontSize=15, leading=22, textColor=colors.HexColor("#173c54"), spaceBefore=10, spaceAfter=9),
        "body": ParagraphStyle("Body", fontName=font, fontSize=10, leading=16, spaceAfter=9, wordWrap="CJK"),
        "small": ParagraphStyle("Small", fontName=font, fontSize=8.5, leading=13, textColor=colors.HexColor("#4b5a64"), spaceAfter=7, wordWrap="CJK"),
        "cell": ParagraphStyle("Cell", fontName=font, fontSize=9, leading=13, wordWrap="CJK"),
        "head": ParagraphStyle("Head", fontName=font, fontSize=9, leading=13, textColor=colors.white, wordWrap="CJK"),
    }
    s = read_json(OUT / "drone_type_set_data_readiness.json")
    ext = read_json(OUT / "drone_type_external_current_status.json")
    integrity = read_json(OUT / "drone_type_set_review_integrity_2026-10-22.json")
    if s["review_csv_sha256"] != integrity["review_csv_sha256"]:
        raise ValueError("Data report and integrity audit refer to different review decisions")
    story = []

    def p(text, style="body"):
        return Paragraph(escape(str(text)), styles[style])

    def add(text, style="body"):
        story.append(p(text, style))

    def table(headers, values, widths):
        data = [[p(v, "head") for v in headers]] + [[p(v, "cell") for v in row] for row in values]
        t = Table(data, colWidths=[v*mm for v in widths], repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,0),colors.HexColor("#254e68")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0f5f8"),colors.white]),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("LINEBELOW",(0,-1),(-1,-1),0.5,colors.HexColor("#bcccd6")),
        ]))
        story.extend([t, Spacer(1, 8*mm)])

    add("无人机机型图像数据\n构建与审核阶段报告".replace("\n", " "), "title")
    add("课程综合设计 / 2026-10-22 / 数据阶段成果，非完整结果论文", "small")
    add("任务是给定标注目标区域（target_crop）的七类机型分类。使用作者真值框属于额外监督，目前没有端到端检测或模型性能结果。")
    table(["实际完成", "数量", "含义"], [
        ["公开图片逐图视觉审核", "2973张", "全部有可追溯视觉证据"],
        ["视觉候选保留 / 排除", "2720 / 253", "0待审；保留不等于来源准入"],
        ["自行编写爬虫取得候选", "93张 / 4类", "未形成正式独立外部测试集"],
        ["正式来源确认", "0组", "目前不具备独立七类评测条件"],
    ], [62,32,80])
    add("1 公开数据实际范围", "h1")
    add("作者网页ZIP包括2973张图片、2973份YOLO标注和1份类别清单，共5947个文件。当前可访问子集与论文所述7000张有差异，按实际取得数量报告。原ZIP、原图和原标注保留。")
    table(["作者类别", "原图", "视觉保留", "排除", "待审"],
          [[r["class_name"],r["raw"],r["visual_keep"],r["visual_reject"],r["pending"]] for r in s["class_summary"]]+[["合计",2973,2720,253,0]], [66,27,27,27,27])
    add("审核为AI辅助逐图视觉复核，主要依据接触表及必要细节卡；不等同于每张原图的原始分辨率检查，也不是两名独立人工标注。类别沿用作者定义，尚未独立机型鉴定。", "small")

    story.append(PageBreak())
    add("2 质量问题与统一筛选", "h1")
    add("排除协议在任何模型训练和分数产生前确定。原图2970个不同文件SHA-256，一组4张完全重复；1280×720有1731张，416×416有1242张。43445对感知近重复候选是关系数，不能当作重复图片数。")
    names={"wrong_target_box":"错框/错目标","crossfade":"转场叠影","postproduction_overlay":"后期字幕/动画叠加","misaligned_body_crop":"框偏移/主要机身漏裁","burned_in_detector_overlay":"烧录检测框/置信度","inset_intersects_crop":"画中画进入裁剪","promotional_title_card":"宣传标题卡域","exact_duplicate":"完全重复冗余","body_out_of_frame":"机身出画仅剩部件","rendered_target_graphics":"目标区域内渲染图形","body_not_identifiable_in_source_box":"框内机身不可辨认","unverifiable_tiny_target":"原生小框内容不能确认"}
    pairs=sorted(s["exclusion_primary_reasons"].items(),key=lambda v:-v[1])
    table(["互斥主排除原因","数量","互斥主排除原因","数量"],
          [[names[pairs[i][0]],pairs[i][1],names[pairs[i+1][0]],pairs[i+1][1]] for i in range(0,len(pairs),2)], [65,22,65,22])
    add("每图计一次主原因，合计253张。宣传型号标题卡按事先声明的场景范围排除，不能声称每张标题都进入目标裁剪。物体自带品牌标识与后期字幕分别对待。主体仍可辨认的小目标、模糊、逆光和局部截断继续保留。")
    image=Image(str(OUT/'figures/data_review_counts.png'), width=174*mm, height=90.5*mm)
    story.append(image)
    add("图1 公开数据逐类筛选。图片数不等于独立来源数。2720张只是视觉候选，不是正式清洗集。", "small")
    add("Phantom 3此前99项与YH此前10项未决，依据既有证据按统一协议转为排除，旧决定保留；不是额外109张审核。YH210原生13×9像素框仍不能确认，选框保持proposed并排除该图。", "small")

    story.append(PageBreak())
    add("3 来源相关性与自采集覆盖", "h1")
    add("2807张原图有12组保守相关画面约束，含排除项。观察约束只表达组内不能跨集合，不证明组间独立；不同分辨率、编号或场景不能自动成为视频/飞行ID。")
    table(["类别", "保留图的观察约束组", "未填写约束的保留图", "确认来源组"],
          [[r["class_name"],r["kept_observed_constraint_groups"],r["kept_without_correlation_constraint"],r["confirmed_source_groups_for_kept"]] for r in s["class_summary"]],[57,44,44,29])
    add("Bebop、Intel Aero、Mambo各只有一个观察到的保守相关组；不能随机拆连续帧来声称来源独立的训练/测试或K折。Phantom 3尚未映射的样本也不能默认独立。")
    story.append(Image(str(OUT/'figures/external_class_coverage.png'), width=174*mm, height=83.5*mm))
    add("图2 自编爬虫的真实候选覆盖。Commons三类各30张，HalfChrome同一测评Mambo三张。共66视觉暂留、20待定、7排除；Emax、Intel、YH尚无采集原图。", "small")
    add("采集保存页面/图片地址、作者、许可字段、时间和哈希；已记录404、无关跳转和robots 403等失败。Mambo三张同源相关，尚不能据此声称可靠域外泛化；原生配对飞行日志也未落实。", "small")

    story.append(PageBreak())
    add("4 文件核对、可复现性与剩余条件", "h1")
    table(["实际核对项目", "已核对数量", "结果"], [
        ["原图实际解码",integrity["unique_images_decoded"],"未发现解码/尺寸不一致"],
        ["逐图证据、原图与标注哈希",integrity["visually_recorded_unique_images_checked"],"未发现记录不一致"],
        ["相关性约束",integrity["correlation_constraint_images_checked"],"未发现证据不一致"],
        ["原始框选择记录",integrity["box_selection_records_checked"],"22 reviewed / 1 proposed"],
    ], [78,33,63])
    add("以上是实际文件和记录核对，0项不一致；不证明语义标签和来源独立，也不是软件测试或模型评估。没有运行本地软件测试、训练或评估，当前无需GPU。")
    add("4.1 接下来必须解决的条件", "h1")
    for t in [
        "追溯原始页面/视频，核实机型和真实来源，补充能支持组隔离的独立数据。来源未知如实记未知，不能生成flight_id。",
        "补齐自采集缺类，解决20项待定标签及最终目标框，核查同来源、裁剪衍生和跨集合重复。网页搜索词不能自动当标签。",
        "根据能够落实的数据确定研究范围和方法，来源准入后再正式清洗、来源隔离约7:3划分、训练折内预处理/调参、冻结和最终评估。",
        "至少四种不同原理模型需要实际训练与公平比较。数据未准入、模型未训练时不能填写性能、消融、最优方法或创新效果。",
    ]:
        add(t)
    add("4.2 源码与产物入口", "h1")
    for label,path in [
        ("采集与逐图决定过程","reports/data_collection_record.md"),
        ("全量视觉筛选CSV，非正式清洗集","data/metadata/drone_type_set_visual_cohort.csv"),
        ("最新就绪条件与阻碍","outputs/drone_type_set_data_readiness.json"),
        ("完整性核对原始报告","outputs/drone_type_set_review_integrity_2026-10-22.json"),
        ("分步骤命令与执行边界","docs/RUNNING.md"),
        ("四题答案及课程要求对应","reports/thinking_questions.md / scoring_map_v2.md"),
    ]:
        add(label+"："+path,"small")
    add("结论边界：公开图片视觉审核和数据阶段报告已完成；正式清洗集、独立划分、模型实验、完整结果论文及展示PPT尚未完成。", "small")

    output=ROOT/'output/pdf/data_audit_report.pdf'
    output.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=18*mm,bottomMargin=18*mm,
                          title="无人机机型图像数据构建与审核阶段报告",author="UAV course project")

    def frame(canvas,document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#c9d5dd"))
        canvas.line(18*mm,15*mm,192*mm,15*mm)
        canvas.setFillColor(colors.HexColor("#657784"));canvas.setFont(font,8)
        canvas.drawString(18*mm,10*mm,"数据阶段 / 2026-10-22 / 未含模型结果")
        canvas.drawRightString(192*mm,10*mm,str(document.page))
        canvas.restoreState()
    doc.build(story,onFirstPage=frame,onLaterPages=frame)
    print(output)
    return output
