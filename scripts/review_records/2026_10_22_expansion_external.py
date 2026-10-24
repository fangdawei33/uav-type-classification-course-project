"""Transcribe the 63 observations actually viewed on 2026-10-22.

Default prepares six Mambo box proposals. Inspect them before --record-inspected.
This dated ledger never decides labels for future images or overwrites history.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from PIL import Image, ImageDraw, ImageOps
from uav_project.common import ROOT, OUT, META, rows, read_json, write_rows, save_json, digest, relative, resolve, now, protect_frozen

FOLDER = OUT / "review_batches/external_expansion_2026-10-22"
SNAPSHOT = OUT / "drone_type_external_visual_review_expansion_2026-10-22.json"
# Each index is the explicit position in the frozen, viewed queue.json (1-based).
# keep is a visual candidate decision, never final identity/source approval.
OBSERVATIONS = [
 ("pending", "hull_occlusion", "侧视泡沫护罩占据主体，中央机体辨识受限；来源称Bebop，暂不独立确认机型。"),
 ("candidate_keep", "indoor_hull|underside", "已补看原图，护罩下的中央机体、四臂与电机可见；保留候选，具体身份待核。"),
 ("candidate_keep", "rear_view", "公园道路飞行后视，中央机身及四臂可见，保留自然拍摄背景。"),
 ("candidate_keep", "front_view", "红色机体正面及相机可辨，前景手臂不作为目标。"),
 ("candidate_keep", "studio_background|reflection", "黄色机体正面实物与台面反射；不是把反射当第二个独立样本。"),
 ("candidate_keep", "studio_background|reflection", "黄色机体斜前视实物与反射，和前图施加同组约束。"),
 ("pending", "multiple_targets|small_target", "展台画面有多架小目标，须先确定单个目标及原生裁剪可辨性。"),
 ("candidate_keep", "multiple_targets|foreground_target", "蓝色前景机体可辨，背景还有其他机体，最终框只取前景。"),
 ("candidate_keep", "small_target|landscape", "山林背景的小型黄色飞行机体可见；不能因场景较难自动删除。"),
 ("candidate_keep", "side_view", "黄色机体侧前视可见，与同场景连拍约束同组。"),
 ("candidate_keep", "side_view", "黄色机体与天空背景，机体仍可辨，保留困难尺度。"),
 ("candidate_keep", "low_angle", "黄色机体低角度飞行视图，四臂可见。"),
 ("candidate_keep", "low_angle|near_duplicate_lead", "黄色机体与上一图姿态接近，保守同组，尚非已证明重复。"),
 ("candidate_keep", "low_angle|near_duplicate_lead", "黄色机体低角度近邻构图，同组防止跨集合拆分。"),
 ("pending", "silhouette|exact_model_unverified", "已补看原图，逆光Phantom轮廓可辨，但来源只写Phantom，Phantom 3子类仍待核实。"),
 ("candidate_keep", "ground_view", "白色机体、起落架与云台可见，草土地面场景。"),
 ("candidate_keep", "small_target|human_present", "人物上方白色飞行机体可见，须单独框目标，人物不作标签依据。"),
 ("candidate_keep", "banked_flight", "倾斜飞行机体、云台与机臂可见，保留姿态变化。"),
 ("candidate_keep", "underside|motion_blur", "仰视飞行机体，桨叶运动模糊是自然条件。"),
 ("candidate_keep", "flight_view", "天空背景飞行机体及云台可见，和同摄影序列同组。"),
 ("candidate_keep", "human_present|flight_view", "人物手部附近机体可见，人与无人机同景，目标区域仍待最终框。"),
 ("pending", "unboxing|propellers_unmounted", "机体未装桨，旁有遥控器、平板和配件；与现有开箱图一致，场景准入待定。"),
 ("candidate_keep", "ground_view|clutter", "机体位于桌面杂物间，中央机身与云台可辨。"),
 ("candidate_keep", "handheld|propellers_present", "原图可见桨叶、中央机身与云台；手持不等于未装桨，不按缩略图误判。"),
 ("candidate_keep", "propellers_present", "原图可见薄桨叶及整机轮廓，保留背景与尺度条件。"),
 ("candidate_keep", "handheld|intrinsic_nameplate", "原图可见安装桨叶与机身自带铭牌；非后期添加答案字幕。"),
 ("candidate_keep", "ground_view", "草地上的整机可辨，与同日同作者桌面图施加保守约束。"),
 ("candidate_keep", "exhibition", "展会实拍白色机体与云台可见，复杂人群背景不自动排除。"),
 ("candidate_keep", "low_light", "黄昏林地起飞前景，灯光与机体仍可辨，保留低曝光条件。"),
 ("candidate_keep", "flight_view", "白色机体飞行于建筑及树前，中心结构可辨。"),
 ("candidate_keep", "small_target", "较远飞行机体仍可见，最终按原生目标框核查尺度。"),
 ("candidate_keep", "ground_view", "修复后草地机体照片，机身及云台可见。"),
 ("candidate_keep", "flight_view", "修复试飞机体正面，与同场景试飞照片同组。"),
 ("candidate_keep", "flight_view|near_duplicate_lead", "相近时间试飞机体，相似背景与构图需保守同组。"),
 ("candidate_keep", "underside", "同次试飞仰视图，正常机体与旋转桨叶可见。"),
 ("pending", "unboxing|propellers_unmounted", "机体在包装盒内、桨叶未安装，保留原图，运行状态范围待决。"),
 ("pending", "disassembled|open_shell", "机壳打开，电路板与布线暴露；不当作正常运行完整机体。"),
 ("pending", "equipment_layout|propellers_unmounted", "套装平铺含未装桨机体及大量配件，场景范围待定。"),
 ("candidate_keep", "flight_view|sunlight", "自然光下飞行机体与云台可见，保留原始场景。"),
 ("candidate_keep", "flight_view|mountains", "山景背景机体可辨，摄影作品署名不是独立机型鉴定。"),
 ("candidate_keep", "flight_view|stone_wall", "石墙背景机体可见，与同摄影作者作品保守相关约束。"),
 ("candidate_keep", "flight_view|high_dynamic_range", "强反差山景中的机体可见，保留困难光照。"),
 ("candidate_keep", "flight_view|forest", "树林背景中的白色机体与云台可见。"),
 ("candidate_keep", "underside", "来源标作Phantom 3 4K，仰视机体可见；沿用来源类，未独立认证。"),
 ("candidate_keep", "flight_view|background_watermark", "机体可辨，摄影署名位于背景，最终框不包含右下角署名。"),
 ("candidate_keep", "dark_background|underside", "暗背景中的仰视机体可见，来源4K名称不自动解释成Phantom 4。"),
 ("candidate_keep", "propeller_guards|flight_view", "海滩飞行机体带护桨，保留配件变化并标记。"),
 ("pending", "exact_subtype_unverified", "已补看原图，Phantom 4系列机体可见；来源正文只写Phantom，Pro具体子型待核。"),
 ("candidate_keep", "flight_view|exact_identity_not_authenticated", "完整Phantom 4系列候选及云台可见，保留来源候选类，仍不独立认证Pro。"),
 ("pending", "small_target|carried_target|exact_subtype_unverified", "人物携带的机体较小且局部遮挡，目标裁剪与Pro子型证据不足。"),
 ("candidate_keep", "small_target|same_scene", "桥边飞行机体较小但轮廓可见，与其他同作者同日桥景同组。"),
 ("candidate_keep", "small_target|same_scene|near_duplicate_lead", "同一桥景近邻图，机体可见，不能计作独立来源。"),
 ("candidate_keep", "ground_view|same_scene", "石地机体及云台可辨，与同一桥边拍摄系列同组。"),
 ("candidate_keep", "flight_view|same_scene", "树冠及建筑尖塔前的飞行机体可辨，与桥景同组。"),
 ("candidate_keep", "flight_view|same_scene", "同场景低角度机体可见，外观身份仍待最终认证。"),
 ("pending", "small_target|ground_target|exact_subtype_unverified", "人物及车辆为画面主体，地面无人机较小，最终框及Pro子型待核。"),
 ("candidate_keep", "accessories_outside_target|natural_truncation", "原图左上方Mambo机身可辨，旁有发射器和弹丸；框只选机体，原生截断保留。"),
 ("candidate_keep", "closeup|natural_truncation|intrinsic_brand", "原图中央机身近景及乐高式模块可见，品牌是实物自带，非后期字幕。"),
 ("candidate_keep", "top_view|natural_truncation", "原图顶视中央机身与四臂可辨，个别护罩接近图边。"),
 ("candidate_keep", "underside_closeup|natural_truncation", "原图底面机身、传感器和配件可见；不是单独脱落零件照片。"),
 ("candidate_keep", "attached_accessory|natural_truncation", "原图中央机体与安装附件可辨，护罩局部自然截断。"),
 ("reject", "no_identifiable_target", "原图只见人物、手机和沙发，没有可确认的无人机目标；网页alt说有无人机不能代替图像证据。"),
 ("candidate_keep", "top_view", "原图整机俯视可辨，机体与护罩完整范围待框核。"),
]
GROUPS = [
 ([1,2], "ext_bebop_pesce_20150106", "同原作者、同日展会画面，与既有Pesce组保守合并。"),
 ([5,6], "ext_bebop_weiwei_20141220_studio", "同作者同日黑背景台面及反射，保守相关，不声称独立拍摄。"),
 ([7,8], "ext_bebop_ifa_20150904", "同作者同日IFA展台，画面展台背景一致。"),
 ([9,10,11,12,13,14], "ext_bebop_weiwei_20141215", "同作者同日黄色Bebop与山林/天空连拍，延用既有同日保守组。"),
 ([16,17,18,19,20,21], "ext_p3_zimin_20150809", "同作者同日同场活动照片及机体，组内不得跨划分。"),
 ([23,27], "ext_p3_andres_20160920", "同作者同日同地点的机体，保守同组，尚未断言是近重复。"),
 ([24,25,26], "ext_p3_bidgee_20150919", "同作者同日Lake Albert相似机体及场景，同组约束。"),
 ([32,33,34,35], "ext_p3_superbass_20160605", "同作者同日修复试飞系列，有直接机体和场景连续性。"),
 ([40,41,42,43], "ext_p3_jasonblackeye_greece_cohort", "同摄影作者希腊作品的宽松保守分组；日期不同，不声称同一飞行。"),
 ([50,56], "ext_p4_saber_guardian_20230530", "同日同活动人物及地面场景，与既有同活动相关组保守合并。"),
 ([51,52,53,54,55], "ext_p4_raboe_20170519_schwerin", "同作者同日Schwerin桥边拍摄系列，组内不得拆分。"),
 ([57,58,59,60,61,62,63], "ext_trustedreviews_mambo_article", "同一原始测评文章，多张同一木质桌面；不是七次独立采样。"),
]
BOXES = {57:(0,0,575,285),58:(0,0,820,461),59:(165,0,680,461),60:(0,0,820,461),61:(0,0,820,461),63:(215,10,685,430)}
ORIGINAL_VIEWED = {1,2,15,24,25,26,48,57,58,59,60,61,62,63}


def inputs():
    queue = read_json(FOLDER / "queue.json")
    if len(queue) != 63 or len(OBSERVATIONS) != 63:
        raise ValueError("This dated transcription covers exactly 63 viewed candidates")
    raw = {r["sample_id"]:r for r in rows(META/"drone_type_external_raw.csv") + rows(META/"drone_type_article_raw.csv")}
    for r in queue:
        if r["sha256"] != digest(resolve(raw[r["sample_id"]]["path"])):
            raise ValueError("Viewed input changed")
    return queue, raw


def prepare():
    protect_frozen()
    queue, raw = inputs()
    destination = FOLDER/"mambo_boxes"
    destination.mkdir(exist_ok=True)
    manifest = FOLDER/"box_proposals.csv"
    if manifest.exists():
        raise FileExistsError("Dated proposal manifest already exists")
    cards = []
    for index, box in BOXES.items():
        r = raw[queue[index-1]["sample_id"]]
        with Image.open(resolve(r["path"])) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
        x1,y1,x2,y2=box
        if not 0<=x1<x2<=rgb.width or not 0<=y1<y2<=rgb.height:
            raise ValueError("Invalid proposed bounds")
        marked=rgb.copy()
        ImageDraw.Draw(marked).rectangle((x1,y1,x2-1,y2-1),outline="red",width=3)
        canvas=Image.new("RGB",(1200,430),"white")
        for offset, image in ((0,marked),(600,rgb.crop(box))):
            thumb=ImageOps.contain(image,(590,390));canvas.paste(thumb,(offset+(600-thumb.width)//2,25+(390-thumb.height)//2))
        ImageDraw.Draw(canvas).text((8,4),r["sample_id"]+" | PROPOSED | source / target crop",fill="black")
        path=destination/(r["sample_id"]+".jpg")
        if path.exists():raise FileExistsError(path)
        canvas.save(path,quality=95)
        cards.append({"sample_id":r["sample_id"],"class_id":r["candidate_label"],"x_min":x1,"y_min":y1,
            "x_max_exclusive":x2,"y_max_exclusive":y2,"coordinate_space":"EXIF-transposed RGB display pixels",
            "image_sha256":r["sha256"],"status":"proposed_not_final","reviewer":"Codex_AI_assisted_visual_review",
            "evidence":relative(path),"evidence_sha256":digest(path)})
    write_rows(manifest,cards)
    for offset in (0,3):
        sheet=Image.new("RGB",(1200,1290),"white")
        for i,r in enumerate(cards[offset:offset+3]):
            with Image.open(resolve(r["evidence"])) as im:sheet.paste(im,(0,i*430))
        sheet.save(destination/f"proposals_{offset//3+1}.jpg",quality=95)
    print("Six proposed target regions prepared; no observations imported.")


def record():
    protect_frozen()
    if SNAPSHOT.exists():raise FileExistsError("Dated observations already recorded")
    queue,raw=inputs()
    current=rows(META/"drone_type_external_review.csv")
    by_id={r["sample_id"]:r for r in current}
    groups=rows(META/"drone_type_external_reviewed_groups.csv")
    proposals=rows(META/"drone_type_external_box_proposals.csv")
    new_boxes=rows(FOLDER/"box_proposals.csv")
    group_by_index={i:(group,basis) for indices,group,basis in GROUPS for i in indices}
    updates=[];records=[]
    for index,(q,observation) in enumerate(zip(queue,OBSERVATIONS),1):
        sid=q["sample_id"];r=raw[sid];decision,flags,note=observation
        if by_id[sid]["visual_decision"]!="pending" or by_id[sid].get("evidence"):
            raise ValueError("Refuse to replace an existing observed decision")
        group,basis=group_by_index.get(index,("",""))
        # Use the frozen, actually viewed queue sheet, not a subsequently regenerated page.
        evidence=q["queue_sheet"]
        updated={**by_id[sid],"visual_decision":decision,"label_status":"source_supported_candidate_not_independently_authenticated",
            "box_status":"proposed_not_final" if index in BOXES else "pending", "source_group":group,
            "source_group_status":"reviewed_correlation_constraint" if group else "unverified",
            "flags":flags,"note":note,"reviewer":"Codex_AI_assisted_visual_review",
            "evidence":evidence+"; position="+str(q["queue_position"])}
        updates.append(updated)
        records.append({**updated,"sheet":evidence,"sheet_position":q["queue_position"],
            "evidence_sha256":digest(resolve(evidence)),"image_sha256":r["sha256"],
            "original_image_inspected":index in ORIGINAL_VIEWED,"contact_sheet_inspected":True,
            "inspection_resolution":"original_file_viewed_may_be_display_downscaled" if index in ORIGINAL_VIEWED else "620x300_thumbnail_on_viewed_contact_sheet",
            "detail_snapshot":r["detail_snapshot"],"detail_sha256":r["detail_sha256"]})
        if group:
            groups.append({"sample_id":sid,"source_group":group,"status":"reviewed_correlation_constraint","basis":basis,
                "evidence":evidence,"source_url":r["source_url"],"detail_snapshot":r["detail_snapshot"],
                "independent_from_public":"unverified","original_flight_id":""})
    for name,data in (("review",current),("groups",rows(META/"drone_type_external_reviewed_groups.csv")),
                      ("boxes",rows(META/"drone_type_external_box_proposals.csv"))):
        p=FOLDER/("before_"+name+".csv")
        if p.exists():raise FileExistsError(p)
        write_rows(p,data)
    replacements={r["sample_id"]:r for r in updates}
    write_rows(META/"drone_type_external_review.csv",[replacements.get(r["sample_id"],r) for r in current])
    write_rows(META/"drone_type_external_reviewed_groups.csv",groups)
    write_rows(META/"drone_type_external_box_proposals.csv",proposals+new_boxes)
    write_rows(FOLDER/"updates.csv",updates)
    save_json(SNAPSHOT,{"recorded_at":now(),"review_type":"AI-assisted contact-sheet screening with 14 selected original-file views",
        "queue_sha256":digest(FOLDER/"queue.json"),"records":records,"new_box_proposals":new_boxes,
        "all_63_originals_individually_viewed":False,"original_files_individually_viewed_count":len(ORIGINAL_VIEWED),
        "all_63_contact_sheet_thumbnails_actually_viewed":True,"final_labels_or_boxes_approved":False,
        "source_independence_confirmed":False,"software_tests_run":False,"training_run":False})
    print("Recorded 63 observed candidates; six new proposed boxes; final source/identity approval still pending.")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-inspected",action="store_true")
    args=parser.parse_args()
    record() if args.record_inspected else prepare()
