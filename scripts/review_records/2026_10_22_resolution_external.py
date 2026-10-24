"""Review ledger for 32 pending acquired images; preparation does not approve data.

Run without flags to generate immutable box cards, inspect them, then explicitly
transcribe the recorded observations with --record-inspected. No models run.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from PIL import Image, ImageOps, ImageDraw
from uav_project.common import META, OUT, rows, write_rows, resolve, relative, digest, save_json, now, protect_frozen

FOLDER = OUT / 'review_batches/external_resolution_2026-10-22'
SNAPSHOT = OUT / 'drone_type_external_visual_review_resolution_2026-10-22.json'
# One intended target per original; normalized EXIF-transposed display coordinates.
# Pending labels stay pending; geometry never establishes a drone subtype.
ITEMS = [
(1, (.29,.52,1,1), 'candidate_keep', 'source_description_supported_not_authenticated', 'natural_truncation', '目标是右下黄色Bebop；遥控器不作目标。主体可辨，保留自然截断。'),
(2, (.65,.27,1,.78), 'candidate_keep', 'source_description_supported_not_authenticated', 'natural_truncation|defocus', '目标为右后方黄色Bebop，前景黄色设备是遥控器，不能用整图代替目标。'),
(3, (.62,.48,1,1), 'candidate_keep', 'source_description_supported_not_authenticated', 'natural_truncation', '右下Bebop机身可辨，桨臂自然出画；来源明确第一代Bebop。'),
(4, (.52,.32,.70,.47), 'pending', 'exact_generation_unresolved', 'small_target|date_identity_conflict', '原文只支持DJI，2014记录与候选Phantom3存在冲突；不能按目录确认代际。'),
(5, (.02,.23,1,.90), 'pending', 'exact_generation_unresolved', 'damaged|challenge_condition', '整机严重烧损但可见；保留原始证据及损伤挑战标记，确切代际未证实。'),
(6, (.035,.055,.98,.95), 'pending', 'exact_generation_unresolved', 'damaged|challenge_condition', '损伤整机仍可见，来源只称MEHEL靶机，不能擅定Phantom3。'),
(7, (.56,.02,.95,.44), 'pending', 'background_target_identity_unresolved', 'multiple_targets|defocus|partial_occlusion', '候选是右后方白色机，前景灰色折叠机排除在选择之外；后景代际仍不明。'),
(8, (0,0,.49,1), 'pending', 'partial_target_identity_unresolved', 'natural_truncation|equipment_layout', '左侧白色局部机身可见，不能用相机/手机/镜头替代机体，代际未核实。'),
(9, (.18,.47,.38,.70), 'pending', 'exhibition_instance_identity_unresolved', 'multiple_targets|small_target|mesh_occlusion', '展示网内多架机；框选右侧白色候选，型号与实例对应仍不充分。'),
(10, (.015,0,.80,.63), 'candidate_keep', 'comparison_title_supported_not_authenticated', 'multiple_targets|partial_occlusion|propellers_unmounted|challenge_condition', '白色Phantom是候选目标，黑色Mavic不是；框内仍有不可消除的干扰，单独标记挑战条件。'),
(11, (.33,0,1,1), 'candidate_keep', 'comparison_title_supported_not_authenticated', 'multiple_targets|partial_occlusion|propellers_unmounted|challenge_condition', '右侧白色Phantom为候选；遮挡的黑色机部分仍进框，不能声称干净单目标。'),
(12, (.32,0,1,.85), 'candidate_keep', 'comparison_title_supported_not_authenticated', 'multiple_targets|partial_occlusion|propellers_unmounted|challenge_condition', '后景白色Phantom可辨，前景Mavic遮挡；裁剪保持真实遮挡。'),
(13, (.36,0,1,.91), 'candidate_keep', 'comparison_title_supported_not_authenticated', 'multiple_targets|partial_occlusion|natural_truncation|propellers_unmounted|challenge_condition', '右后白色Phantom机身可辨，右臂出画，Mavic干扰保留并标记。'),
(14, (.22,.35,.82,.81), 'pending', 'phantom4_pro_subtype_unresolved', 'handheld|partial_occlusion', '白色机体可见；正文只称Phantom4，目录Pro不能替代子型证据。'),
(15, (.01,.37,1,.98), 'pending', 'description_date_conflict', 'date_identity_conflict', '图像可用，Professional说明与2016年5月记录冲突；不据单项信息自动改标签。'),
(16, (.52,.41,.975,.70), 'pending', 'comparison_instance_assignment_unresolved', 'multiple_targets|partial_occlusion', '框选右侧实例，但这张配对照片缺乏可核对的明确实例型号说明。'),
(17, (.515,.31,.98,.71), 'candidate_keep', 'explicit_extracted_child_supports_right_instance', 'multiple_targets|possible_shared_parent', '右侧实例与已登记WM331A Phantom4Pro提取子图对应；父图和子图须保持同一相关组。'),
(18, (.20,.80,.55,1), 'pending', 'phantom4_pro_subtype_unresolved', 'natural_truncation|partial_occlusion', '底边手持机体仍可辨；保留截断条件，具体Pro子型未核实。'),
(19, (.400,.397,.435,.432), 'pending', 'small_target_exact_subtype_unresolved', 'small_target', '天空白色四旋翼可定位；原生小目标不足以据此认证Pro，不因小而删除。'),
(20, (.14,.235,.825,.81), 'pending', 'phantom4_pro_subtype_unresolved', 'in_flight', '飞行机体和传感区域可见；来源只给地点，不能认证Pro。'),
(21, (0,.16,.98,.65), 'pending', 'hull_occlusion_identity_unresolved', 'protective_hull|partial_occlusion', '护罩遮住主要机身；来源虽写Bebop，但可见外形不足以独立核实，保留待定。'),
(22, (.56,.322,.90,.39), 'candidate_keep', 'source_title_supported_not_authenticated', 'small_target|multiple_targets', '框选展台右上黄色飞行目标；避开后期/背景展台型号文字，其他目标不并作新样本。'),
(23, (.20,.14,.405,.93), 'pending', 'exact_generation_unresolved', 'backlit', '逆光机体可辨，不以低照度排除；来源仅Phantom，Phantom3代际仍未证实。'),
(24, (.015,.02,.54,.69), 'candidate_keep', 'source_title_supported_not_authenticated', 'unboxing|propellers_unmounted|challenge_condition', '左侧Phantom3整机可见，未装桨是独立场景标签，不混称飞行场景。'),
(25, (.035,.11,.955,.965), 'candidate_keep', 'source_title_supported_not_authenticated', 'unboxing|propellers_unmounted|challenge_condition', '盒内Phantom3主机体可辨；保留开箱条件与包装背景。'),
(26, (.115,.025,.915,.98), 'candidate_keep', 'source_title_supported_not_authenticated', 'disassembled|challenge_condition', '拆壳机体有明确Phantom3来源；仅作拆解挑战候选，不混入正常整机主评测。'),
(27, (.015,.075,.345,.63), 'candidate_keep', 'source_title_supported_not_authenticated', 'unboxing|propellers_unmounted|challenge_condition', '选择左侧未装桨Phantom3，充电器/遥控器/备用桨不作为目标。'),
(28, (.105,.14,.88,.88), 'pending', 'phantom4_pro_subtype_unresolved', 'motion_blur', '地面机体可辨并保留旋翼运动模糊；原文只支持Phantom家族。'),
(29, (.57,.43,.865,.57), 'pending', 'phantom4_pro_subtype_unresolved', 'small_target|partial_occlusion|handheld', '后方人员手持机体可见；不能从目录确认Pro子型。'),
(30, (.45,.90,.585,1), 'pending', 'phantom4_pro_subtype_unresolved', 'small_target|natural_truncation', '底边地面白色机体可辨；保留小目标截断，不把人员或车辆当目标。'),
(31, (.025,0,.995,1), 'candidate_keep', 'article_identity_supported_dataset_class_mapping_pending', 'disassembled|challenge_condition', 'EZPilot开壳机体保留为拆解挑战候选；原版EZPilot与作者Emax类别对应仍待核实。'),
(32, (.27,.35,.90,.925), 'candidate_keep', 'official_rtf_context_supported_not_authenticated', 'unboxing|propellers_unmounted|challenge_condition', 'Intel官方开箱照片中央碳臂整机为目标，遥控器不是目标但矩形框仍含部分遥控器；只作为开箱挑战候选。'),
]

def prepare():
    protect_frozen()
    inputs=rows(FOLDER/'pending_manifest.csv')
    cards=[]
    for number,bounds,*_ in ITEMS:
        r=inputs[number-1]
        rgb=ImageOps.exif_transpose(Image.open(resolve(r['path']))).convert('RGB')
        box=tuple(round(v*(rgb.width if j%2==0 else rgb.height)) for j,v in enumerate(bounds))
        x1,y1,x2,y2=box
        if not (0<=x1<x2<=rgb.width and 0<=y1<y2<=rgb.height): raise ValueError((r['sample_id'],box))
        marked=rgb.copy(); ImageDraw.Draw(marked).rectangle((x1,y1,x2-1,y2-1),outline='red',width=max(2,rgb.width//300))
        card=Image.new('RGB',(1200,460),'white')
        for off,im in [(0,marked),(600,rgb.crop(box))]:
            thumb=ImageOps.contain(im,(590,400));card.paste(thumb,(off+(600-thumb.width)//2,30+(400-thumb.height)//2))
        d=ImageDraw.Draw(card); d.text((5,6),f"{number} | {r['sample_id']} | source / proposed target",fill='black')
        d.text((5,435),f'{rgb.size} exclusive xyxy={box}; geometry review only',fill='black')
        path=FOLDER/(r['sample_id']+'_box.jpg')
        if path.exists(): raise FileExistsError(path)
        card.save(path,quality=95)
        cards.append({'sample_id':r['sample_id'],'class_id':r['candidate_label'],
            'x_min':x1,'y_min':y1,'x_max_exclusive':x2,'y_max_exclusive':y2,
            'coordinate_space':'EXIF-transposed RGB display pixels','image_sha256':r['sha256'],
            'status':'proposed_not_final','reviewer':'Codex_AI_assisted_visual_review',
            'evidence':relative(path),'evidence_sha256':digest(path)})
    write_rows(FOLDER/'new_box_cards.csv',cards)
    allcards=rows(META/'drone_type_external_box_proposals.csv')+cards
    for start in range(0,len(allcards),4):
        sheet=Image.new('RGB',(1200,1840),'white')
        for j,r in enumerate(allcards[start:start+4]):
            sheet.paste(Image.open(resolve(r['evidence'])),(0,j*460))
        path=FOLDER/f'box_sheet_{start//4+1:02}.jpg'
        if path.exists(): raise FileExistsError(path)
        sheet.save(path,quality=94)
    print('Prepared 32 new cards and 12 sheets including existing16; no records approved.')

REVISIONS = {
2:(.60,.27,1,.78), 3:(.57,.46,1,1), 5:(0,.23,1,.95), 6:(.025,.055,.99,.99),
8:(0,0,.52,1), 9:(.60,.445,.75,.625), 16:(.48,.34,.99,.65),
19:(.395,.397,.44,.445), 22:(.50,.322,.90,.40), 25:(.035,.07,.955,.965),
26:(.05,.025,.97,.99), 27:(.015,.075,.30,.53), 28:(.08,.12,.90,.90),
31:(.025,0,1,1), 32:(.27,.31,.91,.985)}
OLD_REVISIONS = {
'dta_4feae19ca5e40e4d044f':(73,19,1024,558),
'dta_c301d4045fec684aa774':(359,500,3926,2013),
'dta_13c0460a5844cdf1779e':(215,10,685,445)}

def prepare_revisions():
    protect_frozen()
    raw={r['sample_id']:r for name in ('drone_type_external_raw.csv','drone_type_article_raw.csv') for r in rows(META/name)}
    numbered={int(r['number']):r['sample_id'] for r in rows(FOLDER/'pending_manifest.csv')}
    base=rows(META/'drone_type_external_box_proposals.csv')+rows(FOLDER/'new_box_cards.csv')
    changes={numbered[n]:b for n,b in REVISIONS.items()};changes.update(OLD_REVISIONS)
    revised=[];history=[]
    for previous in base:
        sid=previous['sample_id']
        if sid not in changes:continue
        r=raw[sid];rgb=ImageOps.exif_transpose(Image.open(resolve(r['path']))).convert('RGB')
        b=changes[sid]
        box=tuple(round(v*(rgb.width if j%2==0 else rgb.height)) for j,v in enumerate(b)) if max(b)<=1 else b
        x1,y1,x2,y2=box
        if not (0<=x1<x2<=rgb.width and 0<=y1<y2<=rgb.height):raise ValueError(box)
        marked=rgb.copy();ImageDraw.Draw(marked).rectangle((x1,y1,x2-1,y2-1),outline='red',width=max(2,rgb.width//300))
        card=Image.new('RGB',(1200,460),'white')
        for off,im in [(0,marked),(600,rgb.crop(box))]:
            thumb=ImageOps.contain(im,(590,400));card.paste(thumb,(off+(600-thumb.width)//2,30+(400-thumb.height)//2))
        d=ImageDraw.Draw(card);d.text((5,5),sid+' | REVISED geometry | source / crop',fill='black')
        d.text((5,435),str(box)+' exclusive xyxy; previous card retained',fill='black')
        path=FOLDER/(sid+'_box_v2.jpg')
        if path.exists():raise FileExistsError(path)
        card.save(path,quality=95)
        revised.append({**previous,'x_min':x1,'y_min':y1,'x_max_exclusive':x2,'y_max_exclusive':y2,
            'evidence':relative(path),'evidence_sha256':digest(path)})
        history.append({'sample_id':sid,'previous':previous,'revised':revised[-1],
            'reason':'Visual inspection: expand to retain visible extremities / correct intended white target / tighten equipment background; no generated pixels.'})
    write_rows(FOLDER/'box_revisions.csv',revised)
    save_json(FOLDER/'box_revisions.json',{'recorded_at':now(),'records':history})
    for start in range(0,len(revised),3):
        sheet=Image.new('RGB',(1200,1380),'white')
        for j,r in enumerate(revised[start:start+3]):sheet.paste(Image.open(resolve(r['evidence'])),(0,j*460))
        path=FOLDER/f'revised_sheet_{start//3+1:02}.jpg'
        if path.exists():raise FileExistsError(path)
        sheet.save(path,quality=94)
    print('Prepared',len(revised),'revised cards; previous evidence preserved.')


def record():
    protect_frozen()
    if SNAPSHOT.exists():raise FileExistsError(SNAPSHOT)
    inputs=rows(FOLDER/'pending_manifest.csv'); current=rows(META/'drone_type_external_review.csv')
    byid={r['sample_id']:r for r in current};oldboxes=rows(META/'drone_type_external_box_proposals.csv')
    revisions={r['sample_id']:r for r in rows(FOLDER/'box_revisions.csv')}
    oldboxes=[revisions.get(r['sample_id'],r) for r in oldboxes]
    newboxes=[revisions.get(r['sample_id'],r) for r in rows(FOLDER/'new_box_cards.csv')]; cards={r['sample_id']:r for r in newboxes}
    for name,data in [('review',current),('boxes',rows(META/'drone_type_external_box_proposals.csv'))]:
        path=FOLDER/f'before_{name}.csv'
        if path.exists():raise FileExistsError(path)
        write_rows(path,data)
    observations=[];scope=[]
    for number,_,decision,label_status,flags,note in ITEMS:
        r=inputs[number-1];sid=r['sample_id']; card=cards[sid]
        u={**byid[sid],'visual_decision':decision,'label_status':label_status,
            'box_status':'reviewed_geometry_only','flags':flags,'note':note,
            'reviewer':'Codex_AI_assisted_visual_review','evidence':card['evidence']}
        byid[sid]=u
        observations.append({**u,'sheet':card['evidence'],'sheet_position':'1',
            'evidence_sha256':card['evidence_sha256'],'image_sha256':r['sha256'],
            'detail_snapshot':r['detail_snapshot'],'detail_sha256':r['detail_sha256'],
            'contact_sheet_inspected':True,'target_box_card_inspected':True,
            'original_image_inspected':number==1,
            'inspection_resolution':'full original and target crop displayed in review card; display may downscale; no upscaling invents detail',
            'final_label_authenticated':False})
        scope.append({'sample_id':sid,'condition_scope':'challenge_only' if 'challenge_condition' in flags else 'standard_candidate',
            'label_status':label_status,'flags':flags,'training_approved':False,'test_approved':False,
            'reason':note,'evidence':card['evidence'],'evidence_sha256':card['evidence_sha256']})
    # Existing16 cards are inspected together; geometry review is not label approval.
    for p in oldboxes+newboxes:p['status']='reviewed_geometry_only'
    for p in oldboxes:
        byid[p['sample_id']]['box_status']='reviewed_geometry_only'
    write_rows(META/'drone_type_external_review.csv',[byid[r['sample_id']] for r in current])
    write_rows(META/'drone_type_external_box_proposals.csv',oldboxes+newboxes)
    write_rows(META/'drone_type_external_condition_scope.csv',scope)
    write_rows(FOLDER/'review_updates.csv',[byid[r['sample_id']] for r in inputs])
    save_json(SNAPSHOT,{'recorded_at':now(),'review_type':'AI-assisted visual review; not human expert authentication',
        'records':observations,'existing_box_ids_geometry_reviewed':[p['sample_id'] for p in oldboxes],
        'geometry_status':'reviewed_geometry_only','final_labels_or_boxes_approved':False,
        'source_independence_confirmed':False,'software_tests_run':False,'training_or_model_evaluation_run':False})
    print('Recorded32 observations and48 geometry reviews; original data preserved.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--record-inspected',action='store_true');parser.add_argument('--prepare-revisions',action='store_true')
    args=parser.parse_args()
    record() if args.record_inspected else prepare_revisions() if args.prepare_revisions else prepare()
