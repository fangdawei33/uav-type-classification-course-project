"""03k. Export data-only figures and a report from actual release records."""
from collections import Counter
from _bootstrap import PROJECT_ROOT
from uav_project.common import ROOT, OUT, rows, read_json, resolve, relative, save_json, digest, now
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def export():
    data = ROOT/'data/final/uav_type_v3'
    records = rows(data/'clean_manifest.csv')
    counts = rows(data/'class_counts.csv')
    feasibility = rows(data/'split_feasibility.csv')
    summary = read_json(data/'data_summary.json')
    split = read_json(OUT/'expansion/processing_v3/split_draft/receipt.json')
    folder = OUT/'figures/expanded_data_v3'
    folder.mkdir(parents=True, exist_ok=True)
    names = [r['class_name'] for r in counts]
    values = [int(r['retained_images']) for r in counts]
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 180})
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    bars = ax.barh(names, values, color=['#2a7f78']*7+['#a23b55']*5)
    ax.bar_label(bars, padding=3)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values)*1.18)
    ax.set_xlabel('Retained target crops')
    ax.set_title('12 source-defined labels: class counts after recorded filtering')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(folder/'class_counts.png')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    bars = ax.barh(names, [int(r['conservative_constraint_groups']) for r in feasibility], color='#426b83')
    ax.bar_label(bars, padding=3)
    ax.invert_yaxis()
    ax.set_xlim(0, 5)
    ax.set_xticks(range(6))
    ax.set_xlabel('Conservative constraint groups (not authenticated flights)')
    ax.set_title('More images do not establish independent source coverage')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(folder/'constraint_groups.png')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,4), layout='constrained')
    sizes = np.array([min(int(r['width']), int(r['height'])) for r in records])
    bins = [0,16,24,32,48,64,96,128,256,512,1024,2048,4096]
    hist, _ = np.histogram(sizes, bins=bins)
    ax.bar(range(len(hist)), hist, color='#2a7f78')
    ax.set_xticks(range(len(hist)), [f'{bins[i]}-{bins[i+1]-1}' for i in range(len(hist))], rotation=45, ha='right')
    ax.set_ylabel('Crops')
    ax.set_xlabel('Native target short side (pixels)')
    ax.set_title('Native target resolution, before any model resizing')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(folder/'target_sizes.png')
    plt.close(fig)
    # A deterministic overview, never a claim that all images were visually read.
    sheet = Image.new('RGB', (1440, 1200), 'white')
    draw = ImageDraw.Draw(sheet)
    chosen = []
    for index, item in enumerate(counts):
        candidates = [r for r in records if r['label']==item['label']]
        candidates.sort(key=lambda r:(min(int(r['width']), int(r['height'])),r['sample_id']))
        r = candidates[len(candidates)//2]
        x,y = index%4*360,index//4*400
        with Image.open(resolve(r['image_path'])) as image:
            preview = ImageOps.contain(image,(330,300))
            sheet.paste(preview,(x+(360-preview.width)//2,y+(305-preview.height)//2))
        draw.text((x+8,y+315),r['class_name'],fill='black')
        draw.text((x+8,y+335),r['sample_id'],fill='black')
        draw.text((x+8,y+355),f"Native size: {r['width']} x {r['height']}",fill='black')
        chosen.append({'sample_id':r['sample_id'],'path':r['image_path'],'sha256':r['sha256']})
    sheet.save(folder/'class_examples.jpg',quality=95)
    table = '\n'.join('| '+r['class_name']+' | '+r['retained_images']+' | '+
        feasibility[i]['conservative_constraint_groups']+' |' for i,r in enumerate(counts))
    text = f'''# 十二类无人机数据处理报告

更新时间：{now()}。本报告仅描述实际完成的数据处理，不包含主任务模型结果。

## 数据与清洗

Drone-type-Set原七类保留2720张。VisioDECT作者版本1原始20924张已实际下载并解码；通用Phantom3300张与现有细分类重叠，单独隔离。目标框缺失、多框、文件完全重复和一次抽样无目标问题排除后，形成20018张候选。

本轮20018张全部导出原生尺寸RGB目标裁剪，逐张比对保存前后像素，旧七类同时核对原有裁剪一致性。没有上采样伪造原始分辨率，没有调用模型。

对全部候选进行质量数值统计，得到9张低对比度、1张偏暗的重点观察队列。实际查看这10张裁剪和原图对照页后，排除8张没有可辨目标的天空裁剪，保留2张困难样本；不因模糊、目标小或偏暗自动删除。当前保留{len(records)}张、12类。原图和既有v2未覆盖，排除裁剪仍保留在候选图目录，通过清单控制加载。

文件完全重复此前已隔离14张。本轮尺寸加RGB像素哈希没有发现新的完全重复或跨标签像素冲突。相似关系使用64位差分哈希距离不超过3和16x16灰度均值绝对差不超过6，两条件同时满足记录约束；保留510条关系，不能称为510张重复图或独立飞行证据。

## 各类数量

| 作者定义类别 | 保留裁剪 | 保守约束组 |
|---|---:|---:|
{table}

![图1 各类别保留数量](../outputs/figures/expanded_data_v3/class_counts.png)

图1：旧七类与新增五类分别着色。多数新增类数量显著高于旧类，后续训练和评价必须处理类别不平衡。

![图2 原生目标尺寸](../outputs/figures/expanded_data_v3/target_sizes.png)

图2：目标框短边分布。{summary['small_targets_short_side_below_24']}张短边小于24像素，作为尺度困难条件保留；模型缩放不能恢复不存在的细节。

![图3 保守约束组](../outputs/figures/expanded_data_v3/constraint_groups.png)

图3：约束组来自已有相关场景、未明来源的整类保守桶、作者类别/天气桶和相似关系合并。共{summary['constraint_groups']}组，不是已认证的独立视频数。

![图4 十二类原生裁剪示例](../outputs/figures/expanded_data_v3/class_examples.jpg)

图4：各类别按原生短边排序取中间位置的一张确定性示例，不是专家身份核实或全量视觉审核证明。

## 划分的实际可行性

Bebop、DJI phantom 3、Intel Aero RTF和Mambo各只有一个保守约束组。保持这些组不跨集合时，无法为全部12类同时获得训练与测试样本。因此没有生成宣称完整12类独立评测的活动训练协议，也没有随机拆散画面绕过约束。

已保存一个可审阅的诊断草案：开发{split['counts']['development']}张，留出{split['counts']['holdout']}张，实际留出比例{split['holdout_fraction']:.2%}。选择规则为每类在完整约束组的组合中尽量接近30%，只使用标签和数量，不使用模型成绩。全部12类保留在开发池，留出仅覆盖8类，4类仅开发；不能把它报告为完整12类成绩。该草案不是活动训练协议，未运行训练、评估或全12类交叉验证。天气桶也未认证为独立拍摄序列。

草案目录：outputs/expansion/processing_v3/split_draft/，包括manifest.csv、class_support.csv、receipt.json。data/final/uav_type_v3/clean_manifest.csv中的split仍为unassigned。

## 过程与局限

全部原图文件身份与目标裁剪有哈希追踪；复核决定写入带日期记录。VOC坐标原点约定尚未证实，新增裁剪沿用作者xyxy直接取整，不擅加1像素修正。类别名沿用作者目录，未独立鉴定Mavic具体商用子型。已实际完成144对固定抽样和本轮10项质量队列查看，不代表20010张逐图专家审核全部完成。

作者数据来自视频抽帧，相邻帧和同背景画面不等于独立样本。旧类与新类来源相关，不能混合随机拆分后声称跨域泛化。网页弱标签135张仍独立保存在v2，不能拿新增公共数据冒充自编爬虫采集。合成辅助实验独立，不把其指标移用到本任务。

## 文件与运行入口

- data/final/uav_type_v3/clean_manifest.csv：当前保留裁剪的唯一加载入口。
- candidate_inventory.csv：包含8张本轮排除裁剪在内的20018项派生记录。
- excluded_visual_quality.csv、excluded_pixel_duplicates.csv：排除原因，不递归把目录全部图片载入训练。
- files_sha256.csv、freeze.json、data_summary.json：数据快照和处理状态。
- scripts/04h_process_expanded_images.py：已完成裁剪及像素一致性核对，支持日志续跑。
- scripts/03j_prepare_expanded_quality_review.py与review_records/2026_10_22_expanded_quality.py：已查看队列及决定转录。
- scripts/04i_finalize_expanded_data.py：已完成像素去重、相似约束和快照。
- scripts/05c_prepare_expanded_split_draft.py：仅诊断草案，不启动训练。
- scripts/04j_audit_expanded_release.py：实际文件完整性核对。

以上是已完成命令，无需用户重跑。未进行本地模型推理、学习表征提取、训练、模型评价、软件测试或PDF生成。
'''
    report = ROOT/'reports/expanded_data_report.md'
    report.write_text(text,encoding='utf-8')
    save_json(folder/'export_receipt.json', {'created_at':now(),'report_sha256':digest(report),
        'figures':{p.name:digest(p) for p in sorted(folder.glob('*')) if p.suffix in {'.png','.jpg'}},
        'class_examples':chosen,'class_example_selection':'median native short-side rank, per label',
        'class_examples_visually_reviewed':False,'input_manifest_sha256':digest(data/'clean_manifest.csv'),
        'models_or_software_tests_run':False})
    print('Data report and four figures exported:', report, flush=True)


if __name__=='__main__':
    export()
