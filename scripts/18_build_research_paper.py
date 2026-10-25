"""Build the research manuscript, preserving course evidence in appendices.

Reads previously measured results. Does not train, infer, tune or alter the split.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import textwrap

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import pandas as pd
from uav_project.research_math import LATEX
from uav_project.paper_document import render_paper

RUN=ROOT/"outputs/imported_results/uav12_a3479ced386e1dea/outputs/experiments/uav_type_v3_v1"
REPO="https://github.com/fangdawei33/uav-type-classification-course-project"


def read(path):return json.loads(path.read_text(encoding="utf-8-sig"))
def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]+["| "+" | ".join(map(str,row))+" |" for row in rows])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--refresh-figures",action="store_true");args=parser.parse_args()
    if args.refresh_figures:
        from uav_project.research_figures import build_research_figures
        build_research_figures(ROOT,RUN)
    if not (ROOT/"reports/research_figures/figure_receipt.json").is_file():raise FileNotFoundError("Run with --refresh-figures first")
    course=(ROOT/"reports/final_report.md").read_text(encoding="utf-8")
    measured=read(RUN/"evaluation/measured_results.json")
    result={r["model"]:r for r in measured["results"]}
    metric=lambda key:result[key]["metrics"]
    manifest=pd.read_csv(RUN/"manifest.csv",low_memory=False)
    train=manifest.loc[manifest.split.eq("development")]
    envs={k:read(RUN/f"models/qwen3_vl/{k}/training_receipt.json") for k in ["standard","sqrt_class_balanced"]}
    env_differences=[k for k in envs["standard"]["environment"] if envs["standard"]["environment"][k]!=envs["sqrt_class_balanced"]["environment"].get(k)]
    if set(env_differences)!={"variant","balance_rule"}:raise ValueError("Cannot present weighting as a single configuration ablation: "+str(env_differences))
    def section(heading):
        lines=course.splitlines();start=lines.index(heading);level=len(heading)-len(heading.lstrip("#"));stop=len(lines)
        in_code=False
        for j in range(start+1,len(lines)):
            if lines[j].startswith("```"):
                in_code=not in_code
                continue
            if in_code:
                continue
            match=re.match(r"^(#+) ",lines[j])
            if match and len(match[1])<=level:stop=j;break
        return "\n".join(lines[start+1:stop]).strip()
    def old_table(n):
        match=re.search(r"^表"+str(n)+r" [^\n]+\n\n(?:\|[^\n]+\n?)+",course,re.M)
        if not match:raise ValueError(f"Table {n} absent")
        return match[0].strip()
    def remove_tables(text):return re.sub(r"^表\d+ [^\n]+\n\n(?:\|[^\n]+\n?)+", "",text,flags=re.M)
    def remove_figures(text):return re.sub(r"!\[[^\]]*\]\([^\n]+\)\n\n图\d+[^\n]*","",text)
    parts=[];eq_count=0
    def add(text):parts.append(textwrap.dedent(text).strip())
    def eq(key):
        nonlocal eq_count
        eq_count+=1;add(f"<!-- equation:{key}:{eq_count} -->\n\n$$\n{LATEX[key]}\\tag{{{eq_count}}}\n$$")
    def fig(n,name,caption):add(f"![{caption.split('。')[0]}](research_figures/{name}.png)\n\n图{n} {caption}")

    add("# 低分辨率无人机机型识别中的视觉表示与监督适配评估")
    add("""
    ## 摘要

    低分辨率无人机图像中的机身、旋翼与相机结构往往难以辨认，通用预训练能力能否迁移到这一细粒度识别场景，需要在固定数据协议下检验。本文构建含20,010张真实目标裁剪、12个作者类别的数据版本，并以保守场景和相似性约束组划分14,106张训练图像与5,904张测试图像。研究比较手工视觉表示、冻结SigLIP2表示，以及Qwen3-VL-2B的零样本、标准QLoRA和类别加权QLoRA，共13组模型设置。四类传统分类器分别为逻辑回归、K近邻、支持向量机和随机森林，选参在训练集内部的两折分组交叉验证中完成。手工特征调优KNN取得73.07%的测试准确率与61.08%的Macro-F1；Qwen标准适配由零样本的1.59%准确率提高到49.46%，但未超过传统基线。组件消融显示，类别加权未带来一致收益。逐类混淆、原生尺度分层与真实错误样例表明，平均分数不足以描述模型行为。测试仅覆盖8类，内部验证计分仅覆盖6类，来源组也未认证为独立飞行，因而结果用于当前协议下的经验诊断，不构成完整十二类外部泛化结论。数据获取、清洗代码与复现材料随文保留。

    **关键词：** 无人机机型识别；小目标；视觉表示；QLoRA；消融实验；分组评价
    """)
    add("项目代码仓库："+REPO)
    add("""
    ## 1 引言

    无人机识别不仅需要区分飞行目标与背景，还可能需要辨别相似的具体机型。该任务依赖旋翼布局、机身比例、相机和支架等局部线索；当目标只占少量像素时，简单放大无法恢复已经丢失的信息。与此同时，公开图像常来自视频片段或重复转载，同一来源中的相邻帧会使按图片随机划分的评估过于乐观。模型能力、视觉证据与划分协议必须共同审视。

    视觉语言预训练与参数高效适配为这一任务提供了可行工具，但“大模型更强”不等于在任意小样本、细粒度数据上都更准。本文不预设生成式模型胜出，而是提出三个可检验问题：RQ1，当前采集、标注与分组能支持多大范围的评价；RQ2，在相同分类器与划分下，冻结预训练表示相对手工表示有何差异；RQ3，监督适配及类别加权是否分别改善视觉语言模型的闭集标签预测。
    """)
    fig(1,"fig01_motivation","研究动机与三个研究问题。无人机和像素块为机制示意，非测试样例；小目标数量来自本项目清单。图中不预设表示或适配策略优胜。")
    add("""
    本文的主要工作体现在三个方面。首先，建立公开作者数据导入、自编网页爬虫、目标框校验、哈希溯源及原生尺度记录的完整数据流程，并明确自采弱标签与主监督样本的不同角色。其次，在同一冻结测试集上组织三类表示与适配路径，报告四类传统分类器和三种Qwen设置，而非只展示最优模型。最后，以配置开关消融、分类器固定的表示替换、参数搜索与错误案例解释失败模式，保留不支持预期的负结果。

    这些贡献属于数据流程与经验评估；HOG、SigLIP2、LoRA与QLoRA均为已有方法。研究对象是作者标注框裁剪后的机型分类，不将已知目标框下的分类成绩写成端到端检测性能。网页标题与描述用于采集溯源，并未与图像形成真实飞行日志或传感器配对输入。

    ## 2 相关工作

    ### 2.1 无人机机型数据与细粒度识别

    Drone-type-Set围绕无人机类型组织图像并提供目标检测研究基础[1]。本文使用其当前可访问作者子集，并将目标框转为分类输入；下载数量与论文名义数量的差异通过清单披露。对于相似机型，局部外观、视角及背景可能同时影响分类，数据来源和类别覆盖因此比单纯累计图片总数更值得关注。另一公开作者来源VisioDECT在项目来源索引[7]中登记；本实验遵循实际取得的图像和标签，不推断未提供的遥测信息。

    ### 2.2 预训练视觉表示与视觉语言模型

    SigLIP2通过视觉语言预训练学习可迁移的图像表示[2]，为冻结特征提取加轻量分类器提供基础。与此不同，Qwen3-VL-2B-Instruct通过图像与文本指令生成答案[5]，闭集分类还需要将视觉判断映射到固定标签字符串。Tong等在MMVP中指出，多模态模型可能在特定视觉模式辨别上存在系统性短板[8]；这支持开展针对视觉证据的诊断，但其旧模型和不同任务上的结果不能直接解释本项目的Qwen表现。本文通过实测比较两种迁移路径，不用通用基准排名替代机型实验。

    ### 2.3 参数高效适配与研究定位

    LoRA冻结预训练权重并学习低秩更新[3]；QLoRA进一步在量化基座上训练低秩适配参数[4]。本文在固定一轮预算下比较不适配、标准适配和类别加权适配，从输出偏置与类别表现分析后训练的作用。本文没有实现检索增强、强化学习或新的对比学习损失，因而不将这些概念写成方法组件。其研究问题是：在当前视觉质量与分组约束下，现有表示和适配手段实际解决了什么，又留下什么。

    ## 3 方法

    ### 3.1 任务定义与整体框架

    设样本由目标裁剪图像I、作者机型标签y及约束组g组成，标签空间为0至11。约束组是根据场景与相似性构造的保守隔离单元，不等同于认证来源视频。训练集与测试集的样本和约束组均保持不相交：
    """)
    eq("split")
    add("""
    式中D表示样本集合，G表示其约束组集合。主训练集含14,106张图像，测试集含5,904张图像，种子为42，约为7∶3；组不可拆分使比例为70.495%与29.505%。三条路径分别使用手工特征、冻结SigLIP2和生成式Qwen，输出统一映射到十二个标签后在同一测试清单评价。模型、提示词和参数在测试前冻结。图2进一步区分前向路径、冻结模块、可训练适配参数与监督损失。
    """)
    fig(2,"fig02_architecture","整体评估框架与QLoRA训练机制。上方两条路径分别在手工与冻结表示上训练四类分类器；下方Qwen路径仅在语言侧注入LoRA。提示词位置不计算答案损失，标准与加权设置共用图像、提示词、基座和训练预算。多模态接口为基座已有结构，不是本项目新增融合模块。")
    add("""
    ### 3.2 数据准入与视觉证据保留

    采集程序先检查robots和允许来源，再按文章正文选择器提取图片、图注与溯源字段，设置访问延时并保存快照；403或无法确认规则时停止该来源。作者数据经原图、标注和框边界核验后生成RGB裁剪。缺失标签或无效框不能用统计填补生成监督样本；数值特征若缺失，则仅在训练折内填补。

    小图、暗图和低对比度保留质量标记，不在看到测试错误后删除困难样本。短边小于24像素的1,262张裁剪仍在主清单中，这意味着“已完成结构清洗”不等同于“每张图像都有足够机型信息”。网页自采材料缺少完整身份核验，未并入主监督训练或测试。完整采集、缺失统计、前后样例和代码见附录A。

    ### 3.3 手工视觉表示与冻结预训练表示

    手工路径将图像缩放到128×128，提取1,764维HOG、54维HSV统计及直方图、10维uniform-LBP，以及亮度、对比度和原生宽高比3维统计量q。其拼接为：
    """)
    eq("hand")
    add("冻结预训练路径使用google/siglip2-base-patch16-224的图像编码器，并对768维输出执行L2归一化：")
    eq("siglip")
    add("""
    两种表示均分别输入逻辑回归、KNN、RBF-SVM和随机森林。除随机森林外，数值维度经训练折内中位数填补、常量方差筛选和标准化处理。对于保留的非零方差维度，标准化为：
    """)
    eq("scale")
    add("均值μ与标准差σ只从当前训练折估计；来源名、文件名、URL与审核文本不进入特征矩阵。树模型依靠逐维阈值和排序分割，通常无需数值标准化。两条路径的差异是视觉表示，而不是将四个分类器做预测融合。")
    add("""
    ### 3.4 语言侧低秩监督适配

    Qwen3-VL-2B-Instruct接收目标图像与十二个候选机型说明，按提示输出LABEL_00至LABEL_11中的一个标签字符串。字符串可能包含多个token，不把它误称为单个词元。基座以4-bit NF4存储，在语言侧q/k/v/o投影及gate/up/down投影注入低秩更新；视觉主干保持冻结。对一个适用线性层，实际计算权重可写为：
    """)
    eq("lora")
    add("W_NF4表示量化基座恢复至计算精度后的冻结权重，A和B是可训练低秩矩阵；该式是已有QLoRA参数化，不是新算法。训练只监督assistant答案及编码器返回的对应结束位置。设A_i为样本i未被−100屏蔽的答案位置，样本损失为：")
    eq("answer")
    add("""
    其中P为固定类别提示词，a_i,<t为当前位置之前的答案token，φ为加入适配器后的参数。图像与用户提示位置被屏蔽，不能仅因提示中包含标签说明就视为模型已学会分类。类别加权版本根据训练集计数n_c计算平方根逆频率，并在全体训练样本上归一化：
    """)
    eq("weight")
    add("K=12，N=14,106。标准版本令所有权重为1，加权版本使用上述w_c；每个微批的目标为：")
    eq("objective")
    add("""
    全局归一化使训练样本平均权重为1。不能在大小为1的微批内再除以该批权重和，否则会把类别加权完全抵消。两版均训练1轮，学习率1×10⁻⁵、单卡微批1、梯度累积8、LoRA dropout 0.05、weight decay 0.01、warmup ratio 0.03，共1,764个优化步。未设独立验证集，也未按测试分数选检查点。

    ### 3.5 验证协议与评价指标

    传统模型在训练集内执行两折分组交叉验证。为保留全部十二类训练样本，只有可分组验证的六类计分；其余只有一个训练约束组的类别始终留在训练折。测试有真实样本的类别集合为S_te={2,3,6,7,8,9,10,11}，而S_CV={2,7,8,9,10,11}。对支持集合S，Macro-F1定义为：
    """)
    eq("macro")
    add("P_c与R_c分别为类别精确率和召回率，分母为零时按实现置零。预测为不在测试真实支持集合内的其他标签仍会造成漏判，并未从准确率中剔除。Accuracy、平衡准确率和加权F1同时报告。网格搜索只最大化内部验证的平均分数：")
    eq("selection")
    add("λ是分类器超参数；两种表示各在基础四类模型中选出KNN后，搜索k∈{3,5,9,15}、投票权重∈{uniform,distance}，各8组、16次折内拟合，再在全部训练图像上重新拟合。CV与测试计分类别不同，不能直接把其分数差全部解释为过拟合；模型族与参数共用同一验证折，还存在非嵌套选择偏差。")
    add("""
    ## 4 实验与结果

    ### 4.1 数据与实验设置

    原始获取总表含32,077张原图，其中包含角色不同的真实、弱标签和第三方合成材料；其总数不等于主训练集规模。主实验由原七类2,720张及新增五类17,290张真实裁剪组成。训练覆盖十二类，固定测试只覆盖八类。
    """)
    add(old_table(1));add(old_table(8))
    fig(3,"fig03_dataset_profile","主清单的类别、集合与原生尺度分布。训练与测试的类别支持不一致；互斥尺度区间反映模型实际获得的视觉证据。")
    add("主实验在Google Colab的NVIDIA A100-SXM4-40GB上完成，随机种子42；早期T4四类辅助实验不混入本结果。主要环境为Python 3.13.15、PyTorch 2.11.0+cu130、Transformers 4.57.1、scikit-learn 1.6.1。其余版本、运行入口和日志见附录E。四种基础分类器的具体参数见表7，内部两折分配见表6。")
    add(old_table(7));add(old_table(6))
    add("""
    ### 4.2 主结果与公平比较

    十三组设置评价相同5,904张冻结测试图像，按测试八类Macro-F1降序展示。十组传统结果由两种表示、各四个基础分类器和各一个调优结果组成；Qwen为零样本、标准和加权适配三组。调优后的SigLIP2-KNN是测试前由CV选出的传统参照，不能因手工KNN最终更高就改写成事前已选最优。Qwen标准适配同样为预先声明的主研究设置。
    """)
    add(old_table(9))
    fig(4,"fig04_model_comparison","相同测试清单上的总体性能。Accuracy与Macro-F1共享方法顺序；后者对八个真实支持类别等权平均。颜色对应表示家族，展示排序不参与模型选择。")
    add("手工调优KNN取得73.07%准确率和61.08% Macro-F1；标准QLoRA对应49.46%和36.68%。冻结SigLIP2并未在所有分类器上超过手工表示。这些结果支持按任务核验预训练迁移，不能泛化为手工方法普遍胜过大模型，也不能仅凭参数量解释优劣。")
    add("""
    ### 4.3 监督适配与类别加权消融

    为回答RQ3，表21以组件开关组织已经执行的三组设置。图像输入、候选标签提示和冻结测试清单一致；标准与加权两次训练还共享基座revision、训练清单、代码哈希、优化器参数、LoRA位置与一轮预算。训练回执environment字段的差异仅为variant和balance_rule。消融A比较无适配与标准适配，检验加入监督训练的整体作用；消融B比较标准与加权适配，检验损失权重规则的边际作用。
    """)
    variants=[("qwen3_vl_zero_shot","− 监督适配","关闭","关闭"),("qwen3_vl_lora_standard","标准设置","开启","关闭"),("qwen3_vl_lora_sqrt_class_balanced","+ 类别加权","开启","开启")]
    rows=[]
    for model,name,sft,weight in variants:
        m=metric(model);rows.append([name,sft,weight,f"{m['accuracy']*100:.2f}",f"{m['macro_f1_supported_labels']*100:.2f}"])
    add("表21 监督适配与类别加权组件消融 分数单位为百分数\n\n"+table(["设置","监督适配","全局类别权重","Accuracy","Macro-F1"],rows))
    zero=metric(variants[0][0]);std=metric(variants[1][0]);weighted=metric(variants[2][0])
    add(f"加入标准监督适配后，准确率变化为{(std['accuracy']-zero['accuracy'])*100:+.3f}个百分点，Macro-F1变化为{(std['macro_f1_supported_labels']-zero['macro_f1_supported_labels'])*100:+.3f}个百分点；从标准到加权，准确率变化为{(weighted['accuracy']-std['accuracy'])*100:+.3f}个百分点，Macro-F1变化仅{(weighted['macro_f1_supported_labels']-std['macro_f1_supported_labels'])*100:+.3f}个百分点。后一个变化不支持整体改进，更不能把极小差值称为统计显著收益。")
    fig(5,"fig11_ablation","适配组件消融。（a）已有三组配置的测试分数；（b）加权相对标准适配的逐类F1变化。总体微小变化伴随类别间得失，不是所有类别一致改善。")
    add("零样本有5,809/5,904张输出标签0，表现为标签输出偏置；标签0在测试中没有真实样本。监督适配显著改变了这一输出分布，但单次实验不能分离格式学习与视觉识别各自贡献。加权试图提高低频类的损失占比，却不增加其来源组数，也不会补回小图的视觉细节。当前没有多种子重复，故这组消融报告方向与观察值，不作机制因果或显著性保证。")
    add("""
    ### 4.4 分类器固定的视觉表示替换

    为回答RQ2，图6固定基础分类器家族和参数，对比手工与冻结SigLIP2表示。输入维度、方差筛选后的特征空间及必要预处理随表示变化，因此它检验的是整条表示路径，不是HOG、颜色或纹理单个特征块的去除实验。
    """)
    rows=[]
    for clf,label in [("logistic","逻辑回归"),("knn","KNN"),("svm","RBF-SVM"),("forest","随机森林")]:
        hm=metric("handcrafted_"+clf)["macro_f1_supported_labels"]*100;sm=metric("siglip2_"+clf)["macro_f1_supported_labels"]*100
        rows.append([label,f"{hm:.2f}",f"{sm:.2f}",f"{sm-hm:+.2f}"])
    add("表22 相同分类器下的表示替换 对应测试Macro-F1\n\n"+table(["分类器","手工 / %","SigLIP2 / %","变化 / 百分点"],rows))
    fig(6,"fig12_representation","固定基础分类器的视觉表示替换。四组对照沿用同一划分与模型参数；结果依赖表示和决策边界的组合，不能仅由表示规模排序。")
    add("""
    ### 4.5 参数搜索与训练过程

    两条传统路径均在内部CV选中KNN，网格最优配置为k=3且距离加权。表10报告调优前后结果及两折波动，表11给出百分点变化。调优只使用训练集内部验证，最终测试变化作为冻结后的观察。
    """)
    add(old_table(10));add(old_table(11))
    fig(7,"fig05_grid_search","KNN参数敏感性。色标为相同支持六类的两折平均验证Macro-F1；边框标记选中的组合，测试指标没有进入搜索。")
    add("手工KNN调优后测试有小幅提升，SigLIP2-KNN的CV分数上升而测试分数下降，说明内部最优并不保证测试收益。只有两折与一次划分，不能据此声称收益稳定。")
    fig(8,"fig14_training","标准与类别加权QLoRA的原始训练损失日志。仅显示实际记录的loss行，不将末尾摘要损失拼成新的训练点；两版目标权重不同，损失数值不能直接用来排名。")
    add("训练损失总体下降说明模型优化了各自训练目标，但没有独立验证曲线，不能凭曲线下降认定泛化提高。本文不补画不存在的验证轨迹，也不把标准与加权目标的绝对大小解释为模型能力差异。")

    add("""
    ## 5 分析与讨论

    ### 5.1 逐类表现与混淆结构

    总体准确率被样本多的类别影响，Macro-F1又可能掩盖不同类别的失败形式。图9比较手工调优KNN与标准QLoRA，图10补充冻结SigLIP2的逐类F1。所有十二个预测标签和无效输出列保留，仅省略测试无真实样本的行；否则会隐去预测流向未支持类别的错误。
    """)
    fig(9,"fig07_confusion_comparison","两种代表模型的逐行归一化混淆矩阵。行标签含真实样本数；零项不标注，非零但低于1%的项标为＜1。C00至C11对应类别表中的标签0至11。")
    fig(10,"fig08_class_f1","代表设置的逐类F1。只显示有真实测试支持的八类；其他四类不能由零支持下的F1推断泛化表现。")
    add(remove_figures(section("### 5.1 不同模型的表现差异")))
    add("""
    ### 5.2 定性样例与视觉证据

    图11来自已保存的标准QLoRA逐图预测。样例按预先写入绘图代码的“指定类别×正确性分层，取保存顺序中首项”规则选择：类别2的首个正确与错误、类别7和8的首个错误、类别9的首个正确，以及类别10的首个错误。该选择规则只服务解释，不构成无偏错误率估计，完整sample_id与图像哈希另存于qualitative_cases.csv。
    """)
    fig(11,"fig13_qualitative","真实测试裁剪与标准QLoRA预测。标题为原生图像尺寸，绿色和红色边框分别表示正确与错误；展示采用最近邻放大，不进行细节生成。来源为Drone-type-Set作者子集及VisioDECT作者图像，样本溯源见随图CSV。")
    add("这些样例帮助观察外观、视角与像素证据，但不能替代全量身份审核。错误类别之间可能存在局部结构相近、背景影响或样本来源相关性；未运行注意力或因果干预分析时，不能把某个可见区域断言为模型的真实决策依据。")
    add("### 5.3 特征解释与来源影响")
    add(remove_tables(remove_figures(section("### 5.2 特征重要性与误差来源"))))
    fig(12,"fig09_forest_importance","手工特征随机森林的前十二项不纯度重要性。该解释仅对应随机森林；相关特征可能分摊重要性，不能据此解释KNN或Qwen，也不提供因果关系。")
    add("### 5.4 原生目标尺度与误差分层")
    add(remove_tables(remove_figures(section("### 5.3 小目标与数据质量局限"))).replace("图10", "图13"))
    fig(13,"fig10_scale_diagnostics","固定测试结果的原生尺度分层。上方为三种代表设置的准确率，下方为各区间支持数；类别和来源组成随区间变化，此图为事后描述，不是控制变量的尺度消融。")
    add("### 5.5 过拟合与证据边界")
    add(section("### 5.4 过拟合与欠拟合"))
    add("""
    当前实测消融覆盖监督适配与类别加权；固定分类器的表示替换是另一组受控比较。HOG/HSV/LBP特征块去除、视觉主干解冻、LoRA秩与训练轮数敏感性、尺度筛选后重训及独立来源测试仍待开展。现有测试已用于结果分析，后续方法选择应另行登记，并使用新的独立测试数据验证。

    测试覆盖八类、内部CV计分六类、约束组未经独立飞行认证，以及Qwen无独立验证集，限制了结论的外推。各适配设置只有单次运行，尚不能估计训练随机性引起的方差或置信区间。自采弱标签未进入主监督评价，公开数据训练到自采数据外部测试的泛化能力仍待验证。

    ## 6 结论

    本文以低分辨率无人机机型分类为对象，建立可追溯数据流程，并在固定分组协议下评估手工表示、冻结视觉语言表示和生成式模型监督适配。当前结果中，手工调优KNN优于所比较的预训练与适配路径；标准QLoRA相对零样本有明显改善，类别加权却没有形成一致收益。模型比较、适配消融和定性诊断共同说明，视觉证据、来源覆盖与监督目标需要分开检验，通用预训练规模不能直接替代任务验证。

    下一步应优先补充每类多个可核实独立来源，改善固定验证与测试支持，再开展视觉侧适配和特征块消融。现有结论限定于本次数据版本与测试支持类别，不宣称新的算法突破或完整十二类外部泛化。课程要求的实现过程、关键代码、真实问题、四道思考题与心得体会保留在以下附录。
    """)

    add("## 附录A 数据采集与预处理过程")
    add("### A.1 数据获取与爬虫设计")
    add(remove_tables(section("### 3.1 数据获取")))
    add(old_table(2))
    add("### A.2 缺失 异常 文本与裁剪处理")
    add(section("### 3.2 数据预处理"))
    add("### A.3 特征工程与相关性记录")
    add(section("### 3.3 特征工程"))
    add("## 附录B 数据划分 模型与评价关键代码")
    for heading,title in [("### 3.4 数据集划分与验证集用途","### B.1 训练 验证 测试与泄露控制"),("### 3.5 多模型训练与参数","### B.2 多模型实现"),("### 3.6 评价实现与网格搜索","### B.3 指标与网格搜索实现")]:
        add(title)
        if title.startswith("### B.2"):
            add("逻辑回归以线性类别得分和概率损失学习边界；KNN按特征距离选邻居并投票；RBF-SVM以核映射学习非线性间隔，C控制正则化而gamma控制局部影响；随机森林对随机化决策树进行集成，能表达非线性交互并提供不纯度重要性。四者分别代表线性、邻域、核和树集成原理，均由训练样本建立决策规则。")
        if title.startswith("### B.3"):
            add("Accuracy为全部测试样本预测正确的比例；平衡准确率为实际出现类别召回率的均值；Weighted-F1按真实样本数加权各类F1。Macro-F1对支持集合内的类别等权平均，因而更关注低频类别；所有指标必须与对应支持集合共同解释。")
        add(remove_tables(remove_figures(section(heading))))
    add("### B.4 预测与统计补充")
    for n in [12,13,14]:add(old_table(n))
    add("## 附录C 实验问题与解决方法");add(section("## 6 实验问题与解决方法"))
    add("## 附录D 课程思考题作答")
    for i in range(1,5):
        prefix=next(x for x in course.splitlines() if x.startswith(f"### 7.{i} "))
        add(prefix.replace(f"7.{i}",f"D.{i}",1));add(section(prefix))
    add("## 附录E 环境 运行方式与开源材料")
    add(section("### 1.2 软硬件环境"))
    add(section("## 附录 A 可复现性与开源材料").replace("paper_figures 保存十组320 dpi PNG、SVG矢量源图、尺度诊断表与绘图回执", "research_figures保存统计图、机制示意、真实案例、消融CSV与绘图回执"))
    add("新版研究论文的生成入口为 python scripts/18_build_research_paper.py --refresh-figures；该脚本需要已导入的原始结果JSON、CSV及合法取得的本地样本，不加载模型或重新训练。17_complete_course_report.py保留为原课程过程报告的生成入口。公式在Word中以可编辑Office Math保存，Markdown保留对应LaTeX。")
    add("## 附录F 课程总结与要求对应")
    add(section("## 8 实验总结与心得体会"))
    add("数据获取与爬虫对应3.2和附录A.1；缺失、异常、文本清洗和原始前后样例对应附录A.2；编码、标准化、相关性与特征筛选对应3.3和附录A.3；7∶3划分、随机种子及防泄露对应3.1、3.5和附录B.1；四类模型与参数对应4.1和附录B.2；测试评价、网格搜索及调优前后对比对应4.2、4.5和附录B.3；图表、特征解释及误差分析对应第5章；真实问题对应附录C；四道思考题对应附录D；心得体会对应本附录。正文采用论文组织，而不是将老师八项要求直接作为八章。")
    add("## 参考文献")
    references=section("## 参考文献")
    references=references.replace("[1] AlDosari", "[1] AlDosari")
    add(references)
    add("[8] Tong S, Liu Z, Zhai Y, et al. Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs. CVPR, 2024: 9568–9578. https://openaccess.thecvf.com/content/CVPR2024/html/Tong_Eyes_Wide_Shut_Exploring_the_Visual_Shortcomings_of_Multimodal_LLMs_CVPR_2024_paper.html")

    text="\n\n".join(parts)+"\n"
    # Renumber tables in manuscript order and update all numerical references.
    ids=[int(x) for x in re.findall(r"^表(\d+) ",text,re.M)]
    if len(ids)!=len(set(ids)):raise ValueError("Duplicate table caption")
    mapping={n:i+1 for i,n in enumerate(ids)}
    text=re.sub(r"表(\d+)",lambda m:"表"+str(mapping.get(int(m[1]),int(m[1]))),text)
    text=text.replace("图1概括三条独立比较路径。", "整体框架概括三条独立比较路径。")
    text=text.replace("正文采用论文组织，而不是将老师八项要求直接作为八章。", "")
    if sum(line.startswith("```") for line in text.splitlines())%2:
        raise ValueError("Unclosed Markdown code fence")
    text=text.replace("paper_figures/scale_diagnostics.csv", "research_figures/scale_diagnostics.csv")
    text=text.replace("17_complete_course_report.py   更新报告，--refresh-figures 重画图表", "17_complete_course_report.py   保留原课程过程报告\n18_build_research_paper.py     生成当前研究论文，--refresh-figures 重绘")
    dest=ROOT/"reports/final_report_research.md";dest.write_text(text,encoding="utf-8")
    render_paper(dest,ROOT/"reports/final_report_research.docx",research=True)
    evidence={"freeze_hash":measured["freeze_hash"],"metric_source_sha256":hashlib.sha256((RUN/"evaluation/measured_results.json").read_bytes()).hexdigest(),"models_run":False,"gpu_used":False,"equations":eq_count,"tables":len(ids),"referenced_figure_groups":13,"ablation_configuration_differences":env_differences,"new_method_claim":False,"scope":"Research structure with all course evidence in appendices; measured adaptation ablations only."}
    (ROOT/"reports/research_paper_evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Research paper written: {len(text):,} characters, {eq_count} native equations, {len(ids)} tables; no models run.")


if __name__=="__main__":main()
