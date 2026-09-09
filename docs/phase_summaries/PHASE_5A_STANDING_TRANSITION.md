# Phase 5.1 — 站立交接验证与现成 inbetween 模型核对

2026-09-09，分支 `phase/05a-stand-wait`。工程验证完成，科学分类
`fixed-text-standing-handoff-negative`。

## 结论

站立边界筛选可以收缩任务范围；本轮固定 HSI 文本过渡的可靠性不足。
在12个已满足站立、空手、物体落地与低速代理条件的实际 HOI 末端上，
`stand up and wait` 达标2/12，`Stand upright and remain in place.` 达标1/12。
本轮预设的80%站立成功门槛失败。下一入口是给定两端姿态的 inbetween
适配验证，当前两个专家及已选 HOI 编辑链继续作为输入来源。

外部模型本轮完成了源码、接口和权重可获取性核对。CondMDI官方权重实际
可下载；Kimodo-SMPLX格式更贴近当前系统，但当前配置的账户获得401访问拒绝；
SceneMI公开入口仍缺少可直接使用的预训练/推理交付。本轮没有执行外部模型。

## 范围、输入与实现

输入为 Phase2.34 完整链的 `correct_terminal_draw0`：
P15+guide → relation-preserving edit → HSI body guidance → terminal repair。
源目录：`/data/yujinlun/InfBaGel-mixer/results/experiments/p2-mixer-hsi-integrated469-s42-20260908`。
审计469条实际生成输出，去除原生插值最后两个重复帧后测量末0.3秒。
这些数据已有开发历史；本轮筛选是诊断输入选择，正式 benchmark 应依据
原始动作、场景及统一可行性标准独立构建。

| 物体 | 总数 | 站立代理通过 | 同时满足释放/落地代理 | 本轮选取 |
|---|---:|---:|---:|---:|
| clothesstand |67|57|1|1|
| floorlamp |67|67|13|5|
| monitor |67|55|0|0|
| smallbox |67|50|0|0|
| smalltable |67|51|2|2|
| suitcase |67|45|0|0|
| tripod |67|59|6|4|
| 合计 |469|384|22|12|

22个候选来自19个场景，选取12个不同场景。释放/落地过滤只认可世界Y=0的
地面支撑，未判定桌面等其它支撑面，因此22不能解释成全部可释放任务数量。
各条件失败次数有重叠：物体落地341、平移低速204、旋转低速144、手部离物216。

新增组件 `code/mixer/standing_transition.py`，通过原
`code/test_infbagel_hosi.py` Hydra入口和单一配置片段
`code/config/config_sample_hosi_stand_wait.yaml` 调用。两帧stride3历史来自
实际生成末端；一窗口生成14个新10Hz样本，即1.4秒，按原生规则还原30Hz。
人体体型与物体最终状态继承源动作。HSI接收已存在的空物体通道视图，场景查询
保留实际固定物体；源码不调用HOI网络。

固定 R2 epoch222、500步扩散、CFG=1、R2后验系数场景引导、seed42。
两种文本共享初始噪声、后验随机数及场景采样随机数。静止骨盆目标使用真实
接口 `need_pelvis_dir=true`；此标志控制整个骨盆目标token，没有单独朝向开关。
原生历史编码/解码最大误差0.000358mm，接缝位置最大误差0.000377mm。

## 固定指标与完整结果

站立代理：躯干倾斜<=25°、骨盆高度>=65cm、每帧至少一个足部标记距离地面
<=8cm；释放代理要求双手离物>=8cm、物体最低点距地面<=5cm、物体平移速度
<=0.10m/s、旋转速度<=0.5rad/s。交接还要求尾段人体根速度<=0.15m/s、根角速度
<=0.5rad/s，以及整段根XZ偏移<=10cm。完整定义见
[预注册](../plan/PHASE_5_INFERENCE.md)。这些是明确的运动学代理阈值。

| 指标 | stand up and wait | Stand upright and remain in place. |
|---|---:|---:|
| 站立/完整交接通过 |2/12|1/12|
| 场景几何阈值通过 |12/12|12/12|
| 足部高度条件失败 |8/12|8/12|
| 根平移速度条件失败 |4/12|4/12|
| 根漂移条件失败 |3/12|4/12|
| 根角速度条件失败 |2/12|4/12|
| 手部离物条件失败 |1/12|1/12|
| 平均最大根漂移 cm |7.9079|8.2238|
| 平均尾段最大足部标记高度 cm |9.3791|9.3949|
| 平均尾段最大根速度 m/s |0.1158|0.1255|
| 平均接缝关节速度变化 m/s |0.4871|0.4921|
| 平均窗口生成耗时 s |63.9777|64.4008|

场景几何阈值为全顶点/全帧平均穿透<=5mm、最大穿透<=5cm且在场景范围内；
通过此项表示满足该容差。24条中的实际最大单点穿透为3.335cm。
各失败条件可重叠，完整逐任务结果均保留。

10000次seed42配对任务bootstrap使用GPU现有实现，并经
`tools/paired_bootstrap.py`复核，24项统计的最大差为2.78e-17。
改写文本相对原文本：成功率差−8.33个百分点，95%区间[−25,0]；
根漂移增加0.316cm，区间[0.106,0.549]；足部标记高度差0.0158cm，
区间[−0.0376,0.0717]。改写文本未展示站立支撑收益。
样本为固定12个有开发历史的任务，区间仅描述这批任务的配对差异。

## 身体表面核对

为解释足部标记代理的失败，在原判定完成后重建全部24条SMPL-X网格。
身体最低顶点的Y坐标平均由初态2.4936cm升至尾段5.6696/5.6744cm；
24/24条均升高，平均增量3.1784cm，最小增量1.8116cm。
这是相对世界Y=0的表面高度，初始动作本身也有高度偏离。
该补充量解释了实际几何变化，保持原判定阈值和结论。

模型位置通道与原生人体骨盆的最大偏差为0.0958mm；相比数厘米的高度变化，
这项坐标偏差很小。当前证据记录过渡时的支撑与速度问题，尚未把它们因果
归结于训练动作覆盖、人体体型条件或某一引导项。

[全部12例高度曲线](../../results/experiments/p5-inference-stand-wait-parallel-s42-20260909/analysis/standing_surface_height.png)
· [全部12例首尾姿态](../../results/experiments/p5-inference-stand-wait-parallel-s42-20260909/analysis/standing_endpoint_poses.png)
· [可导出的PDF](../../results/experiments/p5-inference-stand-wait-parallel-s42-20260909/analysis/standing_surface_height.pdf)

## 现成 motion inbetween 模型

| 模型 | 可用性核对 | 与当前任务的适配 |
|---|---|---|
| CondMDI / SIGGRAPH2024 |官方源码已克隆到`/data/yujinlun/CondMDI`。官方随机关节条件权重ZIP通过HTTP下载及范围读取验证，取得真实`args.json`；完整权重未下载，未做本项目推理。|HumanML3D绝对根263维、22关节、20Hz；支持首尾/稀疏/部分关节条件。需要当前SMPL-X动作与其表示间的转换和往返误差检查；模型输入没有场景或动态物体。|
| Kimodo-SMPLX-RP-v1 / 2026 |官方有代码、预训练权重和约束接口。当前账户下载元信息请求返回GatedRepoError/401；没有代用户提交访问申请。|22关节SMPL-X身体、根平移和旋转矩阵输出、30Hz，格式更接近当前系统。官方支持姿态/末端约束；仍需核对体型、坐标与约束后处理，场景/动态物体由外部系统处理。|
| SceneMI / ICCV2025 |官方仓库提供训练说明，项目页未给出可直接获取的预训练权重；公开issue3仍在询问推理代码。|场景条件inbetween与任务最贴近，但当前公开交付不足以直接替换桥段；其TRUMANS训练来源还需与本项目场景划分核对。|

来源：[CondMDI官方代码](https://github.com/setarehc/diffusion-motion-inbetweening)、
[CondMDI项目页](https://setarehc.github.io/CondMDI/)、
[Kimodo官方模型卡](https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1)、
[Kimodo约束接口](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/constraints.html)、
[SceneMI官方代码](https://github.com/woo0818/SceneMI)、
[SceneMI推理代码issue](https://github.com/woo0818/SceneMI/issues/3)。

当前可直接取得权重的候选是CondMDI；获得Kimodo-SMPLX访问权限后，可以优先
测量其较接近原生格式的适配。上述是接口与可用性判断，外部模型在本项目中的
过渡质量仍待测量。官方CondMDI压缩包约1.79GB，真实配置为条件UNet、1000步
cosine扩散、`random_joints`和绝对根表示；下载核对记录保存在
`results/verification/condmdi_availability.json`。

inbetween的具体输入应包含上一段末尾历史、下一段期望起始姿态/短后缀和过渡
时长。站立筛选限制端点的范围，桥段需要实际生成到达目标姿态。若使用benchmark
提供的子任务参考初态，将它明确作为目标条件，并给所有比较方法相同条件。
桥段期间保持已释放物体的实际位姿，并检查人体、足部与场景关系。对比实验中
共享桥段模型，可以分别衡量任务先验与过渡模块的贡献。

## 运行、失败与验证

原始启动未产生Python输出，保留失败manifest；r1完成源审计后因场景键前缀错误
在去噪前失败，修正为原生HOSI使用的场景ID。r2串行完成7个cell后，依据约64s/
窗口及约0.524GiB峰值张量显存的测量停止串行调度；17个剩余cell分到8张共享
RTX3090，9个作业全部exit0。最终24个指定cell各出现一次，已完成结果按完成
状态继承，任务/文本选择未使用其质量结果。被中断窗口的前向次数没有完整记录。

所有正式GPU工作使用已验证`infbagel`环境，resolved配置、机器状态、输入引用、
日志、逐任务动作与指标均保存。源码修正后的完整套件为1119通过/6跳过；
补齐只读G=0基线引用后，该组件14项通过，其中2项为此前跳过的资产检查。
剩余4项依赖历史checkpoint/评测文件，与父分支的历史跳过项一致。
registry校验通过；无专家训练与新HOI采样。

验证入口：`python -m pytest tests`、
`python -m pytest tests/phase2/test_hosi_per_sequence.py -q`、
`python tools/experiment.py validate`、原Hydra入口的resolved配置检查、
`tools/paired_bootstrap.py --expected-sequences 12 --replicates 10000`。
Python路径固定为`/data/yujinlun/anaconda3/envs/infbagel/bin/python`。

主运行目录：
`results/experiments/p5-inference-stand-wait-parallel-s42-20260909`。
`completion_metrics.json`、`analysis/`及四份运行manifest构成完整记录。
其中两个失败manifest、串行中止manifest分别位于同级初始、r1、r2运行目录。
补充表面分析首次因工作目录不符合原生SMPL路径约定而失败，纠正到`code/`后
完成，事件另存于`analysis/postprocess_events.json`。

分支记录了预注册、实现、场景键修正、依据测量的调度变更和完成提交；
源码在串行与并行生成之间保持一致，差异仅为计划文档。
完成记录由`exp/p5a-standing-transition-v1`定位，保持独立推理分支。
下一入口是一个具体的首尾约束inbetween适配实验；本阶段交付截至上述确认。
