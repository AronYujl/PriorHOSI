# 状态条件 HOI/HSI Prior 组合的 HOSI 实验计划

状态：预注册；基线提交 `b9a158f75ab0740c91c9cfc8863a65fa381b014c`<br>
创建：2026-07-11（Asia/Shanghai）<br>
主投：CVPR 2027；若未录用，再改进后投 ICCV 2027，不并行投稿同一工作。

## 1. 主张、边界与接口

InfBaGel 的 synthesized-OMOMO scene 由真实 motion 反推，训练条件分布与真实场景下
“scene 约束 motion”的推理分布不一致。本工作不宣称形式化因果识别，研究主张限定为：
去除 motion-derived scene supervision，并在真实场景中进行状态条件专家组合。

\[
p_{HOI}(H,O\mid T,G_O,P),\qquad p_{HSI}(H\mid S,T,G_H,P)
\]

- HOIPrior 仅用无场景 OMOMO，从零训练人体—动态物体先验。
- HSIPrior 仅用真实 LINGO scene-motion，从零训练导航/静态场景先验。
- 不构造 motion 反推 scene 的 paired-HOSI；原 checkpoint 仅作 baseline。
- 推理时冻结专家，依据状态组合，并用真实场景几何能量形成可行后验。
- 只使用官方 OMOMO/LINGO、CHOIS evaluator/checkpoint 和公平基线官方资产。
- 目标位姿由 oracle/上游提供，不做语义场景识别；当前工作保持 kinematics-based。

统一 `TaskSpec` 包含 instruction、人体初态、动态物体实例/BPS/初始位姿、外部
object/pelvis/static-interaction goals、occupancy/SDF、场景边界/坐标系、seed 和最大
episode 长度。`TaskPlan` 只允许：

`NAVIGATE → APPROACH → ACQUIRE → TRANSPORT → RELEASE → STATIC_INTERACT → DONE`

任意状态可进入 `RECOVER`。每个状态保存原文跨度、模式、目标、终止守卫、超时和最多
重试；结果保存人体/物体运动、状态轨迹、守卫事件、失败原因和分阶段耗时。LLM 只生成
计划，实时转移由确定性守卫执行。

## 2. 模型与组合

两专家统一 16 帧窗口、2 帧历史、500-step diffusion schedule、坐标规范与 232 维表示。
HSI 的物体/接触通道固定为空且从损失排除；默认使用作者已为 LINGO 替换的 OMOMO
normalization。人体 clean prediction 使用按帧和身体组的结构化门控：

\[
\hat x_{0,h}=G(s,t,\Delta x,r)\hat x^{HSI}_{0,h}
 +(1-G)\hat x^{HOI}_{0,h}.
\]

身体组为 root/pelvis、下肢、躯干/头、左臂/手、右臂/手；位置/旋转共享门控。物体位姿
与接触永远来自 HOIPrior。Mixer 输入状态、去噪时刻、专家预测/分歧、文本/目标条件及
各身体组/物体 SDF 风险。冻结专家，在真实 OMOMO、真实 LINGO 和训练 split 内独立配对
的可行 counterfactual 条件上训练，不使用 HOSI GT。损失包括双域保持、目标、接触、FK、
SDF、门控时间平滑和窗口连续性；对抗不稳时只用冻结判别器 feature matching。

状态化能量：NAVIGATE/APPROACH 用人景、足地、路径和目标；ACQUIRE 用手物接近/穿透与
人景；TRANSPORT 用人景、物景、接触保持、对象目标和足地；RELEASE 用目标位姿、低速、
稳定放置和手物分离；STATIC_INTERACT 用人景接触、pelvis goal 和姿态稳定；RECOVER 对
碰撞、接触丢失和超时执行回退/重采样。SDF/距离变换通过可微三线性采样作用于 clean
prediction。

## 3. 分阶段执行与门槛

### 阶段粒度与交接约定

- 一个 Codex session 只实现、验证并关闭一个 phase；关闭后不在同一 session 启动下一 phase。
- 若某 phase 无法在单 session 内可靠完成，必须在写实现前拆成编号 subphase，并同步更新本计划、
  分支名、独立 gate 和 registry component；不得仅在执行过程中口头拆分。
- 每个 phase/subphase 合并前必须写 `docs/phase_summaries/PHASE_<N>.md`，至少包含范围、功能与
  配置改动、成功和失败实验、验证命令、结果/artifact hashes、commits/tag、遗留风险，以及下一
  session 的精确入口。新 session 应先读该总结和本完整计划，再继续工作。

### Phase 0：治理、数据与评测闭环

- 从锁定提交建 `research/state-compositional-priors`；禁止旧 feature。
- 跟踪本计划、`AGENTS.md`、registry、split/task manifests 与 artifact hashes。
- 固定 469 条 Atomic-HOSI 参考：82.09% completion、4.57/8.17 cm pelvis/object error、
  0.14 FS、0.781 contact、23.34 FPS。该值须重新运行后才算通过。
- 锁定 CHOIS 官方 evaluator，补 Table 5 FID 与 Top-1/2/3 R-Precision，记录 upstream、
  checkpoint、输入转换哈希。
- 分报 warm generation、LLM planning、end-to-end latency；CUDA 计时显式同步。
- LINGO 按 scene family 分组，mirror/new-loco/action 变体同侧，seed 42 固定 80/20。
- 8×3090 上从 micro-batch `{32,64,128}` 选最大稳定值，以累积固定 effective batch。

通过：Atomic-HOSI 在论文/复现容差内；HOI 全指标可重复；data/evaluator/checkpoint hash
完整。缺数据、官方 evaluator checkpoint 或真实 GPU 运行时不得宣称通过。

### Phase 1：独立专家

HOIPrior 与 HSIPrior 不共享可学习权重。HSI 排除手持动态物体片段，只保留 locomotion、
静态交互和无物体动作，并用 scene-disjoint validation。先 smoke/短预算，再完整训练；
统一按 optimizer updates 和有效样本量计预算。检查 normalization 越界、文本覆盖、短序列、
scene leakage 和不确定性。门槛：关键原生域指标至少达到对应单模型 baseline 的 95%，无
系统性 contact/penetration/FID 退化。失败先审计表示、坐标、mask、normalization 和 split，
不得用 InfBaGel 权重绕过。

### Phase 2：固定组合可行性

先用 oracle state plan；实现共享噪声/timestep、坐标对齐、clean prediction 与固定身体组
门控。比较顺序路由、全局固定、时间、身体组、时空权重。TRANSPORT 默认 HSI 主导
root/下肢，HOI 主导手臂/手/物体。相对顺序路由，人景穿透至少约降 10%，task success 和
contact 各下降不超过 2 个百分点，才进入可学习 Mixer。失败检查 normalization、noise、
timestep、root trajectory 与进度对齐，必要时测解析式 product-of-experts。

### Phase 3：状态/风险感知 Mixer

冻结 experts，训练轻量 Transformer mixer 与两个冻结域特征编码器。仅从 train split 独立
配对 LINGO scene 与 OMOMO 条件，以 A*/SDF 筛可行初态/目标，不给组合 GT；平衡状态与
模式，并给纯域状态 identity anchor。加入约 2× joint model 与双 joint ensemble 控制容量/
计算。模型选择看 task success、HS/OS penetration、contact、FID/R-Precision Pareto，必须
优于固定门控与原 joint training，不用单一加权总分。

### Phase 4：状态化隐式策略

在 Mixer clean output 上做少量 DIP-style gradient refinement。顺序消融 no guidance、
InfBaGel bump-aware、统一 SDF、状态化 energy。成功率/contact 下降各不超过 3 个百分点时，
场景穿透再降至少约 15%。仅大量内循环有效则只留 Quality；无稳定增益则从主模型移除并
记录负结果。不引入 Isaac Gym/低层控制器；所有预注册物理代理指标均保留。

### Phase 5：状态机与组合评测

主规划器固定 Qwen 约 8B AWQ/Instruct、greedy、固定 prompt/schema/checkpoint hash；大模型
只做解析鲁棒性。守卫：5 cm 手物接触、10 cm 目标成功，稳定速度阈值由训练统计得到。
Acquire/Transport/Release 保留完整 OMOMO instruction 与 `pi/end_pi/seq_length`。失败进入
RECOVER，最多两次后结构化终止。先 40 条 pilot（四类各 10），再约 200 条、至少 40 个
unseen scenes：carry/release、push/drag/re-contact、移动椅后并入 occupancy 再坐、搬运后到
第二 pelvis goal/返回。原 469 与新椅子任务分表。自动指标显著领先后做 20–30 人匿名随机
A/B，每任务至少 10 个判断，并设注意力检查。

### Phase 6：Consistency 与实时化

两专家分别蒸馏，比较 16/8/4 steps；纯域状态只调用单专家，混合状态才双专家。缓存 scene
encoding、裁剪/量化 mixer、Fast 模式减少 refinement。必要时以 Quality teacher pseudo
supervision 蒸馏单学生。单 RTX 3090、batch=1 的 Fast 目标 ≥20 FPS，同时报告 Quality 的
完整质量—速度 Pareto。

## 4. 评测、基线与统计

- HOI：Table 5 的目标/轨迹误差、FS、FID、R-Precision、contact P/R/F1、手物/人物穿透、
  GT difference、FPS。
- HSI：goal error、完成时间、人景 penetration ratio/mean/max、FS、FID、Diversity、
  Multimodality、Precision/Recall/F1。
- Atomic-HOSI：保留 Table 1，并输出 approach/acquire/transport/release 谓词诊断。
- Compositional-HOSI：Episode Success、Ordered Completion、Longest Prefix、SPL/路径效率、
  premature transition、timeout、recovery success。
- 连续性：root 位置/速度/加速度跳变、旋转 geodesic jump、jerk、接触连续与边界 FS。
- 效率：参数、FLOPs/窗口、峰值显存、窗口/episode latency、FPS、planning latency。

必跑基线：InfBaGel synthesized/hybrid；专家顺序路由；固定门控；Adapted MixerMDM/DIP；
原 joint、2× joint、双 joint ensemble；LINGO scheduler+守卫；HOSIG（同采样预算）；HOI
子表 CHOIS 及协议兼容的 DecHOI/ROG/ViHOI。无代码或输入不兼容者只引用公开同协议结果。

组件、数据和推理消融按预注册表执行。主表至少 3 个训练 seed；生成指标重复采样给均值和
95% CI；同任务/seed 用 paired bootstrap 或 permutation，并校正主要多重比较。按物体、模式、
路径长度、拥挤度和失败阶段分层报告。

## 5. 变更日志与 fallback

- 2026-07-11：建立预注册。证据为锁定仓库基线与现有 469-case 复现摘要；尚无本分支重跑
  artifact，因此 Phase 0 状态为未通过。
- 2026-07-11：审计 CHOIS commit `8ec585a`。release 中 evaluator 使用作者绝对路径且缺少
  `options/train_options.py` 与官方 feature checkpoint；接入采取严格 provenance/input/asset
  校验，资产不完整时失败，不用自训练或替代网络冒充官方指标。
- 2026-07-12：CHOIS 缺失依赖通过单独锁定的 `text-to-motion` commit 和显式路径 adapter
  补齐；官方 supplied predictions 对 GT 的 GPU 回归得到 FID 0.6881585766、R-Precision@1/2/3
  0.1520833333/0.2979166667/0.4250，完整哈希见 registry。该结果只验证 evaluator，不替代
  InfBaGel HOI 输出评测。
- 2026-07-12：确认 Codex 普通沙箱会隐藏 NVIDIA driver；提权后 8×RTX 3090 24GB 全部可见，
  `infbagel` 环境 PyTorch 1.13.1+cu117 可访问 8 卡。后续 GPU 命令须优先使用该环境，并在
  沙箱报告 CUDA 不可用时提权复核，不能将权限问题记录为硬件缺失。
- 2026-07-12：首次 Atomic-HOSI reportable manifest 在 GPU workload 启动前终止，原因是命令
  未记录所需的 `code/` working directory；失败已追加到 registry。manifest schema 增加显式
  workdir 后使用新 run id 重启，禁止覆盖或复用已终止 id。
- 2026-07-12：Atomic-HOSI r1 在 resolved-config preflight 终止；发现未使用的
  `guidance.pelvis.seq_len` 引用不存在的 `dataset.seq_len`。移除悬空项并将完整 Hydra resolve
  设为 GPU workload 前置条件；r1 未启动采样且已按负结果登记。
- 2026-07-12：Atomic-HOSI r2 完成全部 469 cases/67 scenes：completion 81.663%，pelvis/object
  error 4.686/8.129 cm、FS 0.1384、contact 0.7805；排除 5 条 warm-up 后 generation FPS
  25.647（按总帧/总生成时间为 25.379），与锁定参考一致。manifest、resolved config、聚合和
  逐样本 artifact hash 已登记；Atomic-HOSI 子门槛通过，Phase 0 仍等待 InfBaGel HOI 全指标。
- 2026-07-12：InfBaGel HOI 完成 438 sequences×3 windows 的原生指标复现，并导出严格同 ID
  的 438 对 prediction/GT（每条 126 帧）供 CHOIS evaluator 使用。object/pelvis goal error
  3.037/3.923 cm、contact P/R/F1 0.7908/0.7276/0.7273、FS 0.3334；batch=438 throughput
  322.56 FPS 只作批吞吐量，不冒充 batch=1 latency。Phase 0 尚待 CHOIS FID/R-Precision。
- 2026-07-12：锁定 CHOIS evaluator 在上述 438 对 matched exports 上得到 FID 0.93342、
  R-Precision@1/2/3 0.17308/0.31010/0.43510、Matching Score 3.82295、Diversity 9.14892；
  evaluator/checkpoint/input hashes 完整。HOI 质量子门槛通过；在合并 Phase 0 前仍需补充显式
  batch=1 HOI latency，并完成 `{32,64,128}` 训练 micro-batch 决策，不能用 batch=438 吞吐替代。
- 2026-07-12：首次 HOI batch=1 timing run 完成生成后，在指标阶段发现 `compute_metrics`
  重新读取全 438 条边界、未遵守 subset 契约，因而失败并登记。修复为以传入的 sequence map
  为唯一计数来源；完整 438 run 结果不受影响，使用新 run id 重测 batch=1。
- 2026-07-12：修复后的 HOI batch=1 timing 在 RTX 3090 上完成；排除 1 个 warm-up batch 后，
  3 windows/126 帧纯生成耗时 7.0871 s，即 17.779 FPS，CUDA 边界均同步。单样本质量指标不
  并入 Table 5 均值；batch=1 timing 子门槛完成，Phase 0 只剩训练 micro-batch 决策。
- 2026-07-12：实现预注册的 8 卡训练 micro-batch 审计：候选 `{32,64,128}` 均执行真实
  OMOMO forward/backward 和一次 optimizer update，固定 global effective batch=1024，分别
  使用累积 `{4,2,1}`；全部候选必须先保存 Hydra resolved config 并形成带哈希的集合，只有所有
  preflight 成功才启动任一 GPU workload；记录各 rank peak allocated/reserved CUDA memory。
  审计代码与训练累积语义先提交，随后才允许 clean-worktree GPU run。
- 2026-07-12：新增阶段粒度与交接治理：一个 session 关闭一个 phase；过大的 phase 必须在实现
  前拆成有独立 gate 的 subphase；每阶段合并前提交标准化工作总结，供下一 session 与本计划共同
  作为唯一进度入口。该变更只约束执行治理，不改变研究假设或 Phase 0 指标协议。
- 2026-07-13：首次 8 卡 micro-batch audit 的 `{32,64,128}` 均完成 resolved-config preflight
  并进入真实 OMOMO loss path，但全部在 `Sampler.p_losses` 尾部因遗留清理变量 `occ_goal`
  未定义而失败；失败 manifest/日志已按原样登记，未选择 batch。该问题与 batch 大小无关，按
  Phase 0 训练路径诊断已仅移除无效的 `occ_goal/occ_temp` 清理引用并增加静态回归测试；须以
  新 run id `r1` 从干净修复提交重跑，不覆盖首次失败。

每个阶段只允许上文给出的诊断/fallback。新增方向必须先在此处追加日期、证据和原因，并在
registry 登记，再实现代码。
