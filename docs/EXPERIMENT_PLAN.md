# 状态条件 HOI/HSI Prior 组合的 HOSI 实验计划

状态：Phase 0、Phase 1A 已通过；Phase 1B D2-V--D2-AD 尚未形成 selectable HOIPrior；
D2-AC0 已完成并分类为 `interaction-adapter-locality-negative-stop`，checkpoint 不可选择，
D2-AC1 不 eligible；D2-AD0 human-local BPS coordinate-contract repair 已完成并分类为
`local-frame-interaction-adapter-locality-negative-stop`，checkpoint 不可选择；
D2-AE0 GPU-native sparse current-state role-relative object-field routing 已于 2026-07-28
完成 plan-only 预注册，尚未实施或启动 workload；
Phase 1C 未启动；
基线提交 `b9a158f75ab0740c91c9cfc8863a65fa381b014c`<br>
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

Phase 0 gate 决定：通过。469-case Atomic-HOSI、完整 HOI 原生/CHOIS 指标、batch=1 timing、
数据/evaluator/checkpoint hashes、LINGO split 与 8 卡 micro-batch 决策均已形成可复现闭环。
Phase 0 当时锁定的 smoke 配置为每卡 micro-batch 128、8 卡、accumulation 1、global
effective batch 1024。该结果继续作为历史 smoke/容量证据；2026-07-13 的训练资源协议修订
不追溯修改 Phase 0，但覆盖其对 Phase 1 正式训练的跨专家约束。完整历史证据见
`docs/phase_summaries/PHASE_0.md`。

### Phase 1 正式训练资源协议（2026-07-13 修订）

- 目标是分别训练出能力最强的 HOIPrior 和 HSIPrior，而不是把 batch 当作跨域受控变量。
- 服务器分配固定为 HOIPrior 使用 4×3090、HSIPrior 使用 8×3090。两者分别在 Phase 1B/1C
  内审计与候选 effective-batch 档位整除兼容的最大稳定 per-GPU micro-batch；
  记录峰值显存、预留余量、吞吐、GPU 数和并行训练造成的 CPU/磁盘/GPU contention。
- 正式 effective batch 默认候选只允许 `{512,1024,2048,3072}`。其他档位必须先做 dated
  plan/registry amendment；禁止 `1536` 等未登记中间值。
- 优先 accumulation 1 和充分利用显存的最大稳定 micro-batch；仅为达到选定 effective-batch
  档位或预注册的优化理由使用 accumulation。effective batch 改变时联合预注册 LR/warmup。
- 公平性在同一专家内部实施：其架构、损失和消融对照固定 hardware/effective batch/数据预算。
  HOI 与 HSI 之间不要求相同 micro-batch、GPU 数、effective batch 或 optimizer-update 数。
- 训练预算以 processed windows/frames（并报告 epochs）为主；optimizer updates 必须记录，
  但只作为 effective batch 推导出的计数，不作为 HOI/HSI 跨域相等约束。
- 机器可验证的权威协议为 `experiments/training_resource_protocol.json`；Phase 1B/1C 必须在各自
  实现和 reportable training 前登记最终 hardware、micro-batch、accumulation、effective batch、
  LR/warmup 与 processed-window/frame budget。

### Phase 1：独立专家（拆分为 1A–1D）

Phase 1 原范围包含两套数据契约、两次从零完整训练、两套原生域评测与最终联合验收，无法在
单 session 内可靠实现和验证，因此在任何 Phase 1 代码前拆为以下 subphase。registry 的
`phase` 仍使用 `p1`，run component 和分支分别标识 `data`、`hoi`、`hsi`、`gate`；每个
subphase 独立总结为 `PHASE_1A.md` 等文件。

#### Phase 1A：数据契约、表示与专家脚手架

在 `phase/01a-data` 上固化 OMOMO-only HOI 与过滤后的 LINGO HSI dataset/config contract，
实现通道 mask、相同 232 维表示、normalization/坐标审计和从零初始化断言；HSI 排除手持动态
物体窗口，只保留 locomotion、静态交互和无物体动作，并严格使用 scene-disjoint split。
执行 CPU/unit test、单卡最小 batch 和 8 卡各一次 smoke update，不做筛选性完整训练。

门槛：数据计数/filter/split hashes 固定，无 scene-family leakage；两专家无共享可学习参数且
均拒绝 released InfBaGel checkpoint 初始化；smoke loss 有限、通道 mask 正确、resolved config
和 manifest 完整。通过后总结并 tag `exp/p1a-data-v1`。

#### Phase 1B：HOIPrior 从零训练与原生域评测

在 `phase/01b-hoi` 上只训练 HOIPrior，并固定使用 4×RTX 3090 服务器。先审计显存并从
`{512,1024,2048,3072}` 选择
正式 effective batch，同时联合预注册 LR/warmup 和 processed-window/frame 预算；先 smoke 后短预算
筛选，再对锁定配置执行完整训练；运行 HOI 原生指标与 CHOIS FID/R-Precision，并审计
normalization 越界、文本覆盖、短序列、contact/penetration 与不确定性。

Phase 1B 的执行 worker 固定为 `10.181.9.214`，但 `10.184.17.253` 仍是代码、计划和 registry
的权威开发/集成端。首个 reportable workload 前必须按 `docs/MULTI_SERVER_TRAINING.md` 完成并归档
worker preflight：精确 Git commit、4 卡型号/显存/驱动、机器本地 Python 与依赖快照、可写空间、
OMOMO-only 数据快照及 Phase 1A source hashes、`ROOT_DIR`/工作目录、时钟与 SSH/回收路径。代码只以
Git commit 发布，数据只从不可变 snapshot 读取，运行中的 worktree/results/checkpoints/registry
禁止双向同步。四卡机不得接收 LINGO `data/dataset` 或 synthesized-OMOMO `Scene*` 监督资产。
四卡机固定使用 `/home/yujinlun/data`；因八卡机到四卡机 TCP/22 被网络策略阻断、反向端口可达，
所有服务器间 Git/rsync 由四卡机使用 source-restricted 专用密钥主动发起。Windows 只用于首次
复制公钥文本，不作为大文件中转或实验 artifact 权威副本。

门槛：HOI 关键原生域指标达到对应单模型 baseline 至少 95%，无系统性 contact、penetration、
FID 退化；首次失败后的修改只允许按下文 dated remediation 预注册逐项实施。通过后总结并 tag
`exp/p1b-hoi-v1`，不得用 released checkpoint 绕过失败。

2026-07-13 Phase 1B 执行预注册如下；以下预算、候选和选择规则在任何 Phase 1B 实现或
GPU workload 前锁定：

- 数据仍严格来自 Phase 1A hash 锁定的 scene-free OMOMO。训练根中的 4,304 条 sequence
  以 `SHA256("42:" + sequence_name)` 排序，固定前 5%（216 条）作为 Phase 1B 内部
  sequence-disjoint validation，其余只用于训练；官方 438-sequence OMOMO test 不参与
  筛选或 checkpoint 选择，只在正式配置锁定后做原生域最终评测。划分必须形成 tracked
  manifest 并记录 Phase 1A source/contract hash。
- HOIPrior 固定为只接收 noisy 232-D motion、timestep、完整 instruction embedding、动态物体
  BPS、object goal 与 `pi/end_pi/seq_length` 的 8-layer、512-width、16-head Transformer
  clean-motion predictor；无 scene argument、scene encoder 或 scene loss。窗口/历史/diffusion
  固定为 16/2/500。目标函数固定为五个 representation field 的显式重建项，加手/足 FK、
  速度和 object-goal consistency；训练使用 AdamW、AMP、gradient clipping 与 EMA，不比较
  新架构或新损失方向。
- 4×RTX 3090 容量审计依次保留 per-GPU micro-batch `{128,256,512,768}` 的全部成功、OOM、
  failed/aborted 结果；accumulation 均为 1，对应 effective batch
  `{512,1024,2048,3072}`。每个候选处理 24,576 windows（393,216 frames），完成相应数量的
  真实 forward/backward/optimizer updates。候选只有在 loss/关键梯度有限、四卡无外部
  contention，且每卡至少保留 `max(2 GiB, 10%)` 显存 headroom 时才可选；选择满足该条件的
  最大 micro-batch。容量审计 LR 为 `1e-4`、无 warmup，只作容量/吞吐证据。
- 锁定 effective batch 后，seed 42 短预算只比较两个优化候选：A 为 peak LR `1e-4`、
  warmup 786,432 windows；B 为 peak LR `3e-4`、warmup 1,572,864 windows。每个候选固定处理
  3,145,728 windows（50,331,648 frames），其 optimizer updates 由选定 effective batch
  推导；validation 固定 32,768 windows。以预先固定的内部 validation total loss 为主、
  contact/FK 分项为诊断，有限且最低者胜；相等时选较低 LR。官方 test 不参与选择。
- 正式训练 seed 固定为 `42`，不再运行 seed 123/314。固定处理 61,440,000 windows
  （983,040,000 frames），validation cadence 为每 3,072,000 windows，checkpoint cadence
  为每 6,144,000 windows；最终 checkpoint 由固定预算末端 EMA 权重确定，不按官方 test
  cherry-pick。每个 seed 的 optimizer updates、epochs、峰值显存、吞吐、wall time 均记录。
  checkpoint/resume 先在独立 smoke 中实际中断续训验证，正式 run 仍保留相同 resume 能力。
- seed 42 正式 checkpoint 做一次完整 438-sequence×3-window autoregressive native/CHOIS
  评测。主结果报告 seed-42 point estimate，不报告跨 seed SD/Student-t CI；对 438 个 matched
  sequence 做 10,000 次 bootstrap（seed 42）作为 per-sequence 不确定性。
  同时报告 normalization 越界、NaN/Inf、短序列、文本覆盖、contact、hand/human penetration、
  FS、FID、Matching Score、R-Precision@1/2/3 和 Diversity。
- 95% gate 作用于 seed-42 point estimate。higher-is-better 指标须 `>=0.95×baseline`；lower-is-better
  指标须 `<=baseline/0.95`。锁定 baseline 为 Phase 0 的 object/pelvis trajectory error
  `3.037/3.923 cm`、FS `0.3334`、contact P/R/F1 `0.7908/0.7276/0.7273`、human-object
  penetration `2.5893`（ratio `0.1376`）、FID `0.93342`、R-Precision@1/2/3
  `0.17308/0.31010/0.43510`。任一缺失、非有限或无法解释的系统性退化均使 gate 失败。

2026-07-13 容量审计 `p1-hoi-memory-capacity-s42-20260713` 的四个候选均在首个有限 loss 的
backward 后失败、均非 OOM，原因是 trainer 在 AMP 初始 scale 下发现非有限梯度时先行抛错，
使 GradScaler 无机会执行其动态降 scale。该 immutable failed run 保留且不作容量选择。允许的
优化诊断仅限：四个 rank 同步丢弃发生 overflow 的未提交 accumulation group、统一将 scale
减半，不计入 processed windows 或 optimizer update；连续最多允许 16 次 overflow，之后仍
非有限则 fail fast。必须记录累计 overflow 次数与初/终 scale，并在后续有限、非零关键梯度和
真实 optimizer update 后才报告 stable。使用全新 run ID 原样重跑四个已注册候选，不改变
micro-batch、accumulation、预算、LR、headroom 或选择规则。

容量重跑 `p1-hoi-memory-capacity-r2-s42-20260713` 在无 contention 的 4×RTX 3090 上完成。
mb128/256/512/768 的每卡 peak reserved 分别为 1,287,651,328 / 1,855,979,520 /
2,973,761,536 / 4,066,377,728 bytes，吞吐分别为 820.240 / 840.080 / 782.590 /
738.762 windows/s；每个候选均在初始 scale 65,536 下同步跳过 13 个 overflow group、降至
scale 8 后完成全部 48/24/12/8 个真实 update，无 OOM、loss/关键梯度有限。按预注册规则锁定
per-GPU micro-batch 768、accumulation 1、effective batch 3,072，每卡最小 headroom
21,229,666,304 bytes。由此筛选每候选固定 1,024 updates，A/B warmup 分别为 256/512 updates；
正式训练每 seed 固定 20,000 updates，validation/checkpoint cadence 分别为 1,000/2,000 updates。
筛选的 validation 和 checkpoint 均只在固定预算末端执行，即 3,145,728 windows / 第 1,024
个 update；每次 terminal validation 固定覆盖 32,768 windows。

2026-07-14 首次完整 native run `p1-hoi-eval-native-s42-20260714` 在 438 条序列均加载、
第 0 个自回归窗口完成后，于第 1 个窗口的 object-point 坐标变换失败：OMOMO 物体旋转来自
NumPy `float64`，而锁定 BPS 点为 `float32`，混合 dtype 进入 `torch.bmm`。该失败 run 保留且
无 OOM、无部分 CHOIS 输出。允许的表示诊断修复仅在该坐标变换入口将旋转和平移显式对齐到
BPS tensor 的 device/dtype，并新增 mixed-dtype 回归测试；不得改变 checkpoint、数据、模型、
采样步数、序列数或指标。修复提交发布到 worker 后，以新 run ID 原样重跑完整协议。

2026-07-14 Phase 1B seed-42 gate 决定：失败。`p1-hoi-eval-native-r1-s42-20260714`
完成 438 sequences×3 windows，所有指标有限，scene 未加载、文本覆盖 100%、短序列为 0、
人体位置 normalization 越界为 0，物体越界为 4/63,072；但 object/pelvis goal error 为
`32.963/46.163 cm`、FS 为 `0.4817`、contact P/R/F1 为
`0.6476/0.1823/0.2561`。锁定 CHOIS evaluator 得到 FID `32.2125`、R-Precision@1/2/3
`0.0769/0.1587/0.2236`。只有 human-object penetration `1.1209`（ratio `0.0762`）通过
95% 阈值；penetration 的 per-sequence bootstrap 仅覆盖 181 条，是既有 evaluator 明确排除
woodchair/whitechair/largebox/largetable/plasticbox/trashcan 的协议结果，不是缺失 run。
完整 gate 与 seed-42 sequence bootstrap 在
`experiments/results/p1_hoi_phase1b_gate_s42_20260714.json`。不得 merge、创建
`exp/p1b-hoi-v1` 或开始 Phase 1C。后续 Phase 1B session 只允许先预注册表示、坐标、mask、
normalization、数据契约或优化诊断；其他方向不由本次失败决定授权。

筛选在同一 commit `800a9fd1e2ec5fcdad1f05d855609e8960aaafd9` 完成。候选 A/B 的
terminal validation total 为 182.709418 / 166.539836，FK 为 3.615757 / 3.293625；contact
accuracy 为 0.570313 / 0.570430。按预注册规则锁定 B：peak LR `3e-4`、warmup 1,572,864
windows（512 updates）。两者均在用户明确允许且 manifest 已记录的 GPU0 外部 contention 下
完成，选择不使用 throughput 或官方 test。

2026-07-13 用户将全研究实验统计协议修订为 single-seed-42：Phase 1B 及未来所有 phase 的
screening、training、main-table、evaluation 只运行 seed 42，不再运行额外训练 seed，也不报告
跨 seed SD/Student-t CI；仍保留按样本/序列的注册 bootstrap/permutation 不确定性。该修订只
改变重复 seed 数和相应统计汇总，不改变数据、模型、训练预算、指标或 95% gate 阈值。

##### 2026-07-14 Phase 1B 修复重试预注册

首次失败的形态是“teacher-forced validation 持续下降，但三窗口自回归生成全面退化”，因此
不先扩大网络或加入新的推理模块。对当前数据和代码的只读审查给出六项可检验证据：

- OMOMO train 的 `597,868/597,868` 个 HOI 窗口都标记 `need_pelvis_dir=true`，baseline 也启用
  pelvis endpoint condition；当前 HOIPrior dataset 却令 `goals[:3]` 恒为零，sampler 还显式丢弃
  evaluator 传入的 `pelvis_goal`。窗口首末 pelvis 的 XZ 位移均值为 `0.423 m`、中位数
  `0.346 m`、95 分位 `1.062 m`，`98.99%` 大于 1 cm；这不是可忽略的零条件，且直接对应失败的
  `46.163 cm` pelvis goal error。模型已有 9-D goal 输入槽，恢复该条件无需改变 API 或容量。
- 当前 `object_goal` consistency 对每个训练窗口的最后输出帧都施加序列最终目标。确定性抽取
  100,000 个训练窗口时，窗口末端与目标帧的物体位置平均相差 `61.716 cm`、中位数
  `39.151 cm`，只有 `0.718%` 的窗口末端与目标帧重合；该项与窗口内真实轨迹重建形成冲突。
- dataset 将物体旋转表示为相对“当前窗口首帧”参考的矩阵，而现有三窗口 evaluator/sampler
  在后续窗口仍沿用第一窗口参考。按 42/84 source-frame handoff 审查，参考错位的平均旋转
  geodesic 为 `0.995 rad`、95 分位为 `2.800 rad`。
- 训练为每个窗口读取其当前 BPS；推理第 2/3 窗口仍使用第一窗口 BPS。训练快照中相隔 42
  source frames 的 BPS RMS 变化均值为 `0.0876`、95 分位为 `0.2214`，不能视为常量。
- 正式训练只做随机 timestep 的 teacher-forced x0 validation；它既不检查从纯噪声生成，也不
  检查窗口交接。`EMA=0.9999` 在 20,000 updates 后仍含约 `13.53%` 的初始权重贡献，且没有
  与 online/较快 EMA 在内部 rollout 上比较。
- 锁定 baseline 的训练路径以 `loss_w_obj_pts=50` 对变换后的物体表面点做物理空间重建；当前
  HOIPrior 虽已保留相同 rest-object 资产与 FK=50，却只回归归一化的平移/旋转矩阵元素，没有
  物体表面项。这是接口不变、已有资产可复用的几何归纳偏置缺口，不需要扩大网络。

这些证据优先支持“共享状态/条件契约与优化选择错误”，尚不支持模型容量不足。修复保持
8-layer、512-width、16-head、232-D clean-x0、16/2/500、完整 instruction 和 scene-free API
不变；不引入 CFG 双前向、cross-attention 重写、额外 latent、scene/SDF guidance、mixer、
released checkpoint 初始化或更大网络。训练期几何辅助若被触发也不得改变 expert 的输入输出
接口，因此后续 HSIPrior 和 Mixer 不需要 HOI 专用 adapter。

修复重试仍在 `phase/01b-hoi` 完成，按以下顺序设置不可跳过的内部 gate。预计诊断、两项短预算
筛选、一次约 3–4.5 小时正式训练和完整评测可在一个后续 Phase 1B session 可靠完成，因此目前
不拆新 subphase；任一内部 gate 失败即停止，不以扩大模型替代诊断，也不开始 Phase 1C。

1. **D0：既有 checkpoint 的无重训诊断。** 只用 Phase 1A 锁定 train/internal-validation，
   official 438 test 不参与。以
   `SHA256("42:hoi-remediation:" + sequence_name)` 固定选择 128 条可形成连续三窗口的内部
   validation sequence；另在 timestep `{0,1,10,50,100,250,499}` 各取 512 个固定窗口。
   对 terminal checkpoint 的 online、EMA-0.9999 权重分别报告逐 field x0 error、现有非零
   text/BPS/object-goal 的 permutation sensitivity、单窗口 GT-history 生成、三窗口
   GT-history/rebased rollout 和全生成 history rollout；pelvis condition 在既有路径中的 100%
   缺失作为 structural failure 单独报告。必须把第 1/2/3 窗口误差分开，不能再以
   teacher-forced total loss 代理生成。
2. **D1：统一且可组合的 `WindowStateCodec`。** dataset、训练验证、sampler、evaluator 以及未来
   HSI/Mixer 只调用同一套 encode/decode/rebase。GT 的 encode→decode、相邻窗口 GT handoff 和
   global→local→global 最大绝对误差须 `<=1e-5`，旋转 geodesic 须 `<=1e-4 rad`。每个新窗口以
   第一帧 history 的全局物体旋转为参考，使用
   `R_rel(t)=R_global(t) R_ref^T`；decode/handoff 时将 9-D 输出投影至 `SO(3)` 后恢复全局旋转。
   BPS 必须用当前已生成物体 pose、不可变 `rest_object_geo` 和 hash-verified `code/bps.pt` 固定
   basis 按训练同一算法重算；GT replay 与存储 BPS 的最大绝对误差须 `<=1e-4`，不得读取未来
   GT BPS，也不得用第一窗口
   BPS 冒充后续窗口条件。若无法复现该 BPS contract，Phase 1B 在训练前阻塞并重新预注册，
   不静默换成另一种 object feature。
   同一 codec 还须把每个 OMOMO 窗口最后 source frame 的 pelvis XZ endpoint 按 baseline 语义
   变换到当前 window-local frame，按 baseline 保持 raw metres 写入既有 `goals[:3]`（Y 固定为
   0，不套用 position min/max normalization）；`goals[6:9]` 继续承载
   序列最终 object goal。sampler 不再丢弃 `pelvis_goal`，且 GT replay 与 legacy OMOMO loader 的
   XZ 最大绝对误差须 `<=1e-5`。这只填充既有 9-D goal slot，不增加 condition token、模型参数
   或 232-D 输出字段，也不改写 Phase 1A 的已关闭 artifact/gate。
3. **D1 的目标/损失修正。** `object_goal` 仍是序列最终外部目标和模型 condition，但 direct
   consistency 只在 `end_pi == seq_length` 的 terminal window 施加；其他窗口只接受真实轨迹
   reconstruction、FK、object-surface 和 velocity 监督。目标、history、BPS、object rotation
   和 progress 均由 codec 在当前窗口坐标下生成。两个核心候选都恢复 baseline 已有的物体表面重建：将不可变
   rest-object points 以预测/GT 物体 pose 变换到同一物理坐标，对非 history 帧施加
   `object_surface_weight=50`。保持 `fk_weight=50`、`velocity_weight=0.1`、terminal
   `goal_weight=1`，不改变 232-D field reconstruction 权重。
4. **D2：固定 processed-window 的两档核心筛选。** 两个候选都从随机权重开始，seed 42，处理
   `6,144,000` windows（`98,304,000` frames），每 `3,072,000` windows 做一次 32,768-window
   teacher-forced validation，并在预算末端运行 D0 固定 128-sequence internal rollout：

   | 候选 | micro/GPU × 4 | accum | effective batch | peak LR | warmup windows | updates |
   |---|---:|---:|---:|---:|---:|---:|
   | R-1024 | 256 × 4 | 1 | 1,024 | `1e-4` | 1,572,864 | 6,000 |
   | R-3072 | 768 × 4 | 1 | 3,072 | `3e-4` | 1,572,864 | 2,000 |

   容量审计只证明 3,072 可运行，不再自动决定质量最优档位。每个候选在同一次训练中保存
   online、EMA-0.999 和 EMA-0.9999 三种 terminal 权重，不增加训练数据或测试预算。相对既有
   失败 checkpoint 在同一内部 rollout 上，合格候选必须同时满足：object/pelvis goal error
   各 `<=0.70×`，physical contact F1 至少增加 `0.10`，MPJPE 和 FS 均不超过 `1.10×`，所有
   指标有限；matched text/BPS/pelvis-goal/object-goal condition 的误差还必须以 paired bootstrap
   显著低于独立 permutation。若多个权重/档位合格，先最小化 object/pelvis error ratio 的几何
   均值，再最大化 contact F1；2% 内仍相等时选择 effective batch 1,024。official test、CHOIS 和 throughput
   均不得用于该选择。若没有核心候选合格且不满足下述 D2-G 的唯一触发条件，则停止 Phase 1B
   并先做新的 dated amendment，不选择“相对最好但未合格”的配置进入正式训练。
5. **D2-G：唯一的条件式几何 fallback。** 只有 codec/condition gate 已通过、最佳核心候选的
   object/pelvis ratio 均 `<=0.70`，且 finite、MPJPE、FS 和 condition permutation 等其余 D2
   条件全部合格、唯独 contact F1 未达门槛时，才在已选 batch/LR 上追加一个同为 `6,144,000`
   windows 的随机初始化候选。它保持 232-D 输出不变并继承核心候选的 object-surface/FK loss，
   只额外以既有语义中的前两路 GT 手部 contact channel 对应左右手—物体表面距离加入
   `contact_geometry_weight=10`；后两路稀疏 channel 仍只按原 232-D label 重建，不假设其为手部。
   fallback 必须重新通过完整 D2 门槛；若仍未过，停止 Phase 1B 并先做新的 dated amendment，
   不继续堆叠 loss 或模型模块。
6. **D3：正式重训。** 用 D2 锁定的 config 从随机权重重新训练，禁止从筛选 checkpoint 续训。
   seed 仍只为 42；预算仍为 `61,440,000` windows / `983,040,000` frames。若选择 batch 1,024，
   派生为 60,000 updates、warmup 1,536 updates、validation/checkpoint cadence 3,000/6,000
   updates；若选择 batch 3,072，则仍为 20,000、512、1,000/2,000 updates。terminal 权重类型
   由相同 internal rollout 规则锁定，不按 official test 或中间 checkpoint cherry-pick。
7. **D4：一次正式 gate。** 配置和 terminal checkpoint 锁定后才运行一次完整 438-sequence
   native export 与一次 pinned CHOIS；继续使用 seed-42 point estimate、10,000 sequence
   bootstrap 和原 95% gate。若任一 gate 失败，保留全部结果并停止；只有全部通过才写
   `PHASE_1B.md`、merge 和创建 `exp/p1b-hoi-v1`。

稳定运行的交接规则：reportable 训练必须在 detached worker session 中启动。完成 resolved
config/preflight、初始稳定区间、有限 loss/gradient、显存 headroom 和首个可恢复 checkpoint
验证后，不要求 Codex 持续轮询；应报告实测吞吐与剩余时间并结束当前交互，等待用户发送
“继续”后再检查完成状态。该规则只节约监控，不降低 checkpoint、manifest 或失败登记要求。

2026-07-15 Phase 1B 修复重试在 D2 按预注册规则停止。`R-1024` 与 `R-3072` 均从随机
初始化完成各 `6,144,000` windows，训练 loss/关键梯度和两次 teacher-forced validation 均
有限；terminal checkpoint SHA-256 分别为
`d7931a3221c11903a8f9856355a16a493107ed78ad7947120906eece2b22ec23` 与
`48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4`。固定 128-sequence
三窗口 internal rollout 对 online、EMA-0.999、EMA-0.9999 共六个记录评估后无一合格。
最接近的 `R-3072 online` 虽通过 pelvis ratio `0.3412`、contact F1 增量 `0.1241`、FS ratio
`0.3930` 和 finite gate，但 object-goal ratio 为 `10.1958`、MPJPE ratio 为 `1.8086`，且
matched text/BPS 均未显著优于 permutation；因此不是 contact-only failure，D2-G 不得启动。
提交的选择器 run `p1-hoi-d2-selection-s42-20260715` 返回
`stop-no-eligible-candidate`、`selected=null`、`d2_g_allowed=false`。完整六记录、contention、run
与 artifact hash 位于
`experiments/results/p1_hoi_phase1b_remediation_d2_s42_20260715.json`。本次不得运行 D3、D4、
official test 或 CHOIS，不得 merge/tag，也不得开始 Phase 1C；未来任何修复方向必须先新增 dated
plan/registry amendment，且不得提升本次任一不合格 checkpoint。

2026-07-15 Phase 1B D2-P 机制诊断预注册。用户在 D2 负结果收口后要求继续；本 amendment
只授权解释既有 D2 failure，不撤销 `stop-no-eligible-candidate`，也不授权新的训练或 checkpoint
选择。现有证据显示 `R-1024/R-3072 online` 的 object-goal error 在第 1 窗口已经达到
`138.9/145.1 cm`，而非只在第 2/3 窗口 handoff 后增长；两者 pelvis/object-goal permutation
有效，但 text/BPS permutation 不显著。既有训练均有限，且 object-goal 项仅占未加权平均 total
约 `0.09%/0.11%`，不能仅凭 teacher-forced total 或增加预算断言原因。

1. **D2-P0：CPU contract replay。** 使用同一固定 internal-validation 数据，逐项验证 dataset
   goal global→local→normalized、sampler goal normalization、metric global target、history
   encode/decode 和当前 reference/BPS 语义。pelvis/object/history max abs error 必须 `<=1e-5`，BPS
   replay 必须 `<=1e-4`；不得读取 future GT 来生成 condition。任一项失败即分类为
   `coordinate-contract-defect` 并停止 GPU 诊断，不得静默换表示。
2. **D2-P1：固定 teacher-x0 对照。** 只读取 R-1024 与 R-3072 terminal checkpoint 的 online
   weights，使用 D0 已锁定的 512 internal windows、timestep `{0,1,10,50,100,250,499}` 和完全
   配对的 noise，分别报告 terminal/non-terminal fieldwise x0 error，以及 matched 对 text、BPS、
   pelvis、object-goal permutation 的逐样本差与 10,000-replicate paired bootstrap。D0 online
   已归档结果只作诊断参照，不重新选择旧 checkpoint；EMA 不参与本诊断。
3. **D2-P2：单窗口 reverse-chain trace。** 在相同 internal set 的固定 32 sequences 上，仅对
   R-1024/R-3072 online 各运行一次 matched 500-step diffusion，记录 reverse step
   `{499,250,100,50,10,1,0}` 的 clean-x0 fieldwise error、state/output range、首窗口 object/pelvis
   endpoint error 和有限性。不得运行三窗口 rollout、official 438 或 CHOIS，也不得用 throughput
   或本诊断选择 checkpoint。
4. **归因门槛。** P0 失败即 `coordinate-contract-defect`；P0 通过但 t=499 matched
   joint-position/object-translation error 超过归档 D0 online 对应值的 `1.10` 倍，且 text/BPS
   paired response 仍不显著，则为 `high-noise-condition-underfit`；若上述 teacher errors 均在
   `1.10` 倍内但 P2 首窗口相应误差仍超过 D2 eligibility，则为 `reverse-process-exposure-gap`；
   其他组合登记为 `mixed-mechanism`。所有分类只产生 blocker/evidence，不自动授权 loss、sampler、
   model 或训练改动。

唯一 reportable run id 为 `p1-hoi-d2p-mechanism-s42-20260715`，seed 42，必须在四卡 worker
使用其固定 Python、clean exact commit、`INFBAGEL_WORKER_EXPERT=hoi` 和
`tools/experiment.py start/finish/register`；实际 GPU 诊断只使用一张无 contention 卡。resolved
config/preflight 必须先归档。完成后保留全部 negative artifact 并停止；任何修复或再训练仍需新的
dated plan/registry amendment。D2-G、D3、D4、merge/tag 和 Phase 1C 继续禁止。

2026-07-15 D2-P 按 P0 gate 停止。reportable run
`p1-hoi-d2p-mechanism-s42-20260715` 在 clean commit `7a3e024d91471958c4b8ddbf6d368ca63d6bfffe`
完成；live preflight 全部通过，但固定 32 个 internal 首窗口中 5 个 BPS replay 的 max-abs error
超过 `1e-4`，最差 `sub10_tripod_006=0.01226154`。pelvis、object-goal、metric target、history
replay 的最大误差分别为 `0`、`0`、`3.58e-7`、`2.98e-7`，均通过。分类为
`coordinate-contract-defect`；工具在加载 checkpoint 或检查 CUDA 前按门槛返回，candidate count、
training updates 和 GPU forward calls 均为 0。完整失败序列与 artifact hashes 位于
`experiments/results/p1_hoi_phase1b_d2p_mechanism_s42_20260715.json`。不得以较小 RMS 取代明确的
max-abs gate，不得继续 P1/P2、D2-G、D3/D4 或后续 phase。任何 BPS mismatch 调查/修复都必须
再做新的 dated Phase 1B amendment。

2026-07-15 Phase 1B D2-B 作者 BPS backend replay 预注册。用户确认 `code/bps.pt` 来自
InfBaGel 作者发布资产，并指定以 `b9a158f75ab0740c91c9cfc8863a65fa381b014c`（除
`.gitignore`/`requirements.txt` 外为作者原始发布）作为 provenance 基准。只读核验确认当前
`code/bps.pt` 与该基准为同一 Git blob `03b0851af882192913c90d3485559c6c034455ed`，文件
SHA-256 为 `fdff7204b4697e105457cb7e39267b9555bc0d8d854dbc92cd67e2d8c3e77042`；作者 dataset
从 split-local `rest_object_geo` 读取网格，而当前 snapshot 的 split-local 与
`data/object/rest_object_geo` 共 13 个 PLY/NPY 均逐文件同 hash。因此 D2-P0 不是资产来源或
复制路径漂移。对 5 个失败窗口逐点审计还显示，每个窗口只有 `1/1024` 个 basis 点不一致；stored
delta 对应的 PLY 顶点残差均小于 `1.4e-7 m`，CPU KNN 所选顶点与 stored 顶点的平方距离差仅为
`0` 或 `5.96e-8`。本 amendment 只检验这是否为作者生成路径与 CPU/CUDA KNN tie backend 的执行
上下文差异，不改变 BPS 表示、资产、阈值或模型。

1. **D2-B0：固定 backend replay。** 在 D2-P 同一固定 32 条 internal-validation 首窗口上，以
   当前 hash-verified `code/bps.pt`、不可变 PLY、GT 当前帧 object rotation 和完全相同的
   `WindowStateCodec.recompute_bps`，分别报告 worker CPU 与单张无 contention RTX 3090 CUDA 的
   max-abs/RMS、失败窗口数、失败 basis 数、选中顶点和最近距离差。不得读取 future GT，不加载
   checkpoint，training updates 与 model forward calls 均为 0。严格 gate 仍是每个窗口
   max-abs `<=1e-4`；RMS 或几何等距不得替代该 gate。
2. **唯一条件式继续。** 只有 CUDA 对全部 32 窗口严格通过 `1e-4`，且 provenance/hash/公式与
   CPU 完全相同，才分类为 `cpu-knn-tie-backend-artifact`。随后可提交最小诊断修复：D2-P 的
   reportable BPS contract replay 必须在实际 sampler 所用的 worker CUDA backend 上执行；CPU
   自动测试仍验证 basis/asset hash、坐标公式、无 future GT 和最近表面等价性，但不得声称 CPU
   component-wise replay 通过。修复后使用新 run id
   `p1-hoi-d2p2-mechanism-s42-20260715` 从 P0 重新开始；CUDA P0 严格通过后，才按原 D2-P 固定
   checkpoint、512 teacher windows、32 reverse traces 和分类规则执行 P1/P2。该 follow-on 仍是
   diagnostic-only，不选择 checkpoint。
3. **停止条件。** 若 CUDA 任一窗口仍超过 `1e-4`，立即登记 `backend-replay-unresolved` 并停止；
   不得用 stored per-frame BPS、future GT、阈值放宽或内部样本拟合 tie-breaker 修补 sampler。
   任何进一步表示/近邻算法修复必须再做 dated amendment。

D2-B 唯一 reportable run id 为 `p1-hoi-d2b-bps-replay-s42-20260715`，seed 42，使用 worker 固定
Python、clean exact commit、`INFBAGEL_WORKER_EXPERT=hoi`、resolved config/preflight 和
`tools/experiment.py start/finish/register`。D2-G、D3、D4、official 438、CHOIS、merge/tag 及
Phase 1C 继续禁止；无论 D2-B/D2-P2 结果如何，都不自动授权新训练。

2026-07-15 D2-B 按 CUDA gate 停止。reportable run
`p1-hoi-d2b-bps-replay-s42-20260715` 在 clean commit
`667c5f3059058ac8b1cc6eb9f9c321a8bba4e573` 完成，live preflight 全部通过。CPU 复现原
5-window/5-basis failure，max-abs `0.01226154`；RTX 3090 CUDA 只消除其中 3 个，但
`sub13_woodchair_047` 与 `sub14_floorlamp_019` 仍失败，CUDA max-abs 分别为 `0.005186975` 与
`0.000869215`，均高于 `1e-4`。两者 selected/stored 顶点平方距离差仍仅为
`1.49e-8/-5.96e-8`，但按预注册不得以等距或 RMS 代替 component-wise gate，故分类为
`backend-replay-unresolved`、`conditional_continuation_allowed=false`。本 run 未加载 checkpoint，
model forward、training update、official/CHOIS 使用均为 0。完整失败与 artifact hashes 位于
`experiments/results/p1_hoi_phase1b_d2b_bps_backend_s42_20260715.json`。D2-P2 不得运行；不得拟合
tie-breaker、读取 stored per-frame BPS/future GT、放宽阈值或继续任何训练/评测。D2-B amendment
至此耗尽，任何进一步 BPS 算法修复必须再做新的 dated Phase 1B amendment。

2026-07-15 Phase 1B D2-C 可证明 BPS tie 几何等价预注册。用户明确授权：仅当 stored 与
recomputed 最近点可由同一份 hash-verified immutable PLY 证明为等距 mesh 点时，BPS GT replay
可使用几何等价 gate；所有非 tie basis 点继续严格要求 component-wise max-abs `<=1e-4`。本
amendment 不改变 sampler 生成的 BPS 数值、232-D representation、模型、loss 或 checkpoint，且
严禁 sampler 读取 stored per-frame BPS 或 future GT。

1. **锁定大样本。** 仅用 OMOMO `internal_validation` 的 29,382 windows，按 13 个 object class
   分组；每类按
   `SHA256("42:hoi-d2c:" + object_name + ":" + sequence_name + ":" + global_window_index)`
   排序取前 64 个，共 832 windows。每类恰为 64；global-window selection SHA-256 为
   `5f13844f9c3c1540d89d19b304e484cba6e84cc8adcb7276ebf4d17fb803db72`，
   sequence/window SHA-256 为
   `e7827e83b88058e9d87dc4d56ccdbfe5929ea245645356fab278364b1aae1f38`。
   official 438 与 CHOIS 均不得参与。
2. **逐 basis gate。** 对每个 window 的 1,024 个 basis 点，先执行原 component-wise 检查。误差
   `<=1e-4` 记为 strict pass；只有超过该阈值的点才可尝试 tie exception，并必须同时满足：
   (a) stored delta 还原出的最近点到该 object 的锁定 PLY 顶点 residual `<=1e-6 m`；
   (b) recomputed 最近点直接来自同一 PLY，且 mesh residual `<=1e-6 m`；
   (c) 两顶点相对同一 BPS basis 的 squared-distance 绝对差 `<=1e-7 m²`；(d) 全部值有限。
   任一条件失败即 unexplained failure，不得用 RMS、表面近似或扩大 tolerance 替代。13 个 PLY
   SHA-256 与 `code/bps.pt` SHA-256 锁入 training protocol，并在运行时逐文件拒绝 mismatch。
3. **D2-C reportable gate。** 唯一 run id 为
   `p1-hoi-d2c-bps-equivalence-s42-20260715`；在 worker 的 CPU 和一张空闲 RTX 3090 CUDA 上对
   同一 832 windows 执行，CPU/CUDA 均须 0 unexplained failures、13/13 class 覆盖、所有值有限
   才通过。记录 strict/tie 数量、每个 tie 的 object/window/basis/vertex/residual/distance gap。
   不加载 checkpoint，不做 model forward 或 training update。若其他 GPU 有外部轻微占用，必须
   登记且不得 kill；所选 GPU 必须无 compute contention，throughput 不参与 gate。
4. **唯一条件式 D2-P 继续。** 只有 D2-C 全部通过，才允许提交最小诊断改动，并以新 run id
   `p1-hoi-d2p3-mechanism-s42-20260715` 从 P0 重跑。P0 使用上述逐 basis gate；通过后才读取既有
   R-1024 online checkpoint
   `d7931a3221c11903a8f9856355a16a493107ed78ad7947120906eece2b22ec23` 与 R-3072 online
   checkpoint `48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4`，严格复用原
   D2-P 的固定 512 teacher windows、timestep `{0,1,10,50,100,250,499}`、paired permutation/
   bootstrap，以及固定 32-sequence single-window 500-step reverse trace。该 run 只做机制分类，
   不选择/提升 checkpoint，不训练，不运行三窗口 rollout、official 438 或 CHOIS。
5. **停止条件。** D2-C 任一 unexplained failure 即登记并停止，不运行 D2-P3。D2-P3 完成后无论
   分类为何也停止；任何 loss/model/sampler/training 修复仍需新的 dated amendment。D2-G、D3、
   D4、merge/tag、Phase 1C 及后续阶段继续禁止。

所有 reportable workload 均必须使用 worker 固定 Python、clean exact commit、
`INFBAGEL_WORKER_EXPERT=hoi`、resolved config/live preflight 与
`tools/experiment.py start/finish/register`，finish 后由 worker 主动回传 authority staging 并校验
完整 tree hash。

2026-07-15 D2-C 按 CPU fail-fast gate 停止。reportable run
`p1-hoi-d2c-bps-equivalence-s42-20260715` 在 worker clean commit
`ff4e1256638ef1d97e2720901fbfa46e3280ed88` 完成，覆盖 13/13 类、每类 64 个，共 832 个
internal-validation windows 和 851,968 个 basis 点。851,882 个点严格满足 component-wise
`<=1e-4`，另有 69 个点同时满足 hash-verified immutable PLY、双侧 mesh residual `<=1e-6 m`
及 squared-distance gap `<=1e-7 m²`，按授权记为几何等价 tie；仍有 17 个点的 gap 位于
`1.0356937307776093e-7` 至 `2.1901005275992702e-7 m²`，全部超过锁定上限，因此分类为
`geometric-equivalence-unexplained-failure`、`conditional_d2p_authorized=false`。这 17 个点分布于
floorlamp 3、largetable 2、plasticbox 2、smallbox 1、suitcase 5、tripod 2、whitechair 1、
woodchair 1；所有点均有限，且 PLY residual 都低于 `1e-6 m`，但不构成获准 tie。

CPU gate 失败后工具按预注册顺序跳过 CUDA：0 windows、0 basis、0 GPU kernels。preflight 如实
记录 GPU 0 上外部 root Python PID 858478 占用 3,184 MiB；未 kill 外部进程，且未用 throughput
作任何选择。run 未加载 checkpoint、未做 model forward/training update，未使用 official 438 或
CHOIS，sampler 未读取 stored per-frame BPS 或 future GT。完整 17-point blocker 与封存 artifact
hashes 位于 `experiments/results/p1_hoi_phase1b_d2c_bps_equivalence_s42_20260715.json`。D2-P3
不得运行；D2-C amendment 至此耗尽，D2-G、D3、D4、merge/tag、Phase 1C 及后续阶段继续禁止。

2026-07-15 Phase 1B D2-D BPS 数值容差校准预注册。用户在 D2-C 失败闭环后明确允许：若
`1e-7 m²` 对作者提供的 float32 BPS/pose/geometry contract 过严，可作有证据的适度调整。
D2-C 封存的 69 个 accepted ties 与 17 个 failures 共 86 个 component-wise mismatch，经同一份
hash-verified immutable PLY、同一 object pose 与 `bps.pt` 在 float64 下只读复核：17 个旧失败点
的最近线性距离差最大仅 `1.2267085047756865e-7 m`（`0.122671 µm`），其中 15/17 的直接 PLY
平方距离差回落到 `1e-7 m²` 以下；全部 86 点的线性距离差也都不超过 `0.122671 µm`。因此原
`1e-7 m²` 已进入 float32 序列化、旋转/矩阵乘法与近等距消减误差量级，不能稳定地区分这些
immutable-mesh 最近点。

1. **锁定调整且不改非 tie gate。** component-wise max-abs `<=1e-4`、stored/recomputed PLY
   residual `<=1e-6 m`、全有限、runtime BPS/PLY hash rejection 均保持不变。只有超过
   component-wise gate 且通过双侧 PLY residual 的点才可尝试数值等价 exception；平方距离差
   上限调整为 `2.5e-7 m²`（约 1 m² 尺度的两个 float32 epsilon），并新增独立的线性最近距离差
   上限 `2.5e-7 m`（`0.25 µm`）。两个上限必须同时通过；不允许仅用 RMS、component 平均或
   object-level 汇总放行。
2. **不相交 holdout。** D2-D 保留 D2-C 的每类 hash 排名前 64 个 calibration windows，并增加
   每类排名 65--128（zero-based rank 64--127）的 64 个 holdout windows；两组各 832、均覆盖
   13/13 类且无交集，总计 1,664 windows / 1,703,936 basis 点。holdout global-window SHA-256
   为 `750378d6933a6e190ceebfe582b00fac16b403dad853bd9c30f2bfe0b8fdc00a`，sequence/window
   SHA-256 为 `f2d8fbc4ee42150727a981fbca1b7ef45b9b0902cfb64f859d797bc8ce9a944e`；combined hashes
   分别为 `e58bc72326ec4ec193b7e8371c9a034f64d09761f188ee863f1ccd63ef21bf87` 与
   `f629e28cb4b277ad53c3bae4df96726a5224457fbbb1c779b4214de8385392ad`。不得在实现前查看
   holdout gate 结果；official 438 与 CHOIS 不参与。
3. **唯一 reportable gate。** run id 为 `p1-hoi-d2d-bps-tolerance-s42-20260715`。worker CPU 与
   一张无 compute contention 的 RTX 3090 必须分别在 calibration/holdout 上达到 0 unexplained、
   0 nonfinite 和 13/13 class coverage；记录 strict/equivalent/failure 数量、平方/线性 gap 与
   residual。不加载 checkpoint，不做 model forward 或 training update；external process 不得 kill，
   throughput 不用于结论。CPU 任一 failure 时可 fail-fast 跳过 CUDA并直接判负。
4. **唯一条件式 D2-P4。** 只有 D2-D CPU/CUDA 全部通过，才允许提交同一双重等价 gate 的最小
   P0 诊断改动，并以新 run id `p1-hoi-d2p4-mechanism-s42-20260715` 从 P0/P1/P2 重跑。仍只读取
   已封存 R-1024 online checkpoint
   `d7931a3221c11903a8f9856355a16a493107ed78ad7947120906eece2b22ec23` 与 R-3072 online
   checkpoint `48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4`，复用原固定
   512 teacher windows、timesteps、paired permutation/bootstrap 及 32-sequence 500-step reverse
   trace。只做机制分类，不选择 checkpoint、不训练、不运行三窗口 rollout、official 或 CHOIS。
5. **停止条件。** D2-D 任一 unexplained failure 即登记并停止，不运行 D2-P4。D2-P4 完成后也
   停止；D2-G、D3、D4、merge/tag、Phase 1C 及后续阶段继续禁止，任何 sampler/loss/model 或
   training 变更仍需新的 dated amendment。

2026-07-15 D2-D 按不相交 holdout gate 停止。reportable run
`p1-hoi-d2d-bps-tolerance-s42-20260715` 在 worker clean commit
`e51a128346c4ec1be46ac48483ec3b73eccdb3d4` 完成。CPU calibration 的 851,968 个 basis 点中，
851,882 strict、86 dual-tolerance equivalent、0 unexplained、0 nonfinite，确认原 D2-C 的 17 个
失败确实来自过严的 `1e-7 m²` 数值界限。独立 holdout 的 851,968 个点中，851,854 strict、
113 equivalent、1 unexplained、0 nonfinite；唯一失败为 `sub11_clothesstand_071`、global window
57370、basis 53。其 component max-abs 为 `0.00205436`，stored/recomputed PLY residual 分别为
`2.15037e-7/3.33200e-8 m`，线性距离差 `1.91293e-7 m` 通过 `2.5e-7 m` cap，但平方距离差
`3.36442e-7 m²` 超过锁定的 `2.5e-7 m²` cap。

由于 calibration/holdout CPU conjunction 为 false，CUDA 按预注册 fail-fast 跳过：0 windows、
0 basis、0 GPU kernels。selected GPU 2 在 preflight 为 15 MiB/0%；GPU 0 的外部 PID 858478
占 3,486 MiB，未 kill，throughput 未用于结论。run 未加载 checkpoint、未做 model forward 或
training update，未使用 official/CHOIS，sampler 未读取 stored per-frame BPS 或 future GT。
完整结果与 hashes 位于
`experiments/results/p1_hoi_phase1b_d2d_bps_tolerance_s42_20260715.json`。D2-P4 不得运行；
不得针对已揭示的 holdout failure 继续事后调阈值。若未来用 scale-invariant linear gate 替代固定
squared-distance gate，必须有新的明确 dated amendment 与全新未查看 holdout。D2-G、D3、D4、
merge/tag、Phase 1C 及后续阶段继续禁止。

2026-07-15 Phase 1B D2-E BPS 线性几何等价预注册。用户在 D2-D 失败闭环后明确要求继续，因而
授权此前指出的下一项允许动作：用与基准距离尺度无关的线性最近距离差替代固定平方距离 gate，
并使用全新未查看 holdout。D2-D 的唯一 holdout failure 在线性差 `0.191293 µm` 上通过既定
`0.25 µm` cap，却因相同几何距离在平方域的尺度放大而失败；继续扩大固定 squared threshold
会重复同一尺度依赖。本 amendment 因此不再用 squared-distance gap 作 accept/reject，但仍逐点
报告该值以保留审计可见性。

1. **唯一 gate。** 非 tie component-wise max-abs `<=1e-4` 保持不变。只有超过该阈值的点才可
   尝试等价 exception，并必须同时满足：stored 与 recomputed 点均可回放到同一份 hash-verified
   immutable PLY；双侧 mesh residual 均 `<=1e-6 m`；两点到同一 BPS basis 的最近距离绝对差
   `<=2.5e-7 m`（`0.25 µm`，即 mesh residual cap 的四分之一）；全部有限。squared-distance gap
   仅报告，不设 acceptance threshold。不得使用 RMS、component 平均、object 汇总、stored
   per-frame BPS 或 future GT 放行 sampler。
2. **新鲜 holdout。** 已揭示的 hash rank 0--127 共 1,664 windows 作为 disclosed calibration；
   每类 rank 128--191 再取 64 个全新 internal-validation windows，共 832 fresh holdout，与此前
   1,664 个无交集并覆盖 13/13 类。fresh global-window SHA-256 为
   `44fdc7154902c922310f54ad2eb97d26ca710902d5c9d76c354e50f650e28316`，sequence/window
   SHA-256 为 `70301049201e3c945570f73103d6313f166f592b615908001b41fcc513016b91`；2,496-window
   combined hashes 分别为 `bdf93ebf796baa345f163194aa13b1720e1410d4b1c62c19a29c3a88ee40dc69`
   与 `258254cf9e277b7e7f40f8fed087a74c128ecfaf7c6279bd47df662066b42e3c`。实现提交前不得查看
   fresh holdout gate 结果；official 438 与 CHOIS 不参与。
3. **唯一 reportable gate。** run id 为
   `p1-hoi-d2e-bps-linear-equivalence-s42-20260715`。worker CPU 与一张无 compute contention 的
   RTX 3090 必须分别在 disclosed calibration/fresh holdout 达到 0 unexplained、0 nonfinite、
   13/13 class coverage；记录 strict/equivalent/failure、mesh residual、linear 与 squared gap。
   不加载 checkpoint、不做 model forward/training update；external process 不得 kill，throughput
   不参与结论。CPU 任一 failure 可 fail-fast 跳过 CUDA并直接判负。
4. **唯一条件式 D2-P5。** 只有 D2-E CPU/CUDA 全部通过，才允许提交相同线性 gate 的最小 P0
   诊断改动，并以新 run id `p1-hoi-d2p5-mechanism-s42-20260715` 重跑 P0/P1/P2。只读取既有
   R-1024 online checkpoint
   `d7931a3221c11903a8f9856355a16a493107ed78ad7947120906eece2b22ec23` 与 R-3072 online
   checkpoint `48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4`，复用固定 512
   teacher windows、timesteps、paired permutation/bootstrap 与 32-sequence 500-step reverse
   trace。只做机制分类，不选择 checkpoint、不训练、不运行三窗口 rollout、official 或 CHOIS。
5. **停止条件。** D2-E 任一 failure 即登记并停止，不运行 D2-P5，且不得继续迭代阈值。D2-P5
   完成后也停止；D2-G、D3、D4、merge/tag、Phase 1C 及后续阶段继续禁止。

2026-07-15 Phase 1B D2-E operational retry amendment。首次 reportable run
`p1-hoi-d2e-bps-linear-equivalence-s42-20260715` 在构造 dataset、读取 fresh holdout 或发起任何
CUDA kernel 之前被 resolved-config 一致性检查拒绝：归档配置为物理 `cuda:2`，detached command
误用了经 `CUDA_VISIBLE_DEVICES` 重映射的 `cuda:0`。该 run 已以 `aborted` 封存；CPU/CUDA
windows、checkpoint、model forward、training update 均为 0，故不是科学 gate failure，也未揭示
rank 128--191 fresh holdout 的任何结果。worker/authority staging tree SHA-256 为
`9899ca7551806d2b2d3d28e7a43d59ea2945c680b50b4d34dd6391822af5fa05`。

1. 原 run id 永不复用。只允许一次新 run
   `p1-hoi-d2e-bps-linear-equivalence-r1-s42-20260715`，保持同一 2,496-window selection/hashes、
   `0.25 µm` linear gate、严格 component/PLY/finite gates、CPU-before-CUDA 顺序及全部禁止项。
2. 唯一操作修正是 resolved config 与 runtime 都直接使用物理 `cuda:2`，不设置
   `CUDA_VISIBLE_DEVICES`，并在 start 前再次断言 GPU2 无 compute process、显存占用 `<=128 MiB`。
   不改阈值、不换样本、不加载 checkpoint、不训练。
3. r1 任一科学或操作 failure 都登记并停止，不再重试。只有 r1 CPU/CUDA 对 disclosed/fresh
   subsets 全部通过，原 D2-P5 条件授权才生效；其余 D2-G、D3、D4、official/CHOIS、merge/tag、
   Phase 1C 与后续阶段仍禁止。

D2-E r1 已在 commit `a4e8a82f78fbdbd6c3c2bbc440da79c9ae1089b4` 通过。CPU/CUDA 对 disclosed
1,664 windows 与 fresh 832 windows 均覆盖 13/13 类、0 unexplained、0 nonfinite；CPU 的
equivalent 点数为 200/115，CUDA 为 192/105。四个子集的最大线性距离差分别为
`0.232105/0.224496/0.225945/0.228633 µm`，均低于锁定的 `0.25 µm`；report-only 最大平方差
为 `3.973856e-7 m²`，双侧最大 PLY residual 为 `2.747639e-7/3.578024e-8 m`。worker GPU2
preflight 为 15 MiB/0%，GPU0 外部 PID 858478 未 kill；未加载 checkpoint、未做 model forward
或 training，未使用 official/CHOIS。sealed artifact tree SHA-256 为
`9845eb3e030918126fad01f3d9dffeccdbb316200bafc83fc3086edf2c71e8cf`，精简结果见
`experiments/results/p1_hoi_phase1b_d2e_bps_linear_equivalence_s42_20260715.json`。D2-P5 条件授权
现已生效；只能提交 P0 contract replay 的最小线性等价改动并运行既定 P0/P1/P2。

D2-P5 已在 commit `1c209ebd9347cd1127015bc4393f9c9469eb6ef9` 完成，run id 为
`p1-hoi-d2p5-mechanism-s42-20260715`。P0 的固定 32 windows 通过全部 coordinate checks；
32,768 个 BPS basis 为 32,765 strict、3 linear-equivalent、0 unexplained/nonfinite，最大 accepted
linear gap `0.0726132 µm`。P1 中两个 checkpoint 在全部 7 timesteps 上，matched 均显著优于
text/BPS/pelvis/object-goal 四种独立 permutation（56/56 bootstrap 95% CI 下界大于 0），排除
“完全未使用条件”。但 t=499 joint-position MSE 相对 D0 reference 的 ratio 为 R-1024 `2.2773`、
R-3072 `2.7653`，而 object-translation ratio 为 `0.3800/0.4699`，不是统一 high-noise underfit。
P2 reverse trace 全部有限、history max-abs `3.5763e-7`；两者 pelvis error `12.570/11.226 cm`
通过 `31.710 cm` threshold，但 object goal error `143.546/148.109 cm` 与 MPJPE
`52.250/60.047 cm` 均失败。因此预注册分类为 `mixed-mechanism`，不得用本诊断选择 checkpoint。

run 未训练、未使用 official/CHOIS、sampler 未读取 stored per-frame BPS/future GT。GPU3 preflight
15 MiB/0%，GPU0 外部 PID 858478 未 kill；sealed artifact tree SHA-256 为
`21775b2b7b6b69365178bcafac9e7da6eee614503babf9f988fdfe653004e623`，精简结果见
`experiments/results/p1_hoi_phase1b_d2p5_mechanism_s42_20260715.json`。D2-P5 是本 amendment 的
终点；Phase 1B gate 仍未通过。不得运行 D2-G/D3/D4、official/CHOIS、merge/tag、Phase 1C 或
后续阶段。任何后续 Phase 1B 方向必须重新 dated preregister。

2026-07-15 Phase 1B D2-F reverse-manifold stabilization 与条件式学习曲线扩展预注册。用户在
D2-P5 关闭后明确授权继续解决 HOIPrior 质量问题。D2-P5 已排除 codec/BPS contract 缺陷和
“完全忽略条件”，但暴露 inference-specific mismatch：物体表面 loss 与最终 decode 都先把
9-D 物体旋转投影到 SO(3)，500-step posterior mean 却直接使用 raw model x0；固定 reverse
trace 的 raw object-rotation element absolute maximum 在 t=0 达到 R-1024 `2.8414`、R-3072
`3.2170`，不可能是合法旋转矩阵。与此同时，R-1024 teacher validation total 在
3,072,000 到 6,144,000 windows 从 `72.1366` 降到 `14.0741`，说明原 10% screening budget
结束时尚未形成平台期。D2-F 不放宽最终质量 gate、不改变 232-D API、模型容量、训练 loss、
数据、diffusion schedule 或 condition；只检验并在通过时固定 reverse-step manifold closure，
然后才允许一个更长但仍属于 screening 的 R-1024 learning curve。

1. **D2-F0：paired reverse-manifold diagnostic。** 唯一 reportable run id 为
   `p1-hoi-d2f-so3-reverse-s42-20260715`。只读取 D2-P5 相同的 R-1024/R-3072 online
   checkpoint 与固定 32 internal-validation sequences；每个 checkpoint 使用完全相同的初始
   Gaussian sample 和逐 step posterior noise，成对运行 `control` 和 `object_so3_x0`。后者仅在
   每个 reverse step 的 model x0 已生成、fixed history 已恢复之后，将预测帧的 channels
   `[219:228]` 投影到 SO(3)，再进入 posterior mean；不 clamp 其他 channel、不改 posterior
   variance、不改变 BPS/rebase。逐个 `{499,250,100,50,10,1,0}` 登记 raw/projected
   orthogonality residual、determinant error、fieldwise x0 error、range、endpoint metrics，且
   sampler 仍不得读取 stored per-frame BPS 或 future GT。
2. **D2-F0 gates。** 两条 paired path 均须 finite 且 history max-abs `<=1e-5`；SO(3) path 的
   predicted-frame maximum `||R^T R-I||_F <=1e-5`、`|det(R)-1| <=1e-5`。只有至少一个
   checkpoint 同时满足既有单窗口 thresholds：object goal `<=8.0087890691 cm`、pelvis goal
   `<=31.7099441657 cm`、MPJPE `<=36.8360687256 cm`，才允许 D2-F1。若没有 checkpoint
   达到绝对 gate，但 R-1024 的 paired SO(3)/control object-goal ratio `<=0.50`、MPJPE ratio
   `<=1.02`、pelvis ratio `<=1.05` 且上述 finite/history/manifold checks 全通过，则唯一分类为
   `sampler-mechanism-positive-training-insufficient` 并允许 D2-F2；否则登记并停止。
3. **D2-F1：固定三窗口 rollout。** 唯一 reportable run id 为
   `p1-hoi-d2f-so3-rollout-s42-20260715`。只评估 D2-F0 达到绝对 gate 的 checkpoint，复用既有
   128-sequence、三窗口 generated-history selection、seed 42、paired condition permutation 与
   10,000 bootstrap；唯一 sampler 差异为已通过的逐 step object SO(3) projection。完整沿用 D2
   object/pelvis/contact/MPJPE/FS/finite/condition gates 和选择顺序。若有 eligible checkpoint，
   只锁定 sampler 与 batch 配置并停止本次 D2-F session；不得启动 D3。若无 eligible candidate，
   只有 R-1024 同时满足上一条 mechanism-positive trigger 时才允许 D2-F2，否则停止。
4. **D2-F2：唯一条件式 R-1024 learning curve。** 唯一训练 run id 为
   `p1-hoi-d2f-r1024-curve-s42-20260715`；必须从 seed-42 随机权重开始，不能 resume screening
   checkpoint。保持 micro/GPU 256×4、accumulation 1、effective batch 1024、LR `1e-4`、warmup
   1,572,864 windows/1,536 updates、现有 loss weights、500-step diffusion 与 232-D 模型；总预算
   18,432,000 windows、294,912,000 frames、18,000 updates，在 6,144,000、12,288,000 和
   18,432,000 保存 online/EMA-0.999/EMA-0.9999。每个 milestone 只先用 online weights 运行同一
   32-sequence SO(3) reverse gate；只有达到绝对单窗口 gate 的 milestone 才运行一次固定 128×3
   full D2 rollout。首个 full-rollout eligible milestone 可锁定配置并终止后续 screening；否则在
   18,432,000 后停止。该学习曲线是 screening，不是 D3，不能以 loss 下降代替 rollout gate。
5. **全局停止与治理。** 所有 GPU workload 仅能在四卡 worker、使用 lifecycle 与全新 run id；
   resolved config 必须先归档，worker checkout 在 reportable run 中不可改变，artifact 仍由 worker
   主动回收。D2-F 任一步 failure/aborted/OOM 都 finish/register。不得调整上述阈值，不得增加
   support clamp、contact loss、高噪声 weighting、模型容量或新 condition。D2-G、D3、D4、
   official/CHOIS、merge/tag、Phase 1C 及后续阶段在 D2-F 内全部禁止。

D2-F0 已在 commit `9f671b1e0b5e3f637e9b3de5f196516224ef78d2` 完成，run id 为
`p1-hoi-d2f-so3-reverse-s42-20260715`。逐 reverse step 投影正确将 applied object rotation 的
最大 orthogonality/determinant residual 压到 R-1024 `9.3107e-7/7.7486e-7`、R-3072
`9.5878e-7/7.7486e-7`，两条路径均 finite，history max-abs 均为 `3.5763e-7`。但 paired
projection 没有修复生成：R-1024 object/pelvis/MPJPE 从 control
`142.955/12.624/52.650 cm` 变为 `146.486/10.793/55.155 cm`，对应 ratio
`1.0247/0.8550/1.0476`；R-3072 从 `146.363/12.225/59.831 cm` 变为
`150.434/12.523/63.210 cm`，对应 ratio `1.0278/1.0244/1.0565`。没有 checkpoint 达到绝对
object/MPJPE gate，R-1024 也远未达到 object ratio `<=0.50` 的条件式学习曲线 trigger；预注册
分类因此为 `sampler-mechanism-negative-stop`，D2-F1 与 D2-F2 均未获授权。

该 run 未训练、未选择 checkpoint、未使用 official/CHOIS，sampler 未读取 stored per-frame BPS
或 future GT。GPU3 preflight 15 MiB/0%，GPU0 外部 PID 858478 未 kill，throughput 未参与任何
判断；sealed artifact tree SHA-256 为
`23c5f8c0283c8c8ae20e42592e3cfeb07a3d0634601c5fe7112a5ca3f2f08726`，精简结果见
`experiments/results/p1_hoi_phase1b_d2f_reverse_manifold_s42_20260715.json`。D2-F 在此负向关闭，
Phase 1B gate 仍未通过；不得运行 D2-G/D3/D4、official/CHOIS、merge/tag、Phase 1C 或后续阶段。
任何后续 Phase 1B 方向必须重新 dated preregister。

2026-07-15 Phase 1B D2-H paired reverse-state exposure remediation 预注册。用户确认作者提供的
InfBaGel checkpoint 已经过一致性蒸馏；该 provenance 事实消除了 checkpoint 训练阶段的不确定性，
但 D2-H 不预设“一致性蒸馏缺失”是当前失败主因，也不复刻作者蒸馏、CFG、scene occupancy 或
guidance。作者实现与当前 HOIPrior 在 condition token 路由、timestep 注入、positional encoding、
Transformer norm/FFN、goal normalization/injection、surface loss 类型和 reverse-state training
coverage 上均存在差异；这些实现差异保持为竞争解释。D2-H 只检验其中一个最小、可证伪机制：
当前纯 teacher `q(x_t|x0)` 训练是否不能纠正模型自己的 reverse-state 偏移。任一 gate 失败都只登记
负结果，不自动授权改 condition、模型、loss、蒸馏或预算。

1. **D2-H0：paired reverse-state exposure diagnostic。** 唯一 reportable run id 为
   `p1-hoi-d2h-exposure-paired-s42-20260715`。只加载两个 sealed online checkpoint：R-1024
   `d7931a3221c11903a8f9856355a16a493107ed78ad7947120906eece2b22ec23` 与 R-3072
   `48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4`；不得加载 EMA 或
   released InfBaGel checkpoint。固定 internal-validation 512-window selection SHA-256
   `9d3f8cc4647018fdf285481ffef95df6eb3c4e6f6ad0b680f85e23b1edeebd71`、seed 42、timesteps
   `[0,1,10,50,100,250,498]` 与 10,000 次 paired bootstrap。对每个 target timestep `t` 先令
   parent `s=t+1` 并用固定 `x0`、condition 与 parent noise 构造 `x_s~q(x_s|x0)`；control 用真实
   `x0` 走一次标准 posterior，intervention 用同一 checkpoint 在 `x_s` 上预测的 `x0` 走同一
   posterior，两条 path 共用 posterior noise。随后在同一 `t` 再调用同一 checkpoint，完整报告
   joint position、joint rotation、object translation、object rotation、contact 的 per-sample MSE、
   state displacement、物理误差和四种 condition permutation。GT 只允许构造 diagnostic oracle
   posterior，不得进入 production sampler condition；sampler 仍不得读取 stored per-frame BPS 或
   future GT。两条 path 均恢复 immutable 两帧 history，不做 SO(3) projection、support clamp、CFG
   或 condition 改写。
2. **D2-H0 mechanism gate。** 两个 checkpoint 均须 finite，history 与 posterior-formula replay
   max-abs `<=1e-5`。在低 timestep `{0,1,10,50,100}` 中，每个 checkpoint 至少 4/5 个 timestep
   的 paired-bootstrap 95% CI 下界须同时证明 joint-position 与 object-translation 的
   `model-parent MSE - oracle-parent MSE > 0`；五个低 timestep 的 mean-ratio 几何均值须分别达到
   joint position `>=1.5`、object translation `>=2.0`。必须报告全部 checkpoint、timestep、field，
   不得按有利 subset 选择。任一 checkpoint 或必需 field 失败即登记 negative 并停止；不得训练。
3. **D2-H implementation-parity measurements。** D2-H0 同时封存不参与 gate 的描述性证据：当前
   HOIPrior 相对 `b9a158f75ab0740c91c9cfc8863a65fa381b014c` 的 condition/timestep token 路由、goal
   数值尺度、各 field clean/noise RMS、每项 loss 的 parameter-gradient norm/cosine 及按 timestep
   分解。该附录用于区分 reverse-state exposure 与实现细节竞争解释；它不得在观察结果后授权新的
   condition/loss/architecture intervention。
4. **D2-H1：唯一条件式 one-step detached reverse-state augmentation。** 仅当 D2-H0 两个
   checkpoint 均通过时，先运行 `p1-hoi-d2h-reverse-aug-smoke-s42-20260715`；smoke 的 posterior
   parity、detachment、history、固定 row split、RNG、`t=499` fallback、required-gradient、finite 与
   resource headroom 全部通过后，才允许唯一训练 run
   `p1-hoi-d2h-r1024-reverse-aug-s42-20260715`。模型保持随机初始化、232-D、16 frames/2 history、
   8×512×16、500-step x0 diffusion、scene-free OMOMO、原 condition API、原 sampler 与原 loss
   weights。前 1,572,864 processed windows（1536 updates）保持原 teacher-q warmup；之后每个
   micro-batch 以固定 row index 分成 50% teacher `q(x_t|x0)` 与 50% one-step exposed rows。
   exposed row 从 `q(x_{t+1}|x0)` 出发，由当前 online model 在 `eval()`/`no_grad()` 下预测 parent
   `x0`，用未投影的原标准 posterior 到 `t`，detach 并恢复 history 后再进行唯一有梯度 forward；
   target 仍为真实 `x0`。`t=499` 回退 teacher-q。不得增加 latent、自条件输入、CFG、scene、
   SO(3) projection、support clamp 或 parent-path gradient。
5. **D2-H1 locked budget and checkpoint use。** 训练只在四卡 worker 上使用 seed 42、4×RTX 3090、
   micro-batch/GPU 256、accumulation 1、effective batch 1024、AdamW、peak LR `1e-4`、原
   1,572,864-window warmup/cosine schedule、6,144,000 processed windows、98,304,000 processed
   frames、6000 updates。sealed R-1024 online checkpoint 只作为固定 control，绝不初始化、resume
   或继续训练 intervention。online weights 是唯一 primary candidate；`0.999/0.9999` EMA 与 online
   validation 同步记录但只作描述。OOM、nonfinite、history/formula/gradient/headroom failure 均按
   negative finish/register 并停止，不得观察后改 batch、accumulation、LR、warmup 或预算。
6. **D2-H1 paired reverse gate。** 唯一 run id 为
   `p1-hoi-d2h-r1024-reverse-gate-s42-20260715`。先在相同 512-window D2-H0 diagnostic 上要求
   joint-position 与 object-translation 的 low-timestep `model-parent - oracle-parent` gap 相对 sealed
   R-1024 均缩小至少 50%。再复用 32-sequence selection SHA-256
   `7f16d1b8f4f3843639d10d0ecd367d1e2073b8b55bb03f4fef9895c960b85663`，以 paired initial/posterior
   noise 比较 500-step single-window control/intervention；intervention/control object-goal ratio
   `<=0.70`、MPJPE ratio `<=0.90`、pelvis-goal ratio `<=1.05`，且不放宽既有 absolute gate：
   object goal `<=8.0087890691 cm`、pelvis goal `<=31.7099441657 cm`、MPJPE
   `<=36.8360687256 cm`、history max-abs `<=1e-5`、全部 finite。任一失败即停止。
7. **D2-H2：条件式固定三窗口 rollout。** 只有上一条全部通过才允许唯一 run id
   `p1-hoi-d2h-r1024-rollout-s42-20260715`，复用 128-sequence×3-window selection SHA-256
   `a6688c81c9295743a924afde35a4f920322cb0a84dcc821c658b3f8f812c99a0` 和既有 D2 gate：相对
   D0 的 object/pelvis ratio 均 `<=0.70`、contact F1 increase `>=0.10`、MPJPE/foot-sliding ratio
   均 `<=1.10`、四种 condition permutation 全部显著、history/finite 通过。不得使用 official 438、
   CHOIS、throughput 或 favorable subset 做选择。即使通过也只产生 internal candidate；不得在
   D2-H 内启动 D3/D4、official/CHOIS、merge/tag 或 Phase 1C。
8. **治理与停止条件。** 所有逻辑变更先在 authority 提交，worker 只主动 fetch 精确 committed
   object；所有 reportable run 使用 `tools/experiment.py start/finish/register` 并封存同一执行上下文
   的 preflight/resolved config/hardware。D2-H 任一 failure/aborted/OOM 都保留并登记；无自动 fallback。
   released checkpoint 初始化、D2-G、dense nonterminal object-goal loss、BPS tolerance 修改、继续
   SO(3) remediation、模型/representation/loss 改写、额外预算、official/CHOIS、D3/D4、Phase 1C+
   、Mixer、LLM state machine、merge 与 tag 全部禁止，除非另有新的 dated amendment 和用户授权。

D2-H0 已在 implementation commit `d612cc1f44cf15c4abfa92580e52c1cb2ef6b8a2` 完成，run id 为
`p1-hoi-d2h-exposure-paired-s42-20260715`。两个 checkpoint 的 finite、immutable history 与共享
production-posterior formula replay 均通过，history/formula max-abs 都为 `0`；低 timestep
`{0,1,10,50,100}` 的 joint-position 与 object-translation paired-bootstrap 95% CI 下界在 R-1024
和 R-3072 都达到 5/5 同时为正。但预注册效应量 gate 失败：R-1024 的低 timestep mean-ratio
几何均值仅为 joint `1.084273`、object translation `1.294629`，R-3072 仅为 joint `1.074639`、
object translation `1.318594`，分别低于 `1.5/2.0`。因此分类为
`reverse-state-exposure-negative-stop`，D2-H1 的条件式授权前提未满足，smoke/training 均未启动。

所有 2 checkpoint × 7 timestep × matched/四种 condition permutation × 5 representation fields、
state displacement 与 6 个 physical metrics 均完整且 finite；paired parent-q/posterior noise 跨路径及
checkpoint hash 一致。broadcasting、batch indexing、history/terminal mask、normalization inversion、
posterior coefficient sharing 与 detach 检查全部通过。描述性 implementation-parity appendix 同时封存
当前/作者 token routing 对照、token/representation 数值尺度以及逐 timestep loss-gradient norm/cosine，
但不授权任何 condition/loss/model intervention。worker artifact tree SHA-256 为
`aafe8e3800a1819cd009a65072a50e7da71389c1263958f2d55a4264224c8924`，精简结果 SHA-256 为
`2b9e1af050070d1340007891cba7be03bdf639d04603bd4f35040b705bba687d`，见
`experiments/results/p1_hoi_phase1b_d2h_exposure_paired_s42_20260715.json`。Phase 1B gate 仍未通过；
本 session 在 D2-H0 停止，任何 D2-H1 或其他新 intervention 都必须等待用户再次确认并按治理要求处理。

2026-07-15 Phase 1B D2-I weighted-objective gradient routing diagnostic 预注册。用户在审阅
D2-H0 负结果后明确授权一个新的、基于 implementation-parity appendix 的无训练诊断；该授权不
撤销 D2-H1 的 failed prerequisite，也不允许训练或观察后修改模型/condition/loss。D2-P5 已在两个
sealed checkpoint、全部七个 timestep 上证明 text/BPS/pelvis/object-goal 四种 permutation 均显著
差于 matched condition；D2-H0 又显示 reverse-state exposure penalty 方向稳定但效应量 gate 失败。
因此本诊断不重复 condition-presence 或 exposure 测试，而检验 appendix 中尚未独立复核的优化机制：
在已训练 checkpoint 上，锁定的 `50×FK + 50×object-surface + 0.1×velocity + terminal-goal`
是否使共享参数的 total-gradient 长期由弱对齐的 auxiliary-gradient 主导。D2-H0 的四样本描述性
appendix 中，total/reconstruction parameter-gradient norm ratio 已达到 R-1024 `35.0--62.3`、
R-3072 `53.3--137.6`；该观察只用于提出假设，D2-I gate 使用与 D2-H0 不相交的 fresh holdout。

1. **D2-I0 唯一 reportable run。** run id 固定为
   `p1-hoi-d2i-gradient-dominance-s42-20260715`，只在四卡 HOI worker 加载 sealed online R-1024
   `d7931a3221c11903a8f9856355a16a493107ed78ad7947120906eece2b22ec23` 和 R-3072
   `48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4`；EMA 和 released
   checkpoint 禁止。primary cohort 为 D0 稳定 window hash ordering 的 ranks `[512,640)`，共
   128 个 nonterminal internal-validation windows，和 D2-H0 ranks `[0,512)` 完全不相交；selection
   SHA-256 为 `cefbee34d09cf7db3015e7dc1aacb2d17259608ae27f466c4cc7a11f3c1714c3`。
   terminal-goal 只作描述的 cohort 为排除 D2-H0 selection 后按
   `SHA256("42:d2i-terminal-fresh:"+sequence_name+":"+pi)` 排序的前 64 个 terminal windows，
   SHA-256 为 `43acfcbcbfd6755e2bd66a991b5314805eee7a246a5b9783ec596f7a95c7fc21`。
2. **固定梯度测量。** seed 42，timesteps `[0,1,10,50,100,250,499]`，每个 cohort 按 selection
   顺序组成不可重排的 16-window blocks，q-noise 由 run id/checkpoint-independent stable seed
   生成；两个 checkpoint 必须复用相同 noise。模型保持 `eval()`，用 `torch.autograd.grad` 计算但
   不写 `.grad`、不创建 optimizer、不调用 step。逐 block 报告以下 loss 的 all-parameter 与
   time/text/BPS/goal-progress/motion-input/transformer/output parameter-group L2 norm 和完整 cosine
   matrix：human reconstruction、object reconstruction、contact、全部 field reconstruction、
   `50×FK`、`50×object-surface`、`0.1×velocity`、terminal goal、auxiliary sum 与 total。报告
   loss values、gradient cancellation index、各 group 对 total squared-norm 的描述性占比，以及
   terminal/nonterminal 分层；不得只报告有利 timestep、checkpoint、loss 或 parameter group。
3. **D2-I0 gate。** gate 只用 primary nonterminal cohort 的 high-noise `{250,499}` 和
   all-parameter gradient。对两个 checkpoint、两个 timestep 都要求：全部 loss/gradient finite；
   model state-dict 前后 SHA-256 完全相同；直接 total gradient 与按锁定权重重放的 component-sum
   gradient relative L2 error `<=1e-5`；8 个 block 的 total/reconstruction norm-ratio 几何均值
   `>=20` 且 10,000 次 paired/block bootstrap（seed 42）95% CI 下界 `>=10`；
   cosine(total,reconstruction) 的 bootstrap 95% CI 上界 `<=0.25`。全部通过才分类为
   `weighted-objective-gradient-dominance-positive-stop`，否则为
   `weighted-objective-gradient-dominance-negative-stop`。terminal goal、逐 component 和逐 parameter
   group 结果均为描述性证据，不得替代 primary gate。
4. **治理与停止。** 实现只能添加 diagnostic/helper/tests/docs，并验证 selection disjointness、stable
   RNG、block pairing、weighted-gradient formula replay、parameter immutability、finite/all-loss/all-group
   reporting 和 zero optimizer updates；不得改变 production model、232-D representation、loss function/
   weights、condition API、diffusion/sampler 或 checkpoint。worker 必须使用 lifecycle、resolved config、
   preflight 与 persistent session，并主动回收 immutable artifact。无论结果正负，本 session 都只登记
   D2-I0 并停止；positive 也不授权 loss-weight sweep、重训、D2-H1、architecture/condition intervention、
   official/CHOIS、D3/D4、Phase 1C、merge 或 tag。任何新动作须用户再次确认并另行 dated amendment。

D2-I0 已在 workload commit `45a5165d6e0c09a50d84ffd8ce2d4bd6c4bb4b45` 完成，run id 为
`p1-hoi-d2i-gradient-dominance-s42-20260715`。两个 checkpoint 在 high-noise `{250,499}` 的
total/reconstruction all-parameter gradient-norm ratio 都远超 magnitude gate：R-1024 几何均值
`90.561/111.423`、bootstrap 95% CI `68.286--119.767 / 86.637--142.027`；R-3072 为
`112.173/125.644`、CI `83.040--149.982 / 89.797--166.585`。但预注册要求的是 dominance 与
持续弱对齐的合取；cosine mean/95% CI 分别为 R-1024 t250
`0.2217 [0.1552,0.2973]`、t499 `0.2350 [0.1756,0.3177]`，R-3072 t250
`0.2205 [0.1295,0.3200]`、t499 `0.2471 [0.1776,0.3098]`，四个 CI 上界都高于 `0.25`。
因此分类为 `weighted-objective-gradient-dominance-negative-stop`；不能把“大范数”单独提升为
mechanism-positive，也不授权修改 loss weight。

fresh primary/terminal selections 与 D2-H0 完全不相交，两个 checkpoint 的 block noise hashes
完全相同；全部 2 checkpoint × 2 cohort × 7 timestep × 10 loss × 8 parameter-group 记录 finite，
direct-total/component formula replay 全局最大 relative L2 为 `4.8854e-7`，state-dict 前后 hash
逐 checkpoint 完全相同，`.grad` buffers 为空，optimizer/update 均为 0。terminal cohort 描述性
结果显示 total/reconstruction ratio 仍为 `71.9--150.3`，但 terminal-goal gradient norm 仅约
`0.069--0.169`，相对 `50×FK` 的约 `59.9--141.4` 很小；该结果不参与 gate。tmux wrapper
将退出值 0 与格式字面量写成原始 `0n` token；该文件未覆盖，metrics/manifest/registry 与 completed
lifecycle 均完整保留。

worker artifact tree SHA-256 为
`f49cd8acaa95517d59cfcffce7b0fbcc98a02d9b2b9dd223714292f516aec7b6`，精简结果 SHA-256 为
`2bf63007253c72669619c5d9d688ab8530fc07a1648326d1e96c8d3664ac1ade`，见
`experiments/results/p1_hoi_phase1b_d2i_gradient_dominance_s42_20260715.json`。本 session 在 D2-I0
负结果处停止；D2-H1、loss-weight sweep、重训和其他 condition/architecture/sampler intervention
均未启动，任何后续方向等待用户再次确认和新的 dated amendment。

2026-07-16 Phase 1B D2-J global-gradient-clipping routing diagnostic 预注册。用户在审阅
D2-I0 的负 gate 后再次明确要求继续推进；该授权只允许新的无训练机制诊断，不恢复 D2-H1，也不
允许观察后自动修改 loss、gradient clipping、model、condition 或 sampler。D2-I0 已在 fresh holdout
上证明 high-noise total/reconstruction gradient-norm ratio 很大，但其 aggregate reconstruction cosine
gate 未通过；完整 appendix 同时显示 total-gradient 与 human reconstruction 的 mean cosine 仅约
`0.029--0.089`，与 object reconstruction 则约 `0.290--0.340`。正式 HOIPrior 训练又在 AMP
unscale 后、每个 optimizer step 前固定执行 global `clip_grad_norm_(..., 1.0)`。因此 D2-J0 不重新
裁定 D2-I0，而独立检验：在真实裁剪公式下，巨大 auxiliary-gradient 是否使固定的一单位全局更新
预算稳定偏离 human reconstruction、同时保留更高的 object-reconstruction 一阶方向效率。

1. **D2-J0 唯一 reportable run。** workload 真实日期固定为 2026-07-16，run id 为
   `p1-hoi-d2j-clip-routing-s42-20260716`。只在四卡 HOI worker 加载 sealed online R-1024
   `d7931a3221c11903a8f9856355a16a493107ed78ad7947120906eece2b22ec23` 和 R-3072
   `48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4`；EMA 和 released
   checkpoint 禁止。primary cohort 为 D0 stable window ordering ranks `[640,768)`，共 128 个
   nonterminal internal-validation windows、8 个不可重排的 16-window blocks；selection SHA-256
   为 `a75012dda01cfd59c413bb622f4d867ffb6c2c48cf5d9dcfba4fe800e172432a`，并必须与 D2-H0
   `[0,512)`、D2-I0 `[512,640)` 都不相交。
2. **固定测量。** seed 42，timesteps `[0,1,10,50,100,250,499]`，q-noise 使用 run-id-based
   stable seed，两个 checkpoint 复用逐 block 完全相同的 noise。模型保持 `eval()`，只用
   `torch.autograd.grad`，不写 model `.grad`、不创建 optimizer、不调用 backward/step。逐 block
   完整报告 joint-position、joint-rotation、object-translation、object-rotation、contact 五个 field
   reconstruction gradient，以及 human/object/reconstruction aggregate、`50×FK`、
   `50×object-surface`、`0.1×velocity`、terminal-goal、auxiliary sum 与 total；报告同 D2-I0 的八个
   parameter groups。对 all-parameter total gradient 按 production `max_norm=1.0` 公式报告 pre-clip
   norm、clip coefficient、post-clip norm，并用 PyTorch `clip_grad_norm_` synthetic replay 验证公式；
   human/object directional efficiency 定义为 clipped-total update 与对应 unweighted reconstruction
   gradient 的 cosine（正比例裁剪不改变其数值）。不得省略 timestep、checkpoint、field 或 group。
3. **D2-J0 mechanism gate。** gate 只使用 fresh primary cohort 的 `{250,499}`。两个 checkpoint、
   两个 timestep 均须：全部 finite；state-dict 前后 SHA-256 完全相同；model `.grad` buffers 为空；
   direct-total/component-sum gradient relative L2 `<=1e-5`；production clipping formula/synthetic replay
   max abs `<=1e-6`；10,000 次 paired block bootstrap（seed 42）证明 pre-clip total-gradient norm 的
   95% CI 下界 `>=50`、clip coefficient 的 CI 上界 `<=0.02`、human directional-efficiency cosine 的
   CI 上界 `<=0.15`，并以 object reconstruction 作预注册负对照，其 cosine CI 下界 `>=0.15`。
   全部合取通过分类为 `gradient-clip-routing-positive-stop`，任一失败分类为
   `gradient-clip-routing-negative-stop`。五个 field 与逐 parameter-group 结果为必报描述证据，不得
   替代 gate 或用 favorable subset 改写结论。
4. **治理与停止。** 实现只能添加 diagnostic/helper/tests/docs，必须测试 selection disjointness、
   stable paired RNG、locked clip norm、production-formula parity、field completeness、parameter/state
   immutability、finite checks 和 zero optimizer updates；不得改变 production training call、loss/
   weights、232-D representation、model/condition/diffusion/sampler 或 checkpoint。worker 必须走完整
   lifecycle、resolved config、preflight、persistent session 和 immutable artifact 回收。无论结果正负，
   本 session 都在 D2-J0 登记后停止；positive 也不授权 clip ablation、loss sweep、重训、D2-H1、
   official/CHOIS、D3/D4、Phase 1C、merge 或 tag，后续 intervention 必须再次获得用户确认并另行
   dated amendment。

D2-J0 已在 workload commit `702fc54b0551f79f42958593c7c464f05100599c` 完成，run id 为
`p1-hoi-d2j-clip-routing-s42-20260716`。两个 checkpoint 的 `{250,499}` pre-clip total-gradient
norm mean/95% CI 分别为 R-1024 `103.525 [85.078,125.597] / 126.110
[106.748,148.512]`、R-3072 `134.646 [113.997,156.408] / 132.067
[111.216,151.357]`；对应 clip coefficients 都约 `0.0066--0.0120`，裁剪饱和 gate 全部通过。
object reconstruction direction-efficiency 的四个 CI 下界均高于 `0.15`，R-1024 t250 与 R-3072
t250/t499 的 human CI 上界也低于 `0.15`。但 R-1024 t499 human mean/95% CI 为
`0.1287 [0.0759,0.1943]`，上界未达到预注册 `<=0.15`，故合取 gate 分类为
`gradient-clip-routing-negative-stop`；不得将其余 3/4 cell 或强裁剪事实单独提升为 mechanism-positive。

完整 2 checkpoint × 7 timestep × 8 block × 14 component × 8 parameter-group 记录 finite，五个
representation field 均完整报告。all-parameter 描述证据显示 high-t joint-rotation/contact cosine
接近零，而 object-translation cosine 为约 `0.562--0.654`；output group 的 high-t human cosine
仅约 `0.037--0.051`，object cosine 约 `0.327--0.417`，其余 condition/transformer groups 亦保留在
artifact 中但不参与 gate。paired noise hashes 完全一致，direct/component gradient replay 最大
relative L2 为 `3.768e-7`，clipping replay max abs 为 `0`，state-dict 前后 hash 一致，`.grad`
buffers 为空，optimizer/update 为 0。首次 preflight invocation 因传入错误的 worker CHOIS root 在
manifest/start 前退出且未产生 preflight 文件；随后使用已 pinned 的真实 root 生成唯一 preflight 并
通过全部检查，未覆盖任何 artifact。GPU0 既有外部进程按协议记录且未干预，workload 隔离在空闲
GPU3。

worker/authority artifact tree SHA-256 同为
`62508057b794e135e9d58b29af3d2b8a7a754fb03327da980e4a5cf8eabaca8e`，精简结果 SHA-256 为
`761703af20bced005b26c8e3650107088b84be9496bc52eda720f36994a7efa5`，见
`experiments/results/p1_hoi_phase1b_d2j_clip_routing_s42_20260716.json`。本 session 在 D2-J0
负结果处停止；clip/loss intervention、D2-H1、smoke、训练、official/CHOIS 与后续阶段均未启动。

2026-07-16 Phase 1B D2-K sealed-AdamW counterfactual update-routing diagnostic 预注册。用户在
D2-J0 negative-stop 后再次明确确认继续；该确认仅允许新的 zero-update 机制诊断，不授权 clip/loss
修改、D2-H1 或训练。D2-J0 已证明 formal training 的 global clip norm `1.0` 在 fresh holdout 上
稳定饱和，并观察到 human/object direction efficiency 分化，但其四格合取 gate 因 R-1024 t499
human CI 上界失败而为 negative。两个 sealed terminal checkpoint 的只读 metadata 进一步确认：
online model 同时保存 119 组完整 AdamW `exp_avg/exp_avg_sq`；R-1024/R-3072 分别位于 optimizer
step `6000/2000`，stored LR `1e-5/3e-5`，且历史一阶矩全局 norm 相对下一次 clipped-current
numerator contribution 约为 `1.106/0.796`。因此 D2-K0 不重设 D2-J 阈值，而检验实际训练优化器的
历史状态和逐坐标二阶预条件是否会恢复 D2-J 中较弱的 human reconstruction 路由。

1. **D2-K0 唯一 reportable run。** run id 固定为
   `p1-hoi-d2k-adamw-routing-s42-20260716`，只在四卡 HOI worker 读取 sealed online R-1024
   `d7931a3221c11903a8f9856355a16a493107ed78ad7947120906eece2b22ec23` 和 R-3072
   `48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4`；EMA 与 released
   checkpoint 禁止。fresh primary cohort 从 D0 stable global ordering rank 768 起扫描，跳过 terminal
   ranks `768,770` 后取前 128 个 nonterminal windows，即实际 ranks `769--897` 内的 128 项；selection
   SHA-256 为 `747c0b1c881e150a8ccdb8675044a877b1ab32f615169ea9e3577dcff0a3f90a`，必须与
   D2-H0 `[0,512)`、D2-I0 `[512,640)`、D2-J0 ranks `640--767` 全部不相交。固定 8 个不可重排的
   16-window blocks、seed 42、timesteps `[0,1,10,50,100,250,499]`、10,000 次 paired block
   bootstrap seed 42；两个 checkpoint 逐 block/timestep 使用 checkpoint-independent stable q-noise。
2. **精确 counterfactual。** 模型保持 `eval()`；使用 `torch.autograd.grad` 重放当前锁定 objective
   的五个 field、human/object/reconstruction aggregate、`50×FK`、`50×object-surface`、
   `0.1×velocity`、terminal goal、auxiliary sum 与 total。只把 total gradient 按 production max norm
   `1.0` 缩放，再从 checkpoint 读取但不修改 AdamW step、`exp_avg`、`exp_avg_sq`、betas、eps、
   weight decay 和 stored LR，按 PyTorch AdamW 下一步公式构造 gradient-like descent direction：
   bias-corrected historical-moment contribution、current clipped-gradient contribution 与 decoupled
   weight-decay contribution之和。逐 block 完整报告 raw-clipped 与 full-AdamW direction 对每个 field/
   aggregate 的 cosine、两者 paired difference、full direction 与 current clipped total 的 cosine，以及
   historical/current/decay 三项在 D2-I/J 同一八个 parameter groups 的 norm/cosine/decomposition；不得
   只报告有利 checkpoint、timestep、field、block 或 group。
3. **D2-K0 rescue gate。** gate 只使用 fresh primary 的 `{250,499}`，并将“optimizer state 恢复
   human routing”定义为严格合取。两个 checkpoint、两个 timestep 都须：全部 finite；model 与 optimizer
   state SHA-256 前后完全相同；model `.grad` buffers 为空；checkpoint optimizer parameter order、step、
   hyperparameters 与 registered training contract 精确一致；direct-total/component gradient replay、
   clip replay、AdamW historical+current+decay decomposition relative L2 均 `<=1e-5`。对 paired blocks，
   full-AdamW human efficiency 减 raw-clipped human efficiency 的 bootstrap 95% CI 下界须 `>=0.05`，
   full-AdamW human efficiency CI 下界须 `>=0.15`，并要求 full-AdamW object efficiency CI 下界
   `>=0.15` 作为 no-object-harm 对照。全部通过分类为 `adamw-human-routing-rescue-positive-stop`，任一
   失败分类为 `adamw-human-routing-rescue-negative-stop`。optimizer/current/history/decay 的其他尺度与
   groupwise 结果均为描述证据，不得替代 gate。
4. **治理与停止。** workload 不得创建 `torch.optim` optimizer、调用 backward/step、写 parameter
   `.grad`、修改/保存 checkpoint 或生成训练状态；counterfactual tensor 计算不得写回 model/moments。
   实现只能添加 diagnostic/helper/tests/docs，且须测试真实 AdamW small-tensor parity、bias correction、
   weight decay、missing-gradient semantics、parameter-order rejection、stable selection/RNG、全 field/group
   reporting、formula replay、state immutability 与 zero updates。不得改变 production model/training/loss/
   representation/condition/diffusion/sampler。worker 必须走 resolved config、live preflight、start/finish/
   register、persistent session 和 immutable recovery。无论 positive/negative，本 session 仅登记 D2-K0
   并停止；不授权 optimizer reset/ablation、clip/loss intervention、D2-H1、smoke、训练、official/CHOIS、
   D3/D4、Phase 1C、merge 或 tag，任何下一动作再次等待用户确认和新的 dated amendment。

D2-K0 已在 workload commit `9afd8f7ad0ff02539a228bbb50af765d05fca5f9` 完成，run id 为
`p1-hoi-d2k-adamw-routing-s42-20260716`。四个 preregistered high-noise cell 均未通过 rescue gate：
R-1024 t250/t499 的 full-AdamW minus raw-clipped human efficiency mean/95% CI 分别为
`0.0003 [-0.0213,0.0272] / 0.0031 [-0.0211,0.0313]`，full human 为
`0.0281 [0.0216,0.0333] / 0.0598 [0.0388,0.0811]`，full object 为
`0.1174 [0.0993,0.1338] / 0.1499 [0.1123,0.1860]`。R-3072 对应三组结果为
`0.0089 [-0.0081,0.0299] / 0.0154 [-0.0085,0.0413]`、
`0.0143 [0.0099,0.0191] / 0.0368 [0.0263,0.0466]` 和
`0.1006 [0.0829,0.1219] / 0.1476 [0.1227,0.1670]`。因此所有 human improvement CI 下界均未达到
`0.05`，full human/object CI 下界亦均未达到 `0.15`，严格分类为
`adamw-human-routing-rescue-negative-stop`；不得将任一正的 point estimate 提升为 rescue 证据。

完整 2 checkpoint × 7 timestep × 8 block × 14 loss-component × 5 direction × 8 parameter-group
记录 finite，paired q-noise hashes 完全一致。R-1024/R-3072 的 direct/component gradient replay 最大
relative L2 为 `3.159e-7/2.836e-7`，clip replay max abs 均为 `0`，AdamW
historical+current+decay decomposition 最大 relative L2 为 `5.365e-8/5.294e-8`。两个 checkpoint
均精确覆盖 119 个 optimizer states，step/next-step 为 `6000/6001` 与 `2000/2001`；model、raw
optimizer 和 device-mapped moment 的前后 SHA-256 完全不变，`.grad` buffers 为空，optimizer 未创建且
update 为 0。small-tensor test 与 PyTorch AdamW 的下一步参数差分达到数值一致，未发现 bias correction、
second-moment denominator、weight decay、missing-gradient、parameter order 或 detach 的实现 defect。

描述证据显示 full-AdamW 与 raw clipped total 的 all-parameter cosine 在四格仅约 `0.080--0.130`，与
current preconditioned contribution 为 `0.781--0.948`、与 historical contribution 为
`0.265--0.558`；historical contribution 对 total/object 的 cosine 接近零或为负，因此没有提供稳定的
human rescue。weight-decay direction norm 约 `1.585`，相对 full direction 的 `1447--2016` 很小。
groupwise full-AdamW 结果虽在 time/BPS/goal-progress 等组有较高局部 human/object cosine，但 output
group 的 high-t human/object cosine 仅约 `0.020--0.030 / 0.042--0.061`。这些是描述性定位证据，不能
自动授权 optimizer、model、condition、loss 或 output-head intervention。

worker/authority artifact tree SHA-256 同为
`2ba1284abc3d1b3ed62a56b03496ac134e932f64ddaf1fd281c91fd66b377a79`，精简结果 SHA-256 为
`35f3debde5a0469964ec6ca534dfb91dcf15e980446eca4c49b3519f357b0bde`，见
`experiments/results/p1_hoi_phase1b_d2k_adamw_routing_s42_20260716.json`。workload 在物理 GPU3
执行，GPU0 的既有外部进程仅记录且未干预；运行 `625.42s`，peak allocated/reserved 为
`3007745024/3087007744` bytes。本 session 在 D2-K0 negative-stop 处停止；D2-H1、smoke、训练、
optimizer/clip/loss/model/condition/sampler intervention、official/CHOIS 与后续阶段均未启动。

2026-07-16 Phase 1B D2-L fixed gradient-balanced auxiliary counterfactual 预注册。用户在 D2-K0
negative-stop 后明确授权继续一个最后的 zero-update loss-routing diagnostic；该授权不允许观察
D2-L cohort 后选择权重、执行 weight sweep、修改 production loss、启动 smoke/D2-H1 或训练。
D2-I 的 sealed artifact 已在 D2-L fresh cohort 选择前提供唯一权重推导依据：对 R-1024/R-3072、
`{250,499}`、每格 8 blocks 的 32 个 all-parameter records，令每 block reconstruction-field target
为 `sqrt(||g_human|| × ||g_object||)`，其几何均值固定为 `0.6279429736100133`。将 D2-I 中
`50×FK` 与 `50×object-surface` 的 norm 除以 50 得 raw norm，32-block 几何均值分别为
`1.7589570087469566/1.3158017183675033`，故唯一 counterfactual weights 固定为
`FK=0.3569973401779424`、`object_surface=0.4772322188400037`。推导源 metrics SHA-256 为
`910998c54487cb127343e783773d3dbf13d24b359caf0442695f066bc271bf56`；velocity `0.1`、terminal
goal `1.0` 与五个 reconstruction fields `1.0` 保持不变。

1. **D2-L0 唯一 reportable run。** run id 固定为
   `p1-hoi-d2l-aux-balance-s42-20260716`，只在四卡 HOI worker 读取 sealed online R-1024
   `d7931a3221c11903a8f9856355a16a493107ed78ad7947120906eece2b22ec23` 和 R-3072
   `48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4`；EMA 与 released
   checkpoint 禁止。fresh cohort 固定为 D0 stable global ordering ranks `898--1025` 的 128 个
   nonterminal windows，无需跳过 terminal，selection SHA-256 为
   `b5faa79316c6bd7aa9df0687a2554d458a459bd331c94648a99380d5c3b43a75`，与 D2-H/I/J/K 的
   ranks `0--897` 全部不相交。固定 8 个不可重排的 16-window blocks、seed 42、timesteps
   `[0,1,10,50,100,250,499]`、10,000 次 paired block bootstrap seed 42；current 与 balanced
   candidate、两个 checkpoint 必须逐 block/timestep 使用完全相同的 sample、condition 与 stable
   q-noise。
2. **唯一 counterfactual 与完整报告。** 模型保持 `eval()`；使用 `torch.autograd.grad` 分别得到五个
   reconstruction field、raw FK、raw object-surface、raw velocity、terminal goal 的 gradients。
   current candidate 必须精确重放 production weights `50/50/0.1/1`；balanced candidate 只将
   FK/surface 换成上述 locked weights，其他 tensor、mask、normalization、model、condition 与 losses
   不变。两者分别按 production global clip norm `1.0` 得 raw-clipped direction，并使用相同 sealed
   AdamW step/moments/hyperparameters 构造 exact next full-AdamW direction；不得创建 optimizer、
   写回 moments 或参数。完整报告 current/balanced clipped 与 full-AdamW 对五个 fields、human/object/
   reconstruction、四个 auxiliary、auxiliary sum、total 的 norm/cosine、paired difference，以及
   historical/current/decay contribution 和 D2-I/J/K 同一八个 parameter groups；不得只报告有利
   checkpoint、timestep、field、block、candidate 或 group。
3. **D2-L0 mechanism gate。** gate 只使用 `{250,499}`，并要求两个 checkpoint、两个 timestep 的
   四格全部通过。每格须全部 finite；model、raw optimizer、mapped moments 前后 SHA-256 不变；
   `.grad` buffers 为空；119-state optimizer contract 精确；current direct/component formula、
   balanced direct/component formula、两种 clip formula、两种 AdamW decomposition relative L2
   均 `<=1e-5`。paired-bootstrap 95% CI 必须同时满足：balanced clipped human efficiency 减 current
   clipped human efficiency 的下界 `>=0.10`；balanced clipped human/object efficiency 下界均
   `>=0.15`；balanced full-AdamW human efficiency 减 current full-AdamW human efficiency 的下界
   `>=0.10`；balanced full-AdamW human/object efficiency 下界均 `>=0.15`。全部通过分类为
   `gradient-balanced-auxiliary-routing-positive-stop`，任一失败分类为
   `gradient-balanced-auxiliary-routing-negative-stop`。其他 timestep、component、direction 与 group
   只作描述证据，不得替代 gate。
4. **治理与停止。** 实现只能添加 D2-L diagnostic/helper/tests/docs；须测试 locked-weight provenance、
   current production replay、balanced formula、paired candidate noise、exact clipping/AdamW parity、
   selection/RNG、全 field/candidate/direction/group reporting、state immutability、missing-gradient 与
   zero updates。不得修改 production loss weights/API、model、232-D representation、condition、
   diffusion、sampler、checkpoint 或训练配置。worker 必须走 exact committed object、clean worktree、
   resolved config、live preflight、`start/finish/register`、persistent session 和 immutable recovery。
   无论 positive/negative，本 session 只登记 D2-L0 并停止；positive 也只说明 fixed-weight smoke 的
   条件式前提已达到，仍不授权 smoke、D2-H1 或训练。negative 则停止继续添加 Phase 1B remediation
   mechanism；是否撰写 closure summary 或进入其他阶段须等待用户再次确认。

D2-L0 已在 workload commit `05aa3cd4e48a13d60bd29e372b623b2dae108d83` 完成，run id 为
`p1-hoi-d2l-aux-balance-s42-20260716`，严格分类为
`gradient-balanced-auxiliary-routing-negative-stop`。固定 balanced weights 对 raw-clipped path
产生强且四格一致的 human routing 改善：R-1024 t250/t499 的 balanced-minus-current human
mean/95% CI 为 `0.3766 [0.3399,0.4129] / 0.3340 [0.3057,0.3591]`，balanced human 为
`0.4183 [0.3805,0.4551] / 0.4225 [0.3890,0.4562]`，balanced object 为
`0.5824 [0.5240,0.6487] / 0.6188 [0.5662,0.6718]`。R-3072 对应三组结果为
`0.3783 [0.3296,0.4289] / 0.3382 [0.2993,0.3765]`、
`0.4096 [0.3710,0.4498] / 0.3887 [0.3510,0.4263]` 和
`0.5986 [0.5171,0.6704] / 0.6131 [0.5472,0.6711]`；raw-clipped gate 的全部 registered checks
均通过。

但 exact sealed-AdamW path 的四格 human checks 全部失败。R-1024 t250/t499 的
balanced-minus-current AdamW human mean/95% CI 为
`0.0480 [0.0441,0.0519] / 0.0332 [0.0254,0.0407]`，balanced AdamW human 为
`0.0705 [0.0643,0.0772] / 0.1107 [0.0926,0.1322]`；R-3072 对应为
`0.0365 [0.0343,0.0388] / 0.0281 [0.0246,0.0312]` 和
`0.0505 [0.0477,0.0544] / 0.0669 [0.0602,0.0731]`。所有 improvement CI 下界未达到
`0.10`，所有 balanced-human CI 下界未达到 `0.15`。balanced AdamW object CI 下界仍为
`0.2180/0.2535/0.1650/0.1913`，四格均通过 object preservation，但不能替代 failed human
conjunct。

机制分层因此是：D2-I-derived fixed reweighting 将 current objective 的 high-t preclip norm 从约
`121--140` 降至约 `1.49--1.77`，clip coefficient 从约 `0.0074--0.0090` 提至
`0.575--0.684`，并在不牺牲 object routing 的情况下显著旋转 raw-clipped direction toward human。
这是“`50×FK/50×surface` 主导即时 total-gradient direction”的强 paired 证据。然而，把该新
gradient 输入旧 checkpoint 的 sealed AdamW second-moment geometry 后，full direction 与 raw-clipped
direction 的 cosine 仅约 `0.182--0.284`；full direction 虽与其 preconditioned current contribution
cosine 约 `0.930--0.983`，human 增益仍被压缩。这表明旧 optimizer state/per-coordinate
preconditioning 不能作为 fixed reweighting 的一步代理；它不证明从 fresh optimizer state 训练该
objective 会成功或失败，但预注册 gate 不允许据此启动 smoke。

完整 2 checkpoint × 7 timestep × 8 block × 2 candidate × 14 loss-component × 5 direction ×
8 parameter-group 记录 finite，paired q-noise hashes 完全一致，D2-I source metrics hash 和 32-record
weight derivation 精确重放。production total value replay max abs 为 `2.384e-7`；R-1024/R-3072
current/balanced gradient replay 最大 relative L2 不超过 `3.493e-7/3.580e-7`，clip replay 均为
`0`，AdamW decomposition 不超过 `5.354e-8/5.248e-8`。119-state optimizer contract 精确，
model/raw optimizer/mapped moments 前后 hashes 一致，`.grad` buffers 为空，optimizer/update 为 0，
production loss 未修改。groupwise raw-clipped human improvement 在 output group 四格约
`0.361--0.408`，而 sealed-AdamW output improvement 仅约 `0.061--0.070`；完整其他 groups 已封存，
均不自动授权 intervention。

worker/authority artifact tree SHA-256 同为
`1dc7789284773aca5605cda3057a4d445a0cbcbd6a2d8a72702a17e0c3783fac`，精简结果 SHA-256 为
`760c76da50d45daa1b99eb47bd274f4f8f83f83a7a22f50643a8ee56f98124da`，见
`experiments/results/p1_hoi_phase1b_d2l_aux_balance_s42_20260716.json`。workload 在物理 GPU3
执行，GPU0 既有外部进程只记录且未干预；运行 `1028.69s`，peak allocated/reserved 为
`3950515712/4005560320` bytes。按预注册停止规则，Phase 1B remediation mechanism 在 D2-L0
negative-stop 后不再自动延伸；D2-H1、smoke、训练、production loss/model/condition/sampler 修改、
official/CHOIS 与后续阶段均未启动，等待用户确认 closure 或新的 dated direction。

2026-07-16 Phase 1B D2-M paired fresh-optimizer balanced-objective smoke 预注册。用户在完整审阅
D2-L0 negative-stop 与其 raw-clipped positive subgate 后，明确授权继续新的 dated direction。
D2-M0 只检验一个受控因果问题：在相同自有 R-3072 online 权重、相同训练样本与 q-noise 下，
丢弃旧 AdamW/EMA/scheduler/scaler/RNG state 后，D2-I-derived locked balanced objective 是否比
原 `50/50/0.1/1` objective 产生可测的 high-noise teacher-forced 与 native rollout 改善。该
授权不允许从 released checkpoint 初始化、不允许 weight sweep、architecture/representation/
condition/sampler intervention，也不授权完整重训或 official/CHOIS。

1. **D2-M0 唯一 reportable run。** run id 固定为
   `p1-hoi-d2m-reset-paired-s42-20260716`，四卡 HOI worker 只读取 R-3072 sealed online
   checkpoint `48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4`。
   选择 R-3072 是因为它在既有、结果观察前锁定的 D2 selector 中是唯一记录的 closest record，
   不是 D2-M 结果后的 subset selection。source 只提供 `model` 权重；不得读取或恢复其 optimizer、
   EMA、scheduler、scaler 或 rank RNG，不得使用其 EMA，也不得读取 released InfBaGel
   checkpoint。两个 candidate 固定为 `current` 与 `balanced`，必须从逐 tensor 完全相同的 source
   online model state 开始，使用 seed 42、完全相同的 distributed sampler order、timesteps 与
   q-noise。current weights 为 FK/object-surface/velocity/terminal-goal=`50/50/0.1/1`；
   balanced weights 为
   `0.3569973401779424/0.4772322188400037/0.1/1`，来源仍唯一锁定为 D2-I metrics
   `910998c54487cb127343e783773d3dbf13d24b359caf0442695f066bc271bf56`，不得 sweep。
2. **短程训练合同。** 每个 candidate 固定 4×RTX 3090、micro-batch/GPU `768`、accumulation
   `1`、effective batch `3072`、64 个真实 AdamW updates，即 `196,608` processed windows /
   `3,145,728` frames。AdamW 固定 betas `(0.9,0.999)`、eps `1e-8`、weight decay `0.01`、
   global clip norm `1.0`、AMP 与既有 overflow handling；LR 固定为 source terminal stored LR
   `3e-5`，warmup `0`、minimum LR ratio `1.0`，不作 LR candidate。optimizer 在第一步前必须为
   empty state，terminal 必须恰有 119 states 且每个 step 为 64。EMA `0.999/0.9999` 只从 source
   online 权重重新 copy 并随训练维护，以保持 checkpoint/resume schema；它们不得进入 D2-M
   evaluation 或 selection。两个 candidate 均只评估 terminal online weights，不做 checkpoint
   selection。完整保留 source/initial/terminal model hashes、optimizer-state counts、训练 RNG audit、
   gradient/AMP/finite、loss、memory、throughput 与 terminal checkpoint hashes。
3. **fresh holdout 与 paired evaluation。** teacher cohort 固定为 D0 stable global window ordering
   ranks `1026--1537` 的 512 个 internal-validation windows，含 5 个 terminal windows，selection
   SHA-256 为 `836781bbcdc3a5960631c7af635eaca62bb53f8a67093312fa761eb140174259`；
   它与 D2-H/I/J/K/L 的 ranks `0--1025` 不相交。固定 timesteps `{250,499}`，source/current/
   balanced 使用相同 sample、condition 与 stable q-noise，报告五个 representation fields、physical
   object/pelvis/MPJPE metrics 及 text/BPS/pelvis/object-goal 四种 condition permutation。native
   cohort 固定为既有 three-window eligible sequence ordering ranks `128--159` 的 32 个此前未用
   sequences，96 个 global window indices 的 SHA-256 为
   `30524c88481f6cb81e8063073d510ad01543be92d91eb4ef9b2b8a376cc4fbae`；
   source/current/balanced 的 matched 三窗口 rollout 使用完全相同 sampler noise，完整报告
   object goal、pelvis goal、MPJPE、object/pelvis translation、object rotation、contact 与 foot
   sliding。bootstrap 固定 10,000、seed 42，teacher 以 window、native 以 sequence 为 paired unit。
4. **D2-M0 gate。** 所有 candidate/run/evaluation 必须 finite；source file/model hashes 精确；
   两个 initial model hashes 相同；旧 optimizer/EMA/scheduler/scaler/RNG load counts 均为 0；
   initial optimizer states 为 0，terminal optimizer states/steps 为 `119/64`；训练 RNG audit
   完全相同；history max abs `<=1e-5`；worker exact commit、clean worktree、data/normalization/BPS
   hashes 与 resolved-config/preflight checks 全部通过。teacher `{250,499}` 两格都须满足：
   `current joint-position MSE - balanced joint-position MSE` 的 paired-bootstrap 95% CI 下界
   `>0`；balanced/current object-translation mean-ratio 的 95% CI 上界 `<=1.05`；balanced/source
   joint-position mean-ratio 的两格几何均值 `<=0.98`，且每格 ratio `<=1.02`；balanced/source
   object-translation mean-ratio 的两格几何均值 `<=1.05`。native matched 三窗口须同时满足：
   `current MPJPE - balanced MPJPE` 与 `current object-goal error - balanced object-goal error`
   的 sequence-paired bootstrap 95% CI 下界均 `>0`，且 balanced/current pelvis-goal mean-ratio
   的 95% CI 上界 `<=1.10`。condition permutations 与其他 field/physical metrics 为完整描述证据，
   不得替代或放宽 gate。全部通过分类为
   `fresh-optimizer-balanced-smoke-positive-stop`，任一失败分类为
   `fresh-optimizer-balanced-smoke-negative-stop`。
5. **治理与停止。** amendment 必须先单独提交，随后 implementation/config/tests/docs 组成第二个
   完整 commit。实现只允许增加 hash-locked Phase-1B-online weight-only initialization、D2-M
   paired orchestration/evaluation/summary 与测试；默认 random-only training contract 和 released
   checkpoint rejection 必须保持。须测试只加载 online model、拒绝 EMA/released/wrong hash、
   optimizer/EMA/RNG reset、paired training RNG、locked weights/budget/LR、fresh selections、
   bootstrap gate、全 field/condition/native reporting、terminal state/hash、production sampler
   不读 future GT/stored BPS。worker 必须走 exact committed object、clean worktree、fully resolved
   config、same-context four-GPU preflight、`tools/experiment.py start/finish/register`、persistent
   session 与 immutable artifact recovery。无论 positive/negative，本 session 只完成 D2-M0 并停止；
   positive 只授权用户随后考虑新的 from-random balanced-objective dated screen，不自动启动它；
   negative 不自动触发 architecture/loss/condition intervention。D2-H1、完整训练、official/CHOIS、
   merge/tag、Phase 1C 与后续阶段仍禁止，等待用户再次确认。

D2-M0 implementation entry point 为 `tools/run_hoi_d2m.py`，训练基配置为
`code/config/config_train_hoi_prior_d2m.yaml`，fresh selection/bootstrap/gate 位于
`code/priors/optimizer_reset.py`，统一 teacher/native evaluation 与 compact summary 分别位于
`tools/evaluate_hoi_d2m.py` 和 `tools/summarize_hoi_d2m.py`。`train_hoi_prior.py` 的默认路径仍只
允许 random initialization 与 `50/50/0.1/1`；只有同时满足 exact D2-M candidate、source hash、
online variant、budget、LR 与 paired-RNG contract 时，才允许 weight-only source load 和 balanced
weights。native physical summary 只作向后兼容的 additive reporting，新增 pelvis/object translation
MAE 与 object-rotation geodesic；production sampler equation、BPS recomputation 与 condition API
未改变。runner 在任何 candidate subprocess 前还会硬验证 D2-I source metrics
`910998c54487cb127343e783773d3dbf13d24b359caf0442695f066bc271bf56`、normalization
`6969c0c05ac3e03d9b014380118bee78ce8999e5b9adeeb8e700f4eba8baa969` 与 BPS
`fdff7204b4697e105457cb7e39267b9555bc0d8d854dbc92cd67e2d8c3e77042`，并将三者作为
reportable manifest assets；任一 mismatch 均在 GPU training 前失败。

2026-07-16 D2-M0 按预注册分类为 `fresh-optimizer-balanced-smoke-negative-stop`。worker 在 exact
commit `f17beabf508aa461ac5452ab3f705095bd0e04a2` 上完成两个 candidate 各 64 个真实 update；
source/initial model hash、旧 state 零加载、terminal optimizer `119/64` 与全部记录 loss finite
均通过。current/balanced 每个 rank 分别出现 `10/2` 次 AMP overflow skip，因此预注册的
zero-overflow finite gate 失败，且两条路径随后消费不同 batch/q-noise，paired training RNG audit
失败。teacher t=250/499 的 balanced joint-position MSE 分别为 current 的 `0.6564/0.7158`，
但 object-translation 分别为 `1.0773/1.0567`，相对 source 的两格几何均值为 `1.0983`，故
teacher object gate 失败。fresh 32-sequence native cohort 上，balanced 相对 current 的
MPJPE/object-goal/pelvis-goal ratio 为 `0.4132/0.8488/0.6045`，但 foot sliding ratio
`1.7310`、contact F1 ratio `0.8434`；这些描述性改善不得覆盖 pairing/teacher 失败。完整 artifact
已由 worker 主动回收到
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2m-reset-paired-s42-20260716`，36 files /
1,215,525,519 bytes，两端逐文件清单 SHA-256
`4e1357d60acd57e49f3ce2762c898cd945032cd57f8dc81eaf66282bff7610d5`。compact result 见
`experiments/results/p1_hoi_phase1b_d2m_reset_paired_s42_20260716.json`，SHA-256
`421a4a04675cb8561750ef8a6823bc4dbcda06a5c8efe0b8f8bca35b43160fd3`。D2-M 不重跑、
不提升 checkpoint、不授权 from-random screen 或完整训练；D2-H1 仍未启动。

2026-07-16 Phase 1B D2-N author-native latest-checkpoint transfer audit 预注册。用户在本地重新执行
released InfBaGel 的 `python test_infbagel_hoi.py` 后报告 native 指标与论文大致一致，并询问
Phase 1B 的差结果是否可能由额外 CHOIS evaluator 或不同指标实现导致。已有不可变证据先排除
这一强解释：Phase 1B formal HOIPrior 的 438-sequence native 失败
`p1-hoi-eval-native-r1-s42-20260714` 本身就是通过作者 `test_infbagel_hoi.py` 的
`compute_metrics` 路径得到，CHOIS 只在其后独立运行并进一步 corroborate；当前
`code/eval_metrics.py` 与作者基线 commit `b9a158f75ab0740c91c9cfc8863a65fa381b014c`
逐文件 SHA-256 均为
`445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547`。因此 D2-N0 不把
“evaluator mismatch 解释既有 formal failure”作为仍可成立的假设，只回答一个更窄的新问题：
D2-M latest balanced checkpoint 在作者 native 438 协议上是否复现其 fresh internal cohort 的
人体/目标改善，及其 foot-sliding/contact tradeoff。

1. **唯一 reportable audit。** run id 固定为
   `p1-hoi-d2n-author-native-paired-s42-20260716`。在同一 worker、同一 exact committed object
   上依次评估 source R-3072 online
   `48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4`、
   D2-M current online
   `76e0d8811fc9f54caa6d4778e2fe9fcaee78fad98bee5f17570b47568f71e31f`
   和 D2-M balanced online
   `ded9a12d4e85179c37e2457475649ccc614ef364b97eaebade0629b2c11d4ed8`。
   三者必须全部报告，不得只运行或保留 balanced；不得读取 EMA、released checkpoint、旧
   optimizer/RNG state。每次 invocation 都重新固定 seed 42，使用相同 438 official test
   sequence ordering、3 windows、500-step production diffusion sampler、online weights、
   condition 与 sampler RNG 协议。GT 只进入作者指标 reference，不得进入 production condition；
   rollout BPS 必须从当前生成物体姿态重算，不得读取 future GT 或 stored per-frame BPS。
2. **作者 native 指标与统计。** 固定使用当前 `code/test_infbagel_hoi.py`
   SHA-256 `22886f8797ceb04a892487393dea9f80e19877bc02dd7a6f39127e7319119524`、
   exact-author `code/eval_metrics.py` 上述 hash 与
   `config_eval_hoi_prior.yaml` SHA-256
   `89c702d96b98289924225c4b163d3b29eb22efe27c50ac799ddd0c71c515aa73`。
   完整报告 end-object/pelvis goal、feet height、foot sliding、contact
   accuracy/precision/recall/F1/percent、MPJPE、human/object translation、object rotation、
   hand/human penetration 与 ratio；保存全部 438 条 per-sequence metrics。对 balanced-source
   和 balanced-current 以 sequence 为 paired unit、bootstrap 10,000、seed 42 报告差值或 ratio
   95% CI。Phase 0 released baseline 只读取 compact aggregate
   `experiments/results/p0_hoi_table5_baseline_s42_20260712.json`
   SHA-256 `76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6`
   作绝对描述对照，不加载 checkpoint，不新增 baseline GPU run。
3. **transfer gate 与解释边界。** 只有全部 finite、三 checkpoint/input/config hashes 精确、
   各 438 条、sampler audit 无 future GT/stored BPS，并且 balanced 相对 source 与 current 在
   `mpjpe`、`end_obj_trans_err`、`xy_points_err`、`obj_trans_dist` 四项的 paired-bootstrap
   95% CI 均证明改善，同时 foot-sliding ratio CI 上界 `<=1.10`、contact-F1
   `balanced-comparator` 差值 CI 下界 `>=-0.02`，才分类为
   `author-native-latest-transfer-positive-stop`；任一失败分类为
   `author-native-latest-transfer-negative-stop`。无论分类，结果只描述 D2-M latest checkpoint
   是否跨 evaluator protocol transfer；不得 retroactively 改写 D2-M gate、选择 checkpoint、
   授权重跑/训练/architecture/loss/condition/sampler 修改，也不得声称已解释一致性蒸馏差异。
4. **执行与停止。** D2-N0 只允许增加非覆盖式三 checkpoint orchestration、fully resolved
   configs、paired summary 与 tests；不得修改作者 native metric formula、HOIPrior model、
   representation、loss、condition API 或 production sampler equation。先提交本 amendment，
   再提交 implementation/config/tests；worker 主动 fetch exact commit，使用单张无 contention
   RTX 3090 在 persistent session 运行，走 `start/finish/register` 与 immutable recovery。
   CHOIS/FID/R-Precision/FPS gate、训练、D2-H1、Phase 1C 及后续阶段均不启动；D2-N 完成后停止
   并报告，等待用户再次确认。

2026-07-16 Phase 1B D2-N r1 launch-preflight amendment。首次 reportable run id
`p1-hoi-d2n-author-native-paired-s42-20260716` 在 workload 启动前即停止：已创建的 immutable
manifest command 将锁定 baseline 文件名误写为
`p0_hoi_table5_baseline_s42-20260712.json`，而正确资产为
`p0_hoi_table5_baseline_s42_20260712.json`。未创建 persistent workload session，三个
checkpoint 均未开始评估，completed candidates 为空；该 run 已以 `aborted` 完成
`finish/register`，manifest/metrics/resolved-config/preflight SHA-256 分别为
`d9a1f9867ecb0dec93239464a444d9afb7a9e57ed735a454b1167485ad4575b2`、
`a7954605193fca3f2b0ab77b33f5a29fd0b331fa2dc752dfa3518e134c650d78`、
`a78565e379d52536f78d905c996de6d1617793d24dfe7ab4e33582045ca353d3`、
`50581e0de70fff1f4e45fe253dc9d1b1115333224b277a199578485488223744`，不得覆盖或复用。

唯一替代 run id 预注册为
`p1-hoi-d2n-author-native-paired-r1-s42-20260716`。r1 必须原样继承上一节锁定的 source/current/
balanced online checkpoint、438 sequences × 3 windows、500-step production sampler、
作者 native evaluator/input hashes、checkpoint order、paired bootstrap、transfer gate 和全部
停止边界；只允许将 runner/config/output 的 run id 改为 r1 并修正 manifest command 中 baseline
路径，不得依据未观察的结果改变任何指标、阈值、checkpoint、condition、loss 或 sampler。
必须重新生成 r1 fully resolved configs 与同一 escalated context 的 preflight/manifest，确认
物理 GPU 3 无 contention 后在 worker-owned persistent session 启动。无论 r1 正负，仍不得
checkpoint selection、训练、D2-H1、CHOIS、FID/R-Precision 或后续阶段。

2026-07-16 D2-N0 r1 在 exact commit
`e55438e36a74f2cf27dd7cec68d2d7c3ca97b64f` 上完成，分类为
`author-native-latest-transfer-negative-stop`。source/current/balanced 三个 online checkpoint
均完整评估 438 sequences × 3 windows，全部 finite，checkpoint/evaluator hashes、production
sampler 无 future GT/stored BPS 契约均通过。balanced 相对 source/current 的 author-native
MPJPE ratio 为 `0.5025/0.5033`，end-object-goal ratio `0.3565/0.3596`，pelvis-goal ratio
`0.6354/0.6232`，object-translation ratio `0.5419/0.5276`；四项 paired-bootstrap 95% CI
均严格证明改善。foot-sliding ratio 也改善为 `0.5796/0.5636`。但 balanced contact F1
`0.3386`，相对 source/current 分别下降 `-0.0868/-0.1030`，95% CI
`[-0.1144,-0.0584]` / `[-0.1311,-0.0744]`，违反预注册的 preservation lower bound
`>=-0.02`，因此完整 gate 失败。相对 released Phase 0 compact baseline，balanced 的 MPJPE、
end-object、pelvis-goal、object-translation、contact-F1 ratio 仍为
`1.5640/3.5709/2.0391/1.8920/0.4656`，不能提升或选择为可用 checkpoint。

该结果也排除“Phase 1B 差结果主要由 CHOIS 或不同 evaluator 造成”：D2-N 使用的
`eval_metrics.py` 与作者基线 commit 逐字节一致，且既有 formal 438 failure 本身已使用同一作者
native path。真实结论是 balanced objective 的人体/目标/平移改善可跨 internal cohort 迁移到
作者 native official-test protocol，但伴随显著 contact-recall/F1 deficit，且总体仍显著落后于
released consistency-distilled baseline。完整 artifact 已由 worker 主动回收到
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2n-author-native-paired-r1-s42-20260716`，31 files /
1,014,191 bytes，两端逐文件清单 SHA-256
`ca219c09db7591cd6f6ba91a65fccee8206583841ffafbba7eccac1ff36c53bf`。compact result 见
`experiments/results/p1_hoi_phase1b_d2n_author_native_paired_r1_s42_20260716.json`，SHA-256
`004595af680fd4040781d63ab6c5a46e5bd016663e8878281e8d1b38d2b8b7bb`。D2-N 不重跑、不选择
checkpoint、不授权训练或 D2-H1；停止并等待新的用户确认与 dated intervention。

2026-07-16 Phase 1B D2-O contact semantic–geometry alignment diagnostic 预注册。用户在 D2-N
确认 balanced checkpoint 的人体/目标/平移与 foot-sliding 改善可迁移、但 contact F1 显著下降后
明确要求继续推进。现有证据不支持直接增加 contact-channel reconstruction 权重：D2-M balanced
训练的 contact smooth-L1 与 threshold-0.5 accuracy 均优于 current，fresh internal rollout 的
contact-channel MSE 也更低，但作者 native 5 cm 几何 contact 的主要退化来自 recall/contact
percent。作者实现同样用四维 contact reconstruction，而 native evaluator 仅以左右手关节到物体
表面的几何距离定义 contact；因此在选择新的 loss/guidance 前，D2-O0 只审计语义标签、生成
contact channel、人体姿态与物体姿态之间的对应关系，不训练、不修改 model/loss/condition 或
production sampler。

1. **唯一 reportable diagnostic。** run id 固定为
   `p1-hoi-d2o-contact-alignment-s42-20260716`。使用 source R-3072 online
   `48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4`、
   D2-M current online
   `76e0d8811fc9f54caa6d4778e2fe9fcaee78fad98bee5f17570b47568f71e31f`
   和 D2-M balanced online
   `ded9a12d4e85179c37e2457475649ccc614ef364b97eaebade0629b2c11d4ed8`；
   三者全部使用 online weights、seed 42、500-step production sampler、matched condition 与
   完全相同的 sequence/window ordering 和按 checkpoint role 独立重放的 sampler noise。不得读取
   EMA、released checkpoint、optimizer/RNG state，不得只保留有利 checkpoint、手、threshold、
   object category 或 sequence。
2. **fresh holdout 与 measurements。** internal-validation 序列按
   `SHA256("42:d2o-contact-alignment:" + sequence_name + ":14,56,98")`、sequence name 与
   sequence id 排序，固定前 64 个同时具备 `pi=(14,56,98)` 的序列，共 192 窗；global window
   indices SHA-256 固定为
   `1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`。这些窗口不与此前
   D2/D2-M 使用的 `pi=(0,42,84)` rollout 窗口重叠。对每个 sequence、window、active frame、
   left/right/union 与 object category 完整封存：
   - GT 四维 contact label、前两维 hand-semantic label threshold `0.5`，以及 GT 人体与 GT
     物体在 `2/5/7.5/10 cm` 阈值下的 hand-object geometry；
   - 每个 checkpoint 的四维 contact reconstruction smooth-L1/MSE、前两维 threshold
     `0.5/0.75/0.95` precision/recall/F1/accuracy/calibration、生成 semantic 与生成 geometry
     agreement；
   - 作者 native 5 cm physical contact accuracy/precision/recall/F1/contact percent，另将
     `2/7.5/10 cm` 作为描述性 threshold sensitivity；
   - 在 GT 5 cm contact frames 上报告四种距离分解：
     generated-human/generated-object、GT-human/generated-object、
     generated-human/GT-object、GT-human/GT-object，从而区分 human-pose、object-pose 与
     joint-coupling contribution；另报告左右手、union、contact run length 与距离分位数。
   paired bootstrap 固定 10,000、seed 42，sequence 为 paired unit。GT 只用于 diagnostic
   reference/decomposition；production condition、BPS 更新与 sampler 不得读取 future GT 或
   stored per-frame BPS。
3. **分类 gate。** 所有 checkpoint/input/config hashes、64 sequences/192 windows、selection
   hash、shared sampler-noise replay、history restoration、finite、all-field/all-threshold
   reporting 与 sampler provenance 必须通过，否则分类
   `contact-alignment-contract-failure-stop`。contract 通过后：
   - 若 GT hand-semantic threshold `0.5` 对 GT native 5 cm geometry 的 union F1 或 recall
     `<0.80`，分类 `label-evaluator-contract-mismatch-stop`；
   - 否则，只有 balanced 相对 source 与 current 的前两维 semantic contact MSE 改善均由
     paired-bootstrap 95% CI 严格证明（`comparator - balanced` CI 下界 `>0`），且 source 与
     current 相对 balanced 的 native 5 cm physical recall 下降也均由 95% CI 严格证明
     （`comparator - balanced` CI 下界 `>0`），才分类
     `semantic-geometry-decoupling-positive-stop`；
   - 其余完整结果分类 `mixed-contact-deficit-stop`。
   这些分类只决定下一次 dated intervention 应优先处理 label/evaluator contract、显式几何对齐
   或其他 mixed mechanism，不选择 checkpoint，也不自动授权 contact loss、guidance、CFG、
   architecture/condition 修改或训练。
4. **实现、执行与停止。** 尽量复用 D2-M rollout、stable RNG、object mesh、native physical
   metric 与 paired-bootstrap utilities；新增独立 contact-alignment utility、fully resolved
   config、summary 与 tests，覆盖 selection determinism/non-overlap、semantic/geometry truth
   table、左右手与 union、threshold completeness、human/object swap decomposition、shared
   sampler noise、history、finite、all-checkpoint reporting 及 production sampler 无 future
   GT/stored BPS。先提交本 amendment，再提交 implementation/config/tests；worker 主动 fetch
   exact commit，在单张无 contention RTX 3090 的 worker-owned persistent session 走
   `start/finish/register` 与 immutable recovery。无论分类为何，本 session 都在 D2-O0
   register/compact result 后停止；不得启动 D2-H1、D2-G、contact remediation smoke/training、
   official 438、CHOIS、Phase 1C 或后续阶段，等待用户再次确认。

D2-O0 implementation entry point 为 `tools/diagnose_hoi_d2o.py`，锁定 selection、统计与分类
utility 为 `code/priors/contact_alignment.py`，immutable compact aggregate 由
`tools/summarize_hoi_d2o.py` 生成。resolved config 必须锁定三个 checkpoint 的绝对路径/hash、
batch size `16`、online-only、500 diffusion steps、三项 semantic 与四项 physical threshold、
以及无训练 stop contract；完整 artifact 保留逐 sequence、逐 active frame 与 object-category appendix，
tracked aggregate 可移除这些大数组但不得移除 aggregate、paired comparisons、contract 或 decision。

2026-07-16 Phase 1B D2-O r1 Python-3.8 compatibility amendment。首次 reportable run
`p1-hoi-d2o-contact-alignment-s42-20260716` 在 exact commit
`141652d025d86ab6385813ae11a2dcb60c6ab620` 上通过 resolved-config、preflight 与
`tools/experiment.py start` 后启动，但在任何 checkpoint rollout/GPU sampling 前，于
ground-truth object-category summary 使用 Python 3.9 `dict | dict` 运算符时，被 worker 的
verified Python 3.8 以 `TypeError` 停止。该 run 已以 `failed` finish/register；manifest、
metrics、resolved-config、preflight 与 run-local registry SHA-256 分别为
`1741b75b751d388b75041174feb4f8750d4f123cb40a6cef0f41f655019ce68d`、
`e0d3306fe7ddbe74d4c44eca3152f2dd1f9cc1f7f40a004a6c1988674295e6e9`、
`1d6172fd7454b65646f961c4a06b5dd6b2398c377133cc876348b676da309f51`、
`6dcfe0e60b4035a5f8a0a58d1b3be0126dd5637bdf1e4df3e422a7b993218ebc`、
`d4920488ebe21267e515de954d59b4e1e88b30128d5f1266bf9b9d5c5a7d8b76`；不得覆盖或复用。

唯一 compatibility retry run id 预注册为
`p1-hoi-d2o-contact-alignment-r1-s42-20260716`。r1 必须原样继承 D2-O0 的三个 online
checkpoint、64 sequences × 3 phase-offset windows、selection SHA-256、batch size 16、
matched condition、shared sampler noise、semantic/physical thresholds、distance decomposition、
bootstrap、classification gate 与全部 no-training/stop boundary。代码变更只允许：
将 object-category summary 的 `dict | dict` 改写为 Python-3.8-compatible construction，
更新 run-id/output constants，并增加会实际执行多 object-category summary 的 regression test；
不得依据未观察的 checkpoint 结果改变任何科学输入或阈值。必须在 authority 与 worker verified
environment 重跑完整 tests，重新生成 r1 resolved config/preflight/manifest，并继续使用无 contention
GPU 1。无论 r1 分类为何，仍只 finish/register/recover D2-O0 并停止，不得训练、D2-H1 或 D2-G。

2026-07-16 D2-O0 r1 在 exact commit
`87636dbbcbb4f7c666d6576590f3a587e5f4add2` 上完成，分类为
`mixed-contact-deficit-stop`。三个 online checkpoint、64 sequences / 192 phase-offset windows、
selection/shared sampler noise、history、finite、model-state immutability、全部 contact fields/
thresholds/decomposition 与 production sampler provenance contract 均通过；history max abs 为
source/current/balanced 的 `5.36e-7/4.77e-7/4.77e-7`。GT hand-semantic 对 GT native 5 cm
geometry 的 union F1/recall 为 `0.8878/0.8355`，超过 `0.80/0.80`，因此不属于
label/evaluator contract mismatch。

balanced 的 first-two semantic contact MSE 为 `0.1500`，低于 source/current 的
`0.2540/0.2421`；paired comparator-minus-balanced 95% CI 分别为
`[0.0660,0.1449]` / `[0.0578,0.1290]`，严格证明 semantic prediction 改善。native 5 cm
physical precision/recall/F1/contact-percent 则为 source
`0.9157/0.3696/0.5266/0.3177`、current
`0.9234/0.3587/0.5167/0.3058`、balanced
`0.9487/0.3147/0.4727/0.2612`；balanced 的 recall point estimate 较低，但 source/current
minus balanced 的 paired 95% CI 为 `[-0.0122,0.1143]` / `[-0.0199,0.0952]`，均跨零，
故未达到严格 semantic–geometry decoupling 分类。三者共同呈现高 precision、低 recall 与短
contact duration：5 cm predicted union run mean 为 `7.98/6.68/6.62` frames，而 GT 为
`54.26` frames。balanced semantic threshold-0.5 contact percent 为 `0.7228`，但 generated
physical contact percent 只有 `0.2612`，其 semantic-to-generated-geometry union F1 为
`0.4643`（source/current `0.5652/0.5727`）；这是描述性解耦证据，但不得覆盖预注册 paired gate。

GT-5 cm contact frames 的 distance decomposition 还表明缺陷不是可单独归因于 human 或 object：
source/current/balanced 的 generated-human/generated-object union mean distance 为
`10.19/9.85/10.35 cm`；替换为 GT human 但保留 generated object 后为
`29.77/31.24/17.91 cm`，保留 generated human 但替换为 GT object 后为
`49.48/50.84/25.71 cm`，GT/GT 为 `1.70 cm`。这说明 generated human/object 之间存在明显
coupled drift；balanced 减少了 cross-pose 绝对错位，却没有恢复足够的相对接触覆盖和持续时间。
该结果只支持“共享 under-contact-duration + semantic/geometry/coupled-pose mixed deficit”，
不自动选择 contact loss、guidance、CFG 或 architecture intervention。

首次 Python-3.8 失败 artifact 已回收到
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2o-contact-alignment-s42-20260716`，7 files /
40,865 bytes，tree SHA-256
`1b775fcb6704d274d4f5c3c52a455ba44e3d09055326a82dbb59b86cd09b81ac`。r1 完整 artifact
已回收到
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2o-contact-alignment-r1-s42-20260716`，8 files /
48,775,188 bytes，tree SHA-256
`823fc7a278d47f67feb07c8acc3c7dd7ac91c7fd588c2e1c9987c273b9238978`。r1 manifest、
metrics、resolved-config、preflight、run-local registry SHA-256 分别为
`5a5fd5418f5322e3754ab62e3e1cba7e3a3184a48390aa8e3287907582989909`、
`aee38b32ffc090483266c7c89d0d4c4a27f96278f8eaaff67444916e87d8bcd3`、
`5fc45c0d03807e45d2cc8d6dd9f92f1942615306b5acf82b07b321e41d14cc61`、
`70e565449bd21d98198ac51861cb993268f3c9857d722b1a864d4955f0b92270`、
`229a5e9c0b3cf761630bd81397a18bdcecece953e917df1a8c5ac3756849c4`。compact result 见
`experiments/results/p1_hoi_phase1b_d2o_contact_alignment_r1_s42_20260716.json`，SHA-256
`9cea710a82d81cb9af0a77499a3f15e8124acf4649476704ce3aca09bdd4ece3`。D2-O 不重跑、
不选择 checkpoint、不启动 contact remediation、训练、D2-H1 或 D2-G；停止并等待新的用户确认。

2026-07-16 Phase 1B D2-Q author-contact-guidance paired counterfactual 预注册。用户在 D2-O
完整结果后明确授权继续执行建议的最小几何/持续时间 intervention。该方向不是旧 D2-G：
D2-G 是未满足触发条件的随机初始化训练 fallback；D2-Q0 只在封存 online checkpoint 上执行
sampling-only counterfactual，不创建 optimizer、不更新参数、不修改 production sampler 默认
行为。D2-O 已排除 label/evaluator contract mismatch，并显示三个 checkpoint 共同存在高
precision、低 recall、短 physical-contact duration；balanced 又同时具有最好的语义接触重建、
`.95` hand-semantic coverage 以及人体/目标/物体平移表现，因此 balanced 在观察 D2-Q0 前固定为
唯一 mechanism-gate candidate，source/current 仍必须完整运行和报告，禁止结果后选择有利
checkpoint。

1. **唯一 reportable run 与 fresh holdout。** run id 固定为
   `p1-hoi-d2q-author-contact-guidance-s42-20260716`。使用 source R-3072 online
   `48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4`、
   D2-M current online
   `76e0d8811fc9f54caa6d4778e2fe9fcaee78fad98bee5f17570b47568f71e31f`
   与 D2-M balanced online
   `ded9a12d4e85179c37e2457475649ccc614ef364b97eaebade0629b2c11d4ed8`。
   internal-validation sequence 按
   `SHA256("42:d2q-author-contact-guidance:" + sequence_name + ":28,70,112")`、
   sequence name、sequence id 排序，固定前 64 个同时具备 `pi=(28,70,112)` 的序列，共
   192 windows；global window indices SHA-256 固定为
   `337ba964d4384bc66664ceeb148eb960632bac3861718063b6f86f43c59c5344`。
   这些 phase offsets 与 D2/D2-M 的 `(0,42,84)` 及 D2-O 的 `(14,56,98)` 均不重叠。
2. **paired sampling intervention。** 每个 checkpoint 都以 online weights、seed 42、
   matched condition、500-step 原始 DDPM posterior、相同 sequence/chunk ordering 各运行
   unguided 与 guided 两条 trajectory；两条 path 的初始 latent 和每一步 posterior noise 必须
   byte-identical，guidance 不得消耗 RNG。guided path 只复现作者
   `b9a158f75ab0740c91c9cfc8863a65fa381b014c` 的 hand-object interaction 核心项：
   - 由当前 22 路 rotation 与相同 rest-human offsets 做 24-joint FK，使用作者 palm indices
     `22/23`；前两路预测 contact semantic 以 `>0.95` 形成 detached mask；
   - 以生成 object pose 变换每个 hash-verified rest mesh 的 deterministic uniform-index
     2,048-vertex subset，空间 hinge threshold 固定 `0.02 m`；
   - 保持作者的 object-COM/object-rotation detach、contact-pair temporal cosine、
     batch-size multiplier、`guidance_scale=1` 与
     `x_prev += grad(-guidance_loss, pred_x0)` 注入顺序；只在 reverse step `>0` 注入；
   - 作者 `guidance_loss.py`、`models/infbagel.py` 与 sample config 的 blob SHA-256 分别锁定为
     `5747721bcc015911c7999692079666f7e5d6204912761d91b8b82d4ba4c4ab24`、
     `6dec84991c685e11d1f8e4f2f928497673210066d38a5901d0537a40504f9d76`、
     `39490dc60c47da7273434d65fa95e44a2d34382c872fe7c1718c14886bdde388`。
   为隔离 hand-object mechanism，不加入作者外层 `×10`、feet-floor `×500` 或任何 scene/
   penetration 项；这三项 omission、2,048-vertex deterministic approximation 与 HOIPrior
   codec 的 differentiable SO(3) decode 必须作为 parity deviation 明确封存。guidance 后必须
   再恢复两帧 immutable history；不得 CFG、support clamp、future GT、stored per-frame BPS、
   released checkpoint、EMA、consistency distillation 或 checkpoint write。
3. **完整 measurements。** 对 source/current/balanced × unguided/guided 全部报告：
   - 每一步 guidance loss、spatial/temporal component、gradient RMS/norm/max、active semantic
     mask coverage、finite 与 history max abs；
   - 24-joint FK palm `22/23` 和 28-joint direct hand `24/26` 的左右手/union
     `2/5/7.5/10 cm` accuracy/precision/recall/F1/contact percent、distance quantiles 与
     per-sequence contact run lengths；
   - object/pelvis goal error、MPJPE、joint/pelvis/object translation error、object rotation、
     foot sliding、四路 contact reconstruction、逐 window 与逐 object-category appendix；
   - paired guided-minus-unguided 10,000-bootstrap（seed 42，sequence paired unit），以及
     source/current 的完整 descriptive response。GT 只用于 reference metric，不得进入
     guidance mask、condition、BPS 或 reverse equation。
4. **mechanism gate。** 任一 checkpoint/variant、hash/selection、paired noise、history
   `<=1e-5`、finite、model-state immutability、posterior-helper reuse、all-field/all-threshold
   reporting 或 sampler provenance 失败，分类
   `author-contact-guidance-contract-failure-stop`。contract 全部通过后，只有预先指定的
   balanced guided 相对 balanced unguided 同时满足以下全部条件，才分类
   `author-contact-guidance-positive-stop`，否则分类
   `author-contact-guidance-negative-stop`：
   - FK-palm union 5 cm recall、F1 与 contact percent 的 guided-minus-unguided paired
     bootstrap 95% CI 下界均 `>0`；
   - FK-palm union 5 cm predicted mean run length 的 paired CI 下界 `>0`，且
     `mean(guided)/mean(unguided) >=1.5`；
   - FK-palm union 5 cm precision 的 guided-minus-unguided CI 下界 `>=-0.02`；
   - guided/unguided 的 MPJPE、object goal、pelvis goal、object-translation MAE 与 foot
     sliding paired mean-ratio 95% CI 上界均 `<=1.10`。
   direct-hand 与 source/current 结果均为必报描述性证据，不能替代上述 balanced FK gate。
5. **实现、执行与停止。** 新增独立 analysis-only sampler/guidance utility、runner、fully
   resolved config、summary 与 tests；production `GaussianDiffusion.sample` 和 evaluator adapter
   不增加 guidance 参数。tests 覆盖作者公式 replay、mask/detach、paired RNG、step-0 omission、
   posterior helper reuse、history restoration、FK/direct-hand separation、finite、selection、
   all checkpoint/variant/field/threshold reporting、gate boundary 及 production sampler 无
   future GT/stored BPS。先单独提交本 amendment，再提交 implementation/config/tests；authority
   与 worker 使用各自 verified absolute Python，worker 主动 fetch exact committed object，在
   worker-owned persistent session 走 resolved-config/preflight/start/finish/register/recovery。
   无论 gate 正负，本 session 只登记 D2-Q0 并停止；不得把 guidance 写入 production default，
   不得运行 official 438、CHOIS、D2-G、D2-H1、任何 smoke/training、Phase 1C 或后续阶段，
   等待用户再次确认。

2026-07-16 D2-Q0 pre-implementation author hand-scale clarification。首版 amendment 将作者
wrapper 的 outer hand-object `×10` 误列为 omission；在 implementation commit、resolved config
和 workload 均尚未产生前，重新审阅 `b9a158f` 的 `apply_hoi_guidance_loss` 与
`apply_hosi_guidance_loss` 确认两者都对
`apply_hand_object_interaction_guidance_loss` 固定乘 `10`。D2-Q0 因而必须保留这个作者
hand weight：实际注入为
`x_prev += grad(-(10 * author_hand_object_core_loss), pred_x0) * guidance_scale`，
其中 `guidance_scale=1`；仍排除 feet-floor `×500`、scene/penetration 项。run id、三个
checkpoint、selection、paired noise、其余公式、2,048-vertex approximation、measurements、
gate 与全部 stop/no-training contract 不变。该修正发生在任何结果观察前，防止以弱十倍的
counterfactual 形成假阴性。

D2-Q0 implementation entry point 为 `tools/diagnose_hoi_d2q.py`，锁定 selection、作者公式、
paired sampler、FK/direct-hand metrics 与 gate 的 utility 为
`code/priors/contact_guidance.py`，immutable compact aggregate 由
`tools/summarize_hoi_d2q.py` 生成。resolved config 必须封存三个 checkpoint、全部 rest mesh、
author baseline blobs、batch/device、作者 hand `×10`、499 个 guidance steps、parity
deviations 与 no-training stop contract；完整 artifact 保留逐 sequence、逐 window、逐
reverse-step 与 paired RNG appendix。

2026-07-16 D2-Q0 completed result：`author-contact-guidance-negative-stop`。authority
implementation commit 与 worker exact commit 均为
`1e9586b24efb29b0bd02e5c1b6b787ea299bc2ae`，两端 checkout clean；worker
`node01` 使用绝对 Python `/home/yujinlun/data/envs/infbagel/bin/python`、`cuda:1`、
batch 8 完成 source/current/balanced × unguided/guided 的 192-window paired
counterfactual，运行时 `3428.291 s`。首次 preflight 因 CHOIS baseline audit root 指向错误
checkout 而在 manifest/workload 前失败并保留；改用 pinned
`/home/yujinlun/data/evaluators/chois_release` commit
`8ec585aa0200fd2a890ffb12897bcf69ae719463` 后 corrected preflight 通过。workload
return code 为 0；没有复用 run id。

全部 contract gate 通过：三个 checkpoint/两种 variant 均 finite、all-field/all-threshold
complete，selection SHA-256 为
`337ba964d4384bc66664ceeb148eb960632bac3861718063b6f86f43c59c5344`，
paired initial/posterior noise 完全一致；source/current/balanced 的 history max abs 分别为
`5.9605e-7/4.7684e-7/5.3644e-7`；每个 guided checkpoint 恰有 `11976`
次 reverse-step guidance、step 0 为零次，作者公式 replay max abs 为 `0`，hand outer
weight 始终为 `10`。checkpoint model state hash 前后不变，parameter grad buffer 清空；
production posterior helper 被复用，production sampler 默认行为未变；sampler 不读 future
GT 或 stored per-frame BPS，rollout BPS 只由当前生成 object pose 重算。

预指定 balanced FK-palm union 5 cm 的局部接触响应显著为正：guided/unguided 的
recall 为 `0.4012/0.2795`，paired difference `0.1217`、95% CI
`[0.0640,0.1846]`；F1 为 `0.4821/0.3554`，difference `0.1267`、CI
`[0.0653,0.1940]`；contact percent 为 `0.3977/0.2742`，difference `0.1235`、CI
`[0.0655,0.1856]`；predicted run mean 为 `8.9701/6.5833` frames，difference
`2.3868`、CI `[0.1950,4.7352]`；precision difference 为 `0.0953`、CI
`[0.0275,0.1752]`。但是 run-mean ratio 只有 `1.3625`，低于固定 `1.5` 门槛。
此外 guided/unguided ratio 95% CI 为：MPJPE `0.9982 [0.9691,1.0294]`、object
goal `1.0157 [0.9670,1.0692]`，二者通过；pelvis goal
`1.0669 [0.9875,1.1572]`、object-translation MAE
`1.1307 [1.0528,1.2182]`、foot sliding `1.5350 [1.2447,1.8914]` 均因 CI
上界超过 `1.10` 失败。balanced direct-hand 5 cm recall/F1/contact-percent difference
分别为 `-0.0021/-0.0048/-0.0007`，CI 均跨零，run ratio 为 `0.8464`；因此 FK
接触改善没有转化为稳健 direct-hand 接触或可接受的整体运动质量。

完整 descriptive comparator 也没有提供可替代的 checkpoint 选择理由。source FK F1
difference 为 `0.0352`、CI `[0.0039,0.0706]`，但 recall/coverage/duration CI 跨零且
duration ratio 仅 `1.0295`；current FK duration difference 为 `1.6248`、CI
`[0.3604,2.9723]`，ratio `1.2221`，其余 FK recall/F1/coverage CI 跨零，且 foot-sliding
ratio CI 为 `[1.0566,1.4345]`。所有 checkpoint 与 field、threshold、category 和
per-sequence appendix 均保留，没有 favorable subset。

Implementation-parity appendix 锁定作者 commit
`b9a158f75ab0740c91c9cfc8863a65fa381b014c` 三个 blob hash。复现部分包括 24-joint
FK palms `22/23`、detached `>0.95` semantic mask、`0.02 m` spatial hinge、
object COM/rotation temporal detach、contact-pair temporal cosine、batch multiplier、
outer hand `×10`、guidance scale 1 和 negative-loss-gradient injection。预注册 deviation
包括省略 feet-floor `×500` 与 scene/penetration、deterministic 2,048 vertices 代替作者
random 10,000 vertices、codec differentiable SO(3) decode，以及当前 500-step DDPM
checkpoint 代替作者 consistency sampler。因此该结果证明 isolated author hand core 能推动
FK 接触，却以持续时间不足和 kinematic/object drift 为代价；它不授权把 guidance 写入
production，也不自动授权其它作者项或新模型/损失 intervention。

完整 immutable artifact 已由 worker 主动回收到
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2q-author-contact-guidance-s42-20260716`，
8 files / `158,718,271` bytes；worker/authority 在固定 `LC_ALL=C` 口径下 tree SHA-256
同为 `5612b73fdb61708acb895cd4bf63700a490edcabb74dce11013782d296895663`。
manifest、metrics、resolved config、首次失败 preflight、corrected preflight、run-local
registry SHA-256 分别为
`8876e724cddd394939f3c5f0634cda5ac1912fb01ef879e5110ac1b8182305bd`、
`34340fe1654a47797ef32a5a78aeca6b556470720129f3d954436f38ca9db025`、
`355d053c2bd396db502aef4d2e9cfd41e7775a378ff120390bf5c968ad877e69`、
`41da5a2a51e615b1cb7dba67346485d4e8f689727c48c9099ea16edb3e162ab1`、
`c3dad39ae9473598afe4481aadfe43415e2251c303889baa21b849f70f0755db`、
`d3dab6891caec3d1bfb4ada9809fd1fbb65166c84687473ce3aea3172a824a8c`。
compact result 为
`experiments/results/p1_hoi_phase1b_d2q_author_contact_guidance_s42_20260716.json`，
SHA-256 `66ff72cca071612d9f07e57d2521bf9e69f4724688cbcfe1760c31cbbbb07f23`。
未选择 checkpoint，未授权或启动 production guidance、optimizer、training、D2-H1、
D2-G、official/CHOIS 或 Phase 1C；D2-Q0 在此停止，任何下一方向都需要新的 dated
amendment 与用户确认。

#### 2026-07-17 Phase 1B D2-R state-subspace routed guidance 预注册

用户在 D2-Q0 停止后明确授权继续深入研究 HOIPrior 的多目标耦合问题。D2-Q0 已给出当前
最强的正向机制证据：balanced online checkpoint 在相同噪声下经 isolated author hand core
引导后，FK-palm union 5 cm recall/F1/contact coverage 均显著提高；但接触 run mean ratio
只有 `1.3625`，且 pelvis goal、object translation 与 foot sliding 的 guided/unguided ratio
CI 超过 `1.10`。重新核对作者 `test_infbagel_hoi.py` 后确认，作者 native contact target
同样是由 GT rotations/rest offsets 重建的 24-joint FK palms `22/23`，所以该正向接触证据
不是 evaluator 口径假象。依赖图审计进一步确认：hand spatial gradient 可同时写入人体 root
position、palm ancestor rotations、object translation 与 object rotation；temporal term 虽
detach object COM/rotation，仍可写 root/upper rotations；semantic mask 已 detach，contact
channels 不受梯度。D2-Q0 因此混合了“移动手臂”“移动整个人体 root”和“移动物体”三个
相互竞争的解，足部/目标退化符合 state-subspace coupling，而不要求先修改 architecture、
representation 或训练 loss。

1. **唯一 reportable run 与 frozen inputs。** D2-R0 run id 固定为
   `p1-hoi-d2r-state-routed-guidance-s42-20260717`，seed 42，只加载 D2-M balanced online
   checkpoint
   `ded9a12d4e85179c37e2457475649ccc614ef364b97eaebade0629b2c11d4ed8`；
   禁止 source/current/released/EMA、optimizer、scheduler 或任何 resume。仍使用 500-step
   production posterior coefficients/equation、matched text/BPS/pelvis/object-goal/progress、
   两帧 immutable history、无 CFG、无 SO(3) reverse projection、无 support clamp。所有
   intervention 均为 analysis-only sampling counterfactual，不改变 production default。
2. **fresh paired holdout。** internal-validation 每 sequence 固定 phase offsets
   `[7,49,91]`，与先前 rollout offsets
   `[0,14,28,42,56,70,84,98,112]` 不重合；按
   `SHA256("42:d2r-state-routed-guidance:" + sequence_name + ":7,49,91")`、
   sequence name、sequence id 排序，取前 64 个满足三 offset 的 sequence，共 192 windows。
   global window-index selection SHA-256 固定为
   `189e3f05e28007b3ba3dab25a6cf6afd63ed981135722ae41987129219bfd9da`。
   五个 variants 必须对每个 window 共享完全相同的 initial noise 与每一步 posterior noise，
   guidance 不得消耗 sampler RNG；每个 variant 均完整报告，禁止 favorable subset。
3. **固定 variants 与 primary candidate。** 所有 guided variants 只使用 D2-Q0 已锁定的
   author hand-object core：FK palms `22/23`、detached semantic channels `0/1 > .95`、
   `0.02 m` spatial hinge、detached temporal object COM/rotation、batch multiplier、outer
   hand `×10`、guidance scale 1，并只在 reverse step `499..1` 注入
   `x_prev += grad(-(10 * hand_core), pred_x0)`；step 0 不注入。比较：
   - `unguided`：production posterior control；
   - `author_all`：D2-Q0 all-state gradient replay；
   - `human_only`：从 `author_all` 精确置零 object translation/rotation channels
     `216:228`，保留 root position 与所有自然非零人体 rotation channels；
   - `upper_raw`：只保留 palm FK ancestor rotations
     `[3,6,9,13,14,16,17,18,19,20,21]` 的各 6D channels，置零 root position、其它人体
     rotations、object 与 contact；
   - `upper_norm`：预指定 primary candidate；先执行与 `upper_raw` 相同的正交坐标投影，再
     对每个 sample 的 mutable frames 乘
     `||g_author_all||_2 / ||g_upper_raw||_2`，使投影前后 state-gradient L2 norm 相等；两者
     同为零时 scale 定义为 1，只有分母为零而分子非零时视为 contract failure，不做 sweep、
     clip 或观察后调参。
4. **coupling 与 parity measurements。** 每个 reverse step/variant 封存 author loss、
   spatial/temporal、semantic coverage、全梯度和实际注入梯度的 norm/RMS/max、五个
   representation fields 的梯度能量、22 个 rotation joints 的能量、routing scale、masked-off
   max abs、history max abs、finite 与 formula replay。明确验证 upper-chain parent mapping、
   broadcasting、batch indexing、history mask、normalization inversion、posterior
   coefficients、detach、step-0 omission、model/checkpoint immutability、paired RNG，以及 sampler
   不读取 future GT/stored per-frame BPS。作者 blobs 与 D2-Q0 相同并重新 hash；deterministic
   2,048 rest vertices、codec differentiable SO(3) decode 与 500-step non-distilled checkpoint
   继续作为 parity deviations 如实报告。
5. **完整评测。** 主 contact 口径使用作者一致的 predicted 24-joint FK palms 对 GT
   rotation/rest-offset FK palms；另完整报告 28-joint direct representation hand contact。
   thresholds 为 `2/5/7.5/10 cm`，左右手与 union 全报；sequence-paired bootstrap 10,000、
   seed 42。native-like kinematics 至少包含 24-joint FK MPJPE、pelvis/object goal error、
   object translation MAE、object rotation geodesic、FK foot sliding；同时报告五个 state fields
   的 guided/unguided displacement、contact precision/recall/F1/coverage/run length 与距离分位数。
6. **mechanism gate 与停止。** 任一 hash/finite/history/noise/formula/routing-mask/norm replay
   contract 失败即分类 `state-routed-guidance-contract-failure-stop`。contract 全部通过后，
   只有 `upper_norm` 相对 `unguided` 同时满足下列条件才分类
   `state-routed-guidance-positive-stop`，否则分类
   `state-routed-guidance-negative-stop`：
   - FK union 5 cm recall、F1、prediction percent、prediction run mean 的 paired-bootstrap
     95% CI 下界均 `>0`；
   - run mean ratio `>=1.5`，precision difference CI 下界 `>=-0.02`；
   - FK MPJPE、pelvis goal、object goal、object translation MAE、object rotation geodesic、
     FK foot sliding 的 `upper_norm/unguided` paired ratio 95% CI 上界均 `<=1.10`。
   无论正负，D2-R0 均只登记 mechanism result 并停止；不得自动采用 production guidance，
   不得启动 official/CHOIS、D2-H1、D2-G、smoke/training、loss/model/representation/condition
   修改、Phase 1C 或后续阶段。任何训练或 production adoption 必须另做新的 dated amendment。

2026-07-17 D2-R0 pre-implementation selection-salt clarification。首个 implementation
selection regression 在任何 implementation commit、resolved config 或 workload 产生前发现：
预注册 SHA-256
`189e3f05e28007b3ba3dab25a6cf6afd63ed981135722ae41987129219bfd9da`
实际由已经用于 cohort 计算的 salt
`SHA256("42:d2r-routed-guidance:" + sequence_name + ":7,49,91")` 得到，而正文/registry
误写为包含额外 `state-` 的 salt；后者会得到不同 cohort SHA-256
`3e534f6f25528792c375329271f4f52c4433e58c205e3bc11ee0aee59c7fa3a3`。D2-R0 固定保留
原预注册 SHA、offsets、64 sequences、192 windows、run id、checkpoint、variants 与 gate，
只把 ordering salt 更正为 `d2r-routed-guidance`；不得使用误写 salt 对应的 cohort。

D2-R0 implementation entry point 为 `tools/diagnose_hoi_d2r.py`，locked selection、
state-subspace masks、per-sample norm replay、paired comparison 与 gate 位于
`code/priors/routed_guidance.py`，compact aggregate 由
`tools/summarize_hoi_d2r.py` 生成，回归覆盖位于 `tests/test_hoi_d2r.py`。runner 的
`--resolve-only` 是唯一 fully resolved config 生成入口；这些 analysis-only 文件不修改
production model、loss、representation、condition API、diffusion posterior 或 sampler default。

2026-07-17 D2-R0 completed result：`state-routed-guidance-negative-stop`。authority
implementation commit 与 worker execution commit 均为
`38e3b7ee3ca4eda06dc51d3fc5bfeef430b6b278`；worker 使用 `cuda:1`、batch 8、
balanced online checkpoint 与锁定的 64 sequences/192 windows selection，runtime
`3705.1325 s`。return code 0，全部 hash/selection/finite/history/paired-RNG/author-formula/
routing-mask/norm-replay/model-immutability/metric-completeness contracts 通过；history 与 author
formula replay max abs 均为 `0`，每个 guided variant 精确执行 11,976 次 update（192 windows
`×499`），step 0 不注入。

预指定 `upper_norm` 相对 `unguided` 的 FK-palm union 5 cm sequence-paired 结果为：recall
差 `+0.065904`、95% CI `[0.004129,0.128325]`；F1 差 `+0.078289`、CI
`[0.015101,0.144815]`；prediction percent 差 `+0.055432`、CI
`[-0.001488,0.113095]`；run mean 差 `+1.626042` frames、CI
`[0.174212,3.173210]`，paired-mean ratio `1.30819`；precision 差 `+0.099143`、CI
`[0.006060,0.197225]`。它因此在 recall/F1/run/precision 上有真实正向作用，却同时失败
coverage CI 与 `>=1.5` duration gate。native-like `upper_norm/unguided` ratio CI 为：FK
MPJPE `1.05865 [1.00841,1.11004]`、pelvis goal
`1.07714 [1.01755,1.14199]`、object goal
`0.97340 [0.93043,1.01719]`、object translation
`1.07970 [1.00179,1.15988]`、object rotation
`0.89071 [0.82186,0.95866]`、FK foot sliding
`1.05755 [0.93506,1.21076]`；四项 CI upper 超过 `1.10`，完整 gate 为 negative。

其余 variants 不能提供隐藏的 favorable solution。`author_all` 的 contact 效果最强：
recall/F1/coverage/run mean 差分别为 `+0.10808/+0.11279/+0.10938/+3.81250`，相应
CI 下界全 `>0`，run ratio `1.72261`；但 object translation、pelvis 与 foot-sliding ratio
CI upper 分别为 `1.11644/1.11442/1.24035`，仍失败保护 gate。`human_only` 将物体
gradient 精确置零后 run ratio 降至 `1.36797`，pelvis ratio CI
`[1.05544,1.22326]` 明确退化。`upper_raw` 不做能量重分配时 recall/F1/coverage/run 均
显著改善，run ratio `1.46634`，FK MPJPE ratio CI `[0.99643,1.05937]`；但仍低于 duration
门槛，且 pelvis/object-translation/foot-sliding CI upper 为
`1.12927/1.11622/1.22464`。

coupling audit 给出更具体的原因。`author_all` 全梯度平均 squared-energy 中
joint positions 占 `73.57%`、joint rotations `20.79%`、object translation `5.20%`、
object rotation `0.44%`、contact `0%`；所以主要 shortcut 是人体 root/joint-position，
而不是物体通道本身。`upper_norm` 删除 root/object 后把全部 L2 能量重分配到 ancestor
rotations，routing scale mean/max 为 `3.0684/88.4413`。尽管当前 step 的 object/root
injected gradient 精确为零，后续 denoiser calls 仍重新耦合所有输出字段；最终 normalized
state displacement 反而达到 joint positions `0.64685`、rotations `1.55019`、object
translation `0.15846`、object rotation `1.23756`、contact `0.21374`，均高于对应
`upper_raw`。因此 D2-R0 证明 state-subspace coupling 存在，却否定“简单 hard mask + 全局
norm conservation 足以解决 HOIPrior”的假设；观察结果后不得自动改 scale、做 trust-region/
PCGrad、增加 feet term 或改变训练目标。

完整 immutable artifact 已由 worker 主动回收到
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2r-state-routed-guidance-s42-20260717`，
8 files / `681,156,185` bytes，worker/authority tree SHA-256 同为
`6ae0d121c026e7d88d20c1212117e8d3c611c2b53d7edfef12e47399ac64588b`。
manifest、metrics、resolved config、preflight/hardware snapshot、run-local registry、returncode
与 workload log SHA-256 分别为
`7231e38105ea3166bcb53b26198659431701f97f445747dadcaac11c8cb64943`、
`f93f4910354008a75a994d7d1fdb6f3c706ae2cc2d5d040121f2882250982477`、
`3383c3f8afafd267963752064ae9a16713703402bf77d1ec05d0ee8f13f7ebd8`、
`02b2ca4e3a837e2fabb14af70d476a32411514f55c7d5c489510eca9a3212604`、
`ac556c4fa3b2fbd5ec26aadb0fd871520e391c860047d04ebf0b19967d952fc6`、
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`、
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
compact result 为
`experiments/results/p1_hoi_phase1b_d2r_state_routed_guidance_s42_20260717.json`，
SHA-256 `a1bb67f622aad70dc88ab978ee0326057d4b1076277487149634b8058aa0417a`。
未选择 checkpoint，未采用 production guidance，未启动 official/CHOIS、D2-H1、D2-G、
smoke/training 或后续 phase；任何下一项 intervention 需要新的 dated amendment。

#### 2026-07-17 Phase 1B D2-S denoiser-response trust frontier 预注册

用户在 D2-R0 完整 negative-stop 后明确确认继续推进。D2-R0 已证明 author hand gradient 的
state-subspace coupling：`upper_raw` 保留了显著 contact response，但 pelvis/object/foot
保护仍失败；`upper_norm` 又把每 sample 的 upper-chain gradient 平均放大 `3.068`、最大
`88.441`，并在后续 reverse steps 重新诱发全 state drift。因此 D2-S0 不观察结果后缩放
D2-R trajectory，也不直接改 production sampler；它在真实 unguided reverse state 上测量固定
input perturbation 经“下一次相同 denoiser”后的局部 response frontier，检验是否存在由无 GT
规则识别的安全信赖域。该诊断只决定后续是否值得预注册 full-trajectory controller，不授权
training、production guidance 或 loss/model/condition intervention。

1. **唯一 reportable run 与 frozen inputs。** run id 固定为
   `p1-hoi-d2s-denoiser-response-frontier-s42-20260717`，seed 42，只加载 balanced online
   checkpoint
   `ded9a12d4e85179c37e2457475649ccc614ef364b97eaebade0629b2c11d4ed8` 的 model weights；
   released、EMA、source/current checkpoint、optimizer/scheduler/scaler/RNG state 与 checkpoint
   write 均禁止。使用当前 500-step production posterior coefficients/equation、matched
   text/BPS/pelvis/object-goal/progress、两帧 immutable history，不做 CFG、SO(3) reverse
   projection 或 support clamp，不改变 model、232-D representation、condition API、loss 或
   production sampler 默认行为。
2. **fresh holdout 与真实 reverse parent state。** internal-validation 每 sequence 固定 phase
   offsets `[21,63,105]`，与先前 rollout offsets
   `[0,7,14,28,42,49,56,70,84,91,98,112]` 全部不重合；按
   `SHA256("42:d2s-denoiser-response:" + sequence_name + ":21,63,105")`、sequence name、
   sequence id 排序，取前 64 个同时具备三个 offset 的 sequence，共 192 windows。global
   window-index SHA-256 固定为
   `77d493519b4f7e91a529e3be1b42c3e62d84d045d11bbf24acaab10c6a41a70d`。每个 window 从
   自身 GT 两帧 history 与 matched condition 启动一条完全 unguided production trajectory；
   target timesteps 固定为 `[0,1,10,50,100,250,498]`，对应 parent
   `[1,2,11,51,101,251,499]`。probe 只读取 parent prediction、共享 posterior noise 与
   posterior state；所有 counterfactual candidate 均不得写回 trajectory 或消耗 sampler RNG。
3. **固定 direction、scale frontier 与 no-GT controller。** 在每个 parent prediction 上只计算
   D2-Q/R 锁定的 author hand gradient：24-joint FK palms `22/23`、detached semantic channels
   `0/1 > .95`、`0.02 m` spatial hinge、detached temporal object COM/rotation、batch multiplier、
   outer hand `×10`。完整报告 `author_all` 与 D2-R `upper_raw` 两个 direction；后者只保留
   palm-ancestor rotations `[3,6,9,13,14,16,17,18,19,20,21]`。两个 direction 都固定评估
   scales `[1,0.5,0.25,0.125,0.0625,0]`，不 sweep 其它值。primary controller 只在
   `upper_raw` 中逐 sample 从大到小选第一个同时满足以下条件的非零 scale，否则选 `0`：
   - candidate 下一步 clean prediction 在 **baseline 下一步 prediction 固定的** semantic mask 下，
     per-sample author hand loss 严格低于 baseline；candidate contact channels 不得通过改变 mask
     规避 objective；
   - 对 joint positions、非 upper-chain rotations、object translation、object rotation 与 contact
     五个 protected group，candidate-minus-baseline 下一步 clean-response RMS 均不超过
     `0.25 ×` baseline-next-minus-parent-clean natural-response RMS。若 natural 与 candidate response
     同为零则该 group 通过；natural 为零但 candidate 非零则该 scale 拒绝。
   选择过程不得读取 target `x0`、GT FK/contact、future frame 或 stored per-frame BPS；scale ordering、
   fixed-mask、groupwise response 与最大合格 scale 必须逐 sample replay。
4. **完整 measurements。** 对七个 target timestep、两个 direction、六个 scale 和 controller
   selected candidate 全部报告 input update norm/RMS/max、五个 representation fields、upper/lower
   rotation energy、next-denoiser field response、response/input amplification、相对 scale-1 的
   local-linearity residual、fixed-mask author spatial/temporal/total loss、semantic-mask/contact-channel
   response、selected-scale histogram 与 rejection reason。GT 只在 controller selection 完成后用于
   reference：五个 field MSE、FK-palm 与 direct-hand `2/5/7.5/10 cm` 左/右/union contact、
   FK MPJPE、pelvis/object goal、object translation、object rotation 与 FK foot sliding。
   sequence 为 paired-bootstrap unit，10,000 replicates、seed 42；不得遗漏不利 timestep、scale、
   direction、field 或 subset。
5. **mechanism gate。** 任一 input/checkpoint/author blob hash、selection、unguided RNG、posterior
   formula、history `<=1e-5`、finite、direction/scale completeness、fixed-mask/per-sample-loss sum、
   protected-response、largest-eligible-scale replay、model/state immutability、production sampler
   provenance 或 no-GT selection contract 失败，分类
   `denoiser-response-frontier-contract-failure-stop`。contract 全部通过后，gate 只使用预指定
   low target timesteps `{0,1,10,50,100}` 和 controller selected candidate；至少 4/5 个 timestep
   必须同时满足：
   - nonzero selected-scale fraction `>=0.50`；
   - baseline-minus-selected fixed-mask author loss paired-bootstrap 95% CI 下界 `>0`；
   - selected-minus-baseline FK-palm union 5 cm recall 与 F1 CI 下界均 `>0`；
   - selected/baseline 的 joint-position MSE、object-translation MSE、FK MPJPE、pelvis goal、
     object goal、object-translation MAE、object-rotation geodesic 与 FK foot-sliding paired mean-ratio
     95% CI 上界均 `<=1.05`。
   满足则分类 `denoiser-response-frontier-positive-stop`，否则分类
   `denoiser-response-frontier-negative-stop`。timestep 250/498、`author_all` 和所有固定 scale
   仍为必报描述证据，不能替代 low-t gate 或用于观察后改 primary。
6. **实现、执行与停止。** 新增 analysis-only helper、runner、resolved config、summary 与 tests；
   tests 至少覆盖 target/parent 边界 `0/1`、`498/499`、fixed-mask per-sample/aggregate author formula
   equivalence、upper mask、scale ordering、zero-denominator trust rule、largest-eligible selection、
   candidate batch indexing、paired RNG、history restoration、posterior helper reuse、all-field/scale/
   timestep reporting、finite/no-GT selection 与 production sampler 无 future GT/stored BPS。先单独
   提交本 amendment，再提交 implementation/config/tests；worker 主动 fetch exact committed object，
   用绝对 Python 在 worker-owned persistent session 走 resolved-config/preflight/start/finish/register
   和 immutable recovery。无论 gate 正负，D2-S0 登记后停止；不得在本 session 启动 full-trajectory
   controller、D2-H1、D2-G、smoke/training、official/CHOIS、loss/model/representation/condition
   修改、Phase 1C 或后续阶段。下一 intervention 仍需新的 dated amendment 与用户确认。

D2-S0 implementation entry point 为 `tools/diagnose_hoi_d2s.py`，locked fresh selection、
fixed-mask per-sample author formula、direction/scale packing、protected-response trust rule、
sequence-paired comparison 与 gate 位于 `code/priors/denoiser_response.py`；immutable compact
aggregate 由 `tools/summarize_hoi_d2s.py` 生成。完整 worker artifact 保留逐 window × timestep ×
direction × scale reference/response 与 controller selection，tracked compact result 移除这些 raw
records 但保留全部 aggregate、bootstrap、contract、gate 和 artifact hashes。

#### 2026-07-17 Phase 1B D2-S0 worker-role failure 与 r1 重试 amendment

首次 reportable run `p1-hoi-d2s-denoiser-response-frontier-s42-20260717` 已在 worker-owned
persistent session 启动，但因启动命令遗漏 `INFBAGEL_WORKER_EXPERT=hoi`，立即停在 runner 的
worker-role guard。该进程未加载 checkpoint、未进入 CUDA diagnostic、未处理 window，也未创建
optimizer 或执行 training update；因此分类为
`denoiser-response-frontier-contract-failure-stop`，不构成 D2-S0 mechanism positive/negative
结果，run id 永不复用。failed artifact 已 `finish/register` 并由 worker 主动回收到
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2s-denoiser-response-frontier-s42-20260717`：8 files / 73,551
bytes，worker/authority tree SHA-256 均为
`d8790cb7ff52108ebea24b44c28be0cbf9281f5faf19485ffbbcf3109597503e`；manifest、metrics、resolved
config、preflight、hardware snapshot 与 run-local registry SHA-256 分别为
`2762ffc03b63756212a21089347e5cb4d9d6bff2615f7041a85113e0cdc61380`、
`2ce1a1583a08c0b989de7224eba489e3c09c6ac42a9a0028fba3acbd7ad9db0b`、
`cbffc175a3f27d19dd43d5242e167fa5910133fffe64c0577c19edf1b7afe1df`、
`ae072e9f18e5a897a39b15f0f099660258c2cf83b0842bceec46f815ec4a6af1`、
`ae072e9f18e5a897a39b15f0f099660258c2cf83b0842bceec46f815ec4a6af1` 与
`6f935450157a9a9f7017669375eda49e819ff38bdbea0b256080aad5a3889630`。

授权一次严格 operational retry，唯一新 run id 为
`p1-hoi-d2s-denoiser-response-frontier-r1-s42-20260717`。r1 完整继承上节已锁定的 checkpoint、
selection SHA、192 windows、timesteps/parents、directions、scales、no-GT controller、bootstrap、
measurements、gate、分类与所有禁止项；不得观察后改变任何科学配置。唯一允许的执行差异是，在
resolved-config/preflight/start 之后的 worker-owned persistent workload environment 中同时显式导出
`INFBAGEL_PYTHON=/home/yujinlun/data/envs/infbagel/bin/python` 与
`INFBAGEL_WORKER_EXPERT=hoi`。summary identity 必须接受该精确 r1 id，测试须覆盖原 id 与 r1 id 之外
的 run id 均被拒绝。先单独提交本 amendment，再提交该 summary/config identity 修正与测试；worker
只能 fetch 精确 committed object。无论 r1 gate 正负，仍只登记 D2-S0 后停止，不得启动
full-trajectory controller、D2-H1、D2-G、smoke/training、official/CHOIS 或后续 phase。

D2-S0 r1 已在精确 commit `4003268403ab9515dc8ed0d5540977dc20948745` 上由 node01
`cuda:1` 完成 192/192 windows，return code 0，runtime `313.510 s`。全部 contract checks 通过：
finite、selection、checkpoint/data/normalization/BPS/author blob、paired RNG、两帧 history、production
posterior helper/formula、no-GT selection、candidate 不回写、model/state immutability 与 sampler 无
future GT/stored BPS 均精确；history、posterior、baseline-next 与 candidate replay max abs 均为 `0`，
per-sample author sum replay max abs 为 `1.90735e-6`。五个 low-t gate 结果如下；loss/recall/F1 均为
selected 相对 baseline 的 paired difference，最后一列是八个 protected ratio 的 95% CI 上界最大值。

| target t | nonzero fraction | fixed-mask loss diff 95% CI | FK 5 cm recall diff 95% CI | FK 5 cm F1 diff 95% CI | max protected ratio CI upper | pass |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | 0.3385 | [0.006099, 0.012929] | [-0.000744, 0.003348] | [-0.000702, 0.002081] | 1.002401 | no |
| 1 | 0.3125 | [0.004202, 0.010036] | [-0.001459, 0.001698] | [-0.002402, 0.001338] | 1.001274 | no |
| 10 | 0.2500 | [0.001418, 0.003581] | [0, 0.001116] | [0, 0.000729] | 1.000608 | no |
| 50 | 0.4062 | [0.003611, 0.006842] | [0.000372, 0.002976] | [0, 0.002542] | 1.001238 | no |
| 100 | 0.4531 | [0.005466, 0.010458] | [0, 0.001860] | [0, 0.001523] | 1.001092 | no |

因此 0/5 low timesteps 通过，低于至少 4/5 的 gate，分类
`denoiser-response-frontier-negative-stop`。所有 low-t fixed-mask proxy loss CI 下界均严格大于 0，
且所有 protected ratio CI 上界远低于 1.05；失败来自 nonzero coverage 全部低于 0.50，以及没有任何
low timestep 同时取得严格正的 recall 与 F1 CI 下界。描述性 t=250/498 nonzero fractions 分别为
`0.5521/0.6094`，max protected ratio CI upper 为 `1.002309/1.004106`，但 contact recall/F1 CI 均跨
0，不能替代 low-t gate。`author_all` 的 nonzero coverage 在七个 timestep 仅
`0.0260--0.4948`，也没有提供更强 frontier。

机制结论是：next-denoiser state coupling 可以被该 trust rule 严格限制，但规则在关键低噪声 steps
过于稀疏；即便 fixed-mask author proxy 显著下降，其物理 FK contact effect 仍接近 0。因此 D2-R 的
耦合问题并不能靠这个无 GT 局部 response selector 转化为可用 controller。未授权 full-trajectory
controller、production guidance、checkpoint selection 或 training。

完整 immutable artifact 已由 worker 主动回收到
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2s-denoiser-response-frontier-r1-s42-20260717`：8 files / 1,898,882,887
bytes，worker/authority tree SHA-256 均为
`349dc66bce9ec30d88cfeff8883fd28d2bc36b1c7645e777c153be3901f33fc3`。manifest、metrics、resolved
config、preflight/hardware snapshot 与 run-local registry SHA-256 分别为
`ac37180f6bd6df9c47c13ace3f1a13b760ecfb813fdf96fc28e0836d07e47631`、
`aebb94792525f9a00e4353de91385d920b0300dcb4575c456176e42acb27a068`、
`37d3ab8704405b486fe4a2eba277a5d25060ec8d330c50ed34fd75c2ea9c95ea`、
`9691dcdbee737850ce8b0c5cae0694b474256dfae84918ddc1744498ae1d7701` 与
`830a9f9d8a6422fa1cdbd4430a84c74f02df9557800729b160d26e8588b1294b`。compact aggregate 为
`experiments/results/p1_hoi_phase1b_d2s_denoiser_response_frontier_r1_s42_20260717.json`，SHA-256
`5aa52fd75d65e0caa7439821e9e79eac291987656036a332da1f32389be1e479`。本 session 到此停止；未启动
D2-H1、D2-G、smoke/training、official/CHOIS 或后续 phase，任何新 intervention 必须另做 dated
amendment 并等待用户确认。

#### 2026-07-21 Phase 1B D2-T author-DDPM update-rule parity screen 预注册

作者自主训练的 InfBaGel diffusion epoch500 已在 official 438-sequence HOI evaluator 上形成有效
HOI 能力，而 D2-H 已证明当前 HOIPrior 存在 model-parent reverse-state exposure 放大。两项证据共同
否定“一致性蒸馏是基础能力的唯一来源”，但尚未区分当前 HOIPrior 的失败主要来自 model/condition/
loss 机制，还是此前与作者 DDPM 明显不同的 update-rule contract。D2-T0 因而只把训练更新规则对齐
到作者 DDPM；保持 HOIPrior 的独立 232-D prior、network、conditions、diffusion、loss、data 与 sampler
不变。它不是作者网络移植，不加载 released 或自主训练作者 checkpoint，也不授权 consistency stage。

1. **唯一 manipulated factor。** control 为 sealed R1024 HOIPrior；intervention 固定为
   `p1-hoi-d2t-author-update-rule-s42-20260721`、subphase `1B-D2-T0`、seed 42。D2-T 将一组不可拆分的
   optimizer/update-rule contract 作为唯一 factor：4×RTX 3090、per-GPU batch 512、gradient
   accumulation 1、effective batch 2048、Adam (`betas=(0.9,0.999)`、weight decay 0)、constant
   LR `1e-4`、FP32、无 warmup/cosine、无 global gradient clipping、无 EMA selection，只保存/评估
   online model。该组合同来自作者 DDPM 的实际训练更新规则；不得只挑其中观察后有利的子项。
2. **fixed controls。** 保持 `WindowStateCodec` 与 232-D representation、16-frame window、2-frame
   immutable history、scene-free independent HOIPrior、512-wide/16-head/8-layer network、现有 timestep/
   text/BPS/goal/progress/history condition routing、500-step x0 DDPM、训练 split、seed 42、random
   initialization、FK/surface/velocity/terminal-goal weights `50/50/0.1/1`、normalization、dataset sampler
   与 processed-window budget `6,144,000` 不变；对应 3,000 optimizer updates。validation 固定
   32,768 windows，在 3,072,000 与 6,144,000 processed windows 评估；checkpoint 同步保留 midpoint
   与 terminal online weights。禁止 CFG、dynamic perception、contact/bump guidance、任何 sampler
   intervention、released/author checkpoint、旧 HOIPrior initialization/resume、CM/teacher/student/target。
3. **execution ownership 与 lifecycle。** authority/integration host `10.184.17.253` 只做开发、CPU/static
   tests、commit、manifest recovery 与未来 HSI 工作，明确禁止运行任何 HOIPrior CUDA workload。
   D2-T 只能在 `infbagel-4gpu`（worker `10.181.9.214`）执行；worker 必须主动 fetch 精确 commit，使用
   `/home/yujinlun/data/envs/infbagel/bin/python`，显式设置 `INFBAGEL_PYTHON` 与
   `INFBAGEL_WORKER_EXPERT=hoi`。先归档 fully resolved Hydra config，再由 clean worker checkout 运行
   `tools/experiment.py start`，核验四张 RTX 3090、无 contention、commit/config/assets hashes 与 worker
   role 后，才可在 worker-owned persistent session 启动。旧 run id、脏 checkout、authority CUDA、
   非四卡 topology 或缺少 lifecycle manifest 均为 contract failure；不得静默降 batch、改 accumulation
   或在当前服务器代跑。
4. **mechanism gate。** 使用与 sealed R1024 相同的 author-native HOI evaluation protocol、official
   438 sequences、500-step unguided diffusion loop、online weights、sequence-paired 10,000-replicate
   bootstrap（seed 42）。除 finite、checkpoint/config/hash/lifecycle、official-438、paired-unit 和禁止
   checkpoint-load contracts 全部通过外，D2-T 相对 sealed R1024 必须同时满足：MPJPE、end-object
   error、xy error、object-translation error 的 paired improvement 95% CI 下界均 `>0`；contact F1
   difference CI 下界 `>=-0.02`；foot-sliding D2-T/R1024 paired mean-ratio CI 上界 `<=1.10`。
   任一失败分类 `author-update-rule-negative-stop`，不选择 checkpoint、不进入 CM。
5. **effective-diffusion gate。** mechanism gate 通过后，再相对 released Phase-0 author-native baseline
   检查 MPJPE ratio `<=1.30`、end-object ratio `<=2.00`、xy ratio `<=1.50`、object-translation ratio
   `<=1.50`、foot-sliding ratio `<=1.10` 且 contact F1 `>=0.60`。mechanism pass 但该绝对 gate 失败，
   分类 `positive-but-not-effective-diffusion-stop`；两 gate 全通过才可称为 usable diffusion HOIPrior。
   单 training seed 只支持该 preregistered mechanism screen，不得外推训练方差或 main-table 稳健性。
6. **实现、测试、artifact 与停止规则。** 新增独立 D2-T config，并对 optimizer class、constant LR、
   FP32、no clipping、no EMA、online validation/checkpoint、3,000 updates、random-only initialization、
   worker host/role 和 model/data/diffusion/loss source invariance 加 fail-closed guards/tests。checkpoint
   必须记录 optimizer/scheduler/clipping/AMP/primary-weight metadata，且 `ema_models={}`、不写 legacy
   `ema_model`。artifact 至少包括 manifest、resolved config、preflight/hardware snapshot、training log、
   metrics/state、midpoint/terminal checkpoint 与 hashes、native evaluation aggregate/per-sequence records、
   bootstrap/gate summary 和 append-only registry completion。任何确定实现 defect、OOM、NaN、hash/
   lifecycle failure 或 gate negative 均登记全部负结果并停止；不得观察结果后改变 update-rule factor、
   训练预算、checkpoint variant、evaluator 或阈值。只有 effective-diffusion gate 通过后，才可另行提出
   并预注册从自主训练 HOIPrior diffusion checkpoint 开始的 consistency stage。

#### 2026-07-21 D2-T display-only Xorg idle preflight clarification

D2-T implementation commit `baa01bf73b6e151693f949fe943505bf8eda3410` 已在 authority 与
`infbagel-4gpu` clean checkout 通过 207 项测试（worker 按 HOI-only contract 跳过 2 项真实 LINGO
资产测试），且尚未执行 `tools/experiment.py start`、GPU smoke 或 training。首次与第二次 immutable
preflight 分别为 `preflight.json` 和 `preflight_r1.json`，SHA-256
`b0f35ae89511eb2dc1e88d04c56fd244bab8a75a9078fa8820bf1bebdd5b8047` 与
`9626a12af3c08f7def479e75dec51b969108517ef30a0101e61242909f320527`；二者除
`four_gpu_idle` 外所有 checks 均通过。GPU 0 只有 Xorg graphics PIDs `2552/3224`，无 CUDA compute
process，P8、约 96--100 MiB used、memory utilization 0%；14 秒 71 samples 的 GPU utilization
min/avg/max 为 `0/0/6%`，但 `nvidia-smi` 单点值稳定显示 `1%`。因此失败来自 preflight 将瞬时
utilization 写死为精确 0%，不是训练 contention。

用户明确确认该长期存在的 Xorg 占用可忽略。仅作 operational clarification：四卡 idle 仍要求
4×RTX 3090、CUDA compute-process 列表为空、每卡 memory used `<=128 MiB`，并把瞬时 GPU
utilization 上限从 `0%` 调整为 `1%`，用于容纳 display-only Xorg driver floor。不得放宽到用户提及
的 10 GiB 训练峰值，不得容纳任何 compute process、P2/P0 训练状态或 `>128 MiB` 的外部 allocation。
preflight 输出必须记录该 tolerance 和逐卡 checks；新增 tests 后提交新的 exact commit，worker 主动
fast-forward，再写入不覆盖旧失败证据的 `preflight_r2.json`。只有 r2 全部通过，才可创建原 D2-T
run id 的 lifecycle manifest 并启动原预注册 workload；scientific config、run id、seed、budget、gates
和所有禁止 checkpoint-load 项均不变。

#### 2026-07-21 D2-T author-native evaluation lifecycle amendment

D2-T training 已在 commit `6d57d34e27aeb748039896052a861fdda386873e` 完整结束：returncode
数值 0、3,000 updates、6,144,000 windows、final online checkpoint SHA-256
`1543af304acf76f385dbd3656a1ca82ea25dcd504ee120f7f63e821d71483647`。training manifest 已完成，
因此不得重新打开或复用其 run id 来承载 GPU evaluation。原预注册 scientific gate 不变，新增唯一
evaluation lifecycle run id `p1-hoi-d2t-native-eval-s42-20260721`，subphase `1B-D2-T0-eval`。

评估只加载 D2-T final checkpoint 的 `model`/online weights，不加载 optimizer、scheduler、RNG、EMA、
released 或 author checkpoint。使用现有 `code/test_infbagel_hoi.py` official 438 author-native protocol、
3 windows/sample、500-step unguided diffusion、seed 42、无 CFG/dynamic perception/guidance、无 CHOIS
export、无 FID/R-precision gate。evaluator static hashes 锁定为 test script
`22886f8797ceb04a892487393dea9f80e19877bc02dd7a6f39127e7319119524`、metrics
`445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547`、config
`89c702d96b98289924225c4b163d3b29eb22efe27c50ac799ddd0c71c515aa73`。

paired control 不重复生成，复用 D2-N 已完成且同 evaluator/seed/order/protocol 的 sealed R1024 online
records：aggregate SHA-256 `d95d3090455e763159a4cac793301f9f4744837bf60b4ed21eaef4a141c9ad2b`、
per-sequence SHA-256 `11c11fcd90c0ce2e67d705bb64c3a78bbe2b0e9f84aff7fcb57cab25087e2a1f`，
438 sequences / 3 windows，checkpoint SHA-256
`ded9a12d4e85179c37e2457475649ccc614ef364b97eaebade0629b2c11d4ed8`。任何 hash、sequence identity、
metric set、normalization、online-weight 或 official-count mismatch 均分类
`author-update-rule-contract-failure-stop`。

mechanism gate 逐 sequence、10,000 bootstrap、seed 42：control-minus-D2-T 的 MPJPE、end-object、xy、
object-translation improvement CI 下界均 `>0`；D2-T-minus-control contact F1 CI 下界 `>=-0.02`；
D2-T/control foot-sliding ratio CI 上界 `<=1.10`。失败分类 `author-update-rule-negative-stop`。
若通过，再相对 Phase-0 released aggregate（JSON SHA-256
`76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6`）检查原锁定 absolute ratios 与
contact F1 `>=0.60`；absolute gate 失败分类 `positive-but-not-effective-diffusion-stop`，两 gate 均通过
分类 `effective-diffusion-hoi-prior-stop`。评估完成后 finish/register/recover 全部 artifacts 并停止；
无论结果如何都不在本 amendment 启动 CM、第二次训练、checkpoint selection 或 sampler intervention。

#### 2026-07-21 D2-T native evaluation display-only preflight clarification

evaluation lifecycle 在任何 GPU workload 启动前保留了两个失败 preflight：`preflight.json` SHA-256
`9435c8cfc3e31e4198263331ebb6b7e9d4403c3349e4a171a1e32cd003a5c3f7` 与 `preflight_r1.json`
SHA-256 `3215d8ae53379ac2ffa00c9d01812b8a522da75aa427ae746e9a8411af73189a`。两次均只有 GPU0
display Xorg floor：memory 100 MiB、P8、compute-process list 为空；其余全部 checks 通过，但瞬时
utilization 分别为 10% 和 7%。随后 30 次连续只读采样显示 GPU0 为 7--10%，其余 GPU 为 0%，
说明原 training-oriented `<=1%` 瞬时门槛会稳定误拒绝单卡 evaluation。

按用户明确授权忽略 Xorg 占用，只为本次 evaluation preflight 新增显式 opt-in
`--allow-gpu0-display-utilization`：仍要求四张卡均为 RTX 3090、compute-process list 为空、每卡
memory `<=128 MiB`、P8，且 GPU1--3 utilization `<=1%`；仅 GPU0 的 display utilization 不进入
idle 判定。默认 preflight 和训练仍使用原 `<=1%` 规则，绝不把该 opt-in 用于训练。下一次只写
`preflight_r2.json`，不得覆盖前两次失败。resolved target/config、run id、device `cuda:0`、checkpoint、
evaluator、seed、438 sequences、scientific gates 均不变；不授权训练、CM 或 sampler intervention。

#### 2026-07-21 D2-T author update-rule parity completion

D2-T 训练与独立 evaluation lifecycle 均已完成并回收到 authority。训练从随机初始化开始，只把
HOIPrior 的 optimizer/update-rule contract 对齐作者 DDPM：4×RTX 3090、per-GPU batch 512、effective
batch 2048、Adam、LR `1e-4`、无 scheduler/warmup/weight decay/clipping/AMP/EMA，固定 3,000 updates、
6,144,000 windows；232-D representation、HOIPrior architecture/conditions/losses/data/sampler 均保持不变。
final online checkpoint SHA-256 为
`1543af304acf76f385dbd3656a1ca82ea25dcd504ee120f7f63e821d71483647`，训练 finite 且 final validation
total loss 2.954167。

official-438 author-native、3 windows/sequence、500-step unguided diffusion 评估得到：MPJPE 34.7367、
end-object 38.5563、xy 17.6001、object translation 57.4467、foot sliding 0.1761、contact F1 0.2764。
相对 sealed R1024 control，四项 lower-is-better paired improvement CI 全为负，contact F1 difference
95% CI 为 `[-0.0957,-0.0294]`；只有 foot-sliding ratio gate 通过。因此 mechanism/effective-diffusion
gate 均失败，分类 `author-update-rule-negative-stop`，checkpoint 不选择，CM 不授权。

compact aggregate 为
`experiments/results/p1_hoi_phase1b_d2t_author_update_rule_s42_20260721.json`，SHA-256
`0d211dd59bff79addcf1848da6ffaa5f4076a66d666e1c9d04cc086611933529`。training/evaluation authority
staging tree SHA-256 分别为
`9173d340ab66dffb08b73ba04b177bdaf171377a7e2cd484e361821d20729387` 与
`e77bb8e6d9bfb112a6bf34e142baeab672ab34a9a6ad5ede4073f199692572dc`。该结果否定“仅 update-rule parity
即可得到有效 diffusion HOIPrior”，但不否定 HOIPrior；下一训练方向必须重新预注册并针对仍未隔离的
architecture/condition/loss mechanism。不得从 D2-T checkpoint 启动 CM 或进行观察后 checkpoint selection。

#### 2026-07-21 Phase 1B D2-U from-random balanced-objective screen 预注册

D2-T 已用随机初始化、作者 DDPM update-rule contract 和 6,144,000-window 固定预算排除“只要把
Adam/LR/effective batch/FP32/无 EMA 等更新规则对齐即可学成 HOIPrior”：其 official-438
MPJPE/end-object/xy/object-translation 均显著差于 sealed balanced control。另一方面，D2-I 的
parameter-gradient audit 显示原 `50×FK + 50×object-surface` 在高噪声时令 auxiliary/reconstruction
gradient-norm ratio 约为 90--126；D2-L 证明锁定的 balanced 权重能直接修正 raw gradient routing，
D2-M/D2-N 则显示 balanced objective 可大幅改善已有模型的 kinematics/object-goal，但 D2-M 只有
64 updates 且受 AMP overflow divergence 与旧训练 lineage 限制。D2-U 因而只检验 H3 loss geometry：
在 D2-T 的 clean from-random update contract 上替换这两个已预先锁定的权重，不改变 architecture、
condition、representation、data、diffusion、sampler 或 optimization。

1. **唯一 manipulated factor。** intervention run id 固定为
   `p1-hoi-d2u-balanced-author-update-s42-20260721`、subphase `1B-D2-U0`、seed 42。相对已完成
   D2-T，唯一改变为 `fk_weight: 50.0 -> 0.3569973401779424` 与
   `object_surface_weight: 50.0 -> 0.4772322188400037`；`velocity_weight=0.1`、
   `goal_weight=1.0` 不变。权重来自 D2-I locked aggregate，不根据 D2-U 结果重新调参。
2. **固定训练 contract。** 必须从随机初始化开始，保留 232-D state、16-frame window、2-frame
   history、512-wide/16-head/8-layer Transformer、500-step x0 diffusion、原 data/split/condition/
   sampler。只在四卡 `infbagel-4gpu/node01` 运行：4×RTX 3090、batch/GPU 512、accumulation 1、
   effective batch 2048、Adam `(0.9,0.999)`、constant LR `1e-4`、FP32、无 warmup/scheduler/
   weight decay/clipping/AMP/EMA，3,000 updates、6,144,000 processed windows，midpoint/final
   checkpoint 与 validation cadence 完全复用 D2-T。禁止加载 released、author、source、current、
   balanced、D2-T、prior、resume 或任何 EMA checkpoint；不得设置 `d2m_candidate`。
3. **实现和 lifecycle。** 新增独立 D2-U Hydra config 与 fail-closed mode；D2-T exact contract
   必须保持不变。代码、config、tests、plan、registry 和 target-only evaluator/gate runner 组成一个
   logical commit。authority 完成 registry validation、targeted/full tests、compile 与 diff check 后，
   worker 只能 worker-initiated fast-forward 到 exact commit。训练前必须归档无 unresolved
   interpolation 的 resolved config、同一 escalated context 的四卡 preflight，并通过
   `tools/experiment.py start` 创建不覆盖旧结果的 manifest；detached workload 不依赖 SSH 存活。
4. **mechanism gate。** D2-U final online checkpoint 使用与 D2-T 完全相同的 author-native HOI
   official-438、每序列三窗口、500-step unguided diffusion evaluator；独立 evaluation lifecycle
   run id 固定为 `p1-hoi-d2u-native-eval-s42-20260721`、subphase `1B-D2-U0-eval`。逐 sequence、10,000 paired
   bootstrap、seed 42：`D2-T minus D2-U` 的 MPJPE、end-object、xy、object-translation improvement
   CI 下界均须 `>0`；`D2-U minus D2-T` contact-F1 CI 下界须 `>=-0.02`；D2-U/D2-T
   foot-sliding ratio CI 上界须 `<=1.10`。D2-T per-sequence records 只读复用，不重新生成。
5. **absolute effective-diffusion gate。** 同时相对 Phase-0 released aggregate 满足 MPJPE ratio
   `<=1.30`、end-object `<=2.00`、xy `<=1.50`、object-translation `<=1.50`、foot-sliding
   `<=1.10` 且 contact F1 `>=0.60`，才分类为 `effective-diffusion-hoi-prior-stop`。mechanism
   gate 通过但 absolute gate 失败，分类 `balanced-objective-positive-but-not-effective-stop`；任一
   mechanism 条件失败分类 `balanced-objective-negative-stop`；contract/hash/lifecycle 失败分类
   `balanced-objective-contract-failure-stop`。
6. **停止规则和 artifact contract。** 无论结果如何，D2-U 训练与 official evaluation 后停止；不得
   自动延长预算、选择 midpoint、修改 contact loss、启动 architecture/condition intervention 或 CM。
   必须保留 resolved config、preflight、manifest、日志、loss/validation、midpoint/final checkpoint
   hashes、RNG/checkpoint-resume 证据、official aggregate/per-sequence/bootstrap/gate JSON、artifact-tree
   hash 和 negative result。只有 absolute gate 通过后，用户才可另行授权选择 final checkpoint，并以
   新 dated amendment 讨论从自主训练 HOIPrior diffusion checkpoint 开始的 consistency stage。

#### 2026-07-21 D2-U balanced-objective completion

D2-U 已在 clean worker commit `c4293735b7a144ddf7a1190e8ecf6e43b9698d18` 完成。训练严格从
随机初始化开始，未加载 released、author、source、current、balanced、D2-T 或任何 prior
checkpoint；固定 D2-T 的 4×RTX 3090、effective batch 2048、Adam、LR `1e-4`、FP32、无
scheduler/warmup/clipping/AMP/EMA 和 6,144,000-window 预算，只把 FK/object-surface 权重改为
`0.3569973401779424/0.4772322188400037`。3,000 updates 全部 finite，无 overflow；final validation
total 为 `0.0822169`，final checkpoint SHA-256 为
`7cb379263f8a72e7f9017e4ada9d521a9e25f7c160c061305a92b9822bda2cad`。

唯一一次 official-438、3 windows/sequence、500-step unguided diffusion evaluation 得到：MPJPE
`17.0285`、end-object `10.0201`、xy `9.5509`、object translation `27.1250`、foot sliding
`0.3101`、contact F1 `0.3391`。相对 D2-T，MPJPE/end-object/xy/object-translation 的 paired
improvement 95% CI 分别为 `[17.2038,18.2254]`、`[26.8360,30.3090]`、
`[7.6452,8.4718]`、`[28.6225,32.0440]`，contact-F1 difference CI 为
`[0.0323,0.0943]`；这直接支持 H3 loss geometry 是主要机制之一。但 foot-sliding ratio CI 为
`[1.6229,1.9071]`，超过 `<=1.10` 保护门槛，且相对 Phase-0 released baseline 的 absolute
MPJPE/end-object/xy/object-translation/contact-F1 gates 仍失败。因此按预注册分类为
`balanced-objective-negative-stop`，不选择 checkpoint、不启动 CM。

D2-U 与此前 sealed balanced checkpoint 的 descriptive comparison 为：MPJPE `17.0285 vs
18.7640`、end-object `10.0201 vs 10.8458`、object translation `27.1250 vs 29.7522` 更好；
xy `9.5509 vs 7.9996`、foot sliding `0.3101 vs 0.2934` 更差；contact F1 `0.3391 vs 0.3386`
基本相同。故从随机训练的 balanced objective 已复现并略推进既有 balanced checkpoint 的主要
kinematic/object 能力，但尚未形成可用 HOIPrior，剩余缺口集中为 foot/contact 与 trajectory
质量的训练期 Pareto，而不是 update rule 或 sampler-only contact heuristic。

训练/evaluation artifacts 已由 worker 主动回收到 authority staging；固定 `LC_ALL=C` tree
SHA-256 分别为
`f480ab0223beb998821932f495f882d3067c2d6ab21ddb5fa7578c09ff43a670` 与
`67704f86e69c12f38d6a0c9382d4ac383396c384c955d81f9264450afc1ce7c3`。compact aggregate 为
`experiments/results/p1_hoi_phase1b_d2u_balanced_author_update_s42_20260721.json`。D2-U 至此停止；
任何下一训练机制必须另行 dated plan/registry amendment，且不得从本 checkpoint 自动启动 CM。

#### 2026-07-22 Phase 1B D2-V from-random balanced long-budget screen 预注册

D2-U 已直接证明 balanced loss geometry 能把 clean from-random HOIPrior 的主要 kinematic/object
指标推进到或略优于此前 sealed balanced checkpoint，但在 6,144,000 processed windows 终止时
internal validation 仍明显下降，且作者自主训练 DDPM 使用了远长于该 screen 的训练周期。D2-V
因此只检验 H4 training-budget insufficiency：在 D2-U 完全相同的 232-D diffusion HOIPrior、data、
condition、architecture、loss、optimization 和 evaluation contract 下，把固定预算扩大十倍。它不
检验 consistency、dynamic perception、sampler guidance、architecture 或 condition routing。

1. **唯一 manipulated factor。** training run id 固定为
   `p1-hoi-d2v-balanced-long-budget-s42-20260722`、subphase `1B-D2-V0`、seed 42。相对 D2-U，
   唯一变化为 `max_processed_windows: 6,144,000 -> 61,440,000`，对应 optimizer updates
   `3,000 -> 30,000`。D2-V 必须重新随机初始化；禁止 resume D2-U 或加载其 checkpoint，因此
   “更长预算”不与 checkpoint lineage 混杂。
2. **固定训练 contract。** 完全保留 D2-U 的 232-D state、16-frame window、2-frame history、
   512-wide/16-head/8-layer Transformer、500-step clean-x0 diffusion、固定 OMOMO split、全部
   conditions、FK/object-surface/velocity/goal 权重
   `0.3569973401779424/0.4772322188400037/0.1/1.0`。只在四卡
   `infbagel-4gpu/node01` 运行：4×RTX 3090、batch/GPU 512、accumulation 1、effective batch
   2048、Adam `(0.9,0.999)`、constant LR `1e-4`、FP32、无 warmup/scheduler/weight decay/
   clipping/AMP/EMA。validation windows 32,768，validation/checkpoint cadence 均保持 D2-U 的
   3,072,000 windows。禁止加载 released、author、source、current、balanced、D2-T、D2-U、
   prior、resume 或任何 EMA checkpoint；不得设置 `d2m_candidate`。
3. **实现和 lifecycle。** 新增独立 D2-V Hydra config、fail-closed mode、tests 和 target-only
   evaluator；D2-T/D2-U exact contracts 必须保持不变。代码、config、tests、plan、registry 和
   evaluator 组成一个 logical commit。authority 通过 registry validation、targeted/full tests、
   compile、resolved-config 和 semantic diff 后，worker 只能主动 fast-forward 到 exact commit。
   workload 前必须归档 fully-resolved config、同一 escalated context 的四卡 preflight，并用
   `tools/experiment.py start` 建立新 manifest；detached workload 不依赖 SSH 存活。
4. **budget mechanism gate。** final online checkpoint 只评估一次，evaluation run id 固定为
   `p1-hoi-d2v-native-eval-s42-20260722`、subphase `1B-D2-V0-eval`。复用 D2-U official-438
   per-sequence records 作为只读 control，使用相同每序列三窗口、500-step unguided diffusion、
   10,000 paired bootstrap、seed 42。`D2-U minus D2-V` 的 MPJPE、end-object、xy、
   object-translation improvement CI 下界均须 `>0`；`D2-V minus D2-U` contact-F1 CI 下界须
   `>=-0.02`；D2-V/D2-U foot-sliding ratio CI 上界须 `<=1.10`。
5. **absolute effective-diffusion gate。** 同时相对 Phase-0 released aggregate满足 MPJPE ratio
   `<=1.30`、end-object `<=2.00`、xy `<=1.50`、object-translation `<=1.50`、foot-sliding
   `<=1.10` 且 contact F1 `>=0.60`，才分类为 `effective-diffusion-hoi-prior-stop`。budget
   mechanism gate 通过但 absolute gate 失败，分类
   `long-budget-positive-but-not-effective-stop`；任一 mechanism 条件失败分类
   `long-budget-negative-stop`；contract/hash/lifecycle 失败分类
   `long-budget-contract-failure-stop`。
6. **停止规则和 artifacts。** 无论结果如何，完成 D2-V fixed-budget training 与一次 official
   evaluation 后停止；不得观察中间 checkpoint 后提前选择、延长预算、改变 loss/contact、启动
   architecture/condition intervention 或 CM。保留 resolved config、preflight、manifest、完整
   log/metrics、全部 cadence checkpoint hashes、resume evidence、official aggregate/per-sequence/
   bootstrap/gate JSON、artifact-tree hash 和任何 negative result。只有 absolute gate 通过后，
   才能等待用户另行授权选择 final checkpoint；任何 consistency stage 都必须以新的 dated
   amendment 从自主训练的 D2-V HOIPrior diffusion checkpoint 开始，本 subphase 不授权 CM。

#### 2026-07-22 D2-V balanced long-budget completion

D2-V 已在 clean worker commit `91e54654b6d9aa05e7cfca384ea7ab488018c298` 完成。D2-U 与
D2-V 的 initial model-state SHA-256 均为
`ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e`；data、model、condition、
loss、optimizer、seed 和 sampling 均相同，唯一变化是从 6,144,000/3,000 增至
61,440,000 processed windows/30,000 updates。因此这是严格 budget-only comparison，而不是从
D2-U resume。训练约 108.08 epochs，全部 loss/gradient finite，无 AMP/EMA；20 个固定 cadence
checkpoints 全部保留。final validation total 为 `0.0489330`，固定 final checkpoint SHA-256 为
`e0705681bbaeed40d353494852494d8b7bdaf4d32da92368c0d2ceedea4c01a4`，吞吐
`3,200.83 windows/s`。internal validation 最低点在 24,576,000 windows (`0.0449481`)，仅作
descriptive evidence；没有选择该中间 checkpoint，official evaluation 仍只使用预注册 final。

唯一一次 official-438、3 windows/sequence、500-step unguided diffusion evaluation 得到：MPJPE
`12.1224`、end-object `3.6807`、xy `4.0103`、object translation `16.1082`、object rotation
`1.0245`、foot sliding `0.3783`、contact recall/F1 `0.5853/0.6286`。相对 D2-U，四个
lower-is-better 指标的 paired improvement 95% CI 分别为 `[4.4763,5.3445]`、
`[5.8280,6.8378]`、`[5.1792,5.9114]`、`[10.1110,11.9449]`，contact-F1 difference CI 为
`[0.2576,0.3218]`。这直接支持 H4 training-budget insufficiency 是主要机制之一：当前
architecture/condition/232-D representation 能在不经过 CM 的情况下学到接近 released baseline 的
kinematic、object-goal 和 contact 能力。

但 D2-V/D2-U foot-sliding ratio CI 为 `[1.1232,1.3275]`，超过预注册 `<=1.10`；相对 released
baseline 的 foot-sliding ratio 为 `1.1347`，也超过 absolute `<=1.10`。其余 absolute
MPJPE/end-object/xy/object-translation gates 均通过且 contact F1 `>=0.60`。此外非 gate 的
descriptive protection 指标也恶化：hand/human penetration loss 为 `0.2641/4.1712`，分别是
released baseline 的 `1.6259×/1.6110×`。因此必须按预注册分类
`long-budget-negative-stop`，不选择 checkpoint、不启动 CM，也不以 favorable subset 或中间
checkpoint 改判。科学上它是当前最强的 from-random HOIPrior capability evidence，但尚不是 sealed
usable expert；剩余缺口已收缩为强 denoiser 上的 foot-sliding/contact/penetration protection，而非
基本 denoiser 无法学习。

training/evaluation artifacts 已由 worker 主动回收到 authority
`/data/yujinlun/InfBaGel-p1b-staging/`；标准 directory tree SHA-256 分别为
`fe1d9ec82369864ab6b3dfd334d01d07c58832027b56f25435e3d8074a15c6f3`（113 files / 7,127,146,664
bytes）与 `05b8a9c6fc0c1021c3fbc2a09308b80246eadee6efb9e4221eb0baf4477c6bb8`（16 files /
362,375 bytes），worker/authority 完全一致。compact aggregate 为
`experiments/results/p1_hoi_phase1b_d2v_balanced_long_budget_s42_20260722.json`。D2-V 至此停止；
任何下一 foot-sliding/penetration training intervention 必须另行 dated plan/registry amendment，
且 consistency 仍不授权。

#### 2026-07-22 Phase 1B D2-W fixed-checkpoint FK foot-sliding frontier diagnostic 预注册

D2-V 已证明 10x budget 能把独立 232-D diffusion HOIPrior 的 MPJPE、object-goal、object-translation
和 contact 提升到接近 released baseline，但 fixed final 的 official foot-sliding 仍为 `0.3783`，相对
D2-U 的 paired ratio CI 为 `[1.1232,1.3275]`。训练期 direct normalized velocity validation loss 同期
从 `0.0010924` 降至 `0.0001033`，而 official evaluator 通过 predicted rotations、root translation 和
rest offsets 做 FK 后再计算近地面 ankle/toe 水平位移；production velocity loss 则只读取 direct
joint-position 与 object-translation channels。D2-W 在任何新训练前，只判断 foot-sliding 缺口主要是
`24.576M -> 61.44M` constant-LR 后半程退化，还是在能力形成时已经存在的 rotation/FK trajectory
loss-geometry 缺口。

1. **唯一 reportable diagnostic。** run id 固定为
   `p1-hoi-d2w-checkpoint-frontier-s42-20260722`、subphase `1B-D2-W0`、seed 42，只能在
   `infbagel-4gpu/node01` 运行。加载同一 D2-V run 的三个 online model checkpoints，仅用于 inference：
   `6,144,000` windows（file SHA-256
   `be8233c0a4c013d973c4140ba5c1f472332f1fdd6be8efa21585deeb250506d3`，model-state SHA-256
   `cfcb5836129d177bf57c60ffd8669ee4516fad77f52b58afd037d063e9aaa0c7`）、预先由 teacher-forced
   internal validation total 唯一确定的 `24,576,000` windows（file SHA-256
   `efab7f55d6a719ac85659de0aa66c2f94235e1875ae5e6951e9c4334017ee9a3`，model-state SHA-256
   `1ee340962d158e12a31d3ad081da37886cd8bbc3eddd80b523de4eb236ba2735`）和 fixed final
   `61,440,000` windows（file SHA-256
   `e0705681bbaeed40d353494852494d8b7bdaf4d32da92368c0d2ceedea4c01a4`，model-state SHA-256
   `f7d134ac98ede806abae322c77816ef21ace427e3905a4cb5e1d4a2a2b4b89fc`）。D2-V 6.144M
   model state 已只读验证与 D2-U final 完全相同；不得加载 optimizer/scheduler/scaler/RNG/EMA，
   不得写 checkpoint 或执行 optimizer update。
2. **固定 internal-only rollout。** 复用 D2-M 在 D2-V 前已经 sealed 的 internal-validation
   three-window native holdout：eligible sequence ranks `128--159`、32 sequences、96 windows、selection
   SHA-256 `30524c88481f6cb81e8063073d510ad01543be92d91eb4ef9b2b8a376cc4fbae`。三个 checkpoint
   必须使用逐 step 完全相同的 500-step diffusion noise、matched text/BPS/goal/progress、generated-history
   handoff 和 current generated-object BPS；只用 online weights，无 CFG、dynamic perception、guidance、
   CHOIS、official-438 或 released/author checkpoint。
3. **指标与 parity。** 每个 checkpoint 必报 per-sequence/aggregate：direct-joint foot sliding、
   rotation-to-FK foot sliding、FK MPJPE、pelvis/object goal error、object translation/rotation error、physical
   contact precision/recall/F1，以及 direct-vs-FK foot trajectory disagreement。FK 使用与 production loss
   相同的 24-joint parents/rest offsets，脚滑使用 official evaluator 相同 ankle/toe indices、Y-up、近地面
   height thresholds 与水平 displacement 公式。D2-W 还必须以 synthetic parity test 证明 torch 实现与
   `code/eval_metrics.py::compute_foot_sliding_for_smpl` 一致，并证明三个 checkpoint 的 q-noise hashes
   完全相同。
4. **唯一 gate。** 10,000 次 paired sequence bootstrap、seed 42。`24.576M` 只有同时满足以下条件才分类
   `midbudget-protection-supported-stop`：`61.44M minus 24.576M` 的 FK foot-sliding difference CI 下界
   `>0`；`24.576M / 61.44M` 的 FK-MPJPE、pelvis-goal、object-goal、object-translation ratio CI 上界均
   `<=1.10`；`24.576M minus 61.44M` physical-contact-F1 CI 下界 `>=-0.02`；且相对 6.144M，
   FK-MPJPE、object-goal 和 object-translation 的 improvement CI 下界均 `>0`。任一失败分类
   `midbudget-protection-negative-stop`；contract/hash/parity/lifecycle 失败分类
   `midbudget-protection-contract-failure-stop`。direct/FK divergence 和 penetration coupling 只作机制解释，
   不得改写 gate。
5. **停止与后续。** 本 subphase 不选择任何 D2-V checkpoint，不使用 internal 或 official 指标追认
   favorable intermediate，不修改 production loss/model/condition/sampler，不训练，不启动 CM。若 gate
   通过，下一 dated proposal 只能测试 from-random fixed mid-budget/learning-rate schedule；若 gate 失败，
   下一 dated proposal 才能测试一个 evaluator-aligned FK-foot temporal loss，并保持 D2-V 的 61.44M
   budget。两条路径都必须重新获得用户确认；penetration 的 smallbox/suitcase contact-collision tradeoff
   保留为后续独立假设，禁止与 foot intervention 捆绑。

#### 2026-07-22 Phase 1B D2-W worker interpreter symlink implementation amendment

D2-W implementation commit `21580b66b4af0c79ca54940afb908df34ff4a4a4` 发布后、任何 run directory、
manifest、checkpoint load 或 GPU workload 创建前，只读 worker preflight 发现 lifecycle guard 的确定缺陷：
规范解释器 `/home/yujinlun/data/envs/infbagel/bin/python` 是指向 `python3.8` 的符号链接，
`sys.executable` 保留前一路径，而 `Path(sys.executable).resolve()` 得到后一路径；原 guard 将这个 canonical
路径与未 canonicalize 的 `EXPECTED_PYTHON` 字符串比较，故合法 worker 环境也必然触发
`D2-W interpreter mismatch`。实测两侧分别为
`.../bin/python3.8` 与 `.../bin/python`，比较结果为 false。用户已确认授权以下最小 amendment。

1. **唯一实现修改。** interpreter identity guard 只改为比较
   `Path(sys.executable).resolve()` 与 `Path(EXPECTED_PYTHON).resolve()`；环境变量仍必须逐字等于规范
   `INFBAGEL_PYTHON`，worker hostname、clean worktree、CUDA、checkpoint/data/hash/parity 等 guard 均不变。
2. **回归要求。** 单元测试必须构造 `python -> python3.8` 符号链接，证明 alias/target 双向 canonical
   比较通过、不同文件失败；D2-W 全部测试、完整 authority suite、worker role-applicable suite 与 registry
   validation 必须通过后才能创建 lifecycle artifact。
3. **科学协议不变。** run id、三个 checkpoint 及 hash、internal selection、paired noise、500-step sampling、
   指标、bootstrap gate 和停止规则完全不变；不训练、不选择 checkpoint、不运行 official-438/CHOIS/CM，
   不修改 production model、loss、condition、sampler 或 evaluator。
4. **失败处理。** 修复后若 worker guard、resolved-config、preflight、checkpoint contract、noise parity 或 finite
   contract 任一失败，保留 artifact 并分类 contract failure；不得通过放宽 guard 或更换解释器继续运行。

#### 2026-07-23 Phase 1B D2-W date-rollover operational retry amendment

原 lifecycle `p1-hoi-d2w-checkpoint-frontier-s42-20260722` 的 manifest 于
`2026-07-22T23:59:37+08:00` 创建，但等待受控 SSH/Git 操作后，worker-owned workload 实际于
`2026-07-23T00:10:32+08:00` 才启动，`metrics.json` 于 `00:11:18+08:00` 写成。由于实际 GPU workload
日期已经跨日，原 `20260722` run id 违反真实日期绑定，即使进程 return code 为 0、推理合约通过，也不得作为
reportable D2-W 科学结果。该 lifecycle 已以 `aborted` finish/register 到 run-local registry，全部文件保留、
run id 永不复用；其已观察数值不得用于改动阈值、checkpoint、selection、指标、gate 或任何后续配置。

授权一次且仅一次严格 operational retry：
`p1-hoi-d2w-checkpoint-frontier-r1-s42-20260723`，subphase `1B-D2-W0-r1`。r1 完整继承
2026-07-22 D2-W 预注册的三个 checkpoint/file/model-state hash、32-sequence/96-window internal selection、
逐 step paired noise、generated-history/current-BPS、500-step online diffusion、全部 metric、10,000 次 paired
bootstrap、唯一 gate、分类和停止规则。唯一允许差异是 run id、subphase、真实日期及由新 commit/lifecycle
自然产生的 manifest/preflight/runtime hash；不得因已观察 aborted output 改变任何科学变量。

r1 必须使用新结果目录和新 resolved config/preflight/manifest，重新绑定全部 input hash；原 failed/passed
preflight、manifest、log、metrics、exit code 与 aborted registry 均不覆盖。r1 无论 gate 正负都 finish/register、
回收并停止，不选择 checkpoint、不训练、不运行 official-438/CHOIS/CM，也不开始 FK-foot loss intervention。

D2-W r1 已在 worker commit `e8f4599d1f4d67b1b90f319c850ed2230280b1ad` 完成，return code 0，
runtime `32.128 s`。selection、三个 checkpoint file/model-state hash、逐 step noise、official foot-sliding
torch/numpy parity 与 finite contract 全部通过；r1 与跨日期 aborted lifecycle 的 selection、checkpoint
contracts/metadata、完整 per-sequence results、comparison、decision 和 contract 逐字段完全一致。两个 immutable
目录均已由 worker 主动回收到 authority staging，worker/authority 标准 tree hash 一致：aborted 为
`224c9cb6837ef08ebaa46c2c3c140bb4cbf3abd30c140c2400c4dfea6a98f36b`（8 files / 281,739
bytes），completed r1 为
`bdbd4d66c7304e5d8f80624bd66b99b6088fbda68b898ea12fa95c3b92cfad4d`（7 files / 277,106
bytes）。compact aggregate 为
`experiments/results/p1_hoi_phase1b_d2w_checkpoint_frontier_r1_s42_20260723.json`。

| checkpoint | FK foot sliding | direct foot sliding | FK MPJPE cm | pelvis goal cm | object goal cm | object trans cm | direct/FK foot disagreement cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6.144M control | 1.0184 | 1.4574 | 24.5116 | 8.9756 | 63.2352 | 45.4860 | 11.2901 |
| 24.576M midpoint | 1.3803 | 1.3443 | 15.5517 | 4.5868 | 56.6646 | 20.6455 | 5.2632 |
| 61.44M final | 0.9332 | 1.0083 | 13.9775 | 3.9919 | 58.4418 | 20.3566 | 3.2849 |

核心 gate 的 `final - midpoint` FK foot-sliding paired 95% CI 为
`[-0.7963,-0.1099]`，方向与“后半程退化”假设相反；midpoint/final 的 FK-MPJPE ratio CI 为
`[1.0286,1.2009]`，pelvis-goal 与 object-translation ratio CI 上界也分别为 `1.4025/1.2015`，且
midpoint 相对 control 的 object-goal improvement CI `[-4.0275,17.6240]` 未严格大于 0。因此分类固定为
`midbudget-protection-negative-stop`，不选择 midpoint。该 32-sequence internal holdout 上三个 checkpoint
physical-contact precision/recall/F1 均为 0，contact-preservation 项只能视为无信息，不能外推到 official-438。

科学上，D2-W 排除了“24.576M 后继续 constant LR 是 D2-V official foot-sliding 缺口主要来源”这一首选
解释；后半程反而降低 direct 与 rotation-to-FK foot sliding 并缩小 direct/FK trajectory disagreement。
结合 production velocity loss 不读取 rotation channels、official metric 完全依赖 rotation-to-FK 的确定
目标错位，下一候选应是单一 evaluator-aligned FK-foot temporal training loss，而不是 mid-budget checkpoint
选择或 schedule-only retry。这仍是基于证据的机制优先级，不是该 loss 必然有效的证明；penetration 必须保持
独立，不得捆绑。D2-W 至此停止，未训练、未启动 CM、未选择或蒸馏任何 checkpoint。

#### 2026-07-23 Phase 1B D2-X FK-foot temporal gradient-routing screen 预注册

D2-V 已证明当前 232-D diffusion HOIPrior 在 61,440,000 processed windows 后能够从随机初始化学到接近
released baseline 的 MPJPE、object-goal、object-translation 与 contact；其唯一 absolute capability gate
失败项是 foot sliding，且 penetration 仍较弱。D2-W 进一步排除了中途 checkpoint 或后半程 constant-LR
过冲作为主要解释，并确认现有 velocity objective 只监督 direct normalized joint/object-translation
channels，而 official foot sliding 完全由 predicted root/rotations 经 FK 后得到。D2-X 因此只检验一个
loss-gradient routing 假设：把既有 velocity residual 中四个足部关节的水平分量对齐到 evaluator 使用的
rotation-to-FK trajectory，是否能降低脚滑且不破坏 D2-V 已形成的能力。

1. **唯一 manipulated factor。** Training run id 固定为
   p1-hoi-d2x-fk-foot-temporal-routing-s42-20260723、subphase 1B-D2-X0、seed 42。相对 D2-V，
   velocity loss 的 tensor shape、元素数、mean-square reduction 与全局 weight 0.1 均不变；只替换
   joints [7,8,10,11] 的 x/z 共 8 个 predicted temporal residual slots。Prediction 侧先用 predicted
   root、22 个 6D rotations、既有 24-joint parents/rest offsets 得到 FK positions，再用既有
   position min/max 映射到 direct position channels 相同的 normalized scale。Target 侧继续使用
   normalized clean direct foot positions。第一个 future frame 的 previous prediction 必须是 immutable
   GT history 最后一帧；其后 previous prediction 使用前一帧 predicted normalized FK position。四个
   足部 y、其余 joint xyz、object translation 以及所有 target residual 保持 D2-V 原实现。不得增加
   新 loss term、可调 weight、height/contact gate 或 sampler correction。
2. **固定训练 contract。** 必须重新随机初始化，保留 D2-V 的 232-D state、16 frames、2 history
   frames、512-wide/16-head/8-layer Transformer、500-step clean-x0 diffusion、OMOMO split、全部
   conditions、FK/object-surface/velocity/goal weights
   0.3569973401779424/0.4772322188400037/0.1/1.0。只在 infbagel-4gpu/node01 运行：
   4×RTX 3090、batch/GPU 512、accumulation 1、effective batch 2048、Adam (0.9,0.999)、
   constant LR 1e-4、FP32、无 warmup/scheduler/weight decay/clipping/AMP/EMA，固定
   61,440,000 processed windows、30,000 updates、32,768 validation windows 以及每
   3,072,000 windows validation/checkpoint cadence。禁止加载 released、author、source、current、
   balanced、D2-T/U/V、prior、resume 或任何 EMA checkpoint；不得设置 d2m_candidate。
3. **实现和 lifecycle。** 在任何 production 代码改动前完成本 plan 与 registry 预注册。新增独立
   D2-X Hydra config、fail-closed mode/host/provenance contract 和 tests；D2-T/U/V 的 exact config、
   loss 与 checkpoint-resume contract 必须保持不变。测试至少证明：routing 关闭时与原 velocity
   formula bitwise/数值等价；开启时足部 x/z formula 正确；第一 future frame 只读取 immutable
   history；非足部、足部 y 与 object residual 不变；rotation/root 获得 velocity gradient 而 direct
   foot x/z 不再获得该分量梯度；随机初始化与 forbidden-checkpoint guard 生效。代码、config、
   tests、plan、registry 和 target-only evaluator 必须组成一个 logical commit。authority 全量验证后，
   worker 只能主动 fast-forward 到相同 commit。训练前归档 fully-resolved config、同一 escalated
   context 的四卡 preflight，并用 tools/experiment.py start 创建不可覆盖的 reportable manifest。
4. **固定 official evaluation。** Evaluation run id 固定为
   p1-hoi-d2x-native-eval-s42-20260723、subphase 1B-D2-X0-eval。只评估 fixed final online
   checkpoint 一次，使用 author-native HOI official-438、每 sequence 三窗口、500-step unguided
   diffusion；无 CFG、dynamic perception、guidance、CHOIS selection、FID 或 R-precision。D2-V
   final official per-sequence records 与 aggregate 作为 immutable paired control，只读复用且不重新
   生成。统计单位为 sequence，10,000 次 paired bootstrap，seed 42。
5. **mechanism/effective gates。** Mechanism gate 要求同时满足：D2-V minus D2-X foot-sliding
   difference 的 paired 95% CI 下界 > 0；D2-X/D2-V 的 MPJPE、end-object、xy 与
   object-translation ratio CI 上界均 <= 1.10；D2-X minus D2-V contact-F1 difference CI 下界
   >= -0.02；D2-X/D2-V 的 hand-penetration-loss 与 human-penetration-loss ratio CI 上界均
   <= 1.10；以及 contract、finite loss/gradient、normalization、history 与 checkpoint provenance
   全通过。Absolute effective-diffusion gate 继续使用 Phase-0 released ratios：MPJPE <= 1.30、
   end-object <= 2.00、xy <= 1.50、object-translation <= 1.50、foot sliding <= 1.10，且
   contact F1 >= 0.60。Mechanism 与 absolute gate 均通过时分类
   fk-foot-temporal-routing-positive-candidate-stop；仅 mechanism 通过时分类
   fk-foot-temporal-routing-positive-but-not-effective-stop；任一 mechanism 条件失败分类
   fk-foot-temporal-routing-negative-stop；contract/hash/lifecycle 失败分类
   fk-foot-temporal-routing-contract-failure-stop。
6. **停止规则和 artifacts。** 无论结果如何，完成 fixed-budget training 与一次 official evaluation
   后停止；不得观察中间 checkpoint 后选择 favorable state、延长预算、同时修 penetration/contact、
   修改 architecture/condition、启动 sampler heuristic 或 consistency distillation。即使两个 gate 均
   通过，本 subphase 也只产生 foot-protected diffusion candidate，checkpoint 选择及后续 penetration
   intervention 必须另行确认；CM 仍不授权。必须保留 resolved config、machine preflight、manifest、
   完整 logs/metrics、全部 cadence checkpoint hashes、resume evidence、loss/gradient routing audit、
   official aggregate/per-sequence/bootstrap/gate JSON、artifact-tree hash 和任何 negative result。

#### 2026-07-23 Phase 1B D2-X penetration finite-mask measurement amendment

在 D2-X implementation 尚未提交、未创建 run directory、未启动任何 GPU workload 时，对 immutable
D2-V official-438 control 的 per-sequence artifact 做 fail-closed schema check，发现
hand_pen_loss_omomo 与 human_pen_loss_infbagel 仅在相同的固定 181 条 sequence 上为 finite，其余
257 条均为 null；D2-V aggregate 的两个 penetration mean 也正是按这 181 条计算。该缺失由既有 author
evaluator/object asset coverage 决定，不是 D2-X 结果或 favorable selection。固定 181-sequence ID
列表按字典序、每行一个 ID 且末尾换行的 SHA-256 为
2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec。

D2-X measurement contract 因而在训练前明确为：MPJPE、end-object、xy、object-translation、
foot-sliding 与 contact-F1 继续使用全部 438 个 paired sequences；两个 penetration ratio 只使用上述
预先由 control 封存的 181 条 finite mask，并且 target 两个字段的 finite/null mask 必须分别与 control
完全相同、两个 target penetration masks 也必须相同。任何 mask identity/count/hash 不一致均分类为
contract failure，不允许按 target 数值重新筛选或用 aggregate-only ratio 绕过。bootstrap replicates、
seed、ratio 上界 1.10、其余 gate、训练唯一变量、run id、预算和停止规则全部不变。

#### 2026-07-23 Phase 1B D2-X pre-optimizer dispatch failure and r1 operational amendment

首次 reportable lifecycle p1-hoi-d2x-fk-foot-temporal-routing-s42-20260723 在 clean worker commit
48e2e6c31d281af8809d35b5c8ce2ac8123205d1、resolved config 与四卡 preflight 均通过后启动，但在
任何 optimizer update、processed window 或 checkpoint 之前 fail-closed。确定原因是
code/train_hoi_prior.py 的 pre-optimizer locked loss-weight dispatch 仍只将 D2-U/D2-V 识别为
balanced weights mode，遗漏已由独立 exact contract 验证的 D2-X；因此合法的
FK/object-surface weights 0.3569973401779424/0.4772322188400037 被旧 generic guard 误判。
该失败不是 loss 数值、显存、数据、梯度或 checkpoint lineage 结果，不能用于判断 D2-X 科学假设。

失败 lifecycle 已永久保留并标记 failed：exit code 1、optimizer updates 0、processed windows 0、
checkpoint 0，未加载 released/author/prior checkpoint、未启动 consistency。manifest/metrics/
resolved/preflight/log/run-local-registry SHA-256 分别为
936984115a66c7142b5254c4cc5d33874e855be46972ceec5e75f69483be8c7e、
e52a6eebb4d3fc40e9c204e5381520ba5b65c42ec37b3d3aae945fa4ef501ff7、
5746293fc4f5455a3619b4aec46e5b899aa51f2751416d29577bd54f83b4f226、
c8e1fee0ca60fcd92d482a838601396ec7f2ee4e0ac9d3c6bd62ea2390cd9ae9、
71fa61f14a230fc05bc1dec3f234c1e258149b6a88602762ad6584baf9a01dd4 和
5e7486c4991990654a448ba10314a11db2ab157d68d0c834c48280ec456279e5；完整 tree 为
11 files / 33,752 bytes，SHA-256
4088e04b1b92d412d25aa1842cb6cd2d6d4191a48c79388fdc3c8229bf16ab95。

用户已授权继续 D2-X，因此只允许一次 operational r1：

1. 唯一实现修复是让 pre-optimizer balanced loss-weight dispatch 明确包含 D2-X，并增加直接回归测试，
   证明 D2-U/D2-V/D2-X 返回相同 locked balanced weights，而普通/D2-T mode 仍返回 50/50。不得改动
   loss formula、routing、weights、data、architecture、condition、optimization、budget 或 evaluator gate。
2. 新 training run id 固定为
   p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723、subphase 1B-D2-X0-r1；新 evaluation
   run id 固定为 p1-hoi-d2x-native-eval-r1-s42-20260723、subphase 1B-D2-X0-eval-r1。旧 training
   id 永不复用，旧未启动 evaluation id 不得绑定到 r1 checkpoint。
3. r1 完整继承原 D2-X 和 penetration-mask amendment 的唯一变量、随机初始化、61,440,000-window/
   30,000-update contract、4×3090、official-438/181 finite-mask statistics、所有 gates、classifications、
   artifact contract 与停止规则。必须形成新完整 commit、worker 主动 fast-forward、重新生成
   resolved config/preflight/manifest；禁止从失败 lifecycle resume 或加载任何 checkpoint。
4. 若同一 dispatch blocker 再现或出现新的 pre-optimizer contract failure，r1 必须保留并停止，不得继续
   自由重试。若 r1 进入稳定训练，则按原 D2-X 协议完成；仍不授权 checkpoint selection、penetration
   intervention 或 CM。

#### 2026-07-23 D2-X r1 completion and negative gate

D2-X r1 在 `3af3facaf73d3bbffcca6d6181bac1ff89909a24`、`infbagel-4gpu/node01` 上从随机初始化完成了
固定的 61,440,000 processed windows / 30,000 optimizer updates。effective batch 为 2,048（4×3090、
每卡 512、accumulation 1），loss/gradient 全部 finite，20 个 cadence checkpoint 和 80 个 RNG sidecar
均保留；未加载 released、author、source、current、balanced、prior 或任何 resume/EMA checkpoint。
最终 online checkpoint SHA-256 为
`b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51`，training manifest/metrics/
resolved/preflight/run-local-registry SHA-256 分别为
`2011ded7310f851d3a1278bd65fe2d19fdca8f7b859d289d7e240b3d8d347d85`、
`0c99ac8b5880b1e7419cc6fe9c4be6388e7f806f066e9863da0ac320336693f3`、
`5e913957d0dde5bd3d589e98248436a14353f831ccfd422a5a96d152f7273130`、
`1ef9206e8521ca73ef78c0dec7bddc1ef1b192e138f3765d1f06ff4ef7187cac` 和
`9dde11270e443f39e86ab77b26f0a80fd097cba2f42b7591d314f87373a1e856`；完整 training tree 为
112 files / 7,127,226,145 bytes，SHA-256 `3f95773270e4701310daac9128c19d822d0a4e887ad7c5ddd40008f1b98a47c6`。

首次 evaluation preflight 只因错误地把 `--chois-root` 指向 checkout 内的资产目录而失败
（未创建 evaluation manifest、未加载 checkpoint、未启动 GPU workload）；该失败文件保留。随后使用
固定的 `/home/yujinlun/data/evaluators/chois_release` 重新捕获 preflight，所有 pinned checks、四卡
idle/无 CUDA compute process 和 Xorg 显示层约束通过。唯一一次 official-438 evaluation 使用 final
online checkpoint、每序列 3 windows、500-step unguided diffusion，无 CFG、dynamic perception、guidance、
CHOIS、FID 或 R-precision。D2-V final records 作为只读 paired control，181-sequence penetration mask
保持完全一致。

target aggregate 为 MPJPE `12.0508`、end-object `3.7402`、xy `4.0505`、object translation `15.9940`、
foot sliding `0.3630`、contact F1 `0.6374`、hand/human penetration loss `0.2454/3.8691`。相对 D2-V，
foot-sliding mean 改善 `0.0152733`，但 10,000-replicate paired bootstrap 95% CI 为
`[-0.0037904, 0.0341575]`，没有满足预注册的 CI 下界 `>0`；其余 kinematic/object/contact/penetration
protection checks 均通过，且相对 released baseline 的所有 absolute checks 也通过。因此分类严格为
`fk-foot-temporal-routing-negative-stop`。这是一个保留 D2-V 能力但未证实 foot-sliding 因果改善的负结果，
不是 evaluator 或 artifact contract 失败；不得从该 checkpoint 选择、延长预算、修改 sampler/contact/
penetration，或启动 consistency distillation。

evaluation manifest/metrics/aggregate/per-sequence/resolved/preflight/run-local-registry SHA-256 分别为
`fa19565eb96155f735a7b8c1569a95e1069267b8641ffe7968e878554fee4550`、
`f2cb76d0c248c4d4b8ce4571758c1937e2e19170199c6e9631df21858dc1c807`、
`3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b`、
`69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`、
`206eb63d15231d5986adf001295810b1a83cf88511f7435379d43f35a8c03617`、
`3262548e948038f3725978e645c4f2f14893253bddfa5349ec80e018b18cf76e` 和
`6739dd94d7181c76020f33d3cdbf24aff5f3fe24495c411363808722fc3ba1a`；evaluation tree 为 16 files /
366,190 bytes，SHA-256 `c4a853d99659ac92ac830621a0e8caf68aea3db9f3d954b3486d3aa4d3d3eb74`，worker/authority
完全一致。compact aggregate 为
`experiments/results/p1_hoi_phase1b_d2x_fk_foot_temporal_routing_r1_s42_20260723.json`，SHA-256
`fdf21f8b0042d1d26ac2a3b4cf8a073a43cd20283f0292d55572aa66de6e42f6`。D2-X 至此停止；
任何新的 HOIPrior 机制必须另行 dated plan/registry amendment，且 consistency stage 仍未授权。

#### 2026-07-23 Phase 1B D2-Y routed-foot residual amplification screen 预注册

D2-X 已证明 evaluator-aligned FK-foot temporal routing 没有显著破坏 D2-V：official-438 上
MPJPE、object、contact 与 penetration protection 均通过，foot sliding 点估计从 `0.3783` 降至
`0.3630`；但 paired mean improvement `0.0152733` 的 10,000-replicate sequence bootstrap 95% CI
`[-0.0037904,0.0341575]` 包含 0，因此不能选择 checkpoint。封存代码与只读 audit 同时确认，
velocity tensor 每 future frame 有 87 个 residual slots，routed foot x/z 仅 8 个，占
`8/87 = 9.1954%`；D2-X final validation 的 unweighted velocity MSE 为 `0.0001015743360`，
乘全局 weight `0.1` 后只占 total validation loss 的 `0.0208%`。固定 seed-42 probe 上 late routed
scalar 只占 weighted velocity scalar 的 `5.42%--5.79%`，routed gradient/FK-gradient norm ratio
仅 `0.00640%--0.01049%`，且 routed 与 FK gradient cosine 为 `-0.7435` 至 `-0.8086`。这支持
“routing 正确但 global mean/weight 使信号过弱”为首选解释，并把“rotation/root 上与 FK 的竞争抵消”
保留为次要竞争解释；official metric 的 30 Hz interpolation、predicted floor 与 near-ground nonlinear
gate 仍构成尚未消除的语义差异。上述 audit 没有发现新的确定 implementation defect；D2-X r0 的
0-update dispatch defect 继续作为独立 lifecycle failure 永久保留。

D2-Y 只检验一个可证伪的 loss-geometry 机制：保持 D2-X 的 8 个 FK-routed residual slots 和所有其他
训练条件不变，仅对这 8 个 squared residual errors 使用固定乘数 `1024`，仍在原 87×14 tensor 上做
一次 global mean，且全局 velocity weight 仍为 `0.1`。`1024` 是预先由封存 late probe 选择的最小
2 的幂，使三个 preregistered timestep strata 的 routed/FK gradient-norm ratio 全部至少为 5%
（线性外推为 `8.00%/6.55%/10.74%`）；不得根据 smoke、中间 validation 或训练结果调整或 sweep。

1. **唯一 manipulated factor。** Training run id 固定为
   `p1-hoi-d2y-routed-foot-amplification-s42-20260723`、subphase `1B-D2-Y0`、seed 42。对
   D2-X routed slots（joints `[7,8,10,11]` 的 x/z）令
   `L_velocity = mean(w_j * (predicted_residual_j-target_residual_j)^2)`，其中 routed slots
   `w_j=1024`，其余 79 slots `w_j=1`。Residual construction、target、first-future immutable GT
   history、later predicted-FK history、tensor shape/element count、全局 velocity weight、FK/
   reconstruction/object-surface/goal formula 全部保持 D2-X 不变；不得增加 height/contact gate、
   penetration term、sampler correction或第二个 loss knob。
2. **固定训练 contract。** 必须重新随机初始化并保留 232-D representation、16 frames、2 history
   frames、512-wide/16-head/8-layer Transformer、500-step clean-x0 diffusion、OMOMO seed-42 split、
   全部 conditions，以及 FK/object-surface/velocity/goal weights
   `0.3569973401779424/0.4772322188400037/0.1/1.0`。只允许在
   `infbagel-4gpu/node01` 用 4×RTX 3090、batch/GPU 512、accumulation 1、effective batch 2048、
   Adam `(0.9,0.999)`、constant LR `1e-4`、FP32、无 warmup/scheduler/weight decay/clipping/AMP/EMA，
   固定 61,440,000 processed windows / 30,000 updates、32,768 validation windows 和每
   3,072,000 windows validation/checkpoint cadence。禁止加载 released、author、D2-V、D2-X、
   prior、resume、EMA 或任何其他 checkpoint；不得设置 `d2m_candidate`。若 reportable lifecycle
   在日期变化后才启动，必须先按启动机器真实 `date` 追加只改 run/subphase 标识的 operational
   amendment，旧 id 不得创建或复用。
3. **固定 internal mechanism diagnostic。** 复用 D2-W 封存的 32-sequence/96-window selection
   （selection SHA-256
   `30524c88481f6cb81e8063073d510ad01543be92d91eb4ef9b2b8a376cc4fbae`），用相同 clean batch、
   timestep、noise、condition dropout 与 D2-X early/mid/final online checkpoints作 paired control。
   D2-Y 必须报告 early/mid/final 和 timestep `0/249/499` 的 routed normalized residual RMS，以及
   root、rotation、input projection、output projection、Transformer parameter-gradient norm；
   同时报告 routed gradient 与 reconstruction、FK、object-surface、goal gradient cosine。Internal
   mechanism gate 固定为 final D2-X minus D2-Y routed-foot normalized residual MSE 在 timestep
   `249` 和 `499` 的 sequence-paired 10,000-replicate bootstrap 95% CI 下界均 `>0`。Internal gate
   只诊断放大是否被优化器吸收，不用于 checkpoint selection。
4. **固定 official evaluation。** Evaluation run id 固定为
   `p1-hoi-d2y-native-eval-s42-20260723`、subphase `1B-D2-Y0-eval`。只评估 fixed final online
   checkpoint 一次，使用 D2-X 完全相同的 author-native official-438、每 sequence 三窗口、
   500-step unguided diffusion；无 CFG、dynamic perception、guidance、CHOIS selection、FID 或
   R-precision。D2-X r1 final aggregate/per-sequence records 和 checkpoint SHA-256
   `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
   `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a` /
   `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51`
   作为 immutable paired control，只读复用。统计单位为 sequence，bootstrap 10,000 次、seed 42；
   penetration 继续绑定 D2-X 已封存的相同 181-sequence finite mask 和
   `2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec` ID hash。
5. **success、negative 与 effective gates。** Official mechanism gate 要求：D2-X minus D2-Y
   foot-sliding difference CI 下界 `>0`；D2-Y/D2-X 的 MPJPE、end-object、xy、
   object-translation、hand-penetration-loss 与 human-penetration-loss ratio CI 上界均
   `<=1.10`；D2-Y minus D2-X contact-F1 difference CI 下界 `>=-0.02`；所有 contract、finite、
   normalization、history、mask 与 provenance checks 通过。Absolute gate 继续相对 Phase-0
   released baseline 使用 MPJPE/end-object/xy/object-translation/foot-sliding ratio
   `<=1.30/2.00/1.50/1.50/1.10` 且 contact F1 `>=0.60`。
   internal 与 official mechanism gates、absolute gate 全通过时分类
   `routed-foot-amplification-positive-candidate-stop`；internal 通过而 official foot gate 失败为
   `routed-foot-amplification-transfer-negative-stop`，区分“信号确实变强但训练 surrogate 未转移到
   near-ground official metric”；official foot 改善但任一 protection gate 失败为
   `routed-foot-amplification-conflict-negative-stop`，支持 gradient conflict/capability tradeoff；
   internal gate 失败为 `routed-foot-amplification-optimization-negative-stop`；mechanism 通过但
   absolute gate 失败为 `routed-foot-amplification-positive-but-not-effective-stop`；artifact/
   lifecycle contract 失败为 `routed-foot-amplification-contract-failure-stop`。
6. **实现、停止规则与 artifact contract。** Plan/registry 后才允许新增 fail-closed D2-Y config、
   weighted reduction、contract、diagnostic、target-only evaluator 与 tests；D2-T/U/V/X 的 exact
   config、loss 和 checkpoint-resume contract 必须保持不变。authority clean validation 后形成一个
   logical commit，worker 只能主动 fast-forward 到相同 Git object；训练前必须归档 fully-resolved
   config、同一 execution context 的 machine preflight，并用 `tools/experiment.py start` 创建
   reportable manifest。必须保留 manifest/resolved/preflight、完整 logs/metrics、20 个 cadence
   checkpoints 与 80 个 RNG sidecars及 hashes、resume evidence、internal per-sequence/gradient/
   bootstrap audit、official aggregate/per-sequence/bootstrap/gate JSON、normalization/mask/provenance
   audits、artifact-tree hash和任何 failure。无论分类如何，完成 fixed-budget training、internal
   diagnostic 和一次 official evaluation 后停止；不得选择中间或最终 checkpoint、延长预算、从
   D2-V/D2-X resume、post-hoc 改 multiplier/gate、启动 penetration intervention、HSIPrior、Mixer 或
   consistency distillation。即使分类 positive，本 subphase 也只产生待用户另行确认的 diffusion
   candidate，绝不自动授权 checkpoint selection 或 CM。

#### 2026-07-24 Phase 1B D2-Y post-training lifecycle date amendment

D2-Y training lifecycle 已于真实日期 2026-07-23 启动，使用已预注册的
`p1-hoi-d2y-routed-foot-amplification-s42-20260723`，并跨午夜完成固定的 61,440,000-window /
30,000-update contract。训练 exit code 为 0；final online checkpoint SHA-256 为
`8734431f89cf8739283828d5fb683212ca43143ae3482ad0473f6ed5717eb7a7`。在 2026-07-24 检查完成时，
原计划中的 internal/evaluation lifecycle 均尚未创建 manifest、加载 checkpoint 或启动 GPU，而其
预注册 run id 仍带 `20260723`。为满足所有新 lifecycle 使用真实启动日期且不得复用旧 id 的规则，
只允许以下 identity-only amendment：

1. internal diagnostic run id 改为
   `p1-hoi-d2y-routed-foot-amplification-internal-s42-20260724`；subphase 仍为
   `1B-D2-Y0-internal`；
2. official evaluation run id 改为 `p1-hoi-d2y-native-eval-s42-20260724`；subphase 仍为
   `1B-D2-Y0-eval`；
3. 旧的 `p1-hoi-d2y-native-eval-s42-20260723` 预注册标识在任何 workload 前被 supersede，
   永不创建、复用或绑定 checkpoint；这不是 evaluator/科学 failure，也不产生 retry entitlement。

除上述两个后训练 lifecycle 标识外，D2-X/D2-Y early/mid/final checkpoint、32-sequence/96-window
selection、timestep/noise/dropout pairing、1024 multiplier、bootstrap、official-438/181 mask、所有
success/negative/protection/absolute gates、artifact contract 与停止规则完全不变。不得由此修改训练、
重新生成 D2-X control、选择 checkpoint、启动 CM、HSIPrior 或 Mixer。

#### 2026-07-24 Phase 1B D2-Y completion：surrogate 吸收但 official transfer 失败

D2-Y 已完成唯一预注册的 from-random training、internal mechanism diagnostic 和 official-438
evaluation。训练严格使用 4×RTX 3090、effective batch 2,048、61,440,000 processed windows /
30,000 updates、seed 42 和 final online weights；20 个 cadence checkpoints、80 个 rank RNG
sidecars、finite loss、required gradients、零 overflow、random-init/no-restored-state contract 均通过。
Final checkpoint SHA-256 为
`8734431f89cf8739283828d5fb683212ca43143ae3482ad0473f6ed5717eb7a7`。该 checkpoint 仅是
negative-result artifact，不得选择、resume 或初始化后续 prior/CM。

固定 32-sequence/96-window internal diagnostic 上，final D2-X minus D2-Y routed residual MSE 在
timestep 249/499 的 paired mean 为 `2.85917e-05 / 3.13497e-05`，10,000-replicate bootstrap
95% CI 分别为 `[2.12425e-05,3.62503e-05]` 和
`[2.39972e-05,3.95998e-05]`，两个下界均大于 0，因此 internal gate 通过：1024× amplification
确实被优化器吸收并降低了 teacher-forced surrogate。但 final D2-Y routed/FK gradient cosine 在
timestep 249/499 为 `-0.5001/-0.4522`，说明 late noisy strata 的目标竞争仍然存在。

唯一 official evaluation 使用真实日期 run
`p1-hoi-d2y-native-eval-s42-20260724`、438 sequences、每序列 3 windows、500-step unguided
diffusion，并只读复用 sealed D2-X records。D2-Y 的 MPJPE/end-object/xy/object translation/
foot sliding/contact F1/hand penetration/human penetration 为
`12.1246/4.8506/3.9676/16.3290/0.3572/0.6351/0.2184/3.4353`。D2-X minus D2-Y
foot-sliding paired mean 为 `0.00585353`，95% CI
`[-0.0162119,0.0279164]` 包含 0，official foot gate 失败；D2-Y minus D2-X contact-F1 CI
`[-0.0231844,0.0184971]` 的下界低于 `-0.02`，且 end-object ratio CI
`[1.22629,1.36180]` 的上界超过 1.10，因此 protection gate 也失败。MPJPE、xy、
object-translation、两项 penetration protection、固定 181-sequence mask，以及相对 released
baseline 的全部 absolute diffusion checks 均通过。

严格分类为 `routed-foot-amplification-transfer-negative-stop`：internal surrogate 显著改善，
但未 transfer 为显著 official foot-sliding 改善；end-object/contact 的保护失败作为额外负证据完整
保留，不能 post-hoc 改成 positive 或选择 checkpoint。已验证的是 weight/dilution 机制被成功改变且
official gate 失败；基于证据的推断是 global-mean dilution 并非 D2-X 的充分解释，剩余主要候选为
temporal surrogate 与 nonlinear near-ground metric 的语义差异，以及 routed/FK/object capability
冲突；尚未证明任何新的 official-semantic loss 或 conflict-resolution geometry 有效。没有发现确定
implementation defect，point-estimate foot 改善仍与注册的 sequence-sampling uncertainty 相容。

training/internal/evaluation artifact trees 已由 worker 主动回收到 authority staging，worker 与
authority SHA-256 全部一致，分别为
`177eb44fa53ee46518a714f04ee2fe864aa2a1d755f4377f3fd47fa8e40bf0f8`（112 files /
7,128,453,002 bytes）、
`d57e48d5d4285b4f42ab090e2aef0b1ee641217ea960382ec0b60c6bf0e8d05f`（6 files /
1,493,189 bytes）和
`3e81a59fb1a97e7043e856fda5c502bae3683f24a179b0423143fe910837bdc0`（15 files /
1,631,051 bytes）。compact aggregate 为
`experiments/results/p1_hoi_phase1b_d2y_routed_foot_amplification_s42_20260724.json`，SHA-256
`0bc2fffd4304bb3411176cf355dacddfe731e9f1d46eb01cc9bfefe3c215f875`；完整 handoff 为
`docs/phase_summaries/PHASE_1B_D2Y.md`。D2-Y 至此停止；未经新的 dated plan/registry amendment
和用户确认，不得继续 HOIPrior 机制、选择 checkpoint、启动 consistency、进入 HSIPrior 或 Mixer。

#### 2026-07-24 Phase 1B D2-Z immutable-GT near-ground routed amplification 预注册（plan-only）

D2-Y 已证实固定 1024× routed-foot amplification 能被优化器吸收：timestep 249/499 的
D2-X minus D2-Y routed residual MSE paired bootstrap 95% CI 下界均大于 0；但 official
foot-sliding improvement CI `[-0.0162119,0.0279164]` 包含 0，且 end-object/contact protection
失败。因此“8 slots 在 global mean 中过弱”不再是充分解释。D2-X/D2-Y 的训练 residual 对所有
future frames 一视同仁，而 official foot sliding 只在 previous-frame ankle/toe 相对 inferred floor
低于 `0.08/0.04 m` 时计入 horizontal displacement；uniform amplification 可能把共享模型容量花在
official metric 不计分的非近地 foot motion 上，并加剧 late noisy strata 已观察到的 routed/FK 冲突。

只读 post-hoc sequence audit 仅用于形成新假设，不修改 D2-Y gate 或结论：D2-Y 相对 D2-X 的
438-sequence foot difference 为 239 improve / 199 degrade，mean `0.00585353`、median
`0.0136075`、SD `0.237995`；按 D2-X foot sliding 四分位分层，最低/最高四分位的 mean improvement
分别为 `-0.149215/+0.162256`，13 个 object family 中 6 个为正、7 个为负。foot improvement 与
end-object/contact change 的 Pearson correlation 仅 `-0.0556/-0.0533`。这支持优先检验
near-ground support mismatch，而不是继续增大 multiplier；这些 exploratory 数字不是 selection gate。

D2-Z 是当前 registry 中下一个未占用的 Phase 1B 标识。本 amendment 只预注册科学机制、gates、
stop rule 与 artifact contract；不授权实现、GPU smoke、training、evaluation、checkpoint loading/
selection 或 consistency。具体 workload 日期和 run ids 延迟到用户另行授权后：必须先在执行日运行
`date`，再 append identity-only lifecycle binding；不得预占或复用本 amendment 日期作为未来
workload 日期。预留 run stem 为 `p1-hoi-d2z-immutable-gt-near-ground-gating-s42-<YYYYMMDD>`，
subphase 为 `1B-D2-Z0`。

1. **唯一 manipulated factor。** D2-Z 保留 D2-Y 的 D2-X FK temporal routing、相同 8 个 foot x/z
   residual slots、相同 target 和固定 multiplier 数值，只把 per-element coefficient 从“所有 routed
   entries 均为 1024”改为 immutable-GT binary near-ground gating。对 current sampled future frame
   `t` 的 residual `position[t]-position[t-1]`、joint `j` 和 x/z component `c`：
   `w(t,j,c)=1024` iff
   `y_GT[t-1,j]-floor_GT(sequence)<H_j`，否则 `w(t,j,c)=1`；`H_7=H_8=0.08 m`，
   `H_10=H_11=0.04 m`。First future residual 的 previous frame 仍是 immutable GT history 最后一帧。
   Gate 必须 stop-gradient，不能读取 prediction、diffusion timestep、contact prediction、object、
   text 或 loss；同一 joint 的 x/z 共用 gate。不得加入 official exponential height factor、soft gate、
   zero-velocity target、learned/adaptive threshold、weight normalization 或 multiplier sweep。
2. **GT floor 与 gate provenance。** `floor_GT(sequence)` 必须只由当前 immutable OMOMO snapshot 中
   该 sequence 的完整 30 Hz `human_joints_aligned.npy[start_idx:end_idx]` 计算，精确复用
   `code/eval_metrics.py::determine_floor_height_and_contacts`；该文件当前 SHA-256 为
   `445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547`。Gate height 使用对应
   10 Hz training-window sampled previous frame的 raw aligned GT y；window-local x/z origin/yaw
   不得进入 height/floor。不得使用 generated/predicted floor、per-window minimum、official test
   motion 或任何新增外部资产。232-D tensor、conditions 和 model input/output 不增加 field；
   derived floor/gate 只能作为 loss metadata。
3. **固定 gate preflight contract。** 实现后、任何 GPU 或 reportable manifest 前，必须在 authority
   CPU 上对 train/internal-validation split 全量生成 deterministic gate audit，记录 algorithm/schema、
   ordered sequence/window IDs、source/split/function hashes、per-joint active counts、floor finite/range、
   overall/per-window occupancy、nonfinite count 和 aggregate SHA-256。封存的
   32-sequence/96-window selection（SHA-256
   `30524c88481f6cb81e8063073d510ad01543be92d91eb4ef9b2b8a376cc4fbae`）必须精确复现：
   joints 7/8/10/11 active counts `1096/1081/1211/1232`，各自 denominator `1344`，overall
   active fraction `4620/5376 = 0.859375`，32 个 GT floor 全 finite 且 min/median/max
   `0.0260237856/0.0450961320/0.0530758165 m`。Mismatch、零 active entries、source hash change
   或 nonfinite floor/gate 均为 contract failure；不得通过改 threshold/floor algorithm 修复后沿用同一
   lifecycle。
4. **固定训练 contract。** 若后续另行授权，D2-Z 必须重新随机初始化，initial model-state SHA-256
   仍应为 `ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e`；保留
   232-D representation、16 frames、2 history frames、512-wide/16-head/8-layer Transformer、
   500-step clean-x0 diffusion、OMOMO seed-42 split（SHA-256
   `019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e`）、conditions 和
   FK/object-surface/velocity/goal weights
   `0.3569973401779424/0.4772322188400037/0.1/1.0`。只允许
   `infbagel-4gpu/node01`、4×RTX 3090、batch/GPU 512、accumulation 1、effective batch 2048、
   Adam `(0.9,0.999)`、constant LR `1e-4`、FP32、无 warmup/scheduler/weight decay/clipping/
   AMP/EMA，固定 61,440,000 processed windows / 30,000 updates、32,768 validation windows 和
   3,072,000-window validation/checkpoint cadence。禁止加载 released、author、D2-V、D2-X、D2-Y、
   prior、resume、EMA 或任何 checkpoint。
5. **固定 internal diagnostic。** 完整训练后复用上述 sealed 32-sequence/96-window selection，
   对 D2-X/D2-Y/D2-Z early/mid/final online checkpoints 和 timestep `0/249/499` 使用相同 clean
   windows、noise 与 condition dropout。必须分别报告 near-ground active 与 inactive routed
   residual RMS/per-sequence MSE、gate occupancy，以及 gated routed contribution 对 root、rotation、
   input/output projection、Transformer 的 gradient norm；报告其与 reconstruction、FK、
   object-surface、goal 的 gradient cosine，并报告相对同一 D2-Z prediction 上 uniform-D2-Y routed
   contribution 的 gradient-norm retained fraction/cosine。Internal checks 只验证 gate provenance、
   finite/nonzero gradient 和竞争机制，不作为 checkpoint selection 或调 multiplier 的依据。
6. **固定 official evaluation 与 controls。** 只允许一次 final-online author-native official-438、
   每 sequence 3 windows、500-step unguided diffusion；无 CFG、dynamic perception、guidance、
   CHOIS selection、FID 或 R-precision。Primary candidate control 为 immutable D2-X r1 records：
   checkpoint/aggregate/per-sequence SHA-256 分别为
   `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51` /
   `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
   `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`。D2-Y
   aggregate/per-sequence SHA-256
   `776e6c35acdaa190ffcbab047b170ed4ab559c23f454714c31ad980db4dd8c70` /
   `ea2cde99372392c5f16446708e3acf3789a68be9f1b7cc95134fd45390b12c02`
   仅作为 uniform-amplification mechanism comparator，不重新生成且不取代 D2-X candidate gate。
   统计单位仍为 sequence、bootstrap 10,000 次、seed 42；penetration 绑定相同 181-sequence mask 和
   `2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec` ID hash。
7. **Success、negative 与 absolute gates。** Candidate mechanism gate 相对 D2-X 要求：
   D2-X minus D2-Z foot-sliding difference CI 下界 `>0`；D2-Z/D2-X 的 MPJPE、end-object、xy、
   object-translation、hand/human penetration ratio CI 上界均 `<=1.10`；D2-Z minus D2-X
   contact-F1 difference CI 下界 `>=-0.02`；所有 gate/lifecycle/provenance/finite/normalization/
   history/mask contracts 通过。Absolute gate 保持相对 Phase-0 released baseline 的
   MPJPE/end-object/xy/object-translation/foot-sliding ratio
   `<=1.30/2.00/1.50/1.50/1.10` 且 contact F1 `>=0.60`。D2-Y comparator 的 paired foot、
   contact、end-object 及其余 protection CI 必须完整报告，但不新增 best-of-two selection。
8. **分类、停止规则与 artifact contract。** Mechanism/protection/absolute gates 全通过时分类
   `immutable-gt-near-ground-positive-candidate-stop`；foot gate 失败但 D2-X protection 通过为
   `immutable-gt-near-ground-transfer-negative-stop`，支持 binary support restriction 减少冲突但
   未解决 rollout/predicted-floor 语义差异；foot gate 通过但任一 protection 失败为
   `immutable-gt-near-ground-conflict-negative-stop`；foot 与 protection 均失败为
   `immutable-gt-near-ground-joint-negative-stop`；absolute gate 失败为
   `immutable-gt-near-ground-positive-but-not-effective-stop`；任何 contract failure 为
   `immutable-gt-near-ground-contract-failure-stop`。Artifacts 必须包括 manifest/resolved/
   same-context preflight、GT-floor/gate audit及 hashes、完整 logs/metrics、20 checkpoints 与 80 RNG
   sidecars、resume evidence、internal per-sequence/gradient records、official aggregate/
   per-sequence/bootstrap/gates、normalization/mask/provenance audits、tree hashes 和所有 failure。
   完成一个 fixed-budget run、internal diagnostic 和一次 official evaluation 后必须停止；不得改为
   soft/continuous gate、zero-velocity objective、PCGrad、sampler/contact/penetration intervention，
   不得 sweep threshold/multiplier、选择 midpoint、resume D2-Y/D2-Z、post-hoc 改 gate，或启动
   consistency、HSIPrior、Mixer。即使 positive，也只产生待用户另行确认的 diffusion candidate。

#### 2026-07-24 Phase 1B D2-Z implementation/lifecycle identity amendment

用户已于 2026-07-24 在 plan-only commit
`2b818726093fc15ff819e76fd12119eb329343bf` 后单独授权开始 D2-Z CPU/code implementation。
Authority 在任何 source change 前执行 `date`，得到 `2026-07-24 10:42:57 CST (+0800)`；因此只绑定
以下当前日期 identities：

- CPU gate audit：`p1-hoi-d2z-gate-audit-s42-20260724`，
  subphase `1B-D2-Z0-gate-audit`；
- training：`p1-hoi-d2z-immutable-gt-near-ground-gating-s42-20260724`，
  subphase `1B-D2-Z0`；
- internal diagnostic：
  `p1-hoi-d2z-immutable-gt-near-ground-gating-internal-s42-20260724`，
  subphase `1B-D2-Z0-internal`；
- official evaluation：`p1-hoi-d2z-native-eval-s42-20260724`，
  subphase `1B-D2-Z0-eval`。

本 amendment 只授权实现 gate audit、dataset-derived loss metadata、binary gated velocity reduction、
config/fail-closed contracts、internal diagnostic、target-only evaluator 和 CPU tests，并允许在 clean
implementation commit 后用 authority CPU 执行已绑定的 deterministic gate audit。仍不授权 GPU
smoke、training、official evaluation、任何 checkpoint loading/selection 或 consistency；CPU gate
audit 不得加载 checkpoint。若任一 GPU lifecycle 未在 2026-07-24 启动，必须在未来用户另行授权的
执行日先运行 `date`，append identity-only replacement，并永久禁用未使用的 20260724 workload id。
不得借 identity replacement 修改 D2-Z scientific mechanism、controls、gates、budget 或 stop rule。

#### 2026-07-24 Phase 1B D2-Z CPU/code implementation contract

已按上述授权完成实现，但尚未执行 CPU gate audit 或任何 GPU/checkpoint workload：

- 保持既有 `code/priors/data.py` 与 `code/priors/losses.py` 的封存 SHA-256 分别为
  `62132421b973b1d77c273f80ce48b81507966c0fe75563acd8c1e2158cb54cc5` 和
  `e14cee19e59e9ac698d4d412ccd388f9d0bf903f22e6774b13cc736087d9d1be`；D2-Z 的
  dataset metadata 与 gated reduction 独立实现在 `code/priors/d2z.py`，防止改变 D2-X/Y
  及更早训练路径。`D2ZPriorWindowDataset` 只向 loss batch 增加 `[14,4]` bool gate，不改变
  232-D `x`、condition 或 model API；`d2z_hoi_training_losses` 复用未改变的 D2-X residual/
  FK/reconstruction 逻辑，只替换 velocity scalar reduction。all-false gate 必须与 D2-X
  velocity/total bit-exact，all-true gate 必须与 D2-Y bit-exact。
- `tools/audit_hoi_d2z_gate.py` 是 authority/CPU-only、0 checkpoint load、0 optimizer update 的
  reportable audit；它对 train/internal-validation 的每个完整 aligned GT sequence 调用哈希锁定的
  official floor 函数，封存 source hashes、ordered sequence/window IDs、per-joint counts、
  per-window occupancy、floors 和 sealed 32-sequence/96-window counts。训练 config 中 audit
  path/SHA 当前故意为 `null`；在 clean implementation commit 产生 reportable audit 并另行
  append hash-binding amendment 前，D2-Z validator 必须 fail closed。
- `tools/diagnose_hoi_d2z.py` 固定 D2-X/Y/Z early/mid/final 与 timestep `0/249/499`，输出
  active/inactive per-sequence residual、gate occupancy、root/rotation/input/output/Transformer
  gradient norms、与 reconstruction/FK/object-surface/goal cosine，以及同一 prediction 上
  uniform-D2-Y vs gated-D2-Z gradient retained fraction/cosine；该 artifact 仅验证合同，不参与
  checkpoint selection。
- `tools/run_hoi_d2z_evaluation.py` 以封存 D2-X records 为唯一 primary control，完整报告但不以
  D2-Y comparator 或 internal diagnostic 作选择；六种 classification、Phase-0 absolute gate、
  penetration finite-mask contract 和所有 control hashes 均保持预注册定义。共享 D2-X evaluator
  只新增默认空的 extra-artifact hash hook，D2-X/D2-Y 行为不变。
- Authority 使用规定 Python 完成 282 项 CPU 测试，全部通过；registry validation 仍须在包含本
  amendment 的 logical commit 前再次通过。测试覆盖 gate 的 immutable-previous-frame/strict-
  threshold 语义、active slot 梯度倍率、all-false/all-true loss parity、audit hash/schema/split/
  coverage fail-closed、from-random config/provenance、internal record contract 和全部六种 evaluator
  classification。测试过程中未调用 CUDA，未加载 checkpoint，未创建 optimizer/training result，
  未选择 checkpoint，也未授权 consistency。

下一步只允许先形成 clean implementation commit，然后由 `tools/experiment.py` 启动已绑定的
authority CPU gate audit。若 audit 与 sealed preflight 任一 count/hash 不一致，立即登记
`contract-failure` 并停止；若通过，再以单独 plan/registry hash-binding commit 固定 audit
artifact 和 worker-local path。该步骤本身仍不授权 worker fast-forward、GPU smoke、training、
internal/official evaluation、checkpoint load/selection 或 consistency。

#### 2026-07-24 Phase 1B D2-Z CPU gate-audit r0 failure / r1 amendment

Implementation commit `3034510418fdc76e334cb8643c74414ded045726` clean 后，authority 通过
`tools/experiment.py` 启动 `p1-hoi-d2z-gate-audit-s42-20260724`。该 run 完成 manifest、
resolved-config 和 CPU preflight 后，在首次 full-split window gate shape validation 发现
`human_joints_aligned.npy` 的锁定 raw shape 是 `[T,28,3]`，而新 helper 错误限制为
`[16,24,3]`，因此在 0 checkpoint load、0 optimizer update、0 CUDA call 时失败；未写出
`gate_audit.json`，无科学 counts/result。失败 manifest SHA-256 为
`ea5dd7473989c50de9fedf8268aa7ba2133dac514755b5fed50c5be8da33083d`，failure metrics 为
`a5e8f25dad08a481a772b2b67a5e9ec5d5129375916e7b1ff74b319c0839b89e`，preflight 为
`fdee59171a6c2c868ac4abecb2302595ee79436ca41d69ad43b40e1122d36b02`，resolved config 为
`e8d25543975f0fd2182ada97e1bd10091549d4792bf915428c9f0dfd2fe756ed`，4-file failure tree
SHA-256 为 `11553516ef660118e2e66b8ca9eb0f277c34aa21af10c8fca782a15b4b6cdc9f`。原 run id 永久禁用，
不得把该 operational failure 混入 D2-Z 科学结论。

Authority 在任何修复 source change 前再次执行 `date`，得到
`2026-07-24 11:10:24 CST (+0800)`。允许唯一的 fail-closed 修复：gate helper 的输入合同从错误的
exact `[16,24,3]` 改为锁定 raw aligned source 的 exact `[16,28,3]`；foot joints
`[7,8,10,11]`、previous sampled frame、thresholds、floor、binary comparison、counts、
selection 和所有科学机制均不变。修复与 regression tests 必须单独提交；随后只允许使用新
`p1-hoi-d2z-gate-audit-r1-s42-20260724`、subphase `1B-D2-Z0-gate-audit-r1` 从头重跑 CPU
audit。r1 仍为 0 checkpoint/GPU/training；若任何 sealed count/floor/hash 不匹配，登记失败并停止。
本 amendment 不授权 worker sync、GPU smoke、training、internal/official evaluation、
checkpoint load/selection 或 consistency。

#### 2026-07-24 Phase 1B D2-Z CPU gate-audit r1 completion / hash binding

Shape-fix commit `1bb5ff5cd71eb67c0697094f7c89b0aed9c0643f` clean 后，reportable authority
CPU run `p1-hoi-d2z-gate-audit-r1-s42-20260724` 完成。Sealed 32-sequence/96-window
preflight 精确复现：

- joint `[7,8,10,11]` active counts 为 `[1096,1081,1211,1232]`，每 joint denominator
  `1344`，总计 `4620/5376=0.859375`；
- floor min/median/max 为
  `[0.026023785583674908,0.04509613197296858,0.05307581648230553]`；
- selection SHA-256 为
  `30524c88481f6cb81e8063073d510ad01543be92d91eb4ef9b2b8a376cc4fbae`；
- independent loader replay 再次得到相同 96-window counts，canonical payload replay 与 artifact
  内的 `30add2511afbc651546a6e5d038532455a22b82fc02b2763e01bd8ebc9b911f4`
  相同。

Full train split 为 4,088 sequences / 568,486 windows，active counts
`[6501264,6638139,7366360,7473398]`、total `27979161/31835216=0.8788745457`；
internal-validation 为 216 sequences / 29,382 windows，active counts
`[333433,337070,378355,382271]`、total `1431129/1645392=0.8697799673`。两边
nonfinite floor/gate 均为 0，zero-active windows 均为 0；train/internal fully-active windows
分别为 139,629/7,000。该高覆盖率是预注册 binary threshold 在真实 split 上的已验证性质，不授权
改 threshold、soft gate 或 weight normalization。

Authority immutable artifacts：

- gate audit：
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2z-gate-audit-r1-s42-20260724/gate_audit.json`，
  SHA-256 `d56f1cbc5297b82d768cd396ab1a49c6e33d4101d156c0375501bf32ae055faa`；
- manifest SHA-256 `2e522c3b15b6f255d76a0d9dd8458bf7fa103b2066525efb5b10818e89032637`；
- preflight SHA-256 `cc0e88ac024a6f4ba5a99e2c81d3eef04e710f95376d9f1fc9424a717a6fa643`；
- resolved-config SHA-256
  `d5e73b58725d94bb17ec8fab5fcf3029681831a97138d2538d109b3bb88e5934`；
- 4-file completed artifact tree SHA-256
  `2c3db75fd2227974ecaf939a1e22dd6cccd6ebaf739ee424d1c8ce6aa55cdbca`
  （69,313,536 bytes；manifest 内嵌 sealed metrics）。

D2-Z training config 现在只绑定 portable checkout-local artifact path
`${repo_root}/results/experiments/p1-hoi-d2z-gate-audit-r1-s42-20260724/gate_audit.json`
与上述 gate-audit SHA；worker 若以后另行获准，必须主动复制该 immutable file、校验 SHA，并在
同一 committed Git object 上通过 dataset/config preflight。当前仍不授权 worker sync、artifact
transfer、GPU smoke、training、checkpoint load、internal/official evaluation、checkpoint
selection 或 consistency；本 session 在 hash-binding commit 后停止等待用户确认。

#### 2026-07-24 Phase 1B D2-Z worker/GPU execution authorization amendment

用户已于本 amendment 前明确授权继续执行 D2-Z worker sync、artifact transfer、GPU smoke 与固定
正式训练，并要求正式训练稳定后停止轮询、待用户通知训练结束。Authority 与 worker 在任何本次
文件修改或 GPU workload 前分别执行 `date`，均得到
`2026-07-24 11:23:24 CST (+0800)`；authority 为 clean
`phase/01b-hoi@4d087689d38d338d4a62abffa70bfb77ab46b0fe`，worker 为 clean
`phase/01b-hoi@2cd4991fa3a39f0b6b0d98912a710a7d4c70a964`。worker 的旧 HEAD 是已封存结果
提交，不是 corruption；必须在本 amendment 形成新的 clean authority logical commit 后由 worker
主动 fast-forward 到完全相同的 Git object，worker 不得编辑 source。

本次只新增 execution authorization，不改变 D2-Z 的 scientific mechanism、threshold、multiplier、
loss reduction、representation、architecture、conditions、split、seed、effective batch、budget、
evaluator、gates、classifications 或 stop rule。授权范围固定如下：

1. worker 主动拉取 exact authority commit，并主动复制 authority 的 immutable
   `p1-hoi-d2z-gate-audit-r1-s42-20260724/gate_audit.json` 到 config 已绑定的
   checkout-local ignored path；必须校验 SHA-256
   `d56f1cbc5297b82d768cd396ab1a49c6e33d4101d156c0375501bf32ae055faa`。
2. worker 使用 `/home/yujinlun/data/envs/infbagel/bin/python`、`ROOT_DIR` 等于 checkout root、
   `INFBAGEL_WORKER_EXPERT=hoi`，完成 role-applicable CPU tests、registry validation、fully
   resolved Hydra config 与同一 GPU execution context 的四卡 idle/preflight archive。任一
   commit、audit、data、asset、config、idle 或 unresolved-interpolation contract 失败即停止，
   不得自由重试 formal run id。
3. 允许一个独立 operational GPU smoke：
   `p1-hoi-d2z-gpu-smoke-s42-20260724`、subphase `1B-D2-Z0-gpu-smoke`。它只在 worker GPU 0
   以 seed 42、随机初始化、真实 D2-Z train windows 和 production model/diffusion/loss path执行
   一次 forward/backward；必须验证 audit hash、binary gate 同时含 active/inactive entries、全部
   loss finite、motion input/output/Transformer 关键 parameter gradient finite 且 nonzero，以及
   peak allocated/reserved memory。它不得创建 optimizer、执行 update、写 checkpoint、加载任何
   checkpoint 或形成科学 selection；无论成功失败都必须保留 manifest、resolved config、
   preflight、metrics 和 log，原 smoke id 不得复用。
4. smoke 通过后允许启动已绑定的唯一 formal training
   `p1-hoi-d2z-immutable-gt-near-ground-gating-s42-20260724`、subphase `1B-D2-Z0`。必须使用
   4×RTX 3090、batch/GPU 512、effective batch 2048、seed 42、随机初始化、61,440,000 windows /
   30,000 updates 和全部既有 exact config；禁止设置或加载 init/resume/weight-init/EMA/released/
   author/D2-V/D2-X/D2-Y/prior checkpoint。通过 `tools/experiment.py start` 创建 reportable
   manifest，并在 worker-owned persistent `tmux` 中运行，不依赖反向 SSH 存活。
5. 只监测 initial stability interval：resolved/preflight/manifest 通过，loss 与所需 gradients finite/
   nonzero，无 overflow，四卡显存满足既有 headroom contract，并产生首个 3,072,000-window
   checkpoint 及四个 RNG sidecar。对该完整原子 checkpoint 做只读 schema/hash/contract
   inspection 作为 resumable evidence，不实际 resume、不创建第二训练 lineage。满足后记录吞吐和
   ETA，停止连续轮询；训练结束、artifact recovery、internal diagnostic 与 official evaluation 等待
   用户再次通知。若此前失败，保留失败并停止，不得复用 id。

本授权不允许 checkpoint selection、D2-Z/D2-Y resume、修改 gate 或旧 gate、internal/official
evaluation、consistency distillation、HSIPrior 或 Mixer。Formal training 即使完成也只产生尚未
评估且不可选择的 D2-Z artifact；后续 lifecycle 仍受原预注册 stop rule 约束。

上述 operational smoke 已实现为 `tools/smoke_hoi_d2z.py`（SHA-256
`504fa048b142bb95ed13426ae7693f0801a9ce3050dd0ccb011dc8a5b16e68e0`）。工具只构造随机
HOIPrior、读取真实 D2-Z batch、执行 production diffusion/loss forward/backward 并原子写 metrics；
source 中不存在 optimizer、`step`、`torch.load` 或 `torch.save`，因此不能更新参数或读写
checkpoint。Authority Python 完成 py_compile 与 15 项 D2-Z tests，全部通过；截至该验证时
GPU smoke/formal training 均未启动。必须先提交该工具、tests 与本记录，再由 worker 主动
fast-forward 到 exact commit 后执行。

#### 2026-07-24 Phase 1B D2-Z GPU smoke r0 failure / r1 amendment

worker 在 exact commit `875a544679c0de7b493d9aa3c4ec55b2c5b547f0` 上通过 resolved config、
四卡 idle preflight 与 `tools/experiment.py start` 启动
`p1-hoi-d2z-gpu-smoke-s42-20260724`。r0 在任何 model forward/backward、optimizer、update 或
checkpoint 操作前 fail-closed：工具先创建 DataLoader iterator/读取 batch，再构造随机 model；
PyTorch DataLoader iterator 会消耗 global Torch RNG，因此 observed initial model SHA-256 为
`c1e9ce860311307f0844d4d31daa5b431583b6781cbddf2bf58f9a653b85554c`，不等于正式训练 rank-0
锁定的 `ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e`。这是独立 smoke
工具的执行顺序 defect，不是 D2-Z dataset/loss/trainer/scientific mechanism defect；formal training
尚未启动，原 smoke id 永久禁用。

r0 manifest/metrics/resolved/preflight/run-local-registry SHA-256 分别为
`e5771a7a02fc511f3f3f2380f289883989848ca15eff866428f982975f63af92` /
`b8ee77f88d74e993c7233c1c4b0b90daea1341537716582f59c892c09fdb25eb` /
`b81490a2941193679b9d9c1b9e85713884ef27ec2ed8076bb06c39cb6d202c26` /
`cfa94d0e598b4ce58b006b771c79197fa0b2df7ca52ab9fe8376ac98d3bcd1c7` /
`2aff76bce456c5ef1da963fe4648b529212ab79a0ecec3c0a1e259b132b79374`。7-file tree
SHA-256 为 `cdd6934b5cb68fc42a71feee1f165695256376b08044ff3e2ced76806193edd3`，worker 主动
回收到 authority staging 后 tree hash 完全一致。

Authority/worker 在任何 fix 前再次执行 `date`，分别得到
`2026-07-24 11:36:41/11:36:42 CST (+0800)`。只允许一个 execution-order fix：在 smoke 工具中
先按 seed 42 构造并校验随机 model，再创建 DataLoader iterator/读取 batch；production trainer、
D2-Z dataset/loss、config、gate、timesteps、batch、gradient checks 和所有科学合同不变。Fresh r1
run id 固定为 `p1-hoi-d2z-gpu-smoke-r1-s42-20260724`、subphase
`1B-D2-Z0-gpu-smoke-r1`。必须先提交本 failure/amendment，再实施单一 fix、更新 tests/tool identity
并形成新 clean commit；worker 主动 fast-forward 后从头生成 r1 resolved/preflight/manifest。若 r1
失败，保留并停止 formal training；若通过，才按已授权合同启动唯一 formal run。

r1 tool fix 已严格只把 model construction/hash validation 移至
`raw_batch = next(iter(loader))` 之前，并将 tool identity 改为 fresh r1；修复后工具 SHA-256 为
`9279665f727abff959d0f069d6509273ec031ac7d196fb772e5d6cd023b57ed2`。Authority py_compile
与 15 项 D2-Z tests 全部通过；新增 regression 明确断言 model hash 语句位于 iterator 之前，并继续
静态禁止 optimizer step、checkpoint load/write。截至该验证时 r1 GPU smoke 与 formal training
均未启动。

r1 GPU smoke 已在 exact commit `f67a6437f1ca261ad78f5a8eceab6daabaeb40b5` 上完成：
initial model SHA 精确为 `ad6980ce...`，真实 8-window batch 的 gate 为 441 active / 7 inactive，
全部 loss finite；motion input/output、Transformer、predicted root translation 与 joint rotations
gradient norm 分别为 `55.2236/107.2537/44.4260/1.08801/0.167582`，均 finite/nonzero。GPU 0
peak allocated/reserved 为 `249,652,224/299,892,736 bytes`，headroom
`24,996,151,296 bytes`；optimizer created/updates、checkpoint loads/writes 均为 0。

r1 manifest/metrics/resolved/preflight/run-local-registry SHA-256 分别为
`d0c224fb9af233ceb3a88722ef0c8704cb6568247061279d5be6350e4dba8fbc` /
`ff66f45ab0fcb72c06c1e8664404a11e3d7950bc57d1e7b766ef0bd98a20d2a4` /
`b81490a2941193679b9d9c1b9e85713884ef27ec2ed8076bb06c39cb6d202c26` /
`fe0e3aab513cbec61d2e6b53872ba4b251ce47fc98e3bc7ea70bd6aaaab60230` /
`ebcd17935cdd28a6165e333d858db94e5e7e24ac33401d9cf3d3cb6ca74a4e89`；7-file tree
SHA-256 `3a36211aaa3143a965828f37c2cbbc53872188162905dd93fe3017f55a12f148`
已由 worker 主动回收且双端一致。Operational GPU smoke gate 通过；只有在本 completion metadata
形成新 clean commit、worker 主动 fast-forward，并重新生成 formal-run resolved config 与四卡
idle preflight 后，才允许启动唯一 D2-Z formal training。

#### 2026-07-24 Phase 1B D2-Z formal training completion

worker 在 exact commit `2634cea35e86cd054c9283fdfddb89fe507dc066`、clean
`phase/01b-hoi` 上完成唯一
`p1-hoi-d2z-immutable-gt-near-ground-gating-s42-20260724`。训练从随机初始化
`ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e`
开始，未加载 released/author/D2-V/D2-X/D2-Y/prior/resume/EMA 或任何 checkpoint state；
weight-initialization record 的 source 为 null、restored components 为空，所有旧 optimizer/EMA/
scheduler/scaler/RNG state counts 均为 0。

固定 61,440,000 processed windows / 983,040,000 frames / 30,000 optimizer updates 完整结束，
exit code 0。Loss 与 required gradients 全程 finite/nonzero，AMP overflow 0；4×RTX 3090、
micro-batch/GPU 512、effective batch 2048、Adam `1e-4`、FP32 和所有 exact contracts 未变。
训练 wall time `19,319.6752 s`，throughput `3,180.1777 windows/s`，每 rank minimum memory
headroom `21,208,694,784 bytes`。Mean training total/velocity 为
`0.0478505/0.0273295`；final 32,768-window validation total/velocity 为
`0.0502446/0.00381642`，finite。20 个 3,072,000-window cadence checkpoints 与 80 个 RNG
sidecars 完整，逐 checkpoint 实际 SHA/bytes 与 metrics 全部一致；terminal online checkpoint
SHA-256 为 `44c1ff8c8cf4abc2c7312923f64183e1a4a307166d187c9fcaff03abdcc162b6`。
该 checkpoint 只是未评估 artifact，尚未选择。

Manifest/metrics/resolved/preflight/run-local-registry/resume-evidence SHA-256 分别为
`39a400060c01056f03c03eff28a6ba83f0e0b88b520394a43a72eaf0903b28df` /
`84b682c4ec78ce80402538c6304a419f8bfcf879b7bf3163ced68d8032d47d09` /
`b81490a2941193679b9d9c1b9e85713884ef27ec2ed8076bb06c39cb6d202c26` /
`a94293326f41243b89a9911d3d1c94a34755bd41cc6d70baf8a0f2c2dd83c38b` /
`f5e2848696009542ad50353de8d572c9498c46945c8d4a69bf438410c8659188` /
`fed416229e281f2097a4a7a5ddb232f920f4a1fd8dd616bbba1a1cec2e43330b`。
完整 113-file / 7,127,278,269-byte training tree 已由 worker 主动回收到 authority staging，
双端 SHA-256 均为
`41de8a4a2b94b225d82d628ca3d074408b33619550f8809e1d6576ef2b1f4726`。

Training lifecycle gate 通过，只授权使用 fixed early/mid/final online checkpoints执行已预注册的
internal diagnostic，然后对 fixed final online checkpoint 运行唯一 official evaluation。不得根据
training/validation loss 或中间 checkpoint 选择模型，不得 resume、增加预算或启动 consistency。

#### 2026-07-24 Phase 1B D2-Z internal diagnostic r0 failure / r1 amendment

首次 fixed internal run
`p1-hoi-d2z-immutable-gt-near-ground-gating-internal-s42-20260724`
在 exact commit `0b94844c58bb2a39c6606acae3aea83c43e4ee9b` 上通过 clean-worktree
preflight 并开始读取预注册的 9 个 fixed D2-X/D2-Y/D2-Z early/mid/final online checkpoints，
但在生成任何 scientific diagnostic metrics 前失败。确定原因是 reporting contract 错误地要求
每条 sealed sequence 的 active 与 inactive residual strata 都非空。固定 32-sequence/96-window
selection 的 active/inactive entries 为 `4620/756`；所有序列均有 active support，但 sequence
indices `[2,5,16]` 是合法的 fully-active sequences，其 inactive count 为 0，因此对应
per-sequence inactive MSE 数学上未定义。该情况未违反既有 gate audit（它只要求全 selection
active 非零，并明确允许 fully-active windows），不构成 D2-Z training、checkpoint、gate 或
scientific mechanism defect。

r0 必须永久保留为 failed，禁止复用 run id。其 manifest/failure-metrics/preflight/
run-local-registry/log/exit-code SHA-256 分别为
`ca44be99e85c90f2f437bc4236566141e3c01f092df9e80b442c5f0571d6263a` /
`39f77a06082696572920003c8e7362e0b9e6d34dc0110dabb000fff9e06152c4` /
`5e73c4f031a190b4d75e6aff8df66df895ce8a78805751118d80aec7d8dd8847` /
`cc9173f31684ed7ce26a978753b6a1ae29289013bd75b2892ac3084e055790a0` /
`8784e793ddedf37cef059bcfb38addd2bb70e11e8633267c266ae47df66b893f` /
`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`。
完整 6-file / 69,454-byte tree 已由 worker 主动回收到 authority staging，双端 SHA-256 均为
`239fb2bf4a4ce7ff456bea8a0ebd265bcde8b12cef14a8ae0e81ba081b472544`。
r0 未创建 optimizer、未训练/写 checkpoint、未使用 official-test sequence、未选择 checkpoint，
也未启动 consistency。

只允许一个 fresh replacement：
`p1-hoi-d2z-immutable-gt-near-ground-gating-internal-r1-s42-20260724`
（subphase `1B-D2-Z0-internal-r1`）。唯一 implementation delta 是：保留每条 sequence 的 exact
support count；count 大于 0 时按原公式报告 per-sequence MSE，count 等于 0 时在 JSON 中报告
`null` 和 count `0`，不得填 0、插值、删除 sequence 或改变 aggregate denominator。Finite
validation 只适用于 count 大于 0 的 strata；全 selection 的 active/inactive aggregate 仍必须
各自非空且 finite。必须增加 regression 覆盖 zero-support 的显式未定义语义。

r1 的 selection SHA、32 sequences/96 windows、全部 9 个 checkpoint paths/SHA、timesteps
`[0,249,499]`、paired noise/dropout、所有 residual/gradient/cosine computations、D2-X/D2-Y/D2-Z
比较、bootstrap、seed 42、GPU device 和 non-selection 用途保持不变。不得借此修改 training、
checkpoint、gate、official evaluator、candidate gate 或 classification。必须先提交本
failure/amendment，再修改工具并形成新的 clean logical commit；worker 主动 fast-forward 后通过
fresh preflight/manifest 执行 r1。只有 r1 完整封存后，才可运行已预注册的唯一 official-438
evaluation。

r1 tool fix 已严格实现上述契约：undefined entries 在内部 tensor 中保持 `NaN` 语义，仅在
serialization 时根据 exact zero count 转为 JSON `null`；count-positive entries 沿用原除法，
aggregate active/inactive RMS 与 denominator 未变。`diagnostic_summary` 现在逐项验证
`null iff count==0`、所有 defined MSE finite、两类 aggregate support 非空，并继续禁止 selection。
工具 identity 已更新为 fresh r1，tool/test SHA-256 分别为
`02799a5037fb8f0884eebabf0fea24e70a1f17dd6d3c572c2b10862a2c2a17cf` /
`acf57c66be14b3e5e0f816cbada0e62adcf0c8e3d1b195d4d2154cf9bd600bfd`。
Authority py_compile、16 项 D2-Z tests 与 registry validation 均通过；截至本验证时 internal r1
尚未启动。

internal r1 已在 exact commit `ff5148a7040cc6c9679393557a395e3f147a43b8`、clean worker
上完成，exit code 0、runtime `12.1607 s`，固定 9 checkpoints × 3 timesteps 的全部 required
records/scalars 通过合同。3 个 fully-active sequences 在每个记录中原位保持
inactive `null/count=0`，其余 defined MSE 及 aggregate active/inactive RMS、gradient norms/
cosines 均 finite；selection use、official-test sequences、optimizer/update/checkpoint write/
selection、consistency 均为 0/false。Manifest/metrics/preflight/run-local-registry/log/exit-code
SHA-256 分别为
`d5f3d3bd04bdd290ec5dbd599d706a99daac78e1ec51259b88f9280fc9a0043d` /
`0540afa33b485f3a893973d827fe0c48bfca08df3e0b3fdd54fa1f14ce9256e3` /
`abca0349a533bf288f5a6ef80bfee447423c13563a7c9875ff2d098959e5626b` /
`b09e80dcc370f31a2aa43f9a39d571ee01477bc5999236a99ea002dd0f1f4f50` /
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` /
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。
完整 6-file / 515,725-byte tree 已由 worker 主动回收，双端 SHA-256 为
`039894526a9044865ea0fcfbdee1ba9c51f148a74860048ba16fdd6b1f31e960`。

该 non-selection diagnostic 给出三项描述性证据。第一，final D2-X/D2-Y/D2-Z 的 active routed
RMS 在 timestep `0/249/499` 分别为
`0.007179/0.005561/0.005675`、`0.008951/0.007029/0.007054`、
`0.010113/0.007667/0.007912`；D2-Z 保持 D2-Y 的大部分 surrogate gain，但没有进一步优于
D2-Y。第二，D2-Z final gated-vs-uniform gradient cosine 为
`0.9949/0.9592/0.9483`，norm retained fraction 为 `0.7767/0.7833/0.8587`；由于 sealed
selection 的 active occupancy 达 `85.94%`，binary gating 主要缩小 uniform signal 而没有形成
明显不同方向。第三，D2-Z gated routed gradient 与 FK 的 cosine 为
`0.6908/-0.5110/-0.1521`，在两个 noisy strata 仍冲突；timestep 249 的 gated all-parameter norm
`0.01431` 也远小于 reconstruction/FK/object-surface norms `0.79246/0.22699/0.37023`。
这些只支持“near-ground restriction 是相对温和、且 noisy-step FK conflict 未消失”的推断；
不能预测或替代 official foot-sliding paired CI，也不能选择 checkpoint。

official evaluator 在实现时绑定的是已失败的 r0 internal identity，故唯一允许的 pre-evaluation
contract amendment 是把 expected internal run id 改为上述 r1，并使 validator 与 sealed r1
schema 一致：逐 sequence 要求 `count==0 iff MSE is null`，count-positive MSE finite，active/
inactive aggregate 均非空；其余 checkpoint/control hashes、target generation、bootstrap、
candidate/protection/absolute gates、classifications 与 stop rule全部不变。必须先提交本 completion/
binding amendment，再修改 evaluator/tests 并形成 clean commit；worker 主动 fast-forward 后才可
创建原已绑定且尚未使用的 `p1-hoi-d2z-native-eval-s42-20260724` manifest。只允许该一次
official-438 run。

evaluator binding fix 已只更新 expected internal run id/SHA，并逐 sequence 验证上述
`null/count` schema 和 aggregate nonempty；target generation、controls、statistics、gates 与
classification code 未变。Evaluator/test SHA-256 分别为
`b02954b3f72fe27162fdcdc0fb56d5bc6e119bdd8fdca58408151d2256dcc8fa` /
`23433022b5b990a8f85f864691d8e5d2a690969fc1a1646053e599ef5831ab40`。
Authority py_compile、17 项 D2-Z tests 和 registry validation 均通过；截至本验证时 official
evaluation 尚未启动。

#### 2026-07-24 Phase 1B D2-Z official completion / joint-negative stop

唯一 `p1-hoi-d2z-native-eval-s42-20260724` 已在 exact commit
`38d7e409a1b6208049d1b4ce358eacfef0dc9f3a`、clean worker 上完成，exit code 0。协议严格为
official 438 sequences、每 sequence 3 windows、500-step unguided diffusion、fixed final online
D2-Z checkpoint；D2-X/D2-Y records 均原样复用，D2-Y 只作 non-selection comparator。

D2-Z 的 MPJPE/end-object/xy/object-translation 为
`12.2655/4.4567/3.8216/16.3945`，foot sliding/contact F1 为
`0.363433/0.630798`，hand/human penetration loss 为 `0.219640/3.447872`。Primary D2-X minus
D2-Z foot-sliding paired mean 为 `-0.000423015`，10,000-replicate sequence bootstrap 95% CI
`[-0.0221524,0.0209344]`，故 registered foot gate 失败。D2-Z minus D2-X contact-F1 mean/CI 为
`-0.00662759/[-0.0274279,0.0137967]`，下界低于 `-0.02`；D2-Z/D2-X end-object ratio CI 为
`[1.12841,1.25042]`，上界高于 `1.10`。其余 MPJPE、xy、object-translation 与两项 penetration
protection 通过，固定 181-sequence finite mask 精确匹配。所有 Phase-0 released-baseline
absolute diffusion checks、provenance/lifecycle、normalization 与 finite contracts 通过。

因此严格 classification 为 `immutable-gt-near-ground-joint-negative-stop`：official foot gate
与 protection gate 均失败。不得因 absolute checks 或任一 point estimate 改写分类；final
checkpoint 仍未选择，consistency 未授权/未启动。

Evaluation manifest/metrics/aggregate/per-sequence/resolved-target/resolved-lifecycle/preflight/
run-local-registry SHA-256 分别为
`c134947136eee8a867222c55584ad744f6a8cffa72742b39a77980f541e50c6e` /
`c20738824f0475294e42551121fd7796c2041fd48949e4630e012ad2d4959ae3` /
`fb58a5ab3bd5ad0336ce02ff9a15cd7d97af8446599b147c9e2c806208a56162` /
`9f0f0e65bd0eaa4fe3ec1f495f6e4a4489c88d842256dccc3a6b9b57a1e9113f` /
`f8f1350678f3a706d6763fff2094826c618e10956f8714002dd0cb0a974efa3e` /
`86610e7c646d08474072b67826ad2a7268b7c8a20cefe93e217f446da3f244ed` /
`981c21f0d565e99ee1b9d9ae5e52667de650ddf0986e6f815448edf1a151b4af` /
`80c908260c73d1c5e27e2f4da4c76e8927575ae329be71d0d0da9171b080c9b9`。
完整 15-file / 389,634-byte tree 已由 worker 主动回收，双端 SHA-256
`9613bb8762dc1ba67e29c068dee966461cfa4ac4284a2d84c4dc671625e13bfe`。

Compact aggregate
`experiments/results/p1_hoi_phase1b_d2z_immutable_gt_near_ground_gating_s42_20260724.json`
SHA-256 为
`c2a0ff494784fac6d42485d189646b8b9618205f8fc8ed6608270ff871c16af4`；
handoff 为 `docs/phase_summaries/PHASE_1B_D2Z.md`。D2-Z 到此停止。任何下一 HOIPrior 机制必须
在新 session 先只读审计、核验 registry 下一 ID，再作 dated plan/registry preregistration；
不得选择或 resume D2-Z、改变旧 gate、重跑 official evaluation、启动 consistency、HSIPrior 或
Mixer。

#### 2026-07-24 Phase 1B D2-AA Table-5 completeness evaluation 预注册

论文 Table 5 的完整 HOI 列为 `Te/Txy/FS/Rprec/FID/Cprec/Crec/Cf1/C%/Pbody/MPJPE/`
`Troot/Tobj/Oobj/FPS`。D2-V/X/Y/Z 的封存 official-438 native evaluations 已覆盖其中除
`Rprec/FID` 外的质量列，但其 preregistered target overrides 明确设置
`save_chois_eval_npz=false`、`fid_rprecision_used=false`；既有 generation FPS 又是
batch-438 throughput，不能冒充论文未充分说明的 batch-1 timing。另一方面，仓库锁定的
CHOIS release evaluator 输出的是 `R-Precision@1/2/3`，论文只给一个未进一步定义的
`Rprec` 标量；D2-AA 不得 post-hoc 把任一个 rank 指标冒充论文标量。用户要求补齐这些
缺失指标后，registry 中下一个未占用 Phase 1B 标识经全量检索确认为 D2-AA。

1. **唯一目的与固定候选。** 唯一 reportable umbrella run id 为
   `p1-hoi-d2aa-table5-completion-s42-20260724`、subphase `1B-D2-AA0`、seed 42，只允许在
   `infbagel-4gpu/node01` 单卡顺序运行。固定、对称地包括 D2-V/X/Y/Z 四个 final-online
   checkpoints，SHA-256 分别为
   `e0705681bbaeed40d353494852494d8b7bdaf4d32da92368c0d2ceedea4c01a4` /
   `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51` /
   `8734431f89cf8739283828d5fb683212ca43143ae3482ad0473f6ed5717eb7a7` /
   `44c1ff8c8cf4abc2c7312923f64183e1a4a307166d187c9fcaff03abdcc162b6`。
   四者全部运行，不得根据已有 native 点估计只补有利候选；不加载 optimizer、EMA、RNG、
   released/author checkpoint，也不训练、resume 或写 checkpoint。
2. **Table-5 quality/export protocol。** 每个 candidate 从固定 checkpoint 重新执行一次
   official 438 sequences × 3 windows、500-step unguided production diffusion，online weights、
   seed 42、无 CFG/dynamic perception/guidance/scene/CHOIS selection，并只新增
   `save_chois_eval_npz=true` 的 matched 438-pair、126-frame、Z-up `global_jpos[T,24,3]`
   export。重新计算的全部 native aggregate/per-sequence 值必须与各自封存 official result
   在浮点容差内一致；它们只作为 deterministic regeneration audit，绝不替代原记录、重开旧 gate
   或成为第二次 selection evaluation。四份 GT export 的 ordered IDs、shape/content tree hash
   必须彼此相同并匹配 Phase-0 GT contract；否则 contract failure。
3. **锁定 CHOIS 指标。** 对四份 prediction/GT export 分别使用 pinned CHOIS commit
   `8ec585aa0200fd2a890ffb12897bcf69ae719463`、text-to-motion commit
   `72df96ec453edea2fbe9603b1d58a955eaf71636`、feature checkpoint SHA-256
   `a125bc15ffd9772686737111c7501ecee0a2d8571d9aca348ec1195ddef78775` 和 Phase-0
   normalization/annotation/GloVe assets。完整报告 FID、Matching Score、
   R-Precision@1/2/3 与 Diversity；必须同时报告 upstream `drop_last=true` 后实际进入 embedding
   指标的 sequence 数与被丢弃 ID，不能只写 export count 438。R-Precision@1/2/3 与
   Matching Score 使用 10,000-replicate、seed-42 sequence bootstrap；非加性 FID 使用固定
   200-replicate paired sequence bootstrap，仅作不确定性描述，不设 capability gate。不得改变
   upstream batching、normalization、feature network 或 retrieval definition来追求论文数值。
4. **本地 efficiency 补充。** 四个 checkpoint 还要各执行一次相同确定性首序列的 warmup 后
   batch-1、3-window timing-only workload，CUDA 在测量边界同步；报告 generation seconds、
   generated frames、FPS 与 end-to-end seconds。该本地 batch-1 数值与 batch-438 throughput
   分列，并与论文 Table-5 FPS 分列引用；论文未公开完整 timing protocol 时不得声称严格同协议。
   Timing subset 的质量指标不得并入 official-438 aggregate。
5. **成功/失败与停止。** 只有 checkpoint/source/data/evaluator/assets hashes 全部匹配、
   clean worker manifest/preflight/resolved configs 完整、四个 candidate native regeneration
   audit 通过、每组 438 matched exports finite/同 ID、四组 CHOIS 点估计和注册 uncertainty
   finite、四组 batch-1 timing finite，才分类
   `table5-completion-pass-nonselection-stop`；任一失败分类
   `table5-completion-contract-failure-stop` 并保留全部 partial artifacts，不得复用 run id。
   D2-AA 没有模型质量 success gate，不排序或选择 checkpoint，不改变
   D2-V/X/Y/Z classification，不授权 consistency、HSIPrior、Mixer 或任何训练。
6. **Artifact contract。** 保留 umbrella manifest、fully resolved lifecycle/candidate/timing/
   CHOIS configs、same-context GPU preflight、commands/logs/exit codes、checkpoint和封存
   aggregate/per-sequence hashes、四组 native regeneration aggregates/per-sequence、438-pair
   NPZ trees/IDs/hashes、CHOIS point/bootstrap records、batch-1 timing、environment/dependency/
   hardware hashes、run-local registry、完整 tree hash及所有失败。大 NPZ 留在 worker/authority
   staging，不进入 Git；Git 只记录 compact aggregate、summary 与 hashes。

#### 2026-07-24 Phase 1B D2-AA CPU/code implementation contract

D2-AA 实现只新增 non-selection orchestration 与 opt-in CHOIS reporting，不修改
`code/test_infbagel_hoi.py`、`code/eval_metrics.py`、production sampler、model、loss、checkpoint
或旧 evaluation artifacts。`tools/run_hoi_d2aa_table5.py` 固定四个 final-online checkpoint、
封存 aggregate/per-sequence hashes、438×3/500-step export、相同首序列 batch-1 timing、source/
resolved/preflight/dirty-worker contracts，任一候选失败即保留 partial artifacts 并 fail closed。
Tool SHA-256 为
`5727c2a8c3e262e7de133258cb46427938025f4178cbefb2f0e293df160f1fb8`。

`tools/run_chois_evaluator.py` 的既有 point-estimate 路径保持原公式和 RNG 顺序；新增参数默认均为
disabled。D2-AA opt-in 后记录 embedded/dropped sequence IDs、对 Matching Score 与
R-Precision@1/2/3 做 10,000-replicate bootstrap，并对 paired FID 做 200-replicate bootstrap。
同时显式暴露 upstream loader 的 `batch_size=32, drop_last=true`：438 exports 中 416 条进入
embedding metrics、22 条被丢弃。这里没有改变 upstream behavior，只修复此前只报告 input count、
未报告 effective count 的 reporting omission。更新后 adapter SHA-256 为
`1038e7e1e7dc2882a2a199396b972ef1eb05335da0877c8f05310ff5ad738b4b`；
新增 7 项 D2-AA tests SHA-256
`ef70c1375d949a70652f3133514435bcf6dfd71f679e6707a051397ae6992827`。

Authority 使用指定 `infbagel` Python 完成 py_compile、7 项 D2-AA tests、22 项 governance tests、
全量 293 tests、registry validation（151 records）与 `git diff --check`。此时尚未 worker
fast-forward、创建 reportable manifest、加载 checkpoint 或启动任何 CPU/GPU evaluator workload。
下一步只能提交本 logical implementation，worker 主动 fast-forward exact object，生成
same-context preflight/resolved configs并以 `tools/experiment.py start` 创建唯一 umbrella
lifecycle；不得在 dirty worker 或未绑定 artifacts 上运行。

#### 2026-07-24 Phase 1B D2-AA completion / Table-5 reporting pass

唯一 umbrella run 已在 exact commit
`82ef8f212e77042abd4d6cedfc03fff16d9756eb`、clean `infbagel-4gpu` worker 上完成，exit code 0，
runtime `1879.9277 s`。四个 fixed final-online checkpoints 均重新运行 official 438 sequences ×
3 windows、500-step unguided diffusion；D2-V/X/Y/Z 各自重新生成的 native aggregate 18 个
scalars 与 per-sequence 5,104 个 scalars 对封存记录的最大绝对差均为 `0.0`、mismatch count
均为 0。四份 GT tree SHA-256 均为
`d439a98ea32f5d67964bc98431fe25bdffc24b63e00b42601c5355445d01742c`。

Pinned CHOIS evaluator 的 D2-V/X/Y/Z FID 分别为
`1.578132/1.775477/1.941435/1.935600`，paired embedded-sequence bootstrap 95% CI 分别为
`[1.188574,2.253517]`、`[1.303928,2.494238]`、`[1.509327,2.574472]`、
`[1.507692,2.567234]`。R-Precision@1/2/3 分别为
`0.149038/0.283654/0.413462`、`0.151442/0.283654/0.420673`、
`0.151442/0.286058/0.415865`、`0.153846/0.293269/0.406250`；Matching Score 分别为
`3.858646/3.879792/3.951129/3.918002`，Diversity 分别为
`8.740726/8.781011/8.397693/8.448088`。所有 additive metrics 使用 10,000-replicate、
seed-42 sequence bootstrap，FID 使用固定 200-replicate paired bootstrap。Upstream
`batch_size=32, drop_last=true` 未改变：每组 438 exports 中实际 416 条进入 embedding metrics，
22 条被丢弃；ordered dropped-ID SHA-256 为
`b7ddcb96dae95814e44d1df8f4fe1791c2c7930ed3ddfca55c3ea3fcde31bd15`。
这修复的是 effective-count reporting omission，不是 metric formula 或 scientific evaluator
defect；论文单一 `Rprec` 仍不得映射为任一 rank 指标。

同一 RTX 3090、相同首序列、batch-1、3-window、排除一次 warmup、CUDA synchronized 的
D2-V/X/Y/Z generation FPS 分别为
`15.2809/18.1699/18.0095/18.0700`；这些只用于本地对称 timing，不与论文未充分披露协议的
FPS 宣称严格同协议。完整 Table-5-aligned point estimates、embedding uncertainty、额外
hand/ratio/contact diagnostics 与论文原表逐列数据已写入
`experiments/results/p1_hoi_phase1b_d2aa_table5_completion_s42_20260724.json`，SHA-256
`d791c04bf1a896f4230a55e77518368cf4c5cb5c691c6ce98de65c18a87914d8`。

Manifest/metrics/resolved/preflight-r1/run-local-registry SHA-256 分别为
`a98f6aeab22859c82fba06646888922deae066be9593a97ba44a3de7f229650a` /
`9512467b4eebf5cce5b0ae40a60de63bda107e135cc45e54d9c8a4e3ffc59889` /
`c70286c63ea9fb16d19188d4b3651b0492d9cdd027d3c6c10198cb025c295495` /
`70bee2758200a7879b5bb6ec0be3d179b527deff408c1dbf63273d347e8e6097` /
`082f77f7199f3229f566622e5117f826b0631d21797d50a617e8efc482a306c1`。
完整 3,601-file / 132,167,442-byte artifact tree 已由 worker 主动回收到
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2aa-table5-completion-s42-20260724`，双端
SHA-256 均为
`1fa7f570f935d58adaf1baddd8db2367ae49aa05804310b0a8c2d1cc7febeb77`。

首次 preflight flatten 因 worker 未安装 `jq` 在任何 scientific work/manifest/checkpoint/GPU
workload 前失败，空 `preflight.json`（SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`）永久保留；
随后只使用 worker `infbagel` Python 生成并绑定 `preflight_r1.json`。原异步 start 已创建 manifest
后的一次重复 start 被 fail-closed overwrite check 拒绝，未覆盖 manifest、未重启 workload、
未复用 run id。两者均为保留的 operational events，不影响点估计或 evaluator 路径。

严格分类为 `table5-completion-pass-nonselection-stop`。D2-AA 不产生新的模型质量 gate，
不改变 D2-V/X/Y/Z 的 negative classifications，不选择任何 checkpoint，不授权训练、consistency、
HSIPrior 或 Mixer。Handoff 为 `docs/phase_summaries/PHASE_1B_D2AA.md`。

#### 2026-07-24 Phase 1B D2-AA integrated-table reporting amendment

用户澄清最终对照必须显式区分并同时列出三类来源：

1. 论文 Table 5 的 InfBaGel 1/8/16 published rows；
2. 作者提供 released consistency-distilled checkpoint 在本地 Phase-0 evaluator 上的结果；
3. 先前从 integration baseline 独立 fork 的 `phase/01b-author-repro` 上，按作者
   diffusion→consistency 训练代码在本地 8×RTX 3090 完整复现所得 CM checkpoint 的历史
   official-438 HOI evaluation。

该 amendment 只整合已存在的不可变记录，不启动 evaluator/GPU、训练、checkpoint selection 或
consistency。禁止修改或覆盖已封存的
`p1_hoi_phase1b_d2aa_table5_completion_s42_20260724.json`；新增 integrated aggregate 必须使用
独立文件
`experiments/results/p1_hoi_phase1b_d2aa_integrated_table_s42_20260724.json`，并逐行记录
`source/protocol/reporting_status`，不能把 paper-reported 与 locally-evaluated 数值冒充同协议。

历史本地作者复现的训练 commit 为
`1e982bc3a35301287e6e6a9d56325e103655f0e1`，使用 seed 42、FP32、8×RTX 3090、
per-GPU batch 256/effective batch 2048；先完整训练 diffusion epoch500 checkpoint
`44a723d20a4bbf13de8c2db78c3c375472dba20be0530993c9e00ab780747aac`，再按作者 schedule
蒸馏至 CM epoch200 checkpoint
`a9f5a72e617c2789f84e902829e0aae9118cddfa264047ce9ec320b1cc0df8ab`。Pipeline exit code 0，
但原 branch registry 将其定义为 exploratory reproduction、`reportable=false`、
`final_table_authorized=false`，且不具备 `tools/experiment.py` reportable manifest；本次用户
授权只允许将其作为明确标注的 historical local reproduction row 引用，不能提升为正式
HOIPrior candidate 或选择证据。

该 checkpoint 的既有 evaluation 使用 seed 42、official 438 sequences × 3 windows、
`sample_type=consistency`、`cm_timesteps=16`，aggregate SHA-256
`97cf420d1f2b2596efc957daee4a812a6fd04ba288eacf45abd9b020a41ad573`。Native point estimates
为 Te/Txy/FS `3.555258/3.764048/0.331956`，Cprec/Crec/Cf1/C%
`0.790091/0.749855/0.745333/0.623777`，Pbody/MPJPE/Troot/Tobj/Oobj
`2.777503/11.819468/7.788341/15.882751/1.018622`。该历史 run 没有 per-sequence uncertainty，
且 `save_chois_eval_npz=false`，所以 FID、Matching、R-Precision@1/2/3、Diversity 都必须报告
为“未评估”，禁止用 released checkpoint 或论文行代填。其 `321.6566 FPS` 是 full-438
descriptive throughput，不是独立 batch-1 或论文 FPS，必须移到额外 timing 列，Table-5 FPS
保持 null。

论文 InfBaGel 1/8/16 的完整 15 列直接引用
`https://arxiv.org/pdf/2604.04843` Table 5；它们的 `Rprec` 是论文单一标量，仍不得与本地
R-Precision@1/2/3 映射。Integrated table 完成后只允许追加 registry completion、更新 handoff
中的 reporting appendix 并提交；不得因此重开 D2-V/X/Y/Z gate、选择 checkpoint 或开始下一
科学 subphase。

#### 2026-07-24 Phase 1B D2-AA integrated-table reporting completion

只读逐项交叉检查通过：论文 InfBaGel 1/8/16 的 45 个 scalars 与已封存论文转录完全一致；
released、D2-V/X/Y/Z 的 75 个 Table-5-aligned scalars 与 D2-AA compact 完全一致；历史
author-code CM e200 的 12 个可用 Table-5 native scalars 与原 official-438 aggregate 完全一致。
历史行的 Rprec/FID/FPS 均保持 null，并记录未导出 CHOIS、没有独立 batch-1 timing 的原因；
full-438 `321.6566 FPS` 只保留为 descriptive throughput。

独立结果为
`experiments/results/p1_hoi_phase1b_d2aa_integrated_table_s42_20260724.json`，SHA-256
由 completion registry 绑定。原 compact SHA-256 复核仍为
`d791c04bf1a896f4230a55e77518368cf4c5cb5c691c6ce98de65c18a87914d8`，未修改。分类为
`integrated-table-reporting-pass-nonselection-stop`：没有运行 GPU/evaluator、训练、distillation
或 checkpoint selection；历史复现没有被提升为 reportable/selectable HOIPrior，D2-V/X/Y/Z
结论与正式 HOIPrior 缺失状态不变。

#### 2026-07-25 Phase 1B D2-AB predicted-support no-slip objective 预注册

D2-V 已证明固定的 232-D representation、architecture、conditions、balanced losses 与
61,440,000-window budget 能从随机初始化学到强 diffusion HOIPrior；D2-W 排除了通过中间
checkpoint selection 解决 foot sliding。D2-X 将足部 x/z temporal gradient 路由到
predicted root/rotations→FK 后，official foot sliding 点估计改善但 paired 95% CI 跨 0。
D2-Y 的固定 1,024 倍 amplification 显著降低 internal routed residual，却没有显著改善
official foot sliding，并破坏 end-object/contact protection；D2-Z 的 immutable-GT binary
near-ground gate 保留 D2-Y 约 78--86% gradient norm、cosine 约 0.95--0.99，仍未改善
official foot sliding。D2-Y/Z 的 routed-foot 与 FK gradient cosine 在 `t=0` 为正、在
`t=249/499` 为负，支持“当前 teacher-forced FK temporal target 在 noisy/mid timesteps
与 reconstruction/FK/object capability 冲突”，但尚未检验由 predicted state 本身定义的
物理支撑与 no-slip target。

用户在严格只读审计后明确批准执行本 subphase。registry 全量扫描确认 D2-AB 是下一个未使用的
Phase 1B identifier。本 campaign 最多允许两次新的完整 from-random training；D2-AB 占用第一
次且当前只授权这一次。第二次只可能是下文严格条件触发、另行 dated amendment 和用户明确授权的
fallback；不得自动启动。

1. **唯一 manipulated factor 与运行身份。** 正式训练 run id 固定为
   `p1-hoi-d2ab-predicted-support-no-slip-s42-20260725`、subphase `1B-D2-AB0`、seed 42。
   唯一改变是用下述 predicted-state/contact-aware differentiable no-slip residual 替换
   D2-X velocity tensor 中 joints `[7,8,10,11]` 的 8 个 x/z routed residual slots。其余
   79 slots、87-slot global mean、global velocity weight `0.1`、reconstruction/FK/object
   surface/terminal-goal losses、data、conditions、model、diffusion 与 production sampler
   全部不变。该机制不调用 official floor helper、DBSCAN、official ankle/toe thresholds、
   official foot-sliding reduction 或 test-set statistics，因此不是 evaluator-threshold trick。
2. **固定 from-random training contract。** 保持 232-D、16 frames、2 history frames、
   500-step clean-x0 diffusion、512-wide/16-head/8-layer scene-free HOIPrior、固定
   seed-42 split、4×RTX 3090 `infbagel-4gpu/node01`、per-GPU batch 512、effective batch
   2,048、accumulation 1、61,440,000 processed windows、30,000 optimizer updates、FP32
   Adam betas `(0.9,0.999)`、LR `1e-4`、无 warmup/scheduler/weight decay/gradient clipping/
   AMP/EMA。loss weights 固定为 FK `0.3569973401779424`、object surface
   `0.4772322188400037`、velocity `0.1`、terminal object goal `1.0`。released、author、
   D2-V/X/Y/Z、consistency 或其他 prior checkpoint 均不得加载；`init_checkpoint`、
   `weight_init_checkpoint` 和首次正式启动的 `resume_checkpoint` 必须为空。训练必须由
   `tools/experiment.py start` 在 clean worker exact committed Git object 上创建 manifest。
3. **train-only support metadata。** 对固定 train split 的 4,088 条 raw 30-Hz immutable
   aligned sequences，令
   \[
   f_s=Q_{0.05}^{linear}\{y^{GT}_{s,t,10},y^{GT}_{s,t,11}\}.
   \]
   pooled strictly-positive clearance 是所有 train sequences、四个 joints
   `[7,8,10,11]` 的 `y-f_s>0` 值；其固定 median 为
   \[
   \ell=0.03925712490454316\ {\rm m}.
   \]
   train floor min/median/max 必须为
   `-0.004783304338343441 / 0.0353932767175138 / 0.06221588589251041 m`。
   metadata 必须绑定 split、`human_joints_aligned.npy`、`start_idx.npy`、`end_idx.npy`、
   `norm.npy` hashes、ordered train sequence indices、per-sequence floors、positive count、
   quantiles 和算法；不得读取 official test。internal-validation loss 可对其自身 immutable
   sequence 仅用相同公式即时计算 floor，但 `ell/kappa` 只能来自 train metadata。
4. **公式级 intervention。** predicted foot positions 来自 model predicted clean x0：
   denormalize root/direct positions，decode rotations，经同一 24-joint FK 得到
   \(p^\theta\)；GT foot positions \(p^{GT}\) 来自 denormalized clean direct-position channels。
   左/右 foot pairs 分别为 \(J_L=\{7,10\}\)、\(J_R=\{8,11\}\)。对作为 residual previous
   state 的位置定义
   \[
   d_{t,q}=-\ell\log\left({1\over |J_q|}
   \sum_{j\in J_q}\exp(-(p^\theta_{t,j,y}-f_s)/\ell)\right),\qquad
   s_{t,q}=\sigma(-d_{t,q}/\ell).
   \]
   第一个 future residual 的 previous state 使用 immutable GT history frame；后续 previous
   state 使用前一 predicted FK state。固定 sampled-frame interval \(\Delta t=0.1s\)：
   \[
   v^\theta_{t,j}={\Pi_{xz}(p^\theta_{t,j}-\bar p^\theta_{t-1,j})\over0.1},
   \qquad
   v^{GT}_{t,j}={\Pi_{xz}(p^{GT}_{t,j}-p^{GT}_{t-1,j})\over0.1},
   \]
   \[
   r^{AB}_{t,j}=v^\theta_{t,j}-
   (1-s_{t-1,q(j)})v^{GT}_{t,j}.
   \]
   OMOMO position ranges are locked as
   `Rx=6.658331632614136 m`、`Rz=6.975271224975586 m`，固定 scalar
   \(\kappa=0.029363068377844033\,s/m\)。8 个 routed slots 使用
   \(L^{AB}_{foot}=\operatorname{mean}\|\kappa r^{AB}\|_2^2\) 对应 element errors；
   其余 79 element errors沿用 D2-X，最后仍对全部 87 slots 和 14 residual frames作一次
   unchanged mean。没有 multiplier、threshold sweep、contact-label gate、SNR weighting、
   gradient projection、rollout exposure 或 sampler intervention。
5. **cheap pre-training diagnostic 与 fail-closed tests。** 在任何 GPU workload 前，CPU-only
   metadata builder 必须复现上述 4,088 sequence/count/floor/clearance/range/constants，且
   synthetic 与真实 train-batch tests 必须证明：first-future previous state 为 immutable GT；
   later previous state 为 predicted FK；support 对 predicted foot height 可微、floor/GT
   stop-gradient；只有 8 个 routed x/z slots 被替换；direct foot x/z 不获得该分量梯度而
   root/rotation 获得有限非零梯度；`s→0` 时 routed residual 退化到 scaled physical
   D2-X target、`s→1` 时退化到 zero-slip target；无 checkpoint load。任一 metadata/hash/
   shape/gradient/contract 失败即停止，不启动 GPU。
6. **registered GPU smoke。** run id
   `p1-hoi-d2ab-gpu-smoke-s42-20260725`，只在 clean exact worker commit 的 `cuda:0`
   上读取固定 real-data batch 8，覆盖 timesteps `0/249/499`，执行一次 random-initialized
   forward/backward。不得创建 optimizer、optimizer update 或 checkpoint；必须记录全部
   losses、support occupancy/quantiles、关键 root/rotation/model gradients、CUDA-synchronized
   peak memory/headroom、initial model hash 和 0 checkpoint load/write。四卡必须可见且无
   compute contention。smoke 任一非有限/零关键 gradient、support collapse、hash/host contract
   失败即保留 artifacts 并停止正式训练。
7. **训练稳定性与停止轮询。** 正式 detached run 必须先通过 resolved-config、same-context
   machine preflight、finite losses/required gradients 的初始稳定区间、注册显存 headroom 和
   至少一个可实际 resume 的 checkpoint。通过后报告 measured throughput、ETA 和 checkpoint
   hash，停止连续轮询并等待用户通知；tunnel interruption 不允许重启、复用 run id 或覆盖。
   正式预算无论结果如何不得延长、选择中间 checkpoint 或 resume D2-V/X/Y/Z。
8. **fixed internal diagnostic。** 训练完成后 run id
   `p1-hoi-d2ab-predicted-support-no-slip-internal-s42-20260725`，使用 sealed
   32-sequence/96-window internal cohort、相同 clean windows/noise/dropout、final-online
   D2-X control 与 D2-AB target，在 timesteps `249`、`499` 报告每 sequence
   \[
   M=\operatorname{mean}_{t,j}s_{t-1,q(j)}\|v^\theta_{t,j}\|_2^2
   \]
   及 support mass/height/no-slip residual。每个 timestep 对 `D2-X minus D2-AB M` 做
   seed-42、10,000-replicate sequence bootstrap；internal mechanism gate 要求两个 CI
   下界均 `>0`。support sanity 还要求 target/control mean support-mass ratio 的 paired
   95% CI 完全落在 `[0.80,1.20]`，避免通过抬脚关闭 support。该 diagnostic 不选择 checkpoint。
9. **固定 native evaluation 与 registered uncertainty。** run id
   `p1-hoi-d2ab-native-eval-s42-20260725`，只加载 D2-AB fixed final-online checkpoint，
   执行一次 official 438 sequences×3 windows、500-step unguided production diffusion；
   CFG/dynamic perception/guidance/scene/consistency 均关闭。primary paired control 是封存
   D2-X final-online aggregate/per-sequence records，checkpoint/aggregate/per-sequence
   SHA-256 分别为
   `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51` /
   `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
   `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`，
   不重新生成 control。所有 paired metrics 使用 sequence 为单位、seed 42、10,000 bootstrap；
   penetration 固定使用 D2-X evaluator 已封存的同一 181-sequence finite mask。FID、
   Matching、R-Precision@1/2/3、Diversity 和 timing 若 evaluator 正常产生必须原样保留和
   报告，但 FID/R-Precision 当前不参与选择，不得删除或反向调 evaluator。
10. **gates 与 classification。** official mechanism gate 要求 `D2-X minus D2-AB`
    foot-sliding paired-difference 95% CI 下界 `>0`。protection gate 要求 D2-AB/D2-X 的
    MPJPE、end-object、Txy、object translation、hand penetration、human penetration paired
    ratio CI 上界均 `<=1.10`，contact-F1 difference CI 下界 `>=-0.02`。absolute released
    baseline gate 要求 MPJPE/end-object/Txy/object translation/FS ratios
    `<=1.30/2.00/1.50/1.50/1.10` 且 contact F1 `>=0.60`。internal、official foot、
    protection 和 absolute gates 全通过才分类
    `predicted-support-no-slip-positive-candidate-stop`；internal 失败为
    `predicted-support-no-slip-optimization-negative-stop`；internal 通过但 official foot
    失败为 `predicted-support-no-slip-transfer-negative-stop`；official foot 通过但 protection
    失败为 `predicted-support-no-slip-conflict-negative-stop`；mechanism/protection 通过但
    absolute gate 失败为 `predicted-support-no-slip-positive-but-not-effective-stop`；
    lifecycle/hash/support contract 失败为
    `predicted-support-no-slip-contract-failure-stop`。无论正负均停止，不选择 checkpoint、
    不延长预算、不自动授权 consistency。
11. **artifact contract。** 保留并双端 hash 验证：dated plan/registry、support metadata 与
    builder resolved record、logical implementation commit、authority CPU/test logs、worker
    pull/preflight/resolved configs、smoke manifest/log/metrics、formal manifest/train logs/
    training state、全部 checkpoints 与 per-rank RNG、validation/training metrics、resume
    demonstration、internal diagnostic manifest/per-sequence bootstrap、native evaluation
    manifest/aggregate/per-sequence/resolved config/logs、optional FID/R@/timing、run-local
    registry、dependency/hardware/data/evaluator hashes、完整 recovered artifact tree 和所有
    operational/scientific failures。大 artifacts 不进 Git；Git 只记录 code/config/tests、
    metadata、compact result、phase summary 与 hashes。
12. **最多一次条件性 fallback，当前不授权。** 只有 D2-AB internal gate 与 support sanity
    均通过、native transfer/protection 结果支持 late-timestep conflict、且 D2-AB final
    no-slip/protection gradient cosine 在 `t=249/499 <= -0.2`、`t=0 >= +0.2`，同时不存在
    support collapse 或 contract failure，才可提出 local objective-gradient projection：
    只投影 no-slip gradient，其他 objective gradients 不变，并必须再次从随机初始化。
    触发时须重新扫描 unused Phase 1B identifier、追加 dated plan/registry、取得用户新的明确
    授权；不得 resume D2-AB。本次批准不包含该第二次训练、consistency、HSIPrior 或 Mixer。

#### 2026-07-25 D2-AB CPU smoke-contract amendment（未启动 workload）

在 GPU/worker workload 之前的 authority CPU 审计中，精确复现 seed-42、batch-8、随机
初始化的 D2-AB smoke forward 得到 support mean `0.0266431123`，但 support 分布并未
collapse（约 `7.6%` entries `>0.05`，且同时存在低/高 support）。原 smoke 工具把
`mean >= 0.05` 当作 collapse gate，会错误拒绝该合法随机初始化；这是 operational
gate 实现缺陷，不是 scientific negative，也未消耗 smoke run id、创建 manifest、加载
checkpoint 或启动 GPU。

修正仅将 smoke 的 support 检查改为方向中性的非退化 contract：所有值有限，分布宽度
大于 `1e-3`，至少 `5%` entries `>0.05` 且至少 `5%` entries `<0.95`；同时记录
`q05/median/q95` 与两侧 occupancy。训练公式、数据、预算、native/internal gates、
run id 和 fallback 授权均不变。修正后仍必须重新运行专项/全量 CPU tests，再在 clean
exact commit 上执行原注册的 `p1-hoi-d2ab-gpu-smoke-s42-20260725`。

同一 CPU import 审计还发现 internal diagnostic 错把不存在的
`priors.optimizer_reset.paired_ratio` 作为 support-sanity bootstrap；在任何 checkpoint
load/evaluation 前已改为同模块既有的 `paired_mean_ratio`（ratio of paired-resampled
means，seed 42、10,000 replicates），与预注册统计量一致。该修正不改变 cohort、metric、
gate 或训练，并须由 CLI import test 覆盖。进一步的 CPU schema test 固定 96 windows
按每窗口全 residual/joint 维均值后再按三窗 sequence 聚合，并验证 aggregate 输入使用
`{"timesteps": ...}` 包装，避免把 per-window tensor 当作 per-sequence scalar。
native-evaluation wrapper 的 CPU resolver test 还必须在 monkey-patch shared D2-X
evaluator 后成功生成 D2-AB resolved record；wrapper 必须调用预先保存的 shared resolver，
不得递归调用已被替换的自身函数。该项仍是 pre-workload operational contract，不改变
任何 evaluator formula、metric 或 gate。

#### 2026-07-25 D2-AB continuous detached training lifecycle amendment（用户确认）

此前 D2-AB 首段使用 `pause_after_windows=3072000` 的目的仅是形成并审计一个可恢复
checkpoint；它不是科学停止点、不是第二次训练预算，也不是要求 Codex 等待后再决定是否
继续。该 checkpoint、paused state、resume evidence 和原始日志均为不可变 operational
证据，不能覆盖或删除。

本 amendment 将 D2-AB 的执行约束改为连续 detached training：

1. **同一 lineage 继续。** 用户已明确确认从
   `p1-hoi-d2ab-predicted-support-no-slip-s42-20260725` 的
   `3072000`-window checkpoint 继续。resume 必须使用同一 Git object
   `3fce4767111f7b4c01b5c2af252f6c3ef362cf43`、同一 run id、同一 seed 42、同一
   232-D representation、architecture、conditions、loss、optimizer、effective batch
   和固定 `61440000` processed-window budget；这不是新的 from-random training，也不占用
   第二机制预算。
2. **唯一 continuation config。** continuation 的 `resume_checkpoint` 必须绑定
   `ceb73ebc3a72d6290fc63e2546533c1565912b905a980381d406ea71b39a2ecc`，且
   `pause_after_windows: null`、`max_processed_windows: 61440000`。不得加载 released、
   author、D2-V/X/Y/Z、其他 prior、EMA 或任何不同 run 的 state；不得修改预算、改变
   checkpoint cadence，或选择中间 checkpoint。
3. **manifest 与 artifact 不覆盖。** 原始 `tools/experiment.py start` manifest、
   初始 resolved config、首段 train log、paused state 和 checkpoint 保持原样。因为 run id
   已经被使用，continuation 不创建第二个 manifest，也不复用/覆盖文件；必须新增
   `resolved_config_resume.yaml`、resume command、same-context resume preflight、
   `resume_initial_stability` 和 `resume` log/exit artifacts。完整训练结束后才对原 manifest
   执行一次 `finish/register`，同时绑定初始与 continuation 两套配置和完整 artifact tree。
4. **Codex 监测边界。** resume 启动后，Codex 只检查一个初始稳定区间：resolved config
   无 unresolved interpolation、四卡无 CUDA compute contention、进程持续运行、trainer
   的 finite loss/required-gradient fail-closed 检查未触发，以及注册的显存余量保持。形成
   稳定快照后立即报告已测 resume 吞吐、总训练耗时估计和 ETA，并停止主动轮询；训练必须在
   worker-owned persistent session 中继续运行。停止轮询不等于停止、暂停、杀进程或创建
   新 checkpoint。tunnel/access interruption 也不授权重启、复用 run id 或覆盖结果。
5. **完成后的固定动作。** 训练仍须跑满 `61440000` windows/`30000` updates；无论中途
   观测到的 validation/loss 如何，均不得提前选择 checkpoint、延长预算或启动
 consistency。只有完整训练结束并完成 artifact recovery 后，才运行已注册的一次 internal
 diagnostic 和一次 native evaluation；本 amendment 不授权 fallback、consistency、
 HSIPrior 或 Mixer，也不改变任何 scientific gate、uncertainty 或 FID/R-Precision
 保留规则。

#### 2026-07-25 D2-AB resume provenance guard amendment（operational）

在 continuation workload 启动前发现：正式 trainer 原本要求
`checkpoint.git_commit == current HEAD`。首段 checkpoint 绑定
`3fce4767111f7b4c01b5c2af252f6c3ef362cf43`，而本 lifecycle amendment 的治理 commit
已经是不同 object；若直接 resume，trainer 会在 GPU DDP 初始化后 fail closed。这是可确定的
provenance/lifecycle 不兼容，不是模型、loss、数据或 evaluator defect；该次未启动
resume GPU workload，旧 checkpoint 未被改写。

允许的修复仅是一个 hash-bound continuation provenance guard：resume config 必须显式绑定
checkpoint source commit、worker target HEAD、两者 `git diff --binary` SHA-256 和固定的
changed-file allowlist（仅本次治理/配置/guard/test 文件）；trainer 仍拒绝所有未显式绑定或
包含其他 source/config/model 变化的 commit transition。exact-commit resume 及所有新训练
仍保持原有 fail-closed 规则。该 guard 不改变 forward、loss、optimizer、sampling、预算或
任何 scientific gate；其 source/target/diff hash 必须进入 continuation resolved config、
 resume contract artifact 和最终 manifest。

#### 2026-07-26 D2-AB completion manifest transition amendment（operational）

完整续训在 `0db60d82e454dd722320832e9f7b3f228a90ef72` 正常结束，但原始
`tools/experiment.py start` manifest 绑定的是首段 workload commit
`3fce4767111f7b4c01b5c2af252f6c3ef362cf43`。因此普通 `experiment.py finish` 的
exact-HEAD guard 会在完成阶段拒绝一个已经通过 resume provenance guard、且 metrics 明确记录
source/target/diff 的合法同一 lineage。这是确定的 lifecycle implementation defect，不是
训练或 scientific defect；训练已完成、checkpoint 未修改、evaluation 尚未启动。

允许的最小修复仅扩展 `tools/experiment.py finish/register` 的 manifest provenance contract：
在显式提供原始 manifest commit、当前 workload commit、`git diff --binary` SHA-256 和固定
changed-file allowlist 时，验证这些值、metrics 的 `resume_commit_provenance` 与最终
`metrics.git_commit`，并把 transition 作为不可变 manifest/registry 字段记录；未显式绑定或
路径/hash 不匹配仍 fail closed。不得放宽 dirty-worktree、run-id、artifact overwrite 或
普通 exact-HEAD checks，也不改变训练、评估、gate、uncertainty、checkpoint selection、
consistency 或 fallback 规则。此修复不启动任何 GPU workload，随后仍只执行已注册的 D2-AB
finish/recovery、internal diagnostic 和 native evaluation。

#### 2026-07-26 Phase 1B D2-AB completion / optimization-negative stop

D2-AB 已完成全部预注册 lifecycle，并严格分类为
`predicted-support-no-slip-optimization-negative-stop`。本 completion 只封存已完成的 smoke、
training、internal diagnostic 和 native evaluation；没有启动第二次完整训练、conditional
fallback、checkpoint selection、consistency、HSIPrior 或 Mixer。

1. **GPU smoke 与固定训练 contract。** `p1-hoi-d2ab-gpu-smoke-s42-20260725` 在
   `cuda:0` 覆盖 timesteps `0/249/499`，random-initialized batch 8 的 loss 与关键
   root/rotation/model gradients 均 finite/nonzero；support 分布通过方向中性的
   non-collapse contract。smoke 没有创建 optimizer、update 或 checkpoint，也没有加载任何
   checkpoint。随后 formal run
   `p1-hoi-d2ab-predicted-support-no-slip-s42-20260725` 在
   `infbagel-4gpu/node01` 完成固定的 `61,440,000` processed windows /
   `30,000` updates、effective batch `2,048`、seed 42、232-D、4×RTX 3090、
   final-online contract。初始 model-state SHA-256 为
   `ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e`，
   final checkpoint SHA-256 为
   `3eb68cc55cae15fd4bd3ff5279131ffd9a35ba0399e8e90557e89cb301631d8e`。
   训练总 wall time `18,382.995 s`，吞吐 `3,342.219 windows/s`，每 rank 最小显存余量
   `21,261,123,584 bytes`；final validation total 为 `0.0488587866`。全部 20 个
   cadence checkpoints 与 80 个 RNG sidecars 已保留。released、author、D2-V/X/Y/Z、
   prior 或 EMA 均未作为初始化来源。
2. **连续 resume provenance。** 首个 `3,072,000`-window checkpoint
   `ceb73ebc3a72d6290fc63e2546533c1565912b905a980381d406ea71b39a2ecc`
   只作为 resumability evidence；训练在 worker-owned persistent session 中持续到完整预算，
   没有被 Codex 停止、重启或换 run id。source commit
   `3fce4767111f7b4c01b5c2af252f6c3ef362cf43` 到 workload target
   `0db60d82e454dd722320832e9f7b3f228a90ef72` 的 binary diff SHA-256
   `9c777e1058ddc78ffdf2455141870e3d08eee37b51621bae4ffa45b32448ec86`
   已进入 continuation config、metrics、manifest transition 和 run-local registry。
3. **internal mechanism gate 失败。**
   `p1-hoi-d2ab-predicted-support-no-slip-internal-s42-20260725` 使用 sealed
   32-sequence/96-window internal cohort、D2-X/D2-AB final-online checkpoints 和
   timesteps `249/499`。对主要量 `D2-X minus D2-AB supported velocity`，sequence bootstrap
   结果分别为：
   - `t=249`：mean `-0.0013696933`，95% CI
     `[-0.0021679224,-0.0006211924]`；
   - `t=499`：mean `-0.0017119791`，95% CI
     `[-0.0029663018,-0.0008443758]`。

   两个 CI 都在 0 以下，说明 D2-AB 的 predicted-support 区域水平足速反而显著更高。
   target/control support-mass ratio CI 分别为
   `[1.005099,1.012231]`、`[1.004567,1.013073]`，完全位于预注册
   `[0.80,1.20]`，因此 support sanity 通过，负结果不能解释为通过抬脚或关闭 support 获得。
   internal lifecycle contract/finite checks 均通过，但 mechanism gate 失败；该 diagnostic
   没有 optimizer、update、checkpoint write/selection 或 official-test 使用。
4. **固定 native evaluation。** `p1-hoi-d2ab-native-eval-s42-20260725` 只加载上述
   D2-AB fixed final-online checkpoint，执行 official 438 sequences×3 windows、
   500-step unguided diffusion；sealed D2-X control records 原样复用。D2-AB point estimates：

   | Te | Txy | FS | Cprec | Crec | Cf1 | C% | Pbody | MPJPE | Troot | Tobj | Oobj |
   |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
   | 3.6840 | 4.0892 | 0.3661 | 0.7896 | 0.5953 | 0.6383 | 0.4775 | 3.7714 | 12.0639 | 8.1101 | 15.9248 | 1.0244 |

   primary `D2-X minus D2-AB` foot-sliding paired difference mean 为
   `-0.0030536116`，95% CI `[-0.0175441732,0.0110958477]`，official improvement
   gate 失败。D2-AB minus D2-X contact-F1 difference CI
   `[-0.0046342095,0.0066400986]` 通过；MPJPE、end-object、Txy、object translation、
   hand penetration 与 human penetration 的 target/control ratio CI 上界全部 `<=1.10`，
   因而 protection gate 全通过。全部 released-baseline absolute diffusion checks 也通过，
   其中 FS ratio `1.0980924 <= 1.10`。固定 181/438 penetration finite-mask contract
   完全匹配。
5. **timing 与 optional metrics。** native runtime `392.837 s`；generation
   `62.556 s`、end-to-end generation `386.303 s`、55,188 frames、CUDA-synchronized
   descriptive throughput `882.212 FPS`。本 evaluator 没有生成 FID 或
   R-Precision（`fid_rprecision_used=false`）；该缺失已原样保留，不删除字段、不以
   D2-AA 或其他 checkpoint 的值代填，也不参与选择。
6. **科学结论。** D2-AB 的 support sanity、protection、absolute、provenance、normalization
   和 artifact contracts 全通过，但 internal optimization direction 与 official FS gate
   均失败。故 predicted-state/contact-aware no-slip objective 在当前固定 objective mixture
   下不是充分机制；这不是 evaluator trick 的失败，也不是确定 training loss/model/math/
   official-evaluator implementation defect。因为预注册 fallback 明确要求 internal
   mechanism gate 先通过，local gradient-projection fallback 的触发条件不成立，剩余第二次
   full-training budget 不得使用。
7. **artifact seal。** compact result 为
   `experiments/results/p1_hoi_phase1b_d2ab_predicted_support_no_slip_s42_20260726.json`。
   smoke tree SHA-256
   `654733afafcde8ffed20f41d0812e46cd4a62d91f3f8ccedb4d9d1c837823bd6`
   （15 files / 91,202 bytes）；training tree
   `e357b0c6e8ed3fdd2d8a0ed8a1ca1ac8dbff461892066bc4bd151928e50063ab`
   （149 files / 7,127,317,639 bytes）；internal tree
   `605cfd89381ceb9eb5e35adc4d274feccd102b06f26fbde6570499d6db9cdd55`
   （18 files / 258,193 bytes）；native-eval tree
   `d56bff74b8982a6f63efd62e784c2759233d9f5591222dbe9deb9922e17b5d42`
   （22 files / 379,944 bytes）。worker/authority tree hashes 均一致。

D2-AB 到此停止且 checkpoint 不可选择。任何未来第二机制必须重新执行 `date`、全量扫描下一
unused Phase 1B identifier、添加新的 dated plan/registry hypothesis，并获得用户新的明确授权；
不得 resume D2-AB，不得自动启动 consistency、HSIPrior 或 Mixer。

#### 2026-07-26 Phase 1B D2-AC part-aware local-object cross-attention interaction adapter 预注册（plan-only）

D2-AB 已按预注册分类为 `predicted-support-no-slip-optimization-negative-stop`：训练稳定且
protection 通过，但 predicted-support 区域足速在 `t=249/499` 均显著向错误方向移动，
official foot sliding 也未改善。因此 D2-AB 的 objective fallback 保持关闭，checkpoint 不可
resume、不可选择、不可初始化任何后续 prior。全量扫描 plan、registry、code、tests 与 tools
确认 D2-AA、D2-AB 已占用而 D2-AC/`d2ac` 尚未出现；D2-AC 是下一个 unused Phase 1B
identifier。

当前最强且最安全的 diffusion control 固定为 D2-X final-online。相对 released local baseline，
D2-X 的 MPJPE、Troot、Tobj、Oobj 已分别只差约 `+0.44%/-0.47%/+1.71%/+1.05%`，但
end-object、FS、contact recall、contact F1、contact coverage 与 Pbody 的缺口仍分别约为
`+23.15%/+8.88%/-18.30%/-12.36%/-20.35%/+49.43%`。本地代码审计同时确认当前
HOIPrior 把完整 `1024×3` BPS flatten 后压成单个 global condition token；16 个 motion
tokens 虽能经 self-attention 间接访问物体条件，但没有显式的 left-hand/right-hand/object-motion
到局部 object geometry 的关系通路。这不是确定 implementation defect，而是与现有 contact/
penetration 缺口一致、可由实验否证的 capacity/routing 假设。

用户已确认采用小型 part-aware、object-aware cross-attention interaction adapter，并放宽原
“最后一次完整训练”上限；但该放宽不构成开放式 architecture search。本 amendment 只锁定一个
adapter 机制、一次 primary full training，以及最多一次严格条件触发的同机制 longer-budget
training。当前 session 只写 plan/registry，不修改 source/config/tests，不创建 workload，不启动
GPU、训练、evaluation、checkpoint selection 或 consistency。实际 lifecycle id 必须在新的
implementation session 首先执行真实 `date` 后，以 dated implementation amendment 绑定，不能
预占未来日期。

1. **假设与可区分的竞争解释。** Primary hypothesis 是：当前 contact recall/coverage、
   hand-object geometry 与 penetration 缺口的一部分来自单一 global-BPS token 无法为不同身体
   role 提供稳定的局部物体关系；在主干中段加入小型 part-to-local-object cross-attention，应在
   不改变 loss、representation 或 sampler 的条件下改善 contact F1/recall 与物理 hand-object
   alignment。竞争解释至少包括：
   - `unused-capacity`：adapter gate 或内部路径没有被优化，full 与 gate-ablated 输出无差异；
   - `unstructured-capacity`：adapter 有贡献，但打乱局部几何对应后效果不变，增益只来自额外
     参数或一般非线性；
   - `objective/distribution-limited`：adapter 使用了正确局部几何，但固定训练 objective 或
     train/rollout gap 仍使 native contact/penetration 不改善。
   Internal causal ablation、local-correspondence permutation 和 native transfer gate 必须分别
   区分这三类解释，attention map 本身不得替代 causal gate。
2. **唯一 manipulated factor 与明确排除项。** D2-AC0 相对 D2-X 只加入下述一个
   interaction adapter。保持现有 global BPS token、D2-X FK-foot temporal routing、全部 loss
   tensor/reduction/weight、optimizer、data、conditions、diffusion、production sampler 和
   evaluator 不变；D2-AB predicted-support objective 必须关闭。不加入 contact/no-slip/
   penetration 新 loss，不改变 contact label supervision，不做 SNR/timestep weighting、
   gradient projection、predicted-history exposure、CFG/guidance、future-GT conditioning、
   threshold/multiplier sweep、token-count sweep、adapter-depth/placement sweep 或中间
   checkpoint selection。adapter 不调用 official FS floor/near-ground helper、contact threshold、
   penetration mask 或 test statistics，因此不是 evaluator trick。
3. **固定 local-object tokenization。** 只读取现有 `code/bps.pt` 的 immutable BPS basis，
   file SHA-256 固定为
   `fdff7204b4697e105457cb7e39267b9555bc0d8d854dbc92cd67e2d8c3e77042`。
   对 basis 坐标执行 deterministic lexicographic-first、farthest-point sampling，距离相等时按
   最小原始 index 决定，固定 16 个 centers：
   `[328,903,503,817,474,1023,382,864,640,431,445,960,547,829,545,756]`。
   每个 basis point 分配给最近 center，tie 按 center 顺序决定；cluster sizes 固定为
   `[39,40,57,61,65,68,70,134,77,64,59,79,43,46,84,38]`，assignment canonical
   SHA-256 固定为
   `b62f91f4eb6c4bf2a9211f0187cd1eb97c25394ee45de155f336079fddeecd`。
   clustering 只依赖固定 basis，不读取 train/validation/test motion、contact 或 evaluator。
   令 \(b_i\in\mathbb R^3\) 为 basis、\(d_i\in\mathbb R^3\) 为当前既有 BPS delta，
   对 cluster \(C_k\) 构造固定 10-D feature
   \[
   u_k=\left[
   {1\over |C_k|}\sum_{i\in C_k}b_i,\;
   {1\over |C_k|}\sum_{i\in C_k}d_i,\;
   \sqrt{{1\over |C_k|}\sum_{i\in C_k}d_i\odot d_i},\;
   {1\over |C_k|}\sum_{i\in C_k}\|d_i\|_2
   \right].
   \]
   不做 train-stat normalization、learned clustering、object-category embedding 或额外 mesh/
   point-cloud encoder。
4. **公式级 adapter。** 保持 4 个 condition tokens 和 16 个 motion tokens 的原顺序，先通过
   原 8-layer Transformer 的前 4 层；只取 contextualized motion token
   \(H_t\in\mathbb R^{512}\)。local object tokens 与三个 role queries 固定为
   \[
   O_k=\operatorname{LN}_o(E_o(u_k)+e_k^{obj}),\qquad
   E_o:10\rightarrow128\rightarrow128,
   \]
   \[
   Q_{t,p}=\operatorname{LN}_q(W_qH_t+e_p^{part}),\qquad
   p\in\{\mathrm{left\ hand},\mathrm{right\ hand},\mathrm{object\ motion}\},
   \]
   \[
   A_{t,p}=\operatorname{MHA}_{d=128,h=4,\mathrm{dropout}=0}
   (Q_{t,p},O,O),
   \]
   \[
   R_t=W_r[A_{t,L};A_{t,R};A_{t,O}],\qquad
   H'_t=H_t+\tanh(\alpha)R_t.
   \]
   \(W_r:384\rightarrow512\)，单个 scalar \(\alpha\) 初始严格为 `0`；这是固定 ReZero
   identity gate，不是 checkpoint-derived prior。所有 Linear/MHA/embedding matrices 与原主干
   一起由 seed 42 从随机初始化，released、author、D2-V/X/Y/Z/AB、prior、EMA 或
   consistency weights 均不得加载。adapter 写回后，完整 token sequence 继续通过原第 5--8
   层；输出仍为 `[B,16,232]`。512-wide locked model 的当前参数量为 `29,673,448`，
   adapter 固定增加 `349,697`，总计 `30,023,145`，增量约 `1.1785%`，不得通过额外 hidden
   layer、第二 adapter 或 enlarged token set 超出 `1.25%` 预注册上限。
5. **BPS 与 production provenance。** Training 只使用当前 loader 已提供的单个
   `object_bps` condition；adapter 不读取 per-frame future BPS、future GT object pose、
   rest-mesh vertices 或 contact labels。Autoregressive production sampling 继续由
   `WindowStateCodec.recompute_bps()` 根据当前生成 object rotation 和 hash-verified rest
   geometry 重算下一窗口 BPS；不得回读 stored per-frame BPS。现有 global BPS token 在 full、
   ablated 与 permuted variants 中始终 byte-matched，确保 causal comparison 只改变新的 local
   relation path。
6. **Mixer/HSIPrior compatibility contract。** HSIPrior architecture、parameters、
   checkpoint schema 和 forward path 均不改变；adapter 只属于 HOIPrior。未来 Mixer 仍只接收
   两个专家在同一 timestep、同一 `WindowStateCodec` frame 下的 clean `[B,16,232]`
   prediction，不读取 adapter token、attention map 或 expert-specific latent。因此 Mixer 不要求
   两专家逐层同构，只要求 232-D field semantics、history、normalization、coordinate frame 和
   clean-x0 API 一致。CPU tests 必须继续证明 HOI/HSI parameter/storage independence 和 codec
   round-trip；若 adapter 迫使 Mixer 增加 HOI-specific coordinate/latent adapter，则 D2-AC
   contract 失败。
7. **cheap pre-training CPU diagnostic。** 在任何 GPU workload 前，authority Python 必须：
   - 复现 BPS file hash、16 centers、cluster sizes、assignment hash、feature shape/dtype/
     finiteness；
   - 证明 adapter-disabled/base model 在共享 trunk state、`eval()`、\(\alpha=0\) 时输出
     max-abs difference `<=1e-6`，并保持 `[B,16,232]` API；
   - 证明初始 backward 时 \(\alpha\) gradient finite/nonzero；在 test-only
     \(\tanh(\alpha)=0.1\) probe 下，object encoder、object/part embeddings、Q/K/V/out
     projections 与 writeback gradients 全部 finite/nonzero。正式 initialization 仍为
     \(\alpha=0\)，probe 不得保存或进入训练；
   - 证明 local feature permutation 在 gate nonzero 时改变 adapter contribution，而完整 token
     reorder 不被错误当作 locality test；固定 causal permutation 必须把
     cluster-delta statistics \(k\leftarrow(k+8)\bmod16\)，同时保留 \(\bar b_k\)、
     \(e_k^{obj}\) 和 global BPS token；
   - 验证 exact parameter count/增量上限、zero/constant/extreme BPS finiteness、role query
     separation、batch/device/dtype propagation、checkpoint variant rejection、HSIPrior
     independence、Mixer clean-output contract，以及 source/static path 中没有 future GT/
     stored per-frame BPS/evaluator threshold。
   任一 hash、parity、gradient、shape、parameter-count、provenance 或 interface check 失败即
   `interaction-adapter-contract-failure-stop`，不得启动 GPU。
8. **registered GPU smoke 与连续 detached lifecycle。** 新 session 以真实日期绑定唯一 smoke
   run id；只在 clean exact authority commit fast-forward 后的 `infbagel-4gpu/node01`
   `cuda:0`，对 fixed real-data batch 8、timesteps `0/249/499` 做 random-initialized
   forward/backward。不得创建 optimizer、update、checkpoint load/write；必须记录 losses、
   initial model hash、\(\alpha\) gradient、test-only nonzero-gate adapter gradients、
   CUDA-synchronized peak memory/headroom 和 4-GPU contention。按每 rank micro-batch 512，
   16 frames、3 roles、4 heads、16 local tokens 估算的 cross-attention score elements 为
   `1,572,864`；真实 smoke 仍必须满足注册 headroom，估算不能替代测量。
   Formal detached training 启动后必须持续运行到固定预算，不设置人为 pause node。通过
   resolved-config/preflight、finite loss/required gradients 初始稳定区间、显存余量和至少一个
   可实际 resume checkpoint 后，Codex 报告 measured throughput、总耗时估计与 ETA，并停止
   主动轮询；worker-owned persistent training 继续运行。停止轮询不等于停止、暂停、kill、
   restart 或 checkpoint selection。
9. **D2-AC0 primary full-training contract。** D2-AC0 占用本 amendment 的第一次完整训练：
   seed 42、固定 split
   `experiments/splits/omomo_hoi_train_validation_seed42.json`
   （SHA-256
   `019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e`）、
   232-D、16/2 frames、500-step clean-x0 diffusion、512-wide/16-head/8-layer trunk、
   4×RTX 3090、per-GPU batch 512、effective batch 2,048、accumulation 1、
   `61,440,000` processed windows / `983,040,000` frames / `30,000` updates。
   固定 FP32 Adam、LR `1e-4`、betas `(0.9,0.999)`、无 warmup/scheduler/weight decay/
   gradient clipping/AMP/EMA；loss weights 固定为 FK
   `0.3569973401779424`、object surface `0.4772322188400037`、velocity `0.1`、
   terminal object goal `1.0`，velocity tensor/reduction 完全沿用 D2-X。首次 start 的
   init/weight-init/resume checkpoint 必须为空；final-online fixed-budget checkpoint 是唯一
   target，不按 validation、internal 或 official test 选择中间 checkpoint。
10. **sealed interaction mechanism diagnostic。** 训练完成后，只加载 D2-AC0 fixed
    final-online checkpoint，在 D2-O 已封存的 internal-validation cohort
    `64 sequences × 3 windows`、phase offsets `(14,56,98)`、selection SHA-256
    `1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`
    上运行三条 500-step paired rollout：
    - `full`：正常 adapter；
    - `gate_ablated`：每一步强制 \(\tanh(\alpha)=0\)，其余 state/weights 不变；
    - `local_correspondence_permuted`：只执行第 7 项固定 \(k\leftarrow(k+8)\bmod16\)
      delta-stat permutation，global BPS 与其余 condition 不变。
    三条 path 的 initial latent、每一步 posterior noise、window/chunk ordering、conditions 与
    history restoration 必须 byte-matched；不得 optimizer、checkpoint write/selection、
    CFG、guidance 或 official-test use。按 sequence、seed 42、10,000 bootstrap 报告
    left/right/union semantic contact P/R/F1/coverage、direct-hand indices `24/26` 与
    FK-palm indices `22/23` 的 `2/5/7.5/10 cm` physical contact P/R/F1/coverage、
    contact run length、GT-contact-frame hand-object distance、penetration、MPJPE、
    object/pelvis goals、FS、learned gate 和 per-role attention entropy。
    Primary internal mechanism gate 使用 direct-hand union 5-cm physical contact F1：
    `full - gate_ablated` 与 `full - local_correspondence_permuted` 的 paired 95% CI
    下界都必须 `>0`；同时两组 `comparator - full` GT-contact-frame mean hand-object
    distance CI 下界都必须 `>0`。任一 ablation gate 失败说明 adapter 未被有效使用；只有
    gate-ablation 通过而 locality permutation 失败，则说明是 unstructured extra capacity，
    不能进入 positive classification。其他阈值/representation 与 attention map 只作完整
    诊断，不得替代 primary causal gate。
11. **固定 native evaluation 与 uncertainty。** Internal 完成后，无论正负都执行一次完整
    reporting evaluation：official 438 sequences×3 windows、500-step unguided production
    diffusion、D2-AC0 fixed final-online weights；CFG/dynamic perception/guidance/scene/
    consistency 均关闭。Locked paired control 是 D2-X final-online，checkpoint/aggregate/
    per-sequence SHA-256 分别为
    `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51` /
    `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
    `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`，
    原样复用且不重新生成。Released aggregate 文件 SHA-256 为
    `76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6`。
    所有 paired metrics 用 sequence unit、seed 42、10,000 bootstrap；penetration 继续使用
    sealed D2-X evaluator 的相同 181-sequence finite mask。FID、Matching、
    R-Precision@1/2/3、Diversity 与 timing 若 evaluator 正常产生必须原样保留和报告；
    FID/R-Precision 当前不参与 selection，不得删除、代填或反向调 evaluator。
12. **native transfer、protection、absolute gate 与 classification。**
    - Native transfer gate 要求 D2-AC0 minus D2-X 的 contact-F1 与 contact-recall paired
      95% CI 下界均 `>0`，且 contact-F1 point estimate 至少关闭 released--D2-X 缺口的
      `25%`：
      \[
      {C_{F1}^{AC}-C_{F1}^{X}\over C_{F1}^{released}-C_{F1}^{X}}\ge0.25,
      \]
      即按当前 sealed points 至少达到约 `0.6598838781`。
    - Protection gate 要求 D2-AC0/D2-X 的 end-object、Txy、FS、Pbody、hand penetration、
      MPJPE、Troot、Tobj、Oobj paired mean-ratio 95% CI 上界全部 `<=1.10`，且 contact
      precision difference CI 下界 `>=-0.02`。固定 penetration finite-mask contract 也必须
      通过。
    - Final effectiveness gate 仍使用 Phase 1B section-wide released-baseline 95% point
      gate：lower-is-better 的 end-object、Txy、FS、Pbody、MPJPE、Troot、Tobj、Oobj
      均 `<=baseline/0.95`，higher-is-better 的 contact P/R/F1 均
      `>=0.95×baseline`。Contact coverage 必报但不单独当 monotone selection metric；
      FID/R-Precision 当前只报告、不选择。
    Contract 失败分类 `interaction-adapter-contract-failure-stop`；full 对 gate ablation 失败为
    `interaction-adapter-unused-optimization-negative-stop`；gate ablation 通过但 locality
    permutation 失败为 `interaction-adapter-locality-negative-stop`；internal 通过但 native
    transfer 失败为 `interaction-adapter-transfer-negative-stop`；transfer 通过但 protection
    失败为 `interaction-adapter-conflict-negative-stop`；mechanism/transfer/protection 通过但
    released 95% gate 失败为
    `interaction-adapter-positive-but-not-effective-stop`；全部通过才为
    `interaction-adapter-positive-candidate-stop`。只有最后一类可把 fixed final-online
    checkpoint 标记为 selectable autonomous diffusion HOIPrior candidate；这不是中间
    checkpoint selection，也不自动授权 consistency。
13. **artifact contract。** 必须保留并在 worker/authority 双端 hash 验证：dated
    plan/registry 与 lifecycle amendment、BPS cluster metadata/builder resolved record、
    logical implementation commit、source/config/tests、authority CPU/parity/gradient logs、
    worker pull/preflight/resolved configs、GPU smoke manifest/log/metrics、formal training
    manifest/log/state、全部 cadence checkpoints 与 per-rank RNG、initial/final model hashes、
    validation/training metrics、resumability evidence、measured throughput/ETA、internal
    three-variant manifest/per-sequence/paired-noise/attention appendix、native manifest/
    aggregate/per-sequence/bootstrap/penetration-mask、optional FID/R@/timing、run-local
    registry、dependency/hardware/data/evaluator hashes、recovered artifact tree、compact
    result、`docs/phase_summaries/PHASE_1B_D2AC.md` 及所有 operational/scientific
    failures。大 artifacts 不进 Git，不覆盖结果、不复用 run id。
14. **最多一次同机制 conditional longer-budget training。** D2-AC1 只有在 D2-AC0 严格
    分类为 `interaction-adapter-positive-but-not-effective-stop` 时才 eligible：即 causal
    gate/locality、native contact transfer 与 protection 全通过，只因 released 95% magnitude
    未达标而停止。触发后仍须真实 `date`、新的非复用 run id、dated plan/registry binding 和
    用户再次明确确认；不得自动启动。D2-AC1 必须以相同 seed 42 和相同 adapter/trunk/loss/
    optimizer/data/evaluator 从随机初始化开始，不得 resume 或加载 D2-AC0/D2-X/D2-AB；
    唯一变化是预算固定为 `122,880,000` processed windows /
    `1,966,080,000` frames / `60,000` updates。D2-AC1 重跑同一 internal/native gates，
    D2-AC0 全部结果仍保留并完整报告。若 D2-AC0 属于 contract/unused/locality/transfer/
    conflict negative，或 D2-AC1 未通过全部 candidate gates，则本 amendment 到此停止；
    不允许继续增加预算、改变 token 数/width/layers/placement、换 role、加入新 loss、做
    parameter sweep 或 checkpoint selection。任何进一步方向须使用下一个 unused Phase 1B
    identifier 和新的用户授权。

本 plan-only amendment 不授权当前 session 的 implementation、GPU smoke、training、
evaluation、checkpoint selection、consistency、HSIPrior 或 Mixer。新的 implementation session
必须首先重新执行 `date`、path/branch/HEAD/status 核验，完整读取 `AGENTS.md`、本计划和
`docs/phase_summaries/PHASE_1B_D2AB.md`，再以真实 lifecycle identities 添加
implementation amendment；source change、CPU tests、worker publication 和 D2-AC0 workload
只能在该入口之后进行。

#### 2026-07-26 Phase 1B D2-AC0 implementation/lifecycle binding amendment

本 session 已在真实日期 `2026-07-26` 重新核验 authority path、`phase/01b-hoi`、
handoff HEAD `61a989adab2f3053230bfcd0ebb702601fcdaab2` 与 clean worktree，并获用户明确
授权连续完成 D2-AC0 的 implementation、CPU contract、worker publication、注册 GPU smoke、
一次完整 from-random training、固定 internal diagnostic、一次 native evaluation 与 artifact
recovery。本 amendment 只绑定已预注册机制和本日未使用 lifecycle identities，不改变 D2-AC0
的 architecture、representation、loss、optimizer、data、budget、gates 或 stop rules：

- implementation logical change：`p1-hoi-d2ac-interaction-adapter-implementation-s42-20260726`；
- authority CPU contract：`p1-hoi-d2ac-cpu-contract-s42-20260726`；
- registered GPU smoke：`p1-hoi-d2ac-gpu-smoke-s42-20260726`；
- formal training：`p1-hoi-d2ac-interaction-adapter-s42-20260726`；
- internal diagnostic：`p1-hoi-d2ac-interaction-adapter-internal-s42-20260726`；
- native evaluation：`p1-hoi-d2ac-native-eval-s42-20260726`。

Implementation must remain a single logical commit containing source, config, CPU tests, this dated
binding and its registry record. The worker may execute only that exact committed Git object. The
first-start init/weight-init/resume checkpoints remain empty; all D2-AC0 workloads use seed 42 and
the fixed final-online target. If the date changes before an unstarted lifecycle, that lifecycle must
receive a new identity-only dated amendment before its manifest/workload; an old identity is never
reused. D2-AC1 remains unauthorized and cannot be started automatically.

The implementation-session preflight also found a transcription defect in the plan-only assignment
hash, before any source or workload was created. The preregistration calculation serialized
`{"algorithm":"lexicographic-seed-farthest-point-16-v1","centers":...,"assignments":...}`
with sorted JSON keys and compact separators and produced the valid 64-hex SHA-256
`b62f91f4eb6c4bf2a9211f0187cd1eb97c25394ee45de155f33607959fddeecd`;
the plan-only text and registry line accidentally omitted the two hex characters `59` and therefore
recorded a 62-character value that cannot be a SHA-256. D2-AC0 binds the valid canonical 64-hex
value. This correction changes no center, assignment, cluster size, token feature, model parameter,
training/evaluation protocol, gate, or authorized scope; CPU validation must reproduce the canonical
payload and corrected hash exactly and fail closed otherwise.

#### 2026-07-26 Phase 1B D2-AC0 evaluator closure 与 GPU-smoke retry amendment

在 implementation tree 的 authority 全套 CPU gate 中，323 项测试首次运行有 321 项通过；
仅 `tests/test_hoi_d2t.py` 与 `tests/test_hoi_d2u.py` 的旧整文件冻结断言仍要求 D2-AC 之前的
`code/priors/models.py` SHA-256。失败只涉及 approved D2-AC architecture-variant extension
所在的 shared file；D2-AC exact shared-trunk、`eval()`、`alpha=0` parity 的 measured
max-abs difference 为 `0.0`，representation/data/loss/diffusion hashes 均未改变。CPU gate
closure 因此只允许把这两个 historical freeze 更新到 approved post-D2-AC model source，
同时保留 exact base-path parity test；不得借此接受任何旧 HOIPrior 行为变化。

本 amendment 也在任何 formal training 前封存剩余 evaluator implementation：

- internal runner 只加载 fixed final-online D2-AC0 checkpoint，并在 sealed D2-O
  `64×3` cohort 上运行 `full`、`gate_ablated`、`local_correspondence_permuted` 三路
  production 500-step rollout；
- primary direct-hand union 5-cm F1 与 GT-contact-frame distance 严格使用
  indices `24/26`。GT-contact-frame 定义为 fixed target direct-hand union distance
  `<5 cm`；无 GT-contact frame 的 sequence 不做零值代填，paired bootstrap 使用三路共享的
  target-derived finite sequence mask，并在 artifact 中记录 count 与 identities；
- 本 authority-only derivation fixes that mask at `57` sequences with ordered-name
  SHA-256 `2fa79d30ab6dd6a915098344c4aa7267cb6c3323c6d2a762b4b704f8757cebaa`;
- FK-palm `22/23`、semantic contact、2/5/7.5/10-cm physical contact、run length、
  MPJPE、object/pelvis goal、FS、learned gate、per-role attention entropy 全部报告；
  internal penetration 复用 official SDF formulas，在 internal native 10-Hz frames 上作为
  descriptive metric，且沿用 official excluded object categories/finite mask，不参与
  internal causal gate；
- native wrapper 只生成 D2-AC target，复用 sealed D2-X aggregate/per-sequence hashes 与
  released aggregate，执行既定 contact transfer、九项 protection、precision、penetration
  finite-mask 和 released 95% gates；不重跑 control、不选择中间 checkpoint；
- runner 接受既有 date-transition rule 下的 dated internal/native lifecycle identity，
  但 scientific protocol、selection、seed、threshold、bootstrap 与 gates 不随日期变化。

原 `p1-hoi-d2ac-gpu-smoke-s42-20260726` 已在 model/trainer implementation commit 上产生
稳定 no-update artifact，但它先于 authority full-suite closure，且旧 smoke metadata 没有
正确区分 batch-8 measured attention tensor 与 formal micro-batch-512 的 registered
`1,572,864` score-element estimate。该 artifact 必须保留，不能覆盖或复用；它不作为
formal-launch 的最终 smoke gate。完成本 amendment、authority full tests 和 clean commit
后，唯一允许的 operational retry 是
`p1-hoi-d2ac-gpu-smoke-r1-s42-20260726`（若未启动前跨日则按 date-transition rule 换新日期）。
Retry 的 architecture/data/timesteps/batch/no-update/no-checkpoint protocol 不变，只绑定最终
committed source tree并同时记录 batch-8 actual score-element shape、formal estimate 与实际
CUDA peak/headroom。Formal training run id、预算、随机初始化与全部 scientific gates 不变。

#### 2026-07-26 Phase 1B D2-AC0 authority CPU lifecycle retry amendment

原 `p1-hoi-d2ac-cpu-contract-s42-20260726` 已在任何 CUDA/optimizer/checkpoint workload
前因 resolved-config helper 错把 checkout root 推导为 `/data/yujinlun/code` 而中止；其
manifest、operational failure、CPU log 和 resolved config 已以 `aborted` 状态封存并注册，
不得覆盖或复用。该事件不构成 scientific contract failure，也没有产生模型、梯度或评测
结果。为满足 append-only lifecycle，唯一的 CPU retry identity 是
`p1-hoi-d2ac-cpu-contract-r1-s42-20260726`。Retry 只修复 run-root/path binding，保持
同一 D2-AC0 source/config/test/seed/BPS hashes、CPU contract、无 optimizer/CUDA/checkpoint
语义；必须在最终 logical implementation commit 的 clean tree 上重新创建 manifest，并在
CPU gate 失败时于 GPU 前停止。该 retry 不改变 GPU smoke r1、formal training、internal/
native run ids 或任何 scientific gate，也不授权 D2-AC1。

CPU retry `p1-hoi-d2ac-cpu-contract-r1-s42-20260726` 已于
`a32707047014abb2618b0b2c0ca5a23f55bfcc69` 完成并封存：329 项 authority tests 全部通过，
contract diagnostic 为 `cpu-contract-passed`，manifest/metrics SHA-256 分别为
`151b49d01c0b16980b1607a8b32e5e2fff24752cb6f5b744b07cddff75d5ddea` /
`b152ff16d90492a9010bab916035b8bd1c9179de38fef5965d05d5019f9d01ec`。该结果确认
interaction-adapter contract 可进入 worker publication；它没有创建 optimizer、执行 CUDA、
加载/写入 checkpoint 或进行 scientific selection。

#### 2026-07-27 Phase 1B D2-AC0 post-training evaluation lifecycle date-transition amendment

D2-AC0 formal training lifecycle 已于注册日期 `2026-07-26` 使用
`p1-hoi-d2ac-interaction-adapter-s42-20260726` 启动，并在 clean worker commit
`273e6d7e693f6664b3cd9d0c45b31b6b20c58496` 上完成固定的 61,440,000-window /
983,040,000-frame / 30,000-update from-random contract。训练 exit code 为 0；fixed
final-online checkpoint SHA-256 为
`fede1c2b2f331407ceba7db16e3a4b30ccc6ffb6c8fc252861662bdcc96c7b96`。完整 immutable
training tree 已由 worker 主动回收到 authority
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ac-interaction-adapter-s42-20260726`，worker 与
authority canonical tree SHA-256 均为
`d3784f0b01b8762ab1e6dcc7b0343ef2aa2147c1ca9672f516ae2f672cd92d98`
（115 files / 7,211,816,400 bytes）。

在真实日期 `2026-07-27` 核验时，authority 与 worker 均为 clean
`phase/01b-hoi@273e6d7e693f6664b3cd9d0c45b31b6b20c58496`；原绑定的 internal/native
identities 尚未创建 manifest、加载 checkpoint 或启动 workload。依照既有 date-transition
rule，只允许以下 identity-only replacement：

1. internal diagnostic 改为
   `p1-hoi-d2ac-interaction-adapter-internal-s42-20260727`，subphase 保持
   `1B-D2-AC0-internal`；
2. native evaluation 改为 `p1-hoi-d2ac-native-eval-s42-20260727`，subphase 保持
   `1B-D2-AC0-native`；
3. 未使用的 `p1-hoi-d2ac-interaction-adapter-internal-s42-20260726` 与
   `p1-hoi-d2ac-native-eval-s42-20260726` 在任何 workload 前 supersede，永不创建、复用或
   绑定 checkpoint；这不是 scientific/operational failure，也不产生 retry entitlement。

本 amendment 不改变 training run/checkpoint、seed 42、sealed D2-O 64×3 cohort、三路 paired
500-step rollout、10,000 次 sequence bootstrap、primary causal/locality gates、official
438×3 evaluator、sealed D2-X/released controls、penetration finite mask、native
transfer/protection/released-95% gates、classification 或 artifact contract。Internal 完成后仍
必须无条件执行一次 fixed native evaluation。D2-AC1、checkpoint selection、consistency、
HSIPrior、Mixer、任何 sweep 或新机制仍未授权。

#### 2026-07-27 Phase 1B D2-AC0 internal penetration-asset loader retry amendment

第一次 replacement internal lifecycle
`p1-hoi-d2ac-interaction-adapter-internal-s42-20260727` 在 clean
`655930f0d9b6bb47fbe116c1d779650cfd3dff63` 上通过 resolved-config、manifest、fixed
checkpoint hash 与 same-context GPU preflight 后启动，但在 21.504 秒后 fail-closed：
`load_penetration_assets()` 对 official asset `floorlamp.ply.npy` 使用 `Path.stem`，错误地产生
key `floorlamp.ply`；随后 fixed cohort 的 object category `floorlamp` 无法查到同一 SDF。Official
evaluator 已封存的实现使用 `file.split('.')[0]`，对该文件产生正确 key `floorlamp`。因此这是
一个确定的 internal evaluator asset-key transcription defect，不是 interaction adapter、
checkpoint、cohort、penetration formula/mask、threshold 或 gate 的 scientific failure。

失败 lifecycle 已以 `failed` finish/register，manifest/metrics/run-local registry SHA-256 分别为
`6c17c6e3b73664927c2f8c432c90e8df0e69a025c0a63a6c818765c1be0ab574` /
`2a1be5639174e7d85b46e7f6fd8fc1082077297d803584e1fd4bf207c7325e11` /
`e59caad3066042779b1b80394d3649d17cfa5ab2db0df3b5c5217fcdf8a2bfd7`，并由 worker 主动
回收到 authority；双端 canonical tree SHA-256 为
`e9eaea941d71624152d7d4d05d4ffc162f8fcd02f9c4e0cdf28ce139ce31d7ce`
（10 files / 46,938 bytes）。该 attempt 没有 optimizer、training update、checkpoint write/
selection、official-test use 或 consistency。

只允许以下最小 closure：

1. penetration SDF key extraction 改为与 immutable official evaluator 完全相同的首个 `.` 前
   basename；不改变 asset bytes、SDF/SMPL-X 公式、excluded categories、finite handling 或任何
   reported metric；
2. 加入 double-suffix regression test，并仅让 internal/native provenance regex 接受 append-only
   retry identity `p1-hoi-d2ac-interaction-adapter-internal-r1-s42-20260727`；
3. r1 必须重新生成 resolved config、same-context preflight 和 manifest，从头运行同一 64×3
   full/gate-ablated/locality-permuted paired rollout；不得复用失败 attempt 的 partial output；
4. native lifecycle 仍为 `p1-hoi-d2ac-native-eval-s42-20260727`，且仅在 r1 完成后运行一次。

除上述 filename-key parity 外，source/config/checkpoint/seed/batch/cohort/noise/bootstrap/
metrics/gates/classification/artifact contract 全部不变。若 r1 再次 contract-fail，则封存并停止；
不得继续 retry、改变 mask、删除 penetration、选择 checkpoint 或启动 D2-AC1/consistency。

#### 2026-07-27 Phase 1B D2-AC0 internal zero-denominator summary r2/native continuation amendment

Internal r1
`p1-hoi-d2ac-interaction-adapter-internal-r1-s42-20260727` 已在 clean
`7481c5ee2465725a857fd961876d8f1b997a0eed` 上从头完成 full、gate-ablated 与
local-correspondence-permuted 三条 fixed 64-sequence paired rollout；三份 raw variant 均为
64/64 sequences、finite、all-fields-reported，并共享 28-sequence official penetration
finite cohort。最终汇总在计算 descriptive
`full_mean / gate_ablated_mean` hand-penetration comparison 时失败，因为
gate-ablated 的合法 `hand_pen_loss_omomo` mean 为严格 `0`，而继承的 D2-M ratio helper
要求 denominator mean 严格大于 `0`。零 penetration 是合法且 lower-is-better 的 evaluator
结果；该失败只说明比值在零 denominator 下数学上未定义，不得将它解释为 evaluator metric、
mask、checkpoint 或 adapter contract 失败。

r1 已以 `interaction-adapter-contract-failure-stop` 封存并由 worker 主动回收。Manifest、
metrics、run-local registry 与 canonical artifact-tree SHA-256 分别为
`073bdb39605e518fd08124ad5175380dab2e4afedb455f1fe39ad624b724f28c` /
`ec79022477e5fb02cafaeb8329e0ee439c9d2612802aecda17607464fb582fa4` /
`89221db4a4cf7195c5a652472f9039f262c21642e9b39f7dc5b7941a93f0eca8` /
`2f7c013814fb294bedd5abe0d9c503e3ff1282369daa396ea1e3ac45ec5f9dd8`
（11 files / 15,375,787 bytes）。Full/gate-ablated/locality-permuted raw SHA-256 分别为
`7dc60b213777334d5e6d4a09bb78cf920a60c81858eedec384f3b5992370e472` /
`761844422a42f5738d1611e4d68bee9963ba7c18b639dcb312bd0a1c84c5f192` /
`f2abed08aa3eecf59179c03958f2e5a3e6671a0cb9409a9250ae727cb8a8ee9a`。
没有 optimizer、training update、checkpoint write/selection、official-test use 或
consistency。

用户在检查该证据后明确授权修复并完成最终评估。只允许以下 deterministic evaluator-summary
closure：

1. 仅为 D2-AC internal descriptive nonnegative penetration comparison 增加显式
   zero-denominator contract。原始 `hand_pen_loss_omomo` /
   `human_pen_loss_infbagel`、official finite mask、per-sequence values、excluded categories、
   SDF/SMPL-X 公式与 aggregate metric 均不变；
2. denominator mean `>0` 时必须调用原 `paired_mean_ratio`，保持 seed 42、10,000 paired
   sequence bootstrap、point estimate、CI 与 per-unit 数据逐字段一致；
3. denominator mean `==0` 时不得加 epsilon/pseudocount、不得 clamp 成有限 ratio、不得用
   infinity 或改变 metric。必须记录 `ratio_defined=false`、`mean_ratio=null`、
   `bootstrap_95_ci=null` 与原因 `zero_denominator_mean`，同时用同一 paired sequence values
   调用原 `paired_difference`，报告绝对 difference point estimate/95% CI；
4. mismatched、empty、non-finite 或 negative penetration vectors 继续 fail-closed；
5. 新 internal identity 唯一为
   `p1-hoi-d2ac-interaction-adapter-internal-r2-s42-20260727`。r2 必须重新生成 resolved
   config、same-context preflight/manifest，并从头运行完整三路 rollout；不得复用 r1 raw
   output；
6. primary internal mechanism/locality gates 仍只使用 registered direct-hand union 5-cm F1
   paired differences 与 GT-contact-frame distance paired differences；descriptive penetration
   ratio/difference 不参与 gate；
7. r2 contract 完成后，无论 mechanism gate 正负，执行一次仍未启动的
   `p1-hoi-d2ac-native-eval-s42-20260727`。Native 必须继续使用未修改的 official 438×3
   production evaluator、sealed D2-X control、released aggregate、181-sequence penetration
   mask、原 paired ratio/difference helpers、阈值与 gates；本 amendment 不改变 native
   evaluator 数学或既有实验口径。

Authority implementation closure 在 plan-only commit
`ef39e62c2d30c9dd0d2575121a7806375d53e23b` 之后完成：新增 helper 只由 internal penetration
summary 调用，native wrapper 明确不引用它；既有 `paired_ratio_fixed` 函数 source SHA-256
在修改前后均为
`2d1e58aab9d340250eb90e3ea176132380e4d27fea069ec3c51d33ff90fe9b08`。
Authority targeted tests 为 67 passed，full suite 为 333 passed，registry validation 为
178 records valid。对 immutable r1 raw artifacts 的只读 summary replay 成功产生
`ratio_defined=false` / null ratio 与原 paired-difference CI，没有写回或提升 r1；正式
mechanism classification 仍只由新 r2 lifecycle 决定。

若 r2 仍发生 contract failure，则封存并停止，不得继续 retry。D2-AC1、checkpoint
selection、consistency、任何 architecture/token/parameter/placement sweep、新 loss、
HSIPrior、Mixer 或其他 HOIPrior 搜索仍未授权。

#### 2026-07-27 Phase 1B D2-AC0 native serialized-field parity r1 amendment

Internal r2 已完成全部 contract，并按预注册规则无条件启动 fixed native lifecycle
`p1-hoi-d2ac-native-eval-s42-20260727`。该 lifecycle 在 clean
`cc931b8b6272e323e25be6cc6c6a6e3a49076558` 上完整运行 immutable official
438-sequence × 3-window、500-step production evaluator，生成了 438 条 target
per-sequence records 和 finite aggregate metrics；official aggregate 中 Troot/Tobj/Oobj
分别为 `8.722931146621704` / `16.810938650698322` / `1.0305662157312816`。Target 与
sealed D2-X 的 `trans_dist`、`obj_trans_dist`、`obj_rot_dist` per-sequence fields 均为
438/438 finite，target penetration fields 也严格保持 sealed 181-sequence finite mask。

Official evaluator 完成后，D2-AC post-evaluator paired summary 在第一个 Troot protection
comparison fail-closed，错误为
`native metric trans_dist contains missing/nonfinite values`。根因是 D2-AC wrapper 的
`PER_SEQUENCE_KEYS` 将 evaluator 已封存的 serialized keys `trans_dist` /
`obj_trans_dist` / `obj_rot_dist` 错误映射成不存在的
`translation_difference` / `object_translation_difference` /
`object_rotation_difference`；这使 `_metric_array` 读到 `None`，并不表示 evaluator raw
records 非有限。D2-X sealed wrapper 与 official records 始终直接使用短字段名，因此该问题是
D2-AC paired-summary field routing defect，不是 checkpoint、sampler、official metric、
finite mask 或 native gate 的科学结果。

失败 lifecycle 已以 status `failed` 原样封存并由 worker 主动回收。Manifest、metrics、
run-local registry、aggregate、per-sequence 与 canonical artifact-tree SHA-256 分别为
`e3134b5567eac018a6b99c49c05276d802db3a4d2b6c7864adbbaac419bbd6d6` /
`0ab818a018cd7465b8e25b831992d5bd2ba5f76a0b89af73c30c34394035218e` /
`dead4f9ebbcb639da24b8629272daa7f9a82eded272c94bbf603cfe4b0433262` /
`995acb311187a1f0bfd8abe2f74358da70998deb9ce5b8c98a99e9e36b99e6c3` /
`dd8803c8efe4b836a09d31dc8c86b6f8230d3de6cb92aa2f30c574b96cb4ad6a` /
`ae7f0e3d9975a2bc4d96058dc1f3c5a965a4fb9870634d10889db97f2a0e1b27`
（14 files / 339,245 bytes）。Authority staging 为
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ac-native-eval-s42-20260727`；
worker/authority tree hash 完全一致。该 attempt 没有训练、optimizer/update、
checkpoint write/selection 或 consistency，也没有形成 native gate classification。

用户已明确要求修复并完成与既有实验一致的最终评估。只允许以下 deterministic native
closure：

1. 将上述三个 D2-AC `PER_SEQUENCE_KEYS` value 改为 official evaluator 与 sealed D2-X
   records 的原 serialized short keys；其他 key mapping、metric formula、aggregate、
   finite handling、penetration mask、bootstrap helper、threshold 与 gate 全部不变；
2. 增加基于 official per-sequence schema 的回归测试，证明 Troot/Tobj/Oobj 从 short keys
   读取，旧的不存在 alias 不被要求；继续 fail-closed 于真实 missing/non-finite values；
3. 用 source hash/test 证明 `code/test_infbagel_hoi.py`、`code/eval_metrics.py`、
   `code/config/config_eval_hoi_prior.yaml`、shared D2-X wrapper、sealed control/baseline 与
   internal zero-denominator helper 均未改变；
4. 新 identity 唯一为 `p1-hoi-d2ac-native-eval-r1-s42-20260727`。必须重新生成 resolved
   target/config、same-context preflight 和 manifest，并从头运行完整 official 438×3
   evaluator；不得复用失败 attempt 的 aggregate/per-sequence/partial output；
5. retry 仍只加载 D2-AC0 fixed final-online checkpoint，复用 sealed D2-X control，不重新生成
   control；paired unit、seed 42、10,000 bootstrap、181-sequence penetration mask、
   transfer/protection/released-95% gates 和 classification precedence 全部不变；
6. 无论 native transfer 结果，最终 D2-AC0 classification 仍受已封存 internal locality
   failure 约束，不得 selectable，也不 eligible for D2-AC1。

若 native r1 仍 contract-fail，则封存并停止，不得继续 retry、修改 evaluator/mask/gate、
选择 checkpoint、启动 D2-AC1/consistency、HSIPrior、Mixer 或任何新 HOIPrior 搜索。

#### 2026-07-27 Phase 1B D2-AC0 native serialized-field parity scope correction

在上述 plan-only amendment 后、worker publication 或 native r1 启动前，authority 使用失败
attempt 已回收的 immutable 438-sequence target records 与 sealed D2-X records 对修正后的
paired-summary 做了只读 replay。Troot/Tobj/Oobj 已能完整计算，但 penetration mask contract
显示 0 finite sequences。原始 records 的
`hand_pen_loss_omomo` / `human_pen_loss_infbagel` 实际仍为两侧相同的 181 finite
sequences；0-mask 同样来自 `PER_SEQUENCE_KEYS` 中不存在的
`hand_object_penetration` / `human_object_penetration` aliases。

因此 native field-parity fix 的完整范围校正为五个、且仅五个 official serialized short-key
identity mappings：

- `trans_dist -> trans_dist`；
- `obj_trans_dist -> obj_trans_dist`；
- `obj_rot_dist -> obj_rot_dist`；
- `hand_pen_loss_omomo -> hand_pen_loss_omomo`；
- `human_pen_loss_infbagel -> human_pen_loss_infbagel`。

回归测试必须同时锁定五个 mappings、真实 missing/non-finite fail-closed behavior 与 sealed
181-sequence penetration mask replay。该 scope correction 不改变 SDF/evaluator formulas、
excluded categories、finite mask、paired statistics、native gates、retry identity 或从头重跑
要求；前一 amendment 中“其他 key mapping 不变”应解释为除这五个已证实 alias defect 外
全部不变。若五-key replay 不能通过完整 comparison contract，则不得发布 worker 或启动
native r1。

Authority implementation closure 在 plan-only commit
`376950ea03652306e448bd8c7e7f27362860dd54` 后完成。D2-AC wrapper 只将五个 confirmed
aliases 改为 official short-key identity mappings，并只扩展 lifecycle regex 以接受已登记的
`p1-hoi-d2ac-native-eval-r1-s42-20260727`；没有改变 official evaluator、shared D2-X
wrapper、internal diagnostic、paired statistics 或 gate code。修正后 native wrapper 与
D2-AC tests SHA-256 分别为
`04b49c17602d13da2f45f2ae47dba191c4a21a5e914ada560994cdde3c0c827c` /
`1c51204f5f8140d95bc5a1abbd5e76cab4812759b349c8e3211385c2707a2c3f`。

Authority targeted D2-AC tests 为 25 passed，full suite 为 335 passed，registry validation
为 180 records valid，`py_compile` 与 `git diff --check` 通过；full-suite log SHA-256 为
`528c707aa2413a23e007dc92580c954cf4832cf5dba7dc0d5bd452ae49264619`。对失败 native
attempt 的 immutable raw records 做只读 replay 后，全部 9 个 protection ratios 与三条
contact paired differences 均成功产生，penetration contract 恢复为精确 181 sequences /
`2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec`，没有写回 artifact
或用于正式 selection。Locked source SHA-256 仍为：official test
`22886f8797ceb04a892487393dea9f80e19877bc02dd7a6f39127e7319119524`、eval metrics
`445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547`、eval config
`89c702d96b98289924225c4b163d3b29eb22efe27c50ac799ddd0c71c515aa73`、shared D2-X
wrapper `b6753a66207492e6ee4addb8f450cb38c5d021401d43430faa9e5c9ed77c6e31`、internal
diagnostic `e9a0157f80695469a53a5333b20685cb3c66d042b0ccd621b86164238764bcc5`。

#### 2026-07-27 Phase 1B D2-AC0 completion record

D2-AC0 的全部已批准 lifecycle 已完成。最终 tracked compact result 为
`experiments/results/p1_hoi_phase1b_d2ac_interaction_adapter_s42_20260727.json`，
phase summary 为 `docs/phase_summaries/PHASE_1B_D2AC.md`。本 completion record 只封存
已经执行的固定机制、失败与结果，不新增方向、训练、selection 或 fallback。

1. **CPU、smoke 与 training。** Authority CPU retry
   `p1-hoi-d2ac-cpu-contract-r1-s42-20260726` 通过 exact BPS/assignment、参数量、
   `[B,16,232]` API、`alpha=0` base parity、初始/activated gradients、local permutation、
   extreme-input、provenance、HSIPrior/Mixer independence 与 static path contract。
   Registered final smoke
   `p1-hoi-d2ac-gpu-smoke-r1-s42-20260726` 在 `cuda:0`、real-data batch 8 上通过，
   measured attention score shape/elements 为 `[8,16,3,4,16] / 24,576`，formal
   micro-batch-512 estimate 为 `1,572,864`，peak allocated/reserved/headroom 为
   `252,510,720 / 304,087,040 / 24,991,956,992` bytes。Formal training
   `p1-hoi-d2ac-interaction-adapter-s42-20260726` 从随机初始化完成
   `61,440,000` windows / `983,040,000` frames / `30,000` updates，wall time
   `19,157.121 s`，throughput `3,207.162 windows/s`，20 cadence checkpoints 与
   80 rank RNG sidecars 完整；fixed final-online SHA-256 为
   `fede1c2b2f331407ceba7db16e3a4b30ccc6ffb6c8fc252861662bdcc96c7b96`。
   Learned alpha/gate 为 `0.0907876045 / 0.0905389935`。
2. **Internal r2。**
   `p1-hoi-d2ac-interaction-adapter-internal-r2-s42-20260727` 在 sealed
   64-sequence/192-window cohort 上从头完成三路 paired 500-step rollout。Full minus
   gate-ablated direct-hand union 5-cm F1 为 `+0.6215448246`，95% CI
   `[0.5397640759,0.7003120412]`；gate-ablated minus full GT-contact distance 为
   `+90.978005 cm`，CI `[81.0602569,100.8264305]`，证明 adapter 被使用。Full minus
   locality-permuted F1 为 `+0.0103920517`，CI
   `[-0.0177715936,0.0375934559]`；permuted minus full distance 为
   `+0.013819838 cm`，CI `[-0.3039546465,0.3092713829]`，两项 locality gate 均失败。
   合法零 hand-penetration denominator 显式记录为 `ratio_defined=false`、null ratio/CI，
   并保留同一 paired values 的 difference `9.308420447e-07`、CI
   `[0,2.792526134e-06]`；没有 epsilon、pseudocount、clamp 或 infinity encoding，
   且该 helper 不进入 native evaluator。Internal classification 为
   `interaction-adapter-locality-negative-stop`。
3. **Official native parity 与结果。** Final native lifecycle
   `p1-hoi-d2ac-native-eval-r1-s42-20260727` 在 clean
   `e6ee3fd9611ede9ee8e0cad20b94bd81e9c13366` 上从头运行 official 438×3、
   500-step unguided evaluator。D2-AC wrapper 的修复仅把五个 paired-summary
   serialized-field aliases 路由到 official short keys：
   `trans_dist`、`obj_trans_dist`、`obj_rot_dist`、`hand_pen_loss_omomo`、
   `human_pen_loss_infbagel`。Official evaluator、eval metrics、eval config 与 shared
   D2-X wrapper SHA-256 仍分别为
   `22886f8797ceb04a892487393dea9f80e19877bc02dd7a6f39127e7319119524` /
   `445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547` /
   `89c702d96b98289924225c4b163d3b29eb22efe27c50ac799ddd0c71c515aa73` /
   `b6753a66207492e6ee4addb8f450cb38c5d021401d43430faa9e5c9ed77c6e31`。
   与 sealed D2-X resolved target 对比，在排除 run/output/checkpoint identity 后无
   semantic config difference；control aggregate/per-sequence 原 hash 复用且未重生成。
4. **Native gates。** D2-AC target 的 end-object/Txy/FS/contact
   precision/recall/F1/coverage/Pbody/hand penetration/MPJPE/Troot/Tobj/Oobj 为
   `5.6473 / 4.2379 / 0.3986 / 0.7876 / 0.6042 / 0.6480 / 0.4913 /
   4.0121 / 0.2518 / 12.4268 / 8.7229 / 16.8110 / 1.0306`。
   Contact F1/recall paired differences 为 `+0.0105639`、`+0.0097080`，但 CI
   `[-0.0088320,0.0303036]`、`[-0.0124421,0.0322983]` 均包含零；released
   contact-F1 gap closure 仅 `0.1175963 < 0.25`。End-object、FS、Pbody 与 hand
   penetration protection CI upper bounds 分别为
   `1.58993 / 1.17165 / 1.19812 / 1.18712`，超过 `1.10`；181-sequence
   penetration mask contract 通过。Released-95% effectiveness gate 也失败。Evaluator
   未生成 FID、Matching、R-Precision 或 Diversity，缺失原样保留且
   `fid_rprecision_used=false`。
5. **Final decision 与 artifact recovery。** Classification precedence 由已经失败的
   internal locality gate 决定，最终严格分类为
   `interaction-adapter-locality-negative-stop`；即使 native transfer/protection/
   effectiveness 也失败，不重新命名 classification。Training/internal/native recovered
   tree SHA-256 分别为
   `d3784f0b01b8762ab1e6dcc7b0343ef2aa2147c1ca9672f516ae2f672cd92d98` /
   `62225323d8a5d3d252d34587165bd2da0ade4ed469ddae1c644e848cd391e753` /
   `83b6a811eab7e519f5f15ce2cfeb36d12bb8814625905ac7f2378caeb8fefa34`。
   Internal initial SDF failure、internal r1 zero-denominator failure、native initial
   serialized-field failure 与原 smoke 均按 append-only contract 保留，未覆盖或删除。

D2-AC0 fixed final-online checkpoint 不可选择、不可 resume、不可初始化后续 prior。
D2-AC1 只有 `interaction-adapter-positive-but-not-effective-stop` 才 eligible；当前
locality-negative classification 不满足该条件，因此 D2-AC1 严格 ineligible 且未授权。
不得自动启动 consistency、HSIPrior、Mixer、checkpoint selection、任何 adapter/token/
parameter/placement sweep、新 loss、SNR weighting、gradient projection、rollout exposure、
CFG/guidance 或新的 HOIPrior 搜索。

#### 2026-07-27 Phase 1B D2-AD0 human-local full-mesh BPS coordinate-contract repair 预注册（plan-only）

本 amendment 只注册一个由 D2-AC0 封存结果直接触发、已经得到用户确认的单变量
coordinate-contract repair，不重新开放 HOIPrior 搜索。Identifier audit 确认 D2-AD/`d2ad`
在本 amendment 前未出现在 plan、registry、source、tests 或 lifecycle id 中，因此 D2-AD
是下一个 unused Phase 1B identifier。当前 plan-only source HEAD 为
`dcf871644b6a1b72116dbab03dcc4fafc755dc28`，branch 为 `phase/01b-hoi`，authority
worktree 在修改前 clean。

1. **封存证据与可证伪假设。** D2-AC0 已证明 adapter 被优化器强烈使用：full minus
   gate-ablated direct-hand union 5-cm physical-contact F1 为 `+0.621545`，且
   gate-ablated minus full GT-contact-frame hand-object distance 为 `+90.978 cm`；
   但 full minus local-correspondence-permuted F1 仅 `+0.010392` 且 CI 包含零，
   permutation minus full distance 仅 `+0.01382 cm` 且 CI 包含零。Native contact
   F1/recall 相对 sealed D2-X 只增加约 `1.66%/1.63%`，同时 end-object 与 FS 分别退化
   `50.99%/9.81%`。因此 D2-AC0 严格保持
   `interaction-adapter-locality-negative-stop`，其 checkpoint 不得选择、resume 或初始化
   D2-AD。

   后续 code audit 发现一个更具体、可独立修复的 coordinate contract：

   - 232-D human/object state 使用 Y-up、window-local XZ origin、initial-root-yaw aligned
     frame；
   - dataset 的 author BPS delta 已执行 `zup_to_yup`；
   - D2-AC local token 的 cluster basis mean 却直接来自 raw `code/bps.pt`，仍是原
     Z-up convention；
   - 即使只把 raw basis mean 或 stored/global delta 做轴变换/旋转，也无法恢复正确
     locality，因为 fixed global queries 的 nearest-point correspondence 会随共同 global
     yaw 改变，component-wise RMS 也不是任意旋转下的不变量。

   D2-AD0 的唯一科学假设是：若 D2-AC locality failure 的主要原因是 adapter-only local
   geometry 没有与 human window-local frame 建立一致的 nearest-point correspondence，则
   在同一 full rest mesh 上直接重算 human-local BPS，应使 causal locality gate 与 native
   contact transfer 改善；这不预先声称该 coordinate mismatch 足以解释全部 HOIPrior
   baseline gap。

2. **只读 authority-CPU 原型证据。** Prototype 只读取 locked split、immutable PLY、
   `code/bps.pt` 和当前首帧 pose，没有创建 checkpoint、optimizer、CUDA workload 或
   per-window condition artifact。

   - BPS file SHA-256：
     `fdff7204b4697e105457cb7e39267b9555bc0d8d854dbc92cd67e2d8c3e77042`；
   - fixed Y-up basis float32 tensor SHA-256：
     `02b4f8f3510e723174010a823630f663ddda9875ad82a2f8de807d2bdccebd7d`；
   - raw-versus-Y-up basis/cluster-mean max abs：
     `1.3970013 / 1.1064382`；
   - sealed D2-O 64-sequence × 3-window cohort selection SHA-256：
     `1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`；
   - 192 windows cover all 13 object classes；在共同 global yaw
     `{-179,-90,-37,53,120,179}` degrees 下，full-mesh local BPS max abs
     `1.4901161e-7`，其 `[B,16,10]` cluster feature max abs
     `2.3841858e-7`；
   - exact query 的 worker-count `1/3/all` 输出逐位一致；
   - 相同 real-window probe 中，只旋转旧/global BPS 的共同 37-degree yaw max-abs
     error 平均 `0.6791 m`、最坏 `1.1257 m`；只修 basis 但保留 global delta 的
     cluster-feature max-abs error 平均 `0.1541`、最坏 `0.5767`；
   - full-local 与 rotate-old delta 的逐点 L2 差异平均 `0.5958 m`；
   - 对 192-window cohort 滚动 relative object pose 后，local BPS 逐点 L2 平均改变
     `0.1691 m`，证明 condition 不是 constant；
   - batch-grouped exact query 在 authority CPU 上对 192 windows/13 objects、
     3 query workers 耗时 `1.4742 s`；real DataLoader prototype 保留
     `batch_size=512,num_workers=4` 时测得约 `308.84 windows/s/rank` 的 condition
     delivery。该值只用于 wall-time planning；registered worker smoke 必须重新实测。

3. **唯一 manipulated factor 与精确 local-BPS 方程。** D2-AD0 相对 D2-AC0 只修复
   adapter local geometry 的 coordinate/query contract。Global BPS condition token、
   232-D state、loss、trunk、adapter 参数、placement、sampler 和 evaluator 全部保持。
   令固定 axis conversion 为 \(C_{Z\rightarrow Y}\)，raw BPS basis point 为
   \(b_i^Z\)，则 human-local fixed query 为

   \[
   b_i^L=C_{Z\rightarrow Y}b_i^Z.
   \]

   对每个 current window，令 \(W\) 为 current human frame 的 world-to-local rotation，
   \(R_O\) 为 current/global object rotation reference，immutable Y-up rest-mesh vertex
   为 \(v_j\)。定义

   \[
   L_O=WR_O,\qquad
   j^*(i)=\arg\min_j\lVert L_Ov_j-b_i^L\rVert_2^2,\qquad
   d_i^L=L_Ov_{j^*(i)}-b_i^L .
   \]

   实现允许利用旋转保距性把 query 送回 rest-object frame 后做同一 exact nearest-vertex
   查询，但输出必须与上述定义一致。共同 global yaw \(G\) 下
   \(W'=WG^{-1},R'_O=GR_O\)，所以 \(W'R'_O=WR_O\)；local BPS 及其
   component-wise RMS 必须保持不变。

4. **Immutable geometry 与 exact builder contract。** 只读取
   `data/object/rest_object_geo/*.ply` 的全部原始 vertices；禁止 100-point/1024-point/
   任何新 mesh subsample、SDF/voxel approximation、mesh encoder、category embedding、
   train-stat normalization 或 per-window local-BPS file/cache。13-file canonical
   PLY manifest SHA-256 为
   `ce8328ef2bf873a79d74fb5fd20cc488551a20d56fe5c5ecabf609824b0654d1`；
   sorted object mapping 为
   `[clothesstand,floorlamp,largebox,largetable,monitor,plasticbox,smallbox,smalltable,`
   `suitcase,trashcan,tripod,whitechair,woodchair]`，mapping SHA-256 为
   `424fc96102c576a1d11b0824cc0ee616d52cd9e39524819f49b207d1598fe41b`。

   Builder 固定使用 `scipy.spatial.cKDTree.query(k=1,eps=0,p=2)`；tree 只缓存每类
   immutable rest mesh 的 spatial index，不缓存任何 window condition。Training collate
   按 object 分组并固定 `local_bps_query_workers=3`，继续使用
   `num_workers=4`；query worker count 是 48-CPU worker 上的 operational ownership，
   不是 scientific sweep。Authority/worker 必须记录 SciPy/dependency hash，并验证
   worker-count 不改变 indices/output。输出固定 float32 `[B,1024,3]`，随后继续使用
   D2-AC 同一 16-way assignment、cluster identities、cluster sizes 与
   `[mean basis,mean delta,RMS delta,mean norm]` 10-D statistics。Cluster basis mean
   改为同一 assignment 上的 \(b_i^L\) mean；assignment 不重新按 Y-up lexicographic
   seed 派生，仍锁定：

   - centers：
     `[328,903,503,817,474,1023,382,864,640,431,445,960,547,829,545,756]`；
   - sizes：
     `[39,40,57,61,65,68,70,134,77,64,59,79,43,46,84,38]`；
   - assignment SHA-256：
     `b62f91f4eb6c4bf2a9211f0187cd1eb97c25394ee45de155f33607959fddeecd`。

5. **Training/rollout causal availability。** Training 只能从 current window 第一帧的
   human frame \(W\)、current object rotation reference \(R_O\) 和 immutable rest mesh
   构造 adapter-only local BPS。不得读取 future pose/contact、stored future/per-frame
   local BPS 或 evaluator statistics。Autoregressive rollout 的第一窗口使用 evaluator
   已提供的 current history frame；后续窗口必须从 generated two-frame history 建立新的
   `WindowFrame` 和 generated object reference，再重算 local BPS。Global BPS token 保持
   D2-X/D2-AC author semantics：第一窗口读取既有 current BPS，后续窗口沿用既有
   current-generated BPS replay。Local BPS 只送入 interaction adapter，不能进入 global
   BPS encoder、loss、evaluator threshold、HSIPrior 或 Mixer。

6. **Architecture 与 parameter lock。** 新 checkpoint variant 固定为
   `d2ad_local_frame_interaction_adapter`。它复用 D2-AC：

   - 512-wide、16-head、8-layer trunk；
   - 4 condition tokens、16 motion tokens；
   - layer 4 后、layer 5 前的单个 adapter；
   - 16 object tokens、3 roles、128 adapter width、4 attention heads、dropout 0；
   - `10→128→128` object encoder、`512→128` query、`384→512` writeback；
   - single scalar `tanh(alpha)` ReZero gate，alpha 严格从 0 初始化；
   - exact adapter/base/total parameters
     `349,697 / 29,673,448 / 30,023,145`，增量 `1.1785% <=1.25%`。

   不改变 role query、token 数、width/depth/placement、global BPS token 或任何 parameter。
   D2-AD0 全部矩阵、embedding、gate 以 seed 42 从随机初始化；不得加载 released、author、
   D2-V/X/Y/Z/AB/AC、prior、EMA、consistency 或任何 weight-init/resume checkpoint。
   D2-AC checkpoint schema 必须被 D2-AD loader 拒绝，反之亦然。

7. **保持不变的 optimization contract。** Fixed split 仍为
   `experiments/splits/omomo_hoi_train_validation_seed42.json`，SHA-256
   `019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e`。
   D2-AD0 只在 `infbagel-4gpu/node01`、4×RTX 3090 上训练；per-GPU batch 512、
   effective batch 2,048、accumulation 1；总预算
   `61,440,000 windows / 983,040,000 frames / 30,000 updates`。Optimizer 仍为
   FP32 Adam、LR `1e-4`、betas `(0.9,0.999)`、weight decay 0、无 warmup/scheduler、
   AMP、gradient clipping、EMA；primary 为 fixed final-online。FK/object-surface/
   velocity/terminal-goal weights 仍为
   `0.3569973401779424 / 0.4772322188400037 / 0.1 / 1.0`；D2-X FK-foot routing
   enabled，D2-AB support objective disabled。Formal run 必须从随机初始化持续完整预算，
   不人为 pause、不选择中间 checkpoint。

8. **Authority CPU fail-fast contract。** 任何 GPU 前必须以 authority Python 完成并归档：

   - BPS、Y-up basis tensor、split、13 PLY、mapping 与 assignment hashes；
   - raw-to-Y-up conversion、same assignment/cluster sizes、`[B,1024,3]` local BPS 与
     `[B,16,10]` features 的 shape/dtype/finiteness；
   - sealed 64×3 cohort、上述 6 yaw 的 local-BPS max abs `<=1e-6`，cluster-feature
     max abs `<=1e-6`；
   - query workers `1/3/all` indices/output exact parity；
   - repeated-call determinism、batch ordering、all 13 object coverage、relative-pose
     sensitivity、zero/constant/extreme input finiteness；
   - dataset-collate 与 evaluation helper exact parity；
   - first training window current-pose parity，以及 generated-history rollout 不读取
     future GT/stored local BPS 的 static and runtime audit；
   - exact parameter count、`[B,16,232]` output、alpha-zero shared-trunk parity
     `<=1e-6`；
   - initial alpha gradient finite/nonzero；test-only `tanh(alpha)=0.1` probe 下
     object encoder/identity/query/QKV/out/writeback gradients finite/nonzero，probe
     不保存、不训练；
   - local correspondence permutation causal effect、role separation、dtype/device/
     batch propagation；
   - base/D2-AC/D2-AD checkpoint provenance rejection；
   - HSIPrior parameter/storage/forward unchanged；Mixer 只消费 clean
     `[B,16,232]` output；
   - static scan 无 future GT、stored per-window local BPS、mesh subsample、evaluator
     threshold/helper、new loss/guidance 进入 D2-AD model path。

   任一失败在 CUDA 前分类并停止：
   `local-frame-interaction-adapter-contract-failure-stop`。

9. **Registered GPU smoke。** Implementation lifecycle 必须按真实日期绑定未使用 id。
   Smoke 只在 worker `cuda:0`，real-data batch 8、timesteps `0/249/499`、random init、
   no optimizer/update/checkpoint load/write；必须使用与 formal training 相同 exact
   full-mesh collator，记录 local-BPS construction wall time、initial model/local-BPS
   hashes、coordinate contract replay、alpha gradient、test-only nonzero-gate adapter
   gradients、CUDA-synchronized peak allocated/reserved/headroom、四卡 visibility/
   contention。Cross-attention score shape/element count不变；formal throughput/ETA 以
   实测为准，不以 authority CPU prototype 代替。

10. **Fixed internal causal diagnostic。** Training 完成后只加载 D2-AD0 fixed
    final-online checkpoint，在 sealed D2-O 64×3 cohort、phase offsets
    `(14,56,98)`、selection SHA-256
    `1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`
    上运行与 D2-AC 相同的三条 paired 500-step rollout：

    - `full`；
    - `gate_ablated`：每一步强制 `tanh(alpha)=0`；
    - `local_correspondence_permuted`：只将 human-local cluster delta statistics
      `k<-(k+8) mod 16`，保留 local basis mean、learned object identity、global BPS
      与其余 condition。

    三条 path 共享 initial latent、每步 posterior noise、condition、window ordering 与
    history restoration。指标、sequence-unit seed-42 10,000 bootstrap、attention entropy
    appendix 和 primary mechanism/locality gates 与 D2-AC 完全相同；official test 禁止，
    no optimizer/update/checkpoint write/selection。Primary gate 仍要求：

    - full minus ablated direct-hand union 5-cm physical-contact F1 CI lower `>0`；
    - full minus permuted 同一 F1 CI lower `>0`；
    - ablated minus full GT-contact-frame hand-object distance CI lower `>0`；
    - permuted minus full 同一 distance CI lower `>0`。

11. **Fixed native evaluation 与 gates。** 无论 internal 正负，都必须执行一次与 sealed
    D2-X/D2-AC protocol-identical 的 official 438 sequences × 3 windows、500-step、
    unguided production evaluation；只改变 target run/checkpoint/architecture identity 和
    adapter-only local-BPS construction。Official evaluator、metric keys、181-sequence
    penetration finite mask、bootstrap seed/replicates 不得调整。Sealed D2-X checkpoint/
    aggregate/per-sequence hashes仍为
    `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51` /
    `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
    `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`，
    不重新生成。Released aggregate hash 仍为
    `76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6`。

    Selection gates 完全复用 D2-AC：

    - D2-AD minus D2-X contact F1 与 recall paired CI lower 均 `>0`；
    - contact-F1 released-gap closure `>=0.25`，对应 point estimate 最低约
      `0.6598838781`；
    - end-object、Txy、FS、Pbody、hand penetration、MPJPE、Troot、Tobj、Oobj
      paired mean-ratio CI upper 全部 `<=1.10`；
    - contact precision difference CI lower `>=-0.02`，penetration finite-mask
      contract 通过；
    - released 95% point-effectiveness gate 保持原 lower/higher-is-better 公式。

    Native output 还必须以 sealed artifacts 对 D2-AC0 作相同 sequence-paired、仅描述性
    comparison，以量化 coordinate repair 相对唯一前驱的改变；该 secondary comparison 不
    参与 checkpoint selection，不触发额外 generation。FID/Matching/R-Precision/Diversity/
    timing 若 evaluator 生成必须原样保留和报告，FID/R-Precision 不参与 selection。

12. **分类、授权边界与 lifecycle。** Classification precedence 固定为：

    - contract failure：
      `local-frame-interaction-adapter-contract-failure-stop`；
    - adapter unused：
      `local-frame-interaction-adapter-unused-optimization-negative-stop`；
    - locality negative：
      `local-frame-interaction-adapter-locality-negative-stop`；
    - native transfer negative：
      `local-frame-interaction-adapter-transfer-negative-stop`；
    - protection conflict：
      `local-frame-interaction-adapter-conflict-negative-stop`；
    - mechanism/transfer/protection 通过但 released-95% 失败：
      `local-frame-interaction-adapter-positive-but-not-effective-stop`；
    - 全部通过：
      `local-frame-interaction-adapter-positive-candidate-stop`。

    只有最后一类允许把 fixed final-online checkpoint 标为 selectable autonomous
    HOIPrior candidate；不得选择中间 checkpoint。D2-AD0 没有自动 longer-budget
    extension 或 fallback；任何 D2-AD1、budget/LR/token/width/depth/placement/role/
    query-worker scientific sweep、新 loss、SNR weighting、gradient projection、
    rollout exposure、CFG/guidance、consistency、HSIPrior 或 Mixer 都需新的 dated plan、
    append-only registry 和用户再次明确确认。

    用户已授权在上述固定 D2-AD0 范围内连续完成 implementation、CPU tests、worker
    publication、registered smoke、from-random full training、fixed internal/native
    evaluation、artifact recovery、compact result、phase summary 与 completion record。
    本 plan-only commit 不改 source、不创建 lifecycle run、不启动 CPU contract/GPU/
    training/evaluation；implementation session 必须先重新读取真实 date，并以未使用的
    dated lifecycle ids 写 implementation binding amendment。跨日的尚未启动 lifecycle
    必须在 workload 前 append identity-only amendment，绝不复用或覆盖 id。

13. **Artifact 与 closure。** 必须保留/hash verify resolved configs、authority CPU logs、
    PLY/BPS/mapping manifests、worker preflight、smoke manifest/log/metrics、formal
    training manifest/log/state、all cadence checkpoints/per-rank RNG、initial/final model
    hashes、local-BPS construction throughput、wall time/ETA、internal full/ablated/permuted
    artifacts、paired-noise/attention appendix、native aggregate/per-sequence/bootstrap/
    penetration mask/optional metrics、run-local registry、dependency/hardware/data/evaluator
    hashes、complete recovered tree、compact result、
    `docs/phase_summaries/PHASE_1B_D2AD.md` 和全部 operational/scientific failures。
    大 artifact 不进入 Git。Logical implementation commit 必须同时含 source、config、
    tests、dated implementation amendment、registry binding 与必要 documentation。

#### 2026-07-27 Phase 1B D2-AD0 implementation/lifecycle binding amendment

Authority 在 plan-only commit
`ccc023f44056a056131c730ff39a2dfae447505b`、clean `phase/01b-hoi` 和真实日期
`2026-07-27` 上进入已授权的 D2-AD0 implementation。以下 identities 在创建本
amendment 前均未使用：

- implementation logical change：
  `p1-hoi-d2ad-local-frame-interaction-adapter-implementation-s42-20260727`；
- authority CPU contract：
  `p1-hoi-d2ad-cpu-contract-s42-20260727`；
- registered GPU smoke：
  `p1-hoi-d2ad-gpu-smoke-s42-20260727`；
- formal from-random training：
  `p1-hoi-d2ad-local-frame-interaction-adapter-s42-20260727`；
- fixed internal diagnostic：
  `p1-hoi-d2ad-local-frame-interaction-adapter-internal-s42-20260727`；
- fixed native evaluation：
  `p1-hoi-d2ad-native-eval-s42-20260727`。

Implementation logical commit 必须包含 source、config、tests、本 amendment、registry
binding 和必要 documentation；在该 committed Git object 通过 authority CPU contract 前
不得发布 worker 或启动 CUDA。若任一尚未启动 lifecycle 跨到新的真实日期，必须先追加
identity-only date-transition amendment 并 supersede 旧 identity；不得创建、复用或覆盖旧
run directory。Scope、single manipulated factor、training budget、random-init provenance、
internal/native gates 和所有 forbidden items 完全继承 D2-AD0 plan-only preregistration。
D2-AD1、checkpoint selection、consistency、HSIPrior 与 Mixer 仍未授权。

#### 2026-07-27 Phase 1B D2-AD0 implementation pre-CUDA verification

在进入正式 authority CPU lifecycle 前，D2-AD0 implementation 已完成并保持
single-factor scope。提交内容包括 local-frame full-mesh builder、D2-AD architecture/config、
training/sampler wiring、fixed internal/native wrappers、CPU tests、registry binding 与本
dated amendment；没有启动 worker publication、CUDA、optimizer、training、checkpoint
load/write、internal/native evaluation 或 selection。

Authority verification 使用指定 `infbagel` Python 完成：

- targeted D2-AD + D2-AC：42 tests passed；
- full CPU suite：352 tests passed；
- `tools/experiment.py validate`：192 registry records、2 splits、2 evaluators、
  1 training protocol valid；
- `py_compile`、`git diff --check` 与 internal/native `--resolve-only` 均通过；
- read-only CPU contract prototype 通过 sealed D2-O 64-sequence × 3-window cohort
  （selection SHA-256
  `1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`）：13/13 object classes、six common-yaw
  checks、query workers 1/3/all、repeated/batch-order determinism、training/evaluator
  parity、generated-history recomputation、parameter/gradient/provenance/static gates；
  local-BPS 与 `[B,16,10]` feature common-yaw max abs 分别为
  `1.1920928955078125e-7` 与 `2.384185791015625e-7`。

该 prototype 未使用 official test、未创建 checkpoint/optimizer、未写入 lifecycle
artifact，也不替代后续正式 CPU manifest。任何正式 GPU workload 仍须在 committed clean
object、same-context preflight 与 registered smoke 通过后才可开始。

#### 2026-07-28 Phase 1B D2-AD0 unstarted lifecycle date-transition amendment

真实日期已跨至 `2026-07-28`。07-27 implementation identity
`p1-hoi-d2ad-local-frame-interaction-adapter-implementation-s42-20260727`
继续记录实际 implementation-start，不改名也不覆盖。跨日前尚未启动的 authority CPU、
GPU smoke、formal training、internal 与 native lifecycle 均未创建 manifest、run directory
或 workload，因此按既定 date-transition rule 将以下旧 identity 标为 superseded：

- `p1-hoi-d2ad-cpu-contract-s42-20260727`；
- `p1-hoi-d2ad-gpu-smoke-s42-20260727`；
- `p1-hoi-d2ad-local-frame-interaction-adapter-s42-20260727`；
- `p1-hoi-d2ad-local-frame-interaction-adapter-internal-s42-20260727`；
- `p1-hoi-d2ad-native-eval-s42-20260727`。

后续唯一有效且此前未使用的 lifecycle identities 为：

- authority CPU contract：
  `p1-hoi-d2ad-cpu-contract-s42-20260728`；
- registered GPU smoke：
  `p1-hoi-d2ad-gpu-smoke-s42-20260728`；
- formal from-random training：
  `p1-hoi-d2ad-local-frame-interaction-adapter-s42-20260728`；
- fixed internal diagnostic：
  `p1-hoi-d2ad-local-frame-interaction-adapter-internal-s42-20260728`；
- fixed native evaluation：
  `p1-hoi-d2ad-native-eval-s42-20260728`。

本 amendment 只改变未启动 lifecycle 的日期 identity。D2-AD0 的 single manipulated
factor、source implementation、random initialization、training budget、loss/optimizer、
internal/native evaluator、gates、classification precedence、artifact contract 与全部
forbidden items 均不变；不授权 D2-AD1、checkpoint selection、consistency、HSIPrior、
Mixer 或任何 sweep。

#### 2026-07-28 Phase 1B D2-AD0 completion record

D2-AD0 的全部已批准 lifecycle 已完成。Tracked compact result 为
`experiments/results/p1_hoi_phase1b_d2ad_local_frame_interaction_adapter_s42_20260728.json`，
phase summary 为 `docs/phase_summaries/PHASE_1B_D2AD.md`。本 record 只封存固定
coordinate-contract repair、运维失败、科学结果与 artifact；不新增 fallback 或研究方向。

1. **CPU、smoke 与 formal training。** Authority CPU contract
   `p1-hoi-d2ad-cpu-contract-s42-20260728` 以 352 tests、exact
   parameter/API/base-parity、coordinate equivariance、query-worker/dataset/evaluator
   parity、activated gradients、provenance、HSIPrior/Mixer independence 和 static scan
   全部通过。Registered smoke `p1-hoi-d2ad-gpu-smoke-s42-20260728` 在
   `cuda:0`、real-data batch 8、timesteps `0/249/499` 上通过，peak
   allocated/reserved/headroom 为
   `252,609,024 / 304,087,040 / 24,991,956,992` bytes，且没有 optimizer 或
   checkpoint activity。Formal training
   `p1-hoi-d2ad-local-frame-interaction-adapter-s42-20260728` 从随机初始化完成
   `61,440,000` windows / `983,040,000` frames / `30,000` updates，wall time
   `47,890.633 s`、throughput `1,282.923 windows/s`；20 cadence checkpoints 与
   80 rank RNG sidecars 完整。Learned alpha/gate 为
   `0.10238598 / 0.10202970`，fixed final-online SHA-256 为
   `f527d970243a42a1534b8db4437cd09dbc25334c832c3a13eb011f81db101c06`。
2. **Fixed internal diagnostic。** 原 internal identity 因 preflight 接收到错误的
   CHOIS asset directory 而在 manifest/workload 前停止，1-file failure tree
   `d0eda6ede4e692acb2ca52ed8286ba4e122b0fc1e4edc2845946d03714898a47`
   原样保留。Corrected retry
   `p1-hoi-d2ad-local-frame-interaction-adapter-internal-r1-s42-20260728`
   从头完成 sealed 64×3 cohort 的三路 paired 500-step rollout。Full minus
   gate-ablated direct-hand union 5-cm F1 为 `+0.6274720`，95% CI
   `[0.5403343,0.7116109]`；gate-ablated minus full GT-contact distance 为
   `+97.93614 cm`，CI `[87.31619,108.58261]`，证明 adapter 被使用。Full minus
   locality-permuted F1 为 `+0.0135183`，CI
   `[-0.0062331,0.0342150]`；permuted minus full distance 为
   `+0.143274 cm`，CI `[-0.057880,0.354987]`，两个 locality gate 均失败。
3. **Fixed official native evaluation。**
   `p1-hoi-d2ad-native-eval-s42-20260728` 完成 official 438×3、500-step
   unguided evaluator，复用 sealed D2-X control，未重生成 control，D2-AC 只作 sealed
   descriptive comparison。D2-AD end-object/Txy/FS/contact
   precision/recall/F1/coverage/Pbody/hand penetration/MPJPE/Troot/Tobj/Oobj 为
   `4.2373 / 4.8036 / 0.42539 / 0.76795 / 0.53300 / 0.58687 /
   0.43497 / 3.4625 / 0.21656 / 12.3847 / 9.2747 / 16.4076 / 1.01478`。
   相对 D2-X，contact F1/recall differences 为
   `-0.0505537 / -0.0614576`，95% CI 分别为
   `[-0.0713216,-0.0293031] / [-0.0852768,-0.0377657]`；released contact-F1
   gap closure 为 `-0.562760`。End-object/Txy/FS/Troot protection ratio CI upper
   为 `1.19830 / 1.23695 / 1.25273 / 1.16017`，均超过 `1.10`；
   precision difference CI lower `-0.0418402 < -0.02`。Native transfer、
   protection 和 released-95% effectiveness gates 全部失败；181-sequence
   penetration mask contract 通过。Evaluator 未生成 FID、Matching、R-Precision 或
   Diversity，未代填。
4. **Final decision 与 recovery。** Classification precedence 由固定 internal
   locality failure 决定，最终为
   `local-frame-interaction-adapter-locality-negative-stop`。CPU/smoke/training/
   internal/native recovered tree SHA-256 分别为
   `514163cc45801253f19dbb6e1789464e791f59a00aa6f1b44cdadf9f348eb7ce` /
   `85ef57f3874ab113d4cac75b813259fb61ae5cff5d1b24ed9078b924223c621a` /
   `d694962309735ecae12f4480d4dcb52c8d191a9a453603fefd8e5f4bbd18b656` /
   `4b80a78745de4d3fecc23399f023d736d4b5ff1f9e7d12e043e70e6bf27055e3` /
   `6d0bcf47eac49aaf1a10341d81bc8d4f1a518ed86344fd145283b17c236c7d0c`，
   worker/authority 一致。

D2-AD0 fixed final-online checkpoint 不可选择、不可 resume、不可初始化后续 prior。
本 D2-AD0 计划没有 D2-AD1/longer-budget fallback；任何新机制、预算或参数方向都必须先有
新的 dated plan、append-only registry hypothesis 和用户明确授权。不得自动启动
checkpoint selection、consistency、HSIPrior、Mixer 或任何 sweep。

#### 2026-07-28 Phase 1B D2-AE0 GPU-native sparse current-state role-relative object-field routing 预注册（plan-only）

本 amendment 只注册用户明确授权的一个新单变量 HOIPrior 实验，不重新开放 HOIPrior
search。修改前的 authority 为 clean `phase/01b-hoi`、HEAD
`45b59330f6d09da9050cedb01e5edb7fa5deefda`（`Close Phase 1B D2-AD0`）。Identifier audit
对 tracked/untracked operational text、全部 reachable Git history/diffs/refs/reflogs、registry、
authority/worker staging 文件名和现有 lifecycle identities 做了大小写不敏感扫描，确认
D2-AE、`d2ae`、`D2-AE0`、`p1-hoi-d2ae-*` 与 `sparse-relation-field` 尚未使用。Locked baseline
`b9a158f75ab0740c91c9cfc8863a65fa381b014c` 是 HEAD ancestor；
`feature/independent-hoi-hsi-priors` 既不是 ancestor，也没有 patch-equivalent cherry-pick。

1. **封存证据与可证伪假设。** D2-AC0/D2-AD0 的 adapter whole-gate ablation 很强，但
   correspondence permutation 基本无效，且 native contact transfer 变差；D2-AD0 formal
   throughput 还因 CPU full-mesh KD-tree path 从 D2-X 的
   `3,243.0357134915853 windows/s` 降到 `1,282.923 windows/s`。作者 released InfBaGel 的
   `occ_temp` 路径说明 current-state spatial relation 配合固定 temporal routing 可能是有用
   归纳偏置，但其 training 使用 noisy object pose 与 clean future `x_start` human/grid anchor，
   sampling 又使用 current `x` object pose 与 previous `x0` human anchor，并读取 synthesized
   `Scene*`。D2-AE0 的唯一假设是：若每个 diffusion step 只从同一当前 `x_t` 构造结构性绑定
   left hand/right hand/pelvis 与当前 sparse object surface 的明确相对场，并在 trunk 前按固定
   temporal segments 写回，则 relation path 的 temporal correspondence 与 role identity 会在
   paired causal diagnostic 中成为必要信息，并在不引入 scene/future leakage 或 CPU dynamic
   geometry 的情况下改善 D2-X native contact transfer。

2. **唯一 manipulated factor 与保持项。** D2-AE0 相对 sealed D2-X 只增加一个
   GPU-native sparse current-state role-relative object-field residual。保持 `[B,16,232]`
   clean-output API、232-D field semantics、16-frame window、2-frame history restoration、
   500-step clean-x0 diffusion、512-wide/16-head/8-layer trunk、原四个 condition tokens、
   global BPS token、D2-X FK-foot temporal routing、全部既有 losses/reductions/weights、
   optimizer/LR/batch/split/budget/sampler、official evaluator，以及 HSIPrior/Mixer clean-output
   contract 不变。不得扩大 D2-AC/D2-AD adapter，也不得改变 point count、width、depth、role、
   placement、anchor、batch、LR、loss、threshold 或训练预算。

3. **Current-state relation source 与 immutable sparse asset。** Relation builder 只接收当前
   diffusion state `x_t [B,16,232]`、现有 `rest_object_points [B,100,3]`、
   `world_to_local_rotation [B,3,3]`、`object_rotation_reference [B,3,3]`，以及 locked
   position/object normalization tensors。D2-X data path 对每个 immutable rest mesh 使用
   `trimesh.load_mesh(process=False)`、float32 Z-up→Y-up，再以
   `linspace(0,N-1,100).round()` 选择固定 vertices；13 个对象在 real D2-X batch 中均为
   100 points，并与该重建 byte-exact。以下 canonical hashes 在本 plan 中锁定：

   - object-name mapping（`sequence-name-second-underscore-field-v1`）：
     `1af35119c1dd54e2ad44c99f3cb91b62c1b88f62ca80cddcc96f4b201ffe0f5b`；
   - per-object source/count/index/point manifest
     （`d2x-rest-object-points-100-yup-linspace-vertex-v1`）：
     `e88d74a7ee434f3e6320c95d1ebb74efdc8fe4740b70ff596e502666a096f7a7`；
   - stacked tensor `[13,100,3]` in fixed object-name order：
     `793dad6a805d0a908087b273590bf171e7bce4c026297cf94d40f8c651fe4cab`。

   Training 已直接提供这 100 points；native evaluator 已加载同一 immutable rest meshes，
   sampler 只允许在 diffusion loop 前以相同固定 indices 建立/缓存 13 个 100-point tensors，
   随后每个 batch 只传 `[B,100,3]`。这不是 full-mesh nearest query、per-window relation cache
   或 dynamic CPU geometry。Relation math 本身必须是 train/sample 共用的 pure PyTorch
   function；不得使用 SciPy、NumPy、trimesh、KD-tree、full-mesh `cdist`、dense occupancy、
   stored future relation 或 collator-side dynamic geometry。

4. **精确几何方程。** 固定 temporal anchors `F=(0,5,10,15)`；固定 roles 按顺序为
   `(left_hand_direct_joint_24,right_hand_direct_joint_26,pelvis_joint_0)`。对每个 anchor
   `tau`，从当前 `x_t` 反归一化 28 个 local joints `J_tau` 与 object translation `o_tau`，
   并计算

   `R_local_tau = world_to_local_rotation @ project_to_so3(R_relative_tau @ object_rotation_reference)`。

   当前 sparse object surface 为
   `S_tau,n = P_rest,n @ R_local_tau^T + o_tau`。实现顺序必须与现有
   `hoi_training_losses()` object-surface transform 逐点 parity：先投影 relative，随后
   `project_to_so3(relative @ reference)`，再左乘 `world_to_local`，最后
   `einsum("bpc,btdc->btpd") + translation`。Role anchor 分别为
   `J_tau,24/J_tau,26/J_tau,0`；每点 feature 固定为
   `[delta_x,delta_y,delta_z,||delta||_2]`，其中 `delta=S-h`。不得加入 contact label、
   SDF、penetration label、scene/category embedding 或任何 threshold。

5. **固定 sparse field encoder。** 所有 temporal/role sets 共享 point encoder
   `phi: 4 -> 128 -> 128`，每层后使用 SiLU。每个 role 只做 point-set mean/max pooling，
   得到 256-D；三个 role 按 left/right/pelvis 顺序结构性拼接为 768-D。Relation vector 为
   `r_tau = LN(W_r g_tau + e_tau)`，其中 `W_r:768->512`，`e_tau` 是四个 learned
   temporal-slot embeddings。不得复用 D2-AC 的“同一 motion token 加 additive role
   embedding”设计。Point order permutation invariance 是预期性质，不得作为 locality
   ablation。

6. **固定 temporal writeback 与参数预算。** 保持原 20-token sequence，不插入 occupancy
   tokens。先算 `H_t=motion_input(x_t)`，再以固定 mapping
   `0..4->0, 5..9->5, 10..14->10, 15->15` 写回
   `H'_t = H_t + tanh(alpha) * r_a(t)`。`alpha` 为单 scalar 且严格初始化为 0；writeback
   位于 condition concat/position embedding 与完整 8-layer trunk 之前。`alpha=0` 时必须与
   共享 D2-X trunk `eval()` max-abs parity `<=1e-6`。所有 sparse-field 和 trunk 参数均由
   seed 42 随机初始化；不得加载 released/author/D2-V/X/Y/Z/AB/AC/AD/prior/EMA/
   consistency checkpoint。Exact parameter contract 为：point encoder `17,152`、projection
   `393,728`、temporal embeddings `2,048`、LayerNorm `1,024`、alpha `1`，increment
   `413,953`；base `29,673,448`；total `30,087,401`；increase `1.3950283%`，硬上限
   `1.50%`。若 CPU 实测不一致或超限，GPU 前分类
   `sparse-relation-field-contract-failure-stop`，不得改 width。

7. **独立 architecture/provenance 与 train/sample symmetry。** D2-AE 使用独立
   architecture variant `d2ae_sparse_relation_field` 与独立 checkpoint contract；released、
   D2-X/base、D2-AC、D2-AD schema 必须 fail-closed。Training 的 relation 只从
   `GaussianDiffusion.q_sample()` 产生的 current `noisy` 构造，绝不读取 clean target。
   Sampling 的每个 500-step model call 只从当步 current `x_t` 构造同一 relation，不得使用
   previous predicted clean `x0` 作为专有 condition。两条路径共享同一 builder、normalization、
   100-point tensor 和 frame/reference contract。HSIPrior 不接受该 variant、不共享参数或
   storage；未来 Mixer 仍只接收 clean `[B,16,232]`。

8. **Authority CPU hard gate。** 任何 GPU workload 前，registered authority CPU lifecycle
   必须完成并归档：path/branch/commit/clean/date；identifier/provenance；100-point
   asset/mapping/tensor hashes；loss surface parity；common global-yaw invariance；relative
   translation/rotation sensitivity；left/right swap 精确 block exchange；nonzero-gate temporal
   anchor permutation sensitivity；point-order invariance；zero/constant/extreme noisy-state 与
   SO(3) finiteness；dtype/device/batch propagation；exact parameter/API/base parity；initial
   alpha finite/nonzero gradient；test-only `tanh(alpha)=0.1` 下 point encoder/projection/
   temporal embeddings/relevant trunk gradients finite/nonzero；probe 不保存且不进入训练；
   train/sampler builder parity；relation source 无 clean/future/Scene/contact；checkpoint
   rejection；HSIPrior independence；Mixer clean API；forbidden-path static scan；full authority
   suite、registry validation 和 `git diff --check`。任一失败立即停止为
   `sparse-relation-field-contract-failure-stop`。

9. **固定 lifecycle identities。** 本 plan-only commit 不实施 source、不启动 CPU/GPU
   workload。Implementation commit 后只允许绑定以下本日未使用 IDs；跨日或 unstarted
   preflight failure 必须 append-only supersede/`-r1`，不得覆盖：

   - plan：`p1-hoi-d2ae-sparse-relation-field-preregister-s42-20260728`；
   - implementation：`p1-hoi-d2ae-sparse-relation-field-implementation-s42-20260728`；
   - CPU：`p1-hoi-d2ae-cpu-contract-s42-20260728`；
   - functional smoke：`p1-hoi-d2ae-gpu-functional-smoke-s42-20260728`；
   - performance：`p1-hoi-d2ae-performance-benchmark-s42-20260728`；
   - formal：`p1-hoi-d2ae-sparse-relation-field-s42-20260728`；
   - internal：`p1-hoi-d2ae-sparse-relation-field-internal-s42-20260728`；
   - native：`p1-hoi-d2ae-native-eval-s42-20260728`；
   - completion：`p1-hoi-d2ae-completion-s42-20260728`。

10. **Single-GPU functional smoke。** 在 exact committed clean worker object 上，以 verified
    worker Python、`INFBAGEL_WORKER_EXPERT=hoi`、real-data batch 8、timesteps `0/249/499`、
    seed 42、random initialization 执行。不得创建 optimizer、update 或 checkpoint；必须记录
    relation values/shapes、alpha gradient、test-only activated gradients、loss/model finiteness、
    peak allocated/reserved/headroom、visible GPUs、resolved config、same-context preflight 与
    manifest。失败按 contract failure 停止。

11. **4-GPU full-micro-batch performance hard gate。** Formal training 前必须在
    `infbagel-4gpu/node01`、4×RTX 3090、per-GPU batch 512/effective 2048、FP32 Adam、seed 42、
    random initialization 上完成独立 sacrificial benchmark：64 warm-up + 256 measured = 320
    updates；measured windows `524,288`；不加载/保存 checkpoint，benchmark weights 不复用。
    CUDA timing必须同步，分别报告 loader wait、H2D、GPU relation build、forward、backward、
    optimizer、DDP、peak allocated/reserved/headroom、CPU/GPU utilization、contention 和
    intermediate shapes。Measured throughput 必须
    `>=2756.580356467847 windows/s`（sealed D2-X 的 85%），完整预算 ETA 必须
    `<=6.20 h`；每卡 headroom 必须 `>=max(2 GiB,10% device memory)`，loss/gradients finite，
    且无 CPU dynamic geometry。失败即
    `sparse-relation-field-performance-negative-stop`；保留全部 benchmark artifacts，不得通过
    point/width/depth/role/anchor/placement/batch/loss/budget 或 workers/threads sweep 重试。

12. **Formal from-random training（仅 performance pass）。** 固定 split
    `experiments/splits/omomo_hoi_train_validation_seed42.json`，SHA-256
    `019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e`；只在
    `infbagel-4gpu/node01` 4×RTX 3090 运行。Per-GPU batch 512、effective 2048、accumulation 1；
    `61,440,000` windows / `983,040,000` frames / `30,000` updates；FP32 Adam、LR `1e-4`、
    betas `(0.9,0.999)`、weight decay 0、no warmup/scheduler/AMP/clipping/EMA；primary
    final-online；FK/object-surface/velocity/terminal-goal weights
    `0.3569973401779424/0.4772322188400037/0.1/1.0`；D2-X FK-foot routing on，D2-AB/new
    losses off。First start 的 init/weight-init/resume 全空，所有旧 model/optimizer/RNG/EMA/
    scaler/scheduler load count 为 0。必须完整跑完预算，不得选择中间 checkpoint。通过 initial
    stability、memory headroom、finite required gradients 和 resumable checkpoint 后记录实测
    throughput/ETA/checkpoint hash，并按 multi-server policy 让 worker-owned persistent session
    独立完成；不得因 control tunnel 中断 restart、复用 run id 或覆盖。

13. **Fixed internal causal diagnostic。** 只加载 fixed final-online，在 sealed D2-O
    64 sequences×3 windows、phase offsets `(14,56,98)`、selection SHA-256
    `1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a` 上运行四条 paired
    500-step rollouts：`full`；每步 gate 强制 0 的 `relation_gate_ablated`；geometry anchor
    block `k<-(k+2) mod 4` 但 target temporal embedding/routing slot 不变的
    `temporal_correspondence_permuted`；projection 前交换 left/right pooled geometry blocks 的
    `left_right_role_swapped`。四路共享 initial latent、每步 posterior noise、condition、ordering
    与 history restoration；official test 禁止，无 optimizer/update/checkpoint write/selection。
    统计 unit 为 sequence、seed 42、10,000 paired bootstrap。必须报告 semantic/direct/FK-palm
    contact、多阈值、coverage/run length、GT-contact distance、penetration、MPJPE、object/pelvis
    goal、FS、alpha/gate、temporal/role block norm/variance/permutation sensitivity 与 paired
    uncertainty。Primary gates 固定为：full-gate-ablated direct union 5-cm F1 CI lower `>0`；
    full-temporal-permuted 同指标 CI lower `>0`；full-role-swapped direct left/right macro-F1
    CI lower `>0`；gate-ablated-full 与 temporal-permuted-full GT-contact mean distance CI lower
    均 `>0`。Classification precedence 依次为 unused、temporal negative、role negative；无论
    internal 正负都继续一次 fixed native evaluation。

14. **Fixed native evaluation 与 selection gates。** 完全复用 D2-AC/D2-AD protocol：official
    438 sequences×3 windows、500-step unguided production diffusion、final-online、seed 42、
    10,000 paired sequence bootstrap；CFG/guidance/scene/dynamic perception/consistency 全 off；
    不重新生成 sealed D2-X。Control checkpoint/aggregate/per-sequence SHA-256 分别为
    `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51` /
    `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
    `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`；released aggregate
    `76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6`；penetration 使用
    sealed 181-sequence finite mask SHA-256
    `2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec`。
    Native transfer 要求 contact F1/recall difference CI lower `>0` 且 released gap closure
    `>=25%`（F1 point estimate 约 `>=0.6598838781`）。Protection 继续要求 end-object/Txy/FS/
    Pbody/hand penetration/MPJPE/Troot/Tobj/Oobj paired mean-ratio CI upper `<=1.10`，contact
    precision difference CI lower `>=-0.02`，penetration mask contract 通过；released-95%
    effectiveness gate不变。D2-AC/D2-AD 只作 sealed descriptive evidence；FID/R-Precision
    即使生成也不参与 selection；evaluator 生成什么就保留什么，不删除、补值或改 metric math。

15. **最终分类、artifact 与停止边界。** Classification precedence 固定为：
    `sparse-relation-field-contract-failure-stop`；
    `sparse-relation-field-performance-negative-stop`；
    `sparse-relation-field-unused-optimization-negative-stop`；
    `sparse-relation-field-temporal-routing-negative-stop`；
    `sparse-relation-field-role-binding-negative-stop`；
    `sparse-relation-field-transfer-negative-stop`；
    `sparse-relation-field-conflict-negative-stop`；
    `sparse-relation-field-positive-but-not-effective-stop`；
    `sparse-relation-field-positive-candidate-stop`。只有最后一类可把 fixed final-online 标为
    selectable autonomous diffusion HOIPrior candidate。所有 operational/scientific failures、
    resolved configs、same-context preflights、manifests/logs/profile、checkpoints/RNG、paired noise、
    internal/native raw/summary/bootstrap、mask、optional evaluator outputs、run-local registry、
    dependency/hardware/data/evaluator hashes 与 recovered trees 必须保留并做 worker/authority
    unified `sha256_path` 核验；worker 发起 non-destructive rsync，禁止 `--delete`。最终写 compact
    result、`docs/phase_summaries/PHASE_1B_D2AE.md` 与 append-only completion record。
    D2-AE1、longer budget、任何 sweep、D2-AC/D2-AD retrain/resume/selection、new loss、SNR/
    timestep weighting、gradient projection、rollout exposure、CFG/guidance、distillation、
    HSIPrior、Mixer、scene encoder/Scene*/occupancy、future clean/GT/stored relation 均未授权，
    不得自动启动。

#### 2026-07-28 Phase 1B D2-AE0 implementation / pre-GPU lifecycle binding amendment

本 amendment 实现且只实现上节 plan-only 已锁定的 D2-AE0 机制，并在任何 reportable CPU/GPU
workload 前封存 source、config、tests 与 lifecycle hard binding。Implementation source head 为
`eded185f7e5ba075ba83fde97282cb1464ddb08f`（`Preregister D2-AE sparse relation routing`）；
截至本记录，authority 未创建 optimizer、未加载/写入 checkpoint，worker publication、CPU
contract、functional smoke、performance benchmark、formal training、internal 与 native 均未启动。

1. **实现边界。** 新 architecture variant `d2ae_sparse_relation_field` 使用 train/sample 共用的
   pure-PyTorch builder，只从当前 `x_t`、现有 immutable `[B,100,3]` rest-object points、
   history-derived window/reference rotations 与 locked normalization 构造
   `[B,4,3,100,4]` role-relative point field。共享 `4->128->128` encoder、mean/max pooling、
   fixed left/right/pelvis concatenation、`768->512` projection、four temporal embeddings、LN 与
   zero-init scalar ReZero gate 按 `0/5/10/15` segments 在全部八层 trunk 前写回。普通
   `PriorWindowDataset` 保持不变；没有 D2-AD collator、SciPy/KD-tree/full-mesh query、CPU dynamic
   geometry、dense occupancy、Scene、contact/clean/future/stored relation 或 evaluator change。
2. **参数、API 与 provenance。** CPU recomputation 锁定 base/increment/total 为
   `29,673,448 / 413,953 / 30,087,401`，increase `1.3950283% <= 1.50%`；输出仍为
   `[B,16,232]`。Seed-42 shared D2-X state 共 119 keys byte-exact，只有 10 个 sparse-field keys
   新增，alpha 初始精确为 0。Released、D2-X、D2-AC、D2-AD schemas 均 fail-closed；resume 还必须
   将 checkpoint 自报的 random/no-source/no-old-state provenance 与当前 fresh seed-42 D2-AE
   initial state hash 精确匹配。HSIPrior 参数/storage independence 与 Mixer clean-output API 不变。
3. **Train/sample 与 causal diagnostic binding。** Training relation 只从 `q_sample()` 返回的
   current noisy state 建立；500-step sampler 每步只用当步 current state，并通过真实
   `HOIPriorSampler` metadata reconstruction 与 real `PriorWindowDataset` window 对七项 metadata、
   surface/features 做 exact parity。Internal runner 固定四路、共享 initial latent 与 499 次
   posterior draws；首窗 exogenous condition/history 共享，分叉后 frame/BPS/local-goal/relation
   metadata 保持各 path-local。Sealed `(14,56,98)` cohort runtime proof 锁定 source starts
   `(0,42,84)`，前窗 sampled tail `[start+42,start+45]` 精确成为下一窗 history。
4. **Performance hard gate 现在 fail-closed。** Registered 4-GPU benchmark 保持
   `4x512`、64 warm-up、256 measured、FP32 Adam，并记录 loader/H2D/relation/forward/backward/
   optimizer/DDP、四 rank relation shapes/device、memory、utilization 与 compute contention。
   Benchmark CLI 必须显式接收 actual-date performance run id 与 intended formal run id（含可选
   严格 `-rN`），summary 中的 `formal_run_id` 必须与 formal config `run_id` 精确一致且两者
   与 benchmark run id 使用同一实际日期。Formal config 新增必填 benchmark summary absolute
   path 与 SHA-256；trainer 必须验证 passing
   classification、`>=2756.580356467847 windows/s`、ETA `<=6.20 h`、headroom、finite losses/
   gradients、四 rank GPU-only relation、无外部 compute contention、零 checkpoint activity，
   并验证 benchmark commit 是 current commit ancestor 且 benchmark/formal tracked runtime source-tree
   hash 完全一致。缺失、tamper 或任何 gate failure 都在 optimizer/GPU training 前拒绝。
5. **Lifecycle identity 与 verification。** CPU/smoke/performance/internal/native IDs 使用 locked stem、
   actual start date 和可选严格 `-rN`；fresh formal start 也要求 actual date，same-run resume 则保留
   checkpoint-bound 原 run id，允许跨午夜而不伪造新 identity。Authority 已通过 D2-AE targeted
   `26/26`、D2-AC/D2-AD/independent/remediation/D2-T/D2-U regressions `115/115`、full suite
   `378/378`（authority 未启用 worker-only LINGO skip）、`py_compile`、registry validation
   （implementation record 前 200 records；包含该 record 后 201 records）与
   `git diff --check`。HOI worker 环境预期在同一 378 项中 skip 2 个 real-LINGO-only tests；
   Official evaluator 与 locked
   metric/helper sources未修改。

下一步只能先提交本 logical implementation，使 authority clean；随后以
`p1-hoi-d2ae-cpu-contract[-rN]-s42-<actual-date>` 注册并执行 authority CPU hard gate。只有 CPU、
single-GPU functional smoke 和 hash-bound 4-GPU performance gate 全部通过，formal training 才能
启动；performance negative 时必须立即按已注册分类停止，不得 sweep 或修改机制。

#### 2026-07-29 Phase 1B D2-AE0 unstarted lifecycle date-transition amendment

真实日期已跨至 `2026-07-29`。Authority 在追加本 amendment 前为 clean
`phase/01b-hoi@993934cb1d27a2fb406b4d3640eda90d8737767a`。已经启动或完成的
2026-07-28 lifecycle identities 全部保持不变，特别是 formal retry
`p1-hoi-d2ae-sparse-relation-field-r1-s42-20260728`：它于
`2026-07-28T19:43:17+08:00` 从随机初始化启动，于
`2026-07-29T00:49:32+08:00` 在同一 run id 下完成，exit code `0`，没有 resume、重训或
checkpoint selection。Formal 完成 `61,440,000` windows / `983,040,000` frames /
`30,000` updates，final-online SHA-256 为
`b7d49046504e9f8367bfd2bce0aeefb1c8590bf9c542b6eed637f05bdfcdd840`。

完整 formal tree 已由 worker 发起 non-destructive recovery 到
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ae-sparse-relation-field-r1-s42-20260728-recovery-r1`；
worker/authority 统一 `sha256_path` 均为
`3c8a987d54dfb63e89d7ec243fb065dc4f84c95808d92eee13b46ab621959428`
（119 files / 7,226,999,632 bytes），随后 checksum dry-run 为零传输。较早的
118-file pre-run-local-registry snapshot
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ae-sparse-relation-field-r1-s42-20260728`
也原样保留，tree SHA-256
`420e2f89d8059e4d9b5d0249001fbb9dbaffd5e591990f8ba7d6fbcdf6e44ae6`；
不得删除或覆盖任一 recovery evidence。

以下已经实际发生的 identities 继续作为唯一历史记录，不得改名、复用或 supersede：

- plan `p1-hoi-d2ae-sparse-relation-field-preregister-s42-20260728`；
- implementation `p1-hoi-d2ae-sparse-relation-field-implementation-s42-20260728`；
- CPU `p1-hoi-d2ae-cpu-contract-s42-20260728`；
- failed functional smoke base 与 completed retry
  `p1-hoi-d2ae-gpu-functional-smoke[-r1]-s42-20260728`；
- completed performance base
  `p1-hoi-d2ae-performance-benchmark-s42-20260728`；
- failed formal base
  `p1-hoi-d2ae-sparse-relation-field-s42-20260728`；
- completed performance retry
  `p1-hoi-d2ae-performance-benchmark-r1-s42-20260728`；
- completed formal retry
  `p1-hoi-d2ae-sparse-relation-field-r1-s42-20260728`。

跨日前从未创建 manifest、run directory 或 workload 的旧 internal、native 和 completion
identities 现被永久 supersede：

- `p1-hoi-d2ae-sparse-relation-field-internal-s42-20260728`；
- `p1-hoi-d2ae-native-eval-s42-20260728`；
- `p1-hoi-d2ae-completion-s42-20260728`。

后续唯一有效且 identifier audit 确认未使用的 identities 为：

- fixed internal：
  `p1-hoi-d2ae-sparse-relation-field-internal-s42-20260729`；
- fixed native：
  `p1-hoi-d2ae-native-eval-s42-20260729`；
- completion：
  `p1-hoi-d2ae-completion-s42-20260729`。

本 amendment 只改变尚未启动 lifecycle 的日期 identity。D2-AE0 mechanism、final-online
checkpoint、sealed cohort、四条 internal paths、500-step sampler、native evaluator、bootstrap、
threshold、gate、classification precedence 与 artifact contract 全部不变；不授权新的
performance benchmark、fresh formal training、resume、checkpoint selection、D2-AE1、
longer budget、sweep、consistency、HSIPrior 或 Mixer。

#### 2026-07-29 Phase 1B D2-AE0 fixed internal causal diagnostic completion

Fixed internal lifecycle 在 clean worker `phase/01b-hoi@190d95d1c634299407b398946b2a01d5737b45d7`
上执行。Base identity
`p1-hoi-d2ae-sparse-relation-field-internal-s42-20260729` 与 retry `-r1`
均在 manifest 和 GPU workload 前停止：两次 preflight 均确认四卡显存、利用率、compute
process、Git、Python、数据、CHOIS checkout/checkpoint 与 NTP contract 正常，但单次快照分别
观察到 GPU 1 和 GPU 3 为瞬时 `P5`，利用率仍为 `0%`、compute process 为空。两个目录不覆盖、
不复用，分别以 2 files / 10,311 bytes /
`015e180d5aa21f093fe7f712d576150f12d47203aac26269f28f56c0015336e3` 和
2 files / 10,320 bytes /
`88f20c8ba3f0c013ba475e04551706ce2194c1904d33db2738dde497175de8bd`
原样保留。随后 20 次一秒间隔只读采样均为四卡 `P8`、`0%`，未改变 idle 判据；`-r2`
从头生成 resolved config 和 preflight，未复用任何 partial output。

成功 run
`p1-hoi-d2ae-sparse-relation-field-internal-r2-s42-20260729` 只加载 fixed final-online
checkpoint SHA-256
`b7d49046504e9f8367bfd2bce0aeefb1c8590bf9c542b6eed637f05bdfcdd840`，
在 sealed D2-O 64-sequence / 192-window cohort
`1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`
上完成四条 paired 500-step causal rollout。Exit code 为 `0`，runtime
`332.670974 s`；29 项 runtime contract 全真，四路径各 64 sequences、24 causal
batch-windows、每 window 500 次 relation forward，paired noise/exogenous condition/initial
history、57-sequence GT-contact finite mask、causal overlap、history restoration、model-state
unchanged 与 GPU current-state relation capture 全部通过。未创建 optimizer，training update、
checkpoint write、checkpoint selection 与 official-test use 均为零。

五个 primary mechanism gates 全部通过：

| comparison | paired point | sequence-bootstrap 95% CI |
|---|---:|---:|
| full − gate-ablated direct-hand union 5-cm F1 | +0.236691 | [0.148411, 0.326983] |
| full − temporal-permuted direct-hand union 5-cm F1 | +0.153893 | [0.081493, 0.226123] |
| full − left/right-swapped direct-hand macro-F1 | +0.178708 | [0.122784, 0.232256] |
| gate-ablated − full GT-contact-frame distance (cm) | +3.509101 | [2.090270, 4.957889] |
| temporal-permuted − full GT-contact-frame distance (cm) | +4.010482 | [2.072867, 6.222641] |

因此 internal classification 为
`sparse-relation-field-internal-positive-continue`：relation path 被使用，固定 temporal
correspondence 与结构性 left/right role binding 均有正 causal evidence。Learned
`alpha=-0.1493664682`、`tanh(alpha)=-0.1482654959`。Full frame-aggregate direct-hand
union 5-cm F1 为 `0.778771`，MPJPE `11.985653 cm`、object goal `94.125465 cm`、
pelvis goal `5.238491 cm`、FS `0.799586`；这些 internal descriptive values 不替代 sealed
D2-X native control，也不用于 checkpoint selection。

Worker 发起 non-destructive recovery 后，成功树在两端均为 17 files /
37,798,242 bytes，tree SHA-256
`044f98f78d52347af0c3120a1a5ca4df25c5e4773256c89c2fd5e6bd77fd0b21`，
checksum dry-run 为零差异。Metrics/manifest/paired-noise/paired-conditioning/
sparse-relation-appendix SHA-256 分别为
`0d1e422386bd181e86ef5d77be80d05972bea92411cbb716644ff0a5f2811ba9` /
`811cb1be8e3383295b7d60b8d8f488a2ddafd5d9d829a652e2a5e894325c80b2` /
`1f4123945ae576b8a12ed83fa115dc32d9bea6df81b67c759f1cf482f088988c` /
`1eaa2380f26368ae8dd754c3be5452949f44492eba9f480ee71078a188434b9d` /
`a38693f5743be3b06b05097c1dc0129f6968eff0ea6e22ce70985c6df6a60815`。

下一步只允许将本 append-only record 提交并 fast-forward 到 worker，然后运行一次固定
`p1-hoi-d2ae-native-eval-s42-20260729`。Internal 正结果不授权 checkpoint selection、
D2-AE1、consistency、longer budget、任何 sweep、HSIPrior 或 Mixer。

#### 2026-07-29 Phase 1B D2-AE0 fixed native evaluation completion

Fixed native run `p1-hoi-d2ae-native-eval-s42-20260729` 在 clean worker
`phase/01b-hoi@5a167347ec4761ec8427b518a36da9157b8fe033` 上只加载 fixed final-online
checkpoint SHA-256
`b7d49046504e9f8367bfd2bce0aeefb1c8590bf9c542b6eed637f05bdfcdd840`。
协议为 official 438 sequences × 3 windows、500-step unguided production diffusion、
online/final-online weights、seed 42 与 10,000 次 paired sequence bootstrap；CFG、guidance、
scene conditioning、dynamic perception 与 consistency 均关闭。Sealed D2-X aggregate /
per-sequence SHA-256
`3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
`69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`
直接复用且未重新生成，sealed D2-X checkpoint SHA-256 为
`b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51`；
released aggregate SHA-256 为
`76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6`。
D2-AC/D2-AD 只作 sealed descriptive evidence。Run local start/end 为
`2026-07-29T01:54:02+08:00` / `2026-07-29T02:07:31+08:00`，exit status 为 completed，
runtime/end-to-end 为 `383.200603 / 375.213926 s`，55,188 frames 的 synchronized
generation 为 `71.085844 s`。未进行 optimizer update、checkpoint write、checkpoint
selection 或训练；FID、Matching、R-Precision 与 Diversity 均未生成。

Internal path/temporal/role mechanism gates 保持全通过，但 native transfer 未通过：

- D2-AE contact F1 `0.64194385`，sealed D2-X 为 `0.63742594`；paired difference
  `+0.00451791`，95% CI `[-0.01809406, 0.02684589]`；
- contact recall difference `+0.00168393`，95% CI
  `[-0.02313870, 0.02638117]`；
- contact precision difference `+0.01557008`，95% CI
  `[-0.00519356, 0.03679141]`；
- released contact-F1 gap closure `0.05029306 < 0.25`；
- target contact F1 `0.64194385 < 0.6598838781`。

Protection gate 也未通过：end-object target/control mean-ratio CI 为
`[1.08425, 1.21382]`，FS ratio CI 为 `[1.03233, 1.17347]`，其 upper bounds 均超过
`1.10`。181-sequence penetration finite-mask contract 通过，finite-mask sequence-ID
SHA-256 为
`2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec`。
Classification precedence 因而首先停在
`sparse-relation-field-transfer-negative-stop`；后续 protection 与 released-95%
failures 保留为证据但不覆盖该 classification。Checkpoint selected/selectable 均为 false。

Metrics/manifest/aggregate/per-sequence/resolved-config/resolved-target/preflight/
run-local-registry SHA-256 分别为
`55927debc01eba5a2a07484695b62aed9cb1f7c29e30d289e84e4371229d60f8` /
`419be60fc35c747c27d585270ff0f504921c79f8993004a48fbebd66b2f4d8db` /
`157acda463036bdf787618c217262c14c77a09a3f409cbeada03de06e9b902a1` /
`8533b66ea3c1fb0928b8a7581bb79c0cc14d594970314a3b7619659daddfb95c` /
`d747b549190c1e3fd8e5f91b12ae8c51db405e0a7e4495d556e94ad63fa7a378` /
`4ee5916806c3aafb600054641a0b7baaa17db8d9479c949a1d7fd4e7f7530ad8` /
`5572d7f53913e50b763da5772dfb4bb2d336bcd42f925916a960e2c64833487e` /
`8fb263138e5bc2f429630dc8e5c57b93fe4019ab818cf062a1dddaf82cc1e972`。
Worker 发起 non-destructive recovery 后，两端完整 native tree 均为 18 files /
3,474,559 bytes，SHA-256
`4f31bb8f61bd40eb4604a25a0802a970686092306faf86efa0b289c856cd34b5`，
checksum dry-run 为零差异。

Detached wrapper 的 `exit_code` 原始 bytes 为 literal `0n`，SHA-256
`3ad4ee182e21c25db763cda6359ecc441b8ea32ea4d6631c012aac7fa7d362dc`；
文件未覆盖，postflight 按 leading return code 解析为 `0`。首次 postflight verifier
错误要求 aggregate-only aliases 出现在 per-sequence rows，失败 artifact SHA-256
`05133fc6afc981ec8b28d7b3ede5c938da9110fc0146a704858045289ed50e15`
原样保留；只修 serialized schema mapping 的 append-only r1 verifier 通过，SHA-256
`c4f2f86ccc341e835fdfe6f87f11fb9ec3d7dfa5db8c1bb4d4abba073ba28d18`。
未改变或重跑任何 metric、mask、reduction、threshold、evaluator 或 native workload。

#### 2026-07-29 Phase 1B D2-AE0 completion record

Completion identity 为 `p1-hoi-d2ae-completion-s42-20260729`。D2-AE0 的
identifier/source/provenance audit、plan-only registration、implementation、
authority CPU contract、functional smoke、4-GPU full-micro-batch performance benchmark、
一次从 seed-42 random initialization 完整运行的 61,440,000-window formal training、
fixed internal causal diagnostic、fixed native evaluation、non-destructive artifact recovery
与 hash verification 均已完成。所有 operational failures、完整 formal tree、20 cadence
checkpoints、80 per-rank RNG sidecars、paired internal artifacts、native raw outputs 与
postflight schema failure/r1 correction均保留且未覆盖。

Performance gate 通过，formal throughput 为 `3,347.042 windows/s`；internal path、
temporal correspondence 与 left/right role-binding gates 全通过。但 native contact F1/
recall improvement 无显著 paired evidence、contact-F1 point/gap-closure gates失败，因此
最终 classification 锁定为
`sparse-relation-field-transfer-negative-stop`。Fixed final-online checkpoint
`b7d49046504e9f8367bfd2bce0aeefb1c8590bf9c542b6eed637f05bdfcdd840`
不可选择、不可 resume、不可初始化后续 prior。

Lifecycle tree SHA-256 完整绑定如下：

- authority CPU：
  `662cf1fa37121d24b660334fa22c5fec1d5114e980271d6e1df58aa67973fae5`；
- functional preflight failure / successful r1：
  `d2bd049d7688c8f5493c0698066f79dcfceeb90f8ff34530da4b4035db4170b5` /
  `c2eed8eef78c720db46fd4064d78bad07fb85f1462e25d113a99a69cea474259`；
- performance base / formal-bound r1：
  `0d62d1c5e1da2272c309cbd1882ebbd785897690f2f5ed02750ee87542ba59bb` /
  `b7042d965a8483afd8b1306e7a81d2a30d067f54f1094dfc8910d88fcb4882c7`；
- formal preworkload failure / complete r1 / earlier preserved snapshot：
  `620c4cd5d6361036d15e0adac58a40adb503e7196a2946c7c25ebc4cd43c0136` /
  `3c8a987d54dfb63e89d7ec243fb065dc4f84c95808d92eee13b46ab621959428` /
  `420e2f89d8059e4d9b5d0249001fbb9dbaffd5e591990f8ba7d6fbcdf6e44ae6`；
- internal base / r1 preflight failures / successful r2：
  `015e180d5aa21f093fe7f712d576150f12d47203aac26269f28f56c0015336e3` /
  `88f20c8ba3f0c013ba475e04551706ce2194c1904d33db2738dde497175de8bd` /
  `044f98f78d52347af0c3120a1a5ca4df25c5e4773256c89c2fd5e6bd77fd0b21`；
- fixed native：
  `4f31bb8f61bd40eb4604a25a0802a970686092306faf86efa0b289c856cd34b5`。

Final decision booleans 为：contract/performance/internal mechanism/path/temporal/role
均 `true`；native transfer/protection/released-95% effectiveness 均 `false`；
checkpoint selected/selectable 均 `false`。

Compact result：
`experiments/results/p1_hoi_phase1b_d2ae_sparse_relation_field_s42_20260729.json`
（SHA-256
`13311ea2cea311904225d22bb20fd88f652f32c5612a84d66c7d2b93b96a4036`）；
phase summary：
`docs/phase_summaries/PHASE_1B_D2AE.md`
（SHA-256
`54bd808a8f01c3d1d538c4c5f9f0e0932e078ef431c14b30a9acc19ee8e0c206`）。

Phase 1B D2-AE0 在此停止。未启动 D2-AE1、longer-budget extension、consistency、
任何 point/width/depth/role/placement/LR/batch/threshold sweep、D2-AC/D2-AD retrain/
resume/selection、新 loss、timestep weighting、rollout exposure、CFG、HSIPrior 或 Mixer；
也不 merge/tag。任何后续 HOIPrior direction 必须重新获得授权并先做 dated plan 与
append-only registry hypothesis。

#### 2026-07-29 Phase 1B D2-AF0 sqrt-alpha-bar current-state reliability routing 预注册（plan-only）

用户在完整审阅 D2-AE0 后，只授权最后一次 HOIPrior 方向预算；该方向结束后，无论结果
正负，Phase 1B 均停止继续搜索，下一次独立 session 从 Phase 1C HSIPrior 的 dated
preregistration 开始。本 amendment 执行前，authority checkout 为
`/data/yujinlun/InfBaGel-release`、branch `phase/01b-hoi`、HEAD
`8c4f731846645a4b0a422c6a1bd0405552b831a9`
（`Close Phase 1B D2-AE0`），worktree clean；核验时间为
`2026-07-29T15:31:12+08:00`。重新扫描 working tree、全部 Git objects/refs/reflogs、
append-only registry、authority/worker staging names 与 worker checkout 后，D2-AF、
`d2af`、`D2-AF0`、`p1-hoi-d2af-*` 和 `sqrt-alpha-bar-reliability` 均未被用作
identifier；历史 JSON SHA 中偶然出现的 `d2af` 字节子串不构成 identifier。Integration
baseline `b9a158f75ab0740c91c9cfc8863a65fa381b014c` 是当前 HEAD ancestor，禁止分支
`feature/independent-hoi-hsi-priors` 不是 ancestor。Source audit 证明下述机制可由现有
train/sample 共用 timestep API 实现，没有结构性阻塞。本 commit 只允许修改本 plan 和
registry；不得包含 source change、checkpoint load、GPU workload、训练或评测。

1. **Sealed evidence 与唯一假设。** D2-AE0 的 fixed internal diagnostic 已证明 sparse
   relation path 被使用、固定 temporal correspondence 有因果作用、left/right role binding
   是结构性的；但 native contact F1/recall 相对 D2-X 只有
   `+0.004518/+0.001684`，paired 95% CI 均跨零，contact F1 仅 `0.641944`，released
   gap closure 仅 `5.03%`，并出现 end-object 与 FS protection failure。D2-AF0 的唯一
   假设是：D2-AE 在高 diffusion noise 的 early reverse steps 对当前 `x_t` 几何给予了与
   late clean steps 相同的 residual scale，因而把不可靠 relation 当作强条件写入 trunk；
   使用同一 diffusion schedule 的 clean-signal reliability
   `sqrt(alpha_bar_d)` 衰减该 residual，可能保留 D2-AE 已证实的 relation/temporal/role
   结构，同时修复 official rollout 的 contact transfer、end-object 和 FS。不得把本
   hypothesis 扩展为 learned timestep gate、SNR/loss weighting、exposure、guidance、
   consistency 或任何新 loss。

2. **唯一 manipulated factor。** D2-AF0 相对 D2-AE0 只将 writeback 从

   \[
   H'_t=H_t+\tanh(\alpha)r_{a(t)}
   \]

   改为

   \[
   \rho(d)=\sqrt{\bar\alpha_d},\qquad
   H'_t=H_t+\rho(d)\tanh(\alpha)r_{a(t)}.
   \]

   `d` 是 model 当前收到的逐样本 diffusion timestep `[B]`；training 必须使用生成当前
   `x_t` 的同一个 `d`，sampling 必须使用 reverse loop 当前的同一个 `d`，mixed-timestep
   batch 必须逐样本 index。`d=499` 是最 noisy/最早 reverse step，`d=0` 是最 clean/最后
   reverse step。不得使用 clean target、predicted `x0`、previous `x0`、future GT、
   contact、Scene、stored relation 或 sampler-only source 计算 `rho`。`rho=1` 只允许作为
   注册的 test/internal counterfactual，不允许用于训练、checkpoint selection 或第二次
   formal run。

3. **Canonical schedule contract。** `rho` 必须直接来自与 `GaussianDiffusion` 共用的
   PyTorch-1.13.1 float32、500-step linear-beta schedule：
   `betas=torch.linspace(0.0001,0.02,500,float32)`，
   `alpha_bar=cumprod(1-betas)`。必须只有一个 project canonical schedule constructor；
   released author utility 不得改动。Canonical raw float32 tensor hashes 为：

   - beta SHA-256：
     `496ec54f35af6fe7b92417f7da8b442f31c9c0070bfdd62dbb16fefc426c8f3e`；
   - alpha-bar SHA-256：
     `55f162cebbe109c67a75b00a10a1d23ea85fb1d18df9a372a3e237df5a8f48d4`；
   - sqrt-alpha-bar SHA-256：
     `5d25c63d6618c77cc31976ee9e2c5645aa41653030fca210594a05254323b440`。

   `rho[0/100/249/400/499]` 必须分别为
   `0.9999499917030334 / 0.8995221257209778 / 0.5297974348068237 /
   0.19632703065872192 / 0.0797039046883583`，并严格单调递减。Model-side schedule
   buffer 必须是 canonical、non-persistent、无参数、无 optimizer state；checkpoint
   metadata 必须记录完整 schedule contract/hash，loader 必须在 `load_state_dict` 前验证
   独立 D2-AF architecture/provenance。D2-AE 与 D2-AF learned `state_dict` schema 可以
   相同，但 D2-AE/released/base/D2-AC/D2-AD checkpoint 即使 tensor shapes 相容也必须被
   D2-AF loader 拒绝；D2-AE loader 也必须反向拒绝 D2-AF。

4. **全部保持项。** D2-AE 的 current-state relation builder、100 immutable rest-object
   points、surface transform、roles `(joint 24,joint 26,joint 0)`、temporal anchors
   `(0,5,10,15)`、`4→128→128` point encoder、mean/max pooling、role concat order、
   `768→512` projection、four temporal embeddings、LayerNorm、single scalar
   `tanh(alpha)`、alpha exact-zero initialization、fixed segment routing、full-trunk
   placement、20 tokens、4 condition tokens、global BPS、D2-X FK-foot routing、
   `[B,16,232]` clean output、2-frame history restoration、500-step clean-x0 diffusion、
   losses/reductions/weights、optimizer、LR、batch、split、budget、sampler 和 official
   evaluator 全部不变。Global `rho(d)` 会同时衰减 anchor 0 的 clean-history relation；
   这是本单变量设计预先承认的代价，不得事后改为 per-anchor gate。参数必须仍为
   base `29,673,448`、relation `413,953`、total `30,087,401`，增量
   `1.3950283%`；不得新增 learnable parameter。Seed-42 fresh initialization 的完整
   non-persistent-buffer model-state SHA-256 必须继续精确为
   `b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c`，
   否则 GPU 前停止。

5. **Authority CPU hard gate。** D2-AF 必须继承 D2-AE 全部 geometry、asset、SO(3)、
   invariance/sensitivity、point permutation、finite、dtype/device/batch、parameter/API、
   train/sample builder parity、checkpoint provenance、HSIPrior/Mixer independence、
   forbidden-source static scan、full suite 与 registry validation contracts，并新增：

   - canonical helper、`GaussianDiffusion.sqrt_alpha_bar` 与 field buffer byte-exact；
   - timestep 必须是与 batch 同 device 的 `torch.long[B]`，负值、`>=500`、错误 shape/
     dtype/device 全部 fail closed；
   - mixed batch `[0,249,499]` 必须逐样本得到注册的三个 `rho`；
   - test-only `tanh(alpha)=0.1` 时，field-level
     `delta_AF(d)=rho(d)*delta_unit_rho`，float32 max abs `<=1e-6`；
   - `alpha=0`、shared D2-X trunk、`eval()` output max abs `<=1e-6`，并要求实际 exact
     zero where representable；
   - initial alpha gradient与 activated point-encoder/projection/temporal-embedding/
     relevant-trunk gradients在 `d=0/249/499` 均 finite/nonzero；
   - training q-sample 的 timestep 与 model timestep exact same；sampler capture 必须
     exact 为 `499,498,...,0`；
   - D2-AE/D2-AF resolved configs 除 identity、mechanism flag/variant、eligibility/
     performance binding 外 exact equivalent；
   - four-rank schedule hash一致，raw relation norm、`rho` 和 attenuated writeback norm
     分开记录；
   - 无 loss/SNR/timestep weighting、gamma/exponent/threshold/clamp、learned schedule、
     per-anchor reliability 或第二 writeback。

   任一失败分类为 `diffusion-reliability-contract-failure-stop`，不得开始任何 GPU
   workload。

6. **One-GPU functional smoke。** 注册 stem 为
   `p1-hoi-d2af-gpu-functional-smoke[-rN]-s42-<actual-date>`，worker 固定
   infbagel-4gpu/node01、1×RTX 3090、real-data batch 8、timesteps
   `0/249/499`、seed 42 random initialization、无 optimizer、zero updates、
   zero checkpoint writes。除 D2-AE smoke 内容外，必须记录三个 `rho`、mixed-batch
   per-sample scaling、raw/attenuated relation values、initial alpha gradient、
   activated gradients、peak allocated/reserved/headroom 和 model/schedule hashes。
   Operational preflight failure保留原目录并使用新 run id；scientific contract failure
   立即停止。

7. **No-training clean-signal premise gate。** Functional smoke 通过后、performance/
   formal 前，运行唯一注册的
   `p1-hoi-d2af-clean-signal-eligibility[-rN]-s42-<actual-date>`。该 diagnostic 不加载
   任何 checkpoint，不创建 model/optimizer，不做 update、rollout、official test 或
   downstream contact/goal/FS 指标；它只检验本方向的输入前提，不能选择旧 checkpoint
   或从多个可训练方向中择优。

   - 数据为完整 internal-validation split：216 sequences、29,382 windows，canonical
     non-shuffle global-index SHA-256
     `eab0bde2dc2ddad7ce2cc1817973ca46b9adaf24b1c906307f865930aeb11eb9`，
     sorted sequence-name SHA-256
     `472768c85c6d6c5b682a31a4d40a879d7a1e3d0b16085923c153db1045223fd8`；
     `num_workers=0`、batch 128。
   - timesteps 固定为 `0/249/499`。每个 timestep 使用 CPU float32
     `torch.Generator`，seed 为 `42 + 1,000,003*d`，按上述 canonical order/batch
     产生 Gaussian noise并记录完整 stream hash；`q_sample` 必须保持原 2-frame history。
   - 对 clean `x0` 和相应 `x_d` 运行完全相同的 pre-encoder pure-PyTorch relation
     builder。每个 window 的 mutable-anchor corruption 定义为 anchors `5/10/15` 上
     `[role,point,delta_xyz+distance]` 的
     `C_d=sqrt(mean((feature_d-feature_clean)^2))`。先在 sequence 内平均，再以 sequence
     为 paired unit、seed 42、10,000 bootstrap。
   - `C249-C0` 与 `C499-C249` 的 paired 95% CI lower 必须都 `>0`；anchor 0 因属于
     immutable history，其 scaling 前 feature 在三 timestep 间 max abs 必须 `<=1e-6`。
     Eligibility manifest 必须绑定已通过的 CPU/smoke summary absolute path、SHA-256
     与相同 source-tree hash；任何相关 code change 都使该 eligibility失效并要求新 id
     从头执行，但不得因此改变科学条件。

   Contract/implementation failure 分类为 `clean-signal-contract-failure-stop`；corruption
   单调 premise gate失败分类为 `clean-signal-premise-negative-stop`。任一失败均停止
   D2-AF0，不运行 performance/formal，也不得换 timestep、subset、noise、metric、anchor
   或改成 per-anchor scaling。

8. **4-GPU full-micro-batch performance hard gate。** 只有上述三层 gate全部通过，才在
   clean、identical committed worker object 上运行
   `p1-hoi-d2af-performance-benchmark[-rN]-s42-<actual-date>`：4×RTX 3090、
   per-GPU batch 512、effective batch 2048、FP32 Adam、64 warm-up + 256 measured
   updates、524,288 measured windows、CUDA synchronized timing、random initialization、
   checkpoint load/write均为零、benchmark weights禁止复用。必须记录 loader wait、H2D、
   GPU relation build、rho lookup/writeback、forward、backward、optimizer、DDP、CPU/GPU
   utilization、contention、intermediate shapes、四 rank schedule hash和 peak/headroom。
   Idle hard gate使用多次 compute-process/memory/utilization采样；P-state单独记录但不因
   单次非 P8 独立否决。

   Sealed D2-AE formal throughput 为 `3,347.0419610997483 windows/s`；D2-AF0 的
   预注册门槛固定为其 95%：

   \[
   throughput\ge 3,179.689863044761\ {\rm windows/s},
   \]

   对应 61.44M-window ETA `<=5.367399778519349 h`。同时 memory headroom 必须
   `>=max(2 GiB,10% device memory)`、losses/gradients finite、无 CPU dynamic
   geometry、无 contention。Benchmark必须一对一绑定 intended formal run id/source
   hashes；formal pre-workload retry 需要新的 benchmark retry identity。Completed
   scientific benchmark未过即分类
   `diffusion-reliability-performance-negative-stop`，不得通过 batch/worker/thread/
   architecture/point/width/role/routing或任何 sweep重试。

9. **唯一 formal training。** 只有 contract、smoke、clean-signal premise 和 performance
   全通过，才运行
   `p1-hoi-d2af-sqrt-alpha-bar-reliability[-rN]-s42-<actual-date>`。固定 seed 42、
   split `experiments/splits/omomo_hoi_train_validation_seed42.json`（SHA-256
   `019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e`）、
   4×RTX 3090、batch 512/GPU、effective 2048、accumulation 1、61,440,000 windows、
   983,040,000 frames、30,000 updates、FP32 Adam、LR `1e-4`、betas
   `(0.9,0.999)`、weight decay 0、no warmup/scheduler/AMP/clipping/EMA。
   FK/object-surface/velocity/terminal-goal weights继续为
   `0.3569973401779424 / 0.4772322188400037 / 0.1 / 1.0`，D2-X FK-foot routing
   enabled，全部新 loss disabled。必须从 seed-42 random initialization 开始；
   init/weight-init/resume均为空，released/D2-X/D2-AC/D2-AD/D2-AE/任何 prior/
   EMA/consistency checkpoint load count全为零。完整运行 fixed budget，只使用
   online/final-online；不得选择 cadence/best-validation checkpoint。稳定区间和至少一个
   resumable checkpoint通过后，按 worker-owned persistent-session规则报告 throughput/
   ETA/hash并停止主动轮询。

10. **Fixed five-path internal causal diagnostic。** Formal完成后只加载 fixed
    final-online，复用 sealed D2-O 64 sequences × 3 windows、phase offsets
    `(14,56,98)`、selection SHA-256
    `1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`。
    五条 paired 500-step rollout固定为：

    - `full_rho`；
    - `unit_rho`：每一步只将 `rho(d)` 强制为 1；
    - `relation_gate_ablated`：每一步 `tanh(alpha)=0`；
    - `temporal_correspondence_permuted`：沿用 D2-AE `k<-(k+2) mod 4`；
    - `left_right_role_swapped`：projection前只交换 left/right pooled blocks。

    除被操纵因子外，五路共享 initial latent、每一步 posterior noise、condition、history、
    ordering 和 restoration；permuted/swapped paths继续使用 canonical `rho(d)`。统计、全部
    contact/distance/penetration/goal/FS/uncertainty和描述性 relation appendix沿用 D2-AE，
    并额外报告每个 timestep/anchor/role 的 raw relation、rho、attenuated writeback norm/
    variance/sensitivity。Primary gates全部 conjunctive：

    - `full_rho-unit_rho` direct-hand union 5-cm F1 CI lower `>0`；
    - `unit_rho-full_rho` GT-contact-frame mean distance CI lower `>0`；
    - D2-AE 原五个 path/temporal/role/distance gates原阈值全部通过。

    `unit_rho` 只证明同一 D2-AF trained model 是否依赖 schedule，不等同于 D2-AF 与
    separately trained D2-AE 的模型比较。结果必须分别保存
    `internal_status={unused|schedule-negative|temporal-negative|role-negative|passed}`；
    internal无论正负都执行下面唯一一次 fixed native，不得以 internal cohort过滤 official
    result。

11. **Fixed native evaluation 与双重比较。** 协议严格沿用 D2-AE：official 438
    sequences × 3 windows、500-step unguided production diffusion、final-online、
    CFG/guidance/scene/dynamic perception/consistency全部 off，paired sequence unit、
    seed 42、10,000 bootstrap、sealed D2-X 181-sequence penetration mask、official
    evaluator/hash/helper/threshold不变。D2-X checkpoint/aggregate/per-sequence继续为
    `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51` /
    `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
    `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`，
    不重新生成。D2-AE 只复用
    sealed aggregate
    `157acda463036bdf787618c217262c14c77a09a3f409cbeada03de06e9b902a1`
    和 per-sequence
    `8533b66ea3c1fb0928b8a7581bb79c0cc14d594970314a3b7619659daddfb95c`；
    不加载/重跑其 checkpoint，不把它作为 initializer、resume、checkpoint selector 或
    candidate。

    Native必须同时通过两层 gate：

    - **D2-AE single-factor repair：** AF−AE contact F1 和 recall paired CI lower
      均 `>0`；AF/AE end-object 与 FS paired mean-ratio CI upper均 `<1.0`；
    - **D2-X candidate：** AF−X contact F1/recall CI lower均 `>0`，contact F1
      `>=0.6598838781`，released–D2-X contact-F1 gap closure `>=25%`。

    D2-AC/D2-AD protection contract原样继承：AF/X end-object、Txy、FS、Pbody、
    hand penetration、MPJPE、Troot、Tobj、Oobj mean-ratio CI upper均 `<=1.10`，
    contact precision difference CI lower `>=-0.02`，penetration finite mask exact；
    released-baseline 95% effectiveness gate原样继承。所有条件均为 AND，不允许
    composite、best-of、metric替换或阈值修改。

12. **Decision、lifecycle 与 stop rule。** Post-training同时保存 `internal_status` 和
    `native_status`；native negative是 headline，internal positive不能救回。单线终态顺序为：

    - `diffusion-reliability-contract-failure-stop`；
    - `clean-signal-contract-failure-stop`；
    - `clean-signal-premise-negative-stop`；
    - `diffusion-reliability-performance-negative-stop`；
    - `diffusion-reliability-ae-repair-negative-stop`；
    - `diffusion-reliability-d2x-transfer-negative-stop`；
    - `diffusion-reliability-conflict-negative-stop`；
    - `diffusion-reliability-positive-but-not-effective-stop`；
    - native全部通过但任一internal gate失败：
      `diffusion-reliability-native-positive-mechanism-unverified-stop`；
    - 全部通过：
      `diffusion-reliability-positive-candidate-stop`。

    只有最后一类可将 fixed final-online 标为 selectable autonomous HOIPrior candidate。
    Lifecycle stems固定为
    `p1-hoi-d2af-{cpu-contract|gpu-functional-smoke|clean-signal-eligibility|
    performance-benchmark|sqrt-alpha-bar-reliability|
    sqrt-alpha-bar-reliability-internal|native-eval|completion}[-rN]-s42-<actual-date>`；
    config default `run_id=null`，每次实际 date现场生成，失败目录保留且retry使用新 id。
    所有 resolved config、same-context manifest/preflight、logs/profile、failure trees、
    checkpoints/RNG、internal five paths/paired noise、native raw/optional outputs、run-local
    registry、hardware/data/dependency/evaluator hashes必须由worker发起non-destructive
    recovery，双端统一 `sha256_path` 和 checksum dry-run；不得 `--delete`。

    本方向禁止任何 gamma/exponent/threshold、learned schedule、per-anchor gate、LR/batch/
    budget/point/width/depth/role/placement sweep，禁止第二次 formal run、longer budget、
    D2-AF1、D2-AE/D2-AC/D2-AD resume/retrain/selection、新 loss、SNR/timestep loss
    weighting、gradient projection、rollout exposure、CFG、consistency、scene、HSIPrior 或
    Mixer。若 pretraining gate失败，formal budget不消耗但本次最后 HOIPrior direction仍
    结束；若formal启动则只允许完整运行该一次预算。最终必须写 compact result、
    `docs/phase_summaries/PHASE_1B_D2AF.md` 和 append-only completion。无论最终分类，
    本 session 均在关闭 Phase 1B 后停止，不自动开始 Phase 1C；下一独立 session 的唯一
    entry point 是 HSIPrior plan-only preregistration。

#### 2026-07-29 Phase 1B D2-AF0 implementation / pre-GPU lifecycle binding amendment

本 amendment 在 plan-only commit
`cbf55ef2c5d667d28698597127767e0b14151f06` 上实现且只实现已经预注册的
D2-AF0。核验时间为 `2026-07-29T17:01:41+08:00`；authority path、branch 和 dirty
状态分别为 `/data/yujinlun/InfBaGel-release`、`phase/01b-hoi` 和仅包含本 logical
implementation 的预期修改。当前没有启动 CPU reportable lifecycle、worker publication、
CUDA workload、optimizer update、checkpoint load/write、训练或评测。

1. **Single-factor source implementation。** 新 architecture variant
   `d2af_sqrt_alpha_bar_reliability` 完整复用 D2-AE 的 current-state geometry、100-point
   sparse assets、roles、anchors、point encoder、pooling、projection、temporal embeddings、
   LayerNorm、single alpha、routing 和 D2-X trunk，只将 field writeback 固定为

   \[
   H'_t=H_t+\sqrt{\bar\alpha_d}\tanh(\alpha)r_{a(t)}.
   \]

   Model 对逐样本 `torch.long[B]` timestep fail closed；training 中同一个 tensor object
   同时交给 `q_sample` 和 model，production sampler 按 `499,...,0` 将当前 reverse
   timestep 交给同一 field。没有 clean/future `x0`、previous predicted `x0`、Scene、
   contact、stored relation、sampler-only relation source、gamma/exponent/threshold、
   per-anchor/learned gate、loss/SNR weighting或第二 writeback。

2. **Canonical schedule 与 model/checkpoint contract。** 新增唯一 project helper
   `code/priors/diffusion_schedule.py`；`GaussianDiffusion` 与 D2-AF field byte-exact
   复用注册的 500-step float32 linear schedule。Field schedule 是 non-persistent buffer，
   不增加 parameter、learned state 或 optimizer state。D2-AF parameter count仍严格为
   base/relation/total `29,673,448 / 413,953 / 30,087,401`，seed-42 initial model-state
   SHA-256仍为
   `b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c`。
   Checkpoint metadata使用独立 `diffusion_reliability_contract`；D2-AE 与 D2-AF loader
   双向拒绝对方 provenance，base/released/D2-X/D2-AC/D2-AD 继续被拒绝。Formal resume
   还绑定 random origin、same-run identity、eligibility/performance path与SHA、source
   contract、schedule/assets以及每 rank RNG sidecar；checkpoint及任一 rank sidecar已存在
   时一律拒绝覆盖。

3. **Pre-training lifecycle implementation。** 新增 authority CPU runner、single-GPU
   functional smoke、no-checkpoint clean-signal eligibility和4-GPU full-micro-batch
   benchmark。所有工具要求现场 actual-date run id、clean exact Git object、resolved config
   先落盘且无 interpolation、exclusive output，并支持 `--resolve-only`。Eligibility不创建
   model/optimizer、不加载 checkpoint，只遍历完整216-sequence internal validation、
   canonical 29,382 windows，以注册的CPU noise streams检验 anchor `5/10/15` relation
   corruption单调性和anchor 0 history exactness；其summary同时绑定已通过的authority CPU
   与worker smoke artifact、SHA及相同 formal source-tree hash。Benchmark固定
   4×512、64 warm-up + 256 measured updates、FP32 Adam、zero checkpoint I/O，并将passing
   eligibility SHA和唯一 intended formal id写入summary；formal trainer在启动前重新验证
   eligibility和performance每个字段。

4. **Performance/preflight hardening。** Worker idle preflight改为3次、间隔1秒的GPU/
   compute-process采样；memory、utilization和external CUDA process仍是硬门禁，P-state只作
   描述性记录，避免D2-AE internal曾发生的瞬时P5误停。Benchmark仍要求无contention、
   CUDA synchronized timing、loss/gradient finite、GPU-only relation build、4-rank schedule/
   source/model identity、memory headroom和
   `throughput >= 3179.689863044761 windows/s`、ETA
   `<=5.367399778519349 h`。失败后禁止任何architecture、batch、worker、thread或科学条件
   sweep。

5. **Fixed post-training tools。** Internal runner固定五路
   `full_rho / unit_rho / relation_gate_ablated /
   temporal_correspondence_permuted / left_right_role_swapped`，共享initial latent、
   每步posterior noise、conditions、history与window ordering；另外保存raw relation、
   canonical rho、raw/attenuated writeback的timestep/anchor/role appendix。Native runner
   即使internal mechanism为负也继续唯一一次official evaluation，只复用sealed D2-AE与
   D2-X aggregate/per-sequence artifacts，不加载/重跑D2-AE checkpoint；AE-repair、D2-X
   transfer、protection、released effectiveness和最终classification precedence均逐项
   fail closed。Official evaluator source、threshold、mask和reduction未修改。

6. **Implemented paths and hashes。** Logical implementation覆盖：

   - base/D2-AF configs：
     `fe7619fbaa8256d664d5f68247ef9ebd56738db05e942bafab659b8eac5186e2` /
     `f248bdd118b1d14275867670f32e5973271c93a9d5a2a991df6c36cb4dc73876`；
   - schedule/diffusion/models/sparse relation：
     `b4d9cf74174d63de30f75acb3f687e87f824e75b147f3a2efcfd3d76befd5b09` /
     `fd8d05c34689cf4697920097bd330e6a25e3424c7460eb3a4e7ef12f45ed17a2` /
     `f7d464e48629a5e6420ea6a21f1ff8130980223cb6f59944a6226a83a952dd12` /
     `d86b49dc4030c5621510e3c66345e592235b03350fca767216de56ab78350ba3`；
   - trainer：
     `0f07d4d0060b4394bbe75e2f86bca385f6c720fdd06abc42fa7099339bba3e2d`；
   - CPU/eligibility/smoke/benchmark：
     `5d741c3f863b577ce3f8eba32b77d11fe4ffdd556a98fdf24ca84fba52c2b3c3` /
     `7c75db0ff38786b240cda39d6c95335ff35d6cac6f542a3ffd7a82b8ac26378d` /
     `71754e302215e8d5dcf37e76ef044ea504a8308d2dfaf3015b63d2d885e6b681` /
     `309fd5c0d4556ce902757b56ee92abed184f83e41fed8670c38e79d3bd69ca4f`；
   - internal/native：
     `29f542ba999c3d00b7a0b0d08814114f1803ca4a7e297bb75bd6c3f2b363109c` /
     `581efaac439e51721a4ada83ff6c852ab2f61be7d7039159539eecd81275ea6a`。

7. **Authority verification before this record。** 使用锁定authority Python完成：

   - D2-AF core/eval/CPU-lifecycle/GPU-lifecycle定向测试 `36/36`；
   - 完整 `unittest discover`：`414/414` pass，0 failure；
   - 全部新增/修改Python文件 `py_compile`；
   - `tools/experiment.py validate`：implementation record前213条registry记录有效；
   - `git diff --check`。

   本 implementation record加入后必须重新运行完整suite、registry validation、
   `py_compile`与diff check，然后提交一个clean logical implementation object。只有该
   object上的reportable authority CPU hard gate通过，才允许worker从authority发起Git
   fast-forward；之后依次为functional smoke、authority eligibility、4-GPU benchmark。
   Formal training仍严格条件化于所有pretraining gates，且无论D2-AF0最终结果如何都不再
   启动下一次HOIPrior实验。

#### 2026-07-29 Phase 1B D2-AF0 authority CPU contract completion record

Reportable authority run `p1-hoi-d2af-cpu-contract-s42-20260729` 在 clean
implementation commit
`cae7d4ed64fbc6c15b046c0d17b0cbdefd365b41` 上完成，classification为
`cpu-contract-passed`，runtime `9.5093920920 s`。Manifest在workload前由
`tools/experiment.py start`创建，resolved config先独占落盘且SHA-256为
`4377122c4f8abbba1c175f15f97f61e7e4034cac0cae3e6908c9aba01da21c45`；
manifest完成后SHA-256为
`9c7a05305a71f7907f1f056c3163037ebaa0daf0781753474e0da55d4fe50476`。

CPU hard gate结果：

- canonical schedule、`GaussianDiffusion`和四个模拟rank field buffer byte-exact，
  sqrt-alpha-bar SHA-256为
  `5d25c63d6618c77cc31976ee9e2c5645aa41653030fca210594a05254323b440`，
  buffer均不进入`state_dict`；
- mixed timesteps `(0,249,499)` 的
  `delta_AF-rho*delta_unit` max abs为 `2.384185791015625e-07`，低于注册的
  `1e-6`；错误shape/dtype/device/range全部被拒绝；
- base/relation/total参数严格为
  `29,673,448 / 413,953 / 30,087,401`，seed-42 initial state SHA-256严格为
  `b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c`；
- alpha=0与shared D2-X trunk output max abs为exact `0.0`；timestep
  `0/249/499` 的initial alpha及test-only activated point/projection/norm/temporal/trunk
  gradients全部finite/nonzero，probe未保存且optimizer update为0；
- training中`q_sample`和model收到同一个timestep tensor object；sampler trace严格为
  `499,...,0`，SHA-256为
  `a3b41318496c448ebc2cfe9a9c2b727b777e00188ac8672160b6b51de2817661`；
- D2-AF拒绝released/base/D2-X/D2-AC/D2-AD/D2-AE及缺失contract的伪D2-AF，
  D2-AE反向拒绝D2-AF；scientific checkpoint load为0；
- 全部继承的D2-AE sparse asset、surface-loss parity、yaw invariance、relative-pose
  sensitivity、left/right exchange、temporal permutation、point-set invariance、
  SO(3)/finite、train/sample builder、HSIPrior storage与Mixer clean-output contracts通过；
- source static scan确认无Scene、future/clean target、previous `x0`、contact、
  stored relation、NumPy/SciPy/trimesh/KD-tree、loss/SNR weighting、learned/per-anchor
  schedule或第二writeback；
- D2-AE与D2-AF resolved formal configs除注册identity/mechanism/eligibility/performance
  bindings外无差异；formal source-tree contract为91 files，SHA-256
  `68269a2cac8eaf6fd2b55b139bb2be5b5dbafde6e7f22496f5a894f18b843145`。

Metrics和完整stdout log均为66,188 bytes、SHA-256
`8726ad247b4b9b3828bbdef444426fa197cdb1b2f4333bfcd663fe6e4308eb7f`。
Authority staging tree包含3 files / 135,456 bytes，统一`sha256_path`为
`df730afb3685171099a7296fee87538e41cc64ae3ea61d50056eb87632221cd2`。
本 lifecycle没有CUDA、optimizer、update、checkpoint load/write、official test、
selection、formal training、consistency、HSIPrior或Mixer。下一步只允许提交本append-only
record，然后由worker发起Git fast-forward并执行same-context preflight与注册的single-GPU
functional smoke。

#### 2026-07-29 Phase 1B D2-AF0 single-GPU functional smoke completion record

Reportable worker run `p1-hoi-d2af-gpu-functional-smoke-s42-20260729` 在
infbagel-4gpu/node01 的 clean commit
`758d54897640e93cc60ac76050b9e769ddf4afbc` 上完成，status/classification为
`stable / functional-smoke-passed`。Manifest在任何CUDA workload前由
`tools/experiment.py start`从与workload相同的worker execution context创建；live
preflight连续三次确认4×RTX 3090无compute process且通过全部host/Python/data/evaluator/
clock/tunnel checks。归档的worker完整suite为`414/414` pass、按HOI-only contract仅skip
2个真实LINGO asset测试；registry validation为216条记录前的215条全部有效。

固定real-data batch 8、mixed timesteps
`(0,249,499,0,249,499,0,499)` 的结果：

- relation surface/features/point encodings/pooled blocks/relation vectors及raw/attenuated
  writeback全部finite、`torch.float32`且位于`cuda:0`，没有collator/CPU dynamic geometry；
- canonical rho严格为
  `0.9999499917030334 / 0.5297974348068237 / 0.0797039046883583`，
  `attenuated-rho*unit` max abs为
  `1.2153759598731995e-07`，低于注册的`1e-6`；
- seed-42 initial model-state SHA-256严格为
  `b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c`，
  field/diffusion schedule SHA-256均严格为
  `5d25c63d6618c77cc31976ee9e2c5645aa41653030fca210594a05254323b440`；
- loss、initial alpha gradient，以及`t=0/249/499` test-only activated
  point-encoder/projection/temporal-embedding/relevant-trunk gradients全部
  finite/nonzero；probe未保存；
- peak allocated/reserved为`270,197,248 / 325,058,560` bytes，device headroom为
  `24,970,985,472` bytes；
- optimizer未创建，update、checkpoint load/write、selection均为0，formal training未启动。

Resolved config、metrics、completed manifest与worker preflight SHA-256分别为
`d3c6865611a258e76a0c22306ab56686c8ac1543f1014ed702ec08b0b5354dec`、
`43862309e7758af25b99c7ad7f45d5882d2010912d48d54b8dabe0877fe9c8af`、
`47cff55236bf2f8698b6db74e17001419c0e058e11188e0a681e441f938f7e1a`和
`86638b0992c16a69ae1da70c4bc37912dbaa171bf7db1ad2f88f7d583dd8e369`。
Worker发起无`--delete` recovery后，checksum dry-run传输0 files；worker与authority的
10-file / 149,421-byte tree统一`sha256_path`为
`61fc8d844c68637e7cf34af4bb9e9b4dc969b71bb001af003fe22309247c0747`，
authority路径为
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2af-gpu-functional-smoke-s42-20260729`。

下一步只允许提交本append-only lifecycle record，然后在authority运行唯一注册的
216-sequence、29,382-window no-model/no-optimizer clean-signal eligibility。该premise gate
通过并恢复/注册前，不得运行4-GPU performance benchmark或formal training。

#### 2026-07-29 Phase 1B D2-AF0 clean-signal eligibility completion record

Reportable authority run
`p1-hoi-d2af-clean-signal-eligibility-s42-20260729` 在clean commit
`d12036e5e79d0e7142e8d163fc9a80a62fea317c` 上完成，status/classification为
`passed / clean-signal-premise-passed`，runtime为`105.1356934551 s`。它绑定已通过的
authority CPU metrics
`8726ad247b4b9b3828bbdef444426fa197cdb1b2f4333bfcd663fe6e4308eb7f`
和recovered worker smoke metrics
`43862309e7758af25b99c7ad7f45d5882d2010912d48d54b8dabe0877fe9c8af`，
三者formal source-tree contract均严格为91 files /
`68269a2cac8eaf6fd2b55b139bb2be5b5dbafde6e7f22496f5a894f18b843145`。
Smoke JSON未被改写；其worker绝对resolved-config引用通过authority只读symlink解析到已恢复、
SHA-256相同的staging artifact，最终prerequisite record保存的是该canonical recovered path。

Diagnostic完整遍历locked internal-validation的216 sequences / 29,382 windows，selection
hash严格为：

- global indices：
  `eab0bde2dc2ddad7ce2cc1817973ca46b9adaf24b1c906307f865930aeb11eb9`；
- sorted sequence names：
  `472768c85c6d6c5b682a31a4d40a879d7a1e3d0b16085923c153db1045223fd8`。

固定CPU noise streams与pre-encoder relation corruption结果：

- `C0 / C249 / C499` sequence mean分别为
  `0.037421006676433013 / 3.7462720501557163 / 4.458573468406191`；
- `C249-C0` paired mean为`3.7088510434792834`，10,000-replicate paired
  bootstrap 95% CI为`[3.6977504341815073,3.720119281574443]`；
- `C499-C249` paired mean为`0.7123014182504745`，95% CI为
  `[0.698195667507825,0.7267265609093563]`；
- 两个CI lower均严格`>0`；frame-0 immutable history在全部timestep与cross-timestep的
  max abs均exact `0.0`，低于`1e-6`。

因此全部三个premise gates通过，performance benchmark与formal training的premise flag为
true。该run没有创建model/optimizer，没有update、checkpoint load/write、rollout、official
test、downstream metric或checkpoint selection。

Metrics、resolved config与completed manifest SHA-256分别为
`c52c0536423d7a17101829cb2b020316b9c6e0f7aa2cf39f33b984ffb39896b4`、
`bf0646a3ec69453a17f54de78a5c7b477a6c0334bab8924e5afdad1cd39a1173`和
`83d1b5b1a6db9b4d1cea8052abadebc6d901fd08229057de4ea1d408ea78b763`。
Authority staging包含4 files / 190,598 bytes，统一`sha256_path`为
`1a9b9a2c6779d9971046ded5bc5ac23639aa73926690a8076b3c018b638bef52`。

下一步只允许提交本append-only lifecycle record，将完全相同的clean Git object和immutable
eligibility summary发布到worker，然后运行唯一注册的4×512、64 warm-up + 256 measured
performance benchmark。若其任一hard gate失败，D2-AF0与Phase 1B立即以
`diffusion-reliability-performance-negative-stop`关闭，不得启动formal training或调整条件。

#### 2026-07-29 Phase 1B D2-AF0 performance benchmark failure record

Reportable worker run
`p1-hoi-d2af-performance-benchmark-s42-20260729` 在clean commit
`1c6c3058478411361bf3e73830f900f660ae516b` 上完成。Process return code为0，固定
64 warm-up + 256 measured updates、4×512 effective batch 2048、524,288 measured
windows全部执行；scientific status/classification为
`failed / diffusion-reliability-performance-negative-stop`。

Hard-gate结果：

- measured synchronized wall：`250.8741843551 s`；
- throughput：`2089.8443630127094 windows/s`，低于注册门槛
  `3179.689863044761`；
- sealed D2-AE throughput fraction：`0.6243854685126333`，低于`0.95`；
- full 61.44M-window ETA：`8.166477355310539 h`，高于上限
  `5.367399778519349 h`；
- minimum memory headroom：`18,993,577,984` bytes，高于要求
  `2,529,604,403`；
- losses/gradients finite、relation GPU-only、四rank relation shapes、initial-state/
  schedule hashes、memory和无external contention contracts全部通过；
- checkpoint load/write为0，320-update sacrificial weights未保存且不可复用。

Mean measured profile totals across ranks为：

- loader wait/H2D：`56.2303234 / 0.3658839 s`；
- relation geometry/point encoder/projection/norm/derived pool-route-rho-writeback/
  complete module：
  `0.5566526 / 1.1547405 / 0.0265346 / 0.0036046 / 0.2908488 /
  2.0323810 s`；
- forward+loss/backward-inclusive-DDP/gradient validation/optimizer：
  `16.0291985 / 170.9401413 / 3.9957207 / 3.0369855 s`。

Relation module和完整forward与sealed D2-AE benchmark的
`1.9570280 / 15.9230519 s`接近；本次固定run的主要descriptive异常是rank-1 loader wait
`154.4085614 s`，其他rank为`53.4023972 / 9.5764329 / 7.5339022 s`，并在其他rank形成
inclusive backward/DDP等待。该证据不能事后授权retry：preregistration明确禁止第二次
benchmark、num-worker/thread/architecture sweep或改变任何科学/执行条件，因此不在
transient rank stall和可复现固定stack bottleneck之间做post-hoc选择。

Benchmark summary、completed manifest、resolved config和preflight SHA-256分别为
`53e9842d0522cf456a86eedc25d2a972cd00db3fb067113ff25f31f6117e1f33`、
`97d13c60dd0e073fdd649aec7b76bde4dad23fdf0fd7e8cef9b9ca04b6a04e54`、
`04f747890fd9e7ad3d40a580223783da849ac80a5eae1d826c9bc9af2f4b45a9`和
`e238a8242f31b5a08b083f1e11044834d922babe5c29681754d0a755396613d1`。
Worker发起无`--delete` recovery后，checksum dry-run传输0 files；worker/authority的
14-file / 1,914,984-byte tree统一`sha256_path`为
`076ed5e3ee80bd5325c661f9a3adbe225e45be963cc2166128cdc5c0faadf895`，
authority路径为
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2af-performance-benchmark-s42-20260729`。

Performance gate失败后已现场验证intended formal目录
`p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729`不存在。Formal training、
formal optimizer/checkpoint、internal和native均未启动。不得retry、调参、运行D2-AF1或
启动任何新的HOIPrior方向。

#### 2026-07-29 Phase 1B D2-AF0 one-time user-authorized performance waiver（plan-only）

在上述失败被完整保留并报告后，用户明确接受已测完整预算ETA
`8.166477355310539 h`，并授权：若没有确定、简单且不改变科学条件的训练时间优化，则直接
运行现有D2-AF0唯一formal budget。该新授权覆盖原先“performance失败即不训练”的执行
stop rule，但不回写历史、不把benchmark改成passed，也不改变其
`diffusion-reliability-performance-negative-stop`分类。

1. **ETA与根因解释锁定。** ETA只由固定预算和实测端到端吞吐外推：

   \[
   61{,}440{,}000 / 2{,}089.8443630127094 / 3600
   = 8.166477355310539\ {\rm h}.
   \]

   即每个2048-window update约`0.9800 s`。新增
   `sqrt(alpha_bar)`/rho并非主要计算开销：256个measured updates中，D2-AF relation
   module为`2.0323810 s`（约`7.94 ms/update`），sealed D2-AE为`1.9570280 s`
   （约`7.64 ms/update`）；完整forward分别为`16.0291985 / 15.9230519 s`，仅增加约
   `0.41 ms/update`。主要wall增长来自rank-skewed DataLoader/DDP critical-path wait：
   rank-1 loader wait为`154.4085614 s`，其他rank为
   `53.4023972 / 9.5764329 / 7.5339022 s`，其余rank相应在inclusive backward/DDP中等待。

2. **不做post-hoc execution sweep。** 当前没有已证实能够消除上述rank skew的单一安全
   toggle。`num_workers`、CPU affinity、prefetch/pinning、线程或I/O布局变更都需要新的
   full-micro-batch比较才能证明有效；第二次benchmark和这些sweep继续禁止。
   `profile_every_update=true`也保持不变：其同步不是已测rank-1 loader stall的根因，事后
   改动会改变注册execution contract而没有可报告的同条件证据。因此本waiver选择用户授权的
   “直接训练”分支，不改模型数学、训练循环计算、data loader配置或instrumentation。

3. **Waiver的精确范围。** 只允许启动一次原intended formal identity
   `p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729`；启动时仍须满足actual-date规则，
   且该目录必须此前不存在。原benchmark不重跑，320-update sacrificial weights仍不可复用。
   Formal仍从seed-42随机初始化，4×512/effective 2048、30,000 updates、61.44M windows、
   FP32 Adam、LR/loss/budget/split/checkpoint cadence和全部D2-AF0科学条件完全不变。
   不允许第二次formal、resume旧方向、checkpoint selection、D2-AF1、longer budget、
   consistency、HSIPrior或Mixer。

4. **Fail-closed implementation。** Formal trainer不得简单删除performance检查或伪造
   passing summary。它必须同时绑定：

   - 原failed benchmark JSON的absolute path、SHA-256、run id、failed status/
     classification、实测throughput/ETA及全部non-speed contracts；
   - 一份tracked、immutable、SHA-bound waiver JSON；
   - waiver中的唯一formal run id、用户授权事实、benchmark SHA、原/目标Git commit、
     exact transition diff SHA、允许改变的governance/validator/config/test路径和目标
     formal source-tree contract；
   - `formal_runs_maximum=1`、benchmark retry/sweep=false、training conditions
     unchanged=true、random initialization=true。

   原benchmark的throughput/ETA checks必须在formal lifecycle中继续保存为false；
   新状态只能表示为`failed-waived / user-authorized-performance-waiver`，不得表示为
   `performance-gate-passed`。Benchmark中memory、finite loss/gradient、GPU-only relation、
   optimizer/checkpoint I/O、four-rank identity、contention、eligibility和schedule等任一
   non-speed contract不通过时，waiver无效并停止。

5. **Source transition与重新验证。** 为接受waiver所需的source修改只允许涉及performance
   validator、base/D2-AF config binding及对应tests/documentation；不得修改models、
   diffusion schedule、relation builder/encoder/routing、loss、optimizer或training-loop
   数学。由于原CPU/smoke/eligibility/benchmark是在旧formal source hash上完成，waiver必须
   以source/target commit和exact Git diff hash显式授权这次validator-only transition，
   而不是重写旧artifact。目标commit上必须重新通过完整authority suite、registry
   validation、static source/diff audit和resolved-config fail-closed测试；不重跑scientific
   performance benchmark。

6. **Formal后的原评测不变。** 训练完成后仍只使用fixed final-online，依次执行已注册的
   five-path internal和一次fixed native evaluation。Internal/native gates、统计、sealed
   controls、最终科学分类和selectability条件全部不变；compact result和phase summary必须
   同时报告原performance failure、用户waiver、实际formal wall/throughput及最终结果。

下一步仅允许提交本plan-only waiver和append-only registry hypothesis；随后实现上述最小
hash-bound validator/config/tests，创建并提交immutable waiver contract，通过authority
verification后由worker fast-forward相同clean Git object并启动唯一formal run。

#### 2026-07-29 Phase 1B D2-AF0 performance waiver implementation / contract record

一次性waiver已按上述plan实现，且未改写原benchmark。Validator/config/tests logical
implementation commit为
`9c908ad87dce8806eb052b2a2627160b0a1bbe72`；tracked immutable contract commit为
`69d8cb025c89c0e776d0a4c03a8c158bbd0a3265`，文件
`experiments/contracts/p1_hoi_d2af_performance_waiver_s42_20260729.json` 的SHA-256为
`8a2d11c0febea603ac74328fbcd51622982740c4bef48597a0af71de7a53da97`。

实现保持两条分离路径：

- 原passing benchmark仍要求`status/classification/throughput/eta/formal_authorized`
  全部通过，且waiver fields必须为空；
- 原failed benchmark只允许exact五项
  `classification / eta / formal_authorized / status / throughput`为false，其他
  memory、finite、GPU-only、timing、optimizer/checkpoint I/O、four-rank identity、
  contention、schedule、eligibility、source identity和sweep contracts必须全通过。
  之后才读取tracked waiver，并返回
  `status=failed-waived`、
  `classification=user-authorized-performance-waiver`、
  `formal_authorization=explicit-single-run-waiver`和
  `original_gate_passed=false`。

Waiver exact绑定：

- benchmark summary：
  `53e9842d0522cf456a86eedc25d2a972cd00db3fb067113ff25f31f6117e1f33`；
- eligibility：
  `c52c0536423d7a17101829cb2b020316b9c6e0f7aa2cf39f33b984ffb39896b4`；
- source commit/contract：
  `1c6c3058478411361bf3e73830f900f660ae516b` /
  `68269a2cac8eaf6fd2b55b139bb2be5b5dbafde6e7f22496f5a894f18b843145`；
- target implementation commit/contract：
  `9c908ad87dce8806eb052b2a2627160b0a1bbe72` /
  `299d7a900c6a96264dd698c50ef476ea78d2b2efdfbb3b0e375d27d99101cc3e`；
- exact binary Git diff：
  `24d0dbb8abd96b56f6e745b0f08fcabeb0a50792a4737d8332ca6158aadec7c3`；
- changed paths严格为base/D2-AF config、trainer performance validator、
  `tests/test_hoi_d2af.py`和既有plan/registry六项；models、diffusion、relation、
  loss、optimizer、training loop、DataLoader和profiling均未修改；
- 唯一formal id：
  `p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729`，formal runs maximum 1，
  random initialization，benchmark retry/sweep/reclassification均false。

Authority在clean contract commit上使用真实recovered eligibility/benchmark和tracked
waiver直接调用formal validator，全部authorization与waiver checks为true；原五项
benchmark failed checks继续原样保存。Target source contract由current worktree与target
Git object两种算法独立重算，均为91 files /
`299d7a900c6a96264dd698c50ef476ea78d2b2efdfbb3b0e375d27d99101cc3e`。

Source implementation提交前的authority verification为：

- 完整`unittest discover`：`419/419` pass；
- D2-AF waiver定向测试：passing path、exact failure+waiver、missing waiver、
  extra non-speed failure、benchmark/waiver tamper全部通过；
- `py_compile`、registry validation（219 records）和`git diff --check`通过；
- formal output、metrics、state和checkpoint均不存在；CUDA workload、optimizer update、
  checkpoint load/write、internal和native均未启动。

下一步只允许提交本append-only lifecycle record，随后由worker发起Git fast-forward到相同
clean object；现场验证worker Python/data/artifacts、actual date、formal目录不存在、
resolved config无interpolation、same-context manifest/preflight和waiver validator后，
启动唯一formal run。不得运行第二次performance benchmark或任何优化sweep。

#### 2026-07-29 Phase 1B D2-AF0 checkpoint-race operational continuation（plan-only）

用户报告GPU workload早于ETA结束后，authority通过worker的loopback-only control channel
完成只读取证。核验时authority为`/data/yujinlun/InfBaGel-release`、branch
`phase/01b-hoi`、HEAD
`7202d32a7375e7197886c4f873688fd472e2c803`、worktree clean；worker checkout为
`/home/yujinlun/data/work/InfBaGel-release`、相同branch/HEAD且clean；核验时间为
`2026-07-29T23:15:09+08:00`至`2026-07-29T23:22:30+08:00`。重新扫描authority
working tree、Git history/refs/reflogs、registry、authority staging、worker checkout与
worker artifacts后，
`p1-hoi-d2af-checkpoint-race-continuation-s42-20260729`及stem
`d2af-checkpoint-race-continuation`均未使用。本commit只允许追加本plan与registry，
不得修改source、config或tests，不得加载checkpoint或启动GPU workload。

1. **失败事实与分类。** 唯一formal run
   `p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729`并未完成。它从seed-42随机初始化
   正常运行到第三次cadence save，在attempted `9,216,000` windows /
   `4,500` updates处以return code `1`退出；没有`training_state.json`或
   `metrics.json`，manifest继续保留`running`，internal/native均未启动。失败记录
   `operational_checkpoint_race_failure.json`的SHA-256为
   `a66fec685afb5cbb4079619de9417b7171af7e29244723f1deac9d4ba306d1b1`。
   根因是`code/train_hoi_prior.py::_save_checkpoint()`中每个rank在写自己的RNG
   sidecar前检查了主checkpoint和全部rank sidecar；rank 0/1/3先写后，rank 2把这些合法
   peer文件误判成overwrite collision。该错误不涉及D2-AF relation数学、loss、gradient、
   CUDA OOM、数据、磁盘容量或scientific gate，分类固定为
   `ddp-checkpoint-sidecar-existence-race-operational-failure`。

2. **失败与可恢复状态均不可改写。** 最后完整checkpoint固定为
   `6,144,000` windows / `3,000` updates，SHA-256
   `3c94f7344991cb38aab37fd8356cabe83a84b449d10505e0e46341490605287e`，
   四个rank RNG sidecar均存在并已逐文件hash。第三次cadence已写出的rank 0/1/3
   partial sidecar已无损移动到
   `operational_failures/checkpoint_race_windows009216000`；3 files /
   45,977 bytes，tree SHA-256
   `b5573764eceb388f6a28f10b4ed89b44bbbcdd430213dad490f6c8b5caa7f9dd`，
   内容未修改、未删除。原`train.log`、`returncode.txt`、resolved config、manifest、
   preflight、initial-stability和checkpoint artifacts全部保留，不得覆盖或伪装成成功。

3. **同一scientific lineage，而非第二次实验预算。** 唯一允许动作是从上述exact
   `6,144,000`-window checkpoint在同一formal run id内继续到原固定
   `61,440,000` windows / `30,000` updates。seed、split、model、sqrt-alpha-bar
   routing、100-point relation、loss/reduction/weight、optimizer及其state、LR、batch、
   DataLoader、profiling、checkpoint/validation cadence、budget和final-online规则全部
   不变；不得从`9,216,000` partial state恢复，不得重启from-random、创建第二manifest/
   formal id、延长预算或选择中间checkpoint。崩溃前未保存的1,500 updates丢弃并由exact
   RNG/optimizer state重放；accepted lineage仍为30,000 updates，但实际GPU总成本须另报
   已失败的4,500加continuation的27,000 updates，不得把重放隐藏为正常30,000-update wall。
   该continuation不授权D2-AF1、任何sweep、consistency、HSIPrior或Mixer。

4. **唯一source修复。** `_save_checkpoint()`只允许：

   - 每个rank在任何write前检查自己的RNG sidecar是否已存在；
   - 只有rank 0检查共享主checkpoint是否已存在；
   - 通过collective collision flag使任一rank发现collision时全部rank同步fail closed；
   - 全部rank通过collision preflight后增加barrier，再写各自sidecar；
   - 保留sidecar写完后的barrier、rank-0主checkpoint atomic write和最终barrier。

   peer sidecar绝不再作为collision；own sidecar或主checkpoint仍必须fail closed。不得修改
   checkpoint cadence/schema/value、RNG内容、optimizer/model state、训练循环数学或任何
   D2-AF source。必须增加collective顺序、peer-sidecar、own-sidecar和main-checkpoint
   collision regression tests。

5. **Hash-bound operational resume contract。** 修复后HEAD与checkpoint commit不同，
   因此resume只能通过tracked immutable continuation JSON和既有generic commit-transition
   guard共同授权。continuation contract必须精确绑定：

   - same formal run id与seed 42；
   - source/checkpoint commit `7202d32a7375e7197886c4f873688fd472e2c803`；
   - exact checkpoint path/basename/SHA和`6,144,000` windows / `3,000` updates；
   - 四个rank RNG sidecar SHA；
   - exact failure-record path/SHA/classification/return code；
   - exact partial archive path/tree SHA、文件数/bytes及原checkpoint目录已只清除该组
     partial files；
   - 原performance waiver target formal-source contract
     `299d7a900c6a96264dd698c50ef476ea78d2b2efdfbb3b0e375d27d99101cc3e`；
   - source commit到唯一implementation target commit的
     `git diff --binary` SHA、changed paths与target formal-source contract；
   - science/config/budget unchanged、same-run only、new formal budget false。

   允许的source-transition paths仅为base/D2-AF config、trainer checkpoint/provenance
   guards、D2-AF regression tests及本plan/registry。任何checkpoint/RNG/failure/contract/
   diff tamper、额外source path或current formal-source drift必须在GPU前停止。原failed
   performance benchmark和one-time waiver仍原样保留；不得重分类为performance passed。

6. **Resume execution与artifact规则。** Worker必须由authority committed clean object
   fast-forward到完全相同的target commit，现场验证machine-local Python、四卡空闲、
   data/assets/hash、checkpoint/failure/partial archive和source-transition。原manifest不
   重建；新增且不覆盖
   `resolved_hydra_config_resume.yaml`、`resume_preflight.json`、
   `resume_contract.json`、`resume.log`、`resume_returncode.txt`和
   `resume_initial_stability.json`。Resolved config必须无interpolation，并包含exact
   checkpoint、continuation contract及source/target/diff binding。续训在worker-owned
   persistent session运行；初始稳定、finite loss/gradient、显存和下一个完整cadence
   checkpoint验证后停止主动轮询。

7. **完成路径不变。** 训练完整结束后，`metrics.json`和`training_state.json`必须记录
   resumed-from checkpoint、source/target/diff provenance及完整30,000-update结果。
   Resume进程的内置wall/loss/validation/checkpoint-hash accumulators会从continuation
   启动点重新累计，且raw throughput以累计processed windows为分子，因此不得把该raw值
   报作完整run throughput；必须分别报告continuation wall/throughput、accepted-lineage
   active wall/throughput和包含失败重放的总GPU cost。`tools/experiment.py finish`使用
   既有hash-bound manifest transition，不修改该工具、不放宽dirty/overwrite检查。随后才按
   原注册顺序执行fixed five-path internal、fixed native、non-destructive worker-initiated
   recovery、双端tree/hash验证、compact result、`PHASE_1B_D2AF.md`和append-only
   completion record。任何新的operational failure均保留，不自动改run id、删artifact或
   改变科学协议。

下一步仅允许提交本plan-only amendment与append-only registry hypothesis；随后实现上述
最小checkpoint-race fix、D2-AF专用continuation validator/config/tests，并创建绑定唯一
implementation commit的immutable continuation contract。所有authority/worker CPU与
preflight contracts通过前不得恢复GPU训练。

#### 2026-07-29 Phase 1B D2-AF0 checkpoint-race continuation implementation record

上述plan-only amendment已提交为
`3b9c0c53bff1e09ec880a6795fd1fad550bc2495`。本logical implementation严格限制为checkpoint
save同步、D2-AF resume provenance config/validator与regression tests；未修改model、
diffusion schedule、relation builder/encoder/routing、loss、optimizer、DataLoader、
batch、budget、sampler或evaluator。

- `_save_checkpoint()`不再由每个rank扫描全部peer sidecar。每个rank只形成own-sidecar
  collision flag，rank 0额外形成main-checkpoint flag；NCCL collective MAX使任一collision
  在全部rank同步失败。collective通过后另有pre-write barrier，随后保持原own-sidecar
  atomic write、post-sidecar barrier、rank-0 main atomic write和final barrier。
- Base/D2-AF config只新增默认null的tracked continuation path/SHA字段；这些字段不进入
  scientific `_resume_contract`，因此不会改变旧checkpoint的model/optimizer/data contract。
- 原performance waiver仍要求其旧target formal-source contract。只有same-run resume且
  current source因本次registered fix变化时，validator才进一步要求一份exact
  checkpoint-race continuation contract；fresh run、其他checkpoint或无contract的
  changed-source resume继续fail closed。
- Continuation validator绑定原manifest、6,144,000-window checkpoint及四rank RNG、
  failure JSON、9,216,000 partial archive、原waiver target source contract、
  source-to-implementation binary diff、current execution diff和scientific-unchanged
  booleans。Generic resume commit-transition allowlist只增加本次D2-AF config/tests/contract
  paths；D2-AB exact/guard behavior保持通过。
- Regression覆盖peer sidecar不构成collision、own sidecar/main checkpoint仍拒绝、
  remote collision collective传播、changed-source waiver必须有continuation，以及
  checkpoint/RNG/failure/partial/source/execution/science全部绑定的positive fixture。

Implementation-stage定向verification为：D2-AF core `22/22`、D2-AF CPU/GPU lifecycle
`15/15`、D2-AB resume regression `16/16`通过，`py_compile`通过。此时尚未创建immutable
continuation JSON，未加载worker checkpoint，未启动GPU、optimizer、internal或native。
下一步先运行完整authority suite、registry validation、static diff audit并提交本logical
implementation；随后以该commit为唯一implementation target创建tracked contract和
append-only binding record。Contract commit不得修改formal runtime source。

完整authority verification随后通过：`unittest discover`为`424/424`，registry
validation为222 records，`py_compile`和`git diff --check`通过。Source commit
`7202d32a7375e7197886c4f873688fd472e2c803`的D2-AF formal-source contract重算为
91 files /
`299d7a900c6a96264dd698c50ef476ea78d2b2efdfbb3b0e375d27d99101cc3e`；
implementation worktree同算法为91 files /
`daa57294f4d25db4591a2ef6bcbe8157ca812b99b3b1dfe4c6c01aaf23c2ffd4`。
Changed paths严格为base/D2-AF config、trainer、D2-AF tests和既有plan/registry六项；
models、diffusion、relation、loss、DataLoader、sampler、evaluator、`tools/experiment.py`
均无diff。额外4-process CPU Gloo contract证明无collision时四rank全部通过、rank 2
own-sidecar collision时四rank全部同步拒绝。上述verification仍未读取scientific
checkpoint或启动GPU workload。

#### 2026-07-29 Phase 1B D2-AF0 checkpoint-race continuation contract binding

唯一immutable continuation contract已在implementation commit
`b7248bba3e77234c8f2a5993d8bf3ee8a1db2757`之后创建为
`experiments/contracts/p1_hoi_d2af_checkpoint_race_continuation_s42_20260729.json`，
SHA-256为
`1a4ddf3b220b96f7aea0f1de7c0b8fd3fd9458eb913d284aaacc85a7fa226424`。
该contract只授权原run
`p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729`从唯一完整的
6,144,000-window / 3,000-update checkpoint继续同一manifest；不创建新formal run、
不从随机初始化重启、不使用不完整的9,216,000-window保存点，也不改变模型、relation、
loss、optimizer、batch、budget、DataLoader或evaluation协议。

Contract精确绑定：

- 原running manifest SHA-256
  `985192f686de2d4330cb82c826b648a08d12b7ed55c0bd4c8d196951d05b589b`；
- resume checkpoint SHA-256
  `3c94f7344991cb38aab37fd8356cabe83a84b449d10505e0e46341490605287e`
  及四rank RNG sidecar hashes；
- preserved operational failure SHA-256
  `a66fec685afb5cbb4079619de9417b7171af7e29244723f1deac9d4ba306d1b1`
  与partial archive tree SHA-256
  `b5573764eceb388f6a28f10b4ed89b44bbbcdd430213dad490f6c8b5caa7f9dd`；
- source commit `7202d32a7375e7197886c4f873688fd472e2c803`、implementation target
  `b7248bba3e77234c8f2a5993d8bf3ee8a1db2757`及binary diff SHA-256
  `19778c2dac54ae080b241dabb1215dd55d6defa0e231c301ccee2ed48d43498a`；
- source/target formal-source contract SHA-256分别为
  `299d7a900c6a96264dd698c50ef476ea78d2b2efdfbb3b0e375d27d99101cc3e`
  和
  `daa57294f4d25db4591a2ef6bcbe8157ca812b99b3b1dfe4c6c01aaf23c2ffd4`。

JSON syntax、contract SHA、source transition diff与changed paths均在authority重算通过。
Continuation最终accepted lineage仍固定为61,440,000 windows / 30,000 updates；失败尝试
执行4,500 updates，continuation从3,000继续27,000 updates，因此实际GPU执行成本必须显式
报告为31,500 updates，不能将重放隐藏在raw resumed throughput中。本binding commit之后，
formal workload完成并关闭manifest前不得再改变execution HEAD。下一步仅允许worker发起
Git fast-forward、验证相同clean object与machine-local Python，生成不覆盖原文件的resolved
resume config/preflight/contract/log artifacts，并恢复原persistent lineage。

#### 2026-07-30 Phase 1B D2-AF0 formal training completion and recovery record

唯一授权的formal run
`p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729`已在worker
`phase/01b-hoi@044227fe512a9ee6d1c2a1bc898d3b8a2c6ca706`完成，并于
`2026-07-30T04:56:20+08:00`闭合。`resume_returncode.txt=0`；accepted lineage严格为
61,440,000 windows / 983,040,000 frames / 30,000 optimizer updates。原失败尝试执行
4,500 updates，continuation执行27,000 updates，因此实际GPU updates为31,500；该重放成本
保留为operational accounting，不改变固定科学预算。训练仍从seed-42随机初始化，released、
D2-X/D2-AE/D2-AC/D2-AD、prior、EMA或consistency checkpoint加载数均为0。

Fixed final-online checkpoint为
`p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729_windows061440000.pth`：

- checkpoint SHA-256：
  `483c63ecaeb6dbf5a0a54400e0eecec722ff6df6d72226ce263e7fe053e412e2`；
- model-state SHA-256：
  `7b6e333724f21490c96a0599103cc7eb087b9452e64a8d3c2b9a5ce85ae704bb`；
- 参数量：30,087,401；optimizer state为129项，step min/max均为30,000；
- learned alpha / `tanh(alpha)`：
  `-0.0925037190 / -0.0922407731`；
- final validation total：`0.0502382831`；loss、model、optimizer与required gradients均
  finite，AMP overflow为0；
- 每rank最小显存headroom为19,369,492,480 bytes。

Cadence完整性为20个main checkpoints与80个rank RNG sidecars，missing/extra均为0。
必须保留原checkpoint-race failure
`a66fec685afb5cbb4079619de9417b7171af7e29244723f1deac9d4ba306d1b1`、
partial archive tree
`b5573764eceb388f6a28f10b4ed89b44bbbcdd430213dad490f6c8b5caa7f9dd`
以及两个未启动GPU的旁路operational failure目录。

Timing只允许报告一致口径：

- continuation-only throughput：`3232.575359 windows/s`；
- accepted-lineage artifact throughput：`3218.215477 windows/s`；
- serialized raw `3591.750399 windows/s`使用全预算windows除以continuation-only wall，
  不得作为formal完整训练吞吐。

Worker lifecycle已执行`finish`与run-local `register`。关键SHA-256为：

- manifest：
  `49371a577a037444aef47fd5fda64f5d147ecd712247308b99b675d1edee55d3`；
- run-local registry：
  `7d1ba1bd99cf2c4dcf95a2300522f828e832d70b0f12afe3213c547471dedf50`；
- metrics：
  `25b172f21d78d97412cb4eeeb79b43566d7e488286c383127a4edf0272c11903`；
- training state：
  `8dcb3ea4e1e39d661bcef138de6ff347731db8eeb88213fe0b4e0ba83204f8a4`；
- resolved resume config：
  `6845d032e48027a35ccbd20169d118cc81429ac652c31f8c615e5474d45fe870`；
- resume preflight：
  `5f404f407d022c7db80fb1c781f69828d7abccbcecbd3f995cdd2049cace4e51`；
- resume contract：
  `35240fb486b891a520ad3f08c9e557594349adc77b4868eca9547b845f540f2f`；
- formal completion verification：
  `a6263835cf79c6b803275c3d9c96c269aa1c2e75b1c8fea3fce4b4b56f7f1ec1`。

Worker随后发起non-destructive `rsync -aH --partial`，未使用`--delete`，将完整156-file /
7,227,356,886-byte formal tree恢复到
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729-recovery-r1`。
Worker与authority的`sha256_path`均为
`9bff3d9a182138ee30ca586b10d71f689e6aa0c7345d2b1052fe0ea15251dc6c`，
重复`rsync --checksum --dry-run`无差异。Fixed final-online不可选择、不可resume、不可用于
任何后续prior初始化；下一步仍只允许预注册的five-path internal和一次fixed native。

#### 2026-07-30 Phase 1B D2-AF0 evaluation provenance hardening（plan-only）

在任何internal/native GPU workload前，对现有D2-AF evaluator做一次纯provenance
hardening。核验时authority为clean
`phase/01b-hoi@d51057c35485d9b5e1abc846a55dc2f4324f9659`，worker仍为clean
`phase/01b-hoi@044227fe512a9ee6d1c2a1bc898d3b8a2c6ca706`。Identifier audit覆盖authority
tracked worktree、Git history/refs/reflogs、staging、authority results及worker checkout/
artifacts；以下identifier均未使用：

- `p1-hoi-d2af-evaluation-provenance-hardening-s42-20260730`；
- `d2af-evaluation-provenance-hardening`；
- `p1-hoi-d2af-sqrt-alpha-bar-reliability-internal-s42-20260730`；
- `p1-hoi-d2af-native-eval-s42-20260730`。

本amendment不改变scientific protocol、checkpoint、cohort、sampler、intervention、metric、
threshold、bootstrap、gate、classification precedence或native evaluator，只使已注册事实
fail closed：

1. **Internal formal-lineage binding。** Internal CLI/resolved config/manifest必须显式接收并
   hash-bind completed formal `manifest.json`、`metrics.json`、`training_state.json`、
   `resume_contract.json`与fixed final-online。Validator必须同时证明：

   - manifest已completed并绑定唯一formal run；
   - metrics/state均为61,440,000 windows、983,040,000 frames、30,000 accepted updates；
   - final checkpoint basename/SHA、model-state SHA、architecture、seed、random origin、
     20 main checkpoints与80 RNG sidecars一致；
   - checkpoint source commit为
     `7202d32a7375e7197886c4f873688fd472e2c803`，execution target为
     `044227fe512a9ee6d1c2a1bc898d3b8a2c6ca706`，binary diff为
     `f0cba48ae5d1ba271750ef5d7c042d1b04e8ec6b5e60df00fca5f19c1db8f609`；
   - operational continuation只来自registered 6,144,000-window checkpoint，原failure/
     partial archive bindings保持一致；
   - fixed final-online checkpoint SHA仍为
     `483c63ecaeb6dbf5a0a54400e0eecec722ff6df6d72226ce263e7fe053e412e2`。

2. **Internal RNG/input identity。** Batch size从“64的任意因数”收紧为严格`8`，因为seed label
   使用chunk index，改变batch会改变实际随机流。五条paths的first window必须比较完整
   `path_local_model_inputs` identity，而不只比较fixed history；至少覆盖history、global
   BPS、local goals、object/world rotation references及全部exogenous model inputs。
   Later windows仍按各自causal generated history分叉，不能被错误要求相同。

3. **Internal raw-artifact closure。** Summary必须记录并hash-bind五个raw variants：
   `full_rho`、`unit_rho`、`relation_gate_ablated`、
   `temporal_correspondence_permuted`、`left_right_role_swapped`，以及paired noise、
   paired conditioning、causal overlap和reliability appendix。Existing seven internal
   decision/gate booleans、paired sequence uncertainty、selection hash与schedule hash必须从
   raw artifacts重算一致；不得新增诊断路径或改变gate math。

4. **Native upstream closure。** Native CLI/resolved config/manifest必须显式hash-bind同一formal
   manifest/metrics/state/resume contract与internal summary。Native preflight必须重新读取并
   验证五个internal raw variants、supporting paired artifacts、seven decision/gate booleans及
   它们的SHA；只接受上述唯一resumed lineage。Internal mechanism无论正负仍执行native，
   只有contract failure停止。

5. **Regression gate。** GPU前至少覆盖：

   - completed resumed-lineage positive fixture；
   - missing/tampered formal manifest、metrics、state或continuation contract；
   - wrong source commit、execution target或binary diff；
   - checkpoint/final lineage mismatch；
   - internal batch size非8；
   - first-window完整input identity mismatch；
   - missing/tampered internal raw variant、gate或hash；
   - native对formal/internal binding的positive与negative fixtures。

6. **Allowed source scope与stop rule。** Logical implementation只允许修改
   `tools/run_hoi_d2af_internal.py`、`tools/run_hoi_d2af_native_evaluation.py`、
   `tests/test_hoi_d2af_eval.py`及本plan/registry；若测试需要，可在同一D2-AF eval test文件
   内增加fixture/helper。不得修改models、diffusion、relation、loss、training、dataset、
   official evaluator或sealed artifacts。任一hardening contract失败分类为
   `diffusion-reliability-contract-failure-stop`，不得启动GPU workload。

Hardening实现与full authority suite通过后，worker只允许Git fast-forward到完全相同的clean
commit，现场验证Python/worker expert/data/evaluator hashes，按实际日期创建未使用run id，
先registered fixed five-path internal，再无条件执行唯一一次fixed 438×3 native。不得启动
D2-AF1、second training、sweep、consistency、HSIPrior或Mixer；Phase 1B completion record
提交前不进入Phase 1C。

#### 2026-07-30 Phase 1B D2-AF0 evaluation provenance hardening implementation verification

上述plan-only hardening已按注册范围实现，未修改model、diffusion、relation builder、
training/loss/data、scientific sampler、metric、threshold、bootstrap或gate math。实现只涉及
两个D2-AF evaluation runner、同一D2-AF eval regression test，以及本append-only
plan/registry：

- internal现在显式绑定completed formal manifest、metrics、training state、resume contract、
  final-online checkpoint与`formal_completion_verification.json`；
- recovered formal cadence必须恰好包含20个main checkpoints与80个rank RNG sidecars，
  并逐文件验证regular/non-symlink、basename、kind/rank/window、bytes、SHA、checkpoint
  schema/commit/progress/architecture/RNG pattern及RNG exact state schema；
- 6,144,000-window resume checkpoint与四个sidecars、checkpoint-race continuation、
  source/target commit及binary diff均交叉绑定；
- internal batch严格锁定为8，五条paths固定24个`chunk × window` stream coordinates；
  first-window完整model-input hashes必须跨paths相同，later windows只允许同一路径causal
  history；
- native不再信任internal summary/support自报布尔，而是从五个raw variants、paired noise、
  paired conditioning、causal overlap与reliability appendix重新验证cohort/order、
  selection、500-step schedule、relation trace、comparisons、seven gates及decision evidence；
- artifact closure只接受metrics run root内的regular files，拒绝absolute path、escape与最终
  symlink；
- 最终审查识别的`rho_*_max_abs`/schedule sentinel NaN比较绕过已在GPU前修复；所有这些
  scalar现在必须为finite real number，并新增self-consistent NaN raw-artifact regression。

真实recovered formal tree验证通过：100个cadence files、7,226,924,444 bytes，全部16项
formal-lineage checks为true。Authority验证为：

- `tests.test_hoi_d2af_eval`：17/17 passed；
- full `unittest discover -s tests`：432/432 passed；
- existing registry：226 records validated；
- project Python 3.8.20 `py_compile` passed；
- `git diff --check` passed；
- adversarial regressions覆盖different noise、future-GT/exogenous forgery、missing window、
  forged cohort、empty relation、zero schedule、nonfinite relation scalar/sentinel、
  nonfinite summary与excessive history。

本verification期间checkpoint load、optimizer update与GPU workload均为0；fixed internal/
native run id仍未使用。下一步先提交该logical implementation，再追加只绑定实际implementation
Git object的governance-only record；随后worker仅可fast-forward到相同clean execution
object并执行fixed internal。Internal contract失败立即停止；internal科学结果正负均不改变
随后唯一native的执行要求。

#### 2026-07-30 Phase 1B D2-AF0 evaluation hardening implementation binding

Logical implementation已提交为
`3d4ff1eb5c57b1b08537859dca8e895bc428a26d`，tree为
`c46b6b4dfad7ecd3b8b90af6cf71e2ed8fc7ecf7`。相对plan-only commit
`7a484fe18dc28e29e30b2966d966825823130c0b`的binary diff SHA-256为
`b07123b8086cd523cbe3c89006ce7264e13f08468ba6ae13844b48bb6ecf8b34`，changed paths严格
为两个D2-AF evaluation runners、一个D2-AF eval test及plan/registry五项。该commit之后
尚未启动checkpoint load或GPU workload。Worker必须fast-forward到包含本binding的最终clean
HEAD；其source代码必须与上述implementation object相同，随后才可创建registered internal
manifest。

#### 2026-07-30 Phase 1B D2-AF0 fixed internal causal diagnostic completion

Reportable worker run
`p1-hoi-d2af-sqrt-alpha-bar-reliability-internal-s42-20260730` 在
infbagel-4gpu/node01 的clean evaluation execution commit
`a4cdcf09f84553159be10c555ff8a6773b65d3aa` 上完成，exit code为0，runtime为
`538.3266074708663 s`。Manifest在GPU workload前由相同execution context创建；resolved
config无unresolved interpolation，worker source/Python/data/evaluator/formal-lineage
preflight全部通过。Run只加载fixed final-online checkpoint
`483c63ecaeb6dbf5a0a54400e0eecec722ff6df6d72226ce263e7fe053e412e2`，
没有optimizer、training update、checkpoint write/selection或official test。

固定D2-O cohort严格为64 sequences / 192 windows、phase offsets `(14,56,98)`、selection
SHA-256
`1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`。五条500-step
paths为`full_rho / unit_rho / relation_gate_ablated /
temporal_correspondence_permuted / left_right_role_swapped`；initial latent、每一步posterior
noise、condition、history、window ordering和restoration按注册协议配对。全部formal
lineage、schedule、current-state/current-timestep、first-window input identity、later-window
path-local causality、raw artifact closure、finite、mask和no-write contracts通过。

Learned alpha/gate为
`-0.09250371903181076 / -0.09224077314138412`，但七个primary mechanism gates全部失败：

- full-rho − unit-rho direct-hand union 5-cm F1：
  `-0.0017534958725371544`，CI
  `[-0.008731448896887319,0.005308128210568779]`；
- unit-rho − full-rho GT-contact-frame distance：
  `+0.026833171113742785 cm`，CI
  `[-0.08000149312555654,0.15226497063856367]`；
- full-rho − gate-ablated direct-hand union 5-cm F1：
  `+0.012897639098904238`，CI
  `[-0.027088665174081348,0.05341855160295159]`；
- gate-ablated − full-rho distance：
  `+0.4767763515114704 cm`，CI
  `[-0.05841018885359712,1.0529788793694062]`；
- full-rho − temporal-permuted direct-hand union 5-cm F1：
  `+0.002165564488603878`，CI
  `[-0.031244155095278628,0.043877801226154534]`；
- temporal-permuted − full-rho distance：
  `+0.23349129590381712 cm`，CI
  `[-0.31420466831855565,0.8945656086402927]`；
- full-rho − role-swapped direct-hand left/right macro-F1：
  `+0.013202214669035858`，CI
  `[-0.008038313502508903,0.035691421866214094]`。

因此`relation_path_used / schedule_reliability_passed /
temporal_routing_passed / role_binding_passed / mechanism_passed`均为false，
`internal_status=unused`，classification严格为
`diffusion-reliability-internal-unused-continue-native`。非零learned gate不替代causal
evidence；internal结果不选择checkpoint，也不取消已注册的唯一native evaluation。

Metrics、manifest、resolved config、preflight和run-local registry SHA-256分别为
`38e5a641707cff9a880fea5d4c90b7d936290e912237e60baa2ecf25bde8ff52`、
`1b3ad9792aa5df897982a482453561d2dfb7e56571404b19b060d9c16013ed96`、
`c05c69adeb171dd018b814df6d72197b59e4c6c70bbe0a6bd9a5eb0243fb1e21`、
`b1eb1aee4c3b2c0f67cc4a9bd810e7a3b7c570dd6db37028c4d44b610cb7ca7c`和
`dc1b0e8526ff636a9642be8262b126a908ed0efed18ecec175a480eefd754efd`。
五条raw paths、paired noise/conditioning、causal overlap和diffusion-reliability appendix
均逐项hash闭合。Worker发起non-destructive recovery后，17 files / 224,452,243 bytes的
worker/authority tree SHA-256均为
`5d28e3abc02dcf62f781270fd0391e44f64f4172b7ff705257995be63faffeee`，
checksum dry-run为0 differences。

Internal正式run之前，一个launch-control wrapper因未canonicalize Python symlink而在创建
run目录/manifest/artifact/GPU workload之前退出，正式run id未被占用；完成后一次tree-hash
wrapper因错误cwd出现`ModuleNotFoundError: tools`，但artifact已成功传输且随后在正确repo
root重算通过。两者均为operational wrapper noise，不是scientific/run failure，也不授权retry。

#### 2026-07-30 Phase 1B D2-AF0 fixed native evaluation completion

Reportable worker run `p1-hoi-d2af-native-eval-s42-20260730` 在相同clean commit
`a4cdcf09f84553159be10c555ff8a6773b65d3aa` 上完成，exit code为0，runtime为
`378.46187578188255 s`。Official evaluator执行438 sequences × 3 windows、500-step
unguided production diffusion、final-online、seed 42和10,000 paired sequence bootstrap；
CFG、guidance、scene conditioning、dynamic perception和consistency全部off。Sealed D2-X与
D2-AE aggregate/per-sequence artifacts只读复用，没有重生成；D2-AE checkpoint未加载，也未
用于initialization、resume、selection或candidate。

D2-AF0 target point estimates为：

- end-object `5.5734798312187195 cm`，Txy `4.607994854450226 cm`，
  FS `0.3595817663400341`；
- contact precision/recall/F1/coverage
  `0.7909342632567542 / 0.5990408567110541 /
  0.6410550040393033 / 0.4834384286439081`；
- Pbody `3.5893259727196254`，hand penetration `0.2268892808331516`；
- MPJPE `12.422095239162445 cm`，Troot/Tobj
  `8.447693288326263 / 17.303804395245994 cm`，Oobj
  `1.0007245875630288`。

第一优先级D2-AE single-factor repair gate失败：

- AF−AE contact F1：
  `-0.0008888491232585847`，CI
  `[-0.02435946905539561,0.022823955375799218]`；
- AF−AE recall：
  `+0.0029022340549778826`，CI
  `[-0.023672364012952776,0.02978128354401244]`；
- AF/AE end-object mean-ratio：
  `1.2920819122177505`，CI
  `[1.224156033921259,1.3639746920239342]`；
- AF/AE FS ratio：
  `0.9013030413462073`，CI
  `[0.8455771344455049,0.960204522985284]`，仅该subgate通过。

D2-X transfer也未通过：AF−X contact F1为
`+0.003629064933324414`，CI
`[-0.016673086777549785,0.024194001573353836]`；recall为
`+0.004586162666655739`，CI
`[-0.018407816634206476,0.02811076350108559]`；released gap closure仅
`0.040398463734960754`，contact F1点估计低于`0.6598838781`。

D2-X protection gate对end-object、Txy和Tobj失败，ratio CI upper分别为
`1.5730099778138071 / 1.1801967402975144 / 1.1099017284465105`；FS、Pbody、
hand penetration、MPJPE、Troot、Oobj和contact-precision checks通过。固定181-sequence
penetration finite mask通过，sequence-ID SHA-256为
`2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec`。
Released-95%-effectiveness gate失败。

按预注册classification priority，headline classification严格为
`diffusion-reliability-ae-repair-negative-stop`；`checkpoint_selected=false`、
`selectable_autonomous_diffusion_candidate=false`、`D2-AF1/consistency=false`，
HOIPrior search closed。

Metrics、manifest、resolved config、resolved target、preflight、run-local registry、
aggregate和per-sequence SHA-256分别为
`94fc71cd3d3fbbe87ac6ec38246e39fb0c965d630fd7626c604f0983a1248f56`、
`10498fb42a02501859cfd0aaab484a7a606ee5f711134d7f009519723655de06`、
`1aacaccee84eddaab141e3f4b31cfd3770db247d887512307ef279b3c549e4a9`、
`d956821b273fd70dec1aa2b5f58db4b16b40ebf8f5ba71895e2ae9b382d593fe`、
`8d9e8d2139e145487ea4193ca4a8b4b2fe89fc9afa8e0e6b5409b3238ff32fe9`、
`31c1571a599350036e9b5c57ee5f0f77ed34f3dea2aae4657188b8e94d792fcc`、
`417c245df047e4fd7724c7ddcc7f0884fffd5bda934fefe465fb904da400f488`和
`7252931861dd2d4e60476a05cd7dd35d67aa7369995687de2dc9bcbc67c8acd5`。
Worker发起non-destructive recovery后，16 files / 2,514,430 bytes的双端tree SHA-256均为
`40a9925468e54966f726b2cccec4f55aa53caa92f2a0da188dccc435ebc5bd21`，
checksum dry-run为0 differences。一次recovery wrapper最初只创建空staging目录而未传文件；
同一空目录随后通过无`--delete` rsync完整恢复并hash闭合，保守分类为post-completion
transfer-wrapper no-op，不改变任何workload/artifact/scientific result。

#### 2026-07-30 Phase 1B D2-AF0 recovery and final closure

D2-AF0的CPU、functional smoke、clean-signal eligibility、failed performance benchmark、
user-authorized waiver、formal checkpoint-race failure/continuation、fixed internal和fixed
native全部按append-only lifecycle保留。Formal/internal/native recovered trees分别为：

- 156 files / 7,227,356,886 bytes /
  `9bff3d9a182138ee30ca586b10d71f689e6aa0c7345d2b1052fe0ea15251dc6c`；
- 17 files / 224,452,243 bytes /
  `5d28e3abc02dcf62f781270fd0391e44f64f4172b7ff705257995be63faffeee`；
- 16 files / 2,514,430 bytes /
  `40a9925468e54966f726b2cccec4f55aa53caa92f2a0da188dccc435ebc5bd21`。

三棵树worker/authority完全一致；worker发起的
`rsync -aH --checksum --dry-run --itemize-changes`均exit 0、无itemized output。
Formal manifest/metrics/state/checkpoint cadence和80个rank RNG sidecars、internal九项raw
artifact closure、native aggregate/per-sequence/author log/resolved target均逐项hash验证。
所有operational wrapper/checkpoint-race记录保留，没有覆盖、删除或伪装为scientific pass。

Tracked compact result为
`experiments/results/p1_hoi_phase1b_d2af_sqrt_alpha_bar_reliability_s42_20260730.json`，
phase summary为`docs/phase_summaries/PHASE_1B_D2AF.md`。Final classification为
`diffusion-reliability-ae-repair-negative-stop`，internal classification为
`diffusion-reliability-internal-unused-continue-native`。Fixed final-online不可选择；
不merge/tag，不启动D2-AF1、第二次training、resume、checkpoint selection、performance/
architecture/worker sweep、consistency或任何新HOIPrior方向。

Phase 1B HOIPrior search在此关闭。本session不启动Phase 1C。下一独立session唯一entry point
是Phase 1C HSIPrior的dated plan-only preregistration；HSIPrior必须从seed-42随机初始化，
不得加载released/author/D2-X/D2-AE/D2-AF或任何HOIPrior checkpoint。

#### Phase 1C：HSIPrior 从零训练与原生域评测

在 `phase/01c-hsi` 上只训练 HSIPrior，固定使用 8×RTX 3090 服务器并沿用 1A 锁定过滤/split；
在该服务器上独立审计 micro-batch 和 `{512,1024,2048,3072}` 中的 effective batch。以 processed windows/frames 锁定 HSI
内部预算，联合预注册 LR/warmup；先短预算再完整训练，运行 LINGO/DIMOS 原生域指标并审计
normalization、文本、短序列、人景 penetration、FS、目标误差和不确定性。

Phase 1C 不得复制首次 Phase 1B“容量最大即正式 batch、teacher-forced loss 即模型选择”的错误：
容量只给出可行上限，正式档位必须在固定 processed-window 预算下以 LINGO internal native
rollout 选择。HSI 必须复用 Phase 1B D1 的 `WindowStateCodec` 人体字段、history/progress 与
global/local handoff；其 object/contact mask 保持 Phase 1A 不变，不引入 HOI 专用 BPS 或几何
loss。具体 batch/LR/warmup 仍须在 Phase 1C 开始前另行预注册，本次不创建分支或运行实验。

门槛：HSI 关键原生域指标达到对应单模型 baseline 至少 95%，无系统性 penetration/FS/FID
退化且 validation 无 scene-family leakage。通过后总结并 tag `exp/p1c-hsi-v1`。

#### Phase 1D：独立专家联合审计与 Phase 1 gate

在 `phase/01d-gate` 上不新增模型方向，仅汇总 single-seed-42 的最终专家结果，验证 checkpoint
provenance、参数不共享、各专家内部训练预算/effective batch 一致性、processed-window/frame
预算、完整 hash 和统计协议；补做预注册的
失败分层与专家不确定性对比，形成进入组合前的不可变 expert contract。联合 contract 还必须
证明两专家接受同一 232-D history、输出同一当前窗口坐标下的 clean x0、使用同一 codec 完成
global/local round-trip，并且组合不需要任何可学习或 expert-specific coordinate adapter。

门槛：1B/1C 均通过各自 95% 原生域门槛，且不存在系统性 contact/penetration/FID 退化；否则
Phase 1 不合入，不进入 Phase 2。通过后写 `PHASE_1D.md`，合入研究分支并 tag
`exp/p1-priors-v1`。

### Phase 2：固定组合可行性

先用 oracle state plan；实现共享噪声/timestep、坐标对齐、clean prediction 与固定身体组
门控。比较顺序路由、全局固定、时间、身体组、时空权重。TRANSPORT 默认 HSI 主导
root/下肢，HOI 主导手臂/手/物体。相对顺序路由，人景穿透至少约降 10%，task success 和
contact 各下降不超过 2 个百分点，才进入可学习 Mixer。失败检查 normalization、noise、
timestep、root trajectory 与进度对齐，必要时测解析式 product-of-experts。两专家预测必须先由
统一 `WindowStateCodec` 表达在同一 frame 后再门控，且只在组合完成后统一 decode/rebase；
若 Phase 2 需要 HOI/HSI 各自的学习式坐标修补器，视为 Phase 1 expert contract 未通过。

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

组件、数据和推理消融按预注册表执行。所有 phase 只使用训练 seed 42；生成指标的重复采样给
均值和 95% CI；同任务/seed 用 paired bootstrap 或 permutation，并校正主要多重比较。按物体、模式、
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
- 2026-07-13：修复后的 r1 audit 在 8×RTX 3090 上全部完成真实 optimizer update；每卡
  micro-batch 32/64/128 分别使用 accumulation 4/2/1，最大 reserved 显存为 2.881/4.855/
  6.713 GB。按预注册选择最大稳定值 128，并锁定 global effective batch 1024。至此 Phase 0
  全部门槛通过。
- 2026-07-13：在 Phase 1 实现前将其拆为 1A 数据/脚手架、1B HOIPrior、1C HSIPrior、1D
  联合验收，以满足单 session 单 phase/subphase 的交接约束；研究假设、数据范围和最终 95%
  原生域门槛不变，只增加各 subphase 的独立 deliverable、gate、summary 和 tag。
- 2026-07-13：Phase 1A 预注册假设：集中式 232 维 schema（84 人体位置、132 人体旋转、
  3 物体平移、9 物体旋转、4 contact）与显式域 mask，可以在不改变作者 OMOMO
  normalization/坐标语义的前提下，使无场景 OMOMO HOI 和 seed-42 scene-family-disjoint、
  无手持物体 LINGO HSI 分别完成从零初始化的真实 forward/backward；HSI 的 16 个
  object/contact 输出维度不参与 loss/梯度，HOI API 不接受 scene supervision，两套专家参数
  对象及 storage 完全独立，任何 released checkpoint 初始化请求立即失败。1A 仅执行数据审计、
  单卡功能 smoke 与 8 卡 global-effective-batch=1024 的一次 optimizer update，不进行训练筛选。
- 2026-07-13：Phase 1A 数据审计发现 no-hand LINGO 中 21,856 个窗口来自长度不超过 48 的
  sequence，其中 21,819 个 48-source-frame 窗口越过声明的 sequence end；按已有 mixed loader
  的有效性诊断规则在 no-hand filter 后排除这些窗口，不改变 seed-42 split。最终 HSI 为
  1,918,042 windows（train/validation 1,740,706/177,336），无 scene-family leakage。固定 OMOMO
  normalization 的 100,000-window 确定性抽查越界率为 0.1906%、最大绝对 normalized 值 1.0814，
  无 NaN/Inf；记录为轻微分布尾部，不重算 normalization。
- 2026-07-13：按研究目标修订 Phase 1 正式训练资源协议。Phase 0/1A 的 effective-batch=1024
  smoke 结果保持历史不可变，但不再约束 HOI 与 HSI 使用相同 batch/GPU 数。两专家分别在
  8×3090 或 4×3090 服务器上最大化与候选档位兼容的稳定 micro-batch；正式 effective batch 默认仅从
  `{512,1024,2048}` 选择，禁止 1536 等非 2 的幂值。公平性改为同一专家内部锁定协议，跨专家
  预算以 processed windows/frames 报告，optimizer updates 作为派生计数。该修订在 Phase 1B
  启动前完成，不重写 `exp/p1a-data-v1`。
- 2026-07-13：资源安排进一步固定为 HSIPrior 使用 8×3090、HOIPrior 使用 4×3090，并将 3072
  加入正式 effective-batch 候选。显存审计和具体 micro-batch/accumulation 选择分别留在 Phase
  1B/1C 内完成，不新增或重开 Phase 1A resource gate；`exp/p1a-data-v1` 保持不可变。
- 2026-07-13：固定双服务器执行拓扑：`10.184.17.253` 作为权威开发/集成端和 Phase 1C HSI
  主机，`10.181.9.214` 作为只执行已提交代码的 Phase 1B HOI worker。新增 Git 单向发布、不可变
  OMOMO-only 数据 snapshot、环境复制/验证、artifact 回收和 append-only registry 单写者协议；
  该运维修订不启动 Phase 1B，也不改变其研究假设、gate 或 batch 候选。
- 2026-07-13：四卡 worker 实测为 Ubuntu 20.04/glibc 2.31、4×RTX 3090 24GB、driver
  580.126.09、125 GiB RAM、`/home` 可用 4.0 TiB，时钟/NTP 正常；与八卡机 OS/glibc 兼容。
  八卡机到 worker 的 TCP/22 超时，但 worker 到八卡机 TCP/22 可达，因此将 transport 固定为
  worker 主动 pull/push，并把 worker 根目录固定为 `/home/yujinlun/data`。首次检查时 GPU 2 有
  外部进程占用约 3.5 GiB；Phase 1B reportable 运行前必须清空四卡或显式登记 contention。
- 2026-07-13：HOI-only worker 首次全测试按预期暴露两项部署资产假设：通用测试强制读取未同步
  的 LINGO norm，HOI FK 数据路径需要尚未同步的 `smpl_models`。不复制 LINGO；以显式
  `INFBAGEL_WORKER_EXPERT=hoi` 仅跳过真实 LINGO 文件测试，同时继续运行 HSI schema/mask/API
  测试，并将 `smpl_models` 纳入 hash-verified HOI worker 资产。该诊断未启动 GPU workload。
- 2026-07-13：四卡 HOI worker provisioning 关闭通过：代码 commit/tag、packed environment、
  scene-free OMOMO snapshot、SMPL、全量 rsync checksum 和 Phase 1A audit hashes 均一致；worker
  role 下 32 tests 为 30 pass/2 个真实 LINGO 文件测试 skip；空闲 GPU 0 完成 batch-2 随机初始化
  真实 forward/backward，loss 1.460899、峰值 reserved 102,760,448 bytes。该 non-reportable smoke
  不作显存容量或 batch 决策；GPU 2 当时有外部进程，Phase 1B 四卡 audit 前仍须四卡空闲或登记竞争。
- 2026-07-13：为 Codex/权威端增加由 worker 主动建立的 loopback-only reverse SSH control plane；
  八卡端只监听 `127.0.0.1:22214`，worker host key 必须独立核验，禁止 `GatewayPorts` 或 LAN 暴露。
  该通道已捕获 `node01` Git/GPU 状态并在 GPU 0 完成最小 CUDA 张量验证；它仅用于启动、观察和
  获取输出，不改变 Git/rsync 仍由 worker 主动发起的所有权，也不允许长训练依赖 SSH 存活或把
  终端输出当作唯一实验记录。该运维修订不启动 Phase 1B。
- 2026-07-13：Phase 1B 在任何实现/GPU workload 前锁定 scene-free HOI Transformer、内部
  sequence-disjoint validation、四档容量审计及 headroom 判据、两个 LR/warmup 短预算候选、
  61,440,000-window 正式预算、checkpoint/evaluation cadence、native/CHOIS 统计协议和
  95% baseline 判据。官方 438-sequence test 只在配置锁定后使用；released checkpoint 仍只作
  baseline，绝不初始化或恢复 HOIPrior。
- 2026-07-14：Phase 1B 首次 seed-42 正式训练虽稳定完成，但 native/CHOIS 能力 gate 失败；在
  不扩大模型、不引入 CFG/scene/SDF/mixer 的前提下预注册修复重试。先用既有 checkpoint 做
  多 timestep 与三窗口 rollout 诊断，再恢复被遗漏的 pelvis endpoint condition、统一窗口状态
  codec、逐窗口旋转/BPS 重基准并将物体目标约束限于真实 terminal window；随后以相同
  processed-window 预算比较 effective batch
  1024/3072，最多允许一次不改变专家接口的几何 contact fallback。正式重训仍只用 seed 42、
  scene-free OMOMO 和随机初始化，官方 native/CHOIS 仅在配置锁定后各评测一次。Phase 1C/1D/2
  同步约束为复用同一状态 codec 与接口，不允许靠专家特有坐标修补增加组合复杂度。

每个阶段只允许上文给出的诊断/fallback。新增方向必须先在此处追加日期、证据和原因，并在
registry 登记，再实现代码。
