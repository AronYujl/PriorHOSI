# 状态条件 HOI/HSI Prior 组合的 HOSI 实验计划

状态：Phase 0、Phase 1A 已通过；Phase 1B 首次训练及 D2-F remediation 均未通过，现已预注册 D2-H reverse-state exposure 诊断与条件式修复；Phase 1C 未启动；基线提交 `b9a158f75ab0740c91c9cfc8863a65fa381b014c`<br>
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
