# 状态条件 HOI/HSI Prior 组合的 HOSI 实验计划

状态：Phase 0、Phase 1A 已通过；Phase 1B 首次训练已稳定完成但能力 gate 失败，现已预注册修复重试；Phase 1C 未启动；基线提交 `b9a158f75ab0740c91c9cfc8863a65fa381b014c`<br>
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
