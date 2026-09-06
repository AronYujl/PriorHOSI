# 状态条件 HOI/HSI Prior 组合：总览、接口与跨阶段约定

本文件于 2026-08-10 从 `docs/EXPERIMENT_PLAN.md` 第 14-40、41-63、64-74、97-121、8300-8341、8342-8361、8362-8478 行原样切出（逐字节复制，未改写、未重排、未修正任何笔误）。
导航：[计划入口](../EXPERIMENT_PLAN.md) · [Phase 1B 索引](PHASE_1B_HOI/README.md)

2026-09-06 当前进度：Phase 2.9 已完成，交付与本轮质量门槛均通过。Armijo 在 744 次
修正中保持完整目标不增，A00 滑步接近重建基准，A01 物景穿透指标比重建下降 13.20%。
相对原 Adam 版 A01，人景与物景穿透深度均显著回升；保留这项质量取舍以及边界查询问题。
重建仍为比较锚点，Armijo 保留为通过本轮门槛的候选。完整 Phase 2、有效 HSI 监督与
learned mixer 训练仍开放，详见 [Phase 2.9 交接](../phase_summaries/PHASE_2I_ARMIJO.md)
和 [Phase 2 计划](PHASE_2_COMPOSITION.md)。

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
