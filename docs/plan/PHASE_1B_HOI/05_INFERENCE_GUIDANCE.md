# Phase 1B-05：基线协议分解与推理期接触引导（P1、P2、P3、P5、P6）

本文件于 2026-08-10 从 `docs/EXPERIMENT_PLAN.md` 第 7435-7677、7934-8270、8883-9107 行原样切出（逐字节复制，未改写、未重排、未修正任何笔误）。
导航：[总览](../OVERVIEW.md) · [Phase 1B 索引](README.md)

#### 2026-08-01 Phase 1B 基线协议分解 P1（released baseline 协议归因，用户批准）

动机：`results/experiments/p0-hoi-table5-baseline-s42-20260712/resolved_config.yaml` 记录该行
实际以 `sample_type: consistency`(:29)、`cm_timesteps: 16`(:142)、`guidance_weight: 1`(:26)、
`w: 1`(:9，CFG 开启)、`load_scene: true` + `add_object_voxel: true`(:17,23) 产生，执行 commit
为 `c358fa4`。而全部 D2-* native 行为 500 步无引导 ancestral DDPM、无 CFG、`load_scene=false`。
二者至少在采样器、步数、引导、CFG、逐步物体占据条件、模型架构六个轴上同时不同，因此
released 与 D2 之间 0.1331 的 contact recall 差（0.72759 vs 0.59445）此前从未被归因。

记录缺陷（本次一并更正）：`docs/HOIPRIOR_EVIDENCE_INDEX.md:11-13` 与
`experiments/results/p1_hoi_phase1b_d2aa_integrated_table_s42_20260724.json` 的 protocol 字段
将 released 行描述为 500 步 native 协议，与其 resolved_config 直接矛盾。

代码轴：`code/test_infbagel_hoi.py` 的自回归 rollout 块在 `ffc548a`(2026-07-13) 后被重写，
`object_points_batch` 与 `obj_bps_data` 的「每步重算 / 冻结」关系反转；released 行在重写前，
全部 D2 行在重写后，故重写本身构成第七个未测量的轴。

设计：released checkpoint 仅作基线（不用于任何初始化），官方 438 序列，2×3 共 6 次纯推理，
不训练、不产生 checkpoint、不改动任何模型或训练代码。

| commit | cm=16 gw=1 | cm=16 gw=0 | cm=1 gw=0 |
| --- | --- | --- | --- |
| `c358fa4`（已发表协议） | A0-old：复现闸门 | A-old：引导份额 | B-old：迭代＋逐步重查询份额 |
| `HEAD`（D2 同 rollout） | A0-new：rollout 漂移份额 | A-new：引导份额稳健性 | B-new：纯 NFE 份额 |

统计沿用既有协议：paired sequence-level bootstrap，seed 42，10000 replicates。
`code/eval_metrics.py` 在 `c358fa4` 与 `HEAD` 之间逐字节相同（`git diff` 为空），
`code/guidance_loss.py` 亦逐字节相同（`5747721b…`），故指标与引导公式不构成额外轴。

闸门：A0-old 须复现 contact F1 `0.7272576950146546` 与 recall `0.72759`。期望逐位复现；
若不复现，先诊断偏差来源，其余五格结果不得用于归因解释，只能作为 HEAD 协议内部比较。

预注册判定规则（记 `R_g`=A0-old recall，`R_u`=A-old recall，`R_x`=D2-X 0.59445，
引导份额 `g = (R_g − R_u) / (R_g − R_x)`）：

- `g ≥ 0.5`：差距主要由推理期引导造成，过去的 released-vs-D2 比较不公平；后续可另行提议在
  同等预算下为 HOIPrior 启用引导作为协议对齐，但必须同时报告 foot sliding、物体平移 MAE 与
  pelvis goal 的代价（D2-Q0 曾测得 foot sliding 比值 1.5350）。
- `g ≤ 0.2`：引导贡献小；为 HOIPrior 加引导属于对被测指标的测试期优化，明确放弃该杠杆，
  转向 B 格所度量的逐步物体占据重查询。
- `0.2 < g < 0.5`：不足以单独判定，须结合 B 格与 rollout 漂移份额再议。

边界：本次不触及 `hoiprior_search_closed`，不改动 native D2 协议，不新增 checkpoint。是否为
HOIPrior 启用 production guidance 或逐步重查询，取决于本次结果，仍需另一次带日期的修订与
用户显式授权（沿用 D2-Q0 的 `next_action` 与本节上文 `不得自动采用 production guidance`）。

流程精简（用户 2026-08-01 授权）：本次为纯推理基线测量，不分配训练 run、不走
`tools/experiment.py start` manifest；可回溯性由「执行 commit + 完整命令行 + Hydra
resolved_config + 官方 438 固定测试集 + 逐字节相同的指标代码」承担。

结果（2026-08-01 执行完毕）。闸门以最强形式通过：A0-old 在 `c358fa4` 上逐位复现已发表行，
sealed aggregate 的全部 18 个指标精确相等（其中非比值子集 16 项）；六次运行各自加
`save_motion_params=true` 重跑一遍，108/108 指标逐位一致，证明管线完全确定性且跨 worktree
稳定。配对序列级 bootstrap（seed 42，10000 replicates，
n=438，沿用 `tools/summarize_hoi_phase1b.py:112` 约定，索引矩阵与仓库既有三处实现逐位相同）：

| 对比 | contact recall | contact f1 | foot sliding（低者优） |
| --- | --- | --- | --- |
| 引导（A0-old − A-old） | +0.0788 [+0.0649, +0.0931] | +0.0588 [+0.0475, +0.0707] | −0.0357 [−0.0505, −0.0209] |
| 引导（HEAD rollout 复现） | +0.0836 [+0.0687, +0.0988] | +0.0612 [+0.0491, +0.0737] | −0.0496 [−0.0650, −0.0344] |
| 迭代 cm16→cm1 | +0.0085 [−0.0046, +0.0213] | +0.0029 [−0.0079, +0.0137] | +0.0184 [+0.0046, +0.0319] |
| rollout 重写 | −0.0084 [−0.0190, +0.0022] | −0.0047 [−0.0130, +0.0035] | +0.0092 [−0.0026, +0.0210] |

判定：`g = 0.5920 ≥ 0.5`，落入预注册的第一档。迭代次数对 contact F1 为零效应（两套 rollout
下 CI 均跨零），rollout 重写在三项接触质量指标上均为零效应。引导**改善** foot sliding，
与 D2-Q0 所测方向相反；已从源码确认 D2-Q0 的 `code/priors/contact_guidance.py` 与 D2-R0 的
`code/priors/routed_guidance.py` 对 `apply_feet_floor_contact_guidance` 的引用次数均为 0，
即二者只实现了作者损失的手-物 ×10 项，缺失脚-地 ×500 项。

勘误（由本次分解直接导致，记录于此，不修改任何已封存哈希绑定件）：

1. 登记门槛 `0.6598838781 = F1_X + 0.25·(F1_released − F1_X)` 是跨协议常量。F1 差距 0.08983
   的分解为引导 0.05884（65.5%）、迭代 0.00290（3.2%）、真实模型差距 0.02809（31.3%）；
   协议对齐后的 25% 闭合门槛应为 ≈ `0.64445`。
2. 协议对齐后重算的 gap closure：D2-AG `0.451`（原记 0.1409）、D2-AC `0.376`（原记 0.1176）
   均达到 ≥0.25；D2-AE `0.161`、D2-AF `0.129` 仍不达标，D2-AD 两种口径下均为负。
   **各运行的总判定不变**——D2-AG 仍因 F1 配对 CI 下界 −0.0082 与 foot sliding 比值 CI 上界
   1.184 而失败——变的是记录在案的失败理由集合。
3. `released_95_percent_effectiveness` 含协议成分：D2-AG contact F1 项 0.894 不通过、协议对齐
   后为 0.973 通过；recall 0.823→0.923、foot sliding 1.203→1.086 仍不通过。故「11 项通过 6 项」
   不是一个纯模型陈述。
4. `experiments/results/p1_hoi_phase1b_d2aa_table5_completion_s42_20260724.json` 的
   `.local_protocol.native_quality` 声明 `sample_type: diffusion, diffusion_steps: 500,
   guidance:false, cfg:false, scene:false`，而该表包含 released 行；该声明对 released 行为假。
   该件被 registry ×3、docs ×4 哈希绑定，按既有先例不就地修改，在此登记事实。

追加测量（同日，同为纯推理）：`p1-hoi-d2x-distance-probe-s42-20260801` 与
`p1-hoi-d2ag-distance-probe-s42-20260801`，对已封存的 D2-X 与 D2-AG final-online checkpoint
按原生协议（500 步无引导 diffusion，无 CFG，`load_scene=false`）重跑并开启
`save_motion_params=true`，目的是取得 HOIPrior 自身的**绝对** GT-contact 帧手-物距离——
该量此前从未入档，D2 各结果只存了消融对照的差值。两次探针均逐位复现其在 worker 上封存的
aggregate（各 18/18 指标），因此本机单机评估与已退役的双机 worker 评估结果完全一致。

距离定义与冻结指标内部一致：SMPL 手关节 22/23 到姿态化物体最近网格顶点，帧池化
（GT×GT 参考实测 1.6981 cm，与在档的 1.70 同口径）。GT-contact 帧均值：released 有引导
3.6836、released 无引导 4.5360（cm16）/ 4.7080（cm1）、D2-X 5.3886、D2-AG 5.3129 cm。
配对 95% CI（n=397，正值表示 HOIPrior 离物体更远）：`D2-X − A-old` **+0.8334
[+0.4906, +1.1824]**，`D2-AG − A-old` +0.8686 [+0.5023, +1.2660]，`D2-X − A0-old`
+1.6587 [+1.3050, +2.0182]，`D2-AG − D2-X` +0.0351 [−0.3247, +0.4150]（跨零）。

由此得三条结论：其一，协议对齐后 HOIPrior 的原始生成几何确实更差 0.83 cm，released 在
已发表行上的 1.66 cm 优势约各半来自真实先验质量与推理期引导；其二，**D2-AG 在几何上与
D2-X 不可区分**，其较高的 contact F1 来自操作点移动而非把手放得更近（同期 foot sliding
由 0.3630 恶化至 0.4009）；其三，失败模式是**大幅偏离而非临界漏检**——D2-X 漏检的
GT-contact 帧中 57.9% 位于 ≥8 cm，仅 16.9% 位于 [5,6) cm。第三条否定了此前
「模型知道何时接触、只差几厘米几何」的字面表述；该表述所依据的 `4.65 cm` 实为 D2-Q0
带引导变体左手对 GT 几何距离的 p25，同分布中位数为 10.76 cm，不能作为中心趋势使用。

附带发现（影响项目内每一个 recall 数字）：438 个序列中有 41 个不含任何 GT 接触帧，
`code/eval_metrics.py:316-320` 在 `TP+FN==0` 时将 recall 记为 0，而聚合为 438 序列无权平均。
因此所有 recall 被同一常数因子 `397/438 = 0.9064` 压低，绝对值与差值等比压缩（真实差值需
乘 1.103），份额与排序不受影响。

#### 2026-08-01 Phase 1B 推理期接触引导 P2（协议对齐，用户批准）

动机：2026-08-01 基线协议分解测得推理期引导占 released-vs-D2X contact recall 差距的 59.2%、
F1 差距的 65.5%，且在 released 模型上**同时改善** foot sliding。HOIPrior 的固定原生协议为
500 步无引导采样，因此此前所有对比中 baseline 带引导而 HOIPrior 不带。本实验为**协议对齐**，
不是模型改进：即使成功也不改变已测得的 0.83 cm [0.49, 1.18] 真实生成几何差距。

合法性前置（已查证，读代码）：引导的每一个输入都来自模型预测或给定资产，**无测试期 GT 泄漏**。
`contact_labels = x_start[:,:,228:232]`，`x_start = pred_x_0.detach().requires_grad_(True)`
（`code/models/infbagel.py:721,677`）；脚-地项的地面高度是硬常数 `0.02`
（`code/guidance_loss.py:82`）；GT 接触标注仅作为窗口 0 的两个种子历史帧进入，且无引导路径
完全相同。作者原始实现 CHOIS 的 GT `contact_labels` 形参在 `trainer_chois.py:2024` 被遮蔽
且从未读取，语义一致。

目标 checkpoint：**D2-X**（`p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723` final-online，
sha256 `b0fa6bdd…`）。不选 D2-AG，因其自条件关系源消费 `current`（`code/priors/diffusion.py:234-236`），
而引导恰好修改 `current`，构成不受控的分布偏移。
对照：同一 checkpoint 的无引导原生评估，已在本机逐位复现其 worker 封存值（18/18）。

损失：作者完整 `apply_hoi_guidance_loss`（`code/guidance_loss.py:88-94`），
即手-物 ×10 **加** 脚-地 ×500。此处与 D2-Q0/D2-R0 的关键差别是后者对
`apply_feet_floor_contact_guidance` 的引用次数为 0，只实现了手-物项；且 D2-Q0 的门槛
checkpoint 物体目标误差为 95.3 cm（D2-X 为 3.74 cm），其 foot sliding 比值 1.5350 是在一个
显著更弱的模型上、缺失脚部保护项的条件下测得，不足以否定完整损失。

事先声明的双臂设计（非事后 sweep；两臂无论结果如何都报告，A 为标题结果）：

- **Arm A（主）**：InfBaGel 忠实移植——完整损失，scale 1.0，梯度对 `x0_hat` 求取后原样加到
  `x_{t-1}`，除最后一步外全程引导。这是**效应量被实测过**的那个配置的直接移植。
- **Arm B**：CHOIS 式 DDPM 对应物——仅末段引导、梯度按 `posterior_variance[t]` 缩放并裁剪，
  依据 `chois_release/manip/model/transformer_object_motion_cond_diffusion.py:520`（1000 步中
  仅 `0 < i < 10`）。动机是稳定性：作者只在 15 步上施加未裁剪原始梯度，而 HOIPrior 有 499 步。

确定性：使用 D2-Q0 的确定性顶点子集，不用作者的 `torch.randperm(...)[:10000]`
（`code/models/infbagel.py:736`），否则同一配置的两次运行不可复现。

功能性 smoke 仅检验**可运行性与有限性**（`nonfinite_values == 0` 为被门控项），不用于在两臂之间
做选择；两臂均在官方 438 上完整运行。

必报代价（D2-Q0 正是失败于此类项）：foot sliding、物体平移 MAE、pelvis goal、end-object、
穿透、MPJPE，以及 `position_outside_rate`（`code/priors/diffusion.py:487`，引导必然抬高该值，
此前无 D2 运行审计过该区间）和 GT-contact 帧手-物距离分布。

已知混淆，将在结论中写明而非略去：其一，引导需要完整物体网格（13086–38353 顶点），而模型
仅条件在 1024 点 BPS 上，属于评估器已使用但模型未见的额外测试期几何；其二，FS 的改善可能
部分由指标形状造成——FS 度量自**预测**关节估计地面（`code/test_infbagel_hoi.py:249`，
`code/eval_metrics.py:101-113`），而引导将支撑脚钉在固定 0.02，二者可能互相自洽。

统计与协议沿用既有：官方 438、三窗口、冻结指标代码、配对序列级 bootstrap（seed 42，
10000 replicates）。不改训练、不产生 checkpoint、不改 500 步计划表。

实现期补充规格（2026-08-01，实现时确定，与上文所引先例一致，记录于此以免事后被当作挑选）：
Arm B 的 `guidance_scale` 取 **1000.0**，即所引 CHOIS `classifier_scale = 1e3`；若沿用 Arm A 的
1.0，方差缩放会使更新量缩小约 3.8e-4，Arm B 将退化为一个意外的空臂。Arm B 的裁剪作用于
**缩放后的更新量**（`clamp_target=update`，界 1.0），与上文「梯度按 posterior_variance[t]
缩放并裁剪」的措辞一致。配置键置于 `code/config/sampler/hoi_prior.yaml`，因为
`code/config/config_eval_hoi_prior.yaml` 与 `code/test_infbagel_hoi.py` 被
`tests/test_hoi_d2{ac,ad,n}.py` 逐字节钉死，改动它们会使既有封存件失效。

事前预测（在任何 GPU 运行之前记录）：以真实 `norm.npy` 估算，Arm A 对归一化到 [-1,1] 的
pelvis-y 通道，在 5 cm 脚部误差时单步加入 ≈2.06、30 cm 时 ≈19。因此**预期 Arm A 会抬高
`position_outside_rate` 乃至产生退化运动**——这正是上文所述「作者只在 15 步上施加未裁剪原始
梯度而 HOIPrior 有 499 步」这一风险的定量形式。若 Arm A 失败而 Arm B 成立，该结果应读作
「作者配置不能按字面移植到多步 DDPM」，而非「推理期引导对 HOIPrior 无效」。
另记一处继承自作者实现的数值风险：时间项做 `v/‖v‖` 且无 epsilon
（`code/guidance_loss.py:58-64`），掌心恰落于物体中心时产生 NaN；由已门控的
`nonfinite_values == 0` 兜底。

结果（2026-08-02 执行完毕，官方 438，配对序列级 bootstrap，seed 42，10000 replicates）。
功能性 smoke 与两臂全量运行的 `nonfinite_values` 均为 0，`position_outside_rate` 均为 0.0，
**上文事前预测的 Arm A 饱和与退化运动没有发生**——最坏情况估计未兑现，很可能因为接触达成后
损失趋零、引导自限。该预测在任何 GPU 运行之前写入，此处如实记录其被证伪。

| 指标 | D2-X 无引导对照 | Arm A | Arm B | released 带引导 |
| --- | --- | --- | --- | --- |
| contact f1 | 0.63743 | **0.73071** | 0.72653 | 0.72726 |
| contact recall | 0.59445 | 0.72332 | 0.70783 | 0.72759 |
| contact precision | 0.78806 | 0.80216 | 0.81223 | 0.79081 |
| foot sliding | 0.36301 | 0.39903 | 0.35565 | 0.33336 |
| obj_trans_dist | 15.99405 | 17.23781 | 16.09432 | 15.72565 |

配对差值（`*` 表示 CI 跨零）：Arm A − D2-X 的 contact f1 `+0.0933 [+0.0730, +0.1146]`、
recall `+0.1289 [+0.1046, +0.1537]`；Arm B − D2-X 的 f1 `+0.0891 [+0.0759, +0.1022]`、
foot sliding `−0.0074 [−0.0242, +0.0096]*`。

**主结论：协议对齐后，带引导的 HOIPrior 与带引导的 released 模型在接触上达到统计学平价。**
Arm A − `a0-old` 的 recall `−0.0043*`、precision `+0.0113*`、f1 `+0.0035*`、
contact_percent `−0.0120*` **全部跨零**；Arm B − `a0-old` 的 f1 `−0.0007*` 亦跨零。
Arm A 与 Arm B 在 recall/precision/f1 上互不可区分。

**引导改变的是几何而非仅操作点。** GT-contact 帧手-物距离（同一 joint-to-vertex 帧池化口径）
均值：D2-X `5.3886`、Arm A `3.9759`、Arm B `4.1725`、released 带引导 `3.6836`；中位数
`3.1835 / 1.7178 / 2.1396 / 1.8012`；[0,2) cm 质量由 `0.359` 升至 `0.547`（Arm A）。
配对位移 Arm A − D2-X `−1.4371 [−1.7204, −1.1532]`，CI 远离零；重新划阈值无法产生分布整体
位移。**Arm A − `a0-old` 在连续残差上为 `+0.2216 [−0.0882, +0.5310]*`，同样跨零**，即平价在
未经阈值化的量上也成立。

代价，按 D2-Q0 的比值表达式（`tools/run_hoi_d2n.py:407` 形式）给出以便直接对照：

| 比值（对无引导对照） | Arm A | Arm B |
| --- | --- | --- |
| obj_trans_dist | 1.0778 [1.0571, **1.099831**] | 1.0063 [1.0000, 1.0128] |
| foot_sliding | 1.0992 [1.0374, **1.1661**] | **0.9797** [0.9354, 1.0276] |

判据说明（避免事后造判据）：**P2 预注册要求必报代价，但未预注册代价阈值**；D2-Q0 的
`≤1.10` 是为 D2-Q0 预注册的，不构成 P2 的门槛。此处不宣布依该线的通过或失败，只如实记录：
**若套用该线，Arm A 在 foot sliding 上以 CI 上界 1.1661 失败、在 obj_trans_dist 上以 1.7e-4
之差擦过；Arm B 两项皆干净通过。** Arm A 另付 MPJPE `+0.2958 [+0.1287, +0.4727]`、
obj_trans_dist `+1.2438 [+0.9168, +1.5868]`、end_obj `+0.4720 [+0.3014, +0.6450]`；
Arm B 对应为 `+0.0285*`、`+0.1003 [+0.0003, +0.2036]`、`+0.0992 [+0.0331, +0.1658]`，
且手部与人体穿透点估计均改善。

因此：**Arm A 为预注册主臂且 contact f1 最高，但 Arm B 是经得起代价审视的配置**——按分量计，
Arm B 取得 Arm A 增益的 recall 88.0%、f1 95.5%、contact_percent 84.7%、距离 84.5%，而物体
轨迹代价近乎为零，foot sliding 反而改善。两臂结果均按预注册报告，不因 Arm B 更好而改写主臂声明。

**D2-Q0 的阴性结论被推翻。** 该子阶段以 foot sliding 比值 1.5350 停止，而其实现对
`apply_feet_floor_contact_guidance` 的引用次数为 0，且其 checkpoint 物体目标误差为 95.3 cm。
使用作者完整损失后，Arm B 的 foot sliding 比值为 0.9797（改善），Arm A 为 1.0992。

残留缺口（引导不能修的部分）：Arm A 的远尾仍厚于 released（≥8 cm 占比 0.127 vs 0.105），
且相对 `a0-old` 仍付出物体轨迹与穿透代价。这与 2026-08-01 测得的 0.83 cm 真实生成几何差距
一致——**接触上的失败是测量假象，其余指标上的差距是真的。**

测量口径提醒：`end_obj_trans_err` 的逐序列构造与已发表 aggregate 并非同一量（aggregate 用
插值前的窗口端点，逐序列用插值后轨迹末帧，相对差 Arm A 0.533%、Arm B 0.658%、对照
0.692%），故其 CI 属逐序列构造；
`mpjpe` 仅在 1e-7 相对量级上因池化次序不同而异；`obj_trans_dist` 与两项穿透为精确一致。
穿透 CI 基于 181 个序列，因冻结代码对六类物体不计算该项。

#### 2026-08-02 Phase 1B P3 关系场谱系 × 推理期接触引导（用户批准）

动机：D2-AG 在无引导原生协议上是**穿透最好的 D2 模型**，且该优势不是"够不着"造成的参与度
假象。D2-AG 与封存 D2-X 的接触参与度**基本相同**——`contact_percent` `0.4770600` vs
`0.4765530`、`contact_recall` `0.5984951` vs `0.5944551`——而 `hand_pen_loss_omomo` 为
`0.1836719` vs `0.2453570`（低 25.1%）、`contact_precision` `0.8111962` vs `0.7880620`。
D2-AG 的 `hand_pen_ratio` `0.1128650` 已经**优于 released 带引导行的 `0.1328600`**。这与
`EP:7912-7916` 记录的 epoch100 情形正相反：那里穿透"变好"伴随 `contact_percent` 跌到 D2-X 的
0.670，是典型的参与度假象；这里参与度持平而穿透真实下降。D2-AG 相对 D2-X 的唯一回退是 foot
sliding（`0.4009183` vs `0.3630100`），而 P2 已测得 **Arm B 的独有性质恰是改善 foot sliding**
（对 D2-X 的比值 `0.9797`，`EP:7650`）。P2 之后相对 released 带引导行仅存的两个最大真实缺口正是
手部穿透（`0.2294200` vs `0.1624020`，+41.3%）与人体穿透（`3.6260650` vs `2.5892670`，+40.0%）。

本实验因此补齐一个 **2×2 因子设计**：{checkpoint: D2-X, D2-AG} × {sampler: 500 步无引导,
500 步 + P2 Arm B}。三格已封存、**只引用不重跑**：

| 格 | run | 树 |
| --- | --- | --- |
| D2-X 无引导 | `p1-hoi-d2x-native-eval-r1-s42-20260723`（本机复现件 `p1-hoi-d2x-distance-probe-s42-20260801`） | worker / `5f7dde7` |
| D2-AG 无引导 | `p1-hoi-d2ag-native-eval-s42-20260801`（本机复现件 `p1-hoi-d2ag-distance-probe-s42-20260801`） | `9d77a6f` / `5f7dde7` |
| D2-X + Arm B | `p1-hoi-p2-guidance-armb-s42-20260801` | `c40dc00` |
| **D2-AG + Arm B** | **本次唯一新增** | `c40dc00` |

**事先声明的第二臂：D2-AE + Arm B。** 目的是在**带引导**的体制下把"稀疏关系场本身"与"关系场
加自条件"分开：D2-AE 与 D2-AG 结构相同、参数数相同（`30,087,401`），唯一差别是变量锚
`5/10/15` 读 `x_t` 还是读 `sg[x0_hat]`。两臂无论结果如何都报告；第二臂不是主结论的依据。

**post-hoc selection 的明确披露。** D2-AG 是**在看到其穿透数字之后**才被选中的——它本身是
`selfcond-relation-source-transfer-negative-stop` 的阴性方向，其 checkpoint 按 `PHASE_1B_D2AG.md`
不可选、不可复用于初始化。本节把这一点写在最前面而不是脚注：这是**对已封存 checkpoint 的事后
挑选**，因此本预注册**同时锁死 checkpoint 集合**——只有 D2-AG 与 D2-AE 两个目标，两臂都必须
报告，**不得再增加第三个 checkpoint、不得在看到结果后换目标、不得把某一臂改称"探索性"**。
本实验不产生 checkpoint、不做 checkpoint selection、不解除 `hoiprior_search_closed`，也不改变
D2-AG/D2-AE 的既有阴性分类。

**对 P2 排除 D2-AG 之决定的显式推翻（不得静默进行）。** `PHASE_1B_P2_GUIDANCE.md:69-72` 与
P2 registry row 明文记录"不选 D2-AG，因其自条件关系源消费 `current`，而引导恰好修改 `current`，
构成不受控的分布偏移"。本节推翻该排除，理由是读代码与已测证据两条，都可查证：

1. **引导路径与架构无关，且不触碰关系场读取的任何量。**
   `HOIContactGuidance.apply`（`code/priors/inference_guidance.py:352-385`）只消费
   `clean`（`prepare_clean_x0` 之后的 x0_hat）、`codec`、`frame`、`rest_human_offsets`、
   `parents_24`、`rest_vertices`，不接触 relation field、BPS 或任何 D2-AE/AG 专有张量；
   其返回值在 `:382` 处把 `result[:, :REPRESENTATION.history_frames] = fixed_history` **重新钉住**，
   因此 D2-AG 关系源的 `s[:, :2] = current[:, :2]` 契约（`sparse_relation.py:447-448`）
   **逐位不受引导影响**。
2. **残余耦合是间接的，且已被实测限定。** 唯一的 D2-AG 专有暴露是：下一步的 `prev_x0`
   （`diffusion.py:276`）是在被引导过的 `x_t` 上算出的 x0_hat。但 D2-AG 的固定内部因果诊断
   已**以良好功效把关系源的因果效应界定在零附近**：把源整体替换回 `x_t` 只改变 union 5-cm F1
   `-0.00411124`，CI `[-0.01317134, +0.00524494]`（`PHASE_1B_D2AG.md:263`），半宽约为两个显著
   效应（`0.305`/`0.184`）的十分之一。**模型不使用该路径的来源信息，所以经由该路径的分布偏移
   其效应有上界。** 这不是"分布偏移不存在"，而是"分布偏移经过一条已被证明惰性的路径"。

该推翻仍留一个真实风险，写在下文风险节，不掩盖：上述界定来自**无引导**轨迹上的诊断，引导后的
`x_t` 严格说在诊断的分布之外。

被操纵因子（单一概念，两个取值）：sampler ∈ {500 步无引导 ancestral DDPM，500 步 + P2 Arm B}。
Arm B 逐字沿用 P2 已执行配置：`arm=b`、`guidance_scale=1000.0`、`last_steps=10`、`clamp=1.0`、
`clamp_target=update`、确定性顶点子集、作者完整 `apply_hoi_guidance_loss`（手-物 ×10 + 脚-地 ×500）。
**不新增、不调整任何引导超参**；改动其中任何一个都会构成未登记 sweep。评测协议与三个已封存格
完全一致：官方 438 序列 × 3 窗口、`load_scene=false`、`sample_type=diffusion`、无 CFG、seed 42、
冻结指标代码、配对序列级 bootstrap（seed 42，10000 replicates，`tools/summarize_hoi_phase1b.py:112`
约定），比值用 `tools/run_hoi_d2n.py:407` 形式。**不训练、不产生 checkpoint、不重跑任何已封存格。**

执行树与树效应控制（在任何 GPU 运行之前固定）。2026-08-02 已测得跨树 `end_obj_trans_err`
`+11.0%` 的树效应（`EP:7884`），故本节不得默认树可比：

- 全部新运行在 pinned worktree `/data/yujinlun/InfBaGel-p2`、commit
  `c40dc00b2ad315f194a01d034413d80c493cf220` 执行，**与 `p1-hoi-p2-guidance-armb-s42-20260801`
  同树同工作目录**，因此本实验最承重的新对比（D2-AG+ArmB vs D2-X+ArmB）**是同树对比，零树风险**。
- `c40dc00..86ad8d8` 对 `code/`、`tools/`、`tests/` 的改动为**空**（仅 docs/registry/results），
  故 `86ad8d8` 与 `c40dc00` 在可执行内容上等价；选 `c40dc00` 只为与已封存格共用同一 Git 对象。
- D2-AG 无引导格的树可比性**已被硬证据确立**：worker 上 `9d77a6f` 执行的
  `p1-hoi-d2ag-native-eval-s42-20260801` 与本机 `5f7dde7` 执行的
  `p1-hoi-d2ag-distance-probe-s42-20260801` 的 `per_sequence_metrics.json` **sha256 逐位相同**
  （`eb701cf4e80a4a6c8198a0af5f914fab98c8bf26aabafee9971ce0827de2d835`），aggregate 的 9 个非
  provenance 键全等（5 个差异键为 checkpoint 路径、chois_export 开关、计时与 per-sequence 路径）。
  `9d77a6f..5f7dde7` 的 diff 只含 `docs/phase_summaries/PHASE_1B_D2AG.md` 与
  `experiments/registry.jsonl`，**零代码改动**，与该逐位复现互为佐证。
- 唯一残留树边界是 `5f7dde7 → c40dc00`（引导实现）。源码层面无引导路径受
  `if guidance is not None and step`（`diffusion.py:294`）与
  `if self.guidance_settings is not None` 守卫，为纯增量；但 P2 本身也**从未在 `c40dc00` 上跑过
  一次无引导对照**。因此本节预注册**两个必做的同树无引导对照**，各约 4 分钟：
  1. **D2-AG 同树无引导对照**（`c40dc00`，`guidance.enabled=false`，其余 override 与主运行逐字
     相同）。判据：`per_sequence_metrics.json` sha256 必须等于
     `eb701cf4e80a4a6c8198a0af5f914fab98c8bf26aabafee9971ce0827de2d835`。
  2. **D2-X 同树无引导对照**（同上，D2-X checkpoint）。判据：sha256 必须等于
     `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`。
  两者**必须在主运行的结果被解读之前执行并记录**。若任一不逐位相等，则该 checkpoint 的无引导
  格改用**同树对照值**，且这一替换与其数值差必须在结论中写明，**不得沿用封存值**。
- **D2-AE 第二臂有一个 D2-AG 没有的树风险，必须补对照。**
  `p1-hoi-d2ae-native-eval-s42-20260729` 执行于 `5a167347`，其后 D2-AF/D2-AG 的实现改动了
  `code/priors/diffusion.py`（+58 行到 `5f7dde7`）、`models.py`（161）、`sparse_relation.py`（383）
  ——全部在 D2-AE 的执行路径上，且**从未被重跑验证为惰性**。因此 D2-AE + Arm B 必须**同时**跑
  一个 `c40dc00` 上的 D2-AE 同树无引导对照，判据为 per-sequence sha256 等于
  `8533b66ea3c1fb0928b8a7581bb79c0cc14d594970314a3b7619659daddfb95c`；不相等时，D2-AE 的无引导格
  以同树对照为准，封存值只作历史记录。

判定规则（用户批准，逐字生效；全部在任何 GPU 运行之前固定）：

- **PRIMARY（缺口闭合）**：D2-AG + Arm B 是否在**两项穿透**上把对 released 带引导行
  （`p0-hoi-protocol-decomp-a0-old-s42-20260801`）的缺口闭合到 **±10% 以内**——
  `hand_pen_loss_omomo` 目标 `0.1624020`，`human_pen_loss_infbagel` 目标 `2.5892670`——
  **同时**保持接触平价：`contact_f1` 对 released `0.7272580` 的配对差 CI 跨零，或点估计不低于
  `0.7272580 − 0.02`。released 行无 `per_sequence_metrics.json`，故对 released 的**代价类**对比
  只有点估计、无 CI；此限制照 `PHASE_1B_P2_GUIDANCE.md:358-360` 原样继承，不得被写成 CI 结论。
- **PROTECTION**：(i) `contact_f1` 相对 **D2-X + Arm B 的 `0.7265270`** 不得显著更差
  （配对序列 bootstrap CI 下界 ≥ `−0.02`）；(ii) `end_obj_trans_err` 相对 **D2-AG 无引导的
  `3.6922000`** 不得显著退化（配对 CI 上界 ≤ `+0.25 cm`，与 P2 实测 Arm B 对 D2-X 的
  `+0.0992 [+0.0331, +0.1658]` 同量级）。任一被违反即记为代价性失败并如实登记。
- **加性/交互检验（本设计的核心新量）**：对每个指标计算
  `interaction = (D2AG_guided − D2AG_unguided) − (D2X_guided − D2X_unguided)`，
  并给出其配对序列级 bootstrap CI（438 序列全部配对，四格同一序列集合）。
  **零交互假设 = 引导对 D2-AG 的效应与它对 D2-X 的效应相同。** 该量的一个关键性质是
  **常数树偏移在其中精确抵消**，因此即便上述同树对照暴露出小的树效应，交互项仍然可解释；
  主效应则不然，这一非对称性在结论中必须写明。
- **报告但不设门**：`contact_precision`、`contact_recall`、`contact_percent`、`foot_sliding`、
  `mpjpe`、`xy_points_err`、`obj_trans_dist`、`obj_rot_dist`、`trans_dist`、
  `hand_pen_ratio`、`human_pen_ratio`、`position_outside_rate`、`nonfinite_values`、
  以及 GT-contact 帧手-物距离分布与 ≥8 cm 尾部占比。18 个 aggregate 指标**全部记录**。
- **门控项**：`nonfinite_values == 0` 且 `position_outside_rate == 0.0`（沿用 P2）。

事前预测（在任何 GPU 运行之前写入，无论对错都保留）。取 Arm B 在 D2-X 上的实测效应
（`contact_f1 +0.0891010`、`contact_recall +0.1133760`、`hand_pen −0.0159370`、
`human_pen −0.2430200`、`foot_sliding −0.0073560`、`end_obj +0.0998790`）并**假设完全加性**，
D2-AG + Arm B 应读到：

| 指标 | 加性预测 | released 带引导 | 预测相对缺口 |
| --- | ---: | ---: | ---: |
| `contact_f1` | 0.7391880 | 0.7272580 | **+1.6%（反超）** |
| `hand_pen_loss_omomo` | 0.1677350 | 0.1624020 | **+3.3%**（现为 +41.3%） |
| `human_pen_loss_infbagel` | 2.6948510 | 2.5892670 | **+4.1%**（现为 +40.0%） |
| `foot_sliding` | 0.3935620 | 0.3333630 | +18.1%（**不被修复**） |
| `end_obj_trans_err` | 3.7920790 | 3.0372440 | +24.9%（**不被修复**） |

即：**若加性成立，本实验一次性把两个仅存的最大真实缺口从 ~40% 压到个位数百分比，而 foot sliding
与 end-object 的缺口原样保留。** 这是一个强的、可被证伪的预测：若实测显著偏离，说明存在真实
交互——最可能的方向是引导在 D2-AG 上抬高 `contact_percent` 之后穿透随参与度回升，即
`EP:7912-7916` 的参与度机制在**反方向**上生效。**该情形必须被记成"D2-AG 的穿透优势部分由参与度
换来"，不得改写为其它叙事。** D2-AE + Arm B 的同法加性预测为 `contact_f1 0.7310450`、
`hand_pen 0.1634420`、`human_pen 2.6173110`、`end_obj 4.3989060`。

主要风险（按可能性排序）：

1. **参与度—穿透耦合。** 引导把 `contact_percent` 从 `0.4770600` 推向 `~0.57`，手更多地接近
   物体，穿透可能随之回升，使加性预测过于乐观。这是本实验最可能失败的方式。
2. **自条件关系源在被引导轨迹上的分布外行为。** 上文的惰性界定测于无引导轨迹；引导后的 `x_t`
   在其分布之外。若出现 `nonfinite_values > 0` 或 `position_outside_rate > 0`，须按门控项判失败
   并保留，**不得调小 `guidance_scale` 重跑**。
3. **`5f7dde7 → c40dc00` 的残余树效应。** 由上述两个同树无引导对照直接检验；交互项对常数树偏移
   免疫，主效应不免疫。
4. **D2-AE 第二臂的树风险**（见上）在数量上大于 D2-AG 臂。
5. **首次执行的代码组合。** 引导与稀疏关系元数据从未同时执行过
   （`diffusion.py:579-604` 同时展开 `**relation_arguments` 与 `**guidance_arguments`）。
   因此在官方 438 之前，每个 checkpoint 各跑一次 `hoi_sequence_limit=4` 的功能性 smoke，
   **只检验可运行性与有限性**，不得用于在配置之间做选择（沿用 P2 的 smoke 契约）。

被既有证据否定的备选：

- **(a) 用 Arm A 而非 Arm B。** Arm A 在 D2-X 上 foot sliding 比值 CI 上界 `1.1661`，而 D2-AG 的
  foot sliding 本就是它相对 D2-X 的唯一回退；两者叠加会把一个已知弱项推得更弱。Arm B 是
  `PHASE_1B_P2_GUIDANCE.md:226-232` 记录的"经得起代价审视"的配置。**两臂皆已在 D2-X 上报告，
  此处选 Arm B 不是新的挑选，而是沿用已声明的代价结论。**
- **(b) 为 D2-AG 调 `guidance_scale`/`last_steps`。** 未登记 sweep，为 `AGENTS.md` 与
  `docs/HOIPRIOR_ITERATION_WORKFLOW.md:20-22` 所禁止。
- **(c) 用 D2-AG checkpoint 做任何形式的初始化、resume 或蒸馏。** 为
  `PHASE_1B_D2AG.md:473-486` 与 `AGENTS.md:11-12` 明文禁止；本实验只把它作为推理输入。
- **(d) 重跑任一已封存格以"统一执行树"。** 违反"不重跑封存件"的规则，且成本远高于两个
  4 分钟同树对照。

治理边界：

- **不训练、不分配训练 run id、不产生 checkpoint、不做 checkpoint selection、不改
  `hoiprior_search_closed`、不改 D2-AG/D2-AE 的阴性分类、不解除 Phase 1B 搜索关闭。**
- 引导保持 default-off；本实验不授权任何生产配置变更。
- 允许改动的文件范围：`docs/EXPERIMENT_PLAN.md`、`experiments/registry.jsonl`、
  `docs/HOIPRIOR_EVIDENCE_INDEX.md`（随本次一并修复，见下）。**零源码改动**——Arm B 与
  D2-AE/D2-AG 加载路径均已在 `c40dc00` 实现且被封存运行验证过，故无实现提交、无 authority
  suite 重跑、无 performance benchmark（执行路径与已封存的 Arm B 运行完全相同，按
  `docs/HOIPRIOR_ITERATION_WORKFLOW.md:68-70` 复用其执行剖面：D2-X+ArmB 生成 `64.92 s`、
  端到端 `203.89 s`）。
- 沿用 P2 的 lean 契约：本测量不调用 `tools/experiment.py start`、不产生训练 manifest，
  溯源由执行 commit、归档 Hydra 配置与 overrides、固定官方测试集与逐字节相同的指标代码承担。
- 一并修复 `docs/HOIPRIOR_EVIDENCE_INDEX.md`：其 header 仍写"through D2-AF0, 2026-07-30"，
  且全文 **0 次**提及 D2-AG——而 D2-AG 是 contact F1 / precision / MPJPE 三项上最好的 D2 模型，
  且是本实验的直接依赖。修复内容为 header 重新定期与 D2-AG 行/条目回填，不改写任何已被哈希
  绑定的封存件。

#### 2026-08-02 Phase 1B P3 结果：预注册代价性失败，引导与关系场在接触上部分冗余

上一节预注册的实验已全部执行完毕。**按预注册的逐字判据，本次为代价性失败（cost failure）**，
分类 `relation-field-guidance-contact-redundancy-cost-negative-stop`；compact result 为
`experiments/results/p1_hoi_p3_relation_field_guidance_s42_20260802.json`。穿透缺口确实被闭合，
但那不是判定：PRIMARY 要求穿透闭合**与**接触平价同时成立，PROTECTION (i) 另有独立下界，
两者各失其一。下文按预注册顺序记录，不把穿透结果提为结论。

执行环境：全部 6 格同树同工作目录，worktree `/data/yujinlun/InfBaGel-p2`、commit
`c40dc00b2ad315f194a01d034413d80c493cf220`；官方 438 序列 × 3 窗口、`sample_type=diffusion`
500 步、`load_scene=false`、无 CFG、seed 42、`--config-name=config_eval_hoi_prior`。新执行 5 格
（3 个同树无引导对照 + D2-AG/D2-AE 两个 Arm B 格），封存的 D2-X + Arm B 格
（`p1-hoi-p2-guidance-armb-s42-20260801`）**只引用、不重跑**。**零源码改动**：`c40dc00..f836ca4`
对 `code/`、`tools/`、`tests/` 的 diff 为空。两个 `hoi_sequence_limit=4` 功能性 smoke
（`p3-smoke-d2ag-armb-s42-20260802`、`p3-smoke-d2ae-armb-s42-20260802`）先行通过，
`nonfinite_values` 与 `guidance_nonfinite_steps` 均为 0，**未用于任何配置选择**。

**三个预注册树对照全部逐位通过（这是一个独立的方法学结果）。**

| 同树无引导对照 | 预注册期望 per-sequence sha256 | 实测 | 判定 |
|---|---|---|:--|
| D2-AG | `eb701cf4e80a4a6c8198a0af5f914fab98c8bf26aabafee9971ce0827de2d835` | 逐位相同 | 通过 |
| D2-X | `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a` | 逐位相同 | 通过 |
| D2-AE | `8533b66ea3c1fb0928b8a7581bb79c0cc14d594970314a3b7619659daddfb95c` | 逐位相同 | 通过 |

因此 `5f7dde7 → c40dc00` 的引导实现**在无引导路径上被实测证明为惰性**，且是在**三种不同架构**
（D2-X 无关系场、D2-AE 稀疏关系场、D2-AG 自条件关系源）上分别证明的。此前只有源码层面的守卫
论证（`diffusion.py:294` 与 `self.guidance_settings is not None`），现在有了跨 438 序列 × 3 窗口 ×
500 步的逐字节证据。**这条边界是本仓库第一次被实测证伪风险而非仅被论证**，其直接后果是：三个封存
无引导格**无需替换、无需声明数值差**，且**主效应与交互项同样可解释**——预注册指出常数树偏移在
交互项中精确抵消而在主效应中不抵消，由于该偏移被实测为严格零，这一非对称性在本次不生效。
D2-AE 那一格的先验风险最大（其封存件执行于 `5a167347`，其后 `diffusion.py`/`models.py`/
`sparse_relation.py` 有 +58/161/383 行改动从未为 D2-AE 复验），现已逐位复验通过。

**6 格 × 18 个 aggregate 指标全表**（released 行为 `p0-hoi-protocol-decomp-a0-old-s42-20260801`，
其协议不可比，只作缺口目标）：

| 指标 | D2-X 无引导 | D2-X+ArmB | D2-AG 无引导 | **D2-AG+ArmB** | D2-AE 无引导 | D2-AE+ArmB | released |
|---|---:|---:|---:|---:|---:|---:|---:|
| `end_obj_trans_err` | 3.74021 | 3.84009 | 3.69220 | **3.74477** | 4.29903 | 4.29399 | 3.03724 |
| `xy_points_err` | 4.05052 | 4.13703 | 3.99541 | **4.09172** | 3.98943 | 4.01859 | 3.92310 |
| `feet_height` | 0.04981 | 0.03619 | 0.05495 | **0.04131** | 0.05303 | 0.04123 | 0.04112 |
| `foot_sliding` | 0.36301 | 0.35565 | 0.40092 | **0.39543** | 0.39896 | 0.39675 | 0.33336 |
| `contact_acc` | 0.73164 | 0.80269 | 0.73806 | **0.77684** | 0.73262 | 0.78809 | 0.80209 |
| `contact_precision` | 0.78806 | 0.81223 | 0.81120 | **0.82237** | 0.80363 | 0.80585 | 0.79081 |
| `contact_recall` | 0.59445 | 0.70783 | 0.59850 | **0.65271** | 0.59614 | 0.68278 | 0.72759 |
| `contact_f1` | 0.63743 | 0.72653 | 0.65009 | **0.69343** | 0.64194 | 0.71027 | 0.72726 |
| `contact_percent` | 0.47655 | 0.56956 | 0.47706 | **0.52243** | 0.47663 | 0.54985 | 0.59832 |
| `gt_contact_percent` | 0.66188 | 0.66188 | 0.66188 | **0.66188** | 0.66188 | 0.66188 | 0.66188 |
| `mpjpe` | 12.05084 | 12.07936 | 12.01292 | **12.04981** | 12.15578 | 12.12313 | 11.99759 |
| `trans_dist` | 8.17009 | 7.79007 | 8.29996 | **7.74430** | 8.16512 | 7.69128 | 8.20885 |
| `obj_trans_dist` | 15.99405 | 16.09432 | 15.41697 | **15.49605** | 15.95525 | 15.93526 | 15.72565 |
| `obj_rot_dist` | 1.03094 | 1.02696 | 1.00919 | **1.00451** | 1.00324 | 0.99115 | 1.02025 |
| `hand_pen_loss_omomo` | 0.24536 | 0.22942 | 0.18367 | **0.17214** | 0.17938 | 0.18113 | 0.16240 |
| `hand_pen_ratio` | 0.14387 | 0.14654 | 0.11287 | **0.11642** | 0.12159 | 0.13067 | 0.13286 |
| `human_pen_loss_infbagel` | 3.86908 | 3.62607 | 2.93787 | **2.74657** | 2.86033 | 2.87652 | 2.58927 |
| `human_pen_ratio` | 0.14619 | 0.15044 | 0.11615 | **0.12155** | 0.12703 | 0.13637 | 0.13764 |

**预注册判据逐条判定：**

- **PRIMARY 穿透闭合：通过。** `hand_pen_loss_omomo` `0.1721393` 对目标 `0.1624020` 为 **+6.0%**
  （P2 之后为 +41.3%）；`human_pen_loss_infbagel` `2.7465671` 对 `2.5892670` 为 **+6.1%**
  （原 +40.0%）。两项均落在预注册的 ±10% 带内。
- **PRIMARY 接触平价：不通过。** `contact_f1` `0.6934294`，低于预注册下限 `0.7272580 − 0.02 =`
  `0.7072577`。released 行无 per-sequence 输出，此处只有点估计、无 CI，此限制照 P2 原样继承。
- **PROTECTION (i)：失败。** `contact_f1` 对 D2-X + Arm B 的配对差为 `−0.0331`
  `[−0.0517, −0.0145]`，**CI 下界低于 `−0.02`**。
- **PROTECTION (ii)：通过。** `end_obj_trans_err` 对 D2-AG 无引导为 `+0.0515`
  `[−0.0091, +0.1178]`，上界远低于 `+0.25 cm`。
- **门控项：通过。** 两个臂、全部 438 序列与全部 aggregate 的 `nonfinite_values` 为 0，
  `position_outside_rate` 与 `object_outside_rate` 为 0.0，`guidance_nonfinite_steps` 为 0。
  **预注册风险 2（自条件关系源在被引导轨迹上的分布外行为）未兑现。**

**事前预测被证伪，且恰好证伪在预注册指定的位置。** 加性预测 `contact_f1` `0.7391880`，实测
`0.6934294`，差 **`−0.0458`**。交互项 `(AG_g − AG_u) − (X_g − X_u)`：

| 指标 | 交互项 [95% CI] | 跨零 |
|---|---:|:--|
| `contact_f1` | **−0.0458 [−0.0591, −0.0329]** | 否 |
| `contact_recall` | **−0.0592 [−0.0759, −0.0428]** | 否 |
| `contact_percent` | **−0.0476 [−0.0603, −0.0353]** | 否 |
| `hand_pen_loss_omomo` | +0.0044 [−0.0073, +0.0162] | 是 |
| `human_pen_loss_infbagel` | +0.0517 [−0.1294, +0.2356] | 是 |
| `foot_sliding` | +0.0019 [−0.0234, +0.0275] | 是 |
| `end_obj_trans_err` | −0.0477 [−0.1324, +0.0389] | 是 |

即：**交互只在接触上显著，在穿透、foot sliding 与 end-object 上一律跨零。** 预注册把加性/交互
检验列为本设计的核心新量并预言"显著偏离即存在真实交互"——这一点成立；但它同时把最可能的机制
指认为参与度—穿透耦合，**那个具体机制被否证**（见下）。被证伪的预测按 P2 的先例原样保留在记录中。

**单调剂量—反应，正是声明的第二臂被设计来暴露的量。** 引导对 `contact_f1` 的增益：

| checkpoint | `contact_f1` 增益 [CI] | `contact_percent` | 对 D2-X 的交互 [CI] |
|---|---:|---:|---:|
| D2-X | +0.0891 [+0.0759, +0.1022] | 0.47655 → 0.56956 | — |
| D2-AE | +0.0683 [+0.0550, +0.0819] | 0.47663 → 0.54985 | **−0.0208 [−0.0359, −0.0057]** |
| D2-AG | +0.0433 [+0.0347, +0.0518] | 0.47706 → 0.52243 | **−0.0458 [−0.0591, −0.0329]** |

机制记录：**稀疏关系场本身就削弱引导的接触效应，把它的源改为自条件则再削弱约一倍**
（D2-AG 对 D2-AE 的交互 `−0.0250 [−0.0384, −0.0120]`，同样不跨零）。这一点**追认了 P2 排除 D2-AG
的部分理由**——但不是以 P2 给出的安全性理由（非有限值与越界率全零），而是以**有效性**理由：
D2-AG 恰是引导最不起作用的 checkpoint。P3 推翻该排除**在安全性上是对的、在有效性上是错的**。
两份记录都不改写：P2 的理由仍如其所写，P3 的推翻仍以其所给的证据成立。

**关系场的穿透优势不是参与度换来的：预先承诺的假象叙事未出现。** 三个无引导格的参与度**基本相同**
——`contact_percent` `0.47655`（D2-X）/ `0.47706`（D2-AG）/ `0.47663`（D2-AE），配对差
`+0.0005 [−0.0182, +0.0192]` 与 `+0.0001 [−0.0196, +0.0195]` 均跨零，`contact_recall` 同样跨零
——而 `hand_pen_loss_omomo` 为 `0.24536` / `0.18367` / `0.17938`、`hand_pen_ratio` 为 `0.14387` /
`0.11287` / `0.12159`。预注册要求"若出现相反发现必须记成'穿透优势部分由参与度换来'"，
**该情形没有出现**，据实记录其未出现；证伪落在接触冗余上，不在参与度上。

**声明的第二臂，按声明报告，不提为头条。** D2-AE + Arm B 对 D2-X + Arm B：`contact_f1`
`−0.0163 [−0.0342, +0.0016]`（**跨零，接触在统计上完好**）、`hand_pen_loss_omomo`
`−0.0483 [−0.0827, −0.0161]`、`human_pen_loss_infbagel` `−0.7495 [−1.2884, −0.2482]`、
`foot_sliding` `+0.0411 [+0.0218, +0.0604]`、`end_obj_trans_err` `+0.4521 [+0.2309, +0.6774]`、
`mpjpe` `+0.0438 [−0.1868, +0.2732]`（跨零）。它对 released 的两项穿透缺口为 **+11.5%** 与
**+11.1%**，即**刚好落在 D2-AG 通过的 ±10% 之外**。**两臂互不支配**：D2-AG 把穿透压得更低但显著
损失接触；D2-AE 保住接触但在 end-object 与 foot sliding 上显著付费。不做任何选择，也未获授权做
选择——两个 checkpoint 的阴性分类原样保留。

**综合结论：引导与稀疏关系场在接触上部分冗余、在穿透上互补。** 两者都把手推向物体，故接触增益
不叠加（上表的单调剂量—反应）；而穿透主要由关系场改善、引导几乎不动它——D2-X `0.24536 → 0.22942`、
D2-AG `0.18367 → 0.17214`、D2-AE `0.17938 → 0.18113`，三条引导主效应的 CI 全部跨零。这与已记录的
HOIPrior 失效模式一致：接触失败是**≥8 cm 的大幅落空**而非近失，已经把手移近的机制会让引导无事可做。

测量口径与说明（不掩盖）：

1. `contact_percent`、`contact_acc`、`gt_contact_percent`、`feet_height`、`xy_points_err`
   **不在** `evaluation/per_sequence_metrics.json` 中，直接对它取配对差会得到 NaN。本次按 P2 先例
   **补算而非记为"仅有 aggregate"**：用冻结的 `code/eval_metrics.py` 几何在各运行自己的
   `motion_params` 与 chois 导出上重算逐序列接触统计。抽取门通过——六格的 `contact_f1`/`recall`/
   `precision`/`percent`/`gt_contact_percent`/`acc`/`foot_sliding` 与已发布 aggregate **完全相等**
   （二进制相等），仅 `feet_height` 因累加顺序差 ≤ `5.96e-09`。
2. `end_obj_trans_err` 的逐序列构造与已发布 aggregate 不是同一个量（前者用插值后末帧，后者用插值前
   窗口端点，相对差 0.54%–0.81%），沿用 P2 的记载；PROTECTION (ii) 的 CI 属逐序列构造。
3. 四个穿透项只对 181 个序列打分（冻结指标代码不为六个物体类计算穿透），六格的未打分集合完全相同。
   主口径用预注册的 438 序列 `tools/summarize_hoi_phase1b.py:112` 约定（`np.nanmean`）；P2 的
   181 子集口径作为次口径与每一条穿透对比并列记录。两者点估计相同，CI 半宽仅在小数第三位有别，
   **没有任何一条判定在两种口径间改变**。
4. released 行无 per-sequence 输出，一切对它的代价类对比只有点估计。
5. D2-AG 是在看到其穿透数字之后被选中的；预注册已披露并锁死 checkpoint 集合为 D2-AG 与 D2-AE，
   两臂均已报告，未增加第三个 checkpoint。

治理与边界（本次全部为否）：不训练、不分配 run id、不调用 `tools/experiment.py start`、不产生
checkpoint、不做 checkpoint selection、不重跑任何封存格、不改 D2-AG/D2-AE 的阴性分类、不解除
`hoiprior_search_closed`、不把引导改为生产默认开启、不进入 consistency / HSIPrior / Mixer、
不就地改写任何已被哈希绑定的封存件。下一会话入口不变：在 `phase/01c-hsi` 上做 dated 的
Phase 1C HSIPrior plan-only 预注册，从随机初始化训练。

#### 2026-08-04 Phase 1B P5 推理期接触 mask 剂量-响应与 GT 上界探针（用户批准）

动机：D2-AI 对 Arm B 引导的响应比 D2-X 弱约 5 倍——`contact_percent` 增量 `+0.09301` 对
`+0.01854`，`contact_recall` `+0.11338` 对 `+0.02339`，`contact_f1` `+0.08910` 对 `+0.02204`。
两者对 GT 参与度的余量几乎相同（`0.18533` 对 `0.17143`），所以这不是"没有空间"。

机制已定位到一行。`code/guidance_loss.py:31`：

```python
contact_labels = pred_contact_semantic > 0.95   # 模型自己预测的接触
```

**引导只在"模型自己已以 >0.95 置信度预测接触"的帧上把手掌拉向物体表面**（阈值 2 cm 死区，
`:36`）。它是**放大器而非发生器**：无法创造参与度，只能锐化模型已经承诺的参与度。据此推断
D2-X 存在大量"预测接触但手离得远"的帧供引导发力，而更长训练把这一不一致区间消除了——这与
D2-AI 的 `hand_pen` 从 `0.24536` 降到 `0.17481` 是同一件事。**该推断本实验直接检验。**

同时确认：`code/guidance_loss.py:42-47` 对物体做 `.detach()`（`pred_seq_com_pos.detach()`、
`pred_obj_rot_mat.detach()`），**引导只移动手与脚，从不移动物体**。故引导在结构上无法改善
`end_obj_trans_err`，实测两臂加引导后该项均**变差**（`+0.09988`、`+0.05898`）。本实验不以
`end_obj` 为目标。

设计：单一操纵因子为**引导接触 mask**，在 D2-AI 最终 checkpoint
（`a190e56c249161c0b52f0aebb097d0d5b95cb0c3810abb664000fc3c2fdda224`）上跑，其余一切与封存
Arm B 逐字相同（`arm=b`、`guidance_scale=1000.0`、`last_steps=10`、`clamp=1.0`、
`clamp_target=update`、438 序列 × 3 窗口、500 步、`load_scene=false`、seed 42）：

| 格 | `contact_mask_source` | `contact_mask_threshold` | 角色 |
|---|---|---:|---|
| A0 | predicted | 0.95 | 基线，**复用已封存的 D2-AI + Arm B，不重跑** |
| A1 | predicted | 0.90 | 剂量 |
| A2 | predicted | 0.75 | 剂量 |
| A3 | predicted | 0.50 | 剂量 |
| A4 | predicted | 0.25 | 剂量 |
| **U** | **ground_truth** | — | **上界探针,不可部署** |

**选择/确认分裂（在任何结果存在之前固定）。** 官方 438 条按
`sha256("42:" + 序列名)[0] & 1` 分为**选择半 A = 209 条**与**确认半 B = 229 条**
（两半 subject 与 object 分布均衡，各 13 类物体；名单存于 compact result）。

- **阈值在 A 半上选定**：取 `contact_f1` 最大者，且须同时满足下述保护门；
- **胜者的数值在 B 半上报告**，该数字**无选择偏差**，因为 B 半未参与选择；
- **两半 × 全部六格 × 18 指标的完整表格一并报告**，剂量-响应曲线本身即结果，不只报一个赢家。

这一分裂是必要的：若在全部 438 条上试五个阈值再挑最好，那个值将混入对这 438 条的过拟合而
无法与真实收益分离，且属未登记 sweep。分裂使额外成本为零——每格本就要跑一次全 438 评测，
分裂只发生在**聚合**阶段。

判定规则（逐字固定）：

- **PRIMARY**：胜者阈值在 **B 半**上对 A0 的 `contact_f1` 配对序列级 bootstrap
  （seed 42、10,000 replicates、`tools/summarize_hoi_phase1b.py:112` 约定）CI 排除零且为正。
- **保护门（任一违反即该阈值不得被选为胜者）**：
  (i) `contact_precision` 相对 A0 的下降，其 CI 下界不得低于 `−0.02`;
  (ii) `hand_pen_loss_omomo` 相对 A0 不得显著变差;
  (iii) `mpjpe` 相对 A0 不得显著变差;
  (iv) 每格 `nonfinite_values == 0`。
- **参与度强制读法**：`contact_percent` 必须与每一项接触或穿透结论并列报告。GT 为
  `0.66188`，A0 为 `0.50899`。降低阈值必然提高参与度，因此**任何接触改善都必须与穿透同时
  检视**，不得孤立叙述。
- **U 格（GT mask）的地位**：**诊断性上界探针，不可部署**（推理期无 GT），
  **不得进入任何主表、不得被选为胜者、不得作为部署配置**。它只回答一个问题：
  若参与度判断完全正确，引导最多能拿回多少接触缺口。
  - 若 **U 也拿不回接触缺口** → mask 不是瓶颈，几何才是,方向立即转向,不再在 mask 上投入;
  - 若 **U 拿回大半** → "改善参与度预测"成为一个有明确收益上界的目标。
- **事前预测（写死，无论对错都保留）**：降低阈值将提高 `contact_percent` 与 `contact_recall`、
  降低 `contact_precision`，且穿透变差；`end_obj_trans_err` 不受益（物体被 detach）。
  U 格的接触增益应显著大于任何 predicted 阈值。**若降低阈值对 `contact_recall` 无效果，
  则"mask 门限制了引导作用面"这一推断被证伪。**

治理边界：

- **零训练、零 checkpoint、不分配训练 run id、不改动任何既有分类、不替换任何封存行。**
- 默认路径必须逐位不变:新增的 `contact_mask_source: predicted` /
  `contact_mask_threshold: 0.95` 为默认值,启用前须由针对性测试证明与 HEAD 逐位一致,
  且封存的 D2-AI + Arm B 格**复用不重跑**。
- GT-mask 路径须为显式非默认值,不可被误启用,并在源码注释中标注其不可部署性。
- 允许改动:`code/priors/inference_guidance.py`、`code/config/sampler/hoi_prior.yaml`、
  `tests/test_hoi_guidance_mask.py`、`docs/EXPERIMENT_PLAN.md`、`experiments/registry.jsonl`、
  `experiments/results/`、`docs/HOIPRIOR_EVIDENCE_INDEX.md`、`docs/phase_summaries/`。
  **`code/eval_metrics.py` 与官方 438 协议零改动**;`code/guidance_loss.py` 优先不动
  （作者代码,且可能被源码哈希钉覆盖）。
- 因改动共享推理路径,GPU 发布前跑一次完整 authority 测试套件。

并行审计（零 GPU，与本实验同期）：查明作者 `end_obj_trans_err` ≈3.0 的来源。用户已复现作者
训练代码并蒸馏得 `3.555`，released 权重本地评测得 `3.037`，论文 Table 5 报 `3.06`——**本地无
任何训练管线能产出 3.0 附近的值**。审计范围:作者训练期物体目标的取帧是否与评测评分帧一致、
是否存在额外的终点监督、蒸馏阶段是否引入教师所无的终点项、以及 `cm_sample_loop` 在末步是否
直接把 `object_goal` 注入物体通道（`code/test_infbagel_hoi.py:407` 有一行被注释掉的、直接以
`object_goal_temp` 计算 `end_obj_trans_err` 的代码,其意图须一并说明）。结论须明确归为
"我们缺失的正当机制""采样期直接注入""本仓库代码无法解释""取帧不对称"之一,不得超出代码所示
妄加推测。

**关于 `end_obj` 的取向,一并写明:** 我们在整条物体轨迹 `obj_trans_dist` 上已达
`14.81981`,反超 released 的 `15.72565` 约 5.8%;差距仅在终点那一帧的对齐。`end_obj` 是
"复现一个被告知的数"的能力,不是运动质量,**追逐它有过拟合单帧的风险**,故本阶段不将其设为
优化目标。

每个阶段只允许上文给出的诊断/fallback。新增方向必须先在此处追加日期、证据和原因，并在
registry 登记，再实现代码。

#### 2026-08-04 Phase 1B P6 推理期手部子项重加权与 GT 参与度上界（用户批准）

动机：P5 的接触 mask 剂量-响应返回预注册空结果——阈值 `0.95 → 0.25` 使
`contact_recall` 仅动 `+0.0000240`。按 P5 预注册逐字规则，"mask 门限制了引导作用面"
这一推断**已被证伪**。但空结果的原因已在真实引导路径上定位，且它指向一个**不同的、可操纵的**
量。

机制（对账精确，72/72 检查，max abs err `0.0`，8 序列，monkeypatch 真实 evaluator）。作者手部项
是 `bs * (loss_contact + loss_consistency)`（`code/guidance_loss.py:69`），两个子项对 mask 宽度
**反向**响应：

| 子项 | thr 0.95 | thr 0.25 | Δ | 方向 |
|---|---:|---:|---:|---|
| `loss_contact` | `0.02117535` | `0.02125364` | `+0.000078` | 升 |
| `loss_consistency` | `1.39546636` | `1.39422995` | `−0.001236` | 降 |
| 和 | `1.41664170` | `1.41548359` | `−0.001158` | **降** |

在 mask 真正变化的 6/36 步上比值为 **15.79×**（闭式预测 `15.58×`）。原因是归一化的量纲不同：
`loss_contact` 是 `bs*T*2` 个槽位上的 L1 **均值**，新增一帧线性贡献 `(d−0.02)/(2·bs·T)`；
`loss_consistency` 掩码的是 **T×T 外积**，新增一帧点亮约 `2k` 个对，减去
`2k·s̄/(bs·T²)`，**对 k 是二次的**。比值 `4k·s̄/(T(d−0.02))` 随 k 增长，故
**mask 放得越宽，过度抵消越严重——没有任何阈值能修好它**。
`loss_consistency` 占手部项 **98.5%**，`loss_contact` 仅 **1.5%**（占总损失值 1.86%，
但占总梯度范数 29.2%）。

同期两项**撤回**，一并记录以保持记录自洽：(a) 我先前称手部项占引导损失 13.1%——错，封存审计给出
`guidance_feet_weighted_mean` `218.69` 对总 `5063.74`，手部项占 **95.68%**；(b) 我先前称 mask 在
阈值 0.25 下扩张 `1000 → 1368` 帧（+36.8%）——该探针**未测量引导真正消费的张量**。在引导发生的
最后 10 步，`x0_hat` 已近干净、接触 logit 已饱和成双峰（`43.67% > 0.95`，`43.75% > 0.25`，
中间几乎为空），阈值扫描只打开 **6/7200 个元素（0.083%）**。

实测梯度几何（作者权重下，即两子项均 ×1 位于 ×10 手部项内）：L2 范数 contact `10.89`、
consistency `23.58`、feet `19.47`、总 `37.34`；max-abs contact `1.330`、consistency `13.380`、
feet `3.524`。三者近正交（余弦 contact–consistency `0.085`、contact–feet `0.001`、
consistency–feet `−0.017`）；对总梯度的余弦 feet `0.713`、consistency `0.409`、contact `0.375`。
**clamp 在封存设置下不是约束**（`449/835200 = 0.0538%` 元素饱和）。

设计：单一操纵因子为**手部两子项的权重与一致性归一化**，其余一切与封存 Arm B 逐字相同
（D2-AI 最终 checkpoint `a190e56c...fdda224`、`checkpoint_weight_variant=online`、`arm=b`、
`guidance_scale=1000.0`、`last_steps=10`、`clamp=1.0`、`clamp_target=update`、
`contact_mask_source=predicted`、`contact_mask_threshold=0.95`、438 序列 × 3 窗口、
`sample_type=diffusion` 500 步、`load_scene=false`、无 CFG、seed 42、`config_eval_hoi_prior`）：

| 格 | `contact_weight` | `consistency_weight` | `consistency_normalization` | 角色 |
|---|---:|---:|---|---|
| B0 | 1 | 1 | author | 基线，**复用已封存的 D2-AI + Arm B，不重跑** |
| B1 | 1 | **0** | author | 直接移除反向抵消项 |
| B2 | **3** | 1 | author | 使接触梯度分量与非接触分量等权 |
| B3 | **10** | 1 | author | 接触主导；**权重轴天花板探针** |
| B4 | 1 | 1 | **masked_pairs** | 修复归一化伪影，保持作者权重 |
| **U** | 1 | 1 | author + **`contact_mask_source=ground_truth`** | **参与度上界探针，不可部署** |

`masked_pairs` 是**对作者算术的刻意偏离**（除以掩码开启的对数而非全部 T² 对）。凡用它得到的
数值**均须标注为非作者协议**，不得与论文 Table 5 直接并列。

**事前预测（写死，无论对错都保留），由实测近正交梯度范数推出，故每条可证伪：**

- B1 移除 `23.58` 的 consistency 分量，总梯度范数降至 `sqrt(10.89² + 19.47²) ≈ 22.3`，
  contact 占梯度范数由 `29.2%` 升至 `≈48.8%`;
- B2 把 contact 缩放到 `32.67`，总 `≈ sqrt(32.67² + 23.58² + 19.47²) ≈ 44.4`，contact 占 `≈73.5%`;
- B3 把 contact 缩放到 `108.9`、max-abs 到 `≈13.3`。元素在
  `|grad · 1000 · posterior_variance| ≥ 1.0` 时触顶，而 `posterior_variance[1..10]×1000` 跨度
  `0.058..0.423`，故 `|grad|` 超过约 `2.4..17` 即被截断：**B3 预期大量饱和**，那会把引导退化为
  符号下降，反而抹掉本实验所操纵的权重配比。**B3 因此是天花板探针，不是候选配置。**
- 效应方向：提高接触压力应改善 `contact_recall` / `contact_percent`，可能改善
  `hand_pen_loss_omomo`，代价是 `contact_precision`;
- **事前声明的代价方向（非事后发现）**：`foot_sliding` 与 `feet_height` **预期变差**——feet 项
  （×500）当前是最强单一操舵项（对总梯度余弦 `0.713`），提高接触项会与它争夺同一批 FK 关节的
  梯度。两项**无论结果如何均为强制报告项**。

判定规则（逐字固定）：

- **PRIMARY**：最佳非 U 格在 **B 半（确认半，229 条）**上对 B0 的 `contact_f1` 配对序列级
  bootstrap（seed 42、10,000 replicates、`tools/summarize_hoi_phase1b.py:112` nanmean 约定）
  CI 排除零且为正。
- **选择/确认分裂**（沿用 P5 规则，在任何结果存在之前固定）：
  `sha256("42:" + 序列名)[0] & 1`，`0 →` 选择半 A（209 条），`1 →` 确认半 B（229 条）。
  **格在 A 半上选定，数值在 B 半上报告**。名单存于 compact result。额外成本为零——每格本就要跑
  全 438 评测，分裂只发生在聚合阶段。
- **保护门（任一违反即该格不得被选为胜者）**：
  (i) `contact_precision` 相对 B0 的下降，其 CI 下界不得低于 `−0.02`;
  (ii) `hand_pen_loss_omomo` 相对 B0 不得显著变差;
  (iii) `mpjpe` 相对 B0 不得显著变差;
  (iv) 每格 `nonfinite_values == 0`。
- **参与度强制读法**：`contact_percent` 必须与每一项接触或穿透结论并列报告。GT `0.66188`，
  B0 `0.50899`。不得孤立叙述接触改善而不同时检视穿透代价。
- **新增强制门 `guidance_clamp_saturation_fraction`（每格必报，并记录其存在理由）**：
  某格饱和率若显著高于 B0 的 `0.0538%`，则该格的空结果**不得**被解读为"该权重无效"，
  **唯一可采纳的读法是"更新被 clamp 吞掉"**，且该格须在更高 `clamp` 或更低 `guidance_scale`
  下重跑，方可下任何结论。此门的存在正是为了防止把饱和误读为无效。
- **U 格（GT 掩码）的地位**：**诊断性上界探针，不可部署**（推理期无 GT），
  **不得进入任何主表、不得被选为胜者、不得作为部署配置**。
  - **U 格有效性闸（先决条件，不通过即 U 作废、不得报告）**：GT 标签按整条序列给出
    （`code/test_infbagel_hoi.py:625`），而引导发生在 `max_len=3` 的自回归窗口内
    （`:837`）、作用于窗口局部张量（`code/priors/diffusion.py:458`）。**必须证明逐窗对齐**：
    在 GT 掩码下，引导窗口覆盖帧上测得的参与度须等于 GT `contact_percent` `0.66188`。
    若不等，则 U **作废**——一个错位的上界探针仍会产出看似合理的数字，这是最坏的失败模式。
- **弃元/收敛判据**：若**全部 B 格为空** **且** **U 也未能关闭 `contact_percent` 缺口**，
  则推理期引导方向**整体关闭**，下一杠杆为**训练侧接触头**（模型自身参与度承诺不足属训练侧属性，
  任何推理期重加权在结构上都无法弥补一个不存在的标签）。

治理边界：

- **零训练、零 checkpoint、不分配训练 run id、不改动任何既有分类、不替换任何封存行。**
- 默认路径必须逐位不变：`contact_weight: 1.0` / `consistency_weight: 1.0` /
  `consistency_normalization: author` 为默认值，**默认时调用作者原函数本身而非其重写式**。
  已由 `tests/test_hoi_guidance_subterms.py` 证明：重写式与作者原式在 5 种形状上**严格相等**
  （max rel err `0.0`），默认路径梯度**逐位相等**。封存的 D2-AI + Arm B 格**复用不重跑**。
- GT-mask 与 `masked_pairs` 均须为显式非默认值，并在源码与 config 注释中标注其不可部署性/
  非作者协议属性。
- 允许改动：`code/priors/inference_guidance.py`、`code/config/sampler/hoi_prior.yaml`、
  `tests/test_hoi_guidance_subterms.py`、`docs/EXPERIMENT_PLAN.md`、`experiments/registry.jsonl`、
  `experiments/results/`、`docs/HOIPRIOR_EVIDENCE_INDEX.md`、`docs/phase_summaries/`。
  **`code/eval_metrics.py` 与官方 438 协议零改动**；`code/guidance_loss.py` **绝不改动**
  （作者代码——正因如此才在我方文件中逐字重写并以对账测试钉死）。
- 因改动共享推理路径，GPU 发布前跑一次完整 authority 测试套件（已完成：21 个测试文件全通过，
  含 `test_hoi_p2_guidance` 30 项与新增 22 项）。

成本：5 格需跑（B0 复用）+ U 格，每格约 3.5 分钟，4 卡并行，合计约 10–15 分钟墙钟。
无训练、不产出 checkpoint。


---


---

## 追记（2026-08-21）：cell U 的 GT 接触掩码在窗口 2、3 上退化，其封存结论作废

追加节，不改上文。**cell U 的定义（本文件 :620、:737）没有问题；它的执行有缺陷。**

`gt_contact_label_batch[seq]`（`code/test_infbagel_hoi.py:771`）存的是**一个 16 帧窗口**的接触
（`code/datasets/infbagel.py:626-629` 已按窗口切片），而 `_gt_contact_window`（`:371-393`）
按 stride 14 把它当**整条序列**索引。step 2 处 `start = 28 >= 16` 落进短序列分支，
返回窗口 0 的**最后一帧重复 16 次**。cell U 消耗完整 16 帧掩码，故其**窗口 2 只有 2/16 帧
正确、窗口 3 是 0/16**。

**量级**：封存上界建立在参与度 **0.7891457382039574**（16591/21024）之上，
正确值是 **0.6612442922374430**（13902/21024）——对该探针存在目的所要界定的那个量本身
的 **+19.35% 相对膨胀**。

**为什么判 VOID 而不是「减弱」。** 退化掩码是**过宽**的，而过宽的接触掩码会在真值无接触的帧上
施加接触引导——这正是损害 precision 的机制。cell U 的封存结论恰是**「变差」**
（`contact_f1 −0.0028307`，显著更差；参与度只关闭 −0.19%）。若结论曾是「掩码有帮助」，
过宽只会低估帮助，结论可从宽存活；但结论是「它变差了」，而过宽本身就是让它变差的机制，
**两者无法分离**。该探针在 3 个窗口里有 2 个**从未实现**它所声称的「完美参与度判断」。

**因此**：推理期接触掩码方向**未被本证据关闭**，把剩余 ~82.66% 参与度缺口归为
**训练侧几何性质**的推论也失去支撑。二者都需在修正掩码上重测。

**未受影响（已验证）**：P5/P6 的 predicted-mask 子项扫描不受影响——9 个兄弟臂全部用
`contact_mask_source=predicted`，且 `gt_contact_label_batch` 的两个读点都在
`_hoi_guidance_uses_ground_truth`（`:395`）闸内。`deployable: false` 仍然正确。

**发现路径**：P14 的闸门 G4（`08_ROLLOUT_DYNAMICS.md` 修订 2）。
P14 的教师强制通路改用独立的逐窗口累加器 `contact_all_gt` **绕过**该来源，
故 cell U 的路径逐位未变，等待自己的更正。

**治理缺口**：`normalization_audit.inference_guidance` 把 `contact_mask_source` 与
`contact_mask_threshold` 记为 `None`——审计不记录一个 run 用了哪种掩码。corrected 预注册
须补上。

更正方案（预注册、patch、GPU 估计、拟 run id）在 `.claude/scratch/cellu_fix/`，
**待用户审查后才重测，旧结果不覆盖**。


---

## 预注册（2026-08-22）：corrected cell U——在修正掩码上重测 GT 接触参与度上界

追加节，不改上文，**不覆盖任何封存结果**。用户批准 2026-08-22：
「同意按独立修正流程重跑 P6 cell-U：保留旧结果并标记 VOID，使用新 run id，
确认正常 predicted-mask 路径不变，并在结果中记录接触掩码来源。修正版完成前不启动 W3。」

run id：`p1-hoi-p6-cellu-corrected-mask-s42-20260822` · 子阶段 `1B-P6`（延续，不新开）
seed 42 · 类型：**推理期探针，不可部署，不是模型成绩**

### 复用的封存出处（逐一核对过）

| 项 | 值 |
|---|---|
| checkpoint | `p1-hoi-d2ai-full-budget-s42-20260803_windows299520000.pth` |
| checkpoint sha256 | `a190e56c249161c0b52f0aebb097d0d5b95cb0c3810abb664000fc3c2fdda224`（本机实测一致） |
| weight variant | `online` |
| 引导配置（两臂共用） | `arm=b`、`guidance_scale=1000.0`、`last_steps=10`、`clamp=1.0`、`clamp_target=update`、`contact_mask_threshold=0.95` |
| B0 封存 run dir | `p1-hoi-d2ai-guidance-armb-s42-20260804` |
| B0 封存 `per_sequence_metrics.json` sha256 | `ca999d2d0196684adb6603aad89ac08dfd44058936d8daf61b829154aab1b2cf` |
| B0 封存 `contact_f1` / `contact_percent` | `0.6757017455384731` / `0.5089874610422556` |
| 封存（退化）cell U 的 Δcontact_f1 | **−0.0028307，CI [−0.0066513, −0.0000785]**，显著更差 |

### 修正后的掩码是什么

`code/datasets/infbagel.py` 新增 `sequence_contact_label(idx)`，返回**整条序列**的接触轨道；
`code/test_infbagel_hoi.py:771` 的来源从 `data_dict['contact_label']`（**一个 16 帧窗口**）
换成该轨道。**`_gt_contact_window` 不动**——它的 stride-14 算术在整序列输入下已实测正确
（438 序列 × 2 次步进，raw 索引差恒为 42 = 14×3）。新来源只在
`_hoi_guidance_uses_ground_truth(cfg)` 为真时被填充，因此 predicted 路径连列表都不再构造。

**与 P14 已落地的 `contact_all_gt` 的关系**：方向相反，互不读取。`contact_all_gt`（`c4c934f`）
是教师强制**绕过**该来源、只取每个窗口自己的前 2 帧；本次**修复**该来源，供 GT-mask 引导
消费完整 16 帧。闸门 G4 验证互不影响。

### PRIMARY（先于结果固定）

**Δcontact_f1 = corrected cell U − B0**，438 序列配对 bootstrap，2000 次复制，seed 42，
共享重抽样索引矩阵，2.5/97.5 百分位线性插值。

| 判定 | 条件 | 含义 |
|---|---|---|
| **`cellu-helps`** | Δcontact_f1 **> 0** 且 CI 不含 0 | 完美参与度判断**有帮助**；推理期掩码方向**未关闭**，旧 abort 被推翻 |
| **`cellu-hurts-confirmed`** | Δcontact_f1 **< 0** 且 CI 不含 0 | 即使掩码正确也变差；旧 abort **在有效证据上重新成立** |
| **`cellu-null`** | CI **含** 0 | 无可检出效应；方向**既未开也未关**，且「剩余 ~82.66% 属训练侧」的归因**保持无支撑** |

**为什么 PRIMARY 用 `contact_f1` 而不用 `contact_percent`。** 后者是四个**只有聚合值、
没有逐序列值**的指标之一（另三个是 `contact_acc`、`gt_contact_percent`、`feet_height`），
**因此没有 bootstrap CI**，不能承担带显著性的裁决。旧记录的「−0.19% of the gap closed」
正是一个无 CI 的点估计，这一点当时没有写明。

### SECONDARY（点估计，无 CI，无裁决权）

参与度缺口关闭比例 = (contact_percent(corrected) − 0.5089874610422556) / (0.66188 − 0.5089874610422556)，
分母 **0.15289**。封存退化值为 −0.19%。报出即可，不作判定。

### 闸门

| 闸门 | 要求 | 失败后果 |
|---|---|---|
| **G1** | corrected cell U 的掩码参与度分数复现 **0.6612442922374430**（13902/21024），容差 1e-12 | 掩码没修好，停 |
| **G2**（scoping，**用户点名要求**） | **B0 重跑逐位复现 `ca999d2d…b2cf`**。补丁改的来源只在 GT 闸内被读，故 predicted 路径必须逐位不变 | 补丁泄漏到 predicted 路径 → corrected 读数作废，且 P5/P6 主扫描也要重新审 |
| **G3** | corrected 掩码在**每个** rollout step 上的相异帧数 > 1，且 step 2 的前两帧不等于窗口 0 末帧 | 退化未消除，停 |
| **G4** | 掩码来源与 P14 的 `contact_all_gt` 互不读取（源码扫描 + 测试） | 两个修复互相污染，停 |
| **G5** | 438 序列、3 窗口、`is_timing_subset=false`，与封存 cell U 同 checkpoint 同 seed | 不可比，停 |
| **G6**（**硬性**，用户点名要求「在结果中记录接触掩码来源」） | 两个 run 的 `normalization_audit.inference_guidance` 都必须报出 `guidance_contact_mask_source`（corrected = `ground_truth`，B0 = `predicted`）与 `guidance_contact_mask_threshold` | 无法从工件判断用了哪种掩码——**停**，不再按「可放行」处理 |

G6 的治理缺口需精确表述：`GuidanceAudit.as_dict()` **根本不发出**这两个键（不是发出 `None`），
所以 2026-08-22 之前的任何封存 run 都无法自证掩码来源。本次实现补上了这两个键，
并对「未绑定 settings 时报 `None` 而不是默认值 `predicted`」加了测试——
「字段缺失」与「字段是 predicted」不能对读者长成一个样。

### 事前预测

**预测 `cellu-null`，或一个小幅 `cellu-helps`。** 退化掩码过宽、会在真值无接触的帧上施加接触
引导，那是损害 precision 的机制；去掉过宽应当至少让「显著更差」消失。是否转正取决于模型的
接触头能否被掩码撬动，而 P5 的剂量-响应扫描是平的，说明杠杆有限。**因此不预测大幅改善。**
若结果是 `cellu-hurts-confirmed`，则旧 abort 在有效证据上恢复，是一个比原来更强的结论。

### 本节不建立什么

1. **不是模型成绩。** GT 接触在推理时不存在；corrected cell U 与旧 cell U 同为不可部署探针，
   其 18 项指标不得进入 `baseline.md`、证据索引头表或任何模型对比表。
2. **不重开 P5/P6 的 predicted-mask 扫描**——已验证不受该缺陷影响。
3. **不改 `_gt_contact_window`**、不改 438 分母、不改 `code/priors/core/`、不改 `recipe/d2ai.yaml`。
4. **不覆盖任何封存结果**；旧 cell-U 值逐字节保留，本次结果作为独立行封存。
5. 即便判 `cellu-helps`，也**不**意味着存在可部署方案——它只界定「若参与度判断完美」的上界。
6. 对 P14 的身高收缩结论零信息，两者正交。

### 依赖与成本

**W3 几何项训练在本次重测完成前不启动**（用户裁决 2026-08-21，2026-08-22 重申）。
2 个 rollout，GPU 0、1（4–7 被 HSIPrior 训练占用）；封存 cell-U run 实测
`end_to_end_seconds` 188.76，故并行约 **3.2 min**，预算上限 **≤12 min、最多 2 GPU**。
CPU 侧配对 bootstrap 秒级。无训练、不产出 checkpoint。
