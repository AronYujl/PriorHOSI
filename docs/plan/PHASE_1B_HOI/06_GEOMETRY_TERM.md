# Phase 1B-06：训练侧手-物几何项（D2-AH 权重恢复、P8/P9 剂量扫描、P10 公式修复）

本文件于 2026-08-10 从 `docs/EXPERIMENT_PLAN.md` 第 7678-7933、9108-9519 行原样切出（逐字节复制，未改写、未重排、未修正任何笔误）。
导航：[总览](../OVERVIEW.md) · [Phase 1B 索引](README.md)

#### 2026-08-02 Phase 1B D2-AH 度量几何目标权重恢复（用户批准）

动机（本次发现）：`code/config/config_train_hoi_prior.yaml:36-37` 的默认值就是作者的
`fk_weight: 50.0` / `object_surface_weight: 50.0`，`config_train_hoi_prior_d2t.yaml:41-42`
以相同数值显式重述、未做任何改动。`config_train_hoi_prior_d2u.yaml:42-43` 在 6,144,000
窗口预算下首次把二者改写为 `0.3569973401779424` / `0.4772322188400037`（取自 D2-I 梯度均衡
聚合），此后 D2-V/X/Y/Z/AB/AC/AD/AE/AF/AG 十个 61,440,000 窗口的运行
**逐字节继承同一对数值**（`config_train_hoi_prior_d2v.yaml:43-44`、`d2x:45-46`、
`d2y:47-48`、`d2z:51-52`、`d2ab:55-56`、`d2ac:57-58`、`d2ad:59-60`、`d2ae:60-61`、
`d2af:61-62`、`d2ag:62-63`）。其中 D2-X 是封存对照，其余**九次失败的 model-side 实验**
共享同一个**从未被当作变量审视的常数**，而设定它的那次决策是在最终正式预算的 1/10 上做出的，
此后再未复核。本实验把该常数恢复为作者取值，这是**恢复一个已知可用的配方**，不是新增机制。

机制（以真实 `data/train/norm.npy` 定量，1 个归一化单位 = 3.32917 / 1.09975 / 3.48764 m）：

1. **手部。** `code/priors/losses.py:156` 的 `hand_fk` 是 4 个手关节 × 3 坐标共 12 个元素的
   **米制 MSE**，作者 `code/models/infbagel.py:840` 同为 MSE，量纲与归约完全对齐。单个手关节偏
   离 e 米时，作者目标付 `50 × e²/12 = 4.1667e²`，我们付 `0.3569973401779424 × e²/12 =
   0.029750e²`；再加上两侧共有、权重为 1 的归一化 `joint_position` 通道
   （`losses.py:132`，84 元素 MSE，x 轴系数 `(1/3.32917)²/84 = 0.0010741`），得作者 `≈4.167e²`、
   我们 `≈0.031e²`，**等效欠权 135.2×**（纯权重比为 `50/0.3569973401779424 = 140.06×`，
   两侧共有的权重-1 通道把它拉低到 135）。一次 **8 cm 的手部偏离**因此对作者值 `0.0267`、
   对我们只值 `0.00020`。
2. **物体。** 作者 `code/models/infbagel.py:861` 的 `loss_object` 用 `smooth_l1`（|x|<1 时为
   `0.5x²`），我们 `code/priors/losses.py:196` 用 MSE。物体纯平移偏离 e 米时作者付
   `50 × 0.5 × e²/3 = 8.3333e²`，我们付 `0.4772322188400037 × e²/3 = 0.15908e²`，
   **等效欠权 52.4×**（纯权重比 `104.77×`，被 smooth_l1 的 1/2 系数减半）。该项直接映射到
   `end_obj_trans_err` 与 `xy_points_err`。

与实测失败模式对齐：pooled gross-miss（有 GT 接触的帧上手离物体 ≥8 cm）在 D2-X+Arm B 下为
`0.1216`，released 带引导为 `0.1052`（`docs/phase_summaries/PHASE_1B_P2_GUIDANCE.md:176-177`）；
gross-miss 帧上手部世界误差**中位数 24.12 cm**；其中 **67.7%** 的帧手部误差 ≥ 2× pelvis 误差。
即失败是**够不着**（reach error），不是**站错位置**（body placement error）——而“够不着”正是
被欠权 135× 的那一项所度量的量。2026-08-01 已记录同向证据：协议对齐后 HOIPrior 的原始生成几何
比 released 差 `0.8334 cm [0.4906, 1.1824]`，且引导只搬动主体分布、搬不动远尾。

被操纵因子（单一概念，两个取值）：`fk_weight: 0.3569973401779424 → 50.0`、
`object_surface_weight: 0.4772322188400037 → 50.0`。除这两个标量外，一切与
`code/config/config_train_hoi_prior_d2x.yaml` 逐字节相同：同一 `code/train_hoi_prior.py` 训练
入口、同一数据契约与 seed-42 split（`experiments/splits/omomo_hoi_train_validation_seed42.json`，
4,088 训练序列 / 568,486 窗口）、`max_processed_windows: 61440000`、
`effective_batch_size: 2048`、`batch_size: 512`、`num_gpus: 4`、`gradient_accumulation_steps: 1`、
Adam `learning_rate: 1e-4`、`scheduler_name: none`、`gradient_clipping: false`、`ema_decays: []`、
`amp: false`、`velocity_weight: 0.1`、`goal_weight: 1.0`、`fk_foot_temporal_routing: true`、
`primary_weight_variant: online`、`checkpoint_interval_windows: 3072000`。**零新增参数**，
`[B,16,232]` 输出契约不动，scene-free 与 composable 契约不动，500 步无引导原生采样协议不动。

预注册的前置诊断与中止条件（在正式训练之前执行，先于任何 run id 分配）：对既有的
**非 reportable** checkpoint——run 目录
`results/p1b-author-diffusion-8x3090-full-r1-s42-20260717/`，文件
`checkpoints/p1b-author-diffusion-8x3090-full-r1-s42-20260717_epoch100.pth`
（101 × 597,868 = **60,384,668** processed windows，为 D2-X 61,440,000 的 98.3%，预算基本
对齐）——按官方 438 序列 × 3 窗口原生协议做一次纯推理评测。该 checkpoint 只作评测输入，
**绝不用于任何初始化**。目的只有一个：检验 D2-T 在 6,144,000 窗口下的崩溃
（MPJPE `34.7367`、end-object `38.5563`、contact F1 `0.2764`，而 foot sliding `0.1761`
是全程最好的一档）是否是**欠训练假象**。

**中止规则（在诊断执行之前固定，不得事后修改）**：若 epoch100 行没有在
`xy_points_err`（D2-X `4.050519689917564`）**与** `end_obj_trans_err`（D2-X
`3.7402085959911346`）**两项上同时优于 D2-X**，则正式训练不启动，本方向以 negative preflight
结案并如实登记。

执行树与树效应控制（同样在诊断执行之前固定）：诊断在 worktree
`/data/yujinlun/InfBaGel-head-baseline`（`5f7dde7`）执行，理由是该树已于 2026-08-01 用
`p1-hoi-d2x-distance-probe-s42-20260801` **逐位复现封存的 D2-X native aggregate 18/18**，
因此拿它产出的 epoch100 行去对比封存的 D2-X 行不是跨树比较。已记录的 epoch500 行则产自
`1e982bc`（`phase/01b-author-repro`），与本行不同树，**不得与本行直接相减**。两树之间隔着
2026-07-13/14 的自回归 rollout 重写，而协议分解只验证过该重写对三个接触质量指标为 null，
`xy_points_err` / `end_obj_trans_err` / penetration 均未测。因此预先固定一条附加规则：
**若 epoch100 未过上述中止判据，则先在同一 head-baseline 树补跑 epoch500 作为树效应对照，
再决定判负**——不得把"作者模型路径的树差异"误记为"预算不足"。若 epoch100 通过判据，则该对照
不必执行。

时序声明：上述中止规则与树效应规则在诊断启动之前即已固定并向用户声明；本预注册提交在诊断
执行期间落盘，但**早于任何诊断结果存在**，提交内容未因任何中间状态改变。

该诊断的解释力被事先限定：它至少混淆 **7 个因子**——(1) 两个损失权重；(2) 重构子项的 L1 与
smooth_l1/MSE 取法（作者 `code/models/infbagel.py:801,804,806` 用 L1、`:861` 的 object 用
smooth_l1；我们 `code/priors/losses.py:132-136,196` 用 MSE/smooth_l1）；(3) 预算
（60.38M vs 61.44M，且作者原始为 501 epoch）；(4) `use_random_frame_bps: true`；
(5) 物体占据条件 `add_object_voxel: true`；(6) 训练期 CFG dropout（`free_p: 0.1`，`w: 1`）；
(7) 多 5.28% 的训练序列（4,304 vs 4,088）。因此它**只能证伪“崩溃是预算不足造成的”这一解释**，
**不能**把效应归因到权重本身。归因只由正式的 D2-AH 单因子训练承担。

评测契约：官方 438 序列 × 3 窗口，只加载固定的 final-online checkpoint（不做 cadence 选择、
不做 best-of-N），配对序列级 bootstrap（seed 42，10000 replicates，沿用
`tools/summarize_hoi_phase1b.py:112` 的索引矩阵约定），对照为已封存的
`p1-hoi-d2x-native-eval-r1-s42-20260723`，其 aggregate 直接复用、不重跑。**两个采样臂，均事先
声明、无论结果如何都报告**：(i) **500 步无引导 ancestral DDPM**，与每一个 D2 行协议一致，是
判定规则所依据的臂；(ii) **P2 Arm B 推理期接触引导**（late-steps-only、按 `posterior_variance[t]`
缩放并对更新量裁剪、`guidance_scale: 1000.0`、确定性顶点子集），与当前最佳配置协议一致。
记录全部 18 个 aggregate 指标，另加复原的 GT-contact 帧手-物距离分布与 ≥8 cm 尾部占比。

判定规则（用户批准，逐字生效）：

- **PRIMARY**：D2-AH − D2-X 的配对改善在 `end_obj_trans_err` **与** `hand_pen_loss_omomo`
  **两项上** CI 均不跨零。选这两项，因为它们是相对 released 带引导行**仅存的两个最大真实缺口**：
  Arm B 的 `end_obj_trans_err` 为 `3.8401` vs `3.0372`（**+26.4%**，配对点估计 `+0.8028 cm`），
  `hand_pen_loss_omomo` 为 `0.22942` vs `0.16240`（**+41.3%**，配对点估计 `+0.06702`）。
- **PROTECTION**：contact F1 不得显著退化（配对 CI 下界不得低于 `−0.02`）**且** MPJPE 对 D2-X
  的比值 ≤ `1.10`。
- **报告但不设门**：foot sliding、≥8 cm 尾部占比、GT-contact 帧距离、`xy_points_err`、contact
  precision/recall/percent、`obj_trans_dist`、`obj_rot_dist`、`trans_dist`、两项 penetration
  ratio、`human_pen_loss`。
- **明确记录：本次 foot sliding 按用户决定不作为门禁。** 理由是 D2-T 的证据预测 50× 的 FK-foot
  项会**改善**它（D2-T foot sliding `0.1761`，是全程最好的一档，而 D2-X 为 `0.36301`）。把一个
  被预测会改善的量设为保护门没有信息量；它仍然全量报告，若反而恶化，须在结论中写明。

主要风险（在任何 GPU 运行之前写明）：D2-T 在 6,144,000 窗口下用的正是这一对权重，得到 MPJPE
`34.7367`，而同预算的 D2-U（balanced 权重）为 `17.0285`。本实验押注**10× 预算会改变这个结论**，
依据有二：其一，同一对 balanced 权重下 D2-V 的 10× 预算把 end-object 从 D2-U 的 `10.0201`
降到 `3.6807`，说明 6.144M 处的读数对预算高度敏感、不足以判定权重；其二，作者自身的预算是
501 × 597,868 ≈ **299.5M** 窗口，为 D2-X 的 **4.88×**，故 61.44M 仍在作者配方的欠训练区间内，
D2-T 的 6.144M 更是其 1/49。**若 D2-AH 复现 D2-T 式崩溃，那是一个真实的阴性结果，必须保留，
不得改用某个中间权重重跑**——那将构成未登记的 sweep，为 `AGENTS.md` 与
`docs/HOIPRIOR_ITERATION_WORKFLOW.md:20-22` 所禁止。

被既有证据否定的备选：

- **(a) 再加一个专用 HOI 模块。** D2-AC（+349,697 参数）未过 locality permutation，
  `end_obj_trans_err` 保护比值 `1.503869023480123`；D2-AE（+413,953 参数）**五个内部因果门全部
  通过且 CI well-separated**，原生迁移仍为零。加参数这条路已被两次独立证伪。
- **(b) 用 released checkpoint 初始化或蒸馏。** 为 `AGENTS.md:11-12` 明文禁止。
- **(c) 改 trunk 的宽度/深度/头数。** 按 D2-V 的证据，容量从来不是绑定约束。
- **(d) 引入训练期条件 dropout / CFG。** 会改变每一个已封存 D2 结果所依赖的无引导采样协议，
  使全部横向对比作废。
- **(e) 增加显式的手-物相对几何损失。** 这是一个真正未试过、且有前景的方向；但它是**新增损失
  项**而非**恢复已知可用配方**，因此排在 D2-AH 之后——必须先有一个权重公平的基线，才谈得上
  在其上加项。

待并入的勘误（由本次准备阶段新算的配对 CI 直接导致，只登记事实，不就地改写任何已被哈希绑定的
封存件）：

1. `docs/phase_summaries/PHASE_1B_P2_GUIDANCE.md` § `What guidance does not fix` 把 Arm B 对
   `a0-old` 的五个差值记为纯点估计。现补 CI：`obj_trans_dist +0.36867` **不是**显著代价
   （CI `[−0.29002, +1.02849]` 跨零）；而两项穿透**是**显著代价——`hand_pen_loss +0.06702`
   `[+0.02348, +0.11408]`、`human_pen_loss +1.03680` `[+0.35443, +1.77463]`。原文只给点估计，
   方向性表述需按此修正。
2. `PHASE_1B_P2_GUIDANCE.md:243-244`（同一陈述亦见 `:48`）的“95.3 cm 对 3.7402 cm”是**指标与
   协议双重错配**的比较：`95.3` 是 D2-O 内部协议的 `object_goal_error_cm`，`3.7402` 是官方原生
   协议的 `end_obj_trans_err`；在那个内部协议上，**每一个训练充分的 D2 模型都读 92.9–95.3**
   （例：D2-AG `92.94030836224556`）。该数字不得再被援引。但“D2-Q0 的门槛 checkpoint 过弱”这一
   结论本身仍然成立，其依据换为该 checkpoint 只有 **64 次 optimizer update** 这一事实。

治理边界：

- **不继承 D2-AG 的 performance waiver。** `EP:7331-7333` 已明文声明该 waiver 一次性、run-id
  绑定、不得被 D2-AH 或任何后续实验继承或援引。D2-AH 只改两个标量乘子，不改变 per-step 计算、
  通信、数据加载、张量形状与显存，执行路径与 D2-X 完全相同，故按
  `docs/HOIPRIOR_ITERATION_WORKFLOW.md:68-70` **复用 D2-X 的封存执行剖面**
  （`3243.0357134915853` windows/s，全预算 ETA `5.263 h`）并在实现记录中写明 benchmark 不适用的
  理由；不申请、也不需要任何 waiver。
- 允许改动的文件范围：`code/config/config_train_hoi_prior_d2ah.yaml`（新增）、
  `code/train_hoi_prior.py`（仅新增与既有 D2-* 同构的 `d2ah` mode/subphase/权重契约分支）、
  `tests/test_hoi_d2ah.py`（新增）、`docs/EXPERIMENT_PLAN.md`、`experiments/registry.jsonl`。
  不得改动 `code/priors/losses.py`、`code/models/*`、`code/priors/diffusion.py`、
  `code/eval_metrics.py`、`code/test_infbagel_hoi.py`，以及任何已被哈希绑定的封存件。
  因 `code/train_hoi_prior.py` 属训练代码，实现提交在向 worker 发布之前须跑一次完整 authority
  suite。
- 本次为用户于 2026-08-02 新授予的一次授权，覆盖 D2-AG 预注册中
  `remaining_formal_experiment_budget: 1` 与 `last_authorized_hoiprior_direction: true` 的表述；
  仍不触及 `hoiprior_search_closed`，不改动 native D2 协议，不允许第二次 formal 训练、resume
  旧方向、checkpoint selection、D2-AH1、更长预算、consistency、HSIPrior 或 Mixer。

#### 2026-08-02 Phase 1B D2-AH 前置诊断判负与方向中止（预注册中止规则生效）

上一节预注册的前置诊断已执行并**未通过其固定中止规则**，预注册的树效应对照随后**确认了**该失败。
按预注册逐字执行：**正式 D2-AH 训练未启动**，未分配 run id，未调用 `tools/experiment.py start`，
未运行任何 GPU 训练负载，未产生任何 checkpoint，`code/config/config_train_hoi_prior_d2ah.yaml`
从未创建。分类 `metric-geometry-weight-restoration-preflight-negative-stop`；compact result 为
`experiments/results/p1_hoi_d2ah_negative_preflight_s42_20260802.json`。

执行环境：worktree `/data/yujinlun/InfBaGel-head-baseline`，commit
`5f7dde73903b78d70e6423d525de819e7f4ebfe3`；official 438 序列 × 3 窗口、`sample_type=diffusion`
500 步、`load_scene=true`、`add_object_voxel=true`、`w=1`、纯推理、无引导
（`code/test_infbagel_hoi.py:384-389` 的 diffusion 分支走 `p_sample_loop`，既不接
`guidance_fn` 也不接 `guidance_weight`，配置里的 `guidance_weight: 1` 在该路径上无效）。
两个 checkpoint 只作评测输入、绝不用于初始化：`..._epoch100.pth` sha256
`db0836f6c822f57b79e059208787aef98fbfd614667875825485aa81ae9806c1`（101 × 597,868 =
60,384,668 窗口，D2-X 的 98.3%，wall 32.2 min），`..._epoch500.pth` sha256
`44a723d20a4bbf13de8c2db78c3c375472dba20be0530993c9e00ab780747aac`（501 × 597,868 =
299,531,868 窗口，D2-X 的 4.875×，wall 32.1 min）。该树用
`p1-hoi-d2x-distance-probe-s42-20260801` **逐位复现封存 D2-X 的全部 18 个 aggregate**，本次重新
核验通过；封存 D2-X 行按预注册**只引用、不重算**。

**中止规则判定（两项均须优于 D2-X，实测两项均劣）：**

| gate 指标 | epoch100 | 封存 D2-X | 差值 | 相对 | 通过 |
|---|---:|---:|---:|---:|:--|
| `xy_points_err` | 5.7623 | 4.050519689917564 | +1.7118 | +42.3% | 否 |
| `end_obj_trans_err` | 5.4176 | 3.7402085959911346 | +1.6774 | +44.8% | 否 |

**epoch100 全 18 指标对封存 D2-X 的差值**（精确值见 compact result；此处只列判读要点）：准确度与
接触侧**全线更差**——`contact_f1` 0.4710（−0.1664）、`contact_recall` 0.3904（−0.2040）、
`contact_precision` 0.7484（−0.0397）、`contact_acc` 0.5902、`mpjpe` 14.9141（+2.8632）、
`trans_dist` 10.6235、`obj_trans_dist` 18.7289、`obj_rot_dist` 1.1247；只有六个**与接触参与度绑定**
的指标看似更好——`foot_sliding` 0.3219、`feet_height` 0.0468、`hand_pen_loss_omomo` 0.1109、
`hand_pen_ratio` 0.07809、`human_pen_loss_infbagel` 1.7777、`human_pen_ratio` 0.08327，见结论四。

**树效应对照（预注册规则：epoch100 未过判据则先补跑 epoch500 再判负）。** 本树 epoch500 对已记录
的 `1e982bc` 树 epoch500 行（源自未跟踪的 `results/p1b-author-diffusion-e500-eval-r1-20260721/hoi.log:584-591`，
历史日志只有四位小数，下表精度随之）：

| 指标 | epoch500@head `5f7dde7` | 已记录 epoch500@`1e982bc` | 差值 | 相对 |
|---|---:|---:|---:|---:|
| `xy_points_err` | 3.2307 | 3.2134 | +0.0173 | +0.54% |
| `end_obj_trans_err` | 3.6866 | 3.3224 | **+0.3642** | **+11.0%** |
| `foot_sliding` | 0.3398 | 0.3409 | −0.0011 | −0.32% |
| `contact_precision` | 0.7799 | 0.7811 | −0.0012 | −0.16% |
| `contact_recall` | 0.6437 | 0.6335 | +0.0102 | +1.61% |
| `contact_f1` | 0.6660 | 0.6615 | +0.0045 | +0.69% |
| `mpjpe` | 12.1974 | 12.0874 | +0.1100 | +0.91% |
| `hand_pen_loss_omomo` | 0.1937 | 0.1931 | +0.0006 | +0.29% |

**五条必须记录的结论：**

1. **中止规则失败。** epoch100 在两个 gate 指标上同时劣于 D2-X（`xy_points_err` +42.3%、
   `end_obj_trans_err` +44.8%），正式训练未启动。
2. **树效应真实存在、指标特异，且救不了这个判负。** 8 项中 7 项在 1.62% 相对以内复现
   （最大者为 `contact_recall` +1.61%），唯独 `end_obj_trans_err` 差 `+0.3642`（+11.0%）。因该指标
   正是两个 gate 之一，这条对照是**承重的**，不是形式主义。但它仍救不了 epoch100：`xy_points_err`
   的树效应只有 `+0.0173`（0.54%），而 epoch100 在该 gate 上差 `1.7118 cm`，约为树效应的 **99 倍**；
   即使把 `+0.3642` 的整段树修正**全额**记到 epoch100 头上（这已经很慷慨，因为该修正是在另一个
   预算上测得的），`end_obj_trans_err` 仍为 `5.0534`，对 `3.7402`。因此这次失败**不得**被记成
   树差异假象。
3. **对真正问题的定量回答。** 同一棵树、同一配方、同一 seed、预算是唯一变量：在 60,384,668 窗口
   （D2-X 的 98.3%）上，作者的度量几何配方在**每一个准确度与接触指标上都远劣于 D2-X**；在
   299,531,868 窗口（**4.875×**）上，它在 `xy_points_err`（−0.8198 cm）、`contact_f1`（+0.0286）、
   `contact_recall`（+0.0493）与 `hand_pen_loss_omomo`（−0.0517）上**胜过 D2-X**，其
   `contact_percent` 0.5311 也比 D2-X 的 0.47655 更接近 GT 0.66188。后三项在事后配对序列 bootstrap
   下 CI 不跨零（`xy_points_err` 无 per-sequence 记录，不可检验）。所以**作者的目标权重确实更好，
   但只在约五倍于我们正式预算处才兑现**。被证伪的是**该补救在 61,440,000 窗口下的可负担性**，
   **不是**"我们的 `fk_weight`/`object_surface_weight` 把度量手部与物体误差分别欠权约 135×/52×"
   这一量纲诊断。
4. **epoch100 的穿透与 foot sliding 优势是参与度假象。** epoch100 `contact_percent` 0.3192，对
   GT 0.66188、对 D2-X 0.47655（仅为 D2-X 的 0.670、GT 的 0.482）；其 `hand_pen_ratio` 0.07809 与
   `human_pen_ratio` 0.08327 相应落到 D2-X 的 0.543 与 0.570。**手够不到物体的模型无法穿透物体。**
   对照 epoch500：`contact_percent` 0.5311（**高于** D2-X）**同时** `hand_pen_loss_omomo` 0.1937
   （低于 D2-X 0.24536），那里的穿透优势才是真的。
5. **新发现：`end_obj_trans_err` 的缺口不归因于 diffusion 训练配方。** 在该指标上 D2-X
   （`3.7402`）与作者自己的 from-scratch diffusion 配方在 4.875× 预算处（epoch500@head `3.6866`）
   **本质持平**（差 1.43%；配对序列 bootstrap 均值 `−0.0679`，CI `[−0.3798, +0.2521]` 跨零），
   而 released checkpoint 读 `3.0372`。因此**此前被列为"仅存的两个最大真实缺口"之一、并因此被本次
   预注册选为 PRIMARY gate**（Arm B 对 released 带引导行 `+0.8028 cm`、+26.4%）的这个缺口，
   **根本不是训练配方造成的**——把训练配方推到作者的极端权重、再给它 4.875× 预算，也只把它移动
   1.4%。后续对该指标的努力应**从 diffusion 训练侧改动上撤出**。另注：自研 CM 复现在该指标上读
   `3.5553`（`docs/phase_summaries/PHASE_1B_D2AA.md:142`），故一致性蒸馏只解释残差
   `0.6493` 中的 `0.1313`，其余 `0.5181` 仍**无解释**。

治理与边界（本次全部为否）：不实现 `config_train_hoi_prior_d2ah.yaml`，不启动 D2-AH，不以任何
中间权重重跑（那将构成未登记 sweep，为 `AGENTS.md` 与 `docs/HOIPRIOR_ITERATION_WORKFLOW.md:20-22`
所禁止），不继承 D2-AG 的一次性 performance waiver，不新增 D2-AH1、不加长预算、不启动第二次
formal 训练、不做 checkpoint selection、不进入 consistency / HSIPrior / Mixer，不就地改写任何已被
哈希绑定的封存件。上一节所列的待并入勘误仍然有效，本次未执行。任何新方向须另做 dated plan
amendment 与 registry hypothesis，并取得用户明确批准。

## 2026-08-05 Phase 1B P8 训练侧手-物相对几何损失（2×2 预算×损失，用户批准）

动机：推理期引导已触顶且方向封闭。P6 胜者 B2（`contact_weight=3`）在确认半给出
`contact_f1` `+0.02896` [`+0.0210`,`+0.0378`]，`contact_percent` `0.50899 → 0.53550`
（GT `0.66188`），关闭参与度缺口的 17.34%；再提权重至 ×10 只多关闭 3.64 点而 clamp
饱和率升至基线 5.35 倍——权重轴已尽。

**Cell U 给出决定性的负判决。** 把模型自己的参与度判断替换成完美 GT 标注，接触指标
**显著变差**：确认半 `contact_f1` `−0.00283` [`−0.00665`,`−0.00008`]、`contact_recall`
`−0.00242`、`contact_precision` `−0.00637`，三项均显著；`contact_percent` `0.50870`
对基线 `0.50899`（缺口 `−0.19%`）。机制：GT 标注比模型自身 `>0.95` 的承诺**更宽**
（引导窗口参与度实测 `0.789`），于是引导被要求在手仍离物体很远的帧上拖拽手掌，而这些位移
在最后 10 步、`clamp=1.0` 之内推不动，反而扰动了原本正确的帧。**参与度判断不是瓶颈——
它已经够好，把它做到完美反而有害。** 剩余 82.66% 的缺口属于训练侧几何。

**一处必须记录的撤回。** 本文档先前多处断言"推理期引导从不移动物体"。**该断言错误。**
作者损失中只有一致性项 detach 了物体（`code/guidance_loss.py:42-47`）；接触铰链项
（`:38`）消费的 `obj_verts` **未 detach**。实测合成批次：物体平移梯度 L2 `2.677682`、
旋转 L2 `5.252323`。物体一直在被引导——只是被拉向**手**而非任务目标，这正解释了为何更强的
接触引导使 `end_obj_trans_err` 退化（`3.83750` 无引导 → `3.91086` B2）而整条轨迹
`obj_trans_dist` 反而改善（`14.81981 → 14.80752`）。

设计：单一操纵因子为**训练损失中新增的 GT 接触帧掩码手-物相对几何项**。

```
losses["hand_object_contact_geometry"] =
    mean over GT-contact frames of  mean_j ( min_v || palm_j - surface_v ||^2 )
```

- 手掌取 FK 关节 `(22, 23)`，与作者引导损失一致（`code/guidance_loss.py:15-16`）；
- 掩码取 GT 接触标注的**前两道** `[:2]`（左右手），阈值 `>0.5`；
- 只作用于 `[history_frames:]` 的活动帧。

**与现有 `hand_fk`（`code/priors/losses.py:156`）不是重复，参照系不同：** `hand_fk` 在手到达
**GT 手位**时最小——**无论物体被预测到哪里**；新项在手与**被预测的物体**互相一致时才最小。
若物体预测偏了，满足 `hand_fk` 反而把手放到一个**不在预测物体表面上**的点，这正是"已承诺接触
但手离物体远"的成因。量级上说得通：物体轨迹误差 `14.81` cm 大于接触失败幅度 `8` cm。

**GPU 前已完成的零成本核验（防止重演 P6 首轮的静默 no-op）：**

| 核验项 | 结果 |
|---|---|
| 手语义通道 | `[:2]`，全数据集均值 `0.581`/`0.590`；`[2:]` 为 `0.008`/`0.012` |
| 通道二值性 | 每道 `unique=2`，故 `>0.5` 与 `>0.95` 选出**完全相同**的帧，阈值不是自由度 |
| 参与度（全 482 文件） | `0.628`；按序列位置 `0.359`(0-10%) → `0.824`(50-75%) → `0.078`(90-100%) |
| 权重 50 下量级 | `8.47`，与 `object_surface` 的 `9.02` 相当，小于 `hand_fk` 的 `26.64` |
| 梯度→手掌 / 物体表面 | L2 `0.0957` / `0.0953` |
| 梯度→非手掌关节 / 历史帧 | **精确为 0** |
| 单元测试 | `tests/test_p8_hand_object_geometry.py` 12/12 通过 |
| 负对照 | 将通道注入为 `[-2:]` → 3 项失败；恢复 → 12/12 通过 |

若当初按 `[-2:]` 实现，参与度仅 `1.9%`，整个实验将退化为 no-op 并被误判为"方向无效"。

臂位（2×2，两臂重用，**不重跑**）：

| 臂 | 运行 id | 预算（windows） | `hand_object_contact_weight` | 状态 |
|---|---|---:|---:|---|
| L0 | `p1-hoi-d2aj-gonogo-treecontrol-d2x-s42-20260804` | 61,440,000 | 0.0 | 重用 D2-X |
| **L1** | `p1-hoi-p8-hand-object-geom-l1-s42-20260805` | 61,440,000 | 50.0 | 新跑 ~15h |
| H0 | `p1-hoi-d2ai-full-budget-s42-20260803` | 299,520,000 | 0.0 | 重用 D2-AI |
| **H1** | `p1-hoi-p8-hand-object-geom-h1-s42-20260805` | 299,520,000 | 50.0 | 新跑 ~22h |

**为什么必须是 2×2 而非固定预算。** 已知预算是唯一有效杠杆（D2-X → D2-AI，4.875×：参与度缺口
关闭 `+7.5%`，`hand_pen` `0.24536 → 0.17481`，`human_pen` `3.86908 → 2.76049`，十个模型侧
干预全部为空），且几何目标被 under-price 约 135×。在固定预算下新增几何项，等于在一个本就吃不饱
的目标上再分一块——它可能因预算不足而测不出，却被记成"该方向无效"。2×2 正是为分离这两种失败。

归因表（在任何结果存在之前固定）：

| L1 | H1 | 结论 |
|---|---|---|
| 显著改善 | 显著且更大 | 新项有效，预算放大其效应 |
| 显著改善 | 显著但不更大 | 新项有效，预算效应饱和或为统计噪声 |
| 空 | 显著改善 | **新项需足够预算才显现**（under-pricing 假说得证） |
| 空 | 空 | 新项无效，或公式/归纳偏置有误 |

判定规则（逐字固定）：

- **PRIMARY**：各臂在 **确认半 B（229 条）** 上对其**同预算基线**（L1 对 L0、H1 对 H0）的
  `contact_f1` 配对序列级 bootstrap（seed 42、10,000 replicates、nanmean 约定）CI 排除零且为正。
- **选择/确认分裂**：沿用 `sha256("42:" + 序列名)[0] & 1`，`0 →` A 半 209 条，`1 →` B 半 229 条。
- **保护门（任一违反即该臂不得被宣布为胜者）**：
  (i) `contact_percent` 相对同预算基线的偏移绝对值不得超过 `0.05`（超出则标记"参与度漂移"，需人工审查）;
  (ii) `mpjpe` 相对同预算基线不得显著变差（CI 下界不得高于 `+0.5` cm）;
  (iii) 每臂 `nonfinite_values == 0`。
- **参与度强制读法**：`contact_percent` 必须与每一项接触或穿透结论并列报告。
  L0 `0.47655`，H0 `0.49045`，GT `0.66188`。
- **评测配置强制项**：所有新臂一律以**显式命令行覆盖**应用 P6 胜者与 P7 胜者
  （`contact_weight=3.0`、`consistency_weight=1.0`、`consistency_normalization=author`、
  `object_goal_weight=1.0`）。**不得**将其写入 `config/sampler/hoi_prior.yaml` 默认值——
  那会破坏已封存结果与当前代码的逐位可比性。
- **提前中止**：L1 若在确认半为空且 CI 上界 `<+0.005`，**H1 仍必须跑完**——"低预算无效"不可
  外推为"高预算无效"，那正是 2×2 要区分的。

事前预测（写死，无论对错均保留）：

- **P1**：L1 确认半 `contact_f1` `+0.010` 至 `+0.025` 且显著。证伪条件：CI 含 0。
- **P2**：H1 确认半 `contact_f1` 均值比 L1 高 `+0.010` 以上。证伪条件：`≤` L1 `+0.005`。
- **P3**：胜者臂 `hand_pen_loss_omomo` 显著降低（CI 上界 `<−0.005`）。证伪条件：所有显著改善
  `contact_f1` 的臂在该项上 CI 含 0。
- **P4**：L1 与 H1 的 `mpjpe` 相对各自基线 CI 上界均 `<+0.3` cm。证伪条件：任一臂 `≥+0.3` cm。

治理边界：

- 允许改动：`code/priors/losses.py`、`code/train_hoi_prior.py`、`code/config/*.yaml`、
  `tests/test_p8_hand_object_geometry.py`、`docs/EXPERIMENT_PLAN.md`、`experiments/registry.jsonl`、
  `experiments/results/`、`docs/HOIPRIOR_EVIDENCE_INDEX.md`、`docs/phase_summaries/`。
- **不可触碰**：`code/guidance_loss.py`（作者代码）、`code/eval_metrics.py`、官方 438 协议入口。
- **任何已发布 checkpoint 不得用于初始化。** 种子 **42** 唯一合法。有效批次取
  `{512, 1024, 2048, 3072}`，**1536 禁用**。
- `hand_object_contact_weight` 默认 `0.0`，故所有既有封存配置的目标函数逐字不变。

成本：L1 ~15h + H1 ~22h，两臂并行，墙钟约 **22h**，合计约 **148 GPU·h**。无新增评测协议。

---

## 2026-08-06 Phase 1B P9 训练侧手-物几何项权重精调（权重 10 与 15，用户批准）

动机：P8 的 2×2 显示手-物相对几何训练项**方向正确但权重 50 过猛**。公平对比（四臂均用
`contact_weight=3` + `object_goal_weight=1` 引导评估）：

| 臂 | 预算 | 几何权重 | `contact_percent` | 距 GT 缺口 | `mpjpe` | `trans_dist` |
|---|---:|---:|---:|---:|---:|---:|
| L0 | 61.44M | 0 | 0.60959 | +0.052 | 12.057 | 7.779 |
| L1 | 61.44M | 50 | 0.64721 | +0.015 | 15.797 | 15.093 |
| H0 | 299.52M | 0 | 0.53519 | +0.127 | 11.745 | 7.535 |
| H1 | 299.52M | 50 | 0.66273 | **−0.00085** | 14.245 | 11.473 |

关键事实：
- H1 的 `contact_percent` 0.66273 **反超 GT 0.66188**——参与度缺口完全关闭，Phase 1B 首次。
- 但 `mpjpe` 恶化 +2.50 cm、`trans_dist` +3.94 cm，**远超保护门 ±0.5 cm**，H1 不得被宣布为胜者。
- 预算维度因果链成立：H1−L1 显示预算缓解了几何项的损伤（mpjpe −1.55 cm），H0−L0 显示预算本身改善运动。
- **结论**：几何项有效且方向正确，weight=50 剂量过猛。需要更小权重。

权重外推（299M 预算，线性：每单位权重 contact +0.00255、mpjpe +0.050）：

| 权重 | contact_percent（线性） | mpjpe（线性） | mpjpe 恶化 |
|---|---:|---:|---:|
| 10 | 0.561 | 12.245 | +0.50 |
| 15 | 0.573 | 12.495 | +0.75 |

但接触在 weight=50 已饱和（0.66273 > GT），边际接触增益随权重递减，故**低权重时每单位权重的
接触收益比线性更大**；而损伤在高权重时可能超线性。预计 weight=10 的实际 mpjpe 恶化 < +0.50
（保护门内）且接触 0.58-0.62。

设计：两臂，固定预算 299.52M，唯一操纵因子为 `hand_object_contact_weight`：

| 臂 | 运行 id | 几何权重 | 预算 |
|---|---:|---:|---:|
| **W10** | `p1-hoi-p8-hand-object-geom-w10-s42-20260806` | 10 | 299.52M |
| **W15** | `p1-hoi-p8-hand-object-geom-w15-s42-20260806` | 15 | 299.52M |

基线：H0（`p1-hoi-d2ai-full-budget-s42-20260803`，weight=0，复用不重跑）与 H1
（`None_windows299520000.pth`，weight=50，P8 已跑）。评估一律用
`contact_weight=3` + `object_goal_weight=1` 引导，与 P8 完全一致。

**本次吸取 P8 教训：`run_id` 必须显式设置**——P8 的 H1 因配置缺 `run_id` 把输出写到了
`results/experiments/None/`，导致误判死锁、误杀、再 resume 的连环事故。

判定规则：

- **PRIMARY**：`contact_f1` 相对 H0 在确认半 B（229 条）的配对序列级 bootstrap
  （seed 42、10,000 replicates、nanmean）CI 排除零且为正。
- **保护门**（任一违反即不得宣布胜者）：
  (i) `mpjpe` 相对 H0 的 CI 下界不得高于 +0.5 cm;
  (ii) `trans_dist` 相对 H0 不得显著变差;
  (iii) `contact_precision` 相对 H0 的下降 CI 下界不得低于 −0.02;
  (iv) 每臂 `nonfinite_values == 0`。
- **参与度强制读法**：`contact_percent` 必须与每一项接触或穿透结论并列。GT 0.66188，
  H0 0.53519，H1 0.66273。**若某臂接触缺口关闭不足、且 mpjpe 恶化显著，该臂即判为"保护门
  冲突"，说明该权重不在甜点区。**
- **权重选择的科学价值**：10 与 15 两点的连线刻画"接触饱和曲线 + 损伤曲线"的交叉区——
  若 10→15 接触仍快速上升而 mpjpe 增速放缓，甜点在 15-20；若 mpjpe 恶化骤增，则 10 已近极限。
- **中止判据**：若 W10 已满足保护门且 contact_f1 显著改善，W15 仍跑完（提供曲线形状）；两者
  取保护门内 contact_f1 最大者。

事前预测（写死）：

- **P1**：W10 满足保护门（mpjpe 恶化 < +0.5）且 contact_f1 相对 H0 显著改善。
- **P2**：W15 的 contact_f1 略高于 W10（接触仍上升），但 mpjpe 恶化接近或超过 +0.5。
- **P3**：若 P2 成立，则 10-15 之间是甜点区下界，后续可在 12-15 精调。

治理边界：

- 允许改动：`code/priors/losses.py`（已实现）、`code/config/*.yaml`、`docs/EXPERIMENT_PLAN.md`、
  `experiments/registry.jsonl`、`experiments/results/`、`docs/HOIPRIOR_EVIDENCE_INDEX.md`、
  `docs/phase_summaries/`。
- **不可触碰**：`code/guidance_loss.py`、`code/eval_metrics.py`、官方 438 协议。
- 种子 42 唯一合法；有效批次 `{512,1024,2048,3072}`，1536 禁用。
- 配置必须含显式 `run_id`（P8 教训）。

成本：两臂各约 22h，并行 4 GPU×2，墙钟约 22h，合计约 176 GPU·h。

---

## 2026-08-07 Phase 1B P9-b 权重下探：weight=5 与 weight=8（用户批准）

动机：P9（weight 10/15）显示接触在 weight≥10 已饱和。剂量-响应（299M 预算，全部引导评估）：

| 权重 | `contact_percent` | 距 GT | `mpjpe` | `trans_dist` | `hand_pen` |
|---|---:|---:|---:|---:|---:|
| 0 (H0) | 0.53519 | +0.127 | 11.745 | 7.535 | 0.165 |
| 10 (W10) | 0.65690 | +0.005 | 12.890 | 9.412 | 0.348 |
| 15 (W15) | 0.64398 | +0.018 | 13.167 | 10.424 | 0.571 |
| 50 (H1) | 0.66273 | −0.0009 | 14.245 | 11.473 | 0.423 |

关键事实：
- **W10 是当前帕累托最优**：接触缺口关闭 96%（0.005/0.127），`mpjpe` 代价 +1.15 cm（W50 的 46%）。
- **W15 反常**（hand_pen 0.571 反而高于 W50 0.423、接触低于 W10），剂量曲线在 10-15 不稳定。
- 三臂均违反保护门 `mpjpe ≤ +0.5`——这是"用运动质量换接触"的显式权衡，非方向无效。
- 问题：**是否存在更低的权重，接触仍达标（>0.64）而 mpjpe 代价更小（<+0.8）？**

设计：两臂，固定预算 299.52M，唯一操纵因子为 `hand_object_contact_weight`：

| 臂 | 运行 id | 几何权重 | 预算 |
|---|---:|---:|---:|
| **W5** | `p1-hoi-p8-hand-object-geom-w5-s42-20260807` | 5 | 299.52M |
| **W8** | `p1-hoi-p8-hand-object-geom-w8-s42-20260807` | 8 | 299.52M |

基线：H0（weight=0）与 W10（weight=10）复用。评估一律 `contact_weight=3` +
`object_goal_weight=1` 引导，与 P8/P9 完全一致。

科学目标（本批要回答的）：
- **W5 是否仍达标**：若 contact_percent > 0.64 且 mpjpe < +0.8，则 weight=5 是真正的"免费"甜点。
- **接触达标的最低权重**：5 vs 8 vs 10 三点确定接触饱和曲线的下界。

判定规则：
- **PRIMARY**：`contact_percent` 距 GT 缺口（0.66188 − 值），以及 `contact_f1` 相对 H0 在确认半 B
  的配对序列级 bootstrap（seed 42、10,000 reps、nanmean）CI 排除零且为正。
- **"免费甜点"判据**（本批特有的定义，非原保护门）：`contact_percent > 0.64`（缺口 < 0.022）
  **且** `mpjpe` 相对 H0 的增量 < +0.8 cm。满足则为理想配置；不满足则该权重不足以达标。
- **保护门**（沿用）：`contact_precision` 下降 CI 下界 ≥ −0.02；`nonfinite_values == 0`。
- 参与度强制读法、治理边界与 P9 相同。`run_id` 必须显式设置（P8 教训延续）。

事前预测：
- **P1**：W5 的 `contact_percent` 约 0.60-0.64（可能略低于达标线 0.64），mpjpe 约 12.3-12.6（+0.5~0.9）。
- **P2**：W8 的 `contact_percent` 约 0.64-0.65（接近达标），mpjpe 约 12.5-12.9。
- **P3**：若 P1/P2 均达标，则甜点在 5-8，后续可精调 6/7。

成本：两臂各约 22h，并行 4 GPU×2，墙钟约 22h，合计约 176 GPU·h。

---

## 2026-08-07 Phase 1B P9-c 权重再下探：weight=1 与 weight=3（用户批准）

动机：P9-b（weight 5/8）显示 **W5 是甜点**——`contact_percent` 0.6516（距 GT +0.010）、
`mpjpe` 12.529（仅 +0.784）。权重 5→50 接触几乎饱和（0.652→0.663）而 mpjpe 从 +0.78 飙到
+2.50，高权重纯浪费。现在再下探到 1 和 3，验证**接触达标的最低权重**——若 W3 或 W1 仍能
`contact_percent > 0.64`，则可用更小的 mpjpe 代价获得接触改善。

剂量-响应（299M，全部引导评估）：

| 权重 | `contact_percent` | 距 GT | `mpjpe` | `mpjpe` Δ |
|---|---:|---:|---:|---:|
| 0 | 0.53519 | +0.127 | 11.745 | — |
| 5 | 0.65161 | +0.010 | 12.529 | +0.784 |
| 8 | 0.65324 | +0.009 | 12.624 | +0.880 |
| 10 | 0.65690 | +0.005 | 12.890 | +1.145 |
| 50 | 0.66273 | −0.001 | 14.245 | +2.500 |

设计：两臂，固定预算 299.52M，唯一操纵因子 `hand_object_contact_weight`：

| 臂 | 运行 id | 几何权重 | 预算 |
|---|---:|---:|---:|
| **W1** | `p1-hoi-p8-hand-object-geom-w1-s42-20260807` | 1 | 299.52M |
| **W3** | `p1-hoi-p8-hand-object-geom-w3-s42-20260807` | 3 | 299.52M |

基线：H0（weight=0）、W5（weight=5）复用。评估一律 `contact_weight=3` + `object_goal_weight=1`。

科学目标：
- **接触跌破达标线的位置**：0→5 之间接触从 0.535 升到 0.652，最陡的部分在哪？W1/W3 定位。
- **接近"零代价"的接触改善**：若 W1 的 mpjpe Δ 接近 0 且接触仍有提升，则是近乎免费的配置。

甜点判据沿用：`contact_percent > 0.64` **且** `mpjpe` Δ < +0.8 cm。

判定规则、参与度强制读法、治理边界与 P9-b 相同。`run_id` 必须显式设置。

事前预测：
- **P1**：W3 `contact_percent` 约 0.63-0.65，mpjpe Δ 约 +0.3~0.6。
- **P2**：W1 `contact_percent` 约 0.60-0.64，mpjpe Δ 约 +0.1~0.4。
- **P3**：若 W3 达标，则甜点上移至 3；若 W1 也达标，则 1 是新的最低达标权重。

成本：两臂各约 22h，并行 4 GPU×2，墙钟约 22h，合计约 176 GPU·h。

---

## 2026-08-09 Phase 1B P10 手-物几何项修复：接触铰链与物体 detach（2×2 全预算，用户批准）

动机：P8–P9c 的八点扫描把 `hand_object_contact_weight` 从 50 降到 3，已经把**剂量**调到帕累托前沿；
但**公式本身从未被审视过**。W3 相对无几何基线 H0 留下两处退化，它们不是"接触换运动"的必然代价，
而是该项两个可定位实现缺陷的直接后果：

| 指标 | H0（无几何） | W3（当前封存） | 变化 |
|---|---:|---:|---|
| `hand_pen_loss_omomo` | 0.16451 | 0.25874 | **+57%** |
| `human_pen_loss_infbagel` | 2.59925 | 4.07788 | **+57%** |
| `end_obj_trans_err` | 3.89162 | 4.68200 | **+0.790 cm** |

**缺陷一：零距离目标，而非铰链。** `masked_hand_object_distance_loss`
（`code/priors/losses.py:40-91`）用 `nearest.square()`（`:87`）把掌关节到**被预测物体表面**的距离
逼向 **0**。而 P8 的掌关节索引正是照抄作者推理引导的（`code/guidance_loss.py:15-16`），
**作者那一侧用的是 2 cm 铰链**——`maximum(dists - 0.02, 0)`（`code/guidance_loss.py:36-39`），
距离小于 2 cm 时梯度**精确为零**。掌关节位于手部网格**内部**，而 `hand_pen_loss_omomo` 计的是手部
**顶点**穿透：把关节-表面距离压到 0，等价于把关节周围一圈顶点推进物体内部。P8 抄了索引，
没抄铰链，把作者刻意留下的 2 cm 死区变成了穿透驱动力。

**缺陷二：物体未 detach。** 该项消费的 `predicted_surface`（`losses.py:75`，来自 `:265-267`）
**带梯度**，因此损失可以通过把**物体**拖向手来降低自己，而不是把手移向物体。这条通路的量级 P6
已在引导侧实测：物体平移梯度 L2 `2.678`、旋转 L2 `5.252`。作者的一致性项**明确 detach 了物体**
（`code/guidance_loss.py:42-47`）——只有铰链项没有。这正预测 `end_obj_trans_err` 的退化。

两个机制**分别且精确地**预测了 W3 的两处退化：缺铰链 → 两个穿透项；缺 detach → 物体终点误差。
本轮**不改剂量**（权重固定 3.0），只修公式。

**一处必须记录的更正。** 封存紧凑结果的 `next_action`、
`docs/phase_summaries/PHASE_1B_P8_P9_GEOMETRY_WEIGHT_SWEEP.md:80`、
`docs/HOIPRIOR_EVIDENCE_INDEX.md:341` 三处均把 W3 的 `end_obj_trans_err` 记作 **"3.99"**。
**该数字是 w=50（H1）的值 3.9898，不是 W3 的。** W3 在
`experiments/results/p1_hoi_p8_p9_geometry_weight_sweep_sealed_s42_20260809.json` 中的实际值为
**4.6820**。对 released `3.0372` 的真实缺口是 **+1.645 cm**，而非 +0.95 cm——该短板此前被低估了
约 73%。**P10 收尾时必须同步修正上述三处文档**（封存 JSON 的数值行不动，只改其叙述性 `next_action`）。

设计：**2×2，两个二值因子，其余一切不变。**

- **F1** `hand_object_contact_hinge` ∈ {`0.0`（现状）, `0.02`} 米
- **F2** `hand_object_contact_detach_object` ∈ {`false`（现状）, `true`}

固定不变：`hand_object_contact_weight = 3.0`、`max_processed_windows = 299520000`、seed 42、
随机初始化、`hoi_architecture_variant: base`、`fk_weight 0.3569973401779424`、
`object_surface_weight 0.4772322188400037`、有效批次 2048（4 GPU × 512）、学习率 `1e-4`、
无 warmup、无梯度裁剪、无 EMA、无 AMP——即
`code/config/config_train_hoi_prior_p9w3.yaml` 的**其余每一个字段**。

臂位（2×2）：

| 格 | hinge | detach | 运行 id |
|---|---:|---|---|
| **A00** | 0.0 | false | **复用**封存 `p1-hoi-p8-hand-object-geom-w3-s42-20260807` |
| **A10** | 0.02 | false | `p1-hoi-p10-geom-hinge-s42-20260809` |
| **A01** | 0.0 | true | `p1-hoi-p10-geom-detach-s42-20260809` |
| **A11** | 0.02 | true | 新跑，run id 在其**实际启动日期**分配 |

**复用有效性门（A00 计为一格的前置条件）。** 必须有一个单元测试证明：改动后的实现在
`(hinge=0.0, detach=false)` 下，对固定随机批次返回的损失值与改动前实现**逐位相同**
（bit-identical）。**该测试不通过，A00 必须重训而不得复用**——否则 A00 与另外三格不再处于同一
目标函数下的对照，整个 2×2 失去内部效度。

评估：官方 438 序列、三窗口、500 步原生协议**零改动**，一律以**显式命令行覆盖**应用 P7 封存引导
配置（`arm=b`、`guidance_scale=1000.0`、`last_steps=10`、`clamp=1.0`、`clamp_target=update`、
`contact_mask_source=predicted`、`contact_mask_threshold=0.95`、`contact_weight=3.0`、
`consistency_weight=1.0`、`consistency_normalization=author`、`object_goal_weight=1.0`），
与 P8/P9 评估**逐字节相同**，四格因此协议匹配。**不得**写入
`code/config/sampler/hoi_prior.yaml` 的默认值。

事前方向性预测（写死，可证伪，无论对错均保留）：

- **H1 铰链主效应**：`hand_pen_loss_omomo` 与 `human_pen_loss_infbagel` 相对同 detach 设置的
  非铰链格**双双下降**。证伪条件：任一项不降。
- **H2 铰链对接触中性**：`contact_percent` 绝对变化 `< 0.02`。理由：评测器的接触判据是掌关节到
  物体顶点 `< 5 cm`（`code/eval_metrics.py:236`），铰链目标为 2 cm，尚余 3 cm 余量。
  证伪条件：变化 `≥ 0.02`。
- **H3 detach 主效应**：`end_obj_trans_err`、`obj_trans_dist`、`trans_dist` 相对同 hinge 设置的
  非 detach 格**三项全降**。证伪条件：任一项不降。
- **H4 detach 的接触代价**：`contact_percent` 在 detach 下**允许**下降，但必须保持 `> 0.60`。
  证伪条件：跌破 0.60。

判定规则：

- **PRIMARY**：某格**可选**，当且仅当在 438 序列配对 bootstrap 对 **A00** 的比较中同时满足
  (i) `hand_pen_loss_omomo` 显著更低；(ii) `human_pen_loss_infbagel` 显著更低；
  (iii) `contact_f1` **未**显著更低。
- **SECONDARY**：在可选格中取 `end_obj_trans_err` 最低者。
- **保护门**：`contact_percent ≥ 0.60`，且 `mpjpe` 相对 W3 的 `12.3134` 不得显著变差。
  违反者无论 PRIMARY 如何，一律取消资格。
- **STOP**：若无任何格满足 PRIMARY，分类为 `geometry-term-repair-negative-stop`，
  该方向关闭，**W3 保持为封存的接触配置**。

不确定性协议（**显式严于 P8/P9——那两轮一次 bootstrap 都没跑**）：

- 438 序列的**配对序列级 bootstrap**，10,000 次重采样，seed 42，**共享重采样索引矩阵**，
  取 2.5/97.5 百分位；
- 对**每一个**上报指标计算，且**同时**对两个基线计算：A00（W3）与 H0（D2-AI 无几何格，
  训练 `p1-hoi-d2ai-full-budget-s42-20260803`，评估 `p1-hoi-p8-eval-h0-guided-s42-20260806`）；
- **不做选择/确认分裂**：四格全部预注册，且不存在对任何连续量的事后选择，此处分裂没有对应的
  多重比较风险。

治理边界：

- 允许改动：`code/priors/losses.py`、`code/train_hoi_prior.py`、
  `code/config/config_train_hoi_prior.yaml`、`code/config/` 下三个新臂配置、`tests/`、
  `tools/`（受版本管理的 bootstrap 工具）、`docs/`、`experiments/`。
- **不可触碰**：`code/guidance_loss.py`（作者代码）、`code/eval_metrics.py`、官方 438 协议入口，
  以及 `code/config/sampler/hoi_prior.yaml` 的默认值。
- 两个新字段的默认值必须为 `hand_object_contact_hinge: 0.0` 与
  `hand_object_contact_detach_object: false`，故所有既有封存配置的目标函数逐字不变
  （与 P8 中 `hand_object_contact_weight` 默认 `0.0` 同一约定）。
- 种子 42 唯一合法；有效批次取 `{512,1024,2048,3072}`，1536 禁用；
  任何已发布 checkpoint 不得用于初始化。
- 每臂配置必须含显式 `run_id`（P8 教训延续）。

成本：三个新臂各约 22h @ 4 GPU。8 卡权威机上两臂并行 → 首波 22h、A11 次波 22h，墙钟约 **44h**，
合计约 **264 GPU·h**。评测四格，其中 A00 复用其封存的 `p1-hoi-p8-eval-w3-guided-s42-20260809`
输出、不重跑，新增三次 438 序列评测。无新增评测协议。

**执行位置更正（2026-08-09，用户指示）。** A11 不再排在次波，而是与 A10/A01 **同时**在 4 卡
worker `10.181.9.214`（`infbagel-4gpu`，node01）上执行，run id `p1-hoi-p10-geom-both-s42-20260809`，
配置 `code/config/config_train_hoi_prior_p10_both.yaml`。三臂并行后墙钟从约 44h 降到约 **22h**，
GPU·h 总量不变。这是**硬件位置**的改动，不是科学改动：worker 与权威机同为 RTX 3090 24GB，
A11 的种子、数据快照、有效批次（4×512=2048）、预算（299,520,000 windows）、目标函数与提交的
Git 对象与另外两臂逐字相同，评测仍在权威机上按 P7 封存引导执行。按 `AGENTS.md`
关于记录硬件替换的要求，A11 的 manifest 与 registry 条目须标明其在 worker 上执行；
worker 独占其 run 目录与 checkpoint 树，权威机不写入。
