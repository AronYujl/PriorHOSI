# Phase 1B-04：预算杠杆（P4 预算-指标曲线、D2-AI/D2-AJ 长预算双臂）

本文件于 2026-08-10 从 `docs/EXPERIMENT_PLAN.md` 第 8479-8882 行原样切出（逐字节复制，未改写、未重排、未修正任何笔误）。
导航：[总览](../OVERVIEW.md) · [Phase 1B 索引](README.md)

#### 2026-08-02 Phase 1B P4 预算-指标曲线（D2-X cadence 评测，用户批准）

动机：一个**此前从未被检视的共享常数**被发现。D2-X 的留出验证损失序列
(`p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723/metrics.json`, 20 个 cadence)
在 **24,576,000 窗口处触底**后单调上升，到封存的 61,440,000 点时 `total` 比最低点高
**+8.4%**、`contact` 项高 **+28.7%**。把同一检查扫过全部 D2 谱系，**九个配置无一例外**：

| run | argmin(total) | total drift @61.44M | contact drift |
|---|---:|---:|---:|
| D2-AF | 21.50M | +12.4% | +28.5% |
| D2-Y | 21.50M | +11.2% | +29.7% |
| D2-AG | 21.50M | +11.0% | +30.5% |
| D2-Z | 21.50M | +10.4% | +29.7% |
| D2-AE | 21.50M | +9.8% | +30.6% |
| D2-V | 24.58M | +8.9% | +27.3% |
| D2-X | 24.58M | +8.4% | +28.7% |
| D2-AD | 21.50M | +5.7% | +26.7% |
| D2-AC | 21.50M | +5.6% | +25.4% |

即 **61,440,000 这个固定预算本身就是谱系里每一个模型都越过其验证最优点约 2.5 倍的共享常数**，
而十个配置正是在这个共同的过训练点上被互相比较的。这一事实此前从未进入任何 D2 判定。

**但验证损失是去噪目标，评测指标是采样后的运动质量，扩散模型中两者可以脱钩。**
封存的 61.44M 点在原生指标上是否真的更差，**从未被测量**。本实验只测这一件事。

设计：对 D2-X 的 **6 个既有 cadence checkpoint** 跑冻结的官方原生协议（438 序列 × 3 窗口、
`sample_type=diffusion` 500 步 ancestral DDPM、`load_scene=false`、无 CFG、seed 42、
`--config-name=config_eval_hoi_prior`、`sampler.pelvis.guidance.enabled=false`）：

| 点 | windows | 角色 |
|---|---:|---|
| C1 | 3,072,000 | 早期，曲线左端 |
| C2 | 12,288,000 | 上升段 |
| C3 | **21,504,000** | **预注册的唯一比较点**（九个配置里七个的 contact argmin） |
| C4 | 24,576,000 | D2-X 的 total argmin |
| C5 | 43,008,000 | 最优点之后 |
| C6 | 61,440,000 | 封存 D2-X 点 |

判定规则（在任何结果存在之前逐字固定）：

- **PRIMARY（单一预注册比较，无多重性）**：`contact_f1` 在 **C3(21.504M) 对 C6(61.44M)** 的
  配对序列级 bootstrap（438 序列，seed 42，10,000 replicates，
  `tools/summarize_hoi_phase1b.py:112` 约定）。
  - 若 **C3 − C6 的 CI 下界 > 0**：过拟合**已兑现到指标空间**。结论为
    「固定 61.44M 预算劣于其内部最优点」，**增加预算方向被证伪**，转向早停/正则/数据方向。
  - 若 **CI 跨零或上界 < 0**：验证损失与指标**脱钩**，增加预算方向**保持开放**，
    本曲线成为它的基线。
  - C3 是**在看到验证曲线之后、但在看到任何指标之前**选定的，依据是九个配置的 contact argmin
    的众数（7/9 为 21.50M）。这一选择连同其依据在此固定，**不得在看到指标后更换比较点**。
- **报告但不设门**：18 个 aggregate 指标 × 6 个点全表。特别记录 `end_obj_trans_err`、
  `xy_points_err`、`hand_pen_loss_omomo`、`human_pen_loss_infbagel`、`foot_sliding`、
  `mpjpe`、`contact_percent`、`contact_precision`、`contact_recall`。
- **门控项**：每格 `nonfinite_values == 0` 且 `position_outside_rate == 0.0`。
- **树对照（承重）**：C6 是封存 D2-X 的同一 checkpoint。其 `per_sequence_metrics.json`
  的 sha256 **必须等于** `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`
  （`5f7dde7` 上 `p1-hoi-d2x-distance-probe-s42-20260801` 已逐位复现封存值）。
  不相等即判本实验执行环境失效，全部结果作废重跑，**不得沿用**。

治理边界（严格）：

- **这是测曲线，不是挑 checkpoint。** 六个点全部报告；封存的 D2-X 行**不被替换、不被重新定义**；
  D2-X 的 `windows061440000` 仍是该谱系唯一的正式最终权重。任何"改用 C3 作为 D2-X"的动作
  **不在本实验授权范围内**，需要另一次用户批准的实验。
- **不训练、不产生 checkpoint、不分配训练 run id、零源码改动**、不解除 `hoiprior_search_closed`、
  不改动任何既有分类。仅推理。
- 执行树 `/data/yujinlun/InfBaGel-head-baseline`、commit
  `5f7dde73903b78d70e6423d525de819e7f4ebfe3`（该树已被证明逐位复现封存 D2-X 的 18 个 aggregate）。
- 六格可并发于空闲 GPU；每格约 3.5 分钟，`hoi_timing_warmup=true`，
  **并发会污染计时**，故本实验的 wall-clock 数字不作性能记录，也不用于任何 ETA 判据。
- 允许改动的文件范围：`docs/EXPERIMENT_PLAN.md`、`experiments/registry.jsonl`、
  `experiments/results/`、`docs/HOIPRIOR_EVIDENCE_INDEX.md`。

事前预测（写死，无论对错都保留）：验证损失的 `contact` 项在 C3→C6 上升 28.7%，若指标空间同向，
`contact_f1` 应在 C3 高于 C6 约 0.01–0.03。**若实测跨零，则说明去噪损失的过拟合不等于采样质量的
过拟合**，这本身是关于扩散模型评估的一个可复用结论，须如实记录而非改述。

已被既有证据否决的备选：

- **(a) 直接跑 299.52M 的加预算实验。** 上表九个配置的一致漂移使其先验变差；且 26 GPU 小时
  对 15 分钟的信息量比不合理。本实验的结果决定它是否还值得做。
- **(b) 把 216 条留出序列并入训练以缓解过拟合。** 留出集正是产生上表证据的**唯一过拟合探测器**；
  并入即拆表。且固定预算下它只把重复次数从 108.1 降到 102.8。
- **(c) 用 8 卡合并训练提速。** `code/priors/losses.py:177` 的 `object_goal` 是**按 micro-batch
  自归一化**的掩码均值，终点窗口仅占 0.719%；micro-batch 从 512 降到 256 使"该 micro-batch 无终点
  窗口"的概率从 2.48% 升到 15.76%，令终点目标项被**相对压低约 13.6%**。换 8 卡会在自称"只改预算"
  的同时**静默改动目标函数**，且该项正是 `xy_points_err` 与 `end_obj_trans_err` 的监督来源。
  8 卡若要使用，须为两个并发的 4 卡任务（micro-batch 保持 512，`world_size in {1,4}` 守卫不变）。

#### 2026-08-02 Phase 1B P4 结果：预注册预测被反方向证伪，预算在 61.44M 处远未饱和

上一节预注册的六格已全部执行完毕（`exit=0`，各 438 序列）。**预注册写死的预测被证伪，且是反方向、
幅度为预测的 4–10 倍。**分类 `budget-not-saturated-preregistered-prediction-falsified`；compact result 为
`experiments/results/p1_hoi_p4_budget_metric_curve_s42_20260802.json`。

**PRIMARY（唯一预注册比较）：**

| | C3 = 21,504,000 | C6 = 61,440,000 | 配对差 | 95% CI |
|---|---:|---:|---:|---|
| `contact_f1` | 0.5291389 | 0.6374259 | **−0.1082871** | **[−0.1340293, −0.0826817]** |

预注册的事前预测是「C3 比 C6 高 0.01–0.03」。实测 **C3 低 0.108，CI 排除零**。按预注册判定规则，
CI 上界 < 0 落在第二支：**验证损失与原生指标脱钩，增加预算方向保持开放，本曲线成为其基线。**
实际上不止脱钩，是**强反相关**。

**承重树对照逐位通过。** C6 格的 `per_sequence_metrics.json` sha256 实测
`69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`，等于封存 D2-X 值，
18 个 aggregate 指标逐项复现。执行环境有效。

**六点 × 18 指标全表：**

| 指标 | 3.07M | 12.29M | 21.50M | 24.58M | 43.01M | 61.44M |
|---|---:|---:|---:|---:|---:|---:|
| `end_obj_trans_err` | 20.52380 | 6.07917 | 7.35438 | 6.51069 | 4.81796 | **3.74021** |
| `xy_points_err` | 29.53357 | 5.53157 | 4.99935 | 4.72658 | 4.11503 | **4.05052** |
| `feet_height` | 0.04469 | 0.03655 | 0.05258 | 0.06503 | 0.03855 | 0.04981 |
| `foot_sliding` | 0.24225 | 0.34337 | 0.48222 | 0.50273 | 0.30564 | 0.36301 |
| `contact_acc` | 0.54276 | 0.58620 | 0.63746 | 0.65110 | 0.70068 | **0.73164** |
| `contact_precision` | 0.62205 | 0.71083 | 0.77955 | 0.76007 | 0.80453 | 0.78806 |
| `contact_recall` | 0.33346 | 0.38405 | 0.46056 | 0.46668 | 0.53769 | **0.59445** |
| `contact_f1` | 0.38264 | 0.45270 | 0.52914 | 0.53422 | 0.60070 | **0.63743** |
| `contact_percent` | 0.28358 | 0.31172 | 0.37309 | 0.37256 | 0.42540 | 0.47655 |
| `gt_contact_percent` | 0.66188 | 0.66188 | 0.66188 | 0.66188 | 0.66188 | 0.66188 |
| `mpjpe` | 19.61843 | 15.15895 | 14.36971 | 13.32621 | 12.98136 | **12.05084** |
| `trans_dist` | 25.44906 | 11.59426 | 10.13523 | 10.17294 | 8.45819 | 8.17009 |
| `obj_trans_dist` | 38.03553 | 22.89349 | 22.91834 | 18.59287 | 18.08528 | 15.99405 |
| `obj_rot_dist` | 1.57839 | 1.27423 | 1.22983 | 1.09348 | 1.06210 | **1.03094** |
| `hand_pen_loss_omomo` | 0.13559 | 0.22671 | 0.17826 | 0.21399 | 0.22291 | 0.24536 |
| `hand_pen_ratio` | 0.10138 | 0.13190 | 0.12356 | 0.12918 | 0.13641 | 0.14387 |
| `human_pen_loss_infbagel` | 2.66693 | 3.90925 | 2.91002 | 3.44096 | 3.53873 | 3.86908 |
| `human_pen_ratio` | 0.12690 | 0.14694 | 0.13475 | 0.13444 | 0.14115 | 0.14619 |

**六项严格单调改善**：`contact_acc`、`contact_f1`、`contact_recall`、`mpjpe`、`xy_points_err`、
`obj_rot_dist`。**且最后一段增量比中段更大**：43.01M→61.44M 上 `contact_f1` **+0.0367**、
`contact_recall` **+0.0568**、`mpjpe` **−0.931**。**无任何饱和信号。**

**穿透与 foot sliding 随预算变差是参与度假象，不是退化。** `contact_percent` 单调 0.28358 → 0.47655，
而 GT 为 0.66188——**HOIPrior 在每一个预算点上都处于欠参与**。3.07M 处 `hand_pen` 仅 0.13559 而
`contact_percent` 只有 0.28358：手根本没靠近物体。这与 `EP:7912-7916` 记录的 epoch100 假象同构。

**预注册门控写坏了，按字面失败如实记录，不重新定义判据。** 门控 `position_outside_rate == 0.0`
在 6 格中 **5 格失败**（最大 8.624e-4；`nonfinite_values` 全格为 0）。原因是欠训练 checkpoint 会生成
超出归一化范围的位置，属预期行为。该门控系照抄 P2/P3——那里每格都是训练完整的 61.44M
checkpoint——未为早期 cadence 写容差或豁免。**看到数字之后不得改判据**，故记为按字面失败并保留为
预注册缺陷；它不影响曲线的有效性（`nonfinite` 全零、树对照逐位相同）。

**执行缺陷（登记但非可报告失败）：** 六格并发启动时未限制每进程线程池（185 线程 × 6 于 112 核），
拖慢了 CPU 侧 dataset/BPS 构建阶段。该失误发生在任何 run id 或 manifest 存在之前，按
`AGENTS.md` 属实现工作而非可报告实验；科学内容不受影响（同 checkpoint、同协议、同 seed、确定性），
且预注册已声明本实验的 wall-clock 不作性能记录。

**本次推翻的三项此前判断：**

1. **「九个配置的验证损失上升 ⇒ 61.44M 越过最优点」被证伪。** 该上升是真实的，但它衡量单步去噪，
   与 500 步链式 rollout 的原生指标**反相关**。验证损失此后不得用于预算、早停或 checkpoint 判定。
2. **「固定预算 61.44M 是使十个 D2 配置全部过训练的共享常数」不成立。** 它仍是一个未受控的共享常数，
   但方向相反：**十个配置都在远未饱和处收口**。
3. **「留出集是唯一过拟合探测器，故不可并入训练」的第二次反驳失去前提**（探测器指错方向）。
   是否并入 216 条仍未被本实验回答。

**本实验不能证明的：** 它读的是 D2-X 单一谱系的既有 cadence，不是一次新的全预算训练；
对 299.52M 的对数外推只作方向参考，`end_obj_trans_err` 的 −62% 由最后一段主导，量级不可信；
单谱系、单 seed。**封存 D2-X 行未被替换，`windows061440000` 仍是该谱系唯一正式最终权重。**

#### 2026-08-03 Phase 1B D2-AI 全预算与 D2-AJ 目标条件通路（双臂，用户批准）

动机：P4（`EP` 上节，compact result `p1_hoi_p4_budget_metric_curve_s42_20260802.json`）在官方协议
上测得 61.44M 处**六项指标严格单调改善且最后一段增量最大**（`contact_f1` +0.0367、
`contact_recall` +0.0568、`mpjpe` −0.931），无饱和信号。**这不是孤证**：`D2-W`
（`p1_hoi_phase1b_d2w_checkpoint_frontier_r1_s42_20260723.json`，2026-07-23 封存）在内部 rollout 上
比较同样的 24.576M 中点与 61.44M 终点，得到 `final_minus_midpoint_fk_foot_sliding.mean = −0.447`，
中点在六项能力保持检查中失败三项，且 `direct_fk_foot_disagreement_cm` 在 6.14M→24.58M→61.44M 上
单调 11.29→5.26→3.28。两次独立测量、不同协议、同一方向。（限制:D2-W 三点的
`physical_contact_precision` 均为 0，其与 P4 的一致仅在运动学侧。)

此前"长预算无正面证据"的判断从未被真正检验:D2-V 名为 long budget，其
`max_processed_windows` 同为 61,440,000——那是相对 6.144M 筛选预算的十倍，**确立**了 61.44M 为正式
预算，没有任何 D2 运行越过它。

**Arm 1 = D2-AI：D2-X 在 299,520,000 窗口（146,250 updates，4.875×）。** 唯一被操纵的因子是
`max_processed_windows`。其余科学设置与封存 D2-X 逐字相同：seed 42、随机初始化、有效 batch 2048
（`batch_size 512 × num_gpus 4 × accum 1`）、lr 1e-4 恒定、无 warmup、无 scheduler、无 EMA
（`minimum_lr_ratio: 1.0`、`weight_decay 0`、`amp false`）、balanced 损失权重
（`fk 0.3569973401779424`、`object_surface 0.4772322188400037`、`velocity 0.1`、`goal 1.0`）、
`fk_foot_temporal_routing: true`、`hoi_architecture_variant: base`、同一 4,088 序列 split。
**216 条留出序列不并入**——本次唯一被操纵的因子必须是预算，否则 Arm 1 与封存 D2-X 不再是单因子对比。

**Arm 2 = D2-AJ：目标条件通路拆分，预算与 Arm 1 对齐。** 当前
`models.py:116-118` 把 `goals`(9) + `progress`(3) 融合进单个 `Linear(12,512)` 并只发出**一个**
条件 token（`:298-303`）。其中 `goals[3:6]` 在训练（`data.py:200,206`）与评测
（`diffusion.py:519,521`）**两侧都从未被写入**，骨盆 y 恒为 0——12 个输入维中 4 维是死的。
released InfBaGel 则用三个独立模块各发一个 token。D2-AJ 改为:

```
pelvis_goal: Linear(2,512)+SiLU+Linear(512,512)   # 仅 xz
object_goal: Linear(3,512)+SiLU+Linear(512,512)
progress:    Linear(3,512)+SiLU+Linear(512,512)
条件 token 4 -> 6；self.position 20 -> 22
```

参数 +525,312（524,288 模块 + 1,024 位置编码），+1.77%。`architecture_variant` 记为
`d2aj_split_goal_tokens`。

**先验诚实声明（写在最前，不作脚注）：本臂更可能是惰性的。** 融合的 `Linear(12,512)` 已是同一批
信息的仿射映射，拆成三个仿射映射再作独立 token，在第一层近似重参数化;增益只可能来自**注意力
能按头、按帧分别加权两个目标**，而非容量。证据索引结论 9 记录接触杠杆不叠加（联合训练可把新通路
吸收为通用残差），同样的吸收风险适用于此。D2-AE 是前车之鉴:五个内部因果门全过、CI 分离良好，
原生迁移仍为零。**运行它的理由**是它是第一个专门瞄准**目标召回**缺口的改动（P4 已确立
`end_obj_trans_err` 与 `xy_points_err` 是目标召回而非预测指标:交给模型的物体目标**就是**
指标所评分帧的 GT 物体平移，全 438 序列吻合到 0.0574 cm;`pelvis_goal` 与第 15 帧 GT 骨盆吻合到
0.0000 cm 而指标评第 14 帧），且它搭在一个无论如何都要付的预算臂上。

**已知混淆（登记，不掩盖）：** D2-AJ 同时（i）拆分 token 与（ii）移除 4 个死输入维。任一效应都可能
来自其中之一。下述诊断第 4 项专门分离它。

判定规则（在任何 GPU 运行之前逐字固定）:

- **Arm 2 的 61.44M go/no-go（用户指定的提前停止规则）。** 两臂的 61,440,000 均为 cadence #20，
  自动写出。在该点对 D2-AJ 跑一次官方 438 序列原生评测（与封存 D2-X 逐字同协议、无引导）。
  - **继续到全预算**当且仅当:`contact_f1`、`end_obj_trans_err`、`xy_points_err` 三者中**至少一项**
    相对封存 D2-X 同预算点（`contact_f1 0.6374259`、`end_obj 3.7402086`、`xy_points 4.0505197`）
    的配对序列级 bootstrap CI **排除零且方向有利**;
  - **否则提前停止**，分类 `d2aj-goal-pathway-null-at-matched-budget-stop`，如实登记并保留
    checkpoint 与全部指标。
  - 该判据**只用 61.44M 点**，不看 Arm 1 的任何结果，不因 Arm 1 表现而调整。
- **Arm 1 无 go/no-go**:它是承重预算臂，跑满 299,520,000 无论中途指标如何。
- **主结果（两臂共同）**:各自最终 checkpoint 对封存 D2-X 的官方 438 序列配对 bootstrap
  （seed 42、10,000 replicates、`tools/summarize_hoi_phase1b.py:112` 约定），**18 个 aggregate
  指标全部报告**。
- **保护门（两臂）**:`nonfinite_values == 0`。**不设 `position_outside_rate == 0.0` 门**——P4 已
  实测该门对欠训练 checkpoint 会按字面失败（6 格中 5 格，最大 8.6e-4），此处改为**记录该值**而非
  设门;这一放宽在见到任何本次数值之前写定。
- **参与度强制读法**:任何穿透或 foot sliding 的改善必须与 `contact_percent` 一并报告。P4 测得该
  值在每个预算点都低于 GT 0.66188（0.28358→0.47655），**HOIPrior 全程欠参与**，孤立的穿透改善
  不得被叙述为几何改善。
- **D2-AJ 内部因果诊断（预注册，共享初始 latent／posterior noise／条件／顺序）**:
  1. **骨盆目标替换**——换成另一序列的骨盆目标。通路被使用当且仅当 `pelvis_goal_error_cm` 退化
     且 CI 排除零，**而 `object_goal_error_cm` 不退化**。
  2. **物体目标替换**——对称。1+2 共同检验**可分离性**，即所声称的机制本身;基线的单融合 token
     在原理上做不到分离，故基线对照应表现出交叉污染。
  3. **token 消融不对称性**——分别置零骨盆／物体 token，与基线置零单融合 token 对比。
  4. **死维对照（分离上述混淆）**——给基线模型的 `goals[3:6]` 灌随机噪声。若基线不受影响，则死维
     从未贡献，D2-AJ 的任何效应可归因于 token 拆分而非死维移除。
  5. **progress token 对照**——替换 `progress`，两模型应同向退化;显著不对称意味着拆分改变了
     progress 的使用方式，那不是本臂的主张。
  - **通路被使用**当且仅当 1 与 2 均呈现所预测的不对称退化且 CI 分离良好。**"被使用"不蕴含
    原生迁移**——D2-AE 正是反例。诊断结果不改变 61.44M go/no-go 的判据。

执行与资源（在启动前固定，因其影响吞吐记录）:

- 两臂**同机并发**，各 4 卡。**Arm 1 → GPU4-7**（该四卡两两 `PIX`），**Arm 2 → GPU0-3**
  （GPU1-2-3 互为 `PIX`，GPU0 为 `NODE`）。跨组为 `SYS`，故**两臂 all-reduce 各自留在本组 PCIe 内，
  不共享链路**。以 `taskset -c` 绑定 NUMA CPU（Arm 1: 28-55,84-111;Arm 2: 0-27,56-83);
  `numactl` 未安装。
- 数据集全部 `mmap_mode="r"`（`data.py:97-159`），两臂**共享同一份 page cache**（27G 数据、
  406G 已缓存、492G 可用），内存不构成竞争。每 rank 峰值显存 3.88 GiB，headroom 21.2 GiB。
- **竞争登记规则**:启动后实测每臂 windows/s，对照封存 D2-X 的 **3243.04 windows/s**。
  **任一臂低于 2757（−15%）即在 registry 与 compact result 中如实登记竞争及其幅度**，不得省略。
  预计单臂 ~26 h（`18945.21 s × 4.875`）。
- **不采用 8 卡合并训练**:`code/priors/losses.py:177` 的 `object_goal` 是按 micro-batch 自归一化的
  掩码均值，终点窗口仅占 0.719%;micro-batch 512→256 会使"该 micro-batch 无终点窗口"的概率从
  2.48% 升到 15.76%，令终点目标项被**相对压低约 13.6%**——而该项正是 `xy_points_err` 与
  `end_obj_trans_err` 的监督来源。合并会在自称"只改预算"的同时静默改动目标函数。

实现边界:

- `d2ai`/`d2aj` 均设 **`d2x_fk_foot_temporal_routing: false`**（`_is_d2x` 的唯一输入，
  `:2733-2734`）与 **`fk_foot_temporal_routing: true`**（真正被消费的 key，`:2587`、`:2911`）。
  二者独立，故两臂继承 D2-X 的 FK-foot routing 而**留在 `_validate_d2x_contract`
  （硬编码 61,440,000／30,000 updates，`:3261-3327`）之外**，并满足其余十二处
  `d2x_mode_off` 断言。
- 两个新模式须加入四个 allow-list:`_uses_author_update_rule`（`:2773-2778`，否则被强制
  AdamW + cosine + EMA）、`_validate_fk_foot_temporal_routing_mode`（`:2786`）、
  `_locked_loss_weights`（`:2870`，否则回落到 50/50）、
  `_validate_author_update_execution_host` 的 `modes` 元组（`:4410-4423`，否则
  `label = next(...)` 在每个 worker 里抛裸 `StopIteration`）。
- `:4425` 的 `hostname == "node01"` 门须**按模式放宽**，仅对 `d2ai`/`d2aj` 允许本机 `ubuntu`;
  既有十二个模式仍须 `node01`。这是唯一触及策略的改动，范围受限，不削弱任何封存模式的守卫。
- 终端预算 299,520,000 不落在 cadence（97.5）上，但退出保存块
  `if last_checkpoint_windows != processed_windows:` 会持久化最终权重，**无需凑整**。
  61,440,000 为 cadence #20，两臂均免费写出。
- 新模式的 ETA gate 若设置，须按 4.875× 预算定尺;沿用 61.44M 的常数会自证否决。
  既有 `D2A{E,F,G}_MAXIMUM_ETA_HOURS` 仅由各自模式契约调用，不影响新模式。
- **允许改动的文件范围**:`code/priors/models.py`、`code/train_hoi_prior.py`、
  `code/config/config_train_hoi_prior_d2a{i,j}.yaml`、`tests/test_hoi_d2a{i,j}*.py`、
  `docs/EXPERIMENT_PLAN.md`、`experiments/registry.jsonl`、`experiments/results/`、
  `docs/HOIPRIOR_EVIDENCE_INDEX.md`、`docs/phase_summaries/`。
  **`code/eval_metrics.py`、`code/test_infbagel_hoi.py` 与官方 438 协议零改动。**
- 因共享 model/training 代码改动，GPU 发布前跑一次完整 authority 测试套件、两臂各一次真实数据
  functional smoke（有限 forward/backward、梯度、显存、输出 API），并跑一次
  full-micro-batch 性能基准（两臂并发改变了计算/通信剖面，不得沿用封存剖面）。
- **released 或任何既有 checkpoint 不得用于初始化**（`AGENTS.md:11-12`）;两臂均随机初始化。
- 本次不解除 `hoiprior_search_closed`，不改动任何既有分类，不替换封存 D2-X 行。

被既有证据否决的备选:

- **(a) 把 216 条留出序列并入本次训练。** 会使 Arm 1 不再是单因子对比。P4 之后其正当性上升
  （过拟合探测器指错方向，见 `EP` 上节），但它应作为**独立的后续实验**，不与预算变量混合。
- **(b) 8 卡合并单臂。** 见上，静默改动 `object_goal` 定价。
- **(c) 为 D2-AJ 调整任何损失权重、LR 或 batch。** 未登记 sweep;两臂必须与封存 D2-X 在预算之外
  逐字同参，否则无法归因。
- **(d) 以验证损失做早停或 checkpoint 选择。** P4 测得留出去噪损失与原生指标**反相关**
  （21.504M 的 `contact_f1` 比 61.44M **低** 0.108，CI 排除零），该序列此后不得用于任何预算判定。

#### 2026-08-04 Phase 1B 双臂结果：预算首次带来广泛显著收益，目标通路为第十次模型侧失败

上节预注册的两臂均已终结。compact result 为
`experiments/results/p1_hoi_d2ai_d2aj_long_budget_arms_s42_20260804.json`。
分类 `budget-positive-goal-pathway-null`。

**承重树对照先行通过。** D2-AJ 变体在 `5f7dde7` 上不存在（那棵树的 `models.py` 不认识
`d2aj_split_goal_tokens`），故两次评测均在主仓库 HEAD 执行。为排除树效应，先跑一次 D2-X 树对照
（`p1-hoi-d2aj-gonogo-treecontrol-d2x-s42-20260804`），其 `per_sequence_metrics.json` sha256 实测
`69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`，与封存 D2-X **逐位相同**。
脚本设计为该对照不通过即退出、不产出任何 D2-AJ 数字。

**Arm 1 = D2-AI（`p1-hoi-d2ai-full-budget-s42-20260803`，completed）。**
299,520,000 窗口 / 146,250 updates / 526.87 epochs / 21.74 h，`loss_finite` 为真，
`amp_overflow_skips` 为 0，终端 checkpoint sha256
`a190e56c249161c0b52f0aebb097d0d5b95cb0c3810abb664000fc3c2fdda224`，参数 29,673,448
（与封存 D2-X 相同）。训练日志末尾有 DataLoader worker 收尾 traceback
（`RuntimeError: DataLoader worker ... killed by signal: Aborted`），**发生在训练完成且终端
checkpoint 落盘之后**；`status: completed`、记录哈希与重算哈希一致、权重全部有限，三者独立确证
其不影响权重。

**配对序列级 bootstrap（438 序列，同树，均无引导，seed 42，10,000 replicates）：
9 项显著更好，0 项显著更差。**

| 指标 | D2-AI | D2-X | 配对差 | 95% CI |
|---|---:|---:|---:|---|
| `obj_trans_dist` | 14.81981 | 15.99405 | **−1.17424** | [−1.59101, −0.76639] |
| `human_pen_loss_infbagel` | 2.76049 | 3.86908 | **−1.10860** | [−1.86033, −0.43539] |
| `trans_dist` | 7.70140 | 8.17009 | **−0.46869** | [−0.65694, −0.28444] |
| `mpjpe` | 11.74146 | 12.05085 | **−0.30939** | [−0.51845, −0.09656] |
| `hand_pen_loss_omomo` | 0.17481 | 0.24536 | **−0.07055** | [−0.11800, −0.02705] |
| `obj_rot_dist` | 0.99656 | 1.03094 | **−0.03438** | [−0.06589, −0.00267] |
| `contact_precision` | 0.81474 | 0.78806 | **+0.02668** | [+0.00980, +0.04528] |
| `hand_pen_ratio` | 0.11839 | 0.14387 | **−0.02548** | [−0.04532, −0.00560] |
| `human_pen_ratio` | 0.12216 | 0.14619 | **−0.02403** | [−0.04414, −0.00422] |

`contact_f1`（+0.01623）、`contact_recall`（+0.02035）、`end_obj_trans_err`（+0.07616）、
`foot_sliding`（−0.00866）均跨零。

**对 released 的缺口：反超项从 4/17 升到 8/17。** 两个最大的真实缺口被压到个位数，
**且这是训练侧收益、不是引导造成的**——无引导的 D2-AI 已达 `hand_pen 0.17481`、
`human_pen 2.76049`：

| | D2-X + Arm B | **D2-AI + Arm B** |
|---|---:|---:|
| `hand_pen_loss_omomo` | +41.3% | **+2.7%** |
| `human_pen_loss_infbagel` | +40.0% | **+1.8%** |
| `xy_points_err` | +5.5% | **−11.7%（反超）** |
| `obj_trans_dist` | +2.3% | **−5.8%（反超）** |
| `mpjpe` / `obj_rot_dist` / `hand_pen_ratio` / `human_pen_ratio` | 落后 | **全部反超** |

**代价在接触侧,如实记录：** `contact_f1` 缺口 +0.1% → **+7.1%**，`contact_recall`
+2.7% → **+12.3%**，`contact_percent` 偏离 +45.2% → **+140.5%**。机制是
D2-X + Arm B 的接触平价主要由引导把参与度从 `0.47655` 抬到 `0.56956` 买来，而
**D2-AI 对同一引导的响应显著变弱**（`0.49045` → `0.50899`）。

**P4 的方向正确、量级错误。** P4 用最后四点对数斜率外推 `contact_f1` 至约 `0.856`，实测
`0.65366`。"未饱和"的结论成立，外推量级不成立——幂律拟合对末段斜率过度自信。

**验证损失反相关在 4.875× 预算上以更强形式复现。** 留出去噪损失在 27,648,000 窗口触底，
到 299,520,000 时上升 **+22.7%**（对比 61.44M 处的 +8.4%），而同一区间原生指标 9 项改善、
0 项退化。**若以验证损失设早停,本次会在约 27.65M 处停止并放弃全部收益。**

**Arm 2 = D2-AJ（`p1-hoi-d2aj-split-goal-tokens-s42-20260803`，aborted）。**
在预注册的 61,440,000 判定点：

| 判据 | D2-AJ | D2-X | 配对差 | 95% CI | 判定 |
|---|---:|---:|---:|---|---|
| `contact_f1` | 0.63753 | 0.63743 | +0.00010 | [−0.02222, +0.02209] | 零 |
| `end_obj_trans_err` | 3.71761 | 3.76611 | −0.04849 | [−0.29809, +0.19882] | 零 |

**无一有利且显著 → `stop_early`**，分类 `d2aj-goal-pathway-null-at-matched-budget-stop`，
于 150,528,000 窗口（计划的 50.3%）终止，**省下约 8 GPU 小时**，49 个 cadence 保留，
manifest 封为 `aborted` 并明记 `operational_failure: false`。仅供参考、不进判定的
`pelvis_goal_error_cm` 为 `+0.19289`，CI [+0.04281, +0.34125]——**显著但方向不利**。

**本臂证实了它自己预注册中写下的先验。** 预注册明写"更可能是惰性的",理由是融合的
`Linear(12,512)` 已是同一批信息的仿射映射,拆分在第一层近似重参数化。实测比惰性更差。
**这是本阶段第十次失败的模型侧干预**:九次加损失/加交互表征,加一次改条件通路,方向一致。

**预注册缺陷,如实登记不静默修补。** 第三个判据 `xy_points_err` **在 per-sequence 输出中不存在**
（对应键为 `pelvis_goal_error_cm`,该映射本就记录于 `tools/run_hoi_d2ac_native_evaluation.py:63`,
撰写预注册时未回查）。故该判据不可评估、**不计为有利项**;`pelvis_goal_error_cm` 仅作参考报告,
事后计入等于放宽判据。判据为"至少一项有利",三项减为两项**只会更严**,不改变结论。

**并发代价实测约为零。** Arm 1 与 Arm 2 并发期 `3816.8` windows/s，Arm 2 停止后独占期
`3843.4`，**独占仅加速 1.007×**；两臂全程高于封存 D2-X 的 `3243.04`，远高于 `2757` 地板，
**按预注册规则无需登记竞争**。布局为 Arm 1 → GPU4-7（NUMA1）、Arm 2 → GPU0-3（NUMA0）、
`taskset` 绑核、数据集 `mmap` 共享 page cache。

**本次不能证明的：** 单 seed、单谱系；未孤立"为何更长预算有效"（预算与每窗口重访次数
526.87 epochs 混淆）；released 行仍协议不可比（16 步 CM + 引导 + CFG + scene/voxel 条件、
50,014,184 对 29,673,448 参数），对其的百分比仅为点估计、无 CI；`end_obj_trans_err` 未改善
且不预期改善——released 的 `3.03724` 来自 16 步 CM 采样，作者自己的扩散配方在同预算下同样补不上。

**下一入口点：接触参与度。** 它现在是唯一同时解释三项剩余接触侧缺口的机制
（`contact_percent` +140.5%、`contact_recall` +12.3%、`contact_f1` +7.1%）。
**D2-AI 让它更显眼而非更轻**：模型在每个预算点都欠参与（无引导 `0.49045` 对 GT `0.66188`），
且对推理引导的响应变弱。任何下一个提案必须直接瞄准参与度，并说明它为何不会像前十次模型侧
干预那样被联合训练吸收。

