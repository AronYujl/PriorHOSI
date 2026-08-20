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


#### 2026-08-21 Phase 1B P13 分布级指标列：预算的唯一干净对照（评测-only，用户批准）

动机：**HOIPrior 谱系有一整列指标从未被测量。** `results/experiments/` 下有 **45 个 run 目录**
持有完整的 438 序列 CHOIS 导出（`chois/{predictions,ground_truth}/*.npz`，键 `global_jpos`，
形状 `[126,24,3]`），其中 **36 个 prediction tree 互不相同**，而 CHOIS 分布级指标
（FID / MatchingScore / R-Precision / Diversity）只在其中 **4 个**上被算过：
D2-X 的 61.44M 点（2026-07-24 D2-AA，见本文件上文 D2-AA 小节）与 P12 的三格（2026-08-20）。
`AGENTS.md` "Reproducibility and reporting" 禁止遗漏预注册指标；Phase 1B 的 95% 原生 gate
本身就同时点名 `CHOIS FID/R-Precision`（`01_GATE_AND_EARLY_DIAGNOSIS.md`），但此后 D2-AI 之后
**没有一个 D2/P 臂报告过 FID**。整条谱系上唯一被测量到的有效杠杆——预算——因此从未在分布层面被检验。

本次只补**一个干净对照**：D2-AI 相对封存 D2-X 的唯一被操纵因子是 processed-window 预算
（本文件 2026-08-03 小节逐字固定），两格的 evaluation config 已实测逐键相同
（`dim_model 512 / num_heads 16 / num_layers 8`、`architecture_variant base`、438 序列、
`windows_per_sample 3`、`seed 42`、无引导原生 500 步、同日 2026-08-04 同树评测；
两份 `evaluation/aggregate_metrics.json` 在排除路径/`model_name`/吞吐后**没有任何配置键不同**）。
所以 FID 上的差就是预算的差。

**这不是训练、不是生成、不是新 checkpoint。** 本实验只把**已经落盘的关节导出**重新喂给
**预训练的外部 text-motion encoder**（`checkpoints/omomo/text_motion_features/model/finest.tar`，
sha256 `a125bc15ffd9772686737111c7501ecee0a2d8571d9aca348ec1195ddef78775`，不重训）。
不加载任何我们自己的 checkpoint，不做任何我们自己的模型推理。

设计（三格，一次调用，一份共享 GT embedding，一份共享 resample index）：

| 格 | 角色 | 预算 | 引导 | 绝对导出路径 |
|---|---|---:|---|---|
| **A** | PRIMARY 对照，封存 D2-X | 61,440,000 / 30,000 updates | 无 | `/data/yujinlun/InfBaGel-release/results/experiments/p1-hoi-d2aj-gonogo-treecontrol-d2x-s42-20260804/chois/predictions` |
| **B** | PRIMARY 处理，D2-AI | 299,520,000 / 146,250 updates | 无 | `/data/yujinlun/InfBaGel-release/results/experiments/p1-hoi-d2ai-native-eval-s42-20260804/chois/predictions` |
| **C** | 仅参考，D2-AI + Arm B | 299,520,000 | Arm B | `/data/yujinlun/InfBaGel-release/results/experiments/p1-hoi-d2ai-guidance-armb-s42-20260804/chois/predictions` |

- **A 是树对照格，不是新测量。** A 的 prediction tree sha256 实测
  `03e23d31e995152e77c69a7433c78ab728b140701cbb823eb33a6ebaf96063c1`，与 2026-07-24 D2-AA
  的 d2x 导出**逐位相同**，故 A 的 FID 已封存为 `1.7754768927164406`，200-replicate paired CI
  已封存为 `[1.3039275341029923, 2.4942378930141667]`
  （`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2aa-table5-completion-s42-20260724/candidates/d2x/chois/metrics.json`）。
  重算 A 的作用是**承重环境对照**，见下文门控 G1。
- **共享 GT 参考分布。** 三格的 `ground_truth` tree sha256 均实测为
  `d439a98ea32f5d67964bc98431fe25bdffc24b63e00b42601c5355445d01742c`；45 个完整导出中 **42 个**
  共享这一个 GT 树，另外 3 个是 P12 的修复后导出（`45bf2efe…` 与 `6f880b31…`）。
  **因此本次三格共用同一个 FID 参考分布，而 P12 连 GT 分布都不同。**
- B 的 prediction tree sha256 `0d787ef3e8d720b70a6aae28c1ab3c981ed0cffc3ed57e7df9edff71b9dc23d0`，
  C 的为 `87b02c5ef3bd9380d30ab974049bca41643723f08b737bc66a5a9cc7eeb1b48e`。

PRIMARY 统计量（在任何结果存在之前逐字固定）：

- **Δ = FID(B) − FID(A)**，配对 bootstrap，**单一预注册比较，无多重性**。
  - **resample 单位**：`paired_embedded_sequence`，**416** 条（不是 438，见下文非声明）。
  - **replicates 2000**，**seed 42**，percentile 2.5 / 97.5，linear 插值。
  - **一份共享 resample index matrix**：每个 replicate 抽一次索引 `I_r`，同一 `I_r` 同时用于
    GT、A、B、C 的 embedding 行，因此 `FID_B(I_r) − FID_A(I_r)` 是**同序列子集上的配对差**，
    与 `tools/paired_bootstrap.py` 的 `shared_resample_index_matrix` 约定同构。
  - **一次调用嵌入一份 GT**：三格必须在同一进程内共用同一个 GT embedding 矩阵，否则两次独立
    进程的 GT embedding 可能在末位比特上不同，配对性即失效。
  - 选 2000 而不是封存的 200，理由在见到任何本次数值之前写定：200 个 replicate 的 2.5 分位是第 5 个
    次序统计量，差值 CI 的尾部分辨率过粗。`RandomState(42)` 的抽样流是**前缀相容**的
    （已实测：2000 次抽样的前 200 次与 200 次抽样逐位相同），故 200-replicate 的封存 CI 可作为
    前缀被逐位复现，见 G1。additive 指标仍用封存的 **10,000** replicates（其分块抽样不具前缀性，
    但两侧都是 10,000，逐位可比）。

预注册停止分类（三支互斥且穷尽，在任何结果存在之前逐字固定）：

1. **Δ 的 CI 下界 > 0** → 分类 **`budget-fid-cost-confirmed-qualify`**。
   4.875× 预算在**分布层面**是有代价的：原生几何/接触侧 9 项显著改善、0 项退化（本文件
   2026-08-04 小节）与分布保真度**方向相反**。动作：给证据索引结论 7 加一条限定
   （"预算的收益不覆盖分布级指标"），并给结论 10 加下文固定的 scope 限定。**不选 checkpoint、
   不改 D2-AI 作为 HOIPrior v1 的决定**（那是用户 2026-08-10 的选择性决定，不由本实验推翻）。
2. **Δ 的 CI 跨零** → 分类 **`budget-fid-null-stop`**。
   分布层面测不到预算代价，指标列在这一对上补齐即止。结论 7、10、11 均不变。
3. **Δ 的 CI 上界 < 0** → 分类 **`budget-fid-gain-confirmed-extend`**。
   "更长预算带来分布级过拟合代价"这一假设被**直接证伪**：预算同时改善原生指标与分布保真度。
   动作：把指标列外推到剩余谱系（作为一次独立的、需另行批准的实验），结论 10 反而被第二个
   独立指标类加强。

门控（每一条都在见到数值之前写定；任一条失败即按字面失败如实记录，不重新定义判据）：

- **G1 承重环境对照。** A 格重算的 `FID` 必须等于 `1.7754768927164406`；A 格 replicate 序列
  **前 200 个**的 2.5/97.5 分位必须等于 `[1.3039275341029923, 2.4942378930141667]`。
  - 一级判据：绝对差 ≤ `1e-9`（视作逐位复现）。
  - 二级判据：若一级失败但绝对差 ≤ `1e-3`，登记为**跨主机环境漂移**并保留残差数值，实验**继续**
    ——PRIMARY 只要求 A 与 B 在**同一个环境**内配对，而 G1 是一个额外的跨主机一致性探针。
    这一放宽写在这里的理由是：封存值是 2026-07-24 在 **worker**
    （`/home/yujinlun/data/envs/infbagel/bin/python3.8`）上算的，而 CHOIS encoder + `scipy.linalg.sqrtm`
    路径的跨主机逐位一致性**从未被证明**（证据索引结论 19 只证明了我们自己的原生评测路径，
    并明写不覆盖其他路径）。
  - 三级：绝对差 > `1e-3` → 本实验执行环境失效，全部结果作废，**不得沿用**。
- **G2 配对身份。** 三格 × {predictions, ground_truth} 的
  `embedded_sequence_ids_sha256` 必须全部等于
  `6a4ffcfea5736a616d3ad5b582d16ec259002b2ff435212f19f4d4868dc57069`，
  `dropped_prediction_sequence_ids_sha256` 必须全部等于
  `b7ddcb96dae95814e44d1df8f4fe1791c2c7930ed3ddfca55c3ea3fcde31bd15`
  （两者已在本次预注册前独立复算并与 D2-AA、P12 的封存值逐位相同）。
- **G3 帧数常量。** 三格全部 438 条导出的 `global_jpos.shape[0]` 必须恒为 **126**（已实测）。
  这一条是承重的：CHOIS 的 `EvaluatorModelWrapper.get_co_embeddings` / `get_motion_embeddings`
  内部按 `np.argsort(m_lens)[::-1]` 重排返回行（上游注释 "the results does not following the order
  of inputs"）。`m_lens` 恒定时该置换是 numpy 不稳定排序的确定性产物（numpy 1.24.4 下为
  `[31,30,1,…,29,0]`），三格相同，故行配对成立；一旦某格帧数不同，该置换变成真实排序，
  **行配对即失效**。
- **G4 输入身份。** 三格 prediction tree sha256 等于上表所列；GT tree sha256 等于
  `d439a98e…`；evaluator 身份为 CHOIS commit `8ec585aa0200fd2a890ffb12897bcf69ae719463`、
  text-to-motion commit `72df96ec453edea2fbe9603b1d58a955eaf71636`、feature checkpoint
  `a125bc15…`、annotations `3fec528a…`、mean/std `a55c020c…`。
- **G5 有限性。** 任一 FID replicate 非有限即失败（adapter 已 fail-closed）。
- **不设** `position_outside_rate` 一类原生门控：本实验不生成运动。

**报告但不设门**（informational，不进入任何判定）：

- 三格各自的 `Diversity`、`MatchingScore`、`R-Precision@1/2/3` 点估计与已注册 CI；
- **C 格**：`FID(C)`、Δ(C−B)（**修复前表示内**的引导主效应）、Δ(C−A)。C 与 B 差的是引导、
  不是预算，故它不能进入 PRIMARY；列它的理由是 P2/P3/P5/P6 从未在分布层面测过引导。
- 剩余谱系导出**明确标注为 exploratory**：45 个完整导出中 36 个 prediction tree 互异，已算 FID 的
  4 个之外**还有 32 个**未测；其中 6 个 run 共享同一个 prediction tree
  （`87b02c5e…`：`p1-hoi-d2ai-guidance-armb` 与 P6 的 b0control/b1/b2/b3/b4 的 08-04 批次），
  那是 P6 已登记的"四格是 B0 的逐位重跑、0/438 条序列有差异"
  （`docs/phase_summaries/PHASE_1B_P6_GUIDANCE_SUBTERM.md:31`），**不是本次发现的新缺陷**，
  但它意味着"42 个待测导出"实际只有 32 个不同的分布。
  另有 P4 的六个 cadence 导出在**另一棵树**
  （`/data/yujinlun/InfBaGel-head-baseline/results/experiments/p1-hoi-p4-cadence-curve-w*`），
  **不在本次范围内**：跨树环境身份需另行论证。

**验证损失问题的固定解读规则**（在见到任何结果之前决定）：

证据索引结论 10 的"留出去噪验证损失与原生指标反相关，不得用来决定预算/早停/checkpoint"是
**只对原生几何与接触指标类**建立的（P4 的 `contact_f1`，D2-AI 的 18 项 aggregate），
**从未对任何分布级指标检验过**。P12 实测其留出 `total` 在 21,504,000 触底、到 299,520,000 上升
**+25.87%**（`results/experiments/p1-hoi-p12-frame-repair-baseline-s42-20260819/metrics.json`），
证据索引记录 D2-AI 在 27,648,000 触底、上升 **+22.7%**。

- **若落在分类 1（Δ 显著 > 0）**：结论 10 **必须**获得如下 scope 限定，措辞在此固定：
  「该反相关是在 500 步原生几何/接触指标类上建立的；在分布级 CHOIS FID 上，预算方向与验证损失
  **同向**而非反相关，故不得再以无限定的形式陈述。」
- **若落在分类 2 或 3**：结论 10 **不变**；分类 3 额外提供第二个独立指标类的同向证据。
- **无论落在哪一支，本实验都不授权用验证损失做早停、checkpoint 选择或预算决定。** 理由在此写死：
  两个格（61.44M、299.52M）**都在验证最优点之后**（D2-X 的 argmin 为 24,576,000，D2-AI 为
  27,648,000），本设计**没有任何一格落在验证 argmin 上**。所以即使分类 1 成立，它说的是
  "越过最优点之后继续加预算，分布保真度变差"，**不是**"停在最优点会更好"。后者需要 P4 cadence
  导出那一列，属另一次实验。

明确的非声明（每一条都写在结果之前）：

1. **本次没有任何一格跨过 2026-08-19 的表示帧修复。** 三格都是修复前表示，且 P12 的三格连
   `ground_truth` tree 都不同（`45bf2efe…` / `6f880b31…` 对本次的 `d439a98e…`），
   即**参考分布本身不同**。**任何 P12 的 FID 行（2.3074 / 2.0113 / 2.8174）都不得与本次数字对置**，
   也不得用来解释本次的 Δ。P12 自己的 `control_minus_primary = 0.510` 同理是跨 GT 分布的差，
   本节不引用它、不修正它。
2. **FID 是 416 序列量，不是 438。** 上游 loader 固定 `batch_size=32, drop_last=True`，438 条中
   416 条进入 embedding，被丢的 22 条是按 `seq_name` 排序后的最后 22 条、全部为
   `sub17_woodchair_*`，且**在每一格都是同一批**（G2 已把它变成一条硬门控）。任何把 FID 说成
   "438 序列指标"的表述都是错的。
3. **本设计只能显示 FID 是否随预算移动，永远不能解释为什么。** 它不分离预算与每窗口重访次数
   （526.87 epochs），不定位是哪个运动学属性驱动了 embedding 空间的位移。
4. **`embedded_sequence_ids` 的顺序不等于 embedding 行的顺序**（G3 记录的上游置换）。它作为
   **集合**是正确的，作为**逐行标签**不是。因此本实验**不得**做任何按序列归因、留一分析或
   "哪些序列拉高了 FID"的叙述——那需要先修上游置换的标签，属另一次实验。
5. **released 的 FID `0.9334244584430564` 是协议不可比的参考点**（16 步 CM + 引导 + CFG +
   scene/object-voxel 条件），只作锚点，不作模型差。
6. **不选 checkpoint、不改任何既有分类、不替换封存 D2-X 行、不改动 D2-AI 作为 HOIPrior v1 的
   决定**、不解除任何既有 stop。单 seed 42，只报点估计与已注册的 sequence 级 bootstrap，
   不声称跨 seed 置信区间。
7. **机制性解读留空。** 本 FID 实际在什么特征空间上计算（movement encoder → motion encoder 的
   哪一层、归一化用的是 `t2m_mean_std_jpos.p` 的哪一组统计量）正由另一路分析给出。
   **本设计的判定规则不依赖那个答案**；答案到达后作为一条独立 amendment 追加到本节之下的
   「机制性解读」槽位，不改动上文任何门控或分类。

治理边界（严格）：

- **不训练、不生成、不产生 checkpoint、不加载任何我们自己的 checkpoint、不分配训练 run id。**
  仅对已落盘导出跑外部 encoder。
- 执行机为**权威机 8 卡 `10.184.17.253`**，encoder 用一张空闲 GPU（`--device cuda`），
  bootstrap 是 CPU 端 numpy/scipy。`AGENTS.md` 的"HOI 评测不在权威机跑"是为了保护**吞吐记录**；
  本实验**不做任何吞吐或 FPS 声明**，且封存的 p0 CHOIS 行
  （`p0-hoi-chois-matched-s42-20260712`）与 2026-08-20 的 P12 CHOIS 行**都是在权威机上跑的**，
  precedent 一致。**本实验的 wall-clock 不作性能记录，不用于任何 ETA 判据。**
- 设 `OMP_NUM_THREADS=4`（实测该 sqrtm/协方差路径 1/4/16 线程分别 603/508/482 ms 每 replicate，
  不设上限在 112 核上会起线程风暴；这只影响 wall-clock，不影响数值）。
- 输出不可覆盖：adapter 的 `atomic_output` 对已存在路径直接拒绝。已存在的 manifest/result/
  checkpoint 一律不覆盖、不删除。

预期的源码改动（**这是本节唯一的实现声明**）：

`tools/run_chois_evaluator.py` 现有的不确定性代码**只做 per-run 的配对 GT-vs-prediction bootstrap**
（`_bootstrap_fid_interval:344-385`：每个 replicate 抽一次索引，同时施加于 GT 与 prediction），
**没有任何跨模型配对差的路径**，且**不持久化 per-replicate 序列**，故两次独立调用的输出无法事后
拼出差值分布。因此需要**一个默认关闭的新参数**，而不是新脚本：

- `--compare-predictions <dir>`（可重复给出，默认空）。给出时：GT 只嵌入一次，
  `--predictions` 与每个 `--compare-predictions` 各嵌入一次，**在同一个 replicate 循环内**用
  同一 `I_r` 算出每格 FID 与**全部两两配对差**的 2.5/97.5 分位，并记录
  `resample_index_sha256`、每格 `embedded_sequence_ids_sha256` 与 per-replicate 序列的 sha256。
- 默认路径（不给该参数）**逐字不变**：点估计公式、RNG 顺序、输出键、`--fid-bootstrap-replicates`
  与 `--bootstrap-replicates` 的语义全部不动，新增键只在给出新参数时出现。这与 2026-07-24
  D2-AA 给同一个 adapter 加 opt-in 参数的先例一致（本文件 D2-AA CPU/code implementation
  contract 小节：「既有 point-estimate 路径保持原公式和 RNG 顺序；新增参数默认均为 disabled」）。
- **不改** `code/eval_metrics.py`、`code/test_infbagel_hoi.py`、官方 438 协议、
  `tools/chois_evaluator.py` 的任何既有函数、以及 CHOIS/text-to-motion 两个第三方 checkout。
- **不改** `code/priors/core/`（冻结契约）。
- 断言加到 `tests/test_research_governance.py`（该 adapter 的既有测试所在模块）；
  按 `docs/EXPERIMENT_CONVENTIONS.md` 第 3 条，**不新建以实验 id 命名的测试文件**。
- 因该 adapter 不在共享 model/diffusion/training/data 路径上，也不改变任何 per-step 计算、通信、
  数据加载、张量形状或显存剖面，**不需要 full-micro-batch 性能基准**；但因它是 runtime 代码改动，
  **需要一次真实数据 functional smoke**：用同一批已落盘导出在 `--fid-bootstrap-replicates 4`
  下跑通并核对默认路径输出与新参数关闭时逐键一致。改动前后各跑一次完整 authority suite。

**没有 Hydra config override fragment**，因为本 adapter 是 argparse CLI、不在 Hydra 路径上；
`docs/EXPERIMENT_CONVENTIONS.md` 第 1 条的等价物是**把完整解析后的命令归档在输出旁的
`resolved.json`**，与 D2-AA 的
`…/p1-hoi-d2aa-table5-completion-s42-20260724/candidates/d2x/chois/resolved.json` 同格式。

允许改动的文件范围：

- `tools/run_chois_evaluator.py`
- `tests/test_research_governance.py`
- `docs/plan/PHASE_1B_HOI/04_BUDGET_AND_LONG_ARMS.md`（本节）
- `experiments/registry.jsonl`（一条 hypothesis、一条 completion）
- `experiments/results/`（一份紧凑 JSON）
- `docs/HOIPRIOR_EVIDENCE_INDEX.md`、`docs/phase_summaries/`
- 输出目录 `results/experiments/p1-hoi-p13-fid-budget-contrast-s42-20260821/`（不进 Git）

成本（实测外推，非声明）：一次调用三格、N=2000 时 bootstrap 约 **16–17 分钟**
（compare 模式实测 508 ms/replicate @ OMP=4，三格两两差需 3 次 sqrtm，约 26 分钟），
加 embedding 与 876×3 个文件的 tree sha256 约 2 分钟，**总计约 30 分钟**，单 GPU 只用于 encoder
前向（秒级）。N=200 时约 4 分钟。

事前预测（写死，无论对错都保留）：**Δ = FID(B) − FID(A) 的 CI 跨零（分类 2）。** 理由：D2-AI 在
原生几何上 9 项显著改善、0 项退化，但接触参与度偏离 GT 从 +45.2% 扩大到 +140.5%
（本文件 2026-08-04 小节），两者对 embedding 空间的推力方向相反；且封存 A 的 200-replicate 边际 CI
已宽达 `[1.304, 2.494]`，配对后即使收窄，0.2 量级的差也很可能仍跨零。**若实测显著为正，
则本节分类 1 生效，且结论 10 的无限定表述被推翻**——这正是本实验值得做的那一支。

已被既有证据否决的备选：

- **(a) 一次性补齐全部 32 个未测导出。** 32 格 × 一次 sqrtm 循环不贵，但它把一个单因子对照
  混进 32 重比较，且其中大量格之间同时变了引导、几何权重与预算，**没有一个是单因子**。
  先做唯一干净的那一对，其结果决定是否值得铺开——与 P4 拒绝 (a) 的同一理由。
- **(b) 用两次独立调用的两条边际 CI 判"不重叠"。** 两条 CI 共享 GT 与同一个索引流，
  彼此不独立；不重叠既非充分也非必要条件，且 200 replicate 的尾部太粗。这是把配对信息扔掉。
- **(c) 把差值计算放进 `tools/paired_bootstrap.py`。** 该工具的数据模型是**按序列名索引的
  per-sequence 标量**，FID 不是 per-sequence 量，塞进去等于在一个 1,261 行、承载封存
  shared-index 约定的工具里开一条新代码路径，风险大于在 adapter 里加一个默认关闭的参数。
- **(d) 让 adapter 只 dump per-replicate 序列、差值在外部算。** 改动更小（~12 行），但差值会由
  一条无 provenance 的临时算式产生，且 GT 会被嵌入两次（两进程的 GT embedding 末位比特不保证相同，
  配对性可能失效）。
- **(e) 把 P12 的三格并进本次对照。** 参考分布不同（GT tree 不同），跨 2026-08-19 修复边界，
  按上文非声明 1 直接禁止。
- **(f) 在 4 卡 worker 上跑。** worker 快照按 `AGENTS.md` 只有 OMOMO 数据，
  `third_party/chois_omomo_evaluator_assets` 与两个第三方 checkout 的身份需重新哈希校验；
  而权威机上这些资产的哈希已被 p0 与 P12 两次封存。本实验无吞吐声明，故无须占用 worker。

<!-- 机制性解读槽位（待另一路分析到达后作为独立 amendment 追加；不改动上文任何门控或分类） -->

##### 2026-08-21 增补：去竖直偏置的 informational 诊断格（用户批准）

**动机（本次会话实测，非本实验产出）。** CHOIS FID 在这条特征路径上由一次**刚体竖直偏移**支配：
编码器消费的是归一化后的绝对关节位置（`[T,24,3]` → `[T,72]` → 逐坐标 `(x−mean)/std`），
而该归一化的竖直轴 std 为 `0.1683 m`，对水平轴的 `0.4317 / 0.4825` 而言最紧，
故 3 cm 在竖直方向折合 `0.178σ`、在水平方向仅 `0.069σ`（z 方向 std 最小的关节为 `0.0209 m`，
3 cm 在它上面是 `1.44σ`）。对 P12+Arm B 减去单一三数均值偏置向量使 FID 由 `2.011` 降到 `0.3835`，
对发布权重用其自身向量则由 `0.9332` 降到 `0.4850`。

**后果，必须在本实验开跑前写下：** 若 FID 由竖直偏置支配，则 `FID(B) − FID(A)` 主要反映
D2-X 与 D2-AI 的**竖直偏置差**，而 `feet_height` 显示那只有 `0.04981 → 0.05090`，即 **0.11 cm**。
因此主草案给出的事前预测（"可能出现无信息的 null"）现在有了机制层面的理由。
**这不改变任何判定规则**，只是把预期信息量如实下调，并促使本增补把该混杂单独测出来。

**增补内容：** 在同一次调用内追加两个 informational 格。

| 格 | 构造 |
|---|---|
| **A′** | A 格（D2-X）减去其自身的一个全局固定竖直/均值偏置向量后重算 FID |
| **B′** | B 格（D2-AI）减去其自身的一个全局固定偏置向量后重算 FID |

**逐条约束（用户 2026-08-21 指定，逐字固定）：**

1. **`Δ = FID(B) − FID(A)` 仍是唯一 PRIMARY。** 三支停止分类
   （`budget-fid-cost-confirmed-qualify` / `budget-fid-null-stop` / `budget-fid-gain-confirmed-extend`）
   与门控 G1–G5 **一字不改**。A′/B′ 与 `Δ′ = FID(B′) − FID(A′)` 全部为 **informational**，
   不进入任何判定、不构成第二个 PRIMARY、不产生多重性修正问题。
2. **偏置必须在同一批 416 条嵌入序列上计算**——与 PRIMARY 完全同一子集，不是 438 条。
3. **每个格只减一个全局固定向量。** 该向量为该格全部 416 条序列、全部 126 帧、全部 24 关节
   对 GT 的均值差，构成一个三数向量，对该格所有序列所有帧所有关节同一地减去。
4. **禁止逐序列校正。** 已实测逐序列校正显著更弱（P12+ArmB 0.906 对全局 0.383），
   因为 FID 是分布统计量，抹掉各序列自身偏移会破坏 GT 本身的序列间方差结构。
   本实验不得实现、不得报告逐序列变体。
5. **校正后的 FID 不是模型的正式成绩。** A′/B′ 只作为"竖直偏置贡献了多少 FID"的诊断读数。
   任何表格、摘要或 phase summary 引用 A′/B′ 时必须在同一处注明它是事后减去全局偏置的诊断量。
   **不得**把 A′/B′ 与发布行的 `0.9334244584430564` 并列比较高下。

**A′/B′ 回答的问题：** 把两个格各自的竖直偏置归零之后，预算是否还在分布层面留下差异。
`Δ′` 与 `Δ` 的对比即"预算对 FID 的作用中有多少只是经由竖直偏置"。

**报告口径。** 输出 JSON 需同时记录：每格实测的三数偏置向量、其竖直分量、A′/B′ 的 FID
点估计，以及 `Δ′` 的点估计与 CI（复用 PRIMARY 的同一条共享 resample index，
使 `Δ` 与 `Δ′` 在同一重采样下可比）。偏置向量本身也要落进 JSON，因为它是可独立复核的量。

**成本。** 每个 informational 格多一次编码器前向（秒级）加一次 bootstrap 循环；不新增 GT 嵌入。

**非声明（追加到主草案的非声明列表）。**

- A′/B′ 不表示"模型经过校正后更好"。它们表示 FID 对一个特定的、可独立测量的缺陷敏感。
- A′/B′ 不授权在导出路径或评测路径上加任何偏置校正。事后减偏置是**诊断，不是修复**；
  修复必须在训练侧找成因，那是另一个需单独批准的实验。
- A′/B′ 的偏置向量跨 A、B 两格不同，因此 `Δ′` 不是"同一校正下的预算效应"，
  而是"各自最优刚体校正之后的剩余预算效应"。这个区别必须随 `Δ′` 一起报告。

##### 2026-08-21 预注册完整性披露：本节的 PRIMARY 点估计在提交之前已被看到

必须写在这里，因为它削弱本预注册的证据力，而读者无法从别处看出来。

**发生了什么。** 本节上文自己要求「一次真实数据 functional smoke……用同一批已落盘导出在
`--fid-bootstrap-replicates 4` 下跑通」。那次 smoke 已在 Stage C 实现阶段执行，**用的正是
A/B/C 三个真实格**，因此产出了：

| 量 | smoke 实测（N=4） |
|---|--:|
| FID(A)，封存 D2-X | 1.7754769073836485 |
| FID(B)，D2-AI | 1.4641165319284255 |
| FID(C)，D2-AI + Arm B | 1.1278846941538632 |
| **Δ = FID(B) − FID(A)** | **−0.311360375455223** |
| Δ 的 4-replicate 区间 | [−0.4748328090834761, −0.14987326254437966] |
| Δ′ = FID(B′) − FID(A′) | −0.22319218234221694 |

**FID 点估计与 replicate 数无关**，故上表的 Δ 就是 N=2000 下将要报告的同一个 PRIMARY 点估计，
它不会再变。**决策统计量（N=2000 的 CI）尚未产生**——4 个 replicate 的区间不是预注册的那个统计量，
它的尾部分辨率正是上文拒绝 N=200 的理由所排除的。但 4 个 replicate 全部落在零以下，已使
分类 3（`budget-fid-gain-confirmed-extend`）成为大概率结果，即**上文那条事前预测（分类 2，
CI 跨零）很可能是错的**。该预测按原样保留，不修改、不重新表述。

**G1 的结果也已经知道。** A 格重算得 `1.7754769073836485`，与封存值 `1.7754768927164406`
相差 **1.4667e-08**：**一级判据（≤1e-9）失败，二级判据（≤1e-3）通过**，故 G1 将按上文字面
记为跨主机环境漂移并继续。这是上文在见到任何数值之前就写好的分支，不是事后放宽——但读者应当
知道，写下那个二级判据时我尚未知道它会被用上。

**这个瑕疵的准确边界。** 真正要求「规则先于数字」的那条性质是成立的：本节正文与登记行的草稿
文件 mtime 为 `2026-08-21 00:14:42`，A′/B′ 增补为 `00:48:11`，而第一份 smoke 输出为
`00:51:52`、偏置诊断输出为 `01:08:25`（`--compare-predictions` 的实现落盘于 `00:49:10`）。
三支分类、G1–G5、事前预测与非声明列表在那之后**一字未改**。失效的是**提交时刻**：本次提交
发生在 smoke 之后，因此上述先后顺序的唯一凭据是 git-ignored scratch 目录里的文件 mtime，
而 mtime 是我能改的。**所以这条先后性是声明，不是证明**，读者应当按「未经时间戳保护的预注册」
来折算它的证据力。

**本该怎么做。** functional smoke 应当跑在**别的**输入上——任意两个非 A/B 的封存导出，或打乱的
序列名——它同样能验证代码路径与默认路径逐键一致性，却不会产出 PRIMARY。这是设计错误而非执行
错误：上文写「用同一批已落盘导出」时没有排除 A/B/C 本身。今后凡预注册里含 runtime 代码改动的
functional smoke，都必须指定**非 PRIMARY 输入**。

**判定不因此改变。** 用户 2026-08-21 明确选择「照常提交并跑，显式披露瑕疵」。三支分类与
G1–G5 按上文字面执行；本次运行的作用是把 N=2000 的 CI、A′/B′ 与全部 informational 量补齐，
并把 Δ 的点估计固定在一份有 provenance 的封存结果里。**任何引用本实验结论的地方必须同时引用
本披露节**，包括证据索引与 phase summary。

#### 2026-08-21 Phase 1B P13 结果：预算在分布层面是 null，被证伪的是我在 smoke 后的改判

分类 **`budget-fid-null-stop`**（预注册三支中的第 2 支）。结论 7、10、11 **全部不变**，
按上文固定的解读规则，结论 10 **不获得**任何 scope 限定。

| 量 | 点估计 | 2000-replicate 配对 CI |
|---|--:|:--|
| **Δ = FID(D2-AI) − FID(D2-X)（PRIMARY）** | **−0.3113603755** | **[−0.7797988497, +0.0582610293]** |
| FID(A) 封存 D2-X，61.44M | 1.7754769074 | [1.3138375616, 2.5989216374] |
| FID(B) D2-AI，299.52M | 1.4641165319 | [1.1376242347, 2.0969544186] |
| FID(C) D2-AI + Arm B 引导（informational） | 1.1278846942 | [0.8734578171, 1.6794956215] |

**PRIMARY 的 CI 跨零。** 4.875× 预算在分布保真度上既测不出代价、也测不出收益。
"更长预算带来分布级过拟合代价"这一假设**没有得到支持**，但也没有被证伪成"预算有分布级收益"——
点估计方向为负（预算更好），幅度 0.311，而区间宽 0.838。

**事前预测是对的，我在 smoke 之后的改判是错的。** 上文披露节写「4 个 replicate 全部落在零以下，
已使分类 3 成为大概率结果，即事前预测（分类 2）很可能是错的」。**实测否掉了这句话。**
4-replicate 区间 `[−0.4748, −0.1499]` 比 2000-replicate 区间窄约 **3.7 倍**，并指向错误的分支。
这正是本节在见到任何数值之前拒绝小 replicate 数的理由，现在有了直接证据。
写在预注册里的那条事前预测原样成立，**不是**我后来那句话。

### 门控

| 门 | 结果 |
|---|---|
| **G1 承重环境对照** | **二级判据**。A 格重算 1.7754769074 对封存 1.7754768927164406，残差 **1.4667e-08**；前 200 replicate 分位 [1.3039276576, 2.4942382264] 对封存 [1.3039275341, 2.4942378930]，残差 1.2349e-07 / 3.3335e-07。一级（≤1e-9）失败，二级（≤1e-3）通过，按上文字面记为**跨主机环境漂移**并继续 |
| **G2 嵌入口径** | 通过。四个集合的 `embedded_sequence_ids_sha256` 全为 `6a4ffcfe…`、`dropped_…` 全为 `b7ddcb96…`、416/22 |
| **G3 帧数常量** | 通过。438 条全为 126 帧；13 个 batch 上**只有一个**置换，即预注册预言的 numpy 不稳定排序确定性产物，故逐行配对成立 |
| **G4 输入身份** | 通过。A `03e23d31…` / B `0d787ef3…` / C `87b02c5e…` / GT `d439a98e…`；CHOIS `8ec585aa…`、text-to-motion `72df96ec…`、feature checkpoint `a125bc15…` |
| **G5 有限性** | 通过。输出中每一个浮点数有限 |

### 报告但不设门

- **引导主效应 Δ(C−B) = −0.3362318378，CI [−0.4750274515, −0.2353553402]，不含零。**
  引导在修复前表示内**显著改善**分布保真度。这是本分支第一次在分布层面测引导——
  P2/P3/P5/P6 都没测过。它差的是引导不是预算，故不进 PRIMARY。
  注意它的区间比 PRIMARY 窄得多：C 与 B 同一 checkpoint、只差引导，两格嵌入在重采样间高度相关，
  配对后方差大幅抵消；A 与 B 是两个不同 checkpoint，协变更少。这正是配对 bootstrap 该有的行为。
- Δ(C−A) = −0.6475922132，CI [−1.1434670920, −0.2837032317]，不含零，且**恰等于** null 的预算项
  与显著的引导项之和；抬起它的是引导项。
- additive（10,000 replicates）：Diversity A 8.7810 → B 9.2987 → C 9.2506；
  MatchingScore 3.8798 → 3.9164 → 3.9092；R-Precision@1 0.1514 → 0.1635 → 0.1635。
- **A′/B′ 去全局竖直偏置诊断（informational，不是模型成绩）：**
  A 1.7755 → **0.6332**（偏置向量竖直分量 **2.792 cm**），B 1.4641 → **0.4100**（**2.305 cm**）。
  即**一个三数刚体向量在 A 上值 1.142 FID、在 B 上值 1.054 FID**。
  去偏之后 Δ′ = **−0.2231921823**，CI [−0.3619676348, −0.1053314834]，**不含零**。
  如实的读法：刚体偏置支配了原始指标，并把估计量的离散度抬高到足以掩盖一个真实的残余差；
  **不是**"预算显著"——两格各减自己的向量，且 A′/B′ 按用户约束不作为正式成绩，
  也不得与 released 的 `0.9334244584430564` 并列。

### 这一格关掉什么，没关掉什么

**关掉的**：这一对上的指标列补齐了，且 null 禁止把这条列作为**预算主张**外推到剩余 32 个未测导出。
**没关掉的**：两条 live lead 都在本实验之外。其一是 A′/B′ 直接给出定价的**刚体竖直偏置**——
本次把它标价为 A 上 1.142、B 上 1.054 FID；其二是接触参与度。
按上文非声明 7，两格都在各自验证 argmin 之后（D2-X 24,576,000；D2-AI 27,648,000），
本实验**不授权**用验证损失做早停、checkpoint 选择或预算决定，无论落在哪一支。
