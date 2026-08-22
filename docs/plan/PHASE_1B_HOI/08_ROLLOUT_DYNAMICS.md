# Phase 1B-08：自回归 rollout 动力学

导航：[总览](../OVERVIEW.md) · [Phase 1B 索引](README.md)

本文件与 01–07 同格式：**预注册与逐次 amendment 的原文逐字节副本**，不是结论摘要。
新开一册而不追加进 `07_REPRESENTATION_FRAME.md`，理由是 07 已于 2026-08-20 结案
（分类 `eval-consistency-null`），把一个在跑的实验追加进已结案文件会重开它的记录。

---

## P14 教师强制诊断：窗口 2、3 的历史来源（2026-08-21 预注册，用户批准）

### 这一节要回答的问题

HOIPrior 的 3 窗口自回归 rollout 会**丢掉被试的绝对身高**。把每条序列的模型骨盆高度
对真值骨盆高度做跨序列最小二乘回归，斜率 b 逐窗口衰减：**第一个生成帧 1.025 → 窗口 1
0.899 → 窗口 2 0.505 → 窗口 3 0.186**。到窗口 3，模型基本不再知道自己在生成哪个被试。

窗口 1 的历史是数据集真值（`fixed_points` 由 `points_orig[:, :2]` 构造），窗口 2、3 的
历史是模型自己上一窗口的输出。**唯一被操纵因子：窗口 2、3 的历史来源（自身输出 → 真值）。**
其余一切（checkpoint、438 条序列、seed、噪声、引导设置、评测路径）不动。

### 命名：本节用 cell G / cell N，不用 PRIMARY / CONTROL

P12 自己的 PRIMARY / CONTROL 是**step-0 帧规则**对照，且**两个 cell 都是无引导的**
（registry 第 280 行口径 amendment）。本节的对照轴是引导开/关，与之正交。为免读者把两套
标签混起来，本节固定用：

| 本节标签 | 含义 | run id | registry 行 |
|---|---|---|--:|
| **cell G** | P12 + Arm B，**有引导**（用户指定的主 cell） | `p1-hoi-p12-guidance-armb-s42-20260820` | 283 |
| **cell N** | P12，**无引导**（用户指定的对照） | `p1-hoi-p12-frame-repair-baseline-eval-guided-r1-s42-20260820` | 279 `primary_eval_run_id` |

`p1-hoi-p12-frame-repair-baseline-eval-guided-control-s42-20260820`（historical 帧规则，
registry 279 `control_eval_run_id`）**不是本节的任何一个 cell**。两个 cell 的 run id 里
的 `-eval-guided-` 是 `tools/hoi_chain.py:85` 的默认 tag，是个误名：cell N 无引导。

共用 checkpoint：`p1-hoi-p12-frame-repair-baseline-s42-20260819_windows299520000.pth`，
sha256 `722d83ee7755b051e2095ccd01d4094bacce99589e679f89379f54661fb43704`，
`checkpoint_weight_variant=online`（**必须显式给**；漏掉它是 registry 第 278 行失败的原因）。

### 携带进来的已测值，以及一处必须先更正的标签错误

下表由 2026-08-21 的只读 CPU 探针在六个封存 CHOIS 导出上复算得到（438 序列 × 126 帧，
pelvis = 关节 0，竖直 = 轴 2，Z-up）。**先更正**：`07_REPRESENTATION_FRAME.md:580-584`
与对应的记忆条目把 `1.0245 / 0.9851 / +2.874 cm / 0.1865` 标为「P12 + Arm B」，
**那是 cell N（无引导）的数**；并据此写下「引导把 frame-0 偏置从 +1.161 推到 +2.874、
同时把 spread 从 4.947 收到 1.569」——**那是帧规则对照，不是引导对照**。

**引导的真实效应很小且方向相反**：frame-0 偏置 +2.874 → **+2.746 cm**，sd 1.569 → 1.560
（P12）；D2-AI 上 +2.191 → +1.960 cm。引导**略微减小**该偏置。

| cell | frame0 b | w1 b | w2 b | w3 b | frame0 a | w3 a | frame0 r | w3 r | frame0 bias | w3 bias |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **cell G**（Arm B） | 1.0251 | **0.8994** | **0.5054** | **0.1855** | 0.0055 | 0.7544 | 0.9853 | 0.3524 | +2.746 cm | +3.673 cm |
| **cell N**（无引导） | 1.0245 | **0.9017** | **0.5092** | **0.1865** | 0.0073 | 0.7573 | 0.9851 | 0.3417 | +2.874 cm | +4.049 cm |

跨谱系不变性（六个导出）：frame-0 b ∈ [0.9981, 1.0251]，w3 b ∈ [0.1855, 0.2198]。
4.875× 预算、推理引导、2026-08-19 表示修复三个杠杆都不动它。

**关于先前被记为「已丢失」的两组数（2026-08-21 提交前更正）**：

- **方差收缩那六个数已被逐位复算**，由本次提升为 `tools/measure_hoi_vertical_tracking.py`
  的探针在 **cell N** 上给出（即上文标签缺陷所指的那个 cell），四位小数全中：
  模型 sd `w1 0.0725 → w2 0.0624 → w3 0.0409` m，真值 sd `0.0752 → 0.0867 → 0.0750` m，
  模型 spread 收窄 **43.6%**。cell G 同形：`0.0724 → 0.0619 → 0.0395`。
  **真值 spread 平到略升，模型 spread 收窄——漂移会加宽，收缩才会变窄。** 该证据有效，可引用。
- **仍未复现、且是另一个量**：「窗口 2 对自身历史 b=0.7467、窗口 3 b=0.4233」这两个斜率。
  探针不计算「对自身消耗的历史」这一回归，产生它们的脚本已丢失。
  **本预注册的任何阈值都不建立在这两个数之上**（ρ 的分母取同 run 的 b_normal(w1)，与它们无关）。

本节新增一个**secondary、无裁决权**的统计量以补上这个洞：**own-history 斜率**——
把窗口 s+1 的输出层回归到它实际消耗的那 2 帧历史的层上。稠密导出里，窗口 s 的关键帧 14、15
落在全局稠密索引 `42s+36` 与 `42s+39`（每窗口保留关键帧 2..15，关键帧 k → 窗口内偏移 3(k−2)）。
该定义是**本节新定的**，不声称复现上面那两个丢失的数；若结果落在 0.75/0.42 附近算佐证，
不落在也不构成矛盾。它直接度量假设 M（「模型对任何被给到的层都收缩」），
是 G4 之外唯一能触及 M 的通道。

### 衰减比几何衰减更快，这是本诊断的立论基础

若每窗口施加一个固定的线性收缩 c，则 b(w_k) = c^k。以 c = b(w1) = 0.8994 代入，
预测 w2 = 0.809、w3 = 0.728；**实测 0.505 与 0.186**。所以逐窗口的收缩**随深度加速**，
这是「条件输入逐窗口更加偏离训练分布」的签名，而不是一个常数收缩在复合。

独立佐证（同一探针的分被试拆分）：真值骨盆高度 sub16 0.892 m / sub17 0.855 m，
相差 3.7 cm；模型输出 0.913 / 0.903，相差 1.0 cm。**模型把被试间差异压到真值的 26%。**

### 三个假设，以及教师强制能与不能区分哪一对

- **假设 E（exposure bias / 自条件）**：收缩来自「模型自己的输出作为条件输入是分布外的」。
  喂真值历史 → 各窗口应回到窗口 1 的水平。
- **假设 M（窗口机制内在收缩）**：模型对**任何**它被给到的绝对高度都收缩。
  喂真值历史 → b 不回升，只是停止复合。
- **假设 F（帧/条件构造差异）**：步 0 的条件经**数据集**路径构造，步 ≥1 经
  `window_codec.encode` 路径构造；两条路径若不等价，差异本身即可造成逐窗口收缩。

**教师强制区分 E 与 {M, F}，不区分 M 与 F。** 后者由下面的闸门 G4 在 CPU 上免费分离。

### 已经靠读代码排除的三个混淆（不必花 GPU）

1. **竖直原点约定不是混淆。** `code/priors/core/window_codec.py:132-134` 的
   `frame_from_global` 置 `origin[..., 1] = 0.0`；`code/datasets/infbagel.py:461` 的
   `init_joints = np.array([joints[0,0,0], 0., joints[0,0,2]])` 同样把 Y 置 0。
   **两条路径都保留绝对竖直通道**，所以步 0 与步 ≥1 在本诊断关心的那个通道上约定一致。
2. **归一化盒不可能分裂。** `code/test_infbagel_hoi.py:580-582` 用数据集自己的
   `min_torch/max_torch/obj_min_torch/obj_max_torch` 构造 codec，与 HSI 侧
   `lingo_only` 那次的盒分裂机制在此不成立。
3. **BPS 重算不消耗随机性。** `recompute_rollout_bps`（`:56-69`）走
   sha256 校验过的固定 basis `code/bps.pt`，按 `sequence_names` 分组分块，
   从不依赖生成值。

### 「同种子、同噪声」是被测量出来的，不是被论证出来的

`code/priors/hoi/diffusion.py:568-570`：每次 `p_sample_loop` 新建一个**私有** generator，
种子为 `(torch.initial_seed() + sample_calls * 1000003) % (2**63-1)`。`torch.initial_seed()`
不随消耗改变，所以噪声流是 `(cfg.seed, 调用序号)` 的纯函数，**与全局 RNG 状态无关**，
因此与教师强制对条件值做的任何事都无关。每窗口 500 次抽样（初始 `x_T` 一次 + 499 步后验
噪声，最后一步为 `zeros_like`），每次 rollout 1500 次。Arm B 引导零抽样，
仅在 `0 < reverse_step < last_steps` 命中，每窗口恰 9 次、每 rollout 27 次，与数据无关。
模型 dropout p=0.1 在 `.eval()` 下失效。

该性质**条件于四件教师强制必须不改的事**，逐条写成闸门（见 G5–G7）：
(C1) batch 大小不变——不得因真值历史短而丢序列；
(C2) `p_sample_loop` 调用次数仍为 3；
(C3) 不新增 `torch.manual_seed`；
(C4) `cfg.seed` 与 `timesteps=500` 不变。

### 报告哪五个统计量，怎么算（写死，不许看到结果后改）

输入只需要 CHOIS npz 导出（`save_chois_eval_npz: true` 已是默认）：
`chois/predictions/*.npz` 与 `chois/ground_truth/*.npz`，键 `seq_name` 与
`global_jpos` `(126,24,3)` float32 **Z-up**，pelvis = 关节 **0**，竖直 = 轴 **2**
（`code/test_infbagel_hoi.py:215,220` 经 `yup_to_zup`）。126 = 3×(16−2)×3，
2 帧历史不导出，所以索引 0 **就是**第一个生成帧。

对每条序列，令 P = 模型骨盆高度在某跨度上的均值，G = 真值同跨度均值。跨度七个：
`frame0`=[0:1]、`w1`=[0:42]、`w2`=[42:84]、`w3`=[84:126]，另加每窗口的首个生成帧
`f0`=[0:1]、`f42`=[42:43]、`f84`=[84:85]。跨 N=438 条序列计算：

| 统计量 | 定义 |
|---|---|
| **跟踪斜率 b** | P 对 G 的一元最小二乘斜率 |
| **截距 a** | 同一拟合的截距（单位 m） |
| **相关 r** | Pearson 相关 |
| **跨序列标准差** | `sd(P)` 与 `sd(G)` **分别**报告 |
| **竖直偏置** | `mean(P − G)`，报为 cm |

**PRIMARY 估计量是窗口均值**（与携带进来的探针同口径，保证可比）；首帧版本是 secondary，
无裁决权。

### 判定阈值（**先于结果固定**，这是本预注册的核心）

定义**恢复分数**

> ρ(w) = ( b_TF(w) − b_normal(w) ) / ( b_ref − b_normal(w) )，w ∈ {2, 3}

其中 **b_ref := 同一 cell、同一 run 的 b_normal(w1)**。取 w1 而不是取 1.0 或取 frame-0 的
1.025 作分母，因为 w1 是**已经吃真值历史**的那个窗口，且用的是同一个窗口均值估计量——
它是「喂真值历史能达到的水平」的同口径经验上界。ρ=1 表示完全回到 w1 水平，ρ=0 表示没动。

`b_TF(w1) ≡ b_normal(w1)`（教师强制不碰步 0），由闸门 G2 逐位保证，所以 ρ 的分母在两臂
之间共享、配对良好。

以封存值代入，**冻结如下绝对带**（若 G1 逐位通过，这些带按构造不变）：

| cell | w | b_normal | b_ref (w1) | 判 **RECOVERED** 需 b_TF ≥ | 判 **NOT RECOVERED** 需 b_TF ≤ |
|---|---|--:|--:|--:|--:|
| G | 2 | 0.5054 | 0.8994 | **0.7812** | **0.6236** |
| G | 3 | 0.1855 | 0.8994 | **0.6852** | **0.3997** |
| N | 2 | 0.5092 | 0.9017 | **0.7840** | **0.6270** |
| N | 3 | 0.1865 | 0.9017 | **0.6871** | **0.4011** |

裁决规则，对 **cell G**（主 cell）执行；cell N 独立执行一遍作复现检查：

- **RECOVERED ⇒ 假设 E（exposure bias / 自条件）**：ρ(w2) ≥ 0.70 **且** ρ(w3) ≥ 0.70，
  **且** Δb(w) = b_TF − b_normal 的配对 bootstrap 95% CI 在两个窗口都不含 0。
- **NOT RECOVERED ⇒ 假设 M 或 F（窗口机制）**：ρ(w2) ≤ 0.30 **且** ρ(w3) ≤ 0.30。
  不要求 CI（这是零方向），但仍报 CI。
- **PARTIAL**：其余一切，含两个窗口落入不同类。此时结论就是 `(ρ(w2), ρ(w3))` 与其 CI，
  **不许宣称哪个机制占主导**；唯一例外是一个窗口 ≥0.70 而另一个 ≤0.30，
  那结论报为「深度依赖」。

**为什么是 0.70/0.30 而不是在 0.5 处一刀切**：0.5 单点会强迫一个连续量给出二元结论，
让噪声决定分类。0.40 宽的不确定带保留了 PARTIAL 这个**如果两种机制都在起作用就应当给出的
答案**。带宽先于结果固定，且两条边界相距 ≥0.157（w2）与 ≥0.285（w3）个 b 单位，
远大于 bootstrap 尺度。

**显著性**：Δb 的配对 bootstrap——一个共享的重抽样索引矩阵同时施加到两臂，
2000 次复制，seed 42，2.5/97.5 百分位线性插值，序列按身份配对，两臂身份集合不同即拒绝运行。
与 `tools/run_chois_evaluator.py` 的配对 bootstrap 纪律同源。

### 事前预测（写下来供事后打分）

**我预测 RECOVERED，或偏 RECOVERED 的 PARTIAL。** 理由：b(w1)=0.90 已经接近 1，
而 w1 正是吃真值历史的窗口；而自由 rollout 的衰减**快于几何**，这是复合式偏离的签名，
而教师强制按构造消除复合。若假设 M 为真（对任何历史都收缩固定比例），
则 b(w1) 本身就应当是那个收缩比，w2、w3 在教师强制下应停在 0.90 附近——注意这与
RECOVERED 的预测**数值上重合**，所以 M 与 E 在 b 上并不总能分开；真正把它们分开的是
G4（编码等价）加上 ρ 是否显著小于 1。

这个预测与我上一轮基于「窗口 2 对自身历史 b=0.7467」给出的「偏 NOT RECOVERED」相反。
改口的原因是那个数已被判定为不可复算并作废，不是因为看到了任何新结果。

### 闸门（全部在本次 run 内执行，不许引用历史通过记录）

| 闸门 | 要求 | 失败后果 |
|---|---|---|
| **G1** | 两个 cell 的**正常** rollout 复现封存结果。首选逐位：`evaluation/per_sequence_metrics.json` 与 sealed 同 sha256（cell G `d55f6b74bc02…`、cell N `b176e061184a…`）。若不逐位，18 项聚合指标每一项相对偏差 ≤1e-6 且必须解释原因 | 逐位失败且超容差 → `tf-reproduce-fail-stop`，**不解读任何 TF 结果** |
| **G2** | TF run 的**窗口 1** 输出与同 cell 正常 run 逐位相同（TF 只碰步 1、2）。在 CHOIS npz 上比较帧 [0:42] | 失败 → 实现污染了步 0，停 |
| **G3** | GT 历史行索引 `seq*max_len+step` 正确：相邻窗口的真值重叠必须成立（窗口 s 的前 2 帧 = 窗口 s−1 的后 2 帧）。已在 6 序列 × 2 边界的真实数据上实测最大偏差 4.768e-7 m，闸门容差 `atol=1e-5` | 失败 → off-by-`auto_regre_num`，停 |
| **G4** | **CPU、无 checkpoint**：对同一窗口的同一真值历史，比较 `window_codec.encode` 与数据集路径（`normalize_torch(transform_points(·, inv(mat)))` + `global_rot_6d`）产出的 `fixed_points`。**这一条分离假设 M 与 F** | 不等价 → 差异本身是结论，TF 结果标注为「受 F 混淆」 |
| **G5** | 两臂 `sample_count` = 438、`windows_per_sample` = 3、`is_timing_subset` = false，batch 大小未变（条件 C1/C2） | 失败 → 噪声流不可比，停 |
| **G6** | 两臂 `normalization_audit` 的 `guidance_sample_calls` = 3、`guidance_applied_steps` = 27（仅 cell G 有此字段）；且 `sampler_body` **确有** `reset_sampling_audit`——`code/test_infbagel_hoi.py:889` 的 `hasattr` 守卫若静默 no-op，`sample_calls` 会从 1 进入、三个窗口拿到全不同的种子，run 与封存结果不再可比。这与 2026-08-16 那次 mix dispatch 缺陷同一类 | 失败 → 停 |
| **G7** | 四个 rollout 同主机同驱动（node01，driver 580.126.09）。CHOIS 路径已知跨主机只到 1.47e-08（证据索引 583-588） | 跨主机 → 结果不得用于逐位比较 |

### 运行方案与 GPU 预算

入口 `code/test_infbagel_hoi.py --config-name=config_eval_hoi_prior`，cwd `$ROOT_DIR/code`，
node01，1 GPU，`batch_size: 1`，`num_gpus: 1`。共用覆盖：

```
exp_name=<RUN_ID>
ckpt_path=$ROOT_DIR/results/experiments/p1-hoi-p12-frame-repair-baseline-s42-20260819/checkpoints/p1-hoi-p12-frame-repair-baseline-s42-20260819_windows299520000.pth
checkpoint_weight_variant=online
```

`seed=42`、`save_chois_eval_npz=true` 与三个输出目录都已在 `config_eval_hoi_prior.yaml`
里按 `${exp_name}` 设好，**不需要也不要覆盖**。step-0 帧规则用默认 `repaired`，不覆盖。

| # | cell | 臂 | 追加覆盖 | 拟 run id |
|--:|---|---|---|---|
| 1 | G | 正常 | `sampler.pelvis.guidance.enabled=true arm=b guidance_scale=1000.0 last_steps=10 clamp=1.0 clamp_target=update` | `p1-hoi-p14-tf-cellg-normal-s42-20260821` |
| 2 | G | TF-full | 同上 + `teacher_forcing_history=full hoi_diagnostic_not_a_model_score=true` | `p1-hoi-p14-tf-cellg-tffull-s42-20260821` |
| 3 | N | 正常 | 无 | `p1-hoi-p14-tf-celln-normal-s42-20260821` |
| 4 | N | TF-full | `teacher_forcing_history=full hoi_diagnostic_not_a_model_score=true` | `p1-hoi-p14-tf-celln-tffull-s42-20260821` |

（所有 `sampler.pelvis.guidance.` 覆盖均带该前缀，表内为省略写法。）

**GPU 预算。** 封存实测：cell G `generation_seconds` 61.854 / `end_to_end_seconds` 197.909；
cell N 60.249 / 231.329。外部墙钟 cell N 236 s。故

| 项 | 估计 |
|---|--:|
| 单 rollout 端到端 | 198–236 s |
| 4 个 rollout | **13.2–15.7 min** |
| 预算上限（含一次重跑余量） | **≤ 25 min，1 GPU** |
| CPU 侧探针（G4 + 五统计量 + bootstrap） | 秒级，无 GPU |

不占 worker，不与 HSIPrior 训练争 GPU。**不训练、不重定价 root-y、不加 betas/rest offsets 条件。**

### 可选臂 C（用户可直接删掉，不影响 PRIMARY）

`teacher_forcing_history=vertical`：只替换历史关节的竖直分量，其余通道仍用模型自己的输出。
它分离「竖直收缩由竖直历史驱动」与「由整体条件偏离驱动」。追加 2 个 rollout，
**+6.6–7.9 min**（总计 ≤35 min）。**exploratory，无裁决权**，不得改变 PRIMARY 的判定。

### 教师强制结果**不是**模型成绩（硬约束）

真值历史在推理时不存在。教师强制臂与 P6 cell U 的 GT-mask 上界探针同类，
先例见 registry 第 264 行 `results.cell_u_outcome.deployable: false`。

1. **TF 臂的 18 项指标不得写进 `/data/yujinlun/report/baseline.md`、证据索引头表，
   或任何模型对比表。** 它们会「变好」，而且部分是因为窗口 2、3 被锚在真值位置上——
   `mpjpe` / `trans_dist` / `xy_points_err` 尤其如此。这是构造使然，不是模型能力。
2. registry 行携带 `config.deployable: false` 与
   `config.kind: "inference-only teacher-forcing history diagnostic; ground truth at inference; no training, no checkpoint change, no evaluator metric change"`。
3. 代码侧强制：`teacher_forcing_history` 非 `off` 时必须同时设
   `hoi_diagnostic_not_a_model_score: true`，否则 fail closed；且该模式会被戳进
   `aggregate_metrics.json` 与 `per_sequence_metrics.json` 的顶层。
4. **默认路径的两个 JSON 不得新增键**，否则 `per_sequence_metrics.json` 的逐字节可复现
   不变量（证据索引 462 行：三次 Arm-B 评测跨两主机逐字节一致）会被我们自己破坏，
   而 G1 正要用它。故戳记只在非 `off` 分支出现；**无该键即等于 off**。

### 停止分类

- `tf-exposure-bias-confirmed`：G1–G7 通过，cell G 判 RECOVERED。结论：逐窗口收缩是
  exposure bias / 自条件，下一步方向是让模型在自己的 rollout 分布上训练
  （scheduled sampling / rollout 微调），且该方向**首次**有实测支持。
- `tf-window-mechanics-confirmed`：G1–G7 通过，cell G 判 NOT RECOVERED，且 G4 等价。
  结论：收缩内在于窗口机制，喂真值历史不救。exposure-bias 类方案**被排除**，
  下一步转向条件通路本身。
- `tf-partial`：判 PARTIAL。结论即 `(ρ(w2), ρ(w3))` 与 CI，两种机制都记为有贡献，
  不排除任何一方。
- `tf-encode-path-defect`：G4 不等价。**这条优先于上面三条**——两条编码路径不等价本身
  是一个可修的缺陷，且它会混淆 TF 的读法。此时 TF 结果标注「受 F 混淆」并保留，
  先修等价性。
- `tf-reproduce-fail-stop`：G1 失败。不解读任何 TF 结果，回到评测路径排查。
- `tf-noise-stream-broken`：G5 或 G6 失败。「同噪声」不成立，两臂不可比，停。

### 本节**不**建立什么

1. **不给出任何模型成绩。** 见上一节。
2. **只覆盖 3 个窗口。** 对 HOSI 要链的几十个窗口，本节给的是外推的起点，不是结论。
3. **不区分假设 M 与假设 F**，除 G4 之外。G4 只比较条件张量，不比较模型对它们的响应。
4. **不解释 frame-0 的 +2.75/+2.87 cm 偏置。** 那是另一个缺陷；窗口 1 不被教师强制，
   所以本节对它零信息。若 TF 下 w2/w3 的偏置回到 frame-0 水平，那只是说偏置是每窗口
   一次性的常量，仍不解释它为何存在。
5. **不解释 CHOIS 侧亏空**（FID 2.307 vs 0.933）。表示往返对 FID 只值 0.0014。
6. **不改** `code/eval_metrics.py`、不改 438 分母、不改穿透六类排除、不改
   `code/priors/core/`、不改 `recipe/d2ai.yaml`、不为让检查通过而改测试或 validator。
7. **对 `feet_height` / `foot_sliding` 不是上界**：地板高度由 DBSCAN 从运动自身估，
   模型可以合法地比真值分数更低。

### 需要的源码改动（已实现，待用户审查后提交）

| 文件 | 改动 |
|---|---|
| `code/config/config_eval_hoi_prior.yaml` | 新增 `teacher_forcing_history: off` 与 `hoi_diagnostic_not_a_model_score: false` |
| `code/test_infbagel_hoi.py` | 世界系真值旋转累加器（**默认关闭**）、步 ≥1 的替换调用点、五个辅助函数、两个 JSON 的条件戳记 |
| `tests/test_research_governance.py` | 两个新测试类 |
| `tests/hoi/test_hoi_evaluation_provenance.py` | 默认键集合改为「恰好四键」，新增戳记必须在 off 守卫之后 |
| `tools/measure_hoi_vertical_tracking.py` | **新增**（先例：`tools/measure_hoi_repr_ceiling.py`、`tools/run_chois_evaluator.py`）。CPU-only、无 checkpoint、拒绝覆盖输出 |

`code/priors/core/` 未改。默认路径不新增张量分配、不新增 RNG 抽样、不新增 JSON 键。

### 已知局限

1. **教师强制路径没有在真实 checkpoint 上跑过。** 预注册阶段禁止 GPU，故替换逻辑只在
   合成张量与真实真值数组上验证过，从未经 `window_codec.encode` 进到活模型。
   第一次真实执行就是 run #2。
2. **闸门是 `ValueError`，rollout 中途抛。** 非 off 的 run 可能在协议深处才死。
   对诊断是正确取舍（宁可死不要静默出错数），但要预期。
3. **`vertical` 模式替换关节轴 1（y-up，rollout 系），探针读轴 2（z-up，导出系）。**
   各自在自己的系里都对，但把一个抄进另一个是个坑。
4. **G4 是 CPU 上的张量比较**，它不能证明模型对两种编码的**响应**相同。
5. **本节不复算携带进来的六个导出的数**，只引用 2026-08-21 探针的复算结果；
   该探针将随本次改动被提升为 `tools/measure_hoi_vertical_tracking.py`。

---

## 预注册内修订（2026-08-21，提交前）：G4 已先行执行，并因此改了三处

G4 是 CPU-only、无 checkpoint、只读的条件张量比较，**在提交本预注册之前就跑完了全协议**
（438 序列 × 步 {1,2} = 876 次比较，14.205 s）。这么做的理由是：它的结果决定本实验怎么读，
所以它应当先于批准，而不是先于结果。它**没有消耗 GPU**，四个 rollout 一个都没启动。

### 修订 1：G4 判 EQUIVALENT，假设 F 在几何通道上被排除

工件 `.claude/scratch/tf_prereg/g4_repaired_all438_v2.json`（默认 `repaired` 帧规则，
即两个 cell 都用的那个）：

| 通道块 | 最大绝对偏差 | 判定 |
|---|--:|---|
| joints `0:84` | 2.384e-07（7.94e-07 m） | EQUIVALENT |
| joints，**仅竖直分量** | **0.000e+00** | **EQUIVALENT（逐位）** |
| human rot 6-D `84:216` | 4.172e-07 | EQUIVALENT |
| object trans `216:219` | 1.788e-07（5.47e-07 m） | EQUIVALENT |
| object rot `219:228` | 5.960e-07 | EQUIVALENT |
| 派生窗口帧 `mat[:3,:3]` vs `world_to_local^T` | 2.384e-07 / 测地 1.529e-05° | EQUIVALENT |
| 派生原点 `mat[:3,3]` vs `frame.origin` | **0.000e+00 m** | **EQUIVALENT（逐位）** |
| `obj_rot_mat_ref` vs `frame.object_reference` | 4.172e-07 / 测地 1.225e-05° | EQUIVALENT |

全部落在 4.768e-7 的 float32 参考之下。竖直通道与帧原点是**逐位相同**，不只是接近：
两条路径的帧都是绕 +y 的旋转、原点的 y 都为 0，所以 `local_y == world_y` 在两条路径上恒等。

**该 null 不是盲点**，两个注入的已知坏帧证明检查能失败：换回修复前的
`matrix_to_euler_angles(root,"ZXY")[...,2]` yaw 读法使帧偏 0.173°(p50)/7.096°(max)、
joints 块偏 7.66e-2 m；注入 1 cm 竖直原点偏移使竖直通道到 9.09e-3 归一化 / 1.00e-2 m。
即竖直通道的 0.0 是**测出来的零**，不是没测。

因此 **`tf-encode-path-defect` 从活分支降为已解决**：非恢复的读法可以落到模型身上。
一条重要的结构性局限随之写明：**G4 结构上无法侦测源于帧的竖直缺陷**——纯 yaw 帧不可能
产生竖直分量差异。

**同时确认一件与本设计无关但值得写下的事**：在 `step0_frame_rule=historical_conjugated`
下 876/876 判 NOT-EQUIVALENT（帧 yaw p50 72.17°、joints p50 0.505 m）。**本节两个 cell
都用默认 `repaired`**，不受影响；P12 那个 historical 帧规则 cell 不是本节的任何 cell。

### 修订 2：发现并修掉教师强制的接触通道缺陷；另有一处涉及已封存工作

`gt_contact_label_batch[seq]` 存的是**一个 16 帧窗口**
（`code/datasets/infbagel.py:628-629` 按窗口切片），而 `_gt_contact_window`
（`code/test_infbagel_hoi.py:371-393`）假定它跨整条序列、按 stride 14 索引。于是在 step 2，
`start = 28 >= 16` 落进短序列分支，返回**窗口 0 的最后一帧重复 16 次**。

独立复核（合成 16 帧、逐帧打标）：

| step | 掩码里的相异帧数 | 前两帧 | 应为 |
|--:|--:|---|---|
| 0 | 16 | 0, 1 | 0, 1 ✓ |
| 1 | 2 | 14, 15 | 14, 15 ✓ |
| 2 | **1** | **15, 15** | **28, 29 ✗** |

**对本诊断**：全协议实测 **128/438 序列（29.2%）在 step 2 接触位整位翻转**。已修：
新增 `contact_all_gt` 逐窗口累加器（行索引与其余四通道同为 `seq*max_len+step`），
直接读每个窗口自己的 `data_dict['contact_label']`，按构造精确；
**不动 `_gt_contact_window`、不动 `gt_contact_label_batch`**，故 cell-U 路径逐位不变、
G1 仍然成立。新增两个回归测试钉住该缺陷与新来源。

**对已封存工作（本节不修，需单独治理动作）**：P6 cell-U 的 GT-mask 上界探针
（`code/test_infbagel_hoi.py:997-1001`）与本缺陷**共用来源与切片器**，且它用的是完整 16 帧
掩码，不是前 2 帧。按上表，它的窗口 2 只有 2/16 帧正确、窗口 3 是 0/16。
**即 cell-U 的「完美接触判断上界」是在 3 个窗口里有 2 个退化的掩码上测出来的。**
本预注册只陈述该发现与证据，不改它——改它会使 cell-U 路径与封存 P6 结果不再逐位一致。

### 修订 3：可选臂 C（`vertical`）撤回

G4 的第三条证伪测试显示该臂被混淆：`vertical` 只替换 y，帧原点的 xz 仍来自模型输出，
注入 20 cm/−15 cm 纯水平漂移后，人体 joints 块**不动**（2.38e-7，竖直仍为 0.0），
而**整个漂移完整出现在物体平移通道上**（0.250 m，恰等于漂移范数）。
即该臂不隔离竖直通道，它把累积的水平漂移转成条件里的人-物相对位置误差，
同时让人体块看起来像干净真值。**撤回，不列入运行方案。** 若日后要隔离竖直通道，
必须同时冻结原点的 xz，那是另一个设计。

### 修订 4：仍未覆盖的一条 step-0 vs step-N 条件分歧

`obj_bps_data`：步 0 载入数据集存好的 BPS，步 ≥1 经 `recompute_rollout_bps`
（`:56-69`）→ `WindowStateCodec.recompute_bps`（对 rest mesh 做 knn）重算。
G4 不覆盖它（它不在 `[1,2,232]` 张量内，且需要物体网格才能测）。
同理未覆盖 `object_goal_batch` / `pelvis_goal_batch` / `object_points_batch` 在步 ≥1 的重表达。

**这不是教师强制引入的混淆**——它在正常 rollout 里同样存在，而教师强制只会让 BPS 的
物体参考更接近真值。但它是假设 F 剩下的唯一未测通道，故写入「本节不建立什么」：
**G4 通过不等于步 0 与步 ≥1 的全部条件构造等价，只等于那 232 维张量与其派生帧等价。**

### 修订 5：结论的解释范围被收紧（2026-08-21，用户裁决）

本诊断替换的是**全部五个历史通道**（关节、人体旋转、物体平移、物体旋转、接触），
不是竖直通道。因此：

> **本节唯一被授权的结论形式是「完整 GT 历史能否恢复身高保持能力」。**

**明确禁止的推断**：把 RECOVERED 归因于 root-y 这一个通道。若判 RECOVERED，
被恢复的是「模型在拿到一份完整的、在分布内的历史时保持被试身高的能力」，
而该历史里同时有正确的水平位置、朝向、物体位姿与接触。任何「所以问题在 root-y」的说法
都需要一个**只冻结竖直通道**的实验来支持，而那个实验（原可选臂 C）已被 G4 的漂移注入测试
证伪并撤回——它会把水平漂移转成物体平移误差，做不到隔离。

同理禁止的还有：把 RECOVERED 读成「重定价 root-y 就能修」。D0 已判该重定价为 abort，
本节不提供任何重新打开它的证据。该 abort 的理由见追溯性证据记录
`docs/phase_summaries/PHASE_1B_D0_ROOT_HEIGHT_PRICING_PREDIAGNOSIS.md`
（**不是预注册实验**：无 run id、无 registry 行、纯 CPU 只读，2026-08-22 补写并逐项复算）。

### 修订 6：W3 训练在 corrected P6 完成前不启动（2026-08-21，用户裁决）

P6 cell-U 的退化接触掩码（修订 2）作为**独立治理更正**处理：旧结果保留、追加缺陷披露、
另写 corrected cell-U 预注册与 diff、经用户审查后重测、**不覆盖旧结果**。
在 corrected P6 完成之前，**W3 几何项重测训练不启动**——W3 的读法依赖接触参与度的上界，
而那个上界当前是在 2/3 个窗口退化的掩码上测出来的。

> **2026-08-22 更新（用户裁决）：本条阻塞已解除，但 W3 仍不启动。**
> corrected P6 已于 2026-08-22 完成（`cellu-null`），故本条的条件已满足。
> 但**该 null 不构成 W3 的正面证据**：它界定的是当前模型上推理期掩码的上界，
> 对训练侧几何项零信息。现行的**新**阻塞是：在**高度目标 Stage A 审查完成前不启动 W3**。
> P14 紧凑结果 `p1_hoi_p14_teacher_forcing_s42_20260821.json` 的 `next_action` 里那句
> 「W3 geometry-term training stays on hold until the corrected P6 cell-U re-measurement lands」，
> 其条件已被满足，并被本条取代；新条件记录在
> `p1_hoi_p6_cellu_corrected_mask_s42_20260822.json` 的 `w3_dependency_status_20260822`。
>
> **2026-08-22 当日再更新：新条件也已解除。** 高度目标 Stage A 审查完成并**中止**
> （`height-target-producer-accuracy-negative-stop`），故按用户裁决 W3 回到排序。
> 解除记录在同一文件的 `w3_hold_resolution_20260822`；理由见
> `docs/phase_summaries/PHASE_1B_HEIGHT_TARGET_PRECHECK.md`。**W3 仍未获批准。**

### 修订 7：新增 λ 判别式，并披露 ρ 带在 w2 上的一个缺陷（2026-08-21，提交前）

修订 2 新定义的 own-history 斜率现在算得出来了。在**两个正常臂**的封存导出上实测
（**此刻不存在任何 TF 数据**，所以下面的一切仍然是事前的）：

| cell | w | b_own（对自身消耗历史） | b_normal（对 GT） | b_ref = b(w1) |
|---|---|--:|--:|--:|
| G | w2 | **0.7041** | 0.5054 | 0.8994 |
| G | w3 | **0.3913** | 0.1855 | 0.8994 |
| N | w2 | **0.7041** | 0.5092 | 0.9017 |
| N | w3 | **0.3968** | 0.1865 | 0.9017 |

这给出两个**点预测**：假设 M（响应斜率与历史是否在分布内无关）预测 `b_TF ≈ b_own`；
假设 E 预测 `b_TF ≈ b_ref`。

**必须披露的缺陷**：把 b_own 代进 ρ，纯 M 预测 **ρ(w2) = 0.497**、**ρ(w3) = 0.294**。
即在 w2 上，NOT-RECOVERED 带（ρ ≤ 0.30）**即使在纯 M 下也不可达**——w2 只可能给出
RECOVERED 或 PARTIAL。w3 的带是可达的（0.294 ≤ 0.30，勉强）。因此：

> **w3 是那个有判别力的窗口；w2 的 PARTIAL 不得被读成「不确定」。**

这个不对称是 b_normal(w2) = 0.509 低于 b_own = 0.704 造成的——自由 rollout 下窗口 2 对 GT
的跟踪比它对自身历史的响应更差，因为它消耗的那份历史本身已经退化了。所以纯 M 也预测
b 会在 TF 下**大幅上升**。原始的 0.70/0.30 带是在还没有 b_own 时定的，看不到这一点。

**新增 secondary 判别式 λ**（此刻固定，先于任何 TF 结果）：

> λ(w) = ( b_TF(w) − b_own(w) ) / ( b_ref − b_own(w) )

λ=0 ⟺ 模型对 GT 历史的响应与对自身输出的响应完全一样（纯 M）；
λ=1 ⟺ GT 历史把窗口 1 的行为完全恢复（纯 E）。

| cell | w | λ ≤ 0.25（M 主导）需 b_TF ≤ | λ ≥ 0.75（E 主导）需 b_TF ≥ |
|---|---|--:|--:|
| G | w2 | **0.7529** | **0.8506** |
| G | w3 | **0.5183** | **0.7723** |
| N | w2 | **0.7535** | **0.8523** |
| N | w3 | **0.5230** | **0.7754** |

**PRIMARY 仍然是 ρ 与它那四行带，裁决规则一字不改**——用户要回答的问题就是
「完整 GT 历史能否恢复身高保持能力」，ρ 正是它。λ 是 **secondary**，只用来把 PARTIAL
拆成 M 主导 / 混合 / E 主导，**不能推翻 PRIMARY 的分类**。

**为什么这不是「看到结果后改阈值」**：输入只有正常臂的封存导出，TF 一轮都没跑。
b_own 之所以现在才能算，是因为修订 2 才刚把这个量定义出来。

**最终事前预测（第三次，也是最后一次）**：**PARTIAL，且 λ 落在 M 主导区**。
点预测 b_TF(w2) ≈ 0.70–0.75、b_TF(w3) ≈ 0.40–0.52。理由：0.7041 与 0.3968 是模型对
「它实际消耗的那份历史」的响应斜率；若该响应是条件机制的性质而非历史分布外的后果，
喂 GT 历史不会改变它。反向考虑同样成立、且正是本实验存在的理由：那两个斜率是在自由
rollout 下测的，被消耗的历史本身已偏离分布，所以它们测的是「对分布外历史的响应」；
GT 历史在分布内，响应可能更高。E 与 M 的差就在这里。

**我在本次会话里改过三次这个预测，三次都留在记录里、不删**：先偏 NOT（基于 0.7467）→
改为偏 RECOVERED（因当时判 0.7467 不可复算）→ 现在回到 PARTIAL / M 主导
（因 0.7041 是干净重新定义并实测的）。第二次改口是被一个错误的「已丢失」判断驱动的，
那个判断本身也是本预注册的更正内容之一。
