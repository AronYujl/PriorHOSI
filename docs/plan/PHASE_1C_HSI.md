# Phase 1C：HSIPrior 从零训练与原生域评测

本文件于 2026-08-10 从 `docs/EXPERIMENT_PLAN.md` 第 8271-8286 行原样切出（逐字节复制，未改写、未重排、未修正任何笔误）。
导航：[总览](OVERVIEW.md)

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

---

## 2026-08-12：Phase 1C 入口修订（baseline 选定、split 重建、原生域 evaluator）

本节为用户批准的 Phase 1C 入口工作预注册。批准范围仅限**第 4 节的 split 重建**与
**第 5 节的几何 evaluator**（均为 CPU-only）；第 6 节的 evaluator 训练、InfBaGel
scene-only 采样路径与 baseline 评测在本节结束时另行申请。

### 1. 已确定的四项决定

| 项 | 决定 |
|---|---|
| HSIPrior 实现基座 | InfBaGel 的 `code/priors/` 脚手架 |
| baseline | **仅**使用已发布的 InfBaGel checkpoint，在 LINGO scene-only 上评测 |
| split | 修复 mirror 标签缺陷后重建为三分 scene-family-disjoint |
| FID/R-Precision | 自训练 LINGO text-motion evaluator，标注为项目内部指标 |

基座决定的依据：InfBaGel 是 LINGO 的 fork（`PositionalEncoding` 30/30 行、
`TimestepEmbedder` 17/17 行逐字保留，`Unet` 86.6%，`Sampler` 72.6%；两侧
`scene_embedding` 21,278,720、`transformer` 12,623,872 参数量完全相同），因此
「基于 InfBaGel」已经等于「基于 LINGO 的设计」。LINGO 自身的 denoiser 只输出 84 维
关节位置，132 维旋转由事后的冻结 MLP + 100 步 SMPL-X 拟合得到，无法参与 mixer 的
分身体组门控，故不作为基座。`code/models/infbagel.py` 的默认 `occ_temp`
（`:185-234`）在 `p_losses` 内用预测物体位姿与扰动 GT 关节构造未来 occupancy，正是
`OVERVIEW.md:8-18` 所拒斥的 motion-derived scene supervision，因此基座只取
`code/priors/`，不取 `code/models/infbagel.py`。

### 2. 证据：LINGO 发布数据存在 scene 标签缺陷

本次测量（全部可复算，见第 7 节验证命令）：

- `start_idx.npy` 共 19,450 条 sequence。令 `H = 9725`，则对全部 `i < H`，
  sequence `i+H` 是 sequence `i` 的**精确 x 镜像**：`transl_aligned` 的 x 逐位取负，
  y/z 逐位相同，长度全等。
- `scene_name.pkl` 中 sequence `9725..19449` 的**每一帧**都标为 `005_mirror`，
  而其真实房间共 110 个。首半段的 110 个标签中没有任何 `*_mirror`。
- 磁盘上 `Scene/` 共 254 个 occupancy grid，其中仅 111 个被标签引用，143 个从未被引用；
  110 个源 scene 的 `<src>_mirror.npy` **全部存在**，且逐位满足
  `grid(<src>_mirror) == grid(<src>)[::-1]`。
- LINGO 官方 loader 同样按 `scene_name[start_idx]` 取 scene
  （`lingo/code/datasets/lingo.py:186`），因此该缺陷属于**发布数据**，不是本仓库引入。

两项后果：

1. **条件错误**：任何按标签取 scene 的模型，会对约 1.12M / 2.28M 个窗口喂入
   `005_mirror` 的几何，而它们实际来自 110 个不同房间。
2. **既有 split 失效**：`005_mirror` 归入 family `005`，而 `005` 在 train 侧，故整个
   镜像半段都在 train。`experiments/splits/lingo_scene_disjoint_seed42.json` 的
   **1,895 / 1,895 条 validation sequence 的精确 x 镜像都在 train 侧**（234,381 帧）。
   该 split 按标签是 family-disjoint 的，按内容是完全重复的，因此它测不出泛化。

`exp/p1a-data-v1` 及其 sealed 数值保持不可变；本节不重写它，只声明其 validation
数值不可用于 Phase 1C 的泛化判断。

### 3. 证据：baseline 的 LINGO 曝露范围

已发布 InfBaGel 由 `config_train_infbagel_mix.yaml:15` 的 `lingo_scene_num: 45`
训练，选择规则为确定性的 `all_scenes[:45]`（`code/datasets/infbagel_mix.py:322`，
按首次出现顺序）。据此复算：

- 其 45 个 LINGO 训练 scene 为 `004`–`057` 区间的低编号房间；
- **`005_mirror` 不在其中**，故 baseline 从未在镜像半段上训练；
- 按 family 归并，76 个 family 中有 **44 个被其训练数据触及，32 个完全未触及**。

因此若在旧 validation 上评测 baseline，20 个 scene 中有 8 个
（`015 023 040 042 045 048-bed 051-bed 052-bed`）是它自己的训练 scene。

### 4. split v2 规格（本次批准实施）

`tools/make_lingo_split.py` 增加 `scene-family-disjoint-v2`，**加法式**扩展，v1 行为
与既有 manifest 的校验结果保持逐字不变。

1. **mirror 重标注**：由 `H = N // 2` 推出镜像半段，并在重标注前**逐条验证**
   `len(seq[i]) == len(seq[i+H])`、`transl.x` 取负、`y/z` 相等；任一条不成立则以
   `SplitError` 失败，不得静默降级。通过后令
   `scene(i+H) := scene(i) + "_mirror"`。重标注结果与验证统计写入 manifest。
2. **test 集**：由第 3 节的规则在工具内**复算** baseline 的 45 个 scene（不硬编码
   scene 列表），取其未触及的 32 个 family 为 test。此举使 baseline 与 HSIPrior 在
   test scene 上同为 zero-shot。
3. **train/validation**：在其余 44 个 family 上按 seed 42、ratio 0.20 划分。
4. **镜像归属**：train family 的镜像 sequence 归入 train；**validation/test family 的
   镜像 sequence 整条丢弃**，不得移入 train——移入即重建第 2 节的泄漏。
5. **不变量**：三个 partition 的 family、scene、sequence 两两不交；每个 family 恰好
   被指派一次；validation/test 只含非镜像 sequence。工具须自检并在违反时失败。

预期规模（非镜像、no-hand）：test 32 family / 37 scene / 3,392 sequence / 410,120 帧。
no-hand 过滤规则（`left_hand_inter_frame == -1 and right_hand_inter_frame == -1`）与
`seq_length <= 48` 的短序列排除沿用 Phase 1A，不在本次修改。

新 manifest 写入新路径，`lingo_scene_disjoint_seed42.json` 原样保留。
`tools/experiment.py` 的 `validate_split` 加法式接受 v2 与其 `test` partition。

### 5. 原生域 evaluator 规格（本次批准实施）

几何来源：`Scene/<scene>.npy` 布尔 occupancy，0.02 m 各向同性，世界 bbox
`[-3,0,-4]..[3,2,4]`，y-up，`True` 为占据；由 EDT 得到有符号距离场后三线性采样。
`/data/yujinlun/datasets/LINGO/Scene_mesh/<scene>/mesh_low.obj`（127 个真实三角网格，
已验证与运动共用世界系）用作独立交叉校验，不作为主口径。

指标分层（Tier 1–3 本次实施；Tier 4 见第 6 节）：

- **Tier 1 几何/物理**：人景穿透 `frame_ratio` / `mean` / `max`；**scene engagement**；
  foot sliding（复用 `code/eval_metrics.py` 的
  `compute_foot_sliding_for_smpl` 与 `determine_floor_height_and_contacts`，与 HOI 表同口径）；
  地面穿透与悬浮高度。
- **Tier 2 任务**：终帧 pelvis goal error（cm）、10 cm 完成率、到达时间。
- **Tier 3 rollout 连续性**：窗口边界的 root 位置/速度/加速度跳变、旋转 geodesic 跳变、jerk。

`HSIPRIOR_DESIGN_PRIORS.md:141-144` 要求的 engagement 量必须与每一个 penetration 与 FS
数字**同表呈现**，不得单独报告 penetration 改善。统计口径沿用
`tools/paired_bootstrap.py` 与 seed 42。

### 6. 本节未批准、需另行申请的部分

LINGO text-motion evaluator 的训练（仅用 train family，冻结后哈希登记于
`experiments/evaluators/`，标注为项目内部指标、不与已发表 FID/R-Precision 可比）；
InfBaGel scene-only 采样路径的修复；baseline 评测。Phase 1C 的 95% 门槛数值在
baseline 评测完成前不成立。

### 7. 治理

- 本节为 Phase 1C 的 dated plan 修订，配套 registry amendment row。
- `code/priors/core/` 不改动，因此不触发跨分支审批。
- `AGENTS.md:197-199` 要求的 seed-42 scene-family-disjoint 算法与「变体同 family 同侧」
  意图在 v2 中**得到恢复**而非放弃：v1 的意图被数据缺陷击穿，v2 修复标签后才真正成立。

---

## 2026-08-12（同日修订）：几何口径纠正与指标公式定版

本节修订上文第 5 节。修订发生在**任何模型测量之前**，依据是数据本身的性质与文献调研，
不是看到结果后的调整。上文第 5 节的几何主口径判断有误，此处纠正并保留原文以存证。

### A. 纠正：occupancy grid 不是几何，不能作为 penetration 的评分基准

上文第 5 节将 `Scene/<scene>.npy` 定为主口径、mesh 定为交叉校验。**这是反的。** 实测：

- scene `004` 的 occupancy 比例为 **0.5119**；按高度分层后，最高的一层是
  **y≈1.98 m 的天花板层，占据率 0.807**。实体家具不可能如此。
- LINGO 官方文档对该文件的措辞是「occupied by scene objects **or unreachable**」。
  因此该网格是**可达性/自由空间体**，不是实体几何。
- 后果：GT 关节落在「占据」体素内的比例聚合为 **7.1%**（分场景 1.37%–8.95%），
  另有 **4.3% 的 GT 关节完全落在世界 bbox 之外**。以该网格计的 penetration，其 GT
  参考值约为 0.07 而非 0，且一个远离所有表面悬浮的模型会**优于 GT**。

**纠正后的口径：**

| 用途 | 几何源 |
|---|---|
| penetration / contact 评分**主口径** | `Scene_mesh/<scene>/mesh_low.obj` |
| 模型的 scene 条件输入 | `Scene/<scene>.npy` occupancy（不变） |
| 次要诊断，须显式标注为「可达性违规」而非穿透 | `Scene/<scene>.npy` |

可行性已核实：110 个被引用的非镜像 scene **全部**有 mesh（缺失 0 个）；`004/mesh_low.obj`
为 watertight，1,012,799 顶点 / 2,025,986 面，bbox 落在网格 bbox 内且地面 y≈0。
`Scene_mesh` 不含 `_mirror`，但 v2 的 validation/test 只取非镜像 sequence，故评测集覆盖完整；
训练侧镜像不参与评分。逐场景 watertight 须在构建时断言，非 watertight 场景改用
generalized winding number 定符号并登记。

预计算 2 cm SDF 网格后三线性采样（DeSeG 在 LINGO 上的同分辨率先例）；分辨率、符号约定、
插值模式与 bbox 外规则必须与数值一同登记。bbox 外样本按「不裁剪、记为正距离」处理，
并单独报告越界比例。

### B. 为什么必须写死公式：同名指标不是同一个量

文献调研的首要结论：`Pene_mean`/`Pene_max` 至少是四个互不兼容的量共用一个符号。同一个
LINGO baseline 被不同论文报为 `Pene_mean` **0.402 / 0.421 / 0.392 / 1397**——DIMOS 官方
代码是「逐帧对顶点求和的深度积分」（随网格分辨率缩放，且 `Pene_max` 不是最深顶点），
TeSMo 是「仅穿透顶点的平均 SDF」，Dyn-HSI 是「穿透顶点计数」，PSI 是「非碰撞比例」。
因此本项目不写「following DIMOS/LINGO」，只写表达式、聚合顺序、单位与符号约定。

### C. 定版公式

采样体：**SMPL-X 顶点（10475）为主口径**，28 关节为快速诊断；两者不可互换，均已登记。
帧率 30 fps；地面 y = 0（LINGO 世界系中精确，无需估计）。

**穿透**（三个量同时报告，均需 **GT 参考行**——LINGO Tab.2 没有，我们加）
1. `pen_ratio` = SDF < −3 cm 的「顶点×帧」比例（TeSMo 阈值）。
2. `pen_depth_mean` = 仅对穿透顶点取 |SDF| 均值，单位 m（TeSMo）。`pen_depth_max` 同理取 max。
3. `pen_burst` = `100 × mean_t[(每帧穿透顶点比例)²]`（Dyn-HSI Eq. 9）。平方项是刻意的
   超线性，使一帧灾难性穿透不被长序列稀释——正对自回归 rollout 的突发型失败。

不采用 DIMOS 的逐顶点求和式。**scene 必须作为 paired bootstrap 的分层因子**：实测 GT
的逐帧穿透率在三个场景间为 30.8%–94.7%，聚合值更多由评测集含哪些场景决定，而非由模型决定。

**engagement**（与每一个 penetration 与 FS 数字同表，`HSIPRIOR_DESIGN_PRIORS.md:141-144`）
1. `contact_count` = 每帧落在表面 +5 cm 带内的顶点**数量均值**。**不得使用二值形式**：
   实测 GT 的「≥1 关节接近表面」在三个场景为 0.746 / 0.996 / 0.9996，已饱和无信号；
   计数形式跨场景为 1.64 / 3.46 / 2.72（28 关节口径），才有区分度。
2. `RDS`（FantasyHSI）= 同噪声、同 seed 下「给 scene 条件」与「不给 scene 条件」两次生成
   的逐关节平均距离。它在结构上免疫「靠远离场景刷低穿透」这一失效模式，并且正是
   design prior #6 要求的成对干预形式。

文献佐证该失效模式是实测而非臆测：SUMMON 的 *w/o contact loss* 消融拿到全表最好的
non-collision 0.995 而 contact 仅 0.194；MOVER 明确写出「悬浮的坐姿 non-collision 更好、
contact 更差」；HSI-GPT2 表中所有方法的 non-collision 挤在 99.69–99.82。**LINGO 谱系中
没有任何一篇在 penetration 旁报告 human-scene engagement**，故此列是可辩护的增强而非偏离。

**足部**（三者不是彼此的单调变换，同时报告）
1. `fs_nemf` — LINGO 引用的 NeMF FS：`s = v·(2 − 2^(h/H))`，H = 4 cm（趾）/ 8 cm（踝），
   位移取 **L1** 且**求和**不取平均，序列先平移使最低足高为 0，单位 cm/frame。
2. `skate_ratio` — GMD/TeSMo：足高 < 5 cm 且单帧滑动 > 2.5 cm 的帧比例。该阈值**不是
   帧率无关的**，30 fps 下须换算为 0.75 m/s 后使用。
3. 现有 `compute_foot_sliding_for_smpl`，仅为与 HOI 表同口径而保留。

**目标到达**
1. `last_dist` 与 `min_dist`：对关节取 min、仅水平 (xz)（DIMOS）。两者同时报告——它们在
   模型到达后又漂走时分离，而那正是自回归的真实失效模式。
2. 成功率同时按 **10 cm**（InfBaGel，即本项目 baseline 的口径）与 **20 cm**（DIMOS/LINGO 谱系）报告。
3. TeSMo 三分解：水平位置 / 朝向 / 高度分列，用以区分「到错地方」与「到了但朝向错」。
4. `time_to_goal` = 首次满足阈值的帧，按 30 fps 折算为秒。

**窗口接缝连续性**（本项目自定义：HSI 文献中无此指标，不得声称沿用）
1. `jerk_ratio` = 接缝处 jerk / 内部 jerk（SEAM 形式）。自归一化、不需 GT、且无法靠整体
   平滑作弊——分子分母会同时下降。
2. TEACH transition distance，对齐与未对齐两种都报。
3. **实现约束**：从第二个窗口起，重叠的 2 个 history frame 必须在计算任何指标前丢弃
   （DIMOS 的 `start_frame = 2`）；所有时序指标在**拼接后的整条序列**上计算，不得逐窗口计算
   后再平均。

**分布/语义指标**（仍属未批准范围，此处只定框架）
`R-Precision` Top-1/2/3、gallery 32 为主要语义数字——自训练 evaluator 下它是唯一可迁移的量，
因为它衡量同一嵌入空间内的相对排序；FID/Diversity/MultiModality 只在本表内部可比，须附
「Real motions」参考行，且 Diversity 与 MultiModality 标「→」而非「↑」。**不报告 MM-Dist**
（Voas et al. MIG 2023：与人类判断相关性一贯最弱，明确建议弃用）。

### D. watertight 审计结果与随之确定的报告规则

`Scene_mesh` 全量审计（127 个 mesh，逐场景表见 `.claude/scratch/watertight_audit.md`）：

- **113 个 watertight，14 个不是**；test 侧 37 个场景**全部有 mesh**，其中 **3 个非 watertight**
  （`085-play_game-chair`、`090-baseball_bat`、`091-take_shower`），validation 侧 0 个，train 侧 9 个。
- 这 3 个场景的符号由 generalized winding number 推出，与其余 34 个不同口径。

是否因此高估穿透？用 GT 关节实测对比（各取前 40 条 sequence）：

| test 场景 | watertight | GT pen<−3cm | 深度 mm |
|---|---|---|---|
| 085-play_game-chair | 否 | 0.0155 | 37.0 |
| 090-baseball_bat | 否 | 0.0084 | 58.6 |
| 091-take_shower | 否 | 0.0252 | 52.0 |
| 017-new_loco | 是 | 0.0022 | 45.4 |
| 058-loco | 是 | 0.0150 | 50.6 |
| 058-loco-1 | 是 | 0.0082 | 40.4 |

**结论：n=3 对 n=3 无法把差异归因于 fallback。** 两组区间显著重叠——watertight 的
`058-loco`（0.0150）高于 fallback 的 `090-baseball_bat`（0.0084）——且 watertight 组内部本身
跨 7 倍（0.0022–0.0150）。这与第 C 节已登记的场景间方差主导一致，不构成新问题。

由此固定两条报告规则：

1. 这 3 个场景在结果表中**必须标注符号推导方式不同**，不得与其余 34 个混为一个无注脚的均值。
2. 逐场景报告（第 C 节已要求，scene 作为 paired bootstrap 分层因子）本身已覆盖该风险；
   不额外引入 fallback 专用校正项。另注意 test 侧存在极小场景（如 `017-new_loco` 仅 2 条
   sequence），逐场景数字须与其 sequence 数同列。

SDF 磁盘缓存位于 `.cache/hsi_sdf/`，经 `.gitignore` 的 `*.npz` 排除，不会污染 worktree
（`tools/experiment.py` 会把未跟踪文件判为 dirty 而拒绝启动 reportable run）。

---

## 2026-08-12（同日第二次修订）：baseline 身份纠正与三模型口径

本节纠正上文**第 3 节**的事实错误，并据此修订第 4 节的 test 集规则。错误由用户指出：
作者 `readme.md:81` 明确写着已发布 checkpoint 只在 OMOMO 上训练，与第 3 节矛盾。核查结论是
**作者没写错，第 3 节错了**。上文原样保留以存证。

### A. 纠正：已发布 checkpoint 从未见过 LINGO

第 3 节称已发布 InfBaGel「由 `config_train_infbagel_mix.yaml:15` 的 `lingo_scene_num: 45`
训练」。该结论是从仓库里存在这份 config **推断**出来的，而不是从产出该 checkpoint 的证据得出。
`readme.md:133` 把 mix 明确写为一个可选项（`--config-name config_train_infbagel_mix`），
其权重从未发布。三项独立证据一致指向 OMOMO-only：

1. `readme.md:81`：`checkpoint/` — Consistency model checkpoints (**trained on OMOMO dataset only**)。
2. checkpoint 内含 `embedding_hand_goal.*`（4 个张量），**不含** `embedding_scene_goal.*`。
   `262f2d9` 中该模块名为 `GoalEncoder(mode='hand')`，`75efccc` 才改名为 `mode='scene'`；
   即该权重早于本仓库的改名，属作者原始产物。
3. 作者自己的采样默认值（`config_sample_infbagel.yaml`）为 `dataset: omomo_test`、
   `model: infbagel`（`is_mix: false`）、`sample_type: consistency`。

因此已发布 checkpoint 记为 **A：OMOMO-only 的 consistency model**（CM 属性依据作者文档与
`sample_type` 默认值，权重本身无法与 diffusion 区分，不声称已实测）。

### B. 随之失效的与随之保留的

- **失效**：第 3 节的 45 个 scene、「44 个 family 被触及 / 32 个未触及」、以及第 4 节据此
  选 test 集的规则。A 既然未见过 LINGO，则对全部 110 个 scene、76 个 family 均为 zero-shot，
  该规则没有换来任何东西。
- **保留且仍然关键**：第 2 节的 mirror 标签缺陷与其修复。那是发布数据的真实缺陷，与 baseline
  身份无关，v2 的价值几乎全部在此。
- **代价已量化**：v2 把 eligible window 切成 train 976,993 / validation 113,927 /
  test 367,443，另有 481,535 个 held-out family 的 mirror window 被整体丢弃（合计
  1,939,898）。train 只占 50.4%。由于 B/C 改为**仅在 LINGO 上训练**，该分配直接压低训练量。
- **因此重建 v3**：`scene-family-disjoint-v3` 去掉 baseline 规则，按 eligible window 数
  以 0.70 / 0.10 / 0.20 三分全部 76 个 family，family-disjoint，seed 42；mirror 规则不变
  （train family 的 mirror 归 train，held-out family 的 mirror 整体丢弃）；test 侧至少 25 个
  scene 以维持逐场景报告。v1/v2 的 manifest 与行为逐字节不变，v3 为加法式新增。
  第 5 节起的几何与指标口径不受影响。

### C. 加载与路由：两条硬约束

1. **A 必须做 key 重映射，且绝不可用 `strict=False` 兜过去。** 实测：A 直接 `strict=True`
   失败（4 missing / 4 unexpected）；把 `embedding_hand_goal.* → embedding_scene_goal.*`
   重映射后 `strict=True` 通过。两侧均为 `Linear(3,512) + Linear(512,512)`，形状完全一致。
   改名是纯粹的重命名：`code/datasets/infbagel.py:327-347` 的取值逻辑与原注释一并保留未改。
   若改用 `strict=False`，`embedding_scene_goal` 会**保持随机初始化**而预训练权重被丢弃
   （已实测该张量与初始化逐位相同），而 `load_scene_goal: true` 会让 goal 经一个随机投影进入
   模型，直接压低 baseline。B/C 原样 `strict=True` 通过。
2. **`is_mix` 不是装饰性开关。** `infbagel.py:1289-1291` 注明非 mix 模式下 scene-goal 条件
   不参与。A 训练于 `is_mix: false`，故作者默认采样配置对 A 是**正确**的；B/C 亦为
   `is_mix: false`。（本次会话早前一度声称「A 需 `is_mix: true`」，那是基于已被推翻的 mix 前提，
   此处一并纠正。）

### D. 三模型口径与 gate 归属

| | 身份 | 训练语料 | sampler | 角色 |
|---|---|---|---|---|
| A | 已发布 checkpoint | **OMOMO only** | consistency | ~~仅参考~~ → **已移出评测矩阵**（2026-08-15） |
| B | **已训练**（epoch222） | **LINGO only**（v3 train 侧） | diffusion | ~~**gate**~~ → **teacher / 蒸馏前对照**（2026-08-15） |
| C | 待训练，从 B 蒸馏 | 同 B | consistency | ~~蒸馏前后对比~~ → **gate（guided consistency 采样）**（2026-08-15） |

**本表的「角色」列已于 2026-08-15（同日第二次修订）改写**，理由与代价见该节 §A。原值保留
为删除线，不是覆盖。评测矩阵自该日起只含 **B 与 C** 两个模型。

现有 `p1b-author-{diffusion,cm}-8x3090-full-r1` 两个 checkpoint **不能**充当 B/C：其 config
的 dataset target 为 `datasets.infbagel.InfBaGelDataset`、folder 为 `data/train`、`lingo`
键数为 0；`data/train/Scene` 是 87 个 `occ_N.npy`，`data/dataset/Scene` 是 254 个
`<scene>.npy`，**交集为 0**。该 checkout 下 24 个 run 全部如此。故 B/C 需重新训练。

口径注意事项：

- A 与 B 同时差在**语料**与**模型类别**两个轴上，故「语料差异有多大」只能由 **A vs C**
  （同为 CM）回答；A vs B 不可归因。
- goal 槽位是双用途的：有手部交互帧时取手部关节，否则取**终帧 pelvis**
  （`code/datasets/infbagel.py:347`，原注释 "use hand goal to locate end pelvis goal"）。
  A 的语料 OMOMO 基本都有手部交互，而 LINGO no-hand 评测集恒为终帧 pelvis，故 A 在该槽位上
  额外承受一次分布偏移——这是 A 只作参考的又一条理由。B/C 训练与评测在该槽位上自洽。
- 由此 `last_dist` / `min_dist` / 成功率是**给定目标的可控性**指标，而非场景理解指标，
  且该目标来自 GT。三个模型同等获得该条件，比较是公平的，但该列必须如实标注。场景理解由
  penetration 与 `RDS` 承担。
- 归一化空间统一：`data/train/norm.npy` 与 `data/dataset/norm.npy` 逐字节相同
  （sha256 前缀 `61bb4b5f2ed5955f`），三个模型同处一个归一化空间，无需分别反归一化。

### E. FPS 与 RDS 的实现口径

- **FPS 沿用仓库既有协议，不另立定义**：`code/test_infbagel_hosi.py:724` 的
  `_seq_fps = _num_frames / _seq_gen_time`，CUDA 同步边界、`timing_warmup_sequences: 5`、
  `batch_size: 1`，聚合量 `aits` / `avg_fps` / `aggregate_fps`；计时只包住采样循环，
  排除 SDF、指标与 IO。与 HOI 表同口径。另加 `rtf = aggregate_fps / 30`（实时倍率，
  对应 LLM state machine 的 long-horizon 用途）与 `denoiser_calls_per_window`——FPS 本身
  跨硬件不可比，调用次数才是机制量。FPS 与几何指标共用同一次采样（batch=1）。
- **`RDS` 必须用 `need_scene=False`，不能用 CFG 的无条件分支**：`infbagel.py:1279-1284`
  的 `need_scene=False` 会把 `scene_emb` 与 `scene_emb_0..3` 全部置零；而
  `is_uncondition=True` / `cfg_scale == -1` 只置零 `scene_embs[1:]`（时序体素），
  保留当前帧场景。用后者会得到被系统性低估的 divergence。该 null-scene 模式已存在，无需新建。
- scene-only 推理**不是消融而是训练内模式**：`InfBaGelMixDataset(lingo_only=…)`
  （`code/datasets/infbagel_mix.py:85`）已是一等公民；缺的只是评测侧接线——
  `code/config/dataset/` 只有 `mix/omomo/omomo_test`，无 LINGO 测试配置，而
  `code/test_infbagel_hosi.py` 已具备自回归 rollout、逐场景 SDF 穿透与 FPS 协议，
  但指向 `data/hosi_test/` 的 67 个合成场景。工作量是把它接到 LINGO 的 test 场景，不是改架构。

### F. 尚未预注册的部分

B/C 的 effective batch / LR / warmup 仍须在 GPU 显存审计得出可行 micro-batch 之后另行联合
预注册（`AGENTS.md` 要求）。作者口径为 4×A100、`batch_size: 512`、diffusion 501 epoch →
CM 201 epoch 蒸馏；本机为 8×3090（24 GB）。由于语料从 OMOMO 换为 LINGO，「同 epoch 数」并不
等于「同训练量」，故预算不变量取 **processed windows**，与本文件开头「以 processed
windows/frames 锁定 HSI 内部预算」一致。C 必须在 B 完成后串行启动。

### G. v3 实测结果与 watertight 重新审计

比例基准为 **eligible source（非镜像）window 池**：镜像只是 train 侧的增广，不参与目标算术，
否则 train 用含镜像计数、held-out 侧用纯源计数去比同一个目标，单位不一致。family 按 source
window 数**降序**指派给当前缺口最大的一侧，seed 42 只用于确定性地打破并列。

| | family | scene | sequence | source window | train 镜像 window | 占源池 / 目标 |
|---|---:|---:|---:|---:|---:|---:|
| train | 48 | 144 | 12,748 | 678,420 | 678,315 | 69.95% / 70% |
| validation | 10 | 12 | 1,152 | 97,161 | 0 | 10.02% / 10% |
| test | 18 | 26 | 2,199 | 194,338 | 0 | 20.04% / 20% |

已独立核验：三侧 family/scene 两两不交；76 个 family 全覆盖且各恰一次；held-out 侧无任何
`_mirror` scene；序列账目 16,099 已指派 + 3,351 丢弃镜像 = 19,450 全量；三侧
`sequence_ids_sha256` 互不相同。**train 的 eligible window 由 v2 的 976,993 增至
1,356,735（+38.9%）**。v1 与 v2 以原调用参数重算后**逐字节相同**，加法式扩展成立。

watertight 重新审计（§D 的表针对 v2 的 37 个 test scene，v3 的 test 集已变）：v3 的
26 个 test scene **全部有 mesh**，其中**仅 1 个非 watertight（`031`）**；validation 12 个
scene 全部有 mesh，其中 1 个非 watertight（`049-bed`）。§D 的两条报告规则不变，只是适用对象
改为 `031`（test）与 `049-bed`（validation）：须标注其符号由 generalized winding number
推出、与其余场景不同口径，且逐场景数字须与 sequence 数同列。相比 v2 的 3/37，fallback 面
更小。

---

## 2026-08-13：B/C 训练预算与 batch/LR/warmup 联合预注册

`AGENTS.md` 要求 effective batch、LR 与 warmup 联合预注册。本节据 2026-08-12 的 GPU 实测
给出取值与其推导。**本节尚有一个未解除的前置条件，见 §4，B 不得在其解除前启动。**

### 1. 实测基础（8×RTX 3090，24 GB，GPU 0 单卡，fp32，real `p_losses` 路径）

| micro-batch | peak alloc | peak reserved | 占 23.57 GiB | step | node 吞吐 |
|---:|---:|---:|---:|---:|---:|
| 64 | 6.16 | 7.14 | 30% | 0.311 s | 1,647 win/s |
| 128 | 7.88 | 12.90 | 55% | 0.459 s | 2,232 win/s |
| 256 | 11.32 | 18.68 | 79% | 0.773 s | 2,650 win/s |
| 384 | — | 失败时 20.14 | — | OOM | — |

5 步与 200 步的 peak reserved 逐位相同，故非短程假象。384 的失败是 allocator 碎片
（已分配 9.89 GiB 而保留 20.14 GiB），不是 24 GB 容量不足。**显存不是约束**：作者口径的
effective 512 只用掉单卡 30%，accumulation factor 为 1。

### 2. 取值与推导

| 项 | 取值 | 依据 |
|---|---|---|
| effective batch | **2048** | 用户决定，取自 `AGENTS.md` 许可集 `{512,1024,2048,3072}` |
| micro-batch × GPU | **256 × 8，accum 1** | 实测可跑；**回退** 128 × 8 accum 2（语义等价，55% 显存，+5.9 h） |
| lr | **2e-4** | 作者 1e-4 @ 512，Adam 用 **√k** 缩放：√(2048/512)=2 |
| warmup | **线性 2,000 updates**（约 1.4%）0→2e-4，其后恒定 | 作者无 warmup；4× batch 下 warmup 是使大 batch 追平小 batch 的标准手段，加它是**减少**结果偏离而非增加偏离 |
| 精度 | **fp32** | AMP 快 17–28% 且 loss 有限，但改变数值口径；baseline 以保真优先 |
| 预算不变量 | **processed windows** | 语料由 OMOMO 换为 LINGO，同 epoch 数 ≠ 同训练量 |

LR 用 √k 而非线性：优化器是朴素 `Adam`（`code/train_infbagel.py:83,86`，`lr` 恒定、无
scheduler）。Adam 的更新被梯度二阶矩归一化，batch 增大 k 倍使梯度噪声标准差降为 1/√k，
故 √k 缩放才保持更新的信噪比；线性缩放是 SGD+momentum 的结论，用于 Adam 通常不稳。

### 3. 预算

OMOMO（`data/train`）共 **597,868** 个 window，故作者训练量为：

| | 作者 epoch | 作者 processed windows | v3 train 等效 epoch | optimizer updates @2048 |
|---|---:|---:|---:|---:|
| B diffusion | 501 | 299,531,868 | 220.8 | 146,255 |
| C consistency | 201 | 120,171,468 | 88.6 | 58,678 |

v3 train 侧为 1,356,735 个 eligible window。若改为「对齐 epoch 数」而非 processed windows，
将处理 2.27× 于作者的训练量（约 115 h），故该不变量是实质性的。

B 的墙钟约 **31.4 h**（2,650 win/s，不含 DDP allreduce，故为下界）。C 的 consistency loss
含 teacher/target 额外前向，其单步成本**未实测**，故 C 的墙钟未知，不在此处编造。

**已知偏离作者的项，全部登记**：语料 OMOMO → LINGO-only；effective batch 512 → 2048
（optimizer updates 由 585,023 降至 146,255，是本次最大的保真代价）；lr 1e-4 → 2e-4；
warmup 无 → 2,000；硬件 4×A100 → 8×3090。墙钟由 micro-batch 而非 effective batch 决定：
effective 2048 相对 1024 只多省 5.9 h，代价是 optimizer updates 再减半。

### 4. 未解除的前置条件

`lingo_only=true` 时 `use_object_keypoints` 仍为 True，`p_losses` 因而对**全零**的
`transformed_obj_verts` 计算 `50 × loss_object` 并叠加 `50 × loss_fk`，即训练模型把 object
keypoint 放到原点；实测总 loss 79–203 由该项主导。这是目标函数语义问题，位于任何
batch/LR 选择之上。**B 不得在其修复并经用户确认前启动。** 修复方案另行提交。

同时登记三项已知缺陷（修复中，不改变本节取值）：scene-only guidance 函数不存在且其分支被
`is_mix` 门控；`_compute_occ_sample` 在 batch > 1 抛错；`get_nearest_free_voxel` 未在
`InfBaGelMixDataset` 上暴露。三者只影响采样与评测路径，不影响训练预算。

> **2026-08-15 订正（第一项已不成立）**：截至 HEAD `886aa16`，scene-only guidance 函数
> **存在**——`code/guidance_loss.py:96` 的 `apply_hsi_guidance_loss`，由
> `code/models/infbagel.py:8` 导入，在 `cm_sample`（`:717`）与 `p_sample`（`:969`）两处调用；
> 且其分支**不由 `is_mix` 门控**，而由 `if not is_object.any():`（`:715` / `:967`）门控。
> 上述两处描述均已被 commit `154b24d` / `130882e` / `886aa16` 推翻。后两项缺陷的登记不变。

---

## 2026-08-13（同日修订）：§4 前置条件解除、v3 接线与 guidance 不对称

本节记录 §4 前置条件的解除，并登记一项**改变三模型可比性**的新发现。

### A. §4 的前置条件已解除

`p_losses` 读取 mix wrapper 的 `use_object_keypoints`（True），而 LINGO 子数据集以
`use_object_keypoints=False` 构造，故 `lingo_only` 下 `loss_object` 对**全零**的
`transformed_obj_verts` 计算，在 `loss_w_obj_pts: 50` 下主导总 loss（实测 79–203）。
现按 per-sample `is_object` 门控，无样本存活时返回 `None`；`loss_fk` 属纯人体监督，保留。

**同一缺陷在 `consistency_loss`（`code/models/infbagel.py:373-427`）另有一份**，该函数训练
C，若不修则蒸馏阶段重现该 bug；已一并修复并保留 `NotImplementedError` 分支。训练循环的
guard 已拆分，使 `loss_object is None` 时 `loss_fk` 仍被累加与记录。

`loss_otrans` / `loss_orot` / `loss_contact` **刻意保持**对 LINGO 样本的零目标监督，
依据是 216:232 通道的 zero-target 决定（未来 mixer 可能消费这些通道）。

**B 的启动条件就此解除。**

### B. `occ_ref` 与 v3 接线

- `compute_occ_ref` 原仅在 `vis=True` 下执行，真实 254 场景下 `scene_occ_ref` 恒为 `[]`，
  `get_nearest_free_voxel` 抛 `IndexError`，guided 采样不可达。实测每场景 72 MB / 2.6 s，
  全量 254 场景为 **18.3 GB**，与 3.0 GB 的 `scene_occ` 并存不可行。`occ_ref` 是 `occ` 的
  纯函数，故改为 `LazyOccRef` 按需计算 + 4 项 LRU；`vis=True` 路径逐位不变。
- **v3 接线附带修复 scene 条件缺陷**：数据集原以 `scene_name[start_ind[idx]]` 取 scene，
  即 §2 的缺陷标签，约半数语料被喂入错误房间的几何。现按
  `scene(w) = scene_name[start_idx[seq]]`（`seq < H`）或
  `scene_name[start_idx[seq-H]] + "_mirror"`（`seq >= H`）逐窗口改正，并同时用于选择与
  `__getitem__` 的 `scene_flag`。该规则在实施前已独立核验可**精确**复现 manifest 的四个
  partition 计数。
- **预算口径的次要更正**：数据集另有 `seq_length <= 48` 过滤（manifest 计数不含），故
  dataset-true 的 train 池为 **1,343,667** 而非 1,356,735（低 0.96%）。预算不变量是
  processed windows，故 effective batch / lr / warmup / 146,255 optimizer updates
  **均不变**，仅等效 epoch 由 220.8 变为 222.9。
- 默认路径（无 `split_manifest`）逐项复核未变：仍为 `all_scenes[:45]` 的 495,179 窗口、
  不构造改正标签、保留发布侧条件，故既有复现仍然有效。

### C. 新发现：guidance 只存在于 consistency 路径（影响 gate 归属）

`guidance_fn` 在 `code/models/infbagel.py` 全文仅 4 处，**全部位于 `cm_sample_loop` /
`cm_sample`**。`p_sample_loop`（`:884`）与 `p_sample`（`:916`）均带 `@torch.no_grad()`
且签名中**没有** `guidance_fn` / `guidance_scale` / `human_dict`。

后果：§D 表中 **B（gate，diffusion）是唯一无法 guided 采样的模型**，而 A/C 皆可。这不是
装饰性差异——HOI 阶段已实测「已发布 baseline 行是 guided 16-step CM 采样，guidance 占 contact
gap 的 59%，迭代步数贡献为零」。若以 guided-A 对 unguided-B 成表，则 protocol 差异与语料、
模型类别差异混在同一轴上，95% 门槛将读在一个被混淆的数字上。

> **2026-08-15 订正（本段的事实前提已不成立）**：本节写于 commit `c3ce25e` 之前。截至 HEAD
> `886aa16`，`p_sample` 的签名已带 `guidance_fn` / `guidance_scale` / `human_dict`
> （`code/models/infbagel.py:919-921`），并在 `:943` 以 `if guidance_fn is None:` 分流、
> 在 `:969` / `:1004` 施加梯度，与 `cm_sample` 结构同构。**「B 是唯一无法 guided 采样的模型」
> 已被推翻**；guided diffusion 可运行，其相对 unguided 的墙钟代价已由并行的 run-B 评测
> session 实测（数字记录在该 session 的 registry，尚未并入本 checkout，故此处不复述）。
> 这一订正正是下文 §D 决定要求实施的那半个口径 1 已经落地的结果，本段只是没有随之更新。
> 它同时移除了 2026-08-15（同日第二次修订）§A 把 gate 迁到 C 的一个反对理由。

三个候选口径（**尚未选定，待用户决定**）：

1. 把 guidance 移植进 `p_sample`，使三模型同为 guided。可比性最强，但属 sampler 实质变更，
   且作者从未运行 diffusion+guidance，无参考行为可对照。
2. 三模型各报 guided 与 unguided 两列，gate 取 unguided（三者原生可比的那一列）。无新
   sampler 代码，代价是采样时间翻倍，且 gate 落在 baseline 未发表过的 protocol 上。
3. 改以 C 为 gate（与 A 同为 CM，正是 §D 已指出的「A vs C 才可归因」）。但 C 依赖 B 先完成，
   gate 因此移到链尾，B 在消耗约 31 h 算力期间无 gate。

倾向 2（最便宜且诚实，unguided 列无论如何都要采），必要时叠加 1；不单取 3，因其使 B 无 gate。

### D. 用户决定：取口径 2（guided 与 unguided 双列，gate 取 unguided）

**本项目 Phase 1C 的评测协议就此定版：**

1. A / B / C 三个模型**各报 guided 与 unguided 两列**。
2. **gate 取 unguided 列**——该列三模型原生可比，不含 protocol 差异。
3. guided 列作为参考，用于量化 guidance 本身的贡献（HOI 阶段实测其占 contact gap 的 59%）。
4. 95% 门槛在 unguided 列上判定；guided 列不参与 gate。

随该决定同时实施口径 1 的一半：为使 unguided 列在 diffusion 侧存在，`p_sample_loop` /
`p_sample` 需接受 `guidance_fn`（可为 `None`）。**硬约束：`guidance_fn=None` 时两函数的输出
必须与改动前逐位相同**——这正是 unguided 列可与既有采样结果比较的前提。

`RDS` 的 null-scene 仍按 §E 规定用 `need_scene=False`（完整置零 `scene_emb` 与
`scene_emb_0..3`），不得用 CFG 的 `is_uncondition=True` / `cfg_scale == -1`（只置零
`scene_embs[1:]`，保留当前帧场景，会系统性低估 divergence）。

评测集：v3 test partition（18 family / 26 scene / 2,199 sequence）按**逐场景上限 20 条、
优先取最长序列、并列按窗口索引升序**的确定性规则抽样，不使用随机种子。理由是长序列才压到
自回归接缝，而接缝连续性是 Tier 3 的测量对象。


---

## 2026-08-15：run B 训练完成、checkpoint 选择与 run-B 评测预注册

### A. Run B 训练完成

Run B 已以 `TRAINER_EXIT=0` 完成 223 个 epoch、146,255 次 optimizer update；已记录的
`29.6 h wall clock` 与 `0.7305 s/update sustained` 原样沿用，不在本节重新测量。epoch 222
平均训练 loss 为 0.2021，共产出 13 个 epoch checkpoint。封口记录见
`results/experiments/p1-hsi-b-lingo-full-s42-20260814/manifest.json:1`，训练登记见
`experiments/registry.jsonl:276`。该 manifest 在本会话任何提交出现之前，以干净工作树封口于
`a61ffe7`，因此记录的是实际运行代码。

### B. 用户决定：checkpoint 选择 = final-epoch-only

**评测 checkpoint 在运行前固定为 `hsi_b_lingo_full_epoch222.pth`；这不是 checkpoint 间的
质量比较，也不是从评测集择优。** 下列 spike 分析只检查固定规则没有落在受扰 checkpoint 上，
不构成选择依据：epoch 160 step 340 的未裁剪梯度事件将 loss 从 0.2038 推至 0.7454；
`epoch160.pth` 写于该 spike 后，checkpoint 邻近 loss 为 0.4830；`epoch180.pth` 仍在恢复中
（0.2147）。epoch 222 已完全恢复至 0.2021，略低于 spike 前 epoch 158 的 0.2032。

### C. 发现（写给后续 run，不改本 run）

`code/train_infbagel.py` 全文没有梯度裁剪。本 run 实测两次未裁剪梯度事件：epoch 10
step 50（5.94x）与 epoch 160 step 340（3.66x）。**本阶段不修改 trainer**——B 已训练完成，
裁剪无法追溯生效；任何 trainer 变更都属于 C 或后续 run 的独立预注册决定，不在本节范围内。

### D. 用户决定：`RDS` 落地实现，不降级、不裁掉

`RDS` 正由并行 agent 在独立 worktree 实现，其接线将在评测运行前落到 `phase/01c-hsi`。
沿用 2026-08-12 §E 的硬约束：null-scene 必须使用 `need_scene=False`，绝不能走 CFG 的
`is_uncondition=True` / `cfg_scale == -1` 分支。

### E. run-B 评测预注册

评测协议在执行前完全由以下既有定版段落约束；本节只引用，不重写或改变任何公式：

- 2026-08-12（同日修订）§C：penetration triple、包含 `RDS` 的 engagement、foot-skate
  triple、goal reaching 与 seam continuity 的固定公式；
- 2026-08-12（同日第二次修订）§D：三模型表、B 的 gate 角色与 goal-slot caveat；
- 2026-08-12（同日第二次修订）§E：FPS 协议及 `RDS` 的 `need_scene=False` 约束；
- 2026-08-13（同日修订）§D：guided/unguided 双列、unguided gate 列及 v3 test 的确定性抽样。

用户在 2026-08-15 另行固定 final-epoch-only checkpoint 选择，并决定 `RDS` 按实现落地而非
降级。今日登记 `p1-hsi-b-layout-4x512-s42-20260814` 与
`p1-hsi-b-eval-preregister-s42-20260815`；后者明确发生在任何 run-B 评测之前。
**Phase 1C 的具体 gate 判据仍由用户暂缓定义，本节不定义任何阈值或 gate 数值。**
当前不存在 run-B 评测指标。

治理修复：`code/config/config_train_hsi_b_lingo_full.yaml:35` 与 commit `a61ffe7` 在 registry
row 存在前已引用 `p1-hsi-b-layout-4x512-s42-20260814`；今日 amendment 关闭该悬空引用。
`experiments/training_resource_protocol.json` 刻意不改：`hardware_assignment` 描述 HSI 所属的
8-GPU hardware pool，不描述单次 run 的用卡数；run B 的 4-of-8 分配已由 manifest override
`visible_and_allocated_gpus=0,1,2,3` 记录，且 `tests/test_research_governance.py:177` 固定
`hardware_assignment["hsi"]["gpu_count"] == 8`，不能为描述单次分配而改写该治理不变量。

---

## 2026-08-15（同日第二次修订）：model C 蒸馏预注册、gate 迁移与三处文档订正

本节预注册 **model C**（从 run B 的 consistency-model 蒸馏），把 gate 从 unguided-B 迁到
guided-C，并订正三处已被 HEAD 推翻的旧描述。**本节不启动任何 GPU 工作负载**；C 的启动仍需
用户对本节给出的那一条具体命令逐条批准。

### A. 用户决定：gate 由 unguided-B 迁至 guided-C；A 移出评测矩阵

2026-08-13 §C 曾列出三个候选口径，其中「以 C 为 gate」被否，唯一理由是它
「使 gate 移到链尾，B 在消耗约 31 h 算力期间无 gate」。**该反对理由已随 run B 训练完成而失效**
（run B 于 2026-08-15 以 `TRAINER_EXIT=0` 收尾，见本文件同日 §A）。B 已训练，不再存在
「B 无 gate 地烧算力」这一风险，故当初否掉 C-gate 的成本论据不再成立。

同时，2026-08-13 §C 的另一条前提也已不成立：**guided 采样不再是 CM 独有**（见下文 §E 第 2 条
订正）。因此「gate 必须落在 unguided 列」这一为求原生可比而设的约束，其必要性已经减弱——
B 与 C 现在两列都能出。

用户据此定版：

1. **gate = C，在 guided consistency 采样下判定。** 理由是 C 才是本阶段要交付的产物形态
   （16 步 CM + guidance），gate 应当判在交付形态上，而不是判在它的 teacher 上。
2. **B 由 gate 降为 teacher 与蒸馏前对照**，仍报 guided / unguided 双列，用于回答
   「蒸馏损失了多少」。
3. **A 整体移出评测矩阵。** A 是 OMOMO-only、从未见过 LINGO，其指标不为迭代提供信息；
   2026-08-12 §D 已记录 A 在 goal 槽位上另受一次分布偏移。评测矩阵自此为 **B 与 C 两个模型**。
4. 2026-08-13 §D 关于 guided / unguided 双列、`RDS` 必须用 `need_scene=False`、以及 v3 test
   的确定性抽样规则**全部不变**，只是 gate 所在的列由 unguided 改为 guided。

**代价，如实登记**：gate 落在 guided 列意味着 gate 数字里含有 guidance 的贡献，而 HOI 阶段
实测 guidance 可占 contact gap 的 59%。这使 gate 更贴近交付形态，但更不容易归因到模型本身。
unguided 列仍然必报，正是为了把这部分拆开。**具体 gate 阈值仍由用户暂缓定义，本节不定义
任何阈值。**

### B. C 的配置

`code/config/config_train_hsi_c_lingo_cm.yaml`（本节新增，未提交）。构造原则是
**作者的 CM 程序 + run B 的数据/场景设定**，逐键核对过
`code/config/config_train_infbagel_cm.yaml` 与 `code/config/config_train_hsi_b_lingo_full.yaml`
两份来源，除下列四项外每一个键都等于其中之一：`exp_name`、`epochs`、
`max_optimizer_updates`、`ckpt_path`。

取自作者 CM 配置：`sample_type: consistency`、`load_state_dict: true`、
`loss_w_obj_pts: 0.5`、`loss_w_fk: 1`、`batch_size: 512`、`ckpt_interval: 20`。
取自 run B：`dataset: lingo_v3_train`、`lingo_only`、`lingo_scene_num: 45`、
`lingo_data_ratio: 0.5`、`empty_omomo_scene: false`、`human_only_ratio: 0.4`、
`scene_type: occ_temp`、`temp_voxel_num: 3`、`max_window_size: 16`、全部 `load_*` 标志、
`seed`/`random_seed: 42`、`precision: bf16_tf32`、`num_gpus: 4`、`effective_batch_size: 2048`、
`gradient_accumulation_steps: 1`、`lr`、`warmup_updates`。

**`is_mix` 在 model 与 sampler 两侧均为 false**，已从**解析后**的配置确认
（`model.infbagel.is_mix == false` 且 `sampler.pelvis.is_mix == false`），不是从
defaults 推断。run B 曾为此专门提交 `e99d3f2`，因为只改一侧是静默错误的。

**teacher checkpoint**：`ckpt_path` 指向 run B 的**末 epoch** checkpoint
`results/hsi_b_lingo_full/checkpoints/hsi_b_lingo_full_epoch222.pth`
（sha256 `931a6f1f…48c5e`，已重新校验），配合 `load_state_dict: true`。这与用户的
final-epoch-only 规则一致，也与作者自己的做法一致——作者 CM 复现run 的 `ckpt_path` 指向其
diffusion run 的 `epoch500`，即同样是末 epoch。

**已核实的加载语义**：`code/utils.py:253-274` 用 `strict=False` 加载，会静默容忍键不匹配
（2026-08-12 §C 记录过 A 因此把 `embedding_scene_goal` 留在随机初始化）。已在 CPU 上逐键比对：
B 的 checkpoint 与新实例化的 `Unet` **各 218 个键，missing 与 unexpected 均为空集**。
B → C 不存在被 `strict=False` 掩盖的不匹配。

**已核实的可训练面**：`code/train_infbagel.py:328-333` 先 `requires_grad_(False)`，再只放开
`embedding_input` / `embedding_output` / `transformer` / `out`。因此 `scene_embedding`（ViT）、
`embedding_language`、三个 goal embedding、`embed_timestep`、`bps_encoder`、
`cfg_scale_embedding` **全部冻结在 B 的取值上**。其中 `cfg_scale_embedding` 值得单独记一笔：
`p_losses` 从不传 `cfg_scale`，所以 B 训练期间该模块无梯度、始终停留在随机初始化，而 C 又把它
冻结——即 C 的 w-conditioning 走的是一个**固定随机投影**。这是作者结构本身的性质（作者的
diffusion→CM 链路完全相同），不是本仓库引入的偏离，故照录不改，仅登记。

**`AGENTS.md:189-191` 的解析配置 preflight 已执行**，产物为
`.claude/scratch/config_train_hsi_c_lingo_cm.resolved.yaml`，**未解析插值计数为 0**。
run B 未做该 preflight，C 补上。

顺带订正一条流传中的说法：**`ROOT_DIR` 并不需要在 `--cfg job --resolve` 之前导出**。
`code/train_infbagel.py:18` 在**模块导入期**就无条件执行 `os.environ['ROOT_DIR'] = '..'`，
早于 Hydra 组装，也会覆盖任何已导出的值。已实测：导出与不导出两次运行**退出码均为 0、输出
逐字节相同**，都没有 `InterpolationResolutionError`。真正需要预先导出的是评测入口
`code/test_infbagel_hosi.py`（其 `ROOT_DIR` 赋值在 `:897`，位于 `@hydra.main` 之后）。
其直接后果是：解析后的配置里所有路径都是**相对 `code/` 的相对路径**（例如
`../results/hsi_b_lingo_full/checkpoints/hsi_b_lingo_full_epoch222.pth`），因此 C 的启动脚本
必须像 B 的 `launch.sh` 一样先 `cd` 到 `code/`。

### C. 四个开放问题的裁定

**0.（前置核实）几何损失权重：`consistency_loss` 与 `p_losses` 的施加方式相同，但只有
`loss_w_fk` 真正生效。**
`code/train_infbagel.py:476-483`（consistency）与 `:488-494`（diffusion）用**完全相同**的两行
把 `cfg.loss_w_obj_pts * loss_object` 与 `cfg.loss_w_fk * loss_fk` 加到基项上，两个目标下这两个
键的语义一致。因此「作者蒸馏时把几何权重降 50–100 倍」这一读法在代码上成立，C 取
`loss_w_fk: 1` / `loss_w_obj_pts: 0.5` 是对的。

但要点明两处：(a) **`loss_w_obj_pts` 在 `lingo_only` 下是死键**——`infbagel_mix.py:460` 对每个
LINGO 样本置 `is_object = False`，而 `lingo_only` 下所有样本都走该分支，故
`consistency_loss` 的 `mask_points` 恒为空、`loss_object` 恒为 `None`（2026-08-13 同日修订 §A
的修复正是如此设计）。50 → 0.5 这一半对 B/C 都无效果，**唯一真正改变目标函数的是
`loss_w_fk` 50 → 1**。(b) 权重虽同名同施加方式，**基项的量级并不相同**：`p_losses` 的基项是
五个重建项之和，`consistency_loss` 的基项是 student 与 EMA target 之间的单个 MSE。作者之所以
必须调低几何权重，机制正在于此——同一个 50 在两个基项下对应完全不同的相对定价。这也说明
「照抄 B 的 50/50」在蒸馏里是错的，而不只是不忠实。

**1. `w`：该键对 CM 训练完全无效，问题本身不成立。**
`w` 经 `sampler/pelvis.yaml` 的 `w: ${w}` 进入 `Sampler.self.w`（`infbagel.py:38`），而
`self.w` 在全文**只被读取一处**：`p_sample` 的 `:928`，即 diffusion **采样**时的 CFG 系数。
两个训练目标都不读它——`p_losses` 从不引用；`consistency_loss` 在 `:306` 用
`w, is_uncond = self.sample_cfg_scale_mixed(...)` **就地覆盖**了这个名字，改为按样本随机抽
CFG 尺度（硬编码 `uncond_prob=0.1`、`w_max=2.0`，即 10% 抽 `w=-1`、其余均匀取 `[0,2)`）。
`code/train_infbagel.py` 全文不出现 `cfg.w`。

结论：作者 CM 配置的 `w: 1` 与 mix 配置的 `w: 0` **不是训练程序差异**，而只是各自 sampling
默认值的继承；run B 的 `w: 0` 同样对 B 的训练零影响，是从 mix 配置带过来的惰性值，
`e99d3f2` 无须处理它是对的。C 取 **`w: 1`**，理由纯属文档性：与作者 plain CM 配置一致，
且与评测配置 `config_sample_infbagel_lingo_hsi.yaml:44` 的 `w: 1` 一致，使归档的解析配置不至于
暗示一个和 gate 实际所用不同的采样尺度。**它不构成对 C 的任何约束，也不需要在 B 与 C 之间对齐。**

**2. `lr`：取 2e-4，与 B 相同，并保留 `warmup_updates: 2000`。**
作者的 1e-4 是在 effective batch 512 上取的；2026-08-13 §2 已用 Adam 的 √k 规则把它折算到
effective 2048 得到 2e-4，C 的 effective batch 与 B 相同，同一推导给出同一结果。另一条独立
理由：作者在 diffusion 与 CM 之间**保持 lr 不变**，因此「与自己的 teacher run 同 lr」才是作者
口径下的忠实关系，照抄字面 1e-4 反而引入一个已知的 batch 失配。warmup 保留 2000 update——在
蒸馏里它比在 B 里更有理由：第 0 步时 student / target / teacher 三者**逐位等于 B**，consistency
残差极小且高度结构化，一个全尺寸的首步会把 student 直接踢离 teacher 的流形。
**反对意见如实登记**：2000 update 在 B 是总量的 1.4%，在 C 是 3.4%，比例不同；且 2e-4 在 B 下
已实测出两次未裁剪梯度事件（见第 4 条）。若用户否决第 4 条的裁剪建议，则此处应回退到 1e-4。

**3. `epochs`：取 90，`max_optimizer_updates: 58678`——即 2026-08-13 §3 已预注册的值。**
「作者 CM 是 201 epoch 对其 501 epoch diffusion，比值 0.40」这个读法**经不起检验**：作者的 mix
配置对是 `config_train_infbagel_mix.yaml` 的 **1001** epoch 对 `config_train_infbagel_mix_cm.yaml`
的 **201** epoch，比值 0.20。作者的 CM 预算是一个**常数 201**，不是比值。故「同比值」没有作者
依据，「同字面 201 epoch」则因语料从 OMOMO 换成 LINGO 而不等于同训练量（201 × 656 =
131,856 update，约为应有量的 2.25 倍）。

本阶段的预算不变量早已定为 **processed windows**（2026-08-13 §2），且 C 的数值当时就已连同 B
一起预注册：作者 C = 201 × 597,868 = 120,171,468 processed window ÷ 2048 = **58,678 optimizer
update**。B 用同一规则得 146,255，实际就以该值收尾。58,678 ÷ 656 update/epoch = 89.45，故
`epochs: 90` 是让 `max_optimizer_updates` 恰好绑定的最小值，第 89 个 epoch 在第 294 步被截断，
而 `epoch == cfg.epochs - 1` 使 `epoch089.pth` 仍被写出——与 B 的 222/223 完全同构。

**`epochs: 90` 是承重的，不是随手取的**：checkpoint 条件为
`epoch % ckpt_interval == 0 or epoch == cfg.epochs - 1`，而 89 % 20 = 9。若写成 `epochs: 91`，
`stop_training` 仍在 epoch 89 触发，但两个条件都不满足，**末 epoch checkpoint 不会被写出**，
而 checkpoint 选择规则是 final-epoch-only。反方向的余量则很薄：90 个 epoch 要凑满 58,678 次
update 需要 ≥ 651.98 step/epoch，实测为 656（余量 0.6%）。C 的 dataset 键与 B 逐项相同，故
656 应当复现；即便偏低，后果也只是预算少 ≤0.6%，而非缺 checkpoint。

**巧合值得点明**：0.401 × 223 = 89.4，与 processed-windows 规则给出的 89.45 几乎相同。原因是
B 的 223 本身就是按 processed windows 对齐到作者 501 的，所以两种算法在此处必然重合。用户的
「0.40 比值」直觉在数值上是对的，只是其归因（作者用比值）不成立。

**一处已登记不改的舍入不一致**：120,171,468 ÷ 2048 = 58,677.47，向下取整为 58,677；而
2026-08-13 §3 登记的是 58,678（向上取整）。B 的 146,255 则取的是向下取整
（299,531,868 ÷ 2048 = 146,255.79）。两行的取整方向不一致，差 1 个 update、2,048 个 window，
占 0.0017%。**此处沿用已预注册的 58,678，不静默改写预注册预算**，仅在此登记该不一致。

**4. 梯度裁剪：建议加，但本节不实施。**
`code/train_infbagel.py:513-519` 在 `backward()` 与 `optimizer.step()` 之间没有任何裁剪。
run B 实测两次未裁剪梯度事件，第二次（epoch 160 step 340，0.2038 → 0.7454）耗掉约 23 个 epoch
才恢复。**在 C 上这件事比在 B 上更危险**：C 只有 58,678 个 update（B 的 40%），而
checkpoint 选择是 final-epoch-only。按 B 的事件率（2 次 / 146,255 update）与 B 的恢复长度
（约 15,000 update）粗算，C 期望遭遇约 0.8 次事件，其中落在「来不及恢复」的尾部区间的概率
约两成——即**约 20% 的概率让 gate 读在一个受损的 checkpoint 上**。

裁剪的忠实度代价比它看上去小：`clip_grad_norm_` 只在梯度范数超过阈值的那些步上生效，其余步
的更新与不裁剪**逐位相同**。也就是说这个偏离是**条件性的**，只改变那些本来就已经异常的步。
这是加它的最强理由。

**但本节仍不实施**，原因有三：(a) 它改的是 `code/train_infbagel.py` 这一共享训练代码，按
`AGENTS.md` 需在首个 GPU 工作负载前跑一次完整 authority suite，并补真实数据 smoke，属于独立的
一个变更单元；(b) 改后 C 的 trainer 将不再等于产出 B 的那个 trainer，这是必须单独登记的偏离；
(c) **阈值无据可取**——B 的日志只记 loss，不记 grad norm，凭空取 `max_norm=1.0` 有真实风险：
若典型范数远大于 1，裁剪会在每一步静默地压低有效 lr，那是比 spike 更坏的、不可见的改动。
故正确顺序是先测一次 grad norm 分布再定值，该测量本身是 GPU 工作负载，需用户批准（见 §F）。

### D. 预算

C 的墙钟不能只由 B 的 0.71223 s/update 线性外推：`consistency_loss` 每步比 `p_losses`
多三次无梯度前向（teacher 的 cond 与 uncond、target 各一次，`infbagel.py:325-348`）外加一次
全参数 EMA（`:262`）。该倍率**不在本 checkout 的记录内**，故按下述方式取值并如实标注来源：

> **2026-08-16 作废：本表整体被本分支的直接实测取代，见下文 2026-08-16 §A。** 表中的
> 2.30 倍率与由它导出的 26.7 h / 107 GPU-h **已被丢弃，不是被批准**；本分支现有自己对
> `consistency_loss` 真实路径的 s/update 实测，不再需要任何跨 checkout 数字。**不得再从
> 2.30 重新推导任何预算。** 本分支自己的数据还直接否定了 2.30：见 2026-08-16 §B。
> 原表原样保留以存证。

| 项 | 值 | 依据 |
|---|---:|---|
| optimizer updates | 58,678 | 2026-08-13 §3 预注册 |
| 布局 | 4 GPU × micro-batch 512 × accum 1 | 同 B，effective 2048 |
| diffusion s/update | 0.71223 | run B 布局实测（`p1-hsi-b-layout-4x512-s42-20260814`） |
| CM / diffusion 倍率 | ~~**约 2.30**~~ **已丢弃** | 见下 |
| **墙钟** | ~~**约 26.7 h**~~ → **实测 10.32 h** | 2026-08-16 §A |
| **GPU-h** | ~~**约 107**~~ → **实测 41.3** | 2026-08-16 §A |

若改用 run B 全程实测的 0.7305 s/update sustained，则为 27.4 h / 110 GPU-h（+2.6%）。倍率取
2.0–2.6 的区间时墙钟为 23.2–30.2 h。**该 2.30 的来源需要用户裁决后才能写成本仓库的依据**：
它来自本机上一对作者复现 run（同主机、同 8×256×accum1 布局、同 fp32、同 OMOMO 语料、同为
291 step/epoch）的 checkpoint 时间戳——diffusion 316.51 s/epoch 对 CM 727.49 s/epoch，
比值 2.298；其 diffusion 侧 1.0877 s/update 与本仓库 commit `ca62f74` 已记录的 1.0845 相差
0.3%，互为佐证。但那两个 run 位于另一个 checkout 且以 `p1b-` 命名，**引用它属于跨分支取数，
按 `AGENTS.md` 需用户批准**，故本节标注其为待批准的外部佐证，而不把它登记为本分支的实测值。
在批准之前，本节的墙钟应读作**估计**：唯一无争议的下界是倍率取 1 时的 11.6 h，而该下界在机制上
不可能达到。

作为对照：若误取「作者字面 201 epoch」，则为 131,856 update ≈ 60 h / 240 GPU-h，是本方案的 2.25 倍。

### E. 三处文档订正（已就地施行，原文以删除线／引用块保留）

1. **§D 三模型表（本文件 2026-08-12 同日第二次修订）**：B 的「gate」角色已改写，见上文 §A。
2. **2026-08-13 同日修订 §C「B 是唯一无法 guided 采样的模型」——已推翻。** 截至 HEAD
   `886aa16`，`p_sample`（`infbagel.py:919-921`）签名已含 `guidance_fn` / `guidance_scale` /
   `human_dict`，`:943` 以 `if guidance_fn is None:` 分流，`:969` 调用
   `apply_hsi_guidance_loss`、`:1004` 调用 `guidance_fn`，与 `cm_sample` 结构同构。该节所写的
   「`guidance_fn` 全文仅 4 处且全部位于 `cm_sample_loop` / `cm_sample`」在当时为真，之后被
   commit `c3ce25e` 改变，而该节未随之更新。
3. **2026-08-13 §4「scene-only guidance 函数不存在且其分支被 `is_mix` 门控」——两半都不成立。**
   函数存在（`code/guidance_loss.py:96` 的 `apply_hsi_guidance_loss`），分支由
   `if not is_object.any():`（`infbagel.py:715` / `:967`）门控，与 `is_mix` 无关。

### F. 本节未做、需用户批准才能做的事

1. **启动 C 的训练**（约 27 h / 107 GPU-h）。
2. **一次 ≤300 update 的 C 冒烟**，用途有三，缺一不可：(a) `consistency_loss` 在
   `lingo_only` 修复（2026-08-13 同日修订 §A）之后**从未被执行过**，C 将是首次；
   (b) 实测 CM/diffusion 倍率，把 §D 的估计换成本分支自己的实测值，从而不必跨分支取数；
   (c) 实测三模型并存下 micro-batch 512 的显存峰值——run B 未写
   `benchmark_metrics_path`，本 checkout 没有 B 的显存记录，而 C 的 student 计算图与
   teacher/target 的无梯度前向在同一区间内共存。
3. **梯度裁剪的实施**（§C 第 4 条），及其所需的 grad-norm 分布测量与完整 authority suite。
4. **引用 §D 那个 2.30 倍率的跨 checkout 来源**。

> **2026-08-16 结清**：第 2 项已执行（≤300 update 的 C 冒烟，实为多组计时/等价性探针，见下节）；
> 第 1 项已获用户批准并于本日启动；第 4 项**不再需要**——2.30 已丢弃，本分支改用自己的实测值，
> 该跨 checkout 引用请求就此撤回。第 3 项仍未实施：grad-norm 分布测量已做（本 run 全程开启
> 仪表），但**阈值仍未定**，裁剪推迟到 update 2,000 之后的分布再决定，见下节 §E。

---

## 2026-08-16：C 的墙钟改为本分支实测、OMP 线程上限、8×256 的进程不中性、裁剪推迟

本节把上文 2026-08-15 §D 的估计预算替换为**本分支自己的实测值**，登记为此所做的三组
探针及其对照，并固定 C 启动时的最终参数。**本节写于 C 启动之前**；其后的 run 由
`p1-hsi-c-lingo-cm-s42-20260816` 登记。

所有探针都跑在真实的 C 配置上（`config_train_hsi_c_lingo_cm`，只覆盖
`exp_name` / `max_optimizer_updates` / `save_checkpoints` / `benchmark_metrics_path` /
`hydra.run.dir`），即真实的 `consistency_loss` 路径、真实 teacher、真实 LINGO v3 train 数据，
不是代理基准。计时口径：稳态取 update 51–300，由仪表自身写下的 per-update `t_mono`
（rank 0）求算，不用 stdout 时间戳（后者仅作交叉核对，两者相差 ≤0.4%）。原始
`grad_norms/*.jsonl` 保留在各探针的 `results/hsi_c_lab_*/`，独立复算脚本与输出为
`.claude/scratch/c_verify_independent.py` / `.out`。

### A. 实测墙钟：10.32 h / 41.3 GPU-h（取代 26.7 h / 107 GPU-h）

固定布局 4 GPU × micro-batch 512 × accum 1（effective 2048）、bf16+TF32、seed 42。
counterbalanced ABBA，每臂两 rep：先 U,O,O,U，再 X,X。

| 臂 | `OMP_NUM_THREADS` | 两 rep | s/update | 臂内散布 | 58,678 update 墙钟 | GPU-h |
|---|---|---|---:|---:|---:|---:|
| uncapped | 未设置 | 1.12008 / 1.12289 | 1.12149 | 0.25% | 18.28 h | 73.1 |
| capped 9 | 9 | 0.63617 / 0.63171 | 0.63394 | 0.70% | 10.33 h | 41.3 |
| **capped 4（本 run 采用）** | **4** | 0.63550 / 0.63120 | **0.63335** | 0.68% | **10.32 h** | **41.3** |

加速比 **1.771×**。独立复算（从原始 `t_mono` 重算，方法为端点跨度／区间数而非逐差均值）
给出 uncapped 1.12193、OMP=9 0.63393、OMP=4 0.63339，与上表相差 ≤0.05%，故上表数字不是
单一脚本的产物。

一并登记一个**同布局的会话间偏移**：更早的布局探针里 4×512 uncapped 臂为 1.14273 s/update
（18.63 h / 74.5 GPU-h），比本次新鲜 uncapped 臂高 1.9%。上表取新鲜臂，因为它与两个 capped
臂同会话、同 ABBA 序列，是唯一无会话混淆的比较基准；归档臂的数字不作废，只是不作比较基准。

### B. 2.30 倍率被丢弃，并且被本分支的数据否定

2026-08-15 §D 的 26.7 h 来自一个 **2.30 的 CM/diffusion 倍率**，其来源是另一个 checkout 的
一对 `p1b-` 复现 run。该引用请求**撤回，该倍率丢弃**——不是获批，而是不再需要，因为本分支
现在有对同一目标函数的直接实测。**任何后续工作都不得再从 2.30 重新推导预算。**

顺带记下它错得有多远：run B 的 4×512 diffusion 布局实测为 0.71223 s/update（uncapped），
本节 4×512 CM 的 uncapped 实测为 1.12149（归档臂 1.14273），故本分支自己的 CM/diffusion
倍率为 **1.57–1.60**，而非 2.30——原估计高了约 44–46%。（口径提示：这是两个不同 run 之间
的比值，两者同分支、同布局、同为 uncapped，不是一次受控 A/B。）

### C. OMP 线程上限是逐位中性的，因此它是调度改动而非数值改动

**验收测试**：capped 与 uncapped 的每一对，比较全部 post-allreduce 梯度范数与 stdout loss。
`O1,O2,X1,X2` × `B1,B2` 共 **8 对直接比较**全部逐位相同（每对 1,200 条范数记录、120 行 loss）；
再加 `U1,U2` 对每个 capped rep、以及 `U1 vs U2`、`O1 vs O2`、`X1 vs X2` 亦全部逐位相同。

**一处必须写明的口径修正**：那 1,200 条记录并非 1,200 个独立值。仪表刻意不使用 `no_sync`，
记录的是 DDP 已跨 rank 平均后的**全局**范数，四个 rank 在每个 update 上写下同一个数
（已逐 update 核验：`all 4 ranks equal at every update: True`，distinct 值恰为 300）。故每对
比较的独立样本是 **300 个 per-update 全局范数 + 120 行 loss**，不是 1,200。把 1,200 当独立值
会把该测试的效力高估 4 倍。

**对照**（否则「全都相同」可能只是测试没有分辨力）：

- *会话间确定性（正对照）*：新鲜 uncapped `U1`/`U2` 与归档 `B1`/`B2` 逐位相同，说明基线本身
  可跨会话复现，比较不是靠噪声掩盖差异通过的。
- *值级负对照（效力证明）*：把 world size 由 4 改为 8（其余一切相同、数据可证同一），
  **1,200/1,200 条全部不同**；update 1 的相对分歧为 **4.6012%**，在 300 个 update 的窗口上
  相对分歧中位 **18.46%**、均值 21.79%、最大 66.98%。即真正改变计算的干预会被大声检出。
- *分辨力下限*：所记范数为 float64，1 ULP 对应的相对步长在 **1.11e-16 – 2.22e-16**
  （中位 1.61e-16）；而 capped 各对在任何一位上都没有差异。

**机制（现场实测，不是推测）**：uncapped 时 12 个进程（4 rank + 8 个 dataloader worker）
在 112 个硬件线程上共开 **1,344 个 OS 线程**；cap 9 降到 276，**cap 4 降到 176**（即 48 个
OpenMP 线程对 56 个物理核，留有余量）。拓扑快照见
`.claude/scratch/omp_probe/topology_O1.txt` 与 `topology_X1.txt`。

**由此产生一条启动约束**：`OMP_NUM_THREADS=4` 必须设在**启动 shell** 上，才能经
`torch.multiprocessing.spawn` 传到各 rank 与 dataloader worker；拓扑快照显示 worker 进程确实
继承了该值。只加一半（例如只在 rank 内设）会静默丢掉 43% 的收益，因此启动后必须读
`/proc/<pid>/environ` 逐进程复核，而不是只看启动命令。

### D. 8×256 布局被否，理由是**进程不中性**，不是慢

**缺陷本身**（作者代码，run B 已带，**本 run 刻意不修**）：

- `code/models/infbagel.py:1257-1259`：在 `if cfg_scale is not None:` 之内，
  `if int(timesteps[0]) == 499 or is_uncondition:` 把 `cfg_scale` 整体覆写为 −1，
  作用域是**整个 rank-local batch**，而判据只看**样本 0** 的 timestep。
- `consistency_loss` 传入的是 `start_timestep = solver.ddim_timesteps[randint(0,25)]`
  （`:271-274`），而 `ddim_timesteps=25`、`timesteps=500`（`:1040-1043`），故 `start_timestep`
  只取 25 个离散值、其最大者恰为 499 ⇒ 该分支**以 p = 1/25 每 rank 每 step 触发**。
- 下游 `:1320-1325` 的训练分支读这个已被覆写的 `cfg_scale`
  （`is_uncond = (cfg_scale == -1)`），把 `scene_embs[1:]`——即全部时序 scene embedding——
  对该 rank-local batch 的**每一个样本**置零。
- 叠加 per-rank seeding `seed + rank`（`code/train_infbagel.py:326-329`），被丢掉 scene 条件的
  样本集合就依赖 world size：4 rank 以 512 为块丢，8 rank 以 256 为块丢。**期望比例两者相同**
  （都是 effective_batch/25 = 4%），差别在方差与实现轨迹，不在均值——这点必须写清，否则会被
  误读成两种布局的条件 dropout 率不同。
- 实测后果：同一配置、可证同一数据下，8×256 与 4×512 **自 update 1 即分歧**（全局梯度范数
  相对差 4.6012%），300 个 update 上 1,200/1,200 条范数全部不同（§C 的负对照就是这一组）。

**为什么不以速度为否决理由**：实测 8×256 为 2.30765 s/update，对 4×512 的 1.14273
（两者皆 uncapped），即每 update 慢 **2.02×**，全程 37.61 h / 300.9 GPU-h。但该比较是在
**uncapped** 下做的，而 8×256 **从未在 cap 下重测**。uncapped 时 8-rank 布局要跑 24 个进程
而非 12 个，压在同样的 112 个硬件线程上，正好被 §C 那条 cap 所消除的超订阅病理打得更重，
因此 2.02× 是**被混淆的数字，不能承担布局裁决**。（24 对 12 这个进程数是从启动拓扑推出的，
8-rank 臂没有取拓扑快照，此处标明为推断而非实测。）进程不中性则是布局内在的、与 cap 无关，
所以否决落在它上面。

**不修该缺陷的理由**：它来自 run B 与作者代码。修了之后 C 的目标函数就不再等于产出其
teacher 的那一个，B↔C 的「蒸馏损失了多少」不再可比。故照录、登记、不修。
附带登记一条不触发的相邻风险：覆写用的是 `torch.full((self.batch_size, 1), -1.0)`，取的是
`cfg.batch_size` 而非运行时 batch；训练侧 `DataLoader(..., drop_last=True)`
（`code/train_infbagel.py:370`）保证两者恒等，故本 run 不受影响。

### E. 梯度裁剪：本 run 只测量，不实施

300-update 探针的范数分布（OMP=4 rep，四 rank 汇总 1,200 条 = 300 个全局范数）：
**min 0.0219、median 0.0602、mean 0.0672、max 0.1910、非有限值 0 个**。

**该分布不足以定 `max_norm`，有两条独立原因**：

1. **它整段落在 warmup 之内**。线性 warmup 为 2,000 update，探针只到 300，即这些范数是在
   目标 lr 2e-4 的 **≤15.0%**（300/2000）下产生的。warmup 之后的范数量级无从由它外推。
2. **趋势本身随窗口摇摆，不构成一个统计量**。同一 rep 的 rank 0：update 251–300 均值比
   1–50 均值高 **+18.6%**；换窗口则得 +6.2%（1–25 对 276–300）、+14.4%（1–100 对 201–300）、
   +74.2%（51–100 对 251–300）、**−32.1%**（1–10 对 291–300）。一个能从 −32% 摆到 +74% 的
   量，无法支撑任何阈值。（2026-08-15 §C 第 4 条担心的正是「凭空取 `max_norm=1.0`，若典型范数
   远大于 1 就等于每步静默压低有效 lr」；此处实测典型范数远**小于** 1，若真取 1.0 则几乎从不
   触发，那是另一种无意义。）

**因此本 run 不加裁剪**，与 2026-08-15 §C 第 4 条的「建议加、本节不实施」一致，只是把「不实施」
的依据由「无数据」升级为「有数据但不足以定值」。本 run **全程开启 `+log_grad_norm=true`**，
裁剪的取值决定改由 **update 2,000 之后（warmup 结束）的分布**作出。

裁剪之所以仍值得要的风险不变，一并重记：B 在 146,255 个 update 上实测两次未裁剪梯度事件，
C 只有 58,678 个 update，而 checkpoint 选择是 final-epoch-only。

### F. 仪表：opt-in，关闭时逐位惰性

`code/train_infbagel.py` 增加 `flush_grad_norms` 与一个由 `+log_grad_norm=true` 开启的分支，
**默认关闭**。关闭时的逐位惰性由本日独立复核（脚本 `.claude/scratch/inert_check.sh`，
12 个 update × 两臂，一臂完全不带该 flag）：

- 两臂的 `epoch000.pth` **sha256 完全相同**
  （`b552009d66129a8058e19a5185c1f80da069c57bee5632141acfbca03c035bad`，179,662,353 bytes，
  218 个张量 / 50.01M 元素；可训练面 12.98M）；
- 两臂 `last_loss` 相同至最后一位（`0.0029015124309808016`）；
- `grad_norms/` 只在开启臂出现，关闭臂不留任何痕迹。

**如实登记仪表自身的成本**：开启后 per-rank peak allocated 由 7,604,502,528 增至
7,604,506,112 bytes，即 **+3,584 bytes（3.5 KiB）**——`torch.stack` 那 218 个标量范数。
不是零，但对 7.08 GiB 的峰值无影响，且不改数值（checkpoint 哈希相同已证）。

两处实现选择随之登记：(a) **刻意不用 `clip_grad_norm_(max_norm=inf)` 来取范数**——torch 1.13.1
的实现算 `max_norm / (total_norm + 1e-6)`、clamp 到 ≤1.0 后就地乘每个梯度，`inf/inf` 得 NaN
会静默改写全部梯度；(b) 范数取自 **post-allreduce** 的梯度（循环不用 `no_sync`），即未来
`max_norm` 真正会作用的那个量，且**留在 GPU 上**不做 per-update `item()`，以免同步点吃掉
本节所测的重叠。

> 一处未复现的旧数字：上一会话曾报告该等价性检查的 checkpoint 哈希为 `581aebeb…`、
> 「45M 参数」。其产物未归档，其配置不可知，本日复核得到 `b552009d…` 与 50.01M。**被验证的
> 命题是「两臂相等」，而不是某个绝对哈希值**；本节记录本日自己测到的值，不沿用旧数字。

### G. C 启动的最终参数

| 项 | 值 |
|---|---|
| run id | `p1-hsi-c-lingo-cm-s42-20260816` |
| config | `code/config/config_train_hsi_c_lingo_cm.yaml`（不改） |
| 布局 | 4 GPU（`CUDA_VISIBLE_DEVICES=0,1,2,3`）× micro-batch 512 × accum 1 = effective 2048 |
| 预算 | 58,678 optimizer update / `epochs: 90` / 656 step/epoch |
| 线程 | `OMP_NUM_THREADS=4`，设在启动 shell 上 |
| 仪表 | `+log_grad_norm=true` |
| 裁剪 | 无（见 §E） |
| 预期墙钟 | 10.32 h / 41.3 GPU-h（§A） |
| 预期 per-rank peak alloc | 约 7.08 GiB |
| teacher | `results/hsi_b_lingo_full/checkpoints/hsi_b_lingo_full_epoch222.pth`，sha256 `931a6f1f…48c5e`（本日重校，与 registry 全 64 位相符） |
| `is_mix` | model 与 sampler **两侧均为 false**，从解析后配置读出 |
| preflight | 解析后配置归档于 run 输出目录，未解析插值 **0** 处（`AGENTS.md:189-191`；run B 缺此项，C 补上） |

启动命令（`cd` 到 `code/` 是承重的：`train_infbagel.py:18` 在模块导入期就设
`ROOT_DIR='..'`，故解析后配置里所有路径都相对 `code/`）：

```
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /data/yujinlun/anaconda3/envs/infbagel/bin/python train_infbagel.py \
  --config-name config_train_hsi_c_lingo_cm +log_grad_norm=true
```

必须 detached 启动（`setsid` + `nohup`，日志落在 run 自己的输出目录），因为它必须活过启动它的
会话；启动后须核验进程存活且其父进程在启动 shell 之外。

---

## 2026-08-16（同日修订）：run C 蒸馏完成、gate 工件校验、裁剪决定落定

本节记录 `p1-hsi-c-lingo-cm-s42-20260816` **实际产出了什么**，并把上文 §E 推迟的裁剪取值
决定就地落定。本节写于 `tools/experiment.py finish` / `register` **之后**：manifest 与 registry
行先于本节存在，故本节不构成该 run 的 provenance，只是其读数。封存时 HEAD 为 `8bdc132`、
工作树干净，manifest 记 `status: completed`、`git.commit 8bdc132…`、`dirty: false`，且**无**
`commit_transition` 段——即干净封存，而非哈希绑定的转移记录。registry 由 278 行增至 279 行，
`tools/experiment.py validate` 通过（279 records / 4 splits / 2 evaluators / 1 training protocol）。

### A. 预算：实测 10 h 25 m 01 s，对 10.32 h 超出 0.94%

| 项 | 预注册（§A/§G） | 实测 |
|---|---:|---:|
| 墙钟 | 10.32 h | **10.4169 h = 10 h 25 m 01 s** |
| GPU-h（4 GPU） | 41.3 | **41.7** |
| s/update | 0.63335 | **0.6380**（循环内）/ 0.6391（墙钟÷update 数） |
| optimizer update | 58,678 | **58,678 / 58,678** |

超出 **0.94%**，故 §C 那条 `OMP_NUM_THREADS=4` 的启动约束在 10 小时量级上确实成立——不是
探针窗口里的假象。循环外的启动＋收尾开销实测仅 **62 s**，即墙钟几乎全部是 update 循环本身。

预算按预注册在 **update 数**上收口，而非 epoch 数：58,678 ÷ 656 = 89.45，故
`max_optimizer_updates` 在 **epoch 89 的第 294 / 656 步**停机（2026-08-13 §3 与 §G 已登记此不
整齐）。`epoch089.pth` 因此是 **update 58,678 处的权重**，其落盘走的是
`code/train_infbagel.py:580-582` 的 `epoch == cfg.epochs - 1` 分支，而不是 `ckpt_interval` 分支。
六个 checkpoint（epoch 000/020/040/060/080/089）加 `_resume.pth`，哈希与字节数全部登记于
`results/hsi_c_lingo_cm/metrics.json`。

### B. 数值健康：一次孤立梯度尖峰，无非有限值

仪表全程开启，四个 rank × 58,678 个 update 共 234,712 条 post-allreduce 范数记录：

- **非有限值 0 个**；日志中 `Traceback` / `RuntimeError` / OOM / `Loss: nan` **0 处**。
- 四个 rank 在**每一个 update** 上写下的范数**逐位相同**（最大绝对散布恰为 `0.000e+00`）。这与
  §C 的口径提示一致：独立样本是 58,678 个全局范数，不是 234,712 个。

唯一的梯度事件：

| 项 | 值 |
|---|---|
| 范数 | **0.830289** |
| 位置 | update **28403**（epoch 43，step 195） |
| 邻居 | update 28402 = 0.037076，28404 = 0.034474（前后各两步均在 0.03–0.07） |
| loss 扰动 | epoch 40–42 的 step-190 带 ~0.001710 → epoch 43 step 220 峰值 **0.001908**（**+11.6%**） |
| 恢复 | epoch 43 step 280 已回到 epoch 40–42 的 step-300 带 `[0.001737, 0.001768]` 之内，即约 85–105 个 update；step 300 = 0.001748 |

即单步冲击、无级联、约百步内完全恢复。对照 run B 在 146,255 个 update 上的两次未裁剪事件
（其中 epoch 160 那次代价约 23 个 epoch），C 的这一次**没有可比的代价**。

### C. 裁剪：仍然不加，而且本 run 给出了「不加是对的」的证据

§E 曾把 `max_norm` 的取值决定推迟到 **warmup 之后（update > 2,000）的分布**。该分布现已实测
（56,678 个 post-warmup update）：

| 统计量 | 值 |
|---|---:|
| min | 0.0130 |
| median | **0.0543** |
| mean | 0.0646 |
| **max** | **0.8303**（= §B 的尖峰） |
| **第二大** | **0.7736**（update 28760，epoch 43 step 552） |

反事实计数（post-warmup）：

| `max_norm` | 会裁掉的 update 数 |
|---|---:|
| 1.0 | **0** |
| 0.7736 | 2 |
| 0.5 | 9 |
| 0.4 | 14 |
| 0.2 | 391 |

**结论：不加裁剪。** 理由不是「没数据」，也不是「懒」，而是这批数据直接反驳了裁剪能起作用的
前提：

1. §E 曾担心的 `max_norm=1.0` 果然**一次都不触发**（0 / 56,678），因此它既不会静默压低有效
   lr，也**抓不到那次 0.830 的尖峰**——两头都落空。
2. 更要紧的是，**那次尖峰不是可分离的离群点**。第二大范数 0.7736 与它只差 6.8%，两者又都长在
   一条连续的尾巴上（0.5 以上 9 个、0.4 以上 14 个）。任何低到能抓住 0.830 的阈值（≤0.77）
   同时会裁掉合法 update；而唯一能把它单独挑出来的阈值区间是 (0.7736, 0.830289)，宽度 7%，
   那是对本 run 的过拟合，不是一个可移植的超参。

因此 §E 的「有数据但不足以定值」在本 run 收尾时升级为：**有数据，且数据说不该定值。** 这条
结论绑定的是 C 这个目标函数与这个预算；换 loss 权重或换预算需重测。

### D. 一处必须订正的窗口口径：末 9 个 epoch 是 median 0.0486 / max 0.6437

因为 checkpoint 选择规则是 final-epoch-only，产出 gate 工件的那段窗口是否平静，本身是要登记的。
**本次封存时的重算订正了一个先前口径**：先前报告的 **median 0.0494 / max 0.3954** 并不是末 9 个
epoch，而是 **epoch 72–80**（即以 epoch 80 结尾的 9 个 epoch）的统计量——差了整整九个 epoch。

以 rank 0 的全部范数重算，实测如下：

| 窗口 | n | median | max | argmax |
|---|---:|---:|---:|---|
| **epoch 81–89（真正的末 9 个 epoch）** | 5,542 | **0.0486** | **0.6437** | update 56602（epoch 86 step 186） |
| epoch 72–80（先前误报为「末 9 个」） | 5,904 | 0.0494 | 0.3954 | update 52937（epoch 80 step 457） |
| 全部 post-warmup 尾部 | 56,678 | 0.0543 | 0.8303 | update 28403 |

**该订正不改变结论的方向，但改变了它的强度**：末 9 个 epoch 在 median（0.0486 < 0.0543）与
max（0.6437 < 0.8303）两项上都确实比全程尾部平静，所以「gate 工件产生于平静区间」仍然成立；
但其 max 是 **0.6437 而非 0.3954**，即该窗口内含一个 0.64 量级的事件（epoch 86），比先前的说法
粗糙得多。逐 epoch 的 max 为：81=0.2500、82=0.1243、83=0.5234、84=0.2360、85=0.1794、
**86=0.6437**、87=0.1919、88=0.3246、89=0.3857。不得再引用 0.3954 作为末 9 个 epoch 的 max。

### E. gate 工件 `epoch089.pth` 的校验（只加载与比对，不采样、不评测）

`epoch089.pth`（sha256 `8527b03ae9003c884b0af94fce26f9e9a063f301c39cdfcacb18962a0e20a0d6`，
179,662,353 bytes）是后续约 40 GPU-h 评测要消耗的对象，故先做无 GPU 的加载与张量比对：

- **能加载**：以本 run 归档的解析后配置新建 `models.infbagel.Unet`（CPU），`strict=True`
  加载成功；`strict=False` 下 **missing 0 / unexpected 0**；state_dict **218 个 key**，与 run B
  的 218 一致。39,774,184 个参数元素。
  （口径：218 个 key 对应 **216 个不同张量**——`positional_encoder.pos_encoding` 与
  `embed_timestep.sequence_pos_encoder.pos_encoding` 共享同一 storage，`embedding_language`
  下那一对同理；`state_dict()` 列出两个名字，`named_buffers()` 去重。这不是缺陷。）
- **确实被蒸馏过，不是 teacher 的副本**：与 teacher
  `hsi_b_lingo_full_epoch222.pth`（sha256 `931a6f1f…48c5e`，本日全 64 位重校相符）逐张量比对，
  218 个 key 全部同名同形，无单侧 key：

| 集合 | 大小 | 逐位相同 | 已移动 |
|---|---:|---:|---:|
| 冻结面（112 个冻结参数 + 2 个 buffer + 2 个别名 key） | 116 | **116** | **0** |
| 可训练面（`embedding_input` / `embedding_output` / `transformer` / `out`） | 102 | 2 | **100** |

  可训练面的相对 L2 变化 min 0.0204、**median 0.4099**、max 0.8677——量级在 10⁻¹，是真蒸馏而
  非数值噪声。**反向失效也已检查**：没有任何应当冻结的张量发生移动（116/116 逐位相同）。

- **两个没动的可训练张量已定性，是良性的**：`embedding_output.weight` / `.bias`。
  `embedding_output` 是 `code/models/infbagel.py:1162` 的 `nn.Linear(232, 512)`，
  **除 `__init__` 外没有任何方法引用它**，即它不在 forward 路径上，因而 `grad` 恒为 `None`、
  Adam 从不更新它，于是原样继承 teacher 的值。119,296 / 39,774,184 = **0.30%** 的参数元素是
  这样的死重。这是作者架构里既有的东西（teacher 侧完全相同），也正是 DDP 必须开
  `find_unused_parameters=True` 的原因。与 12-update 抽查时看到的「100/102 移动」是同一现象，
  且在全程 58,678 个 update 之后仍然如此——**它不会自行消失，不是训练不足**。
- **文件互不相同**：`epoch089.pth` / `_resume.pth` / `epoch080.pth` 三者 sha256、inode 互不相同，
  `nlink` 均为 1，均非符号链接，故无误链或覆写。`_resume.pth`（503,287,345 bytes，含
  `optimizer` / `target_model` / `rng_states` 等）其 `['model']` 与 `epoch089.pth` 逐位相同——
  同一时刻的权重，本应如此，而**文件本身是两个不同对象**。
- **末 9 个 epoch 确实改动了权重**：`epoch080` 与 `epoch089` 之间 218 个 key 有 **100 个不同**
  （恰为可训练且非死重的那 100 个），相对 L2 变化 median 0.0380、max 0.2732。

复算脚本与完整表格：`.claude/scratch/seal_c_verify_ckpt.py`、
`.claude/scratch/seal_c_checkpoint_verify.md`、`.claude/scratch/seal_c_gradnorm_analysis.py`。

### F. 训练 loss 的平台期——**只作为观察登记**

epoch 末（step 650）的 loss 均值自 epoch 30 起进入窄带：epoch 30–88 共 59 个值，
min **0.001641**（epoch 85）、max **0.001874**（epoch 31）、median **0.001710**。全程所有 loss
记录的 min/max 为 0.001545 / 0.003005，非有限值 0 个。

**不得**把这段平台期读成「后半程预算白花」。本项目已实测 held-out 去噪 loss 与原生 rollout
指标**反相关**，训练 loss 的平坦度对 rollout 质量没有已建立的预测力；§D 的 epoch080→089 权重
位移（median 0.0380）也说明模型在平台期内仍在实质移动。后半程预算是否值得，只能由 §G 那次
原生域评测回答。

### G. 下一步（需用户批准后才能启动）

按 2026-08-13（同日修订）§D 的预注册协议，在 LINGO v3 test 上对 `epoch089.pth` 做
guided / unguided 双列评测，**gate 读 unguided 列**。那是**另一个 workload**，需要自己的 run id
与 registry 行，且需用户明确批准方可启动。本节不启动它，也不预判其结果。




## 2026-08-16（同日第二次修订）：FPS 协议拆分、评测分片，以及串行方案被用户否决

本节由用户的一次明确否决触发，并**取代** 2026-08-12（同日第二次修订）§E 中
「FPS 与几何指标共用同一次采样（batch=1）」这一句所隐含的**串行强制**。原文保留、不删除，
但自本节起不再作为启动约束。

### A. 用户决定：不为「FPS 协议纯粹」多付约 50 h 墙钟

用户的判断，原样登记：串行的唯一理由是让 FPS 口径与 infbagel 对齐，而为这份「协议纯粹」
多花 50 多小时墙钟，性价比不成立；并且**对 B 的实时性评估本来就没那么重要——已知蒸馏就是
为了解决 B 推理慢的问题**。

这条理由推翻了本 session 之前给出的反对意见。那条反对意见是「B 的 FPS 是蒸馏加速主张的分母」，
它站不住，因为该主张的机制量是 `denoiser_calls_per_window`（B 1000 / C 16），
**与硬件无关、也与是否分片无关**；B 的 latency 只需要「够得上被引用」，不需要全量协议纯度。

### B. 拆分后的口径

- **质量列**：8 卡分片，episode 级切分，只出几何指标。该模式下一切时间量
  （`aits` / `avg_fps` / `aggregate_fps` / `rtf` / `per_window_wall_seconds` /
  `sampling_seconds` / `total_generation_seconds` 及其 per-scene 均值）**必须写成 `null`
  并带显式 `timing_valid: false` 标记**。8 进程互相争用，这些数字是被污染的；把它们照常填进
  payload 是本项目已经犯过一次的错误类型——一个「看起来权威」的污染数字，日后会被引用。
- **latency 列**：单卡串行、静默主机、**跨 scene 的固定确定性子集**，是 FPS 的唯一来源。
  逐窗口成本近似恒定（模型输入固定为 `[1,16,232]`），但 penetration 分支随 scene 变化，
  因此子集必须跨 scene 取，而不是取前 N 个 episode。

### C. 分片带来的三条硬要求（启动前必须验收）

1. **按窗口数而非 episode 数做负载均衡**。窗口数是各 episode JSON 的 `episode_num`：
   375 episodes、合计 **2271** 窗口、mean 6.06、median 4、**min 2、max 55**。按 episode
   轮转会让某个 shard 独自扛下若干个 55 窗口的 episode，而全局成本等于最慢的那个 shard。
2. **逐 episode 确定性播种，且用「全量枚举下的 canonical ordinal」而非 shard 内序号**。
   现状是全程只 `seed_everything(cfg.seed)` 一次、各 episode 共享一条 RNG 流；分片后每个进程
   会消费**不同的子序列**，结果就会依赖切分方式。验收标准是 1 shard 与 2 shard 的逐 episode
   payload **逐位相同**，且必须附一个「本应失败」的对照（例如换 seed）——本仓库已有过
   在空 glob 上打印 `IDENTICAL (0 values)` 的空洞 PASS，只被一个没能失败的负对照抓住。
   该改动会让 B 的数值不同于「假想的串行单播种 run」，这是可接受的：guided HSI 评测从未
   跑完过，不存在需要保持可比的基线。但因此它必须**无条件生效**，B / C 的四个 cell 同制。
3. **merge 必须大声失败**。episode 数须等于 375、窗口数须等于 2271，重复键、缺失 ordinal、
   缺 shard 一律 raise，而不是把 7/8 合成一份看起来完整的结果；`excluded_as_warmup` 必须按
   canonical ordinal 判定，否则 8 个 shard 会标 40 个 warmup 而串行只标 5 个。

### D. 实测成本（取代此前的解析估算），以及一处必须订正的数字

`p1-hsi-b-eval-epoch222-s42-20260816` 启动前的 ABBA 探针，seed-42 两臂各 2 rep，单卡：

| 口径 | OMP 未设 | `OMP_NUM_THREADS=4` |
|---|---:|---:|
| 逐窗口 | 61.195 / 62.193（mean 61.694） | 60.167 / 61.354（mean 60.761） |
| 单趟（2271 窗口，RDS 已 gate） | 38.92 h | 38.33 h |
| 双趟（RDS 未 gate） | 77.04 h | 74.60 h |

**订正**：本 session 先前向用户报出的「OMP=4 时 59.125 s/window、加速 1.033×」是错的。
该数字的来源需要二次订正：它**不是** seed 43 负对照臂的实测值（那一臂实测 59.211），而是探针
`extrapolation.txt` 里的 tqdm 进度条算术——B1/B2 的 473 s bar 总量除以 8 次窗口采样。两者数值
接近但来源不同；**manifest 是权威**，seed-42 两臂的实测 `per_window_wall_seconds` 是
60.167 / 61.354。仅取 seed-42 两臂，比值是 **1.0154×，且两臂区间重叠**
（未设的最快 61.195 低于设了的最慢 61.354）。因此**训练路径上的 1.771× 并不迁移到评测采样路径**：
run-total 墙钟那 17.7 s 差距里只有约 7.5 s 落在采样上，其余约 10 s 是一次性启动开销。
该上限仍然保留——它已验证逐位中性（比对 126 个值，负对照在 56 个值上不同）、且免费——
但预算时应按「从约 40 h 里省下不到 1 h」计，不是省一半。

### E. 已作废的 run

`p1-hsi-b-eval-epoch222-s42-20260816` 以串行口径启动，跑完 19 / 2271 窗口后按本节决定终止，
manifest 以 `status: aborted` 封存（`git.commit` = `final_git.commit` = `6225429`，
无 `commit_transition`），未产出任何 metric。payload 只在结束时写出，因此没有可用的部分结果。
分片版本将以新的 run id 启动，并同样引用 `p1-hsi-b-eval-preregister-s42-20260815`，
但须在 `closes_preregistration` 之外显式登记本节对 §E 的修订。

### F. 一个与本节无关但已在此前修好的启动阻塞

`6225429` 修复了 `code/datasets/infbagel_mix.py` 借用 `InfBaGelDataset` 未绑定方法的失效：
`087848f` 把 `get_nearest_free_voxel` 改成派发到 `self._get_nearest_free_voxel_direct`，
而 `InfBaGelMixDataset` 并不继承 `InfBaGelDataset`，于是**任何 guided HSI 评测都会在第一次
施加 guidance 时 AttributeError**——包括 Phase 1C 的 gate（C + guided）本身。当时无任何测试
调用过该方法，evaluator 的 `hasattr` 前置检查还会通过。direct 与 materialized 两条实现在真实
400×100×600 网格、53.7% 查询穿透、位移达 0.31 m 的条件下**逐位一致**，故 `087848f` 没有改变
guidance 几何，只是打断了派发。

### G. `:1257` 与 `:1316`：同一判据的两条分支，训练侧与评测侧必须分开记

本节 §D（2026-08-16）记的是**训练侧**机制，经复核**准确、不需修改**。这里补评测侧的实测澄清，
因为它决定了分片逐位一致性必须在哪个步数下验证。

- **评测侧 diffusion：`:1257` 整块不可达。** `p_sample` 的 conditional / unconditional 两次调用
  （`code/models/infbagel.py:924` / `:926`）**不传 `cfg_scale`**，故 `if cfg_scale is not None:`
  为假。仪表实测：500 步下 8000 次 `Unet.forward` 全部 `cfg_scale=None`，`:1259` 的整批覆写
  在**任何 `sample_type=diffusion` 的 cell、任何步数下都不触发**。
- **评测侧真正活跃的是 `:1316`。** 它在 `if is_sample:` 之下带有与 `:1257` **字面相同**的判据
  `int(timesteps[0]) == 499 or is_uncondition`，把 `scene_embs[1:]` 置零。实测 500 步下触发
  8 次 conditional（每次窗口采样的 t=499 首步）加全部 3992 次 unconditional；
  **100 步下 conditional 触发 0 次**——DDIM 网格根本到不了 499。因此「在 100 步下证明的逐位
  一致性」结构性地绕开了一条在 500 步下活着的分支，500 步重测是必需的而非保险。
  重测结果：**159/159 逐位一致**（3 episode / 13 窗口，seed-43 负对照 69/159 不同）。
- **C 的 cell 与 B 不同，但仍然无害。** consistency 调用（`:653`）传 `cfg_scale=w` 且
  `is_sample=True`，故 `:1257` 在 C 的评测中**可达**。但评测恒为 `batch_size=1`
  （evaluator 对 `batch_size != 1` 硬拒），「样本 0 替整个 rank-local batch 决定」在单样本下是
  **空的**，批粒度缺陷在评测中不可能发生；分支按判据触发是设计行为，不是缺陷。
  实测 `batch_sizes_observed: [1]`。
- **不要把上一条读成「该缺陷是评测期的」。** 训练侧 `consistency_loss`（`:309`）传 `cfg_scale=w`
  且**不传 `is_sample`**（`:1218` 默认 `False`），于是 `:1257`→`:1259` 先把整批 `cfg_scale`
  覆写为 −1，`:1319` 的 `elif cfg_scale is not None` 再「按样本」读这个已被污染的向量——
  两条分支都触发，per-sample mask 只是忠实地施加一个已经整批错掉的向量。这正是 §D 的记载，
  也正是 8×256 布局不中性的机制；本节不推翻它，只是把评测侧与训练侧分开。

**结论口径**：批粒度缺陷是**训练专有**的（`is_sample=False` 且 batch > 1）；评测两个 sampler
都不受它影响。仍然**不修**——理由同 §D：修了 C 的目标函数就不再等于产出其 teacher 的那一个。
