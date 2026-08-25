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

**穿透**（全部同时报告，均需 **GT 参考行**——LINGO Tab.2 没有，我们加。方向全部为 ↓）
1. `pen_ratio` = SDF < −3 cm 的「顶点×帧」比例（TeSMo 阈值）。无量纲分数。
2. `pen_depth_mean` = 仅对穿透顶点取 |SDF| 均值，单位 m（TeSMo）。`pen_depth_max` 同理取 max。
   两者在穿透集为空时回填 `0.0`——这是已封口语义，不得改动，`pen_value` 是其阈值 0 的
   非回填对应量。
3. `pen_burst` = `100 × mean_t[(每帧穿透顶点比例)²]`（Dyn-HSI Eq. 9）。平方项是刻意的
   超线性，使一帧灾难性穿透不被长序列稀释——正对自回归 rollout 的突发型失败。
4. **（2026-08-17 增）** `pene_pct_scene` = LINGO `Pene%_scene` = 逐帧「SDF < 0 的顶点数 ÷
   采样体顶点数」再对帧取均值。**阈值 0，是 [0,1] 的分数而非百分数**（键名沿用 LINGO 列名
   以便对照）。GT 参考：全 375 条 **0.05044**，walk 130 条 **0.03358**。
5. **（2026-08-17 增）** `pen_value` = TeSMo `Pen. Value` = 仅对 SDF < 0 的「顶点×帧」取
   |SDF| 均值，单位 m。**阈值 0**；集合为空时输出 `NaN` 而非 `0.0`（评测 harness 会丢弃
   非有限值），使「没有穿透」不被读成「穿透极浅」。GT 参考：**0.03416 / 0.02480**。
6. **（2026-08-17 增）** `pene_sum_mean_floorexcl` / `pene_sum_max_floorexcl` = DIMOS 逐顶点
   求和式的**排除地面接触**版本：逐帧对 `y ≥ 地面 + 2 cm` 的顶点求 `Σ_v |min(SDF_v, 0)|`，
   再对帧取 **mean** / **max**。阈值 0；单位是**「对顶点求和的米」——一个外延量，不是深度**，
   随采样体分辨率缩放，只能与同采样体的数字对比；`pene_sum_max_floorexcl` 是最差**帧总和**，
   不是最深顶点（那是 `pen_depth_max`）。GT 参考：全 375 条 **5.973 / 19.376**，
   walk 130 条 **0.449 / 5.919**；walk 均值为 LINGO 已发表 0.402 的 1.12 倍。
7. **（2026-08-17 增）** `pene_samples` = SDF < 0 的「顶点×帧」计数，即 `pen_value` 的分母。
   **是计数而非分数**，与 `pen_samples` / `pen_sample_frames` / `skate_frames` 同属分母类，
   随采样体分辨率与序列长度缩放（实测单序列 22530–957047），因此**不是可比列**；它存在的
   目的是让读者看出 `pen_value` 由多少样本支撑。注意它同样进入 `scene_summary.metrics_mean`，
   那里的场景均值也是外延量，不得当作指标解读。
8. **（2026-08-17 删）** ~~`pen_frame_ratio`（含至少一个穿透顶点的帧比例，TeSMo `Pen. Ratio`
   的字面形式）~~——实测已饱和，见下方 2026-08-17 节。

> **2026-08-12 原文（已被上文第 4–8 条取代，保留以便追溯）**：「不采用 DIMOS 的逐顶点求和式。」
> 该判断的三条理由中有两条仍然成立（随网格分辨率缩放；`Pene_max` 不是最深顶点），已改为在
> 单位与文档中写明而非弃用；第三条「它是 B 节四量同名问题的最大来源」由**排除地面接触**这一
> 附加限定解决：实测 GT 上 92.07%（walk）／68.33%（全split）的穿透质量位于地面 +2 cm 以下，
> 即未排除的求和式在 GT 上主要度量的是脚踩地面。

**scene 必须作为 paired bootstrap 的分层因子**：实测 GT
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
1. **（2026-08-18 定版）** `fs_nemf` — LINGO 引用的 NeMF FS：`s = v·(2 − 2^(h_eff/H))`，
   H = 4 cm（趾）/ 8 cm（踝），位移 `v` 取 **L2**（`sqrt(dx²+dz²)`），对**四个足关节取平均**，
   除以 **T−1** 个转移，×100，单位 cm/frame。**不做任何预平移**，高度是相对精确地面 y = 0 的
   绝对值；权重中的高度取 `h_eff = max(h, 0)`（见下方 2026-08-18 节 B），而**入带判定仍用原始
   `h < H`**，故 sub-floor 帧仍算接触帧。`fs_nemf_ankle` / `fs_nemf_toe` 是可加分解（各自也除以
   4，故等于该组占四关节均值的份额，**不是**该组自身的两关节均值）。
   > **2026-08-12 原文（已被取代，保留以便追溯）**：「位移取 **L1** 且**求和**不取平均，序列先
   > 平移使最低足高为 0」，且分母为 T。四处偏离与实测倍数见 2026-08-18 节 A。
2. `skate_ratio` — GMD/TeSMo：足高 < 5 cm 且单帧滑动 > 2.5 cm 的帧比例。该阈值**不是
   帧率无关的**，30 fps 下须换算为 0.75 m/s 后使用。
3. 现有 `compute_foot_sliding_for_smpl`，仅为与 HOI 表同口径而保留。**注意 2026-08-18 之后
   `fs_nemf` 与它的差别只剩「软权重 vs 硬门限」与关节/阈值口径，L2 与取平均这两点已一致。**

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

---

## 2026-08-17：穿透列扩充为可与 LINGO/TeSMo/DIMOS 对照的形式，`pen_frame_ratio` 下线

本节只改**指标定义与报告口径**，不改任何模型、采样器或训练配置；已就地施行于 §C「穿透」
（第 4–7 条），原文以引用块保留。**本节没有跑任何 model cell**（B/C、guided/unguided 全无），
只跑了 ground-truth cell。

### A. 为什么改：原有三列无法与文献对照，第四列已饱和

标定探针（`.claude/scratch/dimos_pen_form.py`，全 375 条 GT，验证闸门对已封口值的相对偏差
≤ 2.2e-16）实测出各已发表形式在本项目 GT 上的取值：

| 量（阈值 0，除注明） | 全 375 均值 | walk 130 均值 |
|---|---|---|
| DIMOS 求和式 pen_mean（m，对顶点求和） | 20.452 | 9.161 |
| DIMOS 求和式 pen_max | 37.962 | 17.525 |
| **DIMOS 求和式，排除地面，mean** | **5.973** | **0.449** |
| **DIMOS 求和式，排除地面，max** | **19.376** | **5.919** |
| **LINGO `Pene%_scene`（分数）** | **0.05044** | **0.03358** |
| **TeSMo `Pen. Value`（m）** | **0.03416** | **0.02480** |
| TeSMo `Pen. Ratio` @−3 cm | 0.9960 | 0.9911 |

两条结论：

1. **`pen_frame_ratio` 必须下线。** 它是 TeSMo `Pen. Ratio` 的字面形式，GT 实测 0.9960 /
   0.9911，已饱和；而 TeSMo 自己发表的是 0.0611 / 0.1076。它既无区分度，也无法充当被保留时
   所声称的「对照列」。
2. **排除地面后的 DIMOS 求和式是可用的。** walk 的 0.449 为 LINGO 已发表 0.402 的 1.12 倍——
   这是本项目第一个与 LINGO 数量级对齐的穿透列。未排除地面的 9.161 是 22.8 倍，即 2026-08-12
   「不采用 DIMOS 求和式」的判断在**未排除地面**的前提下是对的。

### B. 地面排除带的高度不是刀刃（本节新测，n=375 与 walk 130）

GT 顶点在**每一条**序列上都低于 y = 0（逐序列最低点的范围 −0.163 m 至 −0.034 m），所以
「地面 + 2 cm」是否把小腿也切掉是一个真问题。实测扫描（`.claude/scratch/floorband_followup.py`）：

| 带高 (m) | 全 375 mean | walk 130 mean | walk / LINGO 0.402 |
|---|---|---|---|
| 0.00 | 6.131 | 0.672 | 1.67x |
| 0.01 | 5.992 | 0.473 | 1.18x |
| **0.02（定版）** | **5.973** | **0.449** | **1.12x** |
| 0.03 | 5.971 | 0.447 | 1.11x |
| 0.05 | 5.970 | 0.446 | 1.11x |
| 0.10 | 5.966 | 0.444 | 1.10x |

穿透质量的高度分布：全 375 为 y<0 占 67.43%、[0, 2 cm) 占 0.90%、y≥2 cm 占 31.67%；
walk 130 为 89.11% / 2.96% / 7.93%。**带高在 ≥ 1 cm 后是平的**（walk 在 0.01–0.10 之间只
变动 ±6%，对照倍数落在 1.10x–1.18x），减量几乎全部来自「排除 y<0」而非带本身。因此 2 cm
落在平坦区，1.12x 这个对照结论对带高选择稳健；**唯一敏感的一步是 0 → 1 cm**，即必须排除
sub-floor 质量，单纯取 y ≥ 0 会给出 0.672（1.67x）。

### C. 验收（三个闸门，全部通过）

1. **与标定探针逐序列一致**（n=375，经新 `metrics.py` 代码路径）：`pene_sum_mean_floorexcl`
   与 `pene_sum_max_floorexcl` 375/375 **逐位相同**；`pene_pct_scene` 最大相对偏差 2.18e-16，
   `pen_value` 3.68e-16（均为 float64 求和顺序，探针按 256 帧分块累加，新代码整条张量一次归约）。
2. **保留指标未被扰动**：v1 episode 集重跑 GT，44 个数值指标 × 375 条与已封口
   `results/lingo_hsi/ground_truth/evaluation/per_sequence_metrics.json` **全部逐位相同**
   （`pen_ratio` 0.0280519 / `pen_depth_mean` 0.0493719 / `pen_depth_max` 0.0997523），
   非数值键（含全为 null 的 `goal_orientation_err_rad`）亦全部相同。唯一差异是键集：
   删除 `pen_frame_ratio`，新增四个键。
3. `pytest tests/hsi tests/core tests/test_research_governance.py`：**160 passed, 1 skipped**，
   与基线相同。无测试断言 `pen_frame_ratio`，故无测试需要改。

### D. 显存代价（评测侧，需在 model cell 前留意）

最长 GT 序列（`046-new_loco`，2316 帧 × 10475 顶点）整条 `compute_metric_record` 的实测峰值为
**4492 MiB**（前置常驻 323 MiB，device reserved 5190 MiB / 24576）。新增的两个 `torch.where`
中间场在该形状上的微基准为 **+569 MiB 峰值**（保留族 263 MiB → 新增族 832 MiB）与
+2.5 ms/序列。若某个 model cell 因此触顶，逐帧分块计算该求和是**逐位等价**的降峰值改法——
探针本身就是按 256 帧分块算的，而它与整条张量的结果 375/375 逐位相同。

### E. GT v2 参考行（`pelvis_goal` = rollout 终点）

以 `lingo_episode_dir=$ROOT_DIR/data/lingo_hsi_test_v2/data` 显式覆写跑出，配置默认仍指向 v1
（用户决定：在 model 重跑之前不改默认值）。输出在 `results/lingo_hsi/ground-truth-v2/`。
`sequence_count` 375、`scene_count` 26。v1→v2 除目标族外**全部逐位相同**，这正是应当发生的
（同一段运动、同一批场景，只有目标点变了）。

| 指标 | 单位 | 全 375 | walk 130 |
|---|---|---|---|
| `pene_pct_scene` | 分数 ↓ | 0.0504436 | 0.0335772 |
| `pen_value` | m ↓ | 0.0341575 | 0.0247974 |
| `pene_sum_mean_floorexcl` | m（对顶点求和）↓ | 5.97327 | 0.449037 |
| `pene_sum_max_floorexcl` | m（对顶点求和）↓ | 19.3765 | 5.91913 |
| `pen_ratio` | 分数 ↓ | 0.0280519 | 0.0128566 |
| `pen_depth_mean` | m ↓ | 0.0493719 | 0.0417524 |
| `pen_depth_max` | m ↓ | 0.0997523 | 0.0759019 |
| `fs_nemf` | cm/frame ↓ | 0.567552 | 1.10467 |
| `last_dist` | m ↓ | 9.19e-11 | 1.93e-10 |
| `min_dist` | m ↓ | 7.20e-11 | 1.36e-10 |
| `success_last_10cm` | 比率 ↑ | 1.000 | 1.000 |
| `success_last_20cm` | 比率 ↑ | 1.000 | 1.000 |
| `success_min_10cm` / `_20cm` | 比率 ↑ | 1.000 / 1.000 | 1.000 / 1.000 |
| `goal_planar_err_m` | m ↓ | 9.19e-11 | 1.93e-10 |
| `goal_height_err_m` | m ↓ | 0.76137 | 0.940565 |
| `jerk_ratio` | 比值 → | 1.19363 | 1.21004 |

`last_dist` 不是**字面**的 0：368/375 条恰为 0.0，其余 7 条为 O(1e-9) m（最大 7.45e-9 m），
即 float32 往返精度，不是口径误差。`success_last_20cm` 与 `success_last_10cm` 双双 1.000。

**v1 的目标口径确实是坏的，并且坏在与预期相反的方向上**：v1 GT 的 `time_to_goal_20cm_s`
均值仅 0.0484 s（约第 1.5 帧），`success_min_20cm` 为 1.000 而 `success_last_20cm` 仅 0.531，
`last_dist` 0.403 m——即 v1 的 `pelvis_goal` 落在轨迹**起点附近**，GT 在开头就「到达」然后走开。
v2 修正后 `time_to_goal_20cm_s` 升到 2.52 s，这是目标移到终点的直接推论。`goal_height_err_m`
（0.761）与 `jerk_ratio`（1.194 / 中位 1.171）两列不受目标点影响，与已登记的 GT 参考值一致。

---

## 2026-08-18：`fs_nemf` 改为忠实于 NeMF 的定义，payload `schema_version` 升到 2

本节只改**指标定义与报告口径**，不改任何模型、采样器或训练配置；已就地施行于 §C「足部」
第 1 条，原文以引用块保留。**本节没有跑任何 model cell**（B/C、guided/unguided 全无），
只跑了 ground-truth cell。审计脚本与逐序列产物在 `.claude/scratch/fs_nemf_audit/`、
`.claude/scratch/fs_faithful_gates.txt`、`.claude/scratch/fs_faithful_falsify.txt`。

### A. 四处偏离与各自的实测倍数

已封口的 `fs_nemf` 相对 NeMF（He et al. 2022，即 LINGO `FS` 列所引）有四处偏离。逐处隔离
测量（全 375 条 GT，ratio-of-means；标定闸门：独立复现已封口 payload，最大绝对偏差
0.000e+00）：

| 偏离 | 已封口 | 定版 | GT 全 375 倍数 | 方向 |
|---|---|---|---|---|
| 水平位移量 | L1 `\|dx\|+\|dz\|` | **L2 `sqrt(dx²+dz²)`** | 1.263x | 虚高 |
| 四足关节归约 | 求和 | **取平均** | 4.000x | 虚高 |
| 分母 | `T` | **`T−1`**（实际求和的转移数） | 0.994x | 虚高 |
| 高度基准 | 先平移使最低足高为 0 | **不平移，绝对 y = 0** | ÷2.458x | 压低 |

净效应 `1.263 × 4.000 × 0.994 ÷ 2.458 = 2.043`，即已封口值是纯 NeMF 形式的 2.043 倍
（全 375）／2.633 倍（walk 130）。加上下节的 clamp 后，已封口／定版 = **2.186x**（全）／
**2.706x**（walk）。

L2 的依据是文本读法与仓库谱系两条，**不是**标定：NeMF 把位移投影到水平面后取「the
**magnitude** v」，二维向量的 magnitude 就是 L2；`code/eval_metrics.py:compute_foot_sliding_for_smpl`
也用 L2。

### B. 为什么是 clamp，而不是纯 NeMF 形式

纯 NeMF 在原始 `h` 上取权重，`h → −∞` 时权重趋于 2：**同样的水平位移，穿得更深的脚趾会被
判为更严重的 foot skate**，即足部列会部分度量穿透——而穿透已有自己的四列。定版把权重中的
高度 clamp 为 `h_eff = max(h, 0)`，权重上界回到 1，耦合消失。

三条支持：

1. **只能减、不能增**：`h_eff ≥ h` 且权重对高度单调递减。GT 上 0.2777 → 0.2597（全）、
   0.4195 → 0.4082（walk）。
2. **在无 sub-floor 足的数据上可证为恒等**：实测 17 条最低足关节高度 ≥ 0 的 GT 序列上，
   clamped − unclamped **恰为 0.000e+00**；`tests/hsi/test_metrics.py` 用精确相等
   （`assertEqual`，非 `assertAlmostEqual`）把这条钉住。
3. **入带判定仍用原始 `h`**，故 sub-floor 帧仍算接触帧、其滑动仍被计入——若改用 `h_eff`
   判定，穿透中的脚滑动会变成免费的，与意图相反。（`H > 0` 时两者等价，代码仍写出原始形式，
   使日后任何让它们不等价的改动必须先与该注释争论。）

**代价要写明**：去掉预平移后，悬空 30 cm 的 rollout 在 `fs_nemf` 上得 0。`skate_ratio`
因绝对 5 cm 门限有同一盲区，**故足部两列都抓不到悬空**；抓它的是 engagement 的
`contact_count` 与 `goal_height_err_m`，这正是 §C 要求三者同表的原因。

### C. 两条不可丢失的结论

**1. 任何已封口 `fs_nemf` 值都不允许用标量换算成新口径。**
预平移不是标量：它**重排序列**（Spearman(已封口, 定版) = 0.522 全 / 0.212 walk），逐序列
倍数跨 **0.164x–7.169x**（中位 2.605x）。而且倍数是数据相关的：GT 的 ratio-of-means 是
2.186x，但**足从不低于 y = 0 的序列反而拿到更大的倍数**——本次实测这 17 条为 5.006x–7.169x
（ratio-of-means 5.752x），因为对悬空序列预平移是把序列**下移**、从而**虚高**已封口值。
（顺带纠正一处易犯的近似：GT 中**没有任何一条**序列的最低足高恰为 0，这 17 条落在
0.000247–0.009535 m，全 375 条的范围是 −0.1433 至 +0.0095 m，故「无 sub-floor ⇒ 预平移是
恒等 ⇒ 倍数恰为 5.023x」是错的，5.023x 只是该组的下界附近。剔除已知的 `4·T/(T−1)` 后，这
17 条的残差倍数中位 1.419、最大 1.782，已超过 L1/L2 各向异性的上界 `sqrt(2) = 1.414`，这本身
就证明预平移在该组里在做虚高。）
**结论**：已封口的 model 数字只能靠**从运动重算**来更新，绝不能换算。

**2. 一个必须记录下来的巧合，以免日后有人「发现」它。**
在 clamp 下，落在 NeMF 已发表 GT 0.512 上的是被**否掉**的 **L1** 读法（0.5141，差 0.4%），
而选定的 L2 给 0.4082（低 20%）。L2 是按 A 节的文本读法与仓库谱系选的，**不是**按这个标定
吻合选的。两个不同 mocap 语料之间 0.4% 的吻合，更可能是运气而非信号；单一已发表聚合值
不能裁定 L1 与 L2。

### D. `schema_version` 1 → 2

2026-08-17 的穿透改动**可以**从键集本身看出来（新增四键、删除 `pen_frame_ratio`）；本次 FS
改动**改的是三个已存在键的含义、不动键集**，故带着它的 payload 与已封口 payload 在结构上
无法区分，而 `fs_nemf` 却相差约 2–7 倍。因此 `code/test_infbagel_lingo_hsi.py` 新增
`METRICS_SCHEMA_VERSION = 2`，两处 payload 共用。

- **schema 1** = 2026-08-12 首次封口的指标集。
- **schema 2** = 2026-08-17 的穿透键集 **＋** 2026-08-18 的忠实 FS 定义。

**只在 `schema_version` 相同的行之间比较 `fs_nemf`。**
一处需要留意的既有产物：`results/lingo_hsi/ground-truth-v2` 写于 `pene_samples` 落地之前，
因此它带着 2026-08-17 的穿透新键但缺 `pene_samples`，且自称 `schema_version` 1。它保持**字节
不变**，作为可追溯的旧 FS 行；新的可比 GT 行是 `ground-truth-v3`。

### E. 验收（五个闸门，全部通过）

1. **生产路径复现审计**（n=375，经 `code/priors/hsi/metrics.py`）：`fs_nemf`、`fs_nemf_ankle`、
   `fs_nemf_toe` 对审计的 `nemf_clamped` 375/375 **逐位相同**（最大相对偏差 0.000e+00）；
   均值 0.259674（全）／0.408211（walk），命中目标 0.2597 / 0.4082。分解可加性
   `|ankle+toe−total|` 最大 1.11e-16。
2. **保留指标未被扰动**：v1 episode 集重跑 GT（写入 `.claude/scratch/gt_v1_fs_faithful`），
   48 个数值键 × 375 条中**只有** `fs_nemf` / `fs_nemf_ankle` / `fs_nemf_toe` 移动，其余
   45 键逐位相同；键集无增删；`scene_summary` 键集无增删。对**已封口** schema-1 v1 payload
   的 40 个非 FS 键做同样比较，同样全部逐位相同。唯一 payload 字段变化是
   `schema_version` 1 → 2。
3. **独立表达式反证**（26 条跨全部 26 个场景的序列）：用纯 Python 显式双重循环（不用 torch
   归约、掩码或向量化 sqrt）重算忠实定义，与生产值最大相对偏差 **4.565e-15**（float64 求和
   顺序）。这一步的目的是切断「生产实现与审计实现是同一个人写的同一种 torch 表达式」这一
   共因。
4. `pytest tests/hsi tests/core tests/test_research_governance.py`：**172 passed, 1 skipped**
   （基线 166 passed / 1 skipped；FS 用例由 4 个扩为 10 个，净增 6，覆盖：L2 非 L1；取平均非求和且除数是实际关节数；T−1 使
   等速滑动与序列长度无关；clamp 在无 sub-floor 输入上精确恒等；clamp 在 sub-floor 输入上
   确实生效并给出闭式值；sub-floor 帧仍算接触帧；以及悬空序列现在得 0 且由 engagement 兜住）。
5. **GT v3 已产出**（下节），`ground-truth-v2` 字节未动。

### F. GT v3 参考行（v2 episode 集 = `pelvis_goal` 取 rollout 终点；`schema_version` 2）

以 `lingo_episode_dir=$ROOT_DIR/data/lingo_hsi_test_v2/data` 显式覆写跑出，配置默认仍指向 v1
（用户决定：在 model 重跑之前不改默认值）。输出在 `results/lingo_hsi/ground-truth-v3/`。
`sequence_count` 375、`scene_count` 26。与已封口 `ground-truth-v2` 相比，**只有三个 FS 键不同**
（另加 v2 缺失的 `pene_samples`），这正是应当发生的。

| 指标 | 单位 | 全 375 | walk 130 |
|---|---|---|---|
| `pene_pct_scene` | 分数 ↓ | 0.0504436 | 0.0335772 |
| `pen_value` | m ↓ | 0.0341575 | 0.0247974 |
| `pene_sum_mean_floorexcl` | m（对顶点求和）↓ | 5.97327 | 0.449037 |
| `pene_sum_max_floorexcl` | m（对顶点求和）↓ | 19.3765 | 5.91913 |
| `pene_samples` | 计数（外延量，非可比列） | 127208 | 112041 |
| `pen_ratio` | 分数 ↓ | 0.0280519 | 0.0128566 |
| `pen_depth_mean` | m ↓ | 0.0493719 | 0.0417524 |
| `pen_depth_max` | m ↓ | 0.0997523 | 0.0759019 |
| **`fs_nemf`** | **cm/frame ↓** | **0.259674** | **0.408211** |
| `fs_nemf_ankle` | cm/frame ↓ | 0.0918998 | 0.136807 |
| `fs_nemf_toe` | cm/frame ↓ | 0.167775 | 0.271404 |
| `last_dist` | m ↓ | 9.18905e-11 | 1.93429e-10 |
| `min_dist` | m ↓ | 7.20223e-11 | 1.36116e-10 |
| `success_last_10cm` / `_20cm` | 比率 ↑ | 1.000 / 1.000 | 1.000 / 1.000 |
| `success_min_10cm` / `_20cm` | 比率 ↑ | 1.000 / 1.000 | 1.000 / 1.000 |
| `time_to_goal_20cm_s` | s | 2.51973 | 5.29231 |
| `goal_planar_err_m` | m ↓ | 9.18905e-11 | 1.93429e-10 |
| `goal_height_err_m` | m ↓ | 0.76137 | 0.940565 |
| `jerk_ratio` | 比值 → | 1.19363 | 1.21004 |

对照：`fs_nemf` 的 walk 值 0.408 是 NeMF 已发表 GT 0.512 的 0.80 倍——同一数量级，但如 C.2
所述不可用作口径裁定。踝 : 趾 = 0.0919 : 0.1678（全），趾部贡献约为踝部的 1.83 倍，与
「趾的 H 更小、但趾更常处于低位」一致。

---

## 2026-08-18（同日第二次修订）：Interactive 穿透口径定版（A 档＋配对差）、28 关节槽位不一致登记、忠实 FS 的悬空盲区

本节只做报告口径的定版与两处登记，**不改任何模型、采样器、训练配置或指标实现**，也**没有跑
任何 model cell 或 GT cell**——三项引用的数字全部来自此前已完成的实测。三项互相独立：A 是用户
已批准的 Interactive 列口径；B 是一处用户决定「只记录、暂不修」的常量不一致；C 是上一节（忠实
FS）引入的一个盲区与一份**尚未确认、也从未执行**的指标删除清单之间的直接冲突。

### A. Interactive 列的穿透口径：绝对列取 A 档，并列同 episode 的配对差（用户已批准）

**决定**：Interactive 指标组报告 floor-excluded 的绝对对
`pene_sum_mean_floorexcl` / `pene_sum_max_floorexcl`（地面 + 2 cm 排除，已实现），**并在其旁
并列一列同一批 episode 上的 model − GT 配对差**，由 `tools/paired_bootstrap.py` 计算
（逐 sequence 先成对求差再 bootstrap，10,000 重采样、seed 42、按名字配对）。

**为什么绝对列本身不干净——这一条必须进表注。** 本项目的 SDF 把「未被房间壳体包住」的一切
都判为实心，**包括每一个被扫描表面的下方与背后的体积**：座面之下、床垫上表面之下、沙发垫
内部都各自围出一块实心区（`code/priors/hsi/scene_field.py:560-596`、`:669`）。这不是缺陷，
而是 room-shell 极性的直接推论：实测 `subfloor_enclosed_fraction = 0.0` ⇒
`interior_is_free_space = True` ⇒ `solid = ~inside`，实测 `solid_fraction` 为 padded bbox 的
**0.326–0.434**。

实测穿透质量的高度归属（GT，全 375 条）：

| 穿透质量所在高度 | 占比 | 它实际度量的是 |
|---|---|---|
| < 地面 + 2 cm | **68.33 %** | 脚踩地面 |
| 地面 + 10–50 cm | **26.28 %** | **身体处于座面／床垫体积内部** |

排除地面把 DIMOS 求和式从 **20.452** 压到 **5.973**（**×0.292**，与上一节 §B「唯一敏感的一步是
0 → 1 cm」一致），但它**碰不到 +10–50 cm 这一项**。

于是两列分工明确：

1. **绝对列承担可对照性**（walk 130 的 0.449 = LINGO 已发表 0.402 的 1.12x），代价是它
   **仍然含有家具内部体积**；这句话必须写在表注里，不得让读者把它读成纯粹的「穿墙深度」。
2. **配对差承担区分度**：家具内部那部分质量主要是「坐在这个物体上」这件事本身的属性，
   GT 与 model 共有，逐 episode 求差可抵掉其中大部分。

**Tier B（重建物体壳体、单独给家具符号）暂不采用，理由不是算力。** 实测冷构建
**102.5 s / 37 MB 每场景**，26 个 test 场景合计 **≈45 min / ≈1 GB**——完全付得起。真正的阻塞是
**符号**：`_resolve_polarity` 是**从 sub-floor slab** 推 inside/outside 的（`:560-596`），把地面
拿掉就摧毁了这个判别器，`interior_is_free_space` 必须显式传入（`:833-839`）；而单独抽出的
扫描家具通常是**开口壳体**，符号会退化到 6-ray 多数表决（`:555`）——该 fallback 在今天的口径下
只有 **1/26** 个场景需要。符号一旦判错是**静默的、且量级很大**（整块体积的内外互换）。

**推翻条件**：出现一个不依赖地面的稳健符号判定；或者拿到逐 episode 的物体实例标注**并带几何**
（LINGO 的 episode 文本只给动作，不给实例，因此它本身不够）。

### B. 已知不一致（用户决定：只登记，暂不修）：28 关节常量的末两个槽位

**这不是 InfBaGel 与 LINGO 之间的分歧**——那种分歧是正当的、可以各自成立。这是一处
**常量与 InfBaGel 自己的两份数据资产之间**的不一致：

| 资产 | 末两个槽位实际是 | 证据 |
|---|---|---|
| `human_joints_aligned.npy`（即 `[0:84]` 通道所对应的数组） | **ring1**（SMPL-X 34 / 49） | FK 匹配距离 **0.000000 m**；到 middle1 为 0.0231 m |
| `rest_human_offsets_aligned.npy`（`fk_hand_loss` 的 FK 来源） | **ring1**（`[0..21,34,49]`） | 23 个非根 offset 全部匹配到 **1.5e-7 m**；按 middle1 读则差 **2.8e-2 m** |
| `code/utils.py:299-300` 的 `SMPLX_JOINTS_24` / `SMPLX_JOINTS_28` | **middle1**（25 / 28 / 40 / 43） | — |

常量的消费者：`code/test_infbagel_lingo_hsi.py`（`joints_ind=SMPLX_JOINTS_28` 调用处，当日为
`:181`，该文件正被并行改动，行号会漂移）与 `code/test_infbagel_hosi.py:718`——**后者与
`phase/01b-hoi` 共用**，因此任何修正都是跨分支改动，需要用户批准。

**影响范围，精确地说**：

- **B 与 C 已发表的数字没有一个被偏置。** 该错配在两个臂上完全对称（同一常量同时用于 GT 与
  model），所以不存在左右偏向。
- 它到达的是 `metric_joints`，因而到达 `rds`（正在从常规评测中下线）、28 关节穿透诊断
  （**主口径是 10475 顶点体，不受影响**），以及边缘地影响 `jerk_ratio`（28 关节中 2 个参与
  平均）与 `min_dist`（对关节取 min，故只在「手是离地面投影 pelvis 目标最近的关节」时才动）。
- **不影响 `[0:84]` 导出路径**——该路径按构造就是 dataset 顺序，不经过这两个常量。

同时登记一条**已审计为正确**的：`code/models/infbagel.py:374`、`:820` 的
`hand_idx_28 = [20, 21, 25, 27]` 是**对的**。两份索引在这四个槽位上落到**同一组四个关节**
（双手腕 + 双 ring1）：FK 预测侧来自 rest offsets（ring1），GT 侧来自
`human_joints_aligned.npy`（同为 ring1），故 **B 与 C 的 `fk_hand_loss` 未受影响**——尽管这条
路径确实跑过（两个 HSI 配置均为 `use_object_keypoints: true`，见
`code/config/config_train_hsi_b_lingo_full.yaml:67` 与 `config_train_hsi_c_lingo_cm.yaml:65`）。

**推翻条件**：一旦 28 关节路径从「诊断」升为**权威报告口径**，就必须把
`code/utils.py:299-300` 末尾的两个 middle1 槽位（`28` / `43`）改为 ring1（`34` / `49`），
并与 `phase/01b-hoi` 协调后一起改。

### C. 忠实 FS 的悬空盲区与「拟删除指标清单」的冲突（本节新发现，必须登记）

上一节定版的忠实 FS 去掉了预平移。预平移原先会把每条序列的最低足高强行压到 0，**因而保证
了每条序列都有入带帧**。去掉之后：**一个全程悬空的 rollout——脚从不进入距地面 H 之内——不产生
任何入带帧，`fs_nemf ≈ 0`，也就是看起来完美。** 上一节写明的兜底是 `contact_count` 与
`goal_height_err_m`，而兜底只有在**它们确实与 FS 同表报告**时才成立。

**冲突在这里**：这两个键都出现在一份**已提出但从未确认、也从未执行**的指标删除清单上。该清单
另外还列了 `contact_count_exterior`、`contact_frame_ratio_saturated_diagnostic`、
`reachability_violation_ratio`、`pen_burst`、`skate_ratio`、`time_to_goal_*`、
`goal_orientation_err_rad` 与 RDS。

**因此定下一条硬约束：在没有先放置一个等效的悬空守卫之前，`contact_count` 与
`goal_height_err_m` 都不得删除。** 注意这条约束与「`skate_ratio` 也在清单上」叠加后更紧：
上一节已实测 `skate_ratio` 因绝对 5 cm 门限有**同一个**盲区，故**足部两列都抓不到悬空**，
清单若同时删掉 FS 的兜底与 `skate_ratio`，悬空这一失效模式就完全无人看守。

同时把这条弱点写明，以免日后误以为兜底很强：**`goal_height_err_m` 本身作为守卫是弱的**——
目标的 y 被置零，所以它其实只是在报告**终点 pelvis 高度**（GT 参考：全 375 为 **0.761 m**、
walk 130 为 **0.941 m**，见上一节 §F）。它能把「悬空 30 cm 结束」与「站在地上结束」分开，但
分不开「全程悬空、最后落地」这一类；真正逐帧盯住接触的是 `contact_count`。

---

## 2026-08-18（同日第三次修订）：generated 臂的 SMPL-X 坐标系订正，`schema_version` 升到 3

用户已批准：**训练侧的问题（身体站不起来）先保留，本节只修评测侧。** 本节不追求让身体站直，
也不为改善输出观感调任何参数——checkpoint 的 root 旋转距真值 124–128°（随机基线 126.5°），
其共轭不变的 root 抖动为 3.47 °/帧均值 / 32.1 p95（GT 为 0.69 / 1.9），那是**被刻意推迟的
训练侧问题**，不是本次修复的失败。

### A. 缺陷：被映射的是平移与 FK 输出，而**不是**姿态

修复前 `code/test_infbagel_lingo_hsi.py`（generated 臂）为：

```python
smpl_translation = yup_to_zup(interpolated_points[:, 0] + translation_offset)
smpl_pose        = yup_to_zup(local_axis)
vertices, metric_joints = _run_smplx_chunks(smpl_pose, smpl_translation, ...)
vertices, metric_joints = zup_to_yup(vertices), zup_to_yup(metric_joints)
```

记 `M = yup_to_zup` 的矩阵（绕 +x 转 90°，`yup_to_zup(v) = M v`，`zup_to_yup(v) = Mᵀ v`）。

**推导。** SMPL-X 的 FK 为 `p_i = p_parent + R_global_parent · offset_i`，其中 `offset_i` 是
模板的 rest offset，**不随 `M` 映射**。把平移映射为 `M(pelvis + offset)` 再把输出映射回
`Mᵀ`，并不是一次往返：

- 根关节 `p_0 = M(pelvis − J_rest0) + J_rest0`，取 `Mᵀ` 后为
  `pelvis + (zup_to_yup(J_rest0) − J_rest0)`；
- 其余关节相对 pelvis 的构型变成 `Mᵀ f_i − Mᵀ f_0`，即**整个身体相对自身 pelvis 被绕 +x
  多转了 90°**。

以参考 betas 的 `J_rest0 = (0.0012, −0.3668, 0.0127)` 代入，根位移恰为

> **`(0, +0.3795, +0.3542)` m**

`code/constants.py:10` 的 `pelvis_shift = [0.001144, -0.366919, 0.012666]` 就是同一个
`J_rest0`，属于仓库内已提交的独立旁证。

**必须写明的一处订正。** 姿态上的 `yup_to_zup` **不是缺陷，必须保留**。
`code/datasets/infbagel.py:90-94` 在合成 `global_rot_6d` 之前先对 `human_orient` /
`human_pose` 施加了 `zup_to_yup`（`code/priors/hsi/data.py:57-59` 训练侧同样如此），因此网络
发出、`quat_ik_torch` 解回的每个局部旋转都带着 `Mᵀ R M` 的共轭；`yup_to_zup` 正是把它还原成
`R`——也就是 SMPL-X 模板所在的坐标系。**把姿态也改成 identity 会留下 9.2 cm 的
pelvis-相对关节误差**（见 §C 的 `full identity` 行）。

### B. 三条相互独立的确认

1. **哪个坐标系是 `human_joints_aligned.npy`？** 对第 0 条序列 161 帧扫遍
   （姿态 × 平移 × 输出）全部 27 种组合：`none|none|none` 为 **9.14e-04 m**（残差来自 28 槽位
   中 25/27 两槽的 middle1/ring1 不一致，见同日第二次修订 §B），其余全部 ≥ 0.12 m。
   即 GT 臂的 identity 路径就是正确路径。
2. **缺陷签名可预测。** 在 SHIPPED 路径下，pelvis 误差向量实测
   `(0.0000, 0.3795, 0.3542)`，**逐帧方差为 0**，与 §A 的预测相差 **1.36e-07 m**；同时
   pelvis-相对构型错 **0.284 m**。两个症状同时出现，正是平移＋输出被映射、模板未被映射。
3. **落盘的 375 个 NPZ 上复现。** 对 `c_guided_v2` 的全部 375 个 schema-1 NPZ：按修复后的
   口径（姿态原样、`zup_to_yup(transl)`、输出不变换）重建，FK pelvis 与文件自带的
   `global_jpos` pelvis 相差 **1.19e-07 m**；按文件自己记录的 `zup_to_yup` 口径重建则差
   **0.3795 m**——恰为 §A 位移的 y 分量。

### C. 修复与验收闸门

修复：`smpl_translation` 不再变换，FK 输出不再变换，`smpl_pose` 的 `yup_to_zup` 保留。

**闸门（用 GT 旋转 + GT pelvis 灌进 generated 臂自己的解码链，与 `ground_truth_motion` 的 FK
输出比，20 episode、两臂槽位一致的 26 个关节）**：

| 候选 | mean | median | p99 | max | pelvis max |
|---|---|---|---|---|---|
| SHIPPED（修复前） | 3.000e-01 | 1.205e-01 | 1.470e+00 | 1.777e+00 | **3.795e-01** |
| full identity（姿态也改 identity） | 2.578e-01 | 1.159e-01 | 1.692e+00 | 1.959e+00 | 1.192e-07 |
| **FIX（只保留姿态变换）** | **6.36e-07** | **0.000e+00** | **2.38e-07** | 2.10e-03 | **1.19e-07** |

中位数为 0 表示修复后多数数值与 GT 臂**逐位相同**。`max` 的 2.1e-03 与 mean 的量级差来自
一处**既有**缺陷，与本次修改无关：`code/utils.py:43` 的 `quaternion_slerp` 把 LERP 分支的权重
写反了（`q1 * step + q2 * (1 - step)`），而该分支在 `dot > 1 - 1e-6` 时触发，于是 `step = 0`
处返回 `q2` 而非 `q1`。258 帧中只有 6 帧（18–20 与 66–68，两个关键帧的邻域）超过 1e-5，
未共轭的对照组同样如此。**本节不修它**：它同时作用于两个臂，改动会移动 GT 臂已封存的数字。

### D. `schema_version`：metrics 2 → 3，motion export 1 → 2

- `METRICS_SCHEMA_VERSION` **2 → 3**：键集不变，但 generated 臂上**每一个由 FK 派生的指标的
  数值**都会变（穿透、FS、jerk、boundary jerk、goal 分解、contact）。这正是该字段存在的场景。
  **GT 臂的 payload 与 schema 2 逐位相同**——那条臂一直走 identity。
- `MOTION_EXPORT_SCHEMA_VERSION` **1 → 2**：两臂现在都记 `"identity"`。
- `results/lingo_hsi/c_guided_v2/` 里已有的 **375 个 NPZ 不删、不改写**。它们是 motion-export
  schema 1 / generated 臂，消费者必须：`global_orient` / `body_pose` **原样使用**（它们已在
  SMPL-X 模板坐标系中），把 `transl` 换成 `zup_to_yup(transl)`，并**无视文件自带的
  `smplx_output_transform: zup_to_yup`、对 FK 输出不作变换**。该口径已在全部 375 个文件上验到
  1.19e-07 m（§B.3）。GT 臂的 schema-1 文件本来就是对的，无需修补。

### E. 上游同一缺陷未修，且是跨分支共享的（必须记）

`code/test_infbagel_hosi.py:713-721` 有**字面等价**的写法：

```python
root_trans = yup_to_zup(points_all.reshape(-1, 28, 3)[:, 0, :].to(device) + transl)
pose_pred  = yup_to_zup(transforms.matrix_to_axis_angle(local_rot_mat_all))...
human_verts, joints = zup_to_yup(human_verts), zup_to_yup(joints)
```

该文件**与 `phase/01b-hoi` 共享**，本节按约束**未作任何修改**。因此：

> **`phase/01b-hoi` 上每一个模型侧的几何数字都带着同一个缺陷**（pelvis 位移
> `(0, +0.3795, +0.3542)`，身体相对自身 pelvis 多转 90°），**欠一次跨分支通报**。
> 按 `AGENTS.md`「Cross-branch communication」，这条通报本身需要用户先批准。

### F. 落盘 payload 的缺陷签名

按 `skate_ratio` 逐个 payload 统计（`results/lingo_hsi/**/per_sequence_metrics.json`，共 60 份）：

- **模型 cell 的 55 份 payload 全部落在 0.0000–0.0014**（B/C、guided/unguided、各分片、
  各 probe、各 smoke），
- GT 家族的 4 份（`ground_truth`、`ground-truth-v1/v2/v3`）全部为 **0.1408**，`gt-probe-6`
  为 0.0299。

模型侧比 GT 低两个数量级，与「脚被转到别处、几乎不进入 5 cm 门限」一致。**这 55 份都必须视为
带缺陷的历史记录**，不得与修复后的数字同表比较（`schema_version` 已能把它们分开）。注意此处的
份数是**文件数**（含分片与 probe），其背后只有 2 个不同的 checkpoint。

### G. 本节未做

- **未跑任何 model cell。** C+guided 的重跑由用户单独放行。
- 未改 `code/test_infbagel_hosi.py`、未改 `code/priors/core/`。
- 未删除、未改写任何已存在的 payload、NPZ 或 checkpoint。

## 2026-08-18（同日第四次修订）：C+guided 在订正后尺子下的全协议结果

`results/lingo_hsi/c_guided_v3/hsi_c_lingo_cm_epoch089-8527b03ae900/`，
`schema_version` 3，375/375 episode，seed 42，`lingo_episode_dir=data/lingo_hsi_test_v2/data`，
`export_motion=true`（375 份 NPZ，39 MB）。对照行 `results/lingo_hsi/ground-truth-v3`。
配对差为同 episode 逐条相减，95% CI 为 10000 次配对 bootstrap（seed 20260818）。
完整表在 `.claude/scratch/c_guided_v3_vs_gt_v3.txt`。

### A. 修复本身把模型侧的数改了多少（`c_guided_v2` → `c_guided_v3`）

同一 checkpoint、同一 episode 集、同一 seed，**只差 FK 坐标修复**（375/375 的
`frame_count`、`window_count`、`skate_denominator_frames` 全等，确认 rollout 未变）。

| 指标 | 修复前 | 修复后 | 倍数 | 方向 |
|---|---:|---:|---:|---|
| `pen_ratio` | 0.01482 | 0.06739 | 4.55× | 原来**低报** |
| `pene_pct_scene` | 0.02552 | 0.09607 | 3.76× | 低报 |
| `pene_sum_mean_floorexcl` | 14.148 | 79.338 | 5.61× | 低报 |
| `pen_depth_mean` | 0.05149 | 0.09127 | 1.77× | 低报 |
| `contact_count` | 546.08 | 1685.32 | 3.09× | 低报 |
| `reachability_violation_ratio` | 0.02637 | 0.10127 | 3.84× | 低报 |
| `fs_nemf` | 0.00100 | 0.08592 | 86.3× | 低报 |
| `last_dist` | 0.27303 | 0.14350 | 0.53× | 原来**高报** |
| `goal_height_err_m` | 0.91812 | 0.53861 | 0.59× | 高报 |
| `success_last_20cm` | 0.44000 | 0.82933 | 1.89× | 低报 |

机制一致：被抬起 `+0.3795 m`、沿 z 平移 `+0.3542 m` 的身体**浮出了碰撞体**，于是穿透与接触
双双低报；同一位移直接加进了 goal 距离，于是 goal 误差高报。**没有一条是「修复让模型变好看」。**

### B. 内建对照：aligned / unaligned 缝隙距离的表现正确

- `transition_distance_aligned` 修复前后最大绝对变化 **2.334e-08**（float 噪声）——**不变**，
  因为对齐口径把全局旋转除掉了；
- `transition_distance_unaligned` 最大绝对变化 **3.916e-02**，相对最高 **18.0%**——**该变**。

这一对量是实现自带的对照组，方向完全符合预期，是「修复是对的、指标实现是干净的」的独立证据。
`boundary_jerk` / `interior_jerk` 只动了 0.7% / 1.8%（秩相关 0.9996 / 0.9968），
说明**缝隙不连续这条结论不依赖该缺陷**。

### C. C+guided v3 对 GT v3：配对差（* = 95% CI 不含 0）

| 指标 | 全 375 模型 | GT | 配对差 | walk 130 模型 | GT | 配对差 |
|---|---:|---:|---:|---:|---:|---:|
| `pene_pct_scene` | 0.09607 | 0.05044 | +0.0456 * | 0.06678 | 0.03358 | +0.0332 * |
| `pene_sum_mean_floorexcl` | 79.338 | 5.973 | +73.4 * | 53.371 | 0.449 | +52.9 * |
| `pen_value` | 0.06827 | 0.03416 | +0.0341 * | 0.06489 | 0.02481 | +0.0401 * |
| `pen_ratio` | 0.06739 | 0.02805 | +0.0393 * | 0.04384 | 0.01286 | +0.0310 * |
| `pen_depth_mean` | 0.09127 | 0.04937 | +0.0419 * | 0.08823 | 0.04175 | +0.0465 * |
| `contact_count` | 1685.3 | 787.8 | +897.5 * | 1243.6 | 539.0 | +704.7 * |
| `reachability_violation_ratio` | 0.10127 | 0.05520 | +0.0461 * | 0.06949 | 0.03931 | +0.0302 * |
| `fs_nemf` | 0.08592 | 0.25967 | −0.1738 * | 0.07758 | 0.40821 | −0.3306 * |
| `last_dist` | 0.14350 | 0.00000 | +0.1435 * | 0.19889 | 0.00000 | +0.1989 * |
| `success_last_10cm` | 0.41333 | 1.00000 | −0.5867 * | 0.34615 | 1.00000 | −0.6538 * |
| `success_last_20cm` | 0.82933 | 1.00000 | −0.1707 * | 0.70769 | 1.00000 | −0.2923 * |
| `goal_planar_err_m` | 0.33900 | 0.00000 | +0.3390 * | 0.42367 | 0.00000 | +0.4237 * |
| `jerk_ratio` | 12.273 | 1.194 | +11.08 * | 12.835 | 1.210 | +11.62 * |
| `boundary_jerk` | 6831.2 | 84.4 | +6746.8 * | 7479.9 | 97.6 | +7382.3 * |
| `transition_distance_aligned` | 0.17967 | 0.00641 | +0.1733 * | 0.19112 | 0.00685 | +0.1843 * |

**除 FS 外每一项都显著更差**；interactive 245 上唯一不显著的是 `success_min_10cm`
（0.99592 对 1.00000，CI 上界触 0）。

### D. 口径决定量级：1.8× 还是 133×

walk 130 上同一套顶点、同一个 SDF：

| 口径 | 模型 | GT | LINGO 公布 | 模型 / LINGO |
|---|---:|---:|---:|---:|
| `pene_pct_scene`（顶点比例，LINGO 自己的口径） | 0.0668 | 0.0336 | 0.038 | **1.76×** |
| `pene_sum_mean_floorexcl`（DIMOS 求和，m） | 53.37 | 0.449 | 0.402 | **133×** |

同一个物理事实，报出来差 75 倍。原因是求和口径 = 顶点比例 × 深度 × 10475，而模型的穿透
**又广又深**（深度 0.0882 对 GT 0.0418，2.11×），两个因子在求和里相乘。

**这不是尾部效应：** walk 上模型 `pene_sum_mean_floorexcl` 中位数 22.34（GT 中位数恰为
0.0000），130 条里 **120 条**超过 LINGO 的 0.402，**103 条**超过其 10 倍。

审稿口径建议：**主表用 `pene_pct_scene`**（与 LINGO 公布的 0.038 同口径、同量级，GT 校准到
0.0336 对 0.038），求和式作为附表并注明是 extensive 量。

### E. FS 行在模型上可能是空的（悬空盲区的实测确认）

- 模型 375 条里 **208 条 `fs_nemf` 恰为 0.0**，GT **0 条**；
- 模型 `skate_frames` 合计 383，GT 19483（**51 倍差**），分母 `skate_denominator_frames` 两侧全等；
- `skate_ratio`：修复前 0.0003（与 F 节记录的 55 份 payload 的 0.0000–0.0014 一致）→ 修复后
  0.0049 → GT 0.1408。修复把它抬了 16 倍，**但仍比 GT 低 29 倍**，且 214/375 仍恰为 0。

即 §C 表里 FS 那个「−0.33 显著更好」**不可作为质量结论引用**。GT 自身也有 74/375 为 0（脚踩实
且不滑，是合法的 0），所以零值本身不构成证据；区分「脚不在带内」与「脚在带内且不动」需要 FK 足高
统计，见下条。

> **待补：** 用导出的 375 份 NPZ 在 CPU 上重建 `metric_joints`（`smplx_output_transform:
> "identity"`），统计 FK 足高分布，与 208 条零值做交叉表。网络自身的 28 槽关节通道
> （`global_jpos`）已测：足高最小值均值 **−0.1018 m**、97.4% 的踝槽落在 8 cm 带内，
> 所以零值**不可能**来自这个通道，只能来自 FK 骨架的姿态。

### F. 对已有结论的影响

- **缝隙不连续（`boundary_jerk` ~80× GT）不受影响**：修复前后只动 0.7%，秩相关 0.9996。
  该结论继续成立。
- **设计先验 #7 的方法论规则被强化，不是被推翻。** 本节就是它最干净的一次演示：修复前模型的
  `pen_ratio` 0.0148 **比 GT 的 0.0281 还低**（看起来是穿透赢），而 `contact_count` 被同时
  低报 3.09×；修复后穿透变成 GT 的 2.40×、接触变成 2.14×。"never claim a penetration win
  without engagement beside it" 正是能最早抓到这个缺陷的规则。
- **#7 引用的那个具体数字（D2-AH `contact_percent` 0.3192 对 GT 0.66188）是 HOI 分支证据，
  不是本节测到的 HSI `contact_count`，本节不据此改写它。** 机制上它可疑：
  `code/test_infbagel_hosi.py:337-356` 的 `contact_percent` 把**经过同一 FK sandwich 的手部
  关节**（`joints[:, 24]`/`[:, 26]`）与**在原生坐标系里摆好的物体顶点**比 5 cm 门限，而位移量
  ‖(0, 0.3795, 0.3542)‖ = **0.519 m**，是门限的 10 倍。这会把 `contact_percent` 系统性压低。
  GT 臂是否同样受影响取决于该分支的 GT 路径，**本节未读、未改**——按 `AGENTS.md` 需用户先批准
  跨分支动作。

### G. 本节未做

- 未跑 FID / R-Precision / MM-Dist：需 worker 上 `/home/yujinlun/data/transfer/` 的冻结
  text-motion evaluator，375 份 NPZ 已按其输入口径导出，属下一步。
- 未修 `code/utils.py:43` 的 `quaternion_slerp` 权重（会作废 `ground-truth-v1/v2/v3` 三行）。
- 未改 `code/test_infbagel_hosi.py`、未改 `code/priors/core/`、未发跨分支通报。
- 未 commit、未 tag、未分配 run id、未跑 `tools/experiment.py start`。

## 2026-08-18（同日第五次修订）：FK 骨架是躺平的——对第四次修订 D、E 两节的更正

第四次修订的 §E 留了一条「待补」：模型侧 208/375 条 `fs_nemf` 为 0，究竟是「脚不在带内」还是
「脚踩实不动」。已用导出的 375 份 NPZ 在 CPU 上重建 `metric_joints` 测完，答案是前者，而且这条
测量顺带**推翻了第四次修订 §D 的口径建议**。

### A. 复现保真度先立住

重建用 `pose = concat(global_orient[:,None], body_pose)`、无任何坐标变换
（`smplx_output_transform: "identity"`）：join 375/375；重算的 `fs_nemf` 对 payload
最大 |Δ| = **5.0e-7**；208 个零值全部复现，**且是同样的 208 条**；`F` 与 payload
`frame_count` 全等。所以下面的数就是 payload 里那些指标真正吃到的骨架。

### B. FK 骨架躺平，网络自己的关节通道却大致直立

| 量 | FK 骨架（指标吃的） | `global_jpos`（网络 28 槽通道） |
|---|---:|---:|
| pelvis→neck 轴偏离 +y | **95.15°** | **15.58°** |
| 22 个身体槽的 y 跨度 | 0.5925 m | 1.5260 m |
| 踝 / 趾 / pelvis / neck 平均高度 (m) | 0.5786 / 0.5794 / 0.5696 / 0.5504 | −0.0225 / −0.0599 / 0.5705 / 1.3425 |
| 每帧每关节 L2（两者之间） | — | **0.6146 m** |
| pelvis 高度之差 | — | **+0.0003 m** |

即：**根平移是对的，挂在它下面的身体是横着的**。踝、趾、pelvis、neck 四个高度挤在 0.55–0.58 m
的同一层，正是躺平的签名。

### C. FS 零值的机制：零残差

2×2 交叉表（全 375）：`fs==0 且 FK 最小足高 > 0.08 m` = **185**，`fs==0 且 ≤0.08` = 23，
`fs!=0 且 >0.08` = **0**，`fs!=0 且 ≤0.08` = 167。按带分开后分离是完全的：208 条零值里
**208/208** 的踝最小高度 > 0.08 m，**208/208 在两个带里贡献的 (frame,joint) 槽数恰为 0**
（唯一一条趾低于 0.04 m 只发生在最后一帧，而 `fs_nemf` 取 `h` 于 `pos[:-1]`，从不采样它）。
FK 槽落在带内的比例只有 **1.07% / 1.31%**（`global_jpos` 是 97.43% / 97.27%）。

**结论：`fs_nemf` 在这个模型上是空的**，208 条为 0 是因为没有任何一帧进入接触带，与滑不滑无关。
把同一公式套在 `global_jpos` 上得 **1.6808 cm/frame**（walk 2.0369），0 个零值——对 GT 的
0.2597 / 0.4082 是约 **5 倍更差**，这才是躺平的 FK 身体一直在掩盖的数。

### D. 躺平不是残留的坐标变换（已排除）

在 6 条序列上把 pose 上那个 `yup_to_zup` 去掉（full identity，即 GT 臂的口径）：
偏离 +y 变成 **108–142°**（比 as-shipped 的 87–107° **更差**），对 `global_jpos` 的 L2
从 0.6361–0.6515 涨到 0.6871–0.7879。**两种口径都不直立，as-shipped 是两者中较好的一个**，
与第三次修订里「去掉它 pelvis 相对配置错 9.2 cm」的测量一致。

同时 GT 臂（identity 口径）的脚**确实**落在 4–8 cm 带内（375/375 条 `fs_nemf` 非零，
`skate_frames` 合计 19483），所以 **FK 机制本身能产出直立身体**。综合两条：躺平来自
**模型的旋转通道**，不是路径缺陷。第三次修订的修复继续成立。

> 仍待判定：解码链本身（6D global → `quat_ik_torch` → `interp_jrot` → `yup_to_zup`）是否
> 无损。已派 GT 旋转的往返测试：把 GT 旋转按 dataset 的方式正向编码进网络表示，再走生成臂的
> 解码链回来做 FK，与 GT 关节数组比。若往返即已躺平，则问题在链上；若往返精确，则旋转通道是
> 模型真的没学会。**这个必须在任务 #22（B smoke）之前回答**——链若是坏的，B smoke 什么也说明不了。

### E. 关节通道也不是刚体（任务 #18 的结果）

在 `global_jpos` 上按 SMPL-X 自己的 `parents[:22]` 量骨长：**21/21 根骨的序列内 CV 都超过
1e-2**，10/21 超过 1e-1；CV 均值 **1.013e-1**，per-(seq,bone) CV 的 p95 0.2795、max 0.6731。

对照组用数据集通道 `data/dataset/human_joints_aligned.npy` 同 28 槽、342 个窗口：
**刚到浮点噪声，CV 均值 2.43e-7，0/21 超过 1e-3**，平均骨长与 SMPL-X FK 骨架四位小数相符。
所以生成通道的 CV 是对照组的约 **4e5 倍**。

平均骨长本身也错：8/21 根偏离对照组 >20%——`pelvis→spine1` **3.93×**、
`pelvis→left_hip` 2.67×、`pelvis→right_hip` 2.46×、collar→shoulder 两侧 0.40×、
ankle→foot 0.42–0.44×、elbow→wrist 0.84–0.88×。

**所以两个头都坏，但坏法不同**：旋转通道给出躺平的身体，关节通道大致直立但被拉伸成非解剖的形状。
`global_jpos` 的「直立」因此也不能当可信参照。

### F. 更正第四次修订 §D 的口径建议

第四次修订建议「主表用 `pene_pct_scene`，因为模型 0.0668 对 LINGO 公布 0.038 只有 1.76×，
和社区同量级」。**这个说法要撤回**：那 0.0668 是**一个躺平身体**与场景的交集，而 LINGO 的
0.038 描述的是直立行人。两者不可比，1.76× 会让模型听起来接近可用，实际上它的身体是横着的。

保留的部分：**GT 侧的校准结论仍然成立**（`ground-truth-v3` 的 0.0336 对 LINGO 0.038，
`pene_sum_mean_floorexcl` 0.449 对 0.402 = 1.12×）——尺子对得上，是被测对象不成立。
`pene_pct_scene` 仍是将来的主表口径，但**在旋转通道修好之前，模型侧的任何几何数都不要与文献并列**，
只能自比（同 `schema_version`、同口径的版本间比较）。

### G. 本节的方法论教训

第三次修订的验收闸门只检查了「pelvis 相对配置」和「GT 臂逐位不变」，**没有检查 FK 骨架是否直立**。
一个 O(1) 的量（pelvis→neck 偏离 +y 的角度）就能在重跑之前抓到它。**将来任何改动 FK 或旋转
表示的修复，闸门必须包含一个绝对姿态量，而不只是相对量与不变性。**

## 2026-08-18（同日第六次修订）：解码链无损，但评测驱动自己把条件旋转掀了 ~100°

第五次修订留的那条「仍待判定」已答：**解码链是无损的，问题在它上游的评测驱动里，而且是我们自己
引入的**。51 条 GT 序列、26 个场景、11638 粗帧。

### A. 往返测试：链无损（判定结果是 (b) 而不是 (a)）

把原始 axis-angle 按 `datasets/infbagel.py:512-527` **逐字**正向编码进网络的 6D global 表示，
再走生成臂的解码链回来做 FK：

| 量（interp 3，米） | 值 |
|---|---:|
| GT 臂 FK 对关节数组 | 0.007158 |
| 往返 FK 对关节数组 | 0.007159 |
| **往返 FK 对 GT 臂 FK** | **2.00e-6**（max 8.19e-4） |
| 去掉 pose 上的 `yup_to_zup` | 0.650 |

pelvis→neck 偏离 +y：往返 FK **8.571°**、GT 臂 FK **8.571°**、GT 关节数组 **8.591°**；
identity 解码 **91.48°**。

逐级最大矩阵元误差（float32 eps ≈1.2e-7）：`rotation_6d_to_matrix` **逐位相同**、
`quat_ik_torch` 4.0e-7、`interp_jrot`（权重修正后）3.0e-7。唯一有损的一级是 `interp_jrot`
**as shipped** = 2.52e-3，全部来自 `code/utils.py:43` 那个交换了的 lerp 权重：触发率
**11.27%**，代价 0.003° 均值 / 0.160° 最大，FK 位置 6.7e-5 m 均值。**它造不出 90° 的倾倒**，
且 interp 3 下两臂共用 `interp_jrot`，在配对比较里抵消。修它把 FK-对-数组从 0.0071586 挪到
0.0071418 m。

### B. 全 375 条都是躺平的，而且没有任何坐标口径能救

pelvis up 轴偏离竖直（97632 帧）：

| | mean | p5 | p50 | p95 |
|---|---:|---:|---:|---:|
| as-shipped 口径 | **90.58°** | 45.47 | 89.51 | 135.98 |
| network 口径（撤销导出的变换） | 90.58° | 45.47 | 89.51 | 135.98 |
| 同 375 条的 GT | **10.10°** | 1.35 | 7.17 | 23.78 |

两行相差 2e-4°——**该统计量是共轭不变的，所以 pose 上任何 yup/zup 选择都救不了它**。逐序列均值
p5 79.0°、p95 101.3°（min 65.4、max 112.9），**375 条无一例外**。单一全局旋转也救不了：
`R₀ŷ` 的 mean resultant length 0.696（GT 0.914），取最优单方向后残差仍有 43.8° 均值 / 80.7° p95。
聚集方向距 **+z 仅 1.25°**——正是「多了一次 `zup_to_yup`」的 90°-about-+x 签名。

### C. 上游缺陷：`:1429` 把 y 旋转施加在 y 是水平轴的帧里

`code/test_infbagel_lingo_hsi.py:1414-1429`：

```python
points_world   = points_world @ desired_rotation.T      # :1418  y-up 关节通道，正确
global_matrices = desired_rotation @ global_matrices     # :1429  已共轭的旋转通道，错帧
```

`desired_rotation = R_y(θ)`，`|θ|` 均值 **122.2°**、p95 255°。对交给去噪器的第 0 步条件身体的
pelvis up 轴：**9.46° → 100.49°**（p50 105.6、p95 172.6）。把同一个 yaw 共轭进通道自己的帧则
仍是 9.46°。连带 `mat_step[:, :3,:3]`（由该通道的 y-euler 导出，每个窗口既归一化关节历史又给模型
输出做 un-shift）从训练口径下的 **0.00°** 变成 as-shipped 的 **97.26°**（p95 175.3）。

**归属：`desired_rotation` 只出现在 `code/test_infbagel_lingo_hsi.py`，由我们自己的提交
`1914cae` 引入**；作者的 `75efccc` 没有，HOI evaluator 没有，**训练路径从不施加它**
（`train_infbagel.py`、`models/infbagel.py`、`datasets/infbagel.py`、`priors/` 全部零命中）。
所以这是**我们的缺陷、且只影响 HSI**。

### D. 因此：不能据此判定 checkpoint 坏

模型在第 0 步就被喂了一个约 100° 倾倒的旋转通道，配一个直立的关节通道。**「旋转通道躺平」对这次
rollout 成立，但归属未定**——模型可能只是忠实复现了被掀倒的条件。

**并且这是两个缺陷不是一个**：`desired_rotation` 绕 SMPL 的 z 轴掀，本该把 up 轴散布在 xy 平面上，
而实测聚集在 +z。还有一个系统性的 90°-about-x 成分**尚未定位**。

### E. 顺带查出的三条（都已测，都在解码之上游）

1. **数据集的窗口正则化几乎是空操作。** y-up root 的真实 heading 均值 **90.72°**（p95 176°），
   而 `datasets/infbagel.py:513-516` 只去掉 **4.54°**（p95 9.6°），因为它在 y 是水平轴的帧里
   取 y-euler。**训练窗口在两个通道里都不是 heading-canonical 的**——直接关系到 Phase 1C 的窗口帧。
2. **`datasets/infbagel.py` 与冻结的 `code/priors/core/window_codec.py` 对 6D 通道口径不一致。**
   两边都用 `zup_to_yup` 共轭，但 shift 角不同：scipy `as_euler('zxy')` 是**外旋**，
   `window_codec.py:141` 的 `matrix_to_euler_angles(root,"ZXY")` 是**内旋**，实测帧错配
   **5.13° 均值 / 16.08° 最大**。今天无害（`config_train_hsi_{b,c}_*` 走 `InfBaGelDataset` 而非
   `PriorWindowDataset`，训练与评测共用外旋口径），但**任何经 `core/` 训练的 Phase 1C run 会踩到**。
3. 两条本文档此前的说法要更正：
   - 「GT 臂逐位复现关节数组到 0.000000 m」只对**导出**成立（`joints_coarse` 就是那个数组本身）。
     GT 的 **FK** 对数组是 interp 3 下 **7.16 mm**、interp 1 下 **2.00 mm**（pelvis 平移到
     2.9e-8 m）。这 ~5 mm 是线性关节插值对 slerp 姿态插值的差。
   - 第三次修订注释里「去掉 `yup_to_zup` 错 9.2 cm」，在**正确旋转**上实测是 **65.0 cm** 且倾倒到
     91.5°。结论（该变换必需）成立，数字不成立。**已就地改正该注释**（148 个测试仍全绿）。

### F. 欠用户的两个决定

1. **怎么修 `:1429`。** 两条路：
   (a) 把 `desired_rotation` 共轭进通道的帧：`Z @ R_y(θ) @ Z.T @ global_matrices`（`Z` =
   `zup_to_yup` 的矩阵形式）——只改评测，不动训练表示；
   (b) 去掉 `datasets/infbagel.py:91,94` 与 `priors/hsi/data.py:58-59` 的 `zup_to_yup` 共轭，
   以及 `:1610` 补偿性的 `yup_to_zup`，让两个通道都活在同一个 y-up 帧里——**这是真正的修法**
   （顺带修好 §E 的 1 和 2），但它改变训练表示，**因此强制重训**。
2. **一次 GPU 探针**（无 run id，只写 `.claude/scratch/`）：固定 ~20 条跨场景 episode，跑两次
   ——as-is 与 `:1429` 共轭后——只比导出的 pelvis up 轴偏离。回到 ~10° ⇒ 权重没问题，纯驱动缺陷；
   仍在 ~90° ⇒ 旋转通道真的没学会。**必须先于任务 #22（B smoke）**：驱动在喂垃圾时，B smoke
   分不出任何东西。
3. `code/utils.py:43` 的 lerp 权重：真实但值 0.017 mm，修它会扰动所有已有 GT 与模型数字。倾向不修。

### G. 本节未做

- 未跑任何 GPU 工作负载；未改 `:1429`；未改 `datasets/infbagel.py`、`priors/hsi/data.py`、
  `code/priors/core/`、`code/test_infbagel_hosi.py`。
- 未 commit、未 tag、未分配 run id。唯一的 tracked 改动是第 §E.3 条那处注释订正。

## 2026-08-18（同日第七次修订）：更正第六次修订的归属——`:1429` 不是缺陷，而旋转通道确实没学会

用户已批准「先按 (a) 修 `:1429`，然后跑探针」。**测量表明 (a) 会把事情改坏，因此没有实施**，
探针的 arm B 不存在，**本轮没有跑任何 GPU 工作负载**。下面每条都独立复核过。

### A. 第六次修订 §C 的归属是错的

`:1429` **复现的就是训练的构图**。`code/datasets/infbagel.py:512-534`：

```python
init_global_orient_euler = R.from_rotvec(init_global_orient).as_euler('zxy')
shift_rot_matrix = R.from_euler('zxy', [0, 0, -init_global_orient_euler[2]]).as_matrix()
global_rot_mat = shift_rot_matrix[None,None] @ global_rot_mat      # :526  旋转通道
...
mat[:3, :3] = np.linalg.inv(shift_rot_matrix.T).T                  # :530  = inv(shift)
joints = joints @ shift_rot_matrix.T                               # :534  关节通道
```

**同一个 `shift_rot_matrix` 同时左乘旋转通道、右乘转置到关节通道**——评测的 `:1418`/`:1429`
是同一个操作。而 `:1447-1449` 的 `shift_rotation` 是在 `:1429` **之后**从通道 root 重算的，
`desired_rotation` 在通道帧里的 zxy-y-euler 恰为 θ，所以 `shift_euler = -θ` **精确抵消它**。
实测：`fixed_global` 与 `data["global_rot_6d"][:2]` **逐位相等**（对训练通道的测地距离
**0.0000**，max 6e-4）。

**所以模型收到的条件输入是逐位正确的。** 第六次修订引用的「9.46° → 100.49°」是在抵消**之前**
量的，不是去噪器实际收到的东西；那个「97.26°」就是 `|θ|` 折到 [0,180]。

而我提的两个候选 `A1 = Z M Zᵀ`、`A2 = Zᵀ M Z` **都不过闸门**：两者都把条件推到离训练分布
**82.41°**（p95 168.9）。**(a) 会破坏条件输入。** 闸门 1（条件身体直立度）根本没有分辨力：
as-is 与「完全不施加」到 4 位小数相同（7.62°），因为在共轭通道里左乘 `R_y` 是绕该通道的 z 转，
而直立度统计对它不变。

### B. 倾倒进的是**输出**，不是输入

`mat[:3,:3] = inv(shift)`，所以 `code/test_infbagel_hosi.py:156`
（`global_jrot_mat = mat[:, None, None, :3, :3] @ global_jrot_mat`）是**合法的 un-shift**。
但评测在训练之外**额外**施加了 `desired_rotation`（`|θ|` 均值 122.2°，训练从不施加），它对
条件被抵消、对**输出**没有被抵消。于是导出的旋转通道带着一个 `|θ|` 量级的倾倒。

实证：导出 root 在粗帧 0 与原始 GT 相差一个转轴离原始帧 z 仅 **0.017° 均值**（p50 0.000，
max 1.133）的旋转，其角度与 `|θ|` 的相关系数 **r = 0.982**；剥掉这一个 `R_z(-φ)` 后，导出的
**GT 历史帧**回到原始 GT 的 **0.012°** 之内。**所以导出的 GT 历史帧的倾倒是纯人工制品。**

### C. 但权重确实坏——一个与坐标口径无关的证明

窗口内相邻粗帧 root 的相对旋转大小是**共轭不变**的（同窗口所有帧共享同一个 line-156 因子），
所以它不可能被任何帧约定解释。全 375 条、第一个窗口（`history_frames = 2`）：

| 粗帧步 | 均值 (°) | p50 | p95 |
|---|---:|---:|---:|
| 0→1（两帧都是 GT 历史） | 1.74 | 0.95 | 6.19 |
| **1→2（最后一帧历史 → 第一帧生成）** | **122.68** | **118.86** | **154.64** |
| 2→3 | 2.63 | 1.42 | 8.57 |
| 3→4 … 6→7 | 2.76–4.24 | 1.08–1.66 | 8.57–15.04 |

**给它逐位正确的旋转历史，模型第一帧生成就甩开 122.68°，然后又平滑下去。** 这是主 session
独立复算的（agent 报 122.7）。配套：全 2271 个窗口逐窗取最优 `R_z` 也只能把 90.58° 压到
**50.35°**（GT 10.11）；只从 GT 条件帧反解 φ（非循环）后，历史帧落在 GT 的 0.012°，而生成帧
2–15 去到 **121.10°** 直立度、离 GT **125.01°** 测地。

**结论：旋转通道没有被学会。** 这不是驱动缺陷能解释的。

### D. 上游根因：`zup_to_yup` 是多余的，它把窗口正则化打废了

`human_orient.npy` **本来就是 y-up**。实测（前 20 万帧）：

| 量 | 值 |
|---|---:|
| 原始 orient `arccos(R[1,1])`（body-y 对 world-y） | 均值 14.21°，p50 **6.59°** |
| 原始 orient `arccos(R[2,2])` | p50 112.38° |
| **共轭后**通道 body-y 对 world-y | 90.15° |
| **共轭后**通道 body-z 对 world-z | **9.96°** ← 通道的竖直变成了 z |

所以 `code/datasets/infbagel.py:91,94`（与 `code/priors/hsi/data.py:58-59`）对**旋转通道**做的
`zup_to_yup` 是**多余的共轭**，之后通道的竖直是 ±z 而关节通道的竖直是 +y。

**后果是窗口正则化基本失效**：

| `|zxy` 的 y-euler`|` | 均值 | p50 | p95 |
|---|---:|---:|---:|
| 共轭后通道（训练实际归零的那个） | 6.77° | 3.64° | 20.30° |
| 原始 y-up orient（真正的 heading） | **90.46°** | **90.71°** | 175.38° |

`:514-516` 想去掉的是初始 heading，但它在一个 y 是水平轴的帧里取 y-euler，于是**只去掉了
90.5° 里的 6.8°**。而 `:534` 把同一个小矩阵用到关节上，所以**两个通道都没有被 heading 正则化**。

**假设（有机制、有量、尚未证伪）：** 训练因此从未提供过 heading-canonical 的窗口——每个窗口
以任意绝对朝向（约 90° 均值、p95 175°）出现。位置通道还有 `mat` 的平移归一化托着，旋转通道
则要在没有任何 canonical 帧的情况下学绝对朝向。这与「关节通道大致直立、旋转通道完全没学会」
的观测一致。**这是一个假设，不是测量结论。**

### E. 因此优先级要变

**在训练表示修好之前，打磨 evaluator 没有收益。** 导出侧的倾倒是人工制品、可以修，但修完数字
不会变好，因为缺陷在 checkpoint 里。`results/lingo_hsi/c_guided_v3` 与 B 各列的 FK 几何数字
**依然全部作废**，但**重跑不能救它们**。

### F. 欠用户的决定（已更新）

1. **真正的修法**：去掉 `datasets/infbagel.py:91,94` 与 `priors/hsi/data.py:58-59` 多余的
   `zup_to_yup`，以及评测侧 `:1610` 的补偿性 `yup_to_zup`，让两个通道活在同一个 y-up 帧里。
   顺带修好窗口正则化。**但它改变训练表示 ⇒ 强制重训**，且 `code/priors/core/window_codec.py`
   是冻结的跨分支契约，必须与 HOI 分支一起动。**这是用户的决定。**
2. **跨分支通报现在更紧急**：`code/test_infbagel_hosi.py:156` 与其中的 `get_mat` 是共享的，
   同一个倾倒**很可能出现在每一个 HOI 模型侧 FK 数字里**（未在 HOI 上验证）。
3. **补上缺的那个数**：旋转通道的失败是否早于蒸馏。`b_guided_shard8` 与 `c_unguided`
   **没有 motion export**，所以无法从已有产物回答。需要批一次 B checkpoint 的 export-only pass
   （B 是 diffusion 阶段模型，`sample_type` 与 checkpoint 路径需用户确认）。

### G. 本节未做

- **未实施 (a)**（测量表明它会破坏条件输入）；未跑任何 GPU；未改任何源文件。
- 未改 `code/priors/core/`、`code/test_infbagel_hosi.py`、`datasets/infbagel.py`、
  `priors/hsi/data.py`。未 commit、未 tag、未分配 run id。
- 未验证窗口 ≥1 的条件一致性（需带插桩的 GPU run）。w≥1 的历史是上一窗口被 line-156 倾倒过的
  输出，而归零 root 的 y-euler 无法撤销一个 tip，故预期也出分布——**未验证**。

## 2026-08-18（同日第八次修订）：坐标表示修正已落地，六道闸门全过；并更正本文档两个数

用户已明确授权改动冻结的 `code/priors/core/` 与共享的 `code/test_infbagel_hosi.py`。改动已落地，
证据在 `.claude/scratch/repfix/FINAL_GATES.txt`。**未 commit、未 tag、未训练、未跑 GPU 训练负载。**

### A. 先更正第七次修订的两个数（我错了）

1. **`arccos(R[2,2])` p50 不是 112.38°，是 92.81°。** 我当时只取了 `human_orient.npy` 前 20 万帧；
   按 37 步长遍历全部 291.6 万帧得 mean 92.52 / **p50 92.81**（n=78805）。92.8° 才是正确读数，
   含义是「heading 任意」而非「竖直是 z」。第七次修订表里的 `arccos(R[1,1])` p50 6.59 是对的
   （全量 7.47），结论「资产本来就是 y-up」不变。
2. **「`zup_to_yup` 是多余的」只对 LINGO 成立，对 OMOMO 不成立。** `data/train`/`data/test` 的
   `human_orient` **确实是 z-up**（`R_stored = Rx(90)·G_yup`），共轭在那里与被共轭的模板恰好相消。
   **一刀切删掉会把 HOI 打坏 0.557 m。** 正确做法是按语料在加载时**功能性判定**世界系
   （FK 必须复现 `human_joints_aligned.npy`），而不是无条件删除。

### B. 第七次修订漏掉的关键资产：`rest_human_offsets_aligned.npy` 本身是被共轭的

它在**两个语料里**都是 `zup_to_yup(y-up 模板)`。主 session 独立核对：资产 left_hip 行
`[0.05614, -0.02347, 0.09454]` 与 `zup_to_yup(constants.rest_pelvis[1])`
`[0.05614, -0.02348, 0.09454]` **逐位相符**。

所以「只去掉旋转的共轭、不动模板」会得到 0.887 m，比现状更差。修法必须是
**root-only 世界系校正 + 把模板反共轭回 y-up**，两者一起。

### C. 训练侧 FK 一直在和一个错位 0.56 m 的骨架比（本轮最重要的发现）

`code/models/infbagel.py` 的 `loss_fk` 用 `quat_fk_torch(local_jrot_mat, rest_human_offsets)`。
共轭旋转 + 共轭模板 ⇒ FK 输出整体被 `Rx(-90)` 转过，而 GT 关节是 y-up 未转的。主 session
独立复现（4000 帧，pelvis 相对）：

| 训练 FK 构图 | 对 `human_joints_aligned.npy` 的均值误差 | 最大 |
|---|---:|---:|
| **旧**：共轭旋转 + 仓库模板 | **0.557769 m** | 1.415811 m |
| **新**：y-up 旋转 + 反共轭模板 | **0.000000 m** | 0.000001 m |

（agent 在另一组样本上报 0.565 m → 4.0e-07 m，量级一致。）

**`loss_fk` 是唯一在几何上锚定旋转链的项，而它一直在对着一个平均错位 0.56 m 的骨架算损失。**
这是「旋转通道从未学会直立」「HOIPrior 迭代多次无效」「infbagel 在 LINGO-only 上复现不出来」
三件事的同一个机制性解释。与 `HSIPRIOR_DESIGN_PRIORS.md` §4 记录的「几何项被低估约 135×」叠加：
不只是权重太小，**它测的东西本身是错的**。

**重训预注册必须记录 `loss_fk` 量级的重定标**——它按构造改变，不是调参。

### D. 六道闸门（改前 → 改后）

| 闸门 | 改前 | 改后 |
|---|---|---|
| A 通道竖直 vs 关节竖直（400 窗口） | mean 94.78 / p50 98.04° | **8.98 / 6.12**（原始资产本身 9.95/6.14） |
| A2 `get_mat` yaw vs dataset yaw（新增） | mean 86.56 / p50 85.34° | **0.000 / max 0.017** |
| B shift 轴 vs ±y | 0.000 | 0.000（两侧精确） |
| B 被去掉的 \|shift\| | p50 3.56° | **p50 90.50°**（真实 heading p50 92.53） |
| B shift 后髋方位角 vs 其圆均值 | p50 85.61° | **p50 0.363 / p95 3.60** |
| C GT→表示→生成臂链 vs 关节数组 | 2.1423 mm @s=1 / 6.6835 @s=3 | **2.1407 / 6.6829**（不更差） |
| D GT 臂足部仍进接触带（24 episode/24 场景） | — | `fs_nemf` 非零 **24/24**，`skate_ratio` 0.2183 |
| **E 旧 dataset vs `core/` 旋转测地** | **mean 8.064 / p95 24.99 / max 163.41°** | **mean 3.6e-06 / max 2.3e-05** |
| E 同上，关节通道 | max 4.627e-01（归一化） | **1.49e-07** |
| F 测试 | 148 passed | **159 passed**；全仓 **281 passed / 3 skipped**（主 session 复核，需设 `INFBAGEL_PYTHON`） |

FK 帧测试：LINGO `共轭旋转+仓库模板` **0.565 m** → `y-up 旋转+反共轭模板` **4.0e-07 m**；
OMOMO 两种写法都精确（7.8e-07），即修法对 HOI 是无损的。

闸门 E 改前比第七次修订记录的更糟：不是 5.13°/16.08°，而是 **mean 8.064° / max 163.41°**，
外加关节通道一处 0.4627 的分歧（第七次修订未提）。**该错配此前没有任何测试覆盖**，本轮补上了。

### E. `ground-truth-v3` 不动（已实测确认）

agent 论证 `test_infbagel_lingo_hsi.py` 的改动 hunk（66-80、96-104、1396、1604-1625）与 GT 臂
（161-355）互不相交；`GroundTruthSource` 定义在该文件 `:161` 内、直接读 `DATASET_ROOT`，
**不经过 `code/priors/hsi/data.py`**，所以那个文件的改动不触及 GT 臂。

主 session 实测验证：当前工作树重算 20 条 GT，与 `ground-truth-v3` **在每一个数值键上逐位相同**，
仅 `schema_version` 4 vs 2。**agent 报告里「重算会偏离 v3（boundary_jerk ≤23%）」这条复现不出来，
应予否决**；那条把原因归给 `metrics.py` 的未提交改动，但 `ground-truth-v3` 本来就是在那份
`metrics.py` 下生成的。**结论：GT 参考行完好，不需要重做。**

### F. 作废的产物

- **模型侧全部作废且不可重算**：`b_guided_shard8`、`b_unguided_shard8`、`c_guided{,_v2,_v3}`、
  `c_unguided`、`latency_*`、`smoke-*`、`rds_gate_smoke`、`shard_bitwise`。那些 checkpoint 拟合的
  132 维通道已不存在，**只有重训能取代它们**。保留它们作为「修复前」的对照。
- HOI 分支封存的 D2-* 各行同理：OMOMO 的通道从 `G_yup·Mᵀ` 变为 `G_yup`。
- 磁盘上的数据资产（`norm.npy`、`human_*.npy`、`transl_aligned.npy`、`Scene*`、split manifest）
  **未改动**，输入哈希不变。

### G. 欠用户的决定与我的建议

1. **Phase 1C 基线**：agent 提议加一条 `representation='legacy_75efccc'` 通道专门用于评测已发布
   checkpoint。**我建议不加**：那是为保住一个坏表示而维护的死代码。「修复前」的数字**已经存在**
   （上面 F 节那些行），GT 参考行完好，所以 before/after 对照现成，只要别删旧行。
2. **`core/window_codec.py` 的同款改动要落到 `phase/01b-hoi`**（`AGENTS.md`「Final integration is
   a graft」，冻结测试自己也这么要求）。**本分支未触碰那个分支**；该指令写进交接报告，由新 session 执行。
3. **`code/datasets/utils.py` 是第五个被改的文件**（新增按语料判定世界系的帮助函数，是 `datasets/`
   与 `priors/hsi/` 唯一的共同上游）。需用户追认。
4. **`tools/audit_prior_data.py:67-69` 复制了旧的正则化逻辑**，现在与 dataset 不一致，其归一化审计
   在更新前无效。属授权范围外，未改。
5. **`tests/hsi/test_representation_frame.py`（新增 365 行、11 个测试）目前未跟踪**，
   `tools/experiment.py` 会因此判定工作树脏而拒绝可上报的 run。需要 `git add` 后才能开跑。
6. 遗留、与本轮无关、但已量化：`interpolate_joints` 用 `linspace(0,T-1,T*scale)` 而 `interp_jrot`
   按 `1/scale` 步进，interp_s=3 时仅 pelvis 就漂 4.86 mm 均值 / 43.3 mm 最大。

### H. 已知局限

- heading 正则化对水平身体退化：3.30% 的窗口残余方位角 >5°（其躯干倾角中位数 57.5°），
  1.13% >45°（中位倾角 87.7°，即躺卧）。任何单轴 heading 都有此性质；直立窗口 p50 0.387°。
- 未验证 LINGO 镜像半区（序号 ≥9725）行为一致；探针与抽样取自非镜像半区。
- 闸门 C 残余的 2.14 mm **全部**来自 `SMPLX_JOINTS_28` 的槽位 25/27（仓库数组里各 28.018 mm），
  关节 0–21 精确。属既有问题，与本轮无关。

## 2026-08-18（同日第九次修订）：归一化盒分裂已修复，`mask_fk` 补齐；两项代价登记为设计先验

用户已批准本轮的两处改动并指定了方案。改动已落地于 `code/datasets/infbagel_mix.py`
与 `code/models/infbagel.py`，新增 `tests/hsi/test_normalization_box.py`（8 个测试）。
**未 commit、未 tag、未分配 run id、未启动任何训练或评测负载。** 证据在
`.claude/scratch/normbox/TABLES.md`（前置调查）与 `.claude/scratch/normfix_accept.json`（本轮验收）。

### A. 缺陷：同一个量，归一化用一个盒，反归一化用另一个盒

`lingo_only` 下的两处证据：

1. `code/datasets/infbagel.py:301-306`：`InfBaGelDataset.__getitem__` 用
   `<lingo_folder>/norm.npy`（OMOMO 常量，sha1 `8dac1c678d2a`）归一化 `joints`。
2. `code/datasets/infbagel_mix.py:164-169`（改前）：mix 的 `unified_min/max` 从
   `<lingo_folder>/norm_inter_and_loco__16frames.npy` 载入——一个 (2,3) 的盒，逐轴 range 只有
   `norm.npy` 的 **[0.39924, 1.04313, 0.39456]**。

于是 `mix.denormalize_torch(sub.normalize(v)) = S v + c`，
`S = [0.39924, 1.04313, 0.39456]`，`c = [-0.03386, -0.07942, -0.12771] m`（水平常数偏移 0.1321 m），
而不是 `v`。归一化器与反归一化器对同一个量不一致，**没有任何读法能把它读成有意设计**，所以它是缺陷。

### B. 四个候选盒（全部 1,343,667 条 v3-train 窗口实测）

| 候选 | box min | box max | 占 [-1,1] 比例 (x,y,z) | 越界值 | 越界窗口 | 逐轴方差失衡 | 水平 SNR≥1 末步（共 500） |
|---|---|---|---|---:|---:|---:|---:|
| **(a) `norm.npy` 行 0-1 = OMOMO（原归一化器）** | [-3.24436, -0.04858, -2.43991] | [3.41397, 2.15093, 4.53536] | 0.436 / 0.939 / 0.422 | 2,922,403 | 168,247 (12.52%) | 54.2× | 10 / 11（竖直 89） |
| (b) `norm_inter_and_loco__16frames.npy`（原反归一化器，LINGO 自己的盒） | [-1.32912, -0.13010, -1.09040] | [1.32912, 2.16427, 1.66177] | 1.092 / 0.900 / 1.069 | 399 | 128 (0.0095%) | 7.9× | 29 / 32（竖直 85） |
| (c) `data/test/..._new.npy` (4,3)，逐元素并集 | [-3.24436, -0.13010, -2.43991] | [3.41397, 2.16427, 4.53536] | 0.436 / 0.900 / 0.422 | 240 | 60 (0.0045%) | 49.9× | 10 / 11（竖直 85） |
| (d) v3-train 现测 min/max | [-1.45022, -0.13343, -1.22812] | [1.45249, 1.93215, 1.71323] | 1.000 / 1.000 / 1.000 | 0 | 0 | 11.7× | 26 / 29（竖直 94） |

(c) **逐位**等于 `infbagel_mix.py:143-146` 那段被注释掉的 `np.minimum`/`np.maximum` 的输出，即作者为
OMOMO+LINGO 混训准备、后来放弃的并集归一化器。(b) 的载入出自 `262f2d9`，代码与历史里都没有理由说明。

### C. 为什么选 (a)：这是 bug 修复，不是创新

调查列出的四条理由都成立（`priors/core/contracts.py:44` 指名 `norm.npy` 并写着 "never recompute"；
`priors/hsi/data.py:151` 与 `core/window_codec.py` 两个方向都已在用它；Phase 2/3 mixer 需要两位专家
共用一个位置盒；(2,3) 形状隐患消失）。**但决定性的理由是另一条**：本分支的协议是「复现 InfBaGel、
只修作者的 bug，把创新留给后续 HSIPrior 迭代」。**盒的选择是设计决策**，而作者的设计就是「OMOMO 的盒
跨语料共用」。换盒是创新，而且会污染复现口径。因此 (a) 是唯一让本轮仍然是 bug 修复的选项。

旁证：`tests/core/test_expert_contract.py:84` 早已断言 `data/dataset/norm.npy` 与
`data/train/norm.npy` 逐位相同、且契约文本含 "never recompute"。修改后代码才真正服从这条契约。

### D. 登记为创新阶段的设计先验（(a) 的两项代价，不在本轮消化）

1. **12.52% 的 v3-train 窗口带有竖直分量低于 −1 的值（最差 −1.077）**，因为 OMOMO 的地板
   （−0.0486 m）高于 LINGO 的（−0.1334 m）。**任何地方都不裁剪**——已核实无 `clip_denoised`，
   `models/infbagel.py` 里唯一的 `clamp` 是 `posterior_variance.clamp(min=1e-20)`。仿射映射可逆，
   越界值被精确表示，所以这不是数据损失，而是网络输出范围的先验偏置。
2. **水平两轴只占用 [−1,1] 的 ±0.436 (x) 与 ±0.422 (z)**，于是 `loss_jpos` 对 1 m 水平误差的定价
   只有 1 m 高度误差的 **1/3.03 与 1/3.17**；逐轴方差失衡 **54×**；水平方向的 diffusion SNR 在
   500 步中于 **t=10/11** 就跌破 1，而竖直方向在 **t=89**。在一个以位移为主的 locomotion 语料上，
   这是把主信号定价过低。**这是后续「有意为之」的换盒（选项 (d)，重新在 v3-train 上取盒）的最强候选**，
   必须先写下来，好让那次改动是一个决定而不是一次重新发现。

### E. 验收（四项全部复现，`.claude/scratch/normfix_accept.json`）

| 检查 | 改前（原样） | 改后 |
|---|---:|---:|
| 200 条真实窗口，world 误差 max | **1.026292 m** | **4.768e-07 m** |
| 同上，local 误差 max | 1.164959 m | 4.768e-07 m |
| mix 与 sub 反归一化之差 max | 1.164959 m | **0.0（逐位相同）** |
| `loss_fk` @ 完美预测（256 窗口真实 batch，未加掩码口径） | **0.024817798** | **6.2424e-14** |
| 同上，生产口径（已加掩码，real `p_losses`） | 0.02487195 | **6.3293e-14** |
| 五项基础 loss @ 完美预测 | 全 0.0 | 全 0.0 |
| evaluator 解码 `S` / `c` | [0.39924, 1.04313, 0.39456] / [-0.03386, -0.07942, -0.12771] | **[1,1,1] / [0,0,0]（精确）** |
| `_compute_occ` 主网格查询中心偏移（frame 0） | **0.13212 m**（p50=p90=max，常数） | **max 2.384e-07 m** |
| `occ_temp` frame 15 系统性偏移（同一 256 batch） | max 0.7319 m，>0.6 m 占 3.91% | **0** |

`loss_fk` 从 0.0248178 掉到 6.2e-14、而五项基础 loss 在两种口径下都恰好为 0，说明
**旋转帧缺陷（`3ded4eb`）与本缺陷两者合起来解释了整个 `loss_fk` 地板，没有剩余项**。

`occ_temp` 在改后仍有 0.079–0.137 m 的残差，那是 `_compute_occ` 对 `x_denorm` 故意施加的
±0.1 m 增广（`models/infbagel.py:211-212`），不是缺陷；系统性成分（改前减改后）才是缺陷，
它归零。全语料口径下该系统性偏移的 p100 是 0.9774 m、>0.6 m 占 4.363%（`TABLES.md` §5）；
256 窗口 batch 只到 0.7319 m / 3.91%，与之一致。

`ground-truth-v3` 不动：`GroundTruthSource`（`test_infbagel_lingo_hsi.py:161-216`）直接读
`DATASET_ROOT` 的原始 `.npy`，类体内不出现 `normalize`/`denormalize`，也不引用 `dataset`/`sampler`。

### F. `mask_fk`：`p_losses` 补上历史帧掩码（与归一化修复分属两个改动）

`code/models/infbagel.py:846-848`（改前）建出 `mask_fk` 后从不使用，而 `consistency_loss`
的 `:404-405` 用了。这是作者代码内部的不一致，所以在修复范围内。它要紧的理由是**可比性**：
C 从 B 蒸馏，掩码不加时 B 的 `loss_fk` 在同一几何上恰好是 C 的 **0.875 倍**
（= (16−2)/16；实测比值 0.8749999472，历史帧精确时），于是两阶段的 `loss_w_fk` 互相之间无法解读。

实质上被掩掉的帧就是 `auto_regre_num` 个历史帧：采样时 `set_fixed_points` 每一步都会覆写它们的
输出，而 `mask_inv` 早已把它们从全部五项基础 loss 里排除——所以未加掩码的那一项是唯一在监督
一个没人读的输出。

**重训预注册必须记录 `loss_w_fk` 的重新推导，并把它归因到这两处改动中的哪一处**：两者都会重定标
`loss_fk`（归一化改动改的是几何，掩码改动改的是分母），混在一起就无法归因。本轮两处改动分处
两个文件，diff 天然可分离。

### G. 波及面比预期大：18 个人体位置归一化点，且是双向的

一处改动修好了所有读 `unified_*` 的点：

- `code/models/infbagel.py`：AST 计数 18 个 `(de)normalize_torch` 调用点，其中 7 个是
  `is_object=True`（用 `unified_obj_*`，两侧本来就一致，从未错），**11 个是人体位置点**
  （`:136`、`:210`、`:378`、`:384`、`:486`、`:579`、`:689`、`:824`、`:828`、`:958`、`:1031`），
  含 `loss_fk` 的 FK（`:824`/`:828`）与 `_compute_occ` 的场景查询（`:136`/`:210`）。
- `code/test_infbagel_lingo_hsi.py`：`:1426`（解码）、`:1477`、`:1514`（编码回通道）。
- `code/test_infbagel_hosi.py`：`:145`、`:530`（解码）、`:604`、`:647`（编码回通道）。

**注意 `:604`/`:647` 是正方向**：自回归续写把世界米制的历史用 mix 的盒 `normalize_torch` 回通道，
再交给 `set_fixed_points`。所以缺陷不只压缩了输出的读数，还把注入模型的历史条件整体缩放了，
整条自回归链都在错的尺度上。本文档第八次修订 F 节列出的模型侧行，因此在「表示已作废」之上
**还叠加了一层约 0.4× 的水平压缩**。`:1427`/`:151`/`:531`/`:607`/`:650` 走 `unified_obj_*`；
`lingo_only` 下 LINGO 不带物体、216: 通道恒为 0，那几处本就是空操作。
三个文件合计 **18 个人体位置点 + 14 个物体点**，一处改动全部修好。

`_compute_occ` 的 goal 查询（`:158`）本身不经反归一化，但对既非 `need_pelvis_dir` 又非
`is_object` 的样本会回落到 `mat_for_query` 的中心，那部分同样被修好（实测 p50 0，max 0.1321 m）。

### H. 测试、性能与遗留

1. **`tests/hsi/test_normalization_box.py`（新增，8 个测试）**。缺陷能活下来是因为
   mix 的 `unified_*` 反归一化器**零测试覆盖**：既有的
   `test_representation_frame.py:228`、`:248`、`:267` 只走 sub-dataset 自洽的那条路，两个对象之间的分歧
   它看不见。新测试用真实 `InfBaGelMixDataset.__init__`（`lingo_only`，8.6 s / 11.7 GB 峰值）
   断言盒的数组本身、跨对象双向往返、以及两阶段 FK 语句的 AST 同构。
   **证伪已做**：把两个源文件回退到原样后，8 个中 **6 个失败**（另 2 个是「物体行不变」与
   「0.875 代数」，按设计本就与改动无关）；再逐位恢复，patch 与备份 `diff` 一致。
2. **完整 authority suite**：`pytest tests -q --tb=short --no-header`（需
   `INFBAGEL_PYTHON`，`OMP_NUM_THREADS=4`）→ **289 passed / 3 skipped**，即基线 281/3 加本轮 8 个。
   `tests/core/test_contract_freeze.py` **4 passed**，未触碰 `core/`，未重新 pin 哈希。
3. **性能**：归一化改动只改四个常量的取值（且改为别名 `lingo_dataset.min/max`，与
   `not lingo_only` 分支既有写法一致），不改形状、算子或通信，无需基准。`mask_fk` 改动确实
   改变 per-step 计算——布尔索引替代密集 MSE。micro-batch 256、fwd+bwd 实测
   **235 µs → 1183 µs（+947 µs）**，峰值分配 **+2.4 MiB**；对照本文档 §1 实测的
   micro-batch 256 单步 **0.773 s**，为 **+0.12%**，回退档 128 时 +0.21%。可忽略。
   另注：`consistency_loss` 一直在付这份开销，run C 的 10 h 25 m 已包含它。
4. **未修、已记录**：`tools/audit_prior_data.py` 本轮未动。更正第八次修订 G.4 的一处措辞：
   该工具用的是 `root/"norm.npy"`（`:55-56`），**盒一直是对的**，所以它报的 `|normalized| > 1`
   比例本来就是在测上面 D.1 那项代价；它失效的原因只是 `:68` 的 `zup_to_yup(orient)` 走的是
   旧的旋转帧（第八次修订 C 节），与本轮的盒无关。
5. **未跟踪文件**：`tests/hsi/test_normalization_box.py` 与第八次修订的
   `tests/hsi/test_representation_frame.py` 一样，在用户 `git add` 之前会让
   `tools/experiment.py` 判定工作树脏并拒绝可上报的 run。

---

## 2026-08-19（第十次修订）：`loss_w_fk` 50 → 3——它自己就足以复现 122.68° 的失败

本节是**预注册参数的修正案**，也是 run B 启动前的最后一项。改动落在
`code/config/config_train_hsi_b_lingo_full.yaml`（`loss_w_fk`、`exp_name`、layout 注释）与
`code/config/config_train_hsi_c_lingo_cm.yaml`（`exp_name`、`ckpt_path`）。证据在
`.claude/scratch/lossfk_reweight/FINDINGS_AB_20260819.md`、`AB_TABLES.md`、`ext_4000/`
与 `.claude/scratch/layout_probe_20260818/consolidated.txt`。第九次修订第 5 条已作废：
两个测试文件均已随 `a4c979c` 提交，工作树干净。

### A. 决定性事实：修完三个表示缺陷之后，`loss_w_fk: 50` 单独仍然复现失败

配对 A/B：同 seed 同数据顺序，update 1 在全部 8 个 rank 上七项损失**逐位相同**，各跑 4000 步，
唯一差别是 `loss_w_fk`。模型**自己预测的**旋转通道（`predicted_noise[:, :, 84:216]` 过仓库自身
`mat @ rotation_6d_to_matrix → quat_ik_torch → quat_fk_torch` 链，根平移置零）离世界 +y 的角度：

| update | 根 R 的 +y 列 w=50 | w=3 | 真值 | FK pelvis→neck w=50 | w=3 | 真值 |
|---|---:|---:|---:|---:|---:|---:|
| 250 | 85.8 | 125.2 | 7.8 | 35.2 | 43.5 | 6.1 |
| 750 | 112.4 | 9.4 | 7.8 | 41.7 | 4.0 | 6.1 |
| 1500 | 104.97 [103.76, 107.53] | 10.53 [9.62, 11.28] | 7.9 | 36.9 | 3.9 | 6.2 |
| 4000 | **122.48 [121.30, 123.61]** | **7.13 [6.88, 7.23]** | 7.83 | 27.15 | 4.65 | 6.13 |

**122.48° 与已登记的、上一个训练好的 checkpoint 的 122.68° 相差 0.2° 以内**（见
`hsi-rotation-channel-not-upright` 与 2026-08-18 第七次修订）。也就是说：`3ded4eb`、`a4c979c`、
`bcb4ad4` 三个表示缺陷全部修好之后，**仅这一个权重就足以把失败复现出来**。这是本节最重要的一句。
w=50 是**单调恶化**（85.8° → 122.48°），其 6-D 第一列范数从 1.01 漂到 0.87；w=3 在 update 750
前就切进正确姿态并稳在真值同级，列范数升到 0.96（真值 1.00）。限制在 t<100 的样本上同样成立
（123.5° 对 7.1°），故不是高噪声区的假象。

### B. 为什么这一项**结构上看不见**这个缺陷

`loss_fk` 只算 4 个手部 + 4 个足部关节（`models/infbagel.py:846-857`），而运动链可以从**任意**根
姿态出发、靠上游补偿把这 8 个末端摆到正确位置。实测于 update 4000：

| | 手部误差 | 足部误差 | `loss_fk` | 根姿态 |
|---|---:|---:|---:|---:|
| w=50 | 0.2130 m | 0.1690 m | 0.02465 | 122.48° |
| w=3 | 0.2126 m | 0.1714 m | 0.02487 | 7.13° |

**末端精度相同、`loss_fk` 相同（差 1% 以内）、根姿态差 115°。** 花 50 倍代价买的那一项，对这轮修复
所针对的缺陷是盲的。补充一条结构性发现：`loss_fk` 对关节 **10、11、12、15** 的旋转通道给出
**恰好为零**的梯度（132 维中的 24 维），这些维度在任何权重下都只由常幅 L1 监督。

### C. 梯度侧：46 倍推力里 45.9 倍是正交的，而 `loss_jrot` 无法反抗

`ext_4000`（w=50，4000 步，8 rank）：

| update | \|g_base\| | 50·\|g_fk\| | FK 梯度占比 | cos(g_base, g_fk) | FK 占优的坐标比例 |
|---|---:|---:|---:|---:|---:|
| 1 | 12.25 | 13433 | 0.9991 | +0.002 | 0.914 |
| 1000 | 1.33 | 131.1 | 0.9899 | −0.038 | 0.906 |
| 2000 | 1.39 | 159.8 | 0.9914 | +0.014 | 0.909 |
| 4000 | 7.59 | 132.5 | 0.9458 | +0.002 | 0.903 |

补上的关键量：`cos(g_jrot, g_fk)`（84:216 输出通道）+0.0065（u1）→ +0.054（u1000）→ **+0.067**
（u4000，跨 rank [+0.063, +0.072]）。该 473,088 维空间里随机方向的基线是 **0.00116**，故它统计上
确实为正、几何上可忽略——FK 对旋转通道的推力里只有 **0.45%** 沿 `loss_jrot` 想去的方向。u4000
分解：FK 的旋转通道推力是 `loss_jrot` 自身的 **46.0 倍**，其中对齐 3.1 倍、正交 **45.9 倍**。

而 `loss_jrot` 结构上无法反抗：它是 L1，`|g_jrot|` **恒等于** `1/sqrt(473088)`，与通道停滞多久无关。
符号梯度不会因为学不动而变响。同期 `loss_jrot` 只降 2.7 倍（0.6356 → 0.2334），`loss_jpos` 降
126 倍（0.4438 → 0.00353）。

### D. 为什么取 3 而不是 2.86——朴素平价值被污染

对**整个** base 目标读出的平价权重 2.86 不可用。u4000 时 `|g_object|` 占 `|g_base|` 的 **99.5%**，
且在参数空间从 0.473 涨到 7.479（16 倍），而其**输出空间**范数几乎不变（0.01420 → 0.01003）。
成因从源码读出而非推断：`lingo_only` 下每个样本都走无物体分支，通道 216:232 携带一个**恰好恒定**
的目标（`datasets/infbagel.py:521-523`、`:633`，再 `normalize(0)`），两个 L1 项对着一个可达的常数
维持不衰减的梯度。有意义的平价权重是：**1.09**（对 84:216 上的 `loss_jrot`）、**0.33**（对
jpos+jrot 全参数）；对齐信号的收支平衡点是 **16.1**。取 3 即 u4000 实测的梯度平价值。

w=3 在所有已记录项上都不劣：五个基础项全部更好（jpos 0.68×、jrot 0.52×、otrans 0.54×、
orot 0.83×、contact 0.86×），两个姿态量更好，6-D 列范数更好，`loss_fk` 打平（1.009×）。
**w=50 没有赢下任何一项。** 且 16.7 倍的权重削减对 `loss_fk` 本身的影响在每一步都在 5% 以内。

### E. 复现保真度的反驳意见，及为何它在这里不成立

「作者的值是 50，改它就不是复现」——这条在归一化盒那里成立，在这里不成立，理由是**被乘的量变了**：
该项的不可约下界从 0.5614 m² 变成 6.2e-14，且修复前它是**被错误的旋转最小化**的（FK 最优补偿代价
是 jrot 0.41，加权买 26.83 付 0.41）。所以**保留这个数字才是改动，而不是不变**。旁证：另一个同样
取 50 的几何项 `loss_w_obj_pts`，在 4000 步里 `loss_object` 全程为 `None`，B 早已偏离更远。

### F. 本修正案**没有**确立什么

4000 步是 146,255 步的 **2.7%**，且落在 2,000 步 warmup 之内——只定方向与量级，不定收敛值。
粗姿态正确之后 FK 压力可能更重要，而两个 arm 都没进入那个区间；若满预算下末端精度不理想，
下一个取值是 **16.1**，那是一个新实验而不是微调。单 seed 42、单数据顺序，未给跨 seed 区间。
姿态量本身饱和：它能区分「坏」（90–122°）与「合理」（4–10°），不能在合理带内排序——w=3 的
4.65° 低于真值 6.13° 只说明姿势更居中平滑，不是超过真值。

### G. layout 与 OMP：改启动，不改 layout

已登记的 0.71223（4×512）与 0.96221（8×256）**两者都是在 `OMP_NUM_THREADS` 未设的条件下测的**，
run `p1-hsi-b-lingo-full-s42-20260814` 自己也是——其 `launch.sh` 只有 `CUDA_VISIBLE_DEVICES`，
这就是它 0.7305 s/update 的原因。按同一协议（每 arm 三次交错 40 步）重测两个 layout × 加帽/不加帽：

| layout | OMP=4 | OMP 未设 | 加帽后 146,255 步 |
|---|---:|---:|---:|
| 8 × 256 | **0.38674** | 0.92446 | 15.71 h |
| 4 × 512 | **0.52189** | 0.71844 | **21.20 h** |

未加帽那一对复现了已登记的那一对，故该修正案是在一个不适用于加帽主机的区间里测出来的，加帽后
排序反转。**但 layout 不变**：改 rank 数在本代码库不是优化中性的——同 effective batch 下 update 1
的全局梯度范数差 **4.60%**，而同 layout 内 trainer 逐位确定，根因是 `models/infbagel.py:1257` 按
sample 0 的 timestep 覆写整个 rank 批次的 `cfg_scale`，叠加 `seed+rank` 分 rank 播种。速度不构成
改它的理由，5.5 小时也不。加帽本身逐位相同，故净收益是 29.6 h → **21.2 h**。

### H. 裁剪：仍不加，但理由已经换了

`|g_base|` 在参数空间于 u1000→u4000 间涨了 **5.7 倍**（1.33 → 7.52），其中 99.5% 来自 §D 那个退化
的恒定物体目标，全程无裁剪。但 **B 的 post-warmup DDP 平均范数分布仍未实测**，C 的不能替代（C 是
consistency 路径，且 2026-08-16 §C 的分布是它自己的）。在这个状态下取阈值，正是 2026-08-13 §E
点明的那个错误。本 run 带 `+log_grad_norm=true` 启动，使下一次决定有据可依，且成本为零。

### I. 运行目录改名，避免覆盖已封存工件

`exp_name` 改为 `hsi_b_lingo_full_v2` 与 `hsi_c_lingo_cm_v2`。原名的输出目录承载
`p1-hsi-b-lingo-full-s42-20260814` 与 `p1-hsi-c-lingo-cm-s42-20260816` 两行 registry 记录按路径与
sha256 引用的工件（B 侧 12 个 checkpoint 加 `metrics.json`），`AGENTS.md:126-127` 禁止覆盖既有结果，
且它们是**不可重算**的修复前「before」侧。C 的 `ckpt_path` 同时重指到修复后的 teacher——从旧
checkpoint 蒸馏只会忠实地保住它那个 122° 的旋转通道。C 的 `loss_w_fk` 有意保持 1：
`consistency_loss`（`models/infbagel.py:363-370`）是对 teacher 的单项、232 通道全覆盖，不存在会被
饿死的 `loss_jrot`，且 1 已经低于 B 实测的 1.09 旋转通道平价值。

### J. 登记为设计先验的第三项候选

第九次修订登记的两项（`norm.npy` 的垂直越界 12.52%；水平轴只占 ±0.436/±0.422 导致水平误差被以
1/3.03、1/3.17 定价）仍然成立，现在加上第三项：**`lingo_only` 下 232 个通道里有 16 个携带恰好恒定
的目标**，两个 L1 项将对着零信息生长梯度路径长达 146k 步。三项都是「有意为之的后续改动」的候选，
不属于本轮缺陷修复。

## 2026-08-21（第十一次修订）：v3 的崩溃归因于 `.eval()`，v4 配方与 3000 步预检

`hsi_c_lingo_cm_v3` 于 update 2688 中止（manifest `p1-hsi-c-lingo-cm-v3-s42-20260821`，status
`aborted`）。它一次性带了三处耦合修复，其中**只有第三处**是崩溃的原因。本节记录归因、被排除的
假设、最终配方 `hsi_c_lingo_cm_v4`，以及它 3000 步预检的全部读数。

所有诊断工件在 `.claude/scratch/cfglr_cal_20260821/`（git-ignored）。

### A. 归因：单点回撤实验

两个 1312 步 arm，除被回撤的那一处外训练条件、数据、seed、batch、评测方法完全一致。fidelity 由
`distill_fidelity.py` 测量：固定 batch 上的一步预测相对误差，与 w 探针同 seed、同 support mask、
同 boundary scaling。

| arm | vs teacher @656 | vs GT @656 | vs GT @1312 | 结论 |
|---|---:|---:|---:|---|
| v2（健康参照） | 0.04094 | 0.05814 | — | — |
| v3 全三修复 | 0.15098 | 0.16598 | — | 崩溃 |
| 撤回修复 2（target 不收 w） | 0.15357 | 0.16870 | 0.43381 | **仍崩溃** |
| 撤回修复 3（teacher/target 不 `.eval()`） | 0.04062 | 0.05774 | 0.06666 | **完全恢复** |

撤回修复 3 后 loss 在每个区块都在 v2 的 1% 之内，且 @656 的 fidelity 比 v2 好 0.7%——**而修复 1
和修复 2 仍然生效**。因此不是修复 2 与修复 3 的组合效应，是修复 3 单独致命。

### B. 机制：EMA 半衰期 13.5 步使一致性条件近乎退化

`consistency_loss` 用 `update_ema(target, student, 0.95)`，半衰期约 13.5 次 update，所以 EMA
target 几乎就是 student 自己，一致性条件近乎退化（把自己回归到自己）。**target 的 dropout 正是打破
这个退化的噪声**。`code/models/` 里没有任何 BatchNorm，故 `.eval()` 唯一改动的就是
`nn.Dropout`——它不是「卫生」，它是承重结构。本文档此前把 `.eval()` 记为三处修复中最无害的一处，
该排序被实测反转。

### C. 被排除的假设：cfg embedding 的学习率

v3 中止时的假设是 cfg embedding 在 2e-4 下发散。3000 步、四个 lr 的扫描否证了它：

| cfg 参数组峰值 lr | vs GT @656 | 相对 v2 |
|---|---:|---:|
| v2（cfg 冻结） | 0.05814 | 1.00x |
| 1e-6 | 0.16562 | 2.85x |
| 4e-6 | 0.16619 | 2.86x |
| 1e-5 | 0.16634 | 2.86x |
| 2e-4（v3 本身） | 0.16598 | 2.86x |

**200 倍的 lr 区间，结果在 0.4% 内一致，且四者都是 v2 的 2.86 倍**——lr 从来不是那个变量。反向证据
同样成立：撤回修复 3 的 arm 让该模块跑满 2e-4，权重 RMS 到 init 的 16.1 倍，loss 全程平坦。故 v4
**不设独立参数组**，cfg 与 trunk 合并为单一 Adam group，2e-4 加原 warmup 2000。

### D. v4 最终配方

保留修复 1（解冻 `student_model.module.cfg_scale_embedding`）与修复 2（target 接收
`cfg_scale=w`），撤回修复 3。teacher 与 target 保持 `init_model` 留下的 `train()` 模式。单一参数
组。`grad_clip_max_norm: 1.0` 与修正后的 pre-clip 诊断保留。teacher 仍为
`hsi_b_lingo_full_v2_epoch222`，冷启动，不从 v3 的 optimizer state 恢复。预算不变：58,678 步。

### E. 3000 步预检读数（`results/hsi_c_cand_precheck`，GPU 4-7，rc=0，非有限值 0）

fidelity，七个 student 在**同一次探针调用**内测量（t=419；跨调用会因加载数量改变到达
`_compute_occ` 的 RNG 状态而有约 0.45% 抖动）：

| student | vs teacher | vs GT |
|---|---:|---:|
| v2 @656 | 0.04094 | 0.05814 |
| **v4 候选 @656** | **0.04062** | **0.05774** |
| 撤回修复 3 @1312 | 0.05397 | 0.06666 |
| **v4 候选 @1968** | **0.05789** | **0.07030** |
| **v4 候选 @3000** | **0.06434** | **0.07554** |
| v2 @58678（已登记收敛终点） | 0.07459 | 0.08464 |
| 崩溃 arm 1e-6 @3000 | 1.12806 | 1.13007 |

候选 @3000 优于 v2 的收敛终点，且是同步数崩溃 arm 的 1/17.5。

分区间 loss 与 pre-clip 梯度：

| updates | v2 loss | 候选 loss | 候选 trunk 中位 | vs v2 | cfg 占比 | 裁剪咬住 | Adam 饱和 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-250 | 0.00122 | 0.00121 | 0.0465 | 0.90x | 0.4% | 0.0% | 0.210 |
| 1001-1250 | 0.00104 | 0.00104 | 0.0677 | 1.03x | 0.3% | 0.0% | 0.203 |
| 2001-2250 | 0.00111 | 0.00105 | 0.2218 | 1.02x | 0.1% | 0.0% | 0.150 |
| 2751-3000 | 0.00102 | 0.00105 | 0.1547 | 1.06x | 0.2% | 0.0% | 0.145 |

trunk 相对 v2 同区块中位 **1.03x**（范围 0.76-1.30x，无趋势）；warmup 结束后 trunk 中位**下降**
0.2218 → 0.1547（0.70x）；cfg 占全局范数**下降** 0.41% → 0.15%；EMA 距离 / 参数范数**下降**
2.057e-01 → 8.522e-02；Adam 饱和 0.197 → 0.168（0.86x，1.0 为完全饱和步长，持平即「除 schedule
之外没有额外加速」）；cfg 权重 RMS 达 init 的 37.4 倍且仍在增长。

`w=-1` sentinel 方向余弦（对 teacher uncond 目标）：

| checkpoint | cosine | 幅度比 |
|---|---:|---:|
| B epoch222（起点） | +1.0000 | 0.50040 |
| v4 候选 @656 / @1968 / @3000 | +0.9604 / +0.8347 / **+0.8687** | — |
| v2 @58678 | +0.7427 | — |
| 三个崩溃 arm @3000 | +0.3007 … +0.3524 | — |

B 的 0.50040 恰为理论值：sentinel 走 uncond 分支，不做 CFG 的 ×2 外推，这自校验了整条测量链。

连续 w（排除 sentinel）@3000：幅度 0.26094 / 0.31258 / 0.07456，余弦 **+0.1464 / +0.1094 /
+0.0160**（w = 0 / 2 / 4）。**三个 w 首次同时为正**，幅度是 B 的 57-63 倍。读法两点：(1) 余弦只能
在可比幅度下比较，B 自己的 +0.233 建立在 0.00457 的噪声量级响应上，不是更好的数；(2) 训练中 w 由
`sample_cfg_scale_mixed` 逐行抽取，10% 为 -1.0、90% 均匀于 [0, w_max=2.0)，故 **w=4 在蒸馏区间之外
2 倍处**，它最弱的 +0.0160 是外推而非缺陷。

### F. 裁剪：实测惰性的保险，同时是一个免费的崩溃探测器

v2 全程 234,712 条记录的 pre-clip 全局范数最大 **0.89946**，从未越过 1.0；候选前 3000 步峰值
**0.53411**，比 v2 同窗口的 0.63943 还低，裁剪 3000 步一次没咬。所以 `max_norm=1.0`
既不破坏与 v2 的可比性，也不是一个调出来的阈值。

但它不能挽救坏配方：三个崩溃 arm 全程带着同一个 max_norm=1.0，仍然崩溃，且到末段裁剪咬住
**44.4% / 59.2% / 82.8%** 的 update，Adam 饱和从约 0.24 涨到约 0.49（1.90-2.10x）。对照候选的 0.0%
与 0.86x，这两列构成**实时**判据。v3 之所以烧掉 2688 步，正是缺这个每步可读的信号。

### G. 仪表成本与尚未回答的两点

`log_cfg_embedding_diagnostics: true` 使每步成本 0.63072 → 0.65570 s（u2751-3000 实测中位，+3.4%），
58,678 步即 10.28 h → **10.69 h**。这个价钱买的是 §F 的实时判据。

两点 3000 步（预算的 5.1%）无法回答，登记为本 run 的观测目标：

1. **候选偏离 teacher 快于 v2**：它在 u3000 就到了 v2 约 u13,120 的 vs-teacher 水平。方向上是预期
   的——修复 2 让 target 依赖 w，均衡点本就不同——但落点未知。
2. **连续 w 的余弦虽为正但小**（+0.15 / +0.11 / +0.02）。

监控一律以**余弦**与 §F 的裁剪比例、Adam 饱和为准，**不得以 w 响应幅度为准**：崩溃 arm 中幅度最高
涨到 67 倍而余弦停在零附近。cfg 权重 RMS 的全程线性外推值（init 的 1884 倍）是一个 3000 步无法验证
的假设，只作上界看，不作预测用。

## 2026-08-22（第十二次修订）：v4 训练健康、teacher 保真达标，但连续 w 仍未学到

`hsi_c_lingo_cm_v4` 已完成并登记（`p1-hsi-c-lingo-cm-v4-s42-20260821`，status `completed`，
HEAD `8ff078a`，58,678 步，exit 0，10.69 h，42.75 GPU-h）。结果分裂：**训练侧与 teacher 保真度全部
通过，运行本身的目的——给 C 装上连续 w 能力——没有达成。**

### A. 与预检逐位连续，故预检证据字面适用

`epoch000.pth` 与 `hsi_c_cand_precheck` 的 `epoch000.pth` **sha256 完全相同**
（`d8742990…`），前 1024 条梯度记录 **1024/1024 逐位一致**。所以被批准的那 3000 步预检不是「类比
适用」，它就是本 run 的前 3000 步。顺带订正一处：epoch 是 **656** 步（89×656+294 = 58,678），
epoch000 落在 **update 656**，不是 652 也不是 768。

### B. 训练健康：全部指标优于或等于 v2

| 项 | v2 | **v4** |
|---|---:|---:|
| post-warmup 范数中位 | 0.0799 | **0.0836** |
| 全程范数最大 | 0.89946 | **0.59631** |
| 裁剪咬住 | 未启用 | **0 / 234,712**（系数最小恰为 1.0） |
| 相邻 loss 比最大 | 1.628 | **1.278** |
| epoch30 起 loss 中位 | 0.000871 | **0.00087** |
| epoch30 起 loss 最大 | 0.002641 | **0.001299** |
| 非有限值 | 0 | **0** |

裁剪反事实（post-warmup，4 rank）：0.2 → 22,888 次，0.4 → 1,508，0.5 → 188，**0.6 及以上 → 0**。
仪表完整性：8 条流各 **58,678** 行，尾部 flush 无丢失。

### C. 开放风险 1 已关闭且良性：漂移减速并饱和

t=419、w=1、八样本固定 batch、七个 student 在**同一次探针调用**内测量：

| student | update | vs teacher | vs GT | 每千步漂移 |
|---|---:|---:|---:|---:|
| v4 epoch000 | 656 | 0.04062 | 0.05774 | — |
| v4 epoch020 | 13,776 | 0.06824 | 0.07936 | +0.00212 |
| v4 epoch040 | 26,896 | 0.07221 | 0.08354 | +0.00031 |
| v4 epoch060 | 40,016 | 0.07333 | 0.08364 | +0.00009 |
| v4 epoch080 | 53,136 | 0.07365 | 0.08404 | +0.00002 |
| **v4 epoch089** | **58,678** | **0.07345** | **0.08433** | **−0.00003** |
| v2 epoch089 | 58,678 | 0.07459 | 0.08464 | — |

漂移速率从 epoch000–020 到 020–040 掉了约 **70 倍**，末段转为微负。落点 **0.98469×** v2（vs
teacher）与 **0.99630×**（vs GT）——比 v2 略好，不是更差。预检时那句「它在 u3000 就到了 v2 约
u13,120 的水平」是真的，但它描述的是一条**减速**曲线的早段。

### D. 开放风险 2 关闭为**否**：连续 w 的方向没有巩固

| checkpoint | sentinel cos | sentinel 幅度比 | cos w=0 / 2 / 4（idx 2） | 幅度比 w=0/2/4 |
|---|---:|---:|---:|---:|
| teacher B ep222 | +1.0000 | 0.50040 | — | — |
| v2 ep089 | **+0.7427** | 0.36531 | +0.176 / −0.006 / −0.121 | 0.004 / 0.005 / 0.001 |
| v4 ep040 | +0.7815 | 0.43854 | +0.094 / −0.081 / −0.094 | 0.192 / 0.203 / 0.064 |
| **v4 ep089** | +0.7126 | **0.44002** | **+0.090 / −0.029 / +0.003** | 0.180 / 0.244 / 0.067 |
| 三个崩溃 arm | +0.301…+0.352 | — | — | — |

对照 update 3,000 的 **+0.1464 / +0.1094 / +0.0160**：w=0 降到 +0.090，**w=2 翻负**到 −0.029，
w=4 停在 +0.003。ddim index 20 同结论（+0.159 / −0.151 / −0.028）。

三条读法必须一起说，否则会读错：

1. **幅度不是健康信号。** v4 的连续 w 响应幅度是 v2 的 49–65 倍（幅度比 0.166–0.244 对 v2 的
   0.003），但崩溃 arm 里幅度涨到 67 倍而余弦停在零附近，这正是「幅度涨、方向不动」的签名。
2. **w=4 在蒸馏区间之外 2 倍处**（训练 `w_max=2.0`），它的近零余弦是外推。**w=2 是区间内的**，它
   转负才是实质发现。
3. **v2 的连续 w 余弦本身是噪声**：它建立在 0.003 的响应上，相对 `ref_l2≈75` 是 3.7e-5。所以
   「v2 的 +0.176 比 v4 的 +0.090 好」是个无效比较。v4 唯一确实超过 v2 的是 **sentinel 幅度**
   0.44002 对 0.36531（理论值 0.50040）。

### E. 机制：目标函数只给连续 w 方向定价约 1%

`w` 进入目标函数**只有一条路**：`x_prev`，即从 teacher 的 CFG 外推预测出发的那一步 DDIM
（`models/infbagel.py:339-341`）。而 CFG 方向本身很小——探针实测 `|cond − uncond|` 占 `|cond|`：

| ddim index | t | 占比 |
|---|---:|---:|
| 2 | 59 | 1.032% |
| 8 | 179 | 1.122% |
| 14 | 299 | 1.132% |
| 20 | 419 | 1.127% |

一致性损失是对 `c_skip·x_prev + c_out·target_pred_x0` 的单项回归，其中携带 w 的成分只占约 1%。
**解冻 embedding 让响应变大了，但没有让目标函数去监督它的方向。** 这解释了为什么 lr 扫描
（200 倍区间）当初一致地无效，也解释了为什么再跑一次同样的目标只会得到第三个 null。

### F. 结论与下一步

`cfg.w` 仍是装饰性的，只有 `w=-1` sentinel 有效——**与 v2 同一状态**。第十一次修订对 v3 的归因
不受影响（崩溃是 `.eval()` 单独造成，cfg lr 已被 200 倍扫描排除）。

**不要在不改变 w 定价方式的前提下再花 GPU 时间做连续 w。** 两个候选杠杆，均未预注册、均需用户批准：

1. 提高 `w_max`，或对大 `|w|` 行过采样，让 CFG 方向在目标里占更大份额；
2. 在 `cond − uncond` 上加显式方向项，而不是只让 w 经由 `x_prev` 传导。

另外独立于此：v4 `epoch089` 在 teacher 保真度上是合格的 gate 工件，可以按 C+guided 协议对
`p1-hsi-c-v2-eval-epoch089-guided-progressfix-s42-20260821` 评测——但因为它与 v2 在单步保真上差
1.6% 以内且 w 状态相同，那次评测要以「gate 读数」立项，不能当作 w 假设的检验。Phase 1C 的 gate
阈值仍由用户定。

## 2026-08-22：修复后 B（完整 diffusion teacher）首次评测，以及 08-16 两行 B 数字的作废

### A. 为什么必须先评 B v2 —— 一个此前未被点明的证据缺口

用户提出的问题是：「目前并没有对完整 diffusion 进行评估，如果完整 diffusion 本身就没训练好，
那对蒸馏的修复是不是没有意义？」这个问题的前提需要订正一半，而订正之后结论反而更强。

**已评过 B，但评的是修复前的 B。** 在册的两行是
`p1-hsi-b-eval-epoch222-shard8-s42-20260816`（guided，4.99 h）与
`p1-hsi-b-eval-epoch222-unguided-shard8-s42-20260816`（unguided，2.68 h），二者都在
commit `101ac84` 上评 `hsi_b_lingo_full_epoch222`。表征修复是 `3ded4eb` + `a4c979c`
（2026-08-18），**晚于**这两次评测。按 `hsi-evaluator-fk-frame-defect` 的结论，修复前的
model-side 几何数字全部无效。因此准确的表述是：

> **修复后的完整 diffusion（`hsi_b_lingo_full_v2_epoch222`，08-19 训练完成）从未被评测过；
> 而 08-16 那两行 B 的几何数字属于表征修复前结果，不得用于当前任何决策。**

**「B 可能本来就没训好」这一担心已被间接排除。** C v2 是从 B v2 蒸馏而来
（`config_train_hsi_c_lingo_cm.yaml:134`，teacher sha256 `5daaf813…`），而 C v2 + guided 的
修复后 gate 读数是四项显著优于 GT（`pen_ratio` 0.77×、`interior_jerk` 0.89×、`skate_ratio`
0.79×、`transition_distance_aligned` 0.84×）、四项与 GT 无显著差异，walk 130 上
`success_last_10cm`/`success_min_10cm` 均为 1.0000 CI [0,0]。**一个没训好的 teacher 蒸不出
一个在穿模、接触与到达目标上全面达到或超过 GT 的 student。** 地基是好的。

### B. 但「保证蒸馏无误 → 只评 student → 反推 teacher」这条捷径在剩下的那个问题上不成立

这是本节要记的主要反对意见，因为它正是用户设想的省时路径。前提「蒸馏是保真的」已被实测否证：

- 蒸馏对 `boundary_jerk` 的主效应是 **−965（student 更好）**，对 `fs_nemf` 是 **+0.178（更差）**；
- guidance × 蒸馏交互显著：guidance 把 B 的 `boundary_jerk` 改善 359，把 C 的只改善 13（不显著）。

而当前 gate 读数里**唯二显著劣于 GT 的就是 `boundary_jerk` 1.60× 与 `jerk_ratio` 1.70×**，
恰好落在蒸馏有已测非零主效应、且方向是**让 student 好看**的那两项上。若该方向在修复后仍成立，
B v2 的 seam 可能比 C 更差，从 C 的 1.60× 读不出 B 是多少。

更关键的一层：**唯一能授权「C→B 反推」的那份证据（2×2 factorial）本身算于修复前，而修复把
seam 结论推翻了约 50×**（`boundary_jerk` 6831 → 134.9）。用来省时间的推理链，其授权凭据与它
想替代的测量，属于同一批作废数据。

### C. 成本前提也已过期

`hsi-eval-sharding-and-seeding` 落地后（40 h → 5 h），实测 8-shard guided B 是 **4.99 h**、
unguided 2.68 h。这不是需要靠推理绕开的开销。

### D. 协议对齐 —— 与 C v2 gate 逐项同口径，只翻 `sample_type`

对照 `p1-hsi-c-v2-eval-epoch089-guided-progressfix-s42-20260821`：

| flag | C v2 gate | 08-16 B（修复前） | B v2（本次） |
|---|---|---|---|
| ckpt | c_lingo_cm_v2 ep089 | b_lingo_full ep222（修复前） | **b_lingo_full_v2 ep222**，sha `5daaf813…` |
| sample_type | consistency | diffusion | **diffusion** |
| use_guidance / seed | true / 42 | true / 42 | **true / 42** |
| shard_count | 1（串行） | 8 | **8** |
| `hsi_progress_fix` | **true** | 不存在（=false） | **true** |
| `hsi_gt_trajectory` / `hsi_lookahead_m` | false / 0.8 | false / 0.8 | **false / 0.8** |
| `export_motion` | true | false | **true** |
| timing | valid（串行） | null | **null（自动）** |

两处必须说明的差异：

1. **`hsi_progress_fix=true` 是本次与 08-16 B 之间最要紧的一处**。该 gate 是让身体真正执行
   sit/lie 姿态转换的开关（最终 pelvis 高度 0.7810 → 0.5205，GT 0.4792），并把 C 的
   `boundary_jerk` 从 134.9 推到 159.0。不开就是拿「站着的 B」比「坐下的 C」。
2. **`export_motion=true` 已验证对数值中性**：`export_sink` 在全部采样完成之后写入
   （`test_infbagel_lingo_hsi.py:1663`），只存 handle 不做 D2H 拷贝，`keep_text` 仅保留一个
   pickle 列表，不消耗 RNG。它的用途是取 caption 做 walk / interactive 分层。

**timing 为 null 由代码保证，而非由 launcher 自律**：`shard_count>1` 走
`_invalidate_timing`（`:1974`），置空全部墙钟聚合并写入 `timing_valid=false` 与原因串；
guided cell 的 RDS 同样自动跳过。

### E. 分片逐位中性 —— 实测，且三个负对照确实失败

`seed_everything(cfg.seed + canonical_ordinal)`（`:1818`）无条件生效且以 **canonical
ordinal** 为键，故单个 episode 的 RNG 流与 `shard_count` 无关。在既有
`results/lingo_hsi/shard_bitwise` 工件上复核：

| 比较 | 相同值 | 不同值 |
|---|---:|---:|
| 500 步 串行 vs 2-shard | 141 | **0** |
| 500 步 串行 vs seed-43（负对照） | 72 | 75 |
| 100 步 串行 vs 2-shard | 235 | **0** |
| 100 步 串行 vs seed-43（负对照） | 119 | 126 |
| 100 步 串行 vs 改 reseed 前的 2-shard（负对照） | 138 | 97 |

三个负对照都按应有的方式失败，所以这不是一次空洞 PASS。**因此 8-shard 的 B v2 与串行的
C v2 gate 可以直接配对比较。**

### F. 启动前的 smoke，以及一个不能当结论读的早期信号

以 `shard_count=375` 取最短 episode（`99-pick_up:009601`，2 窗口）跑通全流程：exit 0，
`schema_version 4`、`sample_type diffusion`、`guided true`、checkpoint sha `5daaf813…`、
`timing_valid false` 带 contention 原因、RDS `available false`。日志第二个窗口打印
`pi: tensor([42])`，证实 `hsi_progress_fix` 生效（未修时该值被置零）。

同一 episode 上的三方对照（**n=1，不是证据，仅记录为待验证方向**）：

| | B v2 | C v2 |
|---|---:|---:|
| `boundary_jerk` | 100.51 | 169.09 |
| `jerk_ratio` | 1.394 | 2.508 |
| `fs_nemf` | 0.672 | 0.393 |
| `skate_ratio` | 0.169 | 0.270 |

**该方向在 375 个 episode 上被推翻，见 §H。** 这一段原样保留，作为「n=1 不是证据」的一个具体
例证：全量上 B v2 的 `boundary_jerk` 是 222.50（2.64× GT），**高于** C 的 159.02（1.88×），
方向与本段完全相反。反向的 `fs_nemf` 是唯一在全量上符号一致的一项。

### G. 本次 run 与作废声明

`p1-hsi-b-v2-eval-epoch222-guided-shard8-s42-20260822`，起于 `ed763b2`（worktree 干净），
GPU 0–7（启动前实测八张各 18 MiB、零 compute process、无他人占用），窗口均衡
[285 283 283 285 285 284 284 282]，输出 `results/lingo_hsi/b_v2_guided_shard8`。
**不覆盖也不复用 08-16 的目录**：`b_guided_shard8` / `b_unguided_shard8` 原样保留。

**作废声明（本节即为登记处，registry 为 append-only，不改旧行）**：
`p1-hsi-b-eval-epoch222-shard8-s42-20260816` 与
`p1-hsi-b-eval-epoch222-unguided-shard8-s42-20260816` 的一切 model-side 几何数字属于
**表征修复前**结果，**不得用于当前决策，也不得与本次或任何修复后行放在同一张表里比较**。
两行的 provenance、成本与协议结论（分片规则、timing 口径、RDS 门控）仍然有效并继续被引用。
本节 §D 的成本数字即取自该 guided 行，这一用法是允许的：它是墙钟，不是几何。

按用户指示，本次只跑 guided；**不自动启动 unguided**、不改 B/C、不加 continuous-w loss、
不重训。continuous w 路线继续暂缓，等本次归因结果再决定。

### H. 结果：seam 缺口不是从 diffusion 继承的，而蒸馏在改善它

`p1-hsi-b-v2-eval-epoch222-guided-shard8-s42-20260822` 完成：8/8 shard `fail=0`，
merge exit 0，墙钟 **5 h 18 m**（13:25:40–18:43:54 本地），对照 4.99 h 的先例。
20 条 merge 锚全部通过，其中两条值得单独点出：`excluded_as_warmup` 为 **5**（不是 40，
说明 canonical-ordinal warmup 规则生效），以及 episode key 集合与 C gate **完全相同**，
所以配对是同一批 375 个 episode。

配对逐序列 bootstrap，10,000 次重采样（重采样 **episode**，故同一次 replicate 内所有指标看到
同一批 episode），seed 42。**归因判据在任何 B v2 聚合值产生之前就已写定**
（`.claude/scratch/bv2_eval_20260822/attribution_rule.md`，当时进度约 70/284 窗口/shard）。

| | GT | **B v2** | B/GT | C v2 | C/GT | B→C delta | 95% CI | |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `boundary_jerk` | 84.44 | **222.50** | **2.64×** | 159.02 | 1.88× | **−63.55** | [−82.50, −46.37] | SIG |
| `jerk_ratio` | 1.194 | 2.295 | 1.92× | 2.177 | 1.82× | −0.118 | [−0.211, −0.026] | SIG |
| `interior_jerk` | 70.43 | 92.38 | **1.31× SIG worse** | 68.59 | **0.97× ns** | −23.83 | [−29.60, −18.63] | SIG |
| `fs_nemf` | 0.2597 | 0.3159 | 1.22× | 0.2915 | 1.12× | −0.0244 | [−0.0382, −0.0111] | SIG |
| `transition_distance_aligned` | 0.0064 | 0.0080 | 1.25× SIG | 0.0067 | 1.04× ns | −0.0013 | [−0.0019, −0.0009] | SIG |
| `goal_planar_err_m` | 0 | 0.0851 m | — | 0.0610 m | — | −0.0240 | [−0.0307, −0.0169] | SIG |

按预注册判据，落在第五行：**delta CI 不含 0 且 delta < 0 → 归因于 diffusion，且蒸馏在改善它。**
08-17 factorial 记录的 `boundary_jerk` −965 方向**在表征修复后依然成立**。

**比 seam 更锐利的一项是 `interior_jerk`**：B v2 高出 GT 22.01，CI [15.84, 28.60] SIG（1.31×），
而 C v2 是 0.97× 且与 GT **无显著差异**。所以蒸馏不只是压低了接缝处的抖动，它把窗口**内部**的
平滑度从「显著差于 GT」拉到了 GT 平价。旋转不变的对照 `transition_distance_aligned` 同向：
B 在 interactive 上 1.44× SIG 差于 GT，C 是 1.17×，**说明 B 的缺口部分是位置性的，不只是三阶抖动** ——
这一点与 C 的情形不同（C 那边残差确实以 jitter 为主）。

`fs_nemf` 本次**非空洞**：B、C、GT 三者各有 **0/375** 个恰好为 0 的 episode（修复前是 208/375），
所以 1.22× / 1.12× 是可读的数字，而不是 FK 身体躺平导致脚从未进入 ankle/toe 带的产物。

**场景交互与到达目标：蒸馏基本中性。** `pen_ratio`、`pen_depth_mean`、`pene_pct_scene`、
`pene_sum_mean_floorexcl`、`contact_count`、`skate_ratio`、`success_last_10cm`、
`success_min_10cm` 的 B→C delta 全部 **ns**，且 `success_min_10cm` 两者都恰好 1.0000。
反过来 B 在 `goal_planar_err_m`（0.0851 vs 0.0610 m）与 `last_dist`（0.0337 vs 0.0284 m）上
**显著差于 C**。

**分层（walk 130 / interactive 245，判别式 `caption == "walk"`）：方向一致，量级不一致。**

| 分层 | B/GT | C/GT | 蒸馏消除了 B 超出 GT 部分的 |
|---|---:|---:|---:|
| walk 130 | 1.70× | 1.29× | **59%** |
| interactive 245 | **3.26×** | 2.28× | 43% |

interactive 的 `jerk_ratio` B→C 是 **ns**。所以 **interactive 在两个模型上都仍是缺口所在，
而 B 在那里是更差的一方**（3.26× vs 2.28×）。

### I. 这回答了用户的问题，但把前提反转了

用户设想的捷径是「先确保蒸馏无误，再评 student 反推 teacher，以省下评 B 的时间」。本次测量
证否了它的前提：**蒸馏对 seam 不透明** —— 它消除 B 的 `boundary_jerk` 超出量的 46%，并把
`interior_jerk` 从显著差于 GT 抬到平价。因此任何只读 C 的做法都会**系统性地美化 teacher**。

推论是：**剩下的 seam 工作在 diffusion 侧，不在 CM 侧。** B v2 的 2.64×（interactive 3.26×）
才是更大的那个缺陷，而 C 在每一个 jerk 指标上都优于 B。这不构成「可以不做蒸馏」的理由 ——
恰恰相反，蒸馏目前在替 diffusion 兜住平滑度。

按用户指示：本次只跑 guided；**未**自动启动 unguided，**未**改动 B/C，**未**加 continuous-w
loss，**未**重训。continuous w 路线继续暂缓，等用户就本归因结果决定下一步。

## 2026-08-23：seam 离线归因（无 GPU 正式运行），以及一处对本文档 §H 的订正

用户接受 §H/§I 的归因并设为常规约束：**seam 主要缺口在 B 的 guided diffusion pipeline；
C 的蒸馏实际改善平滑度，故不得修改或重训 C，也不得再用 C-only 结果推断 B；continuous-w 继续暂缓。**
本节是随后要求的离线分析，全部基于已有的 375×(B v2, C v2) motion export 与三方 sealed metrics，
**未启动任何训练或正式 GPU 运行**。完整读数：`.claude/scratch/seam_attrib_20260822/FULL_REPORT.txt`。

### A. 地基：export 能逐位重建 sealed jerk

`boundary_jerk` / `interior_jerk` 从 export 的 FK 输入重建，与 sealed 值最大相对误差 **1.7e-6**
（6 个 episode，float32 FK 噪声量级）。因此下面每一个量都算在 sealed 指标所用的同一批关节上。
GT **没有** motion export（0/375），所以 GT 只能通过其 sealed per-sequence 指标参与，不能参与通道分解 ——
这是本节的一个硬边界。

### B. Q1 通道分解：root 平移是主导，且 B 在每个关节上都均匀更差

`jerk = mean_over_joints ||third_difference(p)||`，本身即逐关节均值，故 root/非 root 与逐关节
分解是**精确**的重新划分，不是模型。平移/旋转用反事实分离（冻结另一路），且实测两者与
`root` / `local frame` 恒等（max|diff| ≤ 4e-3），说明三阶差分下平移与旋转在此近似可加。

| 通道 | B v2 boundary | C v2 boundary | B/C | interior B/C |
|---|---:|---:|---:|---:|
| 总计 | 222.50 | 159.02 | 1.40× | 1.35× |
| **root / 平移** | 142.18 | 89.35 | **1.59×** | **1.49×** |
| 旋转 / 姿态（去 root） | 125.65 | 101.70 | 1.24× | 1.24× |

**boundary 超出量的 69% 来自 root 平移**（52.83 / 76.77），interior 为 58%。

逐关节：最差 6 个关节占比 B **28.7%** / C **28.7%**，上肢占比 33.1% / 33.6%，下肢 32.0% / 31.0%。
**分布几乎完全相同 —— B 不是在某些关节上更差，而是整体被抬高。** 最差关节是 foot/ankle
（R_foot 314 vs 230），但这是 FK 链末端的放大效应，不是独立缺陷。

### C. Q2 定位：**这不是接缝缺陷**（本节订正 §H 的框架）

| | boundary/GT | interior/GT | 比值之比 | 95% CI |
|---|---:|---:|---:|---|
| B v2 | 2.635 | 1.312 | 2.009 | [1.874, 2.148] |
| C v2 | 1.883 | 0.974 | 1.934 | [1.797, 2.084] |
| **B vs C** | 1.399× | 1.347× | **1.040** | **[0.970, 1.112]** |

两件必须分开的事：

1. **相对 GT，两个模型都有真实的接缝效应**（比值之比 ≈2.0，CI 排除 1）。
2. **但 B 相对 C 的超出量对接缝位置是均匀的**（1.040，CI **含 1**）。按 offset 看，B/C 在
   offset ±6 处同样是 1.24–1.31×，与 boundary 的 1.35–1.55× 没有量级差别。

**所以 §H 把 B 的缺口称作"seam 缺口"是错的框架。** B 相对 C 多出来的抖动是**全程均匀抬高**，
不是缝处特有。`jerk_ratio`（自归一，无法被全局平滑刷分）B 2.295 vs C 2.177 —— 只差 5%，
而绝对水平差 40%，正是这个结论的另一面。

分层：`sit` (n=105) boundary B/C **1.73×**、root B/C **1.93×**；`lie` (n=12) 绝对值最高
（846 vs 730）；`walk` 1.32×。按 contact_count 三分：low 1.18× / mid 1.31× / high **1.59×**，
接触越多差距越大。按窗口序号无单调增长趋势（seam 0..3 为 1.43/1.57/1.18/1.31×）。

### D. Q3 配对异常：重尾，且机制是**物理上不可能的 root 加速度尖峰**

B 更差的 episode 占 **287/375 = 76.5%**（中位比 1.214×），所以效应是普遍的；但**总超出量高度集中**：
前 5 个 episode（1.3%）占 20.2%，前 20 个（5.3%）占 **56.8%**，前 94 个占 97.9%。

最差 12 个几乎全是 **sit / lie 下坐下躺**，root B/C 高达 **15.2×**。最差四分位 vs 其余：
`frac sit/lie` **2.56×**、`B root_speed_max` **2.31×**、`pen_ratio` 2.04×，而
**`GT boundary_jerk` 只有 1.03×** —— 这些**不是** GT 眼中困难的 episode
（spearman(excess, GT) = **0.055**）。是 B 自己在这些动作上坏掉。

**机制，逐帧实测的 root 加速度：**

| | B v2 | C v2 | B/C |
|---|---:|---:|---:|
| per-episode max &#124;acc&#124; 均值 (m/s²) | **19.68** | 11.71 | 1.68× |
| 超过 2g 的帧数/episode | 1.355 | 0.384 | 3.53× |
| 有任一帧超过 5g 的 episode | **31/375** | 10/375 | 3.1× |

个例：`010:000344` (sit down on office chair) B 的 root 加速度峰值 **84.9 m/s² ≈ 8.6g**；
`010:000426`（caption 为 `walk`）峰值 **180.6 m/s² ≈ 18g**，且 pelvis 高度下降 **0.662 m**
（C 为 0.082 m）——**B 在一个纯行走片段里把骨盆甩穿了地板**。walk 130 中 pelvis 低于 0.6 m 的
episode：B **13** 个，C **1** 个。`spearman(excess, B acc_max) = +0.464`，是所有候选里最强的。

**订正本 session 早先的一句判断**：我先看 4 个手挑 episode 后说尖峰"不在接缝上"，这是错的。
全量上 B 的 max-root-jerk stencil 有 **58.1%** 落在任一接缝 ±2 帧内（C 48.3%），而**随机基线只有 9.0%**，
CI 排除基线。**接缝确实是触发点**，两个 arm 都是；B 与 C 的差别主要在尖峰**幅度**
（acc_max 1.68×、超 2g 帧数 3.53×），而不在**位置**（+9.9 pp，CI [+3.5,+16.3]）。

### E. Q4：C 的改善**不是**单纯时间平滑，代价几乎为零

低通滤波的签名是「jerk 降 → 速度同比降 → 幅度缩 → HF 功率大幅降」。实测不符：

| 量 | B v2 | C v2 | B/C |
|---|---:|---:|---:|
| boundary jerk | 222.50 | 159.02 | **1.399×** |
| interior jerk | 92.38 | 68.59 | **1.347×** |
| 平均关节速度 (m/s) | 0.2499 | 0.2258 | 1.107× |
| 平均 root 速度 | 0.2057 | 0.1833 | 1.122× |
| bounding-box 体积 (m³) | 2.036 | 1.963 | **1.037×** |
| HF 功率占比（> Nyquist/2） | 0.0045 | 0.0038 | 1.179× |

**jerk 降 35–40%，而速度只降 11%、幅度只降 3.7%、HF 功率只降 18%。** 低通会让这四个数字同量级下降。
C 是在**不损失运动幅度**的前提下去掉了三阶尖峰 —— 与 D 节一致：它压掉的是那些不可能的加速度脉冲。

**代价**：sealed 指标里 B 优于 C 的只有 **`goal_height_err_m`（0.982×，SIG）** 一项，量级 1.8%。
`skate_ratio` 0.978× 与 `pene_pct_scene` 0.982× 名义上 B 略好但 **ns**。其余全部 C 更好或无差异。
**没有发现 C 为平滑付出的实质代价。**

### F. Q5：guided-only 证据**不能**区分"B 模型/采样器问题"与"guidance 交互问题"

**根本原因是一处此前未记录的曝光不对称**（取自 sealed timing，非代码推断）：

| cell | sampler_steps_per_window | denoiser calls |
|---|---:|---:|
| B v2 guided | **500** | 1000 |
| C v2 guided | **16** | 16 |

`code/models/infbagel.py:1018`(B) 与 `:758`(C) 用的是**同一条规则**
`gradient = autograd.grad(-loss, x_start) * guidance_scale; x_prev = x_prev + gradient`，
**没有 1/steps 归一化，也没有逐步衰减**（C 路径上的 `(1-alpha_cumprod)` 因子是注释掉的）。
于是 guidance 项**每个采样步加一次**：**B 加 500 次，C 加 16 次，相差 31.2×。**

**因此 B 与 C 在本次测量中同时差两件事，且在 375 个 episode 上完全共线：**
(i) 模型/采样器（500 步 ancestral vs 16 步 consistency）；(ii) guidance 剂量（31.2×）。
guided-only 数据里没有任何一格能让其中一个变、另一个不变，故**无法分离**。

三条约束使"guidance 剂量"成为领先假设：guidance 写的正是
`x_start[:, :, :84]`（28×3 位置通道，slot 0 即 root），而缺口 69% 在 root 平移上；缺口对接缝位置均匀，
像逐步过程而非缝处一次性事故；且 31× 剂量**没有**换来穿模优势（`pen_ratio` 0.91× GT，
`pene_pct_scene` 1.04×）。

**但一个免费的反证削弱了它。** 在**修复前**的树内，B guided vs B unguided（同 checkpoint
`931a6f1fff41`、同 seed、同表征，唯一差别是 `use_guidance`）：

| | unguided | guided | g/u | |
|---|---:|---:|---:|---|
| `boundary_jerk` | 7463.39 | 7104.46 | **0.952×** | SIG |
| `interior_jerk` | 579.04 | 608.12 | 1.050× | SIG |
| `jerk_ratio` | 12.838 | 11.693 | 0.911× | SIG |

**在那个表征下，guidance 让 boundary jerk 变好，不是变坏。** 幅度不可迁移（修复前 FK 身体躺平、
208/375 的 `fs_nemf` 恰为 0），但**符号**是 guided-only 数据给不出的信息，它指向采样器而非 guidance。
两个假设仍然共存，必须用实验分开。

### G. 若需要 post-repair unguided 对照：最小 matched 矩阵（**未启动，等批准**）

| cell | 状态 | GPU | 墙钟 |
|---|---|---:|---:|
| B v2 + guided | 已完成 2026-08-22 | 8 | 5.30 h |
| C v2 + guided | 已完成 2026-08-21 | 1 | 1.35 h |
| **B v2 + unguided** | 需要 | 8 | ~2.9 h |
| **C v2 + unguided** | 需要 | 8 | ~0.4 h |

新增成本 **约 3.3 h 的 8 卡墙钟**，两个 run，一个晚上。估算依据：B unguided 修复前实测 2.68 h ×
(5.30/4.99 的修复后漂移) = 2.85 h；C unguided 由 1.35 h 串行 guided 去掉 guidance autograd 再 8 分片。
两者都 8 分片，故 timing 按设计为 null，均非 latency run。

这个 2×2 精确地买到：**模型主效应**（固定 guidance → 500 步采样器本身是否是 jerk 来源）、
**guidance 主效应**（固定模型 → 31× 剂量是否注入 root 尖峰）、**交互**（guidance 是否专门伤害 B）。
另外 unguided cell 才能算 RDS（guidance 开启时被 gate 掉），这是检验 B 是否真的在用场景的唯一途径。

### H. 建议的**单一**最小 B-side 修复，以及验证门槛（等批准，未实施）

**修复：在 B 的 guided 采样路径上给 guidance 增量加一个范数上限（per-step trust region）。**
`code/models/infbagel.py:1018` 之后，把 `gradient` 按其 L2 范数裁剪到一个阈值 `tau` 再加到
`x_prev`，即 `gradient *= min(1, tau / (||gradient|| + eps))`。这是**一行量级**的改动，
config 开关，默认关闭，不触碰 C、不触碰 `code/priors/core/`、不需要重训任何模型。

**为什么是这一个而不是别的：** 诊断指向的是「幅度」而非「位置」或「关节」—— root 平移占超出量 69%，
31/375 个 episode 出现 >5g 的物理不可能加速度，`spearman(excess, acc_max)=+0.464` 是最强预测因子，
而逐关节分布与 C 完全相同（28.7% vs 28.7%）。裁剪增量正是针对「个别步注入过大位移」这一机制，
且它**同时**解释了 76.5% 普遍偏差（每步都略大）与 5.3% 重尾（少数步极大）。
它也无需先解开 §F 的共线性：无论过大的增量来自 guidance 还是来自 500 步链，上限都作用在两者之和上。

**验证门槛（预注册，通过才考虑保留）：**
1. **必要**：`frames_over_5g` 的 episode 数从 **31/375** 降到 **≤10/375**（即 C 的水平），
   且 walk 130 里 pelvis 低于 0.6 m 的 episode 从 **13** 降到 **≤3**。
2. **必要**：`boundary_jerk` 相对 GT 从 **2.635×** 降到 **≤2.0×**，配对 bootstrap CI 排除 0。
3. **不得倒退**：`pen_ratio`、`pene_pct_scene`、`contact_count`、`success_min_10cm`、
   `goal_planar_err_m` 五项相对当前 B v2 guided 均不得显著变差；`min_dist` 不得显著增大
   （否则是用"远离一切表面"换平滑）。
4. **反作弊**：`mean_speed` 与 bounding-box 体积相对当前 B 的下降不得超过 **5%**，
   `jerk_ratio` 必须同时下降 —— 否则该修复只是全局低通，与 §E 判定 C **不是**低通的标准同一把尺。
5. **成本**：单次验证 = 一个 B v2 guided 8-shard run ≈ 5.3 h / 8 GPU。建议先在
   §D 认定的最差 20 个 episode（占超出量 56.8%）上做 ~0.3 h 的 smoke，通过再上全量。

`tau` 的取值本身未预注册最优值；建议用同一批 20 个 episode 扫 3 个值（如 C 的 p99 增量范数的
1×/2×/4×），选满足门槛 1–4 的最小干预档，并把扫描结果与选择理由一并登记。

**未做**：本节没有启动任何训练或正式 GPU 运行，没有修改 B 或 C 的任何源文件，没有加 continuous-w loss。

## 2026-08-23（同日第二次）：matched unguided 两格完成，2×2 判定 —— 抖动来自 guidance，不是采样器

用户批准并运行了 post-repair matched unguided 两格，形成完整 2×2。
`p1-hsi-b-v2-eval-epoch222-unguided-shard8-s42-20260823`（8 分片，2 h 39 m）与
`p1-hsi-c-v2-eval-epoch089-unguided-s42-20260823`（串行，26 m 42 s），新增墙钟合计
**3.10 h**，与批准的 ~3.3 h 一致。完整读数：`.claude/scratch/unguided_2x2_20260823/FULL_REPORT.txt`。

**协议对齐（脚本内 assert 强制）**：B 两格 checkpoint 均为 `5daaf813`，C 两格均为 `f1c09c2425af`，
四格 seed 42、375 episode / 2271 窗口、`hsi_progress_fix=true`、`export_motion=true`。
每格只与自己的 guided 搭档差 `use_guidance` 一项：B 沿用同一 8 分片切分，C 沿用串行。
跨列可比因为分片逐位中性（500 步下 141/141 相同，三个负对照均失败）。
B unguided 的 20 条 merge 锚全通过（含 `excluded_as_warmup` = 5、key 集合与 guided 完全相同）。
**RDS 自动开启**（`:1714` 的 `rds_available = not guided`，无 flag），这是问题 3 唯一的取数途径。

### A. 问题 1：guided → unguided 的变化

| | B+guided | **B+unguided** | C+guided | **C+unguided** | GT |
|---|---:|---:|---:|---:|---:|
| `boundary_jerk` | 222.50 (2.64×) | **127.92 (1.51×)** | 159.02 (1.88×) | **128.85 (1.53×)** | 84.44 |
| `interior_jerk` | 92.38 (1.31×) | **63.72 (0.90×)** | 68.59 (0.97×) | **62.88 (0.89×)** | 70.43 |
| `jerk_ratio` | 2.295 | 2.030 | 2.177 | 2.050 | 1.194 |
| root max &#124;acc&#124; 均值 (m/s²) | 19.68 | **8.84** | 11.71 | **8.58** | — |
| 超 2g 帧数 / episode | 1.355 | **0.019** | 0.384 | **0.029** | — |
| **有超 5g 帧的 episode** | **31/375** | **0/375** | **10/375** | **0/375** | — |
| 超 5g 总帧数 | 132 | **0** | 41 | **0** | — |
| walk 中 pelvis <0.6 m | 13/130 | **0/130** | 1/130 | 1/130 | — |
| root jerk (boundary) | 142.18 | **59.75** | 89.35 | 66.91 | — |

配对 CI：B 的 `boundary_jerk` −94.57 [−116.62, −74.93] SIG，`interior_jerk` −28.67 SIG，
`acc_max` −10.84 [−13.48, −8.47] SIG；C 对应为 −30.17 [−45.39, −17.80] SIG、−5.71 SIG、
−3.13 [−5.07, −1.58] SIG。

**关闭 guidance 后，两个模型的「物理不可能加速度」帧数都精确归零。** 8-23 节认定的机制
（>5g 的骨盆加速度、walk 片段骨盆穿地）**100% 由 guidance 产生**，无一例来自采样器本身。
B 的 `interior_jerk` 甚至降到 **0.90× GT，低于真值**。

**代价是真实且符合预期的**：关掉 guidance 后穿模变差 —— B 的 `pen_ratio` 0.0255→0.0354
（0.91×GT → 1.26×）、`pene_pct_scene` 0.0523→0.0608、`pene_sum` 11.69→13.44，C 同向
（`pene_sum` 9.98→13.91，代价比 B 更大）。`min_dist` 从 0.0048 降到 0.0030，身体贴表面更近。
但**到达目标反而变好**：B 的 `goal_planar_err_m` 0.0851→0.0516 m、`last_dist` 0.0337→0.0235 m，
均 SIG，`success_min_10cm` 保持 1.0000。

### B. 问题 2：主效应与交互（`boundary_jerk`）

| 项 | 量 | 占效应量之和 | |
|---|---:|---:|---|
| 模型主效应 (B−C) | 31.277 | 19.8% | SIG |
| **guidance 主效应 (g−u)** | **62.373** | **39.5%** | SIG |
| **交互 (Bg−Bu)−(Cg−Cu)** | **64.401** | **40.7%** | SIG |

**涉及 guidance 的项合计 80.2%。交互是最大单项 —— guidance 专门伤害 B。**
每个指标都通过了交互恒等式自检。加速度通道同构：`acc_max` 的模型/guidance/交互为
4.115 / 6.985 / 7.713（全 SIG），`frames_over_5g` 为 0.121 / 0.231 / 0.243（全 SIG）。

`jerk_ratio` 的**模型主效应为 ns**（0.049），guidance 主效应 0.196 SIG、交互 0.137 SIG ——
即在自归一化口径下，模型身份根本不重要，guidance 才重要。

一处对 8-23 §C「超出量对接缝位置均匀」的**细化**：B 内部 guidance 效应的 boundary/interior
超出比为 **1.200 CI [1.124, 1.278]**（boundary 1.739× vs interior 1.450×），**排除 1**。
所以 guidance 是一个逐步过程，但**在接缝处咬得更狠**，不是纯均匀抬升。原结论（B-vs-C 超出量
对 offset 均匀）在 guided 口径下成立，此处补上的是它的成因分解。

### C. 问题 3：unguided RDS —— B 确实在用场景

| | RDS 均值 | 95% CI | 中位 | p10 | p90 | RDS<0.005 的 episode |
|---|---:|---|---:|---:|---:|---:|
| B unguided | **0.14650** | [0.13930, 0.15385] | 0.13328 | 0.06789 | 0.24519 | **0/375** |
| C unguided | 0.13115 | [0.12421, 0.13826] | 0.11905 | 0.05270 | 0.23275 | **0/375** |

RDS = 同噪声下「给场景条件」与「置零场景条件」两次 rollout 的分歧度。
**B 的 375 个 episode 里没有一个接近 0，且 B 显著高于 C（+0.01535 [+0.00954, +0.02115]）。**
所以 B 不是在忽略场景条件 —— 它在用，而且比 C 用得更多。**「B 没学会看场景」这个假设被排除。**

### D. 问题 4：判定 —— guidance 剂量，且有一个决定性对照

**关掉 guidance 后，B 与 C 在 `boundary_jerk` 上统计不可区分**：
127.924 vs 128.847，delta −0.923 CI [−3.609, +1.820] **ns**；`jerk_ratio` 也是 **ns**
（−0.0197 [−0.0614, +0.0225]）；只有 `interior_jerk` 差 +0.838 SIG，量级 1.3%。

**即 B-vs-C 的整个抖动差距都是 guidance 中介的。** 8-23 节留下的两个假设由此裁定：

- **guidance 剂量：成立。** 500 次 vs 16 次未归一化增量累加是主因，交互项证明它对 B 尤其有害。
- **B 模型/采样器：基本排除。** 去掉 guidance 后 500 步 diffusion 与 16 步 CM 的接缝抖动
  没有可测差别。

**同时这条对照给出了任何 per-step 修复的天花板**：B unguided 的 1.51× GT 与 C unguided 的
1.53× 不可区分，**说明剩余的 ~1.5× 是两个模型共有的、属于 16 帧自回归 rollout 本身的性质**，
不属于任一采样器。一个完美的增量上限最多把 B 拉到 ~1.5×，且不可能同时保住 guidance 买来的穿模。

**这也修正了 8-23 §F 的领先假设排序**：当时那条「修复前 B guided vs unguided 让
`boundary_jerk` 变好 0.952×」的反证，在修复后的表征下**反号**（变差 1.739×）。修复前那个数
不可迁移，本节以 post-repair matched 设计取代它。

### E. 未做

未修改任何模型源码，未实施 guidance norm cap，未做 norm-cap 实验，未修改或重训 C，
continuous-w 继续暂缓。最差 20 episode 的 norm-cap smoke 等用户在本结果之后决定。

## 2026-08-23（同日第三次）：guidance norm-cap smoke —— 三档 tau 全部失败，且失败原因在跑之前就可测

用户批准实现并运行 §H 预注册的 per-step guidance norm cap，仅限已冻结的最差 20 个
episode。代码 commit `1a3a5a1`，配置开关 `hsi_guidance_norm_cap` 默认 `null`，只作用于
`Sampler.p_sample`（diffusion 路径）。运行目录 `results/lingo_hsi/normcap_smoke_20260823/`，
完整读数 `.claude/scratch/normcap_20260823/FULL_REPORT.txt`，预注册
`.claude/scratch/normcap_20260823/preregistration.md`（在任何 arm 存在之前写定）。

**结论：两个非平凡档全部未通过全部五道门槛，且不是"差一点"，是量级不对。**

### A. 冻结的 20 个 episode，与它们有多极端

按 `B v2 guided boundary_jerk − C v2 guided boundary_jerk` 取前 20，即 §D 认定的集合，
核对其占 375 个 episode 总超出量 **56.8%**，与 §D 一致。102 窗口 = 2271 的 4.5%，
4 个 walk / 16 个 interactive，canonical ordinal 10..285 全部 ≥ `timing_warmup_sequences`=5。
选取方式 `shard_count=375 shard_index=<该 ordinal 的 bin-packing 槽位>`，映射由
`plan_episode_shards` 自己产生，并以精确复现已封存的 8 分片切分
（`[285,283,283,285,285,284,284,282]`、`[49,46,46,47,47,47,47,46]`）验证。

这 20 个远比全集恶劣：B+guided 的 `boundary_jerk` 在这里是 **904.71 = 10.24× GT**
（全集 2.64×），**16/20** 个 episode 出现 >5g 的根加速度，全集 132 个 >5g 帧里有 **85 个**
落在这 20 个 episode 内。

### B. Inertness：两个 cap-off arm 逐位复现已封存结果

| 对照 | 数值指标比较 | 精确不一致 | motion 数组比较 | 逐位不一致 |
|---|---:|---:|---:|---:|
| `probe_b`（cap 未设 + hook）vs 封存 B v2 guided 8 分片 | 1060 | **0** | 440 | **0** |
| `probe_c`（cap 未设 + hook）vs 封存 C v2 guided 串行 | 1060 | **0** | 440 | **0** |

一次同时证明三件事：默认关闭时新代码发出的算术与发布版一致；`torch.autograd.grad`
包装器是只读的；375 分片的单 episode 运行能从另一种分片方式内部逐位复现该 episode ——
这才是 smoke 能与封存基线比较的前提。

### C. 决定性测量：B 的每步增量比 C **小**

只读探针记录 `‖autograd.grad(-loss, x_start)[0]‖₂`（`guidance_weight` 恒为 1，故与实际
加到 `x_prev` 上的增量同量）。B 50,898 个增量，C 1,530 个。

| | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B (`p_sample`) | 15.97 | 32.07 | 61.62 | 105.71 | 138.79 | 235.45 | **782.38** | 46.54 |
| C (`cm_sample`) | 25.95 | 46.62 | 90.40 | 134.35 | 169.05 | 281.99 | 390.02 | 64.94 |
| **B/C** | 0.615 | 0.688 | 0.682 | 0.787 | 0.821 | 0.835 | 2.006 | **0.717** |

**除极端尾部外，B 的每步增量在每个分位数上都小于 C。**差距完全来自步数：guidance 在除
最后一步外的每一步触发，实测 `t_index` 为 B **499 次/窗口**、C **15 次/窗口**（33.3×，
不是比较 `sampler_steps_per_window` 500 vs 16 得到的 31.2×）。每窗口总位移
499×46.54 = 23,225 对 15×64.94 = 974，即 **23.8×**。

因此预注册的 tau 阶梯（C 池化 p99 的 1×/2×/4× = 281.99 / 563.97 / 1127.94）在 B 的
50,898 步中分别只裁掉 **135 / 2 / 0** 步。**tau4 超过 B 的最大增量 782.38，
故第一步逐位相同，归纳可知整条轨迹逐位相同 —— 这是可证明的空档，不需要用 GPU 去确认算术，
因此未运行。**即使换成"C 的 p99"最激进的另一种读法（每 episode p99 的均值 119.42），
也只裁掉 7.4%。结论对读法不敏感。

### D. 两个非平凡档的结果（n=20 均值）

实际裁剪：tau1 176/50,898（0.346%），tau2 3/50,898（0.006%）。

| 指标 | B+guided | **tau1** | **tau2** | B+unguided | C+guided | GT |
|---|---:|---:|---:|---:|---:|---:|
| `boundary_jerk` | 904.71 | **887.39** | **865.24** | 161.19 | 228.59 | 88.37 |
| `interior_jerk` | 269.78 | 248.13 | 243.44 | 70.57 | 91.33 | 75.22 |
| `jerk_ratio` | 3.789 | **3.811** | **3.827** | 2.311 | 2.358 | 1.185 |
| 有 >5g 帧的 episode | 16/20 | **17/20** | 15/20 | 0/20 | 2/20 | — |
| >5g 帧总数 | 85 | 83 | 72 | 0 | 3 | — |
| 平均 max\|acc\| (m/s²) | 91.69 | 89.69 | 84.44 | 10.36 | 18.41 | — |
| `pen_ratio` | 0.07225 | **0.07582** | **0.07416** | 0.08814 | 0.06647 | 0.05503 |
| `contact_count` | 1704.21 | 1823.24 | 1801.26 | 1965.94 | 1729.93 | 1325.21 |
| `mean_speed` | 0.4209 | 0.4061 | 0.3987 | 0.2700 | 0.2712 | — |
| bbox 体积 (m³) | 2.440 | 2.473 | 2.418 | 2.071 | 1.990 | — |

配对 bootstrap（10,000 次，seed 42，重采样 episode）对 B+guided：`boundary_jerk`
tau1 **−17.32 [−156.32, +120.93] ns**，tau2 **−39.47 [−128.56, +10.16] ns**。
门槛 2 要求 ≤334.1，两档都差约 2.6 倍。恒等自检（B+guided 对自身）在所有指标上
delta 为 0 且 ns，无一例外。

**方向也是错的**，不只是幅度不够：`jerk_ratio` 两档都**上升**；`pen_ratio` 两档都**变差**；
`contact_count` 两档都**离 GT 更远**；而门槛 4 的弹性检验显示 `mean_speed` 的下降比
jerk 的下降**更快**（tau1 0.965 vs 0.981，tau2 0.947 vs 0.956）—— 也就是说 cap 那一点
微弱的作用反而更像低通，而不是有针对性的修复。tau1 的 >5g episode 数甚至从 16 升到 17。

五道门槛逐条：tau1 与 tau2 均 G1=FAIL G2=FAIL G3=FAIL G4=FAIL G5=FAIL。

### E. n=20 门槛的三处改写（在任何 arm 运行之前写定）

§H 的门槛是在 375 个 episode 上标定的，其中三条无法逐字迁移到这个子集：

1. **门槛 4 的"幅度下降不超过 5%"在这个子集上任何修好 jerk 的 arm 都过不了** ——
   C+guided 自己就是 B 的 `mean_speed` 0.644×、bbox 体积 0.815×、pelvis 水平路径 0.709×，
   因为 B+guided 的幅度本身就被缺陷抬高了。改写为 §E 真正用的判别式：幅度 ≥ 0.95 ×
   C+guided，且 jerk 的下降必须严格快于幅度的下降。B+unguided 的水平路径 0.653× 与 C 的
   0.709× 相差 8%，说明这个幅度下限是"修好后的运动"的性质，不是 C 的性质。
2. **门槛 2 的"≤2.0× GT"在这个子集上不可达** —— C 自己是 2.587× GT。改写为等强度的
   "关闭 B→C 差距的 ≥84.4%"，即 `boundary_jerk` ≤ 334.1。
3. **门槛 1 的 pelvis<0.6 m 必须限定在 walk** —— 16 个 interactive episode 里，
   B+guided 是 10/16，两个参照 arm 都是 **16/16**：坐下和躺下时骨盆本来就该低。
   限定在 4 个 walk 上：B+guided 3/4（0.273/0.349/0.496/0.774 m），两个参照 arm 都是 0/4。

另有一处自查修正：门槛 3 初稿把 `min_dist` 写成下界，方向反了（作弊方式是它**变大**），
在任何 tau 结果产生之前改为上界 0.00554 并在预注册中标注了修改与理由。

### F. 下一个最小方案：step-count normalization（未实施，等批准）

§C 直接指出了杠杆：**是增量的个数，不是增量的大小。**两个都是一行、config 开关、默认关闭、
只动 diffusion 路径、不碰 C、不碰 `code/priors/core/`、不需要重训：

1. **总剂量归一化** —— 给 B 的增量乘一个常数，使每窗口总位移与 C 相等：
   `s = 974/23225 = 1/23.8`。这直接对准实测的 23.8×。
2. **重新启用已经写在代码里但被注释掉的 `(1 - alpha_cumprod)` 逐步衰减** ——
   `models/infbagel.py` 的 guidance 处原作者写了这个因子又注释掉了，它按噪声水平压低
   高噪声阶段的 guidance。零新参数，是可能的最小改动。这一条还回答了 (1) 回答不了的问题：
   500 步与 16 步的增量即使总量相等，作用也不相等，因为早期（高噪声）的位移会被后续去噪
   冲掉，晚期的不会。

建议用同一批 20 个 episode、同一套门槛跑这两档（约 35 分钟 8 卡），仍然先 smoke 再决定。
门槛 2 的目标 334.1 落在 C+guided 的 228.6 与 B+guided 的 904.7 之间，而 B+unguided 是
161.2，所以正确的剂量归一化在原理上有落点，不像 norm cap 那样连着力点都没有。

**未做**：没有修改或重训 C，没有加 continuous-w loss，没有启动 375 全量实验，没有在三档
失败后继续调 tau。continuous-w 继续暂缓。

## 2026-08-23（同日第四次）：guidance 剂量两档 —— jerk 被彻底解决，代价与门槛的判定另有一层

用户接受 norm-cap 的失败结论，批准在同一批 20 个 episode 上跑两档剂量方案。代码
commit `e9bcf77`：`hsi_guidance_dose_scale`（默认 `null`）与 `hsi_guidance_alpha_decay`
（默认 `false`），两个独立开关，只作用于 `Sampler.p_sample`。运行 15:26–15:59，45/45 job
rc=0，完整读数 `.claude/scratch/normcap_20260823/FULL_REPORT_DOSE.txt`。

**按冻结门槛的判定：两档都 FAIL。**但失败的性质与 norm-cap 完全不同：两档都以极大幅度
通过 G1、G2、G5，各自只在 G3 和 G4 的**单条子条款**上失败，而这些子条款全部是本轮为 n=20
新加的，其中一条（G4 的 `pelvis_path_horiz` 幅度下限）经查是**标定错误** ——
B+unguided 这个正当参照本身也过不了。

### A. 机制确实被改变了（用户要求核实的一项）

每窗口累计**实际施加**的 guidance norm。探针记录的是 `autograd.grad` 输出的范数，
即施加 `guidance_scale`（恒为 1）与任何 knob 之前的量；每个 knob 都是整张张量上的
per-sample 标量，所以施加后的范数可精确推出。

| arm | 每窗口累计 norm | ×B | **×C** | 预测 |
|---|---:|---:|---:|---|
| B+guided（原始） | 23,224.8 | 1.000 | **23.84** | — |
| `decay` | 13,930.4 | 0.600 | **14.30** | 0.596× / 14.21× ✓ |
| `dose1` = 1/23.8 | 1,498.4 | 0.065 | **1.54** | 0.042× / 1.00× ✗ |
| C+guided（参照） | 974.1 | 0.042 | 1.00 | — |

`dose1` 没有落在预测的 1.00× C 上，原因是**原始梯度范数自己涨了**：46.54 → 71.47。
guidance 被削弱后轨迹漂进穿模更深的区域，那里 guidance loss 的梯度更大。这本身是该项
作为回复力在起作用的证据，任何静态预测都不可能算出这个反馈。`decay` 的原始范数也涨了
（46.54 → 51.99），幅度小得多。

### B. 结果（n=20 均值）

| 指标 | B+guided | **dose1** | **decay** | B+unguided | C+guided | GT |
|---|---:|---:|---:|---:|---:|---:|
| `boundary_jerk` | 904.71 | **160.02 (1.81×)** | **148.38 (1.68×)** | 161.19 (1.82×) | 228.59 (2.59×) | 88.37 |
| `interior_jerk` | 269.78 | 71.32 (0.95×) | 68.16 (0.91×) | 70.57 | 91.33 | 75.22 |
| `jerk_ratio` | 3.789 | 2.268 | 2.205 | 2.311 | 2.358 | 1.185 |
| `transition_distance_aligned` | 0.02475 | 0.00855 | **0.00805** | 0.00938 | 0.00942 | 0.00701 |
| 有 >5g 帧的 episode | 16/20 | **0/20** | **0/20** | 0/20 | 2/20 | — |
| >2g 帧总数 | 265 | **1** | **0** | 0 | 25 | — |
| 平均 max\|acc\| (m/s²) | 91.69 | 10.69 | 10.16 | 10.36 | 18.41 | — |
| `pen_ratio` | 0.07225 | 0.07603 | 0.08087 | 0.08814 | 0.06647 | 0.05503 |
| `min_dist` | 0.00504 | 0.00385 | 0.00285 | 0.00254 | 0.00433 | 0.0 |
| `contact_count` | 1704.21 | 1836.64 | 1841.97 | 1965.94 | 1729.93 | 1325.21 |
| `goal_planar_err_m` | 0.13842 | **0.07911** | **0.07157** | 0.06856 | 0.08039 | 0.0 |
| `success_last_10cm` | 0.95 | **1.00** | **1.00** | 1.00 | 1.00 | 1.00 |
| `fs_nemf` | 0.46942 | 0.40152 | 0.39417 | 0.41616 | 0.40100 | 0.32355 |
| `skate_ratio` | 0.11844 | 0.14772 | 0.14922 | 0.15897 | 0.14575 | **0.14702** |
| `mean_speed` | 0.4209 | 0.2646 | 0.2566 | 0.2700 | 0.2712 | — |

配对 bootstrap 对 B+guided：`boundary_jerk` dose1 **−744.69 [−867.56, −626.55] SIG**，
decay **−756.33 [−881.42, −637.46] SIG**；`interior_jerk`、`jerk_ratio` 同为 SIG；
`goal_planar_err_m` 两档 SIG 改善；`pen_ratio`、`pene_pct_scene`、`contact_count` 两档均
**ns**。恒等自检全指标 delta 0 且 ns。

两档在 `boundary_jerk`、`interior_jerk`、`jerk_ratio`、`transition_distance_aligned`、
`fs_nemf`、`skate_ratio`、`goal_planar_err_m` 七项上都比 **C+guided 更接近 GT**；
`skate_ratio` dose1 是 1.005× GT，几乎正中。B+guided 的 0.806× 不是"更好"，是低于真值。

### C. 五道门槛逐条

两档同为 **G1=PASS G2=PASS G3=FAIL G4=FAIL G5=PASS**。失败点只有两处：

**G3 只败在 `contact_count`。**§H 原文的显著性检验两档都是 **ns**（+132.4
[−183.6, +449.3] 与 +137.8 [−187.8, +471.0]），所以按 §H 口径这一条是过的；败的是本轮
新加的"点估计不得比 B+guided 离 GT 更远"（|arm−GT| 511.4 / 516.8 对 |Bg−GT| 379.0）。
`contact_count` 与穿模同向，两档都落在 B+guided 与 B+unguided 之间，与部分剂量削减一致。

**G4 只败在幅度下限，且该下限本身标定有误。**弹性判别式（§E 真正的低通检验）两档在四项
幅度指标上全部通过，而且余量极大：jerk 降到 0.177×/0.164×，幅度只降到 0.61–0.93×，
`jerk_ratio` 两档都下降。败的是绝对下限 0.95 × C+guided：

| 幅度指标 | 下限 | C+guided | B+unguided | dose1 | decay | B+unguided 过下限？ |
|---|---:|---:|---:|---:|---:|:--|
| `mean_speed` | 0.2577 | 0.2712 | 0.2700 | 0.2646 ✓ | **0.2566 ✗** | YES |
| `bbox_vol` | 1.8902 | 1.9897 | 2.0711 | 2.0364 ✓ | 2.0635 ✓ | YES |
| `bbox_xz` | 1.1718 | 1.2335 | 1.3112 | 1.2815 ✓ | 1.2969 ✓ | YES |
| `pelvis_path_horiz` | 1.2301 | 1.2949 | 1.1924 | **1.2244 ✗** | **1.1835 ✗** | **NO** |

`pelvis_path_horiz` 上 B+unguided 自己是 0.921× C，低于我设的 0.95× 下限 —— 我用两个相差
8% 的参照去标定，却只给了 5% 的带宽。这一项的失败是门槛的构造缺陷，不是 arm 的性质。
但 `decay` 在 `mean_speed` 上短 0.4%，而两个参照在该项都通过，那一处是 `decay` 自己的
幅度不足，不能用同一理由解释。

### D. Pareto 与判别标准

两档互不支配：`decay` 在 `boundary_jerk` 上比 `dose1` 好 **−11.64 [−21.26, −2.59] SIG**，
`dose1` 在 `pen_ratio` 上好 0.00483（ns）。真正分开它们的是**与 unguided 端点的比较**：

| | vs B+unguided `pen_ratio` | vs B+unguided `pene_pct_scene` | vs B+unguided `boundary_jerk` |
|---|---|---|---|
| `dose1` | **−0.01211 [−0.02561, −0.00257] SIG** | **−0.01137 [−0.02844, −0.00020] SIG** | −1.18 ns |
| `decay` | −0.00727 [−0.01769, +0.00106] **ns** | −0.00702 **ns** | −12.81 SIG |

**`decay` 在穿模上与"直接关掉 guidance"统计不可区分** —— 它赢的那一项（jerk）unguided
本来就免费拿到。**`dose1` 是整个研究里唯一一个在穿模上显著优于 `use_guidance=false`
而 jerk 与之持平的配置**，也就是唯一一个让"继续开着 guidance"有可测理由的配置。
因此若用户决定放宽 §C 的两条子条款，推荐 **`dose1`**。

### E. Inertness 在第二次改源码之后重新测过

`probe_off2`（两个 knob 都不设，5 个 episode）对封存 B v2 guided：265 个数值指标 0 不一致，
110 个 motion 数组 0 逐位不一致。§A 表中 `probe_off2` 的 0.868× 是 5 episode 子集所致
（那 5 个是 4 个 walk 加一个长 interactive），不是扰动。

**未做**：按用户指令"两个都失败即立即停止"，没有追加任何缩放系数或调参，没有启动 375
全量实验，没有修改或重训 C，没有加 continuous-w loss。norm-cap 保留在树上且默认关闭。
两条子条款是否放宽、以及是否用 `dose1` 上全量，等用户决定。

## 2026-08-23（同日第五次）：dose1 全量运行的门槛 —— 启动前冻结，不再依结果调整

用户接受 §四 的机制结论，选择 `dose1` = 1/23.8，不推进 `decay`，并同意把本轮两条失败子条款
判定为 n=20 smoke 阶段的**错误标定**而不是事后宣布通过。本节在 dose1 的 375-episode 运行
**启动之前**写定全部门槛、方向与比较对象，此后不再修改。所有数值取自已封存的
`p1-hsi-b-v2-eval-epoch222-guided-shard8-s42-20260822`、
`p1-hsi-b-v2-eval-epoch222-unguided-shard8-s42-20260823`、`c_guided_v5_baseline`、
`ground-truth-v3`，计算脚本 `.claude/scratch/normcap_20260823/baseline355.py` 与
`amp_envelope.json`。

### A. 集合划分：最差 20 已被用于选型，故不是 confirmatory

| 集合 | n | 角色 |
|---|---:|---|
| **holdout355** | 355 | **主要 confirmatory**，全部门槛在此判定 |
| full375 | 375 | 次要，§H 原文门槛在此逐字判定 |
| worst20 | 20 | 仅作故障修复诊断，**不参与通过判定**（曾用于选型） |

holdout 的 walk 子集 126 个（全集 130 减去 worst20 里的 4 个）。

### B. holdout355 上的封存基线

| 指标 | B+guided | B+unguided | C+guided | GT | Bg/GT | Cg/GT |
|---|---:|---:|---:|---:|---:|---:|
| `boundary_jerk` | 184.064 | 126.050 | 155.101 | 84.217 | 2.186 | 1.842 |
| `interior_jerk` | 82.389 | 63.331 | 67.304 | 70.163 | 1.174 | 0.959 |
| `jerk_ratio` | 2.2105 | 2.0144 | 2.1668 | 1.1941 | 1.851 | 1.815 |
| `pen_ratio` | 0.02286 | 0.03241 | 0.02285 | 0.02653 | **0.861** | 0.861 |
| `pene_pct_scene` | 0.04938 | 0.05716 | 0.05032 | 0.04855 | 1.017 | 1.037 |
| `min_dist` | 0.00480 | 0.00306 | 0.00393 | 0.0 | — | — |
| `contact_count` | 847.476 | 885.091 | 828.690 | 757.564 | 1.119 | 1.094 |
| `goal_planar_err_m` | 0.08209 | 0.05065 | 0.05993 | 0.0 | — | — |
| `success_last_10cm` | 0.98310 | 0.99155 | 0.98310 | 1.0 | — | — |
| 有 >5g 帧的 episode | **15/355** | 0/355 | 8/355 | — | — | — |
| >5g 帧总数 | 47 | 0 | 38 | — | — | — |
| >2g 帧总数 | 243 | 7 | 119 | — | — | — |
| walk `h_min`<0.6 | **10/126** | 0/126 | 1/126 | — | — | — |

**注意 holdout 上 B+guided 的 `pen_ratio` 是 0.861× GT，即穿模比真值还少。**所以往
unguided（0.03241 = 1.222× GT）方向移动在这个集合上是实打实的退化，比 worst20 上更受约束。

### C. 冻结的门槛

**G1 impossible acceleration（必要）**
- holdout：有 >5g 帧的 episode **15 → ≤8**（C 在同一集合上的水平，§H 原文"降到 C 的水平"）；
  >5g 帧总数 **47 → ≤38**（同上）；walk `h_min`<0.6 **10/126 → ≤2**（§H 的 13→≤3 即 ≤23%，
  作用于 10 得 ≤2；C 为 1，B+unguided 为 0）。
- full375：§H 逐字，episode **31 → ≤10**，walk **13/130 → ≤3**。

**G2 boundary jerk（必要）**
- G2a：`boundary_jerk` **≤2.0× GT**（§H 逐字）—— holdout ≤168.43，full375 ≤168.88。
- G2b：holdout 的 `boundary_jerk` **≤155.101**，即不得差于 C+guided。B+guided 是 184.064，
  所以这是一条实质门槛，不是自动通过项。
- G2c：对 B+guided 的配对 bootstrap（10,000 次，seed 42，重采样 episode）delta 显著为负，
  holdout 与 full375 都要满足。

**G3 场景交互不得退化（必要，按用户本轮修订）**
判据回到 §H 的显著性/非劣规则，**删除本轮新加的"点估计不得比 B+guided 离 GT 更远"**。
对 B+guided 的配对 bootstrap，下列各项不得**显著变差**：
- `pen_ratio`、`pene_pct_scene`：不得显著**升高**；且点估计不得超过 B+unguided
  （holdout 0.03241 / 0.05716）—— 若让出的穿模与直接关掉 guidance 一样多，这个开关就没有意义。
- `min_dist`：不得显著**升高**（变大 = 离表面更远 = 用"躲开一切"换平滑）。
- `contact_count`：不得显著**升高**。方向在此写定：contact 与穿模同向，且四个 cell 里 GT
  最低（757.56 对 B+guided 的 847.48），故"变差"= 变高。两侧 CI 一并报告。
- `success_min_10cm`、`success_last_10cm`：不得显著**降低**。
- `goal_planar_err_m`：不得显著**升高**。

**G4 不得是全局低通（必要，按用户本轮修订）**
不再要求达到 0.95 × C+guided 的固定值。该标定是错的：C+guided 与 B+unguided 这两个正当参照
在 `pelvis_path_horiz` 上相差 8%（1.2949 对 1.1924，即 0.921× C），而我只给了 5% 带宽，
于是 B+unguided 自己也过不了那道门槛。改为**以两个参照张成的包络为准**：
- G4a 包络下限 = **0.95 × min(B+unguided, C+guided)**，同一集合上计算，无上限
  （幅度更大不构成作弊）：

  | 幅度指标 | holdout355 下限 | full375 下限 |
  |---|---:|---:|
  | `mean_speed` | 0.2118 | 0.2142 |
  | bbox 体积 (m³) | 1.8631 | 1.8645 |
  | bbox xz 面积 (m²) | 1.1547 | 1.1557 |
  | pelvis 水平路径 (m) | 1.1345 | 1.1345 |

- G4b 反低通判据（保留）：`boundary_jerk` 的相对下降比 (arm/B+guided) 必须**严格小于**
  上述四项幅度的相对下降比 —— jerk 必须比幅度降得明显更快。
- G4c 保留：`jerk_ratio` 必须相对 B+guided 下降。

**G5 guidance 必须仍然在买到场景合规（必要，用户本轮明确要求）**
holdout 上 `pen_ratio` 与 `pene_pct_scene` 必须**显著优于 B+unguided**（配对 bootstrap CI
在改善方向上排除 0）。这一条是 §四 里区分 dose1 与 decay 的判据，现在提升为正式门槛：
若不满足，正确结论是 `use_guidance=false`，而不是这个开关。

**总判定**：以 **holdout355 上 G1–G5 全部通过**为通过。full375 按 §H 原文并列报告，
worst20 仅作故障修复诊断。若未通过则停止汇报，不追加比例调参。

### D. 修订的诚实记账

按修订后的 G4a，§四 的两档在 worst20 上的幅度下限会变成 `mean_speed` 0.2565、
pelvis 水平路径 1.1328，于是 `dose1`（0.2646 / 1.2244）与 `decay`（0.2566 / 1.1835）
都会通过。这里如实记下这一点，是为了说明修订确实改变了历史判定，而不是假装它只影响未来；
但 §四 的记录保持原样，那一轮按当时冻结的门槛就是 FAIL。

### E. 运行配置

与 `p1-hsi-b-v2-eval-epoch222-guided-shard8-s42-20260822` 完全一致 —— 同一 checkpoint
（`hsi_b_lingo_full_v2_epoch222.pth`，sha `5daaf813`）、seed 42、同一 episode 集合与 8 分片
切分、`hsi_progress_fix=true`、`export_motion=true`、同一评估器 —— 仅增加一个覆盖：
`hsi_guidance_dose_scale=0.042016806722689076`。timing 按设计为 null（分片运行自动作废）。
不运行 `decay`，不追加比例调参，不修改 C，不训练模型。

## 2026-08-23（同日第六次）：dose1 全量确认 —— jerk 与异常加速度全过，穿模/接触未过

`p1-hsi-b-v2-eval-epoch222-guided-dose1-shard8-s42-20260823`，8 分片，**5 h 14 m**
（16:23:06–21:37:19），8/8 shard fail=0，merge exit 0，**22 条 merge 锚全部通过**
（含 `shard_episode_counts`/`shard_window_totals` 与封存 guided cell 完全相同、
`excluded_as_warmup`=5、key 集合相同、checkpoint sha `5daaf813`、375 个 motion export）。
门槛在 commit `1172ca6` 即运行启动前冻结。完整读数
`.claude/scratch/normcap_20260823/confirm.out` 与 `verdict.out`。

### A. 判定：holdout355 与 full375 均 **FAIL**（G3、G5）

| 门槛 | holdout355 | full375 |
|---|---|---|
| G1 异常加速度 | **PASS** | **PASS** |
| G2 boundary jerk | **PASS** | **PASS** |
| G3 场景交互不得退化 | **FAIL** | **FAIL** |
| G4 不得是低通 | **PASS** | **PASS** |
| G5 仍须优于 unguided | **FAIL** | PASS（勉强） |

### B. holdout355 主表（n=355）

| 指标 | B+guided | **dose1** | B+unguided | C+guided | GT | dose1/GT |
|---|---:|---:|---:|---:|---:|---:|
| `boundary_jerk` | 184.064 | **129.547** | 126.050 | 155.101 | 84.217 | **1.538** |
| `interior_jerk` | 82.389 | **64.154** | 63.331 | 67.304 | 70.163 | **0.914** |
| `jerk_ratio` | 2.2105 | 2.0537 | 2.0144 | 2.1668 | 1.1941 | 1.720 |
| `transition_distance_aligned` | 0.00707 | 0.00671 | 0.00689 | 0.00653 | 0.00638 | 1.052 |
| `pen_ratio` | 0.02286 | **0.02921** | 0.03241 | 0.02285 | 0.02653 | **1.101** |
| `pene_pct_scene` | 0.04938 | **0.05610** | 0.05716 | 0.05032 | 0.04855 | **1.156** |
| `min_dist` | 0.00480 | 0.00346 | 0.00306 | 0.00393 | 0.0 | — |
| `contact_count` | 847.476 | **884.223** | 885.091 | 828.690 | 757.564 | 1.167 |
| `success_min_10cm` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| `success_last_10cm` | 0.98310 | **0.99718** | 0.99155 | 0.98310 | 1.0 | 0.997 |
| `goal_planar_err_m` | 0.08209 | **0.05384** | 0.05065 | 0.05993 | 0.0 | — |
| 平均 max\|acc\| (m/s²) | 15.625 | **8.995** | 8.755 | 11.333 | — | — |
| 有 >5g 帧的 episode | **15/355** | **0/355** | 0/355 | 8/355 | — | — |
| >5g 帧总数 | 47 | **0** | 0 | 38 | — | — |
| >2g 帧总数 | 243 | **8** | 7 | 119 | — | — |
| walk `h_min`<0.6 | 10/126 | **1/126** | 0/126 | 1/126 | — | — |
| `mean_speed` | 0.24030 | 0.22643 | 0.22292 | 0.22321 | — | — |
| bbox 体积 | 2.01323 | 1.99808 | 1.98554 | 1.96113 | — | — |
| pelvis 水平路径 | 1.37039 | 1.24824 | 1.19426 | 1.23152 | — | — |

### C. 通过的三条

**G1：异常加速度彻底消失。**>5g episode **15/355 → 0/355**，>5g 帧 **47 → 0**
（上限 8 与 38）；full375 **31 → 0** 与 **132 → 0**（上限 10）；>2g 帧 243 → 8；
walk pelvis<0.6 m 10/126 → 1。平均 max|acc| 15.63 → 9.00 m/s²，已低于 C 的 11.33。

**G2：`boundary_jerk` 184.064 → 129.547 = 1.538× GT**，低于 2.0× 上限，也低于 C+guided 的
155.101 —— 而 B+guided 的 184.064 是高于 C 的。delta **−54.516 [−67.868, −42.828] SIG**。
`interior_jerk` **0.914× GT**（低于真值），`jerk_ratio` 2.211 → 2.054。full375
222.498 → **131.172 = 1.553× GT**，delta **−91.326 [−113.530, −71.673] SIG**。
这个水平与 B+unguided 的 1.497× 基本持平，即已经触到 §2×2 判定的 16 帧自回归 rollout 地板。

**G4：不是低通。**四项幅度下限全部通过且有余量；jerk 相对降到 **0.704×**，而幅度只降到
0.911–1.003×（bbox xz 甚至 1.003× 略微变大）；`jerk_ratio` 下降。

### D. 未通过的两条，以及原因

**G3：穿模与接触按 §H 自己的显著性规则显著变差。**
`pen_ratio` +0.00636 [+0.00444, +0.00855] **SIG**（0.861× GT → **1.101× GT**）；
`pene_pct_scene` +0.00672 **SIG**（1.017× → **1.156×**）；`contact_count` +36.747
[+4.075, +74.703] **SIG**。三项都仍落在 B+unguided 带内，而 success 与 goal 反而**改善**
（`success_last_10cm` 0.98310 → 0.99718 SIG，`goal_planar_err_m` 0.08209 → 0.05384 SIG，
`success_min_10cm` 始终 1.0000）。注意这不是本轮修订过的条款：修订只动了
`contact_count` 的方向定义与 G4 的幅度包络，而 `pen_ratio`/`pene_pct_scene` 的显著性检验是
§H 原文，dose1 在其上确实显著退化。

**G5：在普通 episode 上 guidance 已经不再可证明地买到场景合规。**
`pene_pct_scene` 只比 B+unguided 的 0.05716 好 **1.9%**，且 **ns [−0.00205, +0.00004]**；
`contact_count` 884.223 与 unguided 的 885.091 **实质相等**；只有 `pen_ratio` 保住显著优势
（−0.00320 [−0.00412, −0.00219] SIG）。full375 上 G5 勉强通过（`pene_pct_scene`
−0.00160 [−0.00292, −0.00037] SIG），但 G3 同样失败。

**机制**：1/23.8 是按"让 B 的每窗口总位移等于 C"标定的，它在病态 episode 上正好，在普通
episode 上**过度削弱**。worst-20 掩盖了这一点 —— 那里缺陷大到任何剂量削减都是净胜。

### E. worst-20 故障修复（仅诊断，不参与判定）

| 指标 | B+guided | dose1 | C+guided | GT |
|---|---:|---:|---:|---:|
| `boundary_jerk` | 904.709 | **160.015** | 228.594 | 88.372 |
| `interior_jerk` | 269.779 | 71.317 | 91.335 | 75.218 |
| 有 >5g 帧的 episode | 16 | **0** | 2 | — |
| >5g / >2g 帧 | 85 / 265 | **0 / 1** | 3 / 25 | — |
| walk `h_min`<0.6 | 3/4 | **0/4** | 0/4 | — |

这些数字与 §四 的 20-episode smoke **逐位一致**（`boundary_jerk` 160.01540），
说明 8 分片全量运行复现了 375 分片单 episode smoke，两者可直接比较。

**未做**：按用户指令完成即停。没有重新调整比例系数，没有启动任何后续实验，没有修改或重训 C。
`hsi_guidance_dose_scale`、`hsi_guidance_norm_cap` 保持默认 null，
`hsi_guidance_alpha_decay` 保持默认 false，continuous-w 继续暂缓。

## 2026-08-23（同日第七次）：dose1 失败后的异质性分析 —— 无 GPU，只重读已封存数据

用户接受 full375 的正式 FAIL 判定，`dose1` = 1/23.8 只作为确认 guidance 过量机制的诊断结果，
不提升为 baseline；原始 B guided 也不视为问题已解决。本节在**不实现、不启动任何方案**的前提下，
用已在盘上的五个 cell（B+guided = 剂量 1、`dose1` = 1/23.8、B+unguided = 0、C+guided、GT）做异质性
分析，并为下一轮冻结抽样集与门槛。脚本与全量读数 `.claude/scratch/hetero_20260823/`，
汇总 `FULL_REPORT.txt`。

计算口径：所有数字来自已封存的 per-episode 指标，或对**已导出的 motion** 做 SMPL-X FK 加
预建 2 cm 场景 SDF 查表。没有任何采样、没有模型前向、没有训练。窗口级穿模用 28 关节的
fast diagnostic（`priors/hsi/metrics.py` 明确它与 10475 顶点口径**不可互换**），仅用于
同一 cell 内跨窗口归因；与封存顶点口径的 episode 级 spearman 为 0.9156–0.9420，绝对值从不跨口径比较。

### A. jerk 收益与穿模代价不是同一个现象

| dose1 − B+guided | 总量 | 最差 20 episode 占比 | 最差 50 占比 | 变差的 episode 数 |
|---|---:|---:|---:|---:|
| `boundary_jerk` | −34247 | **48.7%** | 74.6% | 64/375 |
| `pen_ratio` | +2.333 | 26.7% | 43.8% | **301/375** |
| `pene_pct_scene` | +2.580 | 34.0% | 51.8% | **297/375** |
| `acc_max` | −3974 | 49.4% | 75.2% | 82/375 |

**jerk 收益是尾部现象，穿模代价是普遍税。**两者的 episode 级 spearman 只有 **+0.056**
（`pen_ratio` +0.024），中位数四分位表近乎均匀（98/89/89/99）；只有两端极值重合
（top-50 交集 24/50，随机期望 6.7）。

按 B+guided 的 `acc_max` 分层最干净：**最低三层共 225 个 episode 一个 >5g 帧都没有**，
`boundary_jerk` 只降 9.7–26.3，却付出 +0.0039～+0.0052 的 `pene_pct_scene`；全部 132 个 >5g
帧集中在最高两层的 75 个 episode。这就是过度修正的量化形态。

代价确实付在 guidance 原本在做功的地方：
spearman(scale=1 时 guidance 相对 unguided 的穿模改善量, dose1 的穿模代价) = **+0.845**。
按该改善量五分位，最高一档（75 个 episode）承担 **87.1%** 的总穿模代价，同时也拿到最大的
jerk 收益（208.8）；最低一档（75 个，guidance 在 scale=1 时反而**加重**穿模）在 dose1 下
穿模与 jerk 双双改善。真正的两难只存在于约 75 个 episode 上。

按动作：`lie`(12) `boundary_jerk` 846.5 → 231.8，`standup`(23) 与 `lie` 的相对穿模代价最大
（+23.4% / +22.7%）；`walk`(130) 与 `manip`(105) 的 jerk 收益只有 −25.6% / −28.1%。

### B. 窗口级：伤害极度局部，且已经在第 0 窗出现

2271 个窗口中**只有 67 个（2.95%）含 >5g 帧**，192 个（8.45%）含 >2g 帧。这 67 个窗口只持有
**8.18%** 的全部穿模关节-帧（>2g 窗口持有 20.38%）。也就是说：只在这些窗口上削减剂量，
牺牲的场景合规工作约 8%，而全局常数削减的是约 96%。**这是状态相关方案唯一的量化依据。**

窗口序号分布：>5g 率在窗口 0–9 为 3–5%，**窗口 ≥10 恒为 0**（508 个窗口）。

**第 0 窗是可以做因果比较的唯一位置**（各 cell 共享同一段历史）：

| cell（仅第 0 窗，n=375） | max&#124;acc&#124; | >2g 帧 | >5g 帧 | 有 >5g 的 episode | pene(关节口径) | interior jerk |
|---|---:|---:|---:|---:|---:|---:|
| B+guided | 13.276 | 155 | **53** | **18** | 0.06990 | 125.83 |
| dose1 | 7.839 | 4 | 0 | 0 | 0.07662 | 87.29 |
| B+unguided | 7.744 | 4 | 0 | 0 | 0.08028 | 86.94 |
| C+guided | 8.822 | 38 | 14 | 7 | 0.06852 | 92.38 |
| GT | 2.396 | 0 | 0 | 0 | 0.07607 | 52.73 |

两个结论：**整个交易在第 0 窗就已经完整呈现**（jerk 收益与穿模代价同时出现，代价是即时的，
不是 rollout 累积）；而且第 0 窗 B+guided 的穿模已经**低于 GT**（0.06990 对 0.07607），
即在制造不可能加速度的同时已经过度修正。

### C. 在线可得信号：能检测已经出事，不能预测将要出事

在 B+guided 内部，以"窗口 k 是否含 >5g 帧"为标签，特征**只取窗口 k 之前已生成的帧**与 prompt
（1896 个有历史的窗口，49 个正例）：

| 信号 | AUC | 均值(负) | 均值(正) | 在线代价 |
|---|---:|---:|---:|---|
| 上一窗 `acc_max` | **0.939** | 9.28 | 63.81 | 免费 |
| 上一窗 interior jerk | **0.935** | 95.8 | 413.6 | 免费 |
| 历史末端已实现 max root accel | **0.928** | 6.48 | 45.64 | 免费 |
| 历史末 6 帧穿模深度 | 0.880 | 0.0023 | 0.0191 | 一次 SDF 查表 |
| 历史末端 root 速度 | 0.855 | 0.220 | 0.590 | 免费 |
| 本 episode 已发生的 >5g 帧数 | 0.837 | 0.227 | 2.898 | 免费 |
| 历史末 6 帧穿模比例 | 0.819 | 0.0638 | 0.1871 | 一次 SDF 查表 |
| pelvis 高度 | 0.511 | 0.827 | 0.840 | 免费（无信息） |

**但这些 AUC 是误导性的。**留一 episode 交叉验证、阈值取 p95 时规则触发 9.7% 的窗口、召回
87.8%；把正例按"本 episode 之前是否已经出现过 >5g"拆开：

- 之前**已经**出事的窗口：36/36，召回 **100%**
- 之前**没有**出事的窗口：7/13，召回 **53.8%**
- 而且 67 个 >5g 窗口中有 **18 个是窗口 0**，历史根本不存在

**按 episode 计的首次尖峰召回率只有 7/31 = 22.6%。**任何以"已实现历史"为输入的控制器，
本质上是事故检测器而不是事故预测器。

唯一能覆盖窗口 0 的信号在采样循环内部：`code/models/infbagel.py:985` 处
`x_start[:, :, :84]` 经 `denormalize_torch` 与 `transform_points` 得到 `[B, 16, 28, 3]`
的世界坐标关节，**slot 0 就是 root**。对它做二阶差分即得"当前去噪预测在本窗内的 root 加速度"，
纯 tensor 运算，不需要 SMPL-X 前向、不需要额外模型调用，且天然是 per-sample（满足
layout 中立性）。**这条信号在 500 步的哪一步开始可辨，是本轮离线分析唯一无法回答的问题。**

已录制的 raw gradient norm 只存在于最差 20 个 episode 的探针里，且与结果的相关性很弱
（`raw norm max` 对 `Bg acc_max` +0.068、对 >5g 帧数 +0.247；burstiness `max/p50` 对 >5g +0.399），
n=20 且全为极端样本，range-restricted，不可外推。**每步梯度范数不是伤害的编码方式**——
这与本日第三次记录的 norm-cap 失败一致。

### D. 禁止用于控制器的信号

| 信号 | 判定 | 理由 |
|---|---|---|
| 任何 GT 量 | **禁止** | 部署时不存在 |
| episode 总窗口数 / `seq_length` / `end_pi` | **禁止** | 本 harness 由 GT 序列长度导出，不是用户请求；其 AUC 0.474 无论如何也无信息 |
| `contact_labels` 通道 `x_start[:, :, 228:232]` | **不可用** | 全部 375 个 LINGO episode 都是 `is_object=false`，B 的 `is_mix=false`，该通道没有有效监督；未测量即等于未知，不得据此建控制器 |
| 本窗未来帧（相对被修正的帧） | **不适用** | 16 帧窗口是联合去噪的，窗内"未来"不是因果泄漏；跨窗未来才是 |
| 事后评估量（`boundary_jerk`、`pen_ratio`、`fs_nemf`、`rds` …） | **仅事后** | 需要完整轨迹与顶点级 SDF；其中本窗的**起始** seam jerk 在线可算，**结束** seam 不可 |

### E. 问题 4：「任何固定剂量都不可能兼顾两端」没有证据支持，而且离线不可能证明

盘上实际存在的剂量点（每窗累计**已施加** guidance 范数，探针实测）：

| arm | knob | 每窗施加量 | vs B | vs C | 是否有 375 个 episode |
|---|---|---:|---:|---:|---|
| B+guided | 无（released） | 23224.8 | 1.000 | 23.84 | 是 |
| `decay` | `(1-alpha_cumprod)` | 13930.4 | **0.600** | 14.30 | **否，只有 worst-20** |
| `dose1` | 常数 1/23.8 | 1498.4 | 0.065 | 1.54 | 是 |
| B+unguided | `use_guidance=false` | 0 | 0 | 0 | 是 |

**holdout355 上只有 s = 0、1/23.8、1 三个点，区间 (0.042, 1.0)（23.8 倍剂量跨度）完全未测。**

而区间内唯一存在的测量点反对该论断：worst-20 上 `decay` 在 **0.600×** 剂量下
`boundary_jerk` = 148.38，**低于关掉 guidance 的 161.19**，>5g 帧 85 → 0。
如实记下混淆：`decay` 是靠把**晚期低噪步**归零达到 0.600×，不是等比缩放，所以
"60% 剂量是安全的"与"晚期步才是危险的"这两种读法没有被分开——但两种读法都与
"内部常数必然失败"相矛盾。同一张表还给出第二个事实：`decay` 的剂量是 `dose1` 的 9.3 倍，
`pen_ratio` 反而更差（0.08087 对 0.07603，ns）——**总剂量也不决定穿模结果，剂量在调度中的
位置才决定。**

**离线不可能settle它，这一点本身是个结论。**冻结的 G3 是显著性判据（"不得显著升高"）。
任何以已测 per-episode 向量的**仿射函数**做的投影，其点估计与 bootstrap CI 会同比缩放，
显著性判决因此**严格不变**——已实证：`pen_ratio` 的 |delta|/半宽在 g = 1.0 到 0.01 全程恒为
**3.013**（`pene_pct_scene` 2.821、`contact_count` 1.044）。所以

- 在线性假设下，**任何**常数 s ∈ (0,1) 都过不了现在这条 G3，这是**门槛的性质，不是机制的性质**；
- 该不变性只会被"episode 间恢复速度不同"打破，而这一点在唯一可测的子段 [0, 0.042] 上确实成立
  （`pen_ratio` 的个体缺口回补比例：中位数 +0.284，p10 −0.192，p90 +0.807，24.2% 的 episode
  回补 ≤10%，8.5% 回补 ≥90%）。因此**结论是"未定"，不是"不可行"**。

按两个都只用实测数据拟合的模型外推（LINEAR：在 [1/23.8, 1] 上线性；POWER：按
"s = 1/23.8 时已实现的效应份额"逐指标拟合 s^b）：

| 指标 | s=0 | s=1/23.8 | s=1 | 4.2% 剂量已买到 | 拟合 b |
|---|---:|---:|---:|---:|---:|
| `boundary_jerk` | 126.05 | 129.55 | 184.06 | 6.0% | 0.886 |
| `pen_ratio` | 0.03241 | 0.02921 | 0.02286 | **33.5%** | **0.345** |
| `pene_pct_scene` | 0.05716 | 0.05610 | 0.04938 | 13.6% | 0.631 |
| `acc_max` | 8.755 | 8.995 | 15.625 | 3.5% | 1.058 |
| `contact_count` | 885.09 | 884.22 | 847.48 | 2.3% | 1.189 |

**穿模买得早、jerk 买得晚**——正是中等剂量可能成立的形态。

### F. 冻结的分层 smoke 集（替代 worst-20），及其**能与不能**判定什么

worst-20 无法发现过度修正，因为集合里没有一个普通 episode：在那里任何剂量削减都是净胜。
改为在 B+guided 的封存值上做固定分位分层，层内按 canonical ordinal 均匀间隔取样
（全程无 RNG），并用**总体权重** N_s/375 做加权估计，使小样本仍是 375 均值的无偏估计。

分层切点：`boundary_jerk` p20 = 111.84、p80 = 231.90；`pene_pct_scene`/GT p20 = 0.7886、p80 = 1.2222。
S1 = 存在 >5g 帧；否则 S2 = jerk > p80；否则 S4 = pene/GT < p20；S5 = pene/GT > p80；其余 S3。

| 层 | N | 权重 | 取样 n | 窗口 |
|---|---:|---:|---:|---:|
| S1 anomalous | 31 | 0.08267 | 16（全部取自已烧掉的 worst-20） | 84 |
| S2 highjerk_nospike | 46 | 0.12267 | 10 | 47 |
| S3 typical | 195 | 0.52000 | 16 | 102 |
| S4 pene_tight | 58 | 0.15467 | 10 | 46 |
| S5 pene_loose | 45 | 0.12000 | 8 | 85 |

合计 n = 60，364 个窗口，**每个 arm 约 0.84 h**（8 卡墙钟，6.71 GPU-hour）。新烧掉 42 个 episode，
累计已烧 62 个，剩余 confirmatory holdout **313**。

**回溯验证**（因为 `dose1` 已在全部 375 上测过，可以直接给这个设计打分）：加权估计在该 60 集上
给出 `pen_ratio` +0.00835 [+0.00307, +0.01613] **SIG**、`pene_pct_scene` +0.00918 **SIG** ——
**它会抓到 worst-20 漏掉的那次失败。**

**但必须写清它判不了什么。**在 n=60 上加权 bootstrap 的半宽与 375 上的真实效应对比：

| 指标 | n=60 半宽 | n=375 半宽 | 375 上的真实效应 | n=60 能判定？ |
|---|---:|---:|---:|---|
| `boundary_jerk` | 28.57 | 20.64 | −91.33 | 能 |
| `interior_jerk` | 7.55 | 5.93 | −27.85 | 能 |
| `jerk_ratio` | 0.176 | 0.087 | −0.230 | 能 |
| `goal_planar_err_m` | 0.0110 | 0.0054 | −0.0299 | 能 |
| `pen_ratio` | 0.00653 | 0.00220 | +0.00622 | **不能，需延后** |
| `pene_pct_scene` | 0.00745 | 0.00253 | +0.00688 | **不能，需延后** |
| `contact_count` | 105.3 | 37.5 | +41.9 | **不能，需延后** |
| `min_dist` | 0.00199 | 0.00057 | −0.00133 | **不能，需延后** |

**因此本 smoke 集的职责被限定为：G1、G2、G4 与 goal，以及"开关是否真的在预期机制上起作用"
的机制性检查。G3 / G5 的穿模判决在 n=60 上不可判定，smoke 阶段在这两项上通过没有意义，
必须延后到全量。**上表的 `pen_ratio` 一行是关键：把半宽压到效应的一半需要 n≈266，
那已经不是 smoke 了。**这直接推出候选 1 不设 smoke 阶段。**

（60 个 episode 的完整清单见下表，含 canonical ordinal、`shard_count=375` 下的 shard index、
窗口数与所属层，可按 08-23 已验证的分片规则逐位复现。）

| ordinal | shard/375 | sequence_id | 窗口 | 层 | Bg `boundary_jerk` | Bg pene/GT | Bg >5g 帧 | caption |
|---:|---:|---|---:|---|---:|---:|---:|---|
| 2 | 72 | `010:000433` | 5 | pene_tight | 130.91 | 0.6073 | 0 | play guitar with both hands |
| 10 | 80 | `010:000344` | 5 | anomalous | 1017.64 | 1.8904 | 4 | sit down on office chair |
| 14 | 170 | `010:000457` | 4 | pene_tight | 165.14 | 0.5923 | 0 | play guitar with both hands |
| 16 | 172 | `010:000426` | 4 | anomalous | 807.14 | 1.6980 | 6 | walk |
| 23 | 24 | `015:000917` | 7 | anomalous | 1392.01 | 0.1174 | 10 | lie down on sofa and face up |
| 26 | 40 | `015:000944` | 6 | highjerk_nospike | 382.59 | 0.3458 | 0 | get up from lying to sitting |
| 31 | 83 | `015:000970` | 5 | typical | 230.84 | 1.2057 | 0 | sit down on office chair |
| 36 | 88 | `015:000918` | 5 | highjerk_nospike | 1419.87 | 0.9911 | 0 | get up from lying to sitting |
| 44 | 45 | `018-1:001041` | 6 | pene_tight | 103.06 | 0.5114 | 0 | walk |
| 45 | 92 | `018-1:001033` | 5 | pene_loose | 199.96 | 1.4958 | 0 | sit down on toilet |
| 62 | 98 | `024:001787` | 5 | pene_tight | 196.83 | 0.7481 | 0 | walk |
| 64 | 100 | `024:001781` | 5 | pene_loose | 137.14 | 1.3569 | 0 | walk |
| 66 | 102 | `024:001768` | 5 | typical | 127.22 | 0.9668 | 0 | walk |
| 70 | 187 | `024:001765` | 4 | anomalous | 905.33 | 1.1688 | 6 | sit down on yoga ball |
| 71 | 188 | `024:001793` | 4 | anomalous | 999.90 | 1.1145 | 9 | sit down on yoga ball |
| 76 | 193 | `024:001807` | 4 | highjerk_nospike | 423.11 | 1.5332 | 0 | sit down on yoga ball |
| 88 | 198 | `027:002114` | 4 | highjerk_nospike | 569.31 | 0.4696 | 0 | sit down on chair |
| 94 | 204 | `027:002230` | 4 | pene_tight | 187.79 | 0.6563 | 0 | kneel down |
| 101 | 211 | `027:002237` | 4 | highjerk_nospike | 307.19 | 0.8539 | 0 | stand up from seat |
| 105 | 17 | `031:002588` | 9 | anomalous | 907.63 | 2.5153 | 2 | lie down on bed and face to left |
| 108 | 32 | `031:002567` | 7 | anomalous | 1556.42 | 2.4954 | 9 | lie down on bed and face up |
| 110 | 48 | `031:002577` | 6 | typical | 124.10 | 1.1118 | 0 | walk |
| 111 | 107 | `031:002565` | 5 | anomalous | 1060.98 | 1.8193 | 4 | sit down on bed |
| 119 | 213 | `031:002619` | 4 | pene_tight | 175.93 | 0.7878 | 0 | walk |
| 126 | 19 | `038-bed:003170` | 9 | pene_loose | 207.60 | 1.3189 | 0 | sit down in front of drum kit |
| 134 | 115 | `038-bed:003198` | 5 | typical | 130.05 | 0.8126 | 0 | play guitar with both hands |
| 136 | 117 | `038-bed:003127` | 5 | highjerk_nospike | 274.62 | 0.9320 | 0 | type on drum kit while in sitting position |
| 149 | 57 | `044:004178` | 6 | anomalous | 714.05 | 1.7090 | 7 | sit down on couch |
| 152 | 125 | `044:004139` | 5 | anomalous | 494.02 | 1.7886 | 9 | walk |
| 155 | 128 | `044:004223` | 5 | anomalous | 1119.20 | 1.5688 | 7 | walk |
| 157 | 221 | `044:004182` | 4 | highjerk_nospike | 251.61 | 0.5862 | 0 | get up from lying to sitting |
| 160 | 224 | `044:004170` | 4 | anomalous | 572.12 | 0.6445 | 4 | walk |
| 164 | 228 | `044:004225` | 4 | pene_tight | 116.76 | 0.5702 | 0 | drink from vodka with right hand |
| 166 | 3 | `045-new_loco:009707` | 50 | pene_loose | 115.56 | 1.5751 | 0 | walk |
| 174 | 231 | `045:004368` | 4 | pene_loose | 122.25 | 1.3538 | 0 | sit down on chair |
| 177 | 305 | `045:004307` | 3 | typical | 125.22 | 1.2068 | 0 | walk |
| 192 | 235 | `046:004474` | 4 | pene_loose | 170.81 | 1.2281 | 0 | sit down on toilet |
| 202 | 319 | `046:004430` | 3 | typical | 76.23 | 0.9156 | 0 | drink from bottle with right hand |
| 213 | 135 | `056:005718` | 5 | pene_tight | 122.51 | 0.7417 | 0 | talk on phone with right hand |
| 216 | 241 | `056:005711` | 4 | typical | 157.43 | 0.9755 | 0 | sit down on chair |
| 232 | 9 | `061-new_loco:009720` | 41 | typical | 123.07 | 0.9401 | 0 | walk |
| 234 | 62 | `061:006180` | 6 | anomalous | 811.15 | 1.2262 | 2 | sit down on chair |
| 240 | 258 | `061:006120` | 4 | pene_loose | 120.69 | 1.2527 | 0 | sit down on chair |
| 249 | 330 | `061:006207` | 3 | typical | 149.31 | 0.9191 | 0 | walk |
| 259 | 142 | `062:006212` | 5 | highjerk_nospike | 438.45 | 0.7905 | 0 | sit down on chair |
| 262 | 263 | `062:006249` | 4 | anomalous | 592.11 | 1.8140 | 3 | sit down on chair |
| 268 | 335 | `062:006253` | 3 | typical | 107.09 | 0.9056 | 0 | kneel down |
| 271 | 338 | `062:006247` | 3 | pene_tight | 116.34 | 0.7207 | 0 | walk |
| 275 | 145 | `071-write:007258` | 5 | anomalous | 1066.97 | 1.8032 | 2 | read book with right hand |
| 285 | 273 | `071-write:007243` | 4 | anomalous | 629.70 | 1.3803 | 1 | sit down on chair |
| 288 | 276 | `071-write:007263` | 4 | typical | 111.94 | 0.9378 | 0 | read book with right hand |
| 297 | 152 | `082-wash-brush_teeth:008092` | 5 | highjerk_nospike | 261.22 | 1.2400 | 0 | sit down on toilet |
| 304 | 285 | `082-wash-brush_teeth:008087` | 4 | pene_loose | 163.34 | 1.8487 | 0 | sit down on toilet |
| 311 | 292 | `082-wash-brush_teeth:007995` | 4 | typical | 95.31 | 0.7922 | 0 | brush teeth with toothbrush in right hand |
| 326 | 348 | `086-pick_up-put_down:008583` | 3 | typical | 135.79 | 1.0147 | 0 | walk |
| 338 | 67 | `097-drum_kit:009372` | 6 | typical | 134.52 | 0.8936 | 0 | sit down in front of drum kit |
| 342 | 71 | `097-drum_kit:009439` | 6 | pene_tight | 134.28 | 0.5466 | 0 | sit down on chair |
| 350 | 165 | `097-drum_kit:009423` | 5 | highjerk_nospike | 277.25 | 1.7405 | 0 | stand up from seat |
| 354 | 300 | `097-drum_kit:009383` | 4 | typical | 95.70 | 0.9194 | 0 | type on drum kit while in sitting position |
| 367 | 367 | `99-pick_up:009639` | 3 | typical | 145.48 | 1.0130 | 0 | walk |

### G. 在线信号台账（`.claude/scratch/hetero_20260823/SIGNAL_LEDGER.md`），及三处代码事实

**G1 — 决定性的在线信号存在，而且已经被物化。**`models/infbagel.py:985-986` 已经把
`x_start[:, :, :84]` 经 `denormalize_torch` 与 `transform_points(·, mat)` 变成
`global_jpos` = `[B, 16, 28, 3]` 的**世界坐标**关节，slot 0 是 pelvis。采样器自己的帧率是
**10 Hz**（`DATA_STEP=3`，`fps=30`），所以
`acc = (p[:,2:] - 2*p[:,1:-1] + p[:,:-2]) / 0.01` → `[B,14,3]`，
`peak = acc.norm(-1).amax(1)` → `[B]`。纯 tensor 运算，无 SMPL-X、无额外模型调用、无占据查询，
天然 per-sample，且**不需要历史，因此在窗口 0 就能用**。

独立复核（`coarse_proxy.py`，CPU）：对导出的 10 Hz pelvis 做线性插值后重算，
与 `windows.json` 里 FK 30 Hz 的 `acc_max` 在全部 2271 个窗口上最大绝对差
**3.45e-4 m/s²**（相对 1.9e-6）——评估指标的加速度**就是**粗率 rollout 的确定性函数。
作为 >5g 窗口检测器（B+guided，67/2271）：

| 粗率阈值 | 召回 | 精确率 | 触发窗口数 |
|---|---:|---:|---:|
| >1.00 g | 0.866 | 0.417 | 139 |
| >1.20 g | 0.806 | 0.482 | 112 |
| **>1.67 g** | **0.791** | **0.726** | **73** |
| >2.00 g | 0.731 | 0.831 | 59 |
| >2.50 g | 0.627 | 0.913 | 46 |

在 `dose1` 上同一阈值 >1.67g 触发 **0** 个窗口（>1.00g 触发 2 个），与 `dose1` 零个 >5g 帧一致。
**必须定价的保留：以上全部是 `x_start` 在 t→0 时的值。`x_start` 在 t=400/200/50 时是否已经预示
最终值，未测量，需要一次 GPU 探针。**

粗/细峰值比**不是** `interp_s=3`：`linspace(0, T-1, T*scale)` 不落在粗率节点上，二阶差分冲激
在相邻细样本间以漂移的相位分裂，实测 p05 1.561 / p50 2.428 / p95 6.830。因此阈值必须在**粗率上
直接标定**，不能用 3 去换算。

**G2 — `loss` 是 batch 归约后的 0 维标量。**`guidance_loss.py:98`
`F.mse_loss(human_jnts, nearest_free_points) * 20000`，默认 `reduction='mean'` 在
`B*16*24*3` 上归约。要拿 per-sample 值：`diff.pow(2).flatten(1).mean(1) * 20000` → `[B]`
（target 已经是 detached 的整型量）。改动**只涉及 `code/guidance_loss.py` 与
`models/infbagel.py:1005` 这一个调用点，不触及冻结的 `code/priors/core/`**——
`guidance_loss.py`、`models/infbagel.py`、`datasets/infbagel*.py`、`utils.py` 里没有任何
`priors.core` import。

**G3 — 两处已核实的代码事实，本轮不修，只登记。**

1. `apply_hsi_guidance_loss`（`guidance_loss.py:96-99`）**没有 batch-size 补偿**，
   而同文件的三个兄弟项都有：`:69` `loss = bs * (...)`、
   `:84` `loss = human_jnts.shape[0] * loss_feet_floor_contact`、
   `:115` `loss += bs * loss_floor_object * 100`。配合 `reduction='mean'`，
   **HSI guidance 的 per-sample 强度按 1/B 缩放。**评估协议是 B=1，所以现在完全不可见；
   mixer 一旦成批就是活的。已核实，不在本轮修改范围内。
2. `p_sample_loop:943-944` 在**每一步之后**调用 `set_fixed_points(..., fix_mode=True)`，
   把 `img[:, :fixed_points.shape[1], :]` 的全部 232 通道覆盖回历史值（导出记录
   `history_frames=2`，即 16 帧里的 2 帧）；而 loss 的 `human_jnts` 覆盖全部 16 帧。
   **对 G3-2 的读法要精确**：因为 `mse_loss` 是均值，去掉那 2 帧只会把分母从 16 改成 14，
   即所有元素的梯度**一致地**乘 16/14 = 1.143——这被自由常数 `guidance_scale` 完全吸收，
   **不是形状缺陷**。它真正的后果在别处：**`loss` 的数值系统性高报了"不可纠正"的穿模**，
   所以把 `loss` 当作控制器的状态量读取时必须先掩掉那 2 帧。这正是候选 2 要用它的场合。

**G4 — 禁止清单（已核实来源）。**`seq_length` / `episode_num` / 总窗口数及任何派生量
（完成比例、剩余窗口数）由 GT 序列长度导出：
`tools/make_lingo_hsi_episodes.py:112 → :118 → :46-47 → :150 → :187`，再由
`test_infbagel_lingo_hsi.py:1566-1570` 组装；`GroundTruthSource.episode_indices:203-210`
会在 episode 与 GT 不一致时**抛异常**，这本身就是它由 GT 导出的证明。**判定 PROHIBITED。**
（其 AUC 0.474 本来就是随机水平，禁掉不损失任何东西。）
`pi` / `end_pi` **不受影响**：`pi = step*42`、`end_pi = pi + 48`，对窗口序号确定，可用。
`object_goal` 是 GT 终点 pelvis 的字节拷贝（`make_lingo_hsi_episodes.py:179`），禁止。
`scene_goal` 是 `transform_points(zeros, inverse(mat))` = `-Rᵀt` 的原点伪影，不可用。

### H. 候选 1：固定 scale s = 0.40（简单基线，**建议先做这一个**）

**参数来源，逐条可复核**：在**保守的** LINEAR 投影上求满足 jerk 侧全部门槛的最大 s ——
G2b（`boundary_jerk` < C+guided 155.101）给 s ≤ 0.491，G1（>5g episode ≤ 8）给 s ≤ 0.595，
G2a（≤2.0×GT）给 s ≤ 0.725；取交集 s ≤ 0.491，留 10% 安全边际得 s ≤ 0.442，
向下取整到干净值 **s = 0.40 = 2/5**（= `dose1` 的 9.5 倍）。穿模在 s 上单调改善，所以
"清得过 jerk 侧的最大 s"同时也是固定方案能拿到的最好穿模——沿用挑 tau 时已经用过的
"通过门槛中干预最小的一档"。

**实现**：不改任何代码。已有开关 `hsi_guidance_dose_scale=0.40`，默认仍为 null。

**不设 smoke 阶段**，理由在 F 表里：n=60 上 `pen_ratio` 的半宽 0.00653 已经和总体效应 0.00622
同量级，smoke 判不了唯一有疑问的那条门槛，只会白烧 42 个 episode。直接上 375、8 分片，
holdout355 保持不变。

**冻结的门槛**：完全沿用 §C（2026-08-23 第五次）已冻结的 G1–G5，逐字不改，在 holdout355 判定。

**启动前写定的预测**，两个模型都给，运行后可逐条打分：

| 门槛 | LINEAR 预测 | POWER 预测 | 冻结上限 |
|---|---|---|---|
| G1 >5g episode | 3/355 | 3/355 | ≤8 |
| G1 >5g 帧 | 17.6 | 17.6 | ≤38 |
| G1 walk `h_min`<0.6 | 0/126 | 0/126 | ≤2 |
| G2a `boundary_jerk` | 149.92（1.780×GT） | 151.81（1.803×） | ≤168.43 |
| G2b vs C+guided 155.101 | **通过** | **通过** | 必须低于 |
| G4 四项幅度下限 + 反低通 | **全部通过**（jerk 比 0.814 对幅度比 0.944–1.002） | 同 | — |
| G5 `pen_ratio` vs B+unguided | −0.00558 **SIG** | −0.00697 **SIG** | 必须显著更好 |
| G3 `pen_ratio` delta vs B+guided | **+0.00398** | **+0.00259** | 不得显著升高 |
| G3 `pene_pct_scene` delta | +0.00421 | +0.00341 | 不得显著升高 |

即：**把 `dose1` 的两门失败（G3 + G5）压缩成最多一门（G3），G5 从 ns 变成 SIG 通过。**
G3 是唯一存疑项，两个模型对其代价的点估计相差 1.5 倍。

**成本**：与 `dose1` 全量同构（同为 500 步），约 **5.2 h** 8 卡墙钟 / 41.8 GPU-hour。
不新烧 episode。

### I. 候选 2：仅用在线信号的 in-loop 状态相关剂量（**只在候选 1 未通过时才做**）

**为什么必须是 in-loop 而不是按历史。**C 节的数字：67 个 >5g 窗口里 18 个是窗口 0（历史不存在），
留一交叉验证下按 episode 计的**首次尖峰召回率只有 22.6%**。以已实现历史为输入的控制器是
事故检测器，不是预测器。唯一覆盖窗口 0 的量是采样循环内部的 `x_start`（G1 节）。

**方案（per-sample，两级，不引入新的连续超参）**：在 `p_sample` 已算出 `gradient` 之后，
由 `global_jpos[:, :, 0, :]` 求本样本的粗率 pelvis 峰值加速度 `peak_i`，令

    scale_i = 1.0                      if peak_i <= a_lo
    scale_i = s_low (= 候选 1 的 0.40)  if peak_i >= a_hi

之间线性过渡；`gradient = gradient * scale_i.view(-1, 1, 1)`。
`a_lo` / `a_hi` 取 G1 表已标定的 **粗率 1.67 g / 2.50 g**（召回 0.791 精确 0.726 与
召回 0.627 精确 0.913 两档），不是新拟合的自由参数。独立配置开关，默认关闭；
只改 B 的 diffusion guidance 路径，不动 C，不动 `priors/core/`，不训练。

**必须先做一次只读探针，不做就不要实现。**唯一未测的前提是"`x_start` 在 500 步的哪一步开始
预示最终加速度"。探针只写 `.claude/scratch/`、不改剂量，因此采样出的 motion 必须与封存的
B+guided cell **逐位相同**（这是探针本身的通过条件，08-23 已经用同样方式验过三次）。
在 F 节冻结的 60 集上跑，成本约 **0.84 h** 8 卡墙钟 / 6.71 GPU-hour。
探针只读 ⇒ 不产生选择 ⇒ 不烧 episode；若之后要从探针数据里定阈值，只能用其中已烧掉的
16 个 anomalous episode 定，其余 44 个仅用于确认信号在普通 episode 上也存在。

**门槛**：与候选 1 完全相同的 §C 五道门，加两条实现性门槛：
(i) 阈值设到任何观测值之上时，运行必须与封存 B+guided 逐位相同（惰性）；
(ii) 报告实际触发的窗口比例与每窗累计施加范数，确认机制真的动了——这是本日第四次已经
要求过的验证方式。

**成本**：探针 0.84 h + 全量 5.2 h ≈ **6.0 h**。

**两个候选都不实现、不启动，等用户批准。**

### J. 需要用户决定的一件事：G3 的形式

E 节已实证：冻结的 G3 是显著性判据，而**任何常数剂量在线性假设下都过不了它**——
|delta|/半宽在剂量上严格不变（`pen_ratio` 恒为 3.013）。这条性质对候选 1 和候选 2 同样成立
（只要它们是"均匀削弱"）。因此有两条路，**这是用户的决定，我不代为选择**：

- **(a) G3 保持原样。**那么候选 1 应当作"解决问题 4 的一次测量"来跑，并**在启动前就承认
  它很可能过不了 G3**；它的价值在于给出区间内的第一个真实点，把 LINEAR 与 POWER 两个模型
  分开。G1/G2/G4/G5 仍然是真实的通过项。
- **(b) 在启动前把 G3 的穿模两项从"显著性"改为一个写定的非劣边界。**建议的具体形式：
  **必须保留 B+unguided → B+guided 穿模改善量的 ≥50%**，即
  `delta vs B+guided ≤ 0.50 × (B+unguided − B+guided)`，用配对 bootstrap 的 CI 上界判定。
  0.50 的来源写在这里：G5 已经要求"显著优于关掉 guidance"，50% 是在此之上仍使开关有意义的
  最弱边界。按此边界，候选 1 在 LINEAR 下保留 58.3%、在 POWER 下保留 72.9%，**两者都通过**。
  若选 (b)，必须在启动前改文档并冻结，此后不依结果调整。

## 2026-08-24：候选 1（固定 scale 0.45）的门槛 —— 启动前冻结，不再依结果调整

用户选择方案 (b)：在新候选运行前把 G3 定义为**实际非劣门槛**，并明确这是一次
**政策性 trade-off 定义**，**不回改 `dose1` 的既有 FAIL 结论**——2026-08-23 第六次记录的
那次判定按当时冻结的门槛就是 FAIL，保持原样。本节在运行**启动之前**写定门槛、scale、
配置与预测，此后不再修改。

### A. 为什么 scale 从 0.40 改成 0.45

0.40 是在**旧门槛结构**下推出来的：那时只有 jerk 侧从上面约束 s（G2b → s ≤ 0.491、
G1 → s ≤ 0.595、G2a → s ≤ 0.725），**没有任何条款从下面约束 s**，所以取"通过门槛中干预
最小的一档"就落到 0.40。新 G3 是一个保留率下界，**它从下面约束 s**，于是最优点上移。

绑定项是 `pene_pct_scene`：它的 B+unguided → B+guided 改善量只有 0.00777，50% 边界 0.00389。
细网格扫描（对全部门槛取最差归一化边际，再对 LINEAR 与 POWER 两个模型取最小，
`.claude/scratch/hetero_20260823/optscale.out`）：

| s | 最差边际（点估计形式） | 绑定项 | 最差边际（CI 上界形式） |
|---:|---:|---|---:|
| 0.40 | **−0.0829** | G3 pene 保留率 | −0.5028 |
| 0.44 | −0.0107 | G3 pene 保留率 | −0.4026 |
| **0.45** | **+0.0030** | G2b（POWER 侧） | −0.3776 |
| 0.46 | −0.0007 | G2b（POWER 侧） | −0.3525 |
| 0.59 | −0.0470 | G2b | **−0.0470（CI 形式最优）** |
| 0.60 | −0.1250 | G1 >5g episode | −0.1250 |

两条结论都记在这里：

1. **s = 0.45 是点估计形式下唯一使全部门槛在两个模型下都有正边际的值**（+0.0074 / +0.0030）。
   0.44 在 LINEAR 下 pene 保留率差 1.1%，0.46 在 POWER 下 G2b 差 0.07%。窗口只有 0.01 宽，
   这是因为 G2b 与 pene 保留率下界几乎相切，不是标定不当。
2. **CI 上界形式对任何固定 scale 都无解**：最优 s = 0.59 仍差 4.7%，而 s ≥ 0.60 时 G1 破线。
   "pene 的 CI 上界低于 0.00389"需要 s ≳ 0.62，"G2b 优于 C 的 155.101"需要 s ≤ 0.49。
   CI 上界形式实际要求约 **68%** 保留率而不是 50%。因此按用户决定，**判定用点估计，
   CI 与保留率 CI 下界并列报告作为敏感性，不参与判定**。

### B. 冻结的 G3（本轮唯一修改的门槛）

`pen_ratio` 与 `pene_pct_scene` 的判据从"不得显著升高"改为**非劣保留率**：

    improvement = mean(B+unguided) − mean(B+guided)            （正数，guidance 降低穿模）
    retention   = (mean(B+unguided) − mean(arm)) / improvement
    通过条件    : retention ≥ 0.50                              （点估计判定）

在 holdout355 上的具体数值，取自封存 cell：

| 指标 | B+guided | B+unguided | improvement | 50% 边界（delta 上限） | arm 允许的最大值 |
|---|---:|---:|---:|---:|---:|
| `pen_ratio` | 0.02286 | 0.03241 | 0.00956 | 0.00478 | **0.02764** |
| `pene_pct_scene` | 0.04938 | 0.05716 | 0.00777 | 0.00389 | **0.05327** |

**并列报告但不判定**：(i) `mean(arm − B+guided)` 的配对 bootstrap CI（10,000 次，seed 42，
重采样 episode）与其上界对 50% 边界的位置；(ii) 每个 replicate 内重算边界的 retention 的
CI 下界。两者都记录，供后续决定是否收紧。

**G3 其余条款不变**：`min_dist` 不得显著升高、`contact_count` 不得显著升高、
`success_min_10cm` / `success_last_10cm` 不得显著降低、`goal_planar_err_m` 不得显著升高；
`pen_ratio` / `pene_pct_scene` 的点估计仍不得超过 B+unguided。

**G1、G2a、G2b、G2c、G4a、G4b、G4c、G5 全部沿用 2026-08-23 第五次 §C 的冻结原文，逐字不变。**
holdout355 为主要 confirmatory，full375 与 worst20 并列报告，判定规则不变。

### C. 运行配置

与 `p1-hsi-b-v2-eval-epoch222-guided-shard8-s42-20260822` 完全一致 —— 同一 checkpoint
（`hsi_b_lingo_full_v2_epoch222.pth`，sha `5daaf813`）、seed 42、同一 episode 集合与 8 分片
切分、`hsi_progress_fix=true`、`export_motion=true`、同一评估器 —— 仅一个覆盖：

    hsi_guidance_dose_scale=0.45

该键默认 null，为 null 时 `p_sample` 走 released 算术；两个 cap-off 对照臂已逐位复现封存 cell。
**不跑 n=60 smoke**（理由：第七次记录 §F，n=60 上 `pen_ratio` 半宽 0.00653 与总体效应 0.00622
同量级，判不了唯一有疑问的门槛，只会白烧 42 个 episode）。**不实现候选 2 的状态相关控制器，
不修改 C，不训练模型，不顺带修复 §G3 记录的 batch-size compensation。**
timing 按设计为 null（分片运行自动作废）。另一位用户占用 4 张卡，按用户决定仍用 8 卡启动并接受并行。

### D. 启动前写定的预测，两个模型都给，运行后逐条打分

| 门槛 | LINEAR | POWER | 冻结上限 / 下限 |
|---|---|---|---|
| G1 有 >5g 帧的 episode | 4/355 | 4/355 | ≤8 |
| G1 >5g 帧总数 | 22.5 | 22.5 | ≤38 |
| G1 walk `h_min`<0.6 | 0/126 | 0/126 | ≤2 |
| G2a `boundary_jerk` | 152.765（1.814×GT） | 154.641（1.836×） | ≤168.434 |
| G2b vs C+guided 155.101 | **PASS**（余量 1.5%） | **PASS**（余量 0.3%） | 必须更低 |
| `interior_jerk` | 71.920（1.025×GT） | 71.966（1.026×） | — |
| `jerk_ratio` | 2.1205 | 2.1452 | 必须低于 2.2105 |
| **G3 `pen_ratio` 保留率** | **61.8%** | **75.9%** | **≥50%** |
| G3 `pen_ratio` delta / CI 上界 | +0.00365 / +0.00496 | +0.00230 / +0.00291 | 边界 0.00478（仅报告） |
| **G3 `pene_pct_scene` 保留率** | **50.4%** | **60.4%** | **≥50%** |
| G3 `pene` delta / CI 上界 | +0.00386 / +0.00535 | +0.00308 / +0.00426 | 边界 0.00389（仅报告） |
| G3 `contact_count` | 868.57（+21.10） | 870.54（+23.06） | 不得显著升高 |
| G3 `goal_planar_err_m` | 0.06587（−0.01622，改善） | 0.06832（−0.01377） | 不得显著升高 |
| G3 `success_last_10cm` | 0.99118（+0.00809，改善） | 同 | 不得显著降低 |
| G4 四项幅度下限 | **全过**（0.2323 / 2.0045 / 1.2464 / 1.3003） | 全过 | 0.2118 / 1.8631 / 1.1547 / 1.1345 |
| G4b 反低通 | **过**（jerk 比 0.830 对幅度比 0.949–1.002） | 过（0.840 对 0.967–1.000） | jerk 必须降得更快 |
| G5 `pen_ratio` / `pene` vs B+unguided | **SIG / SIG** | **SIG / SIG** | 必须显著更好 |

**最薄的两处余量已经写明**：G2b 在 POWER 下只有 0.3%（154.641 对 155.101），
`pene_pct_scene` 保留率在 LINEAR 下只有 0.4 个百分点（50.4% 对 50%）。这两项任一失败都在
预期之内，且按用户指令**立即停止汇报，不追加其他 scale，不自动启动状态相关方案**。

**方法学说明，避免日后误读**：`card.py` 输出的 G5 行是把已测的 `dose1 − B+unguided` 向量
按比例缩放得到的，因此继承了本日第七次记录 §E 的不变性伪影（缩放不能改变显著性判决），
**不可作为预测**。上表的 G5 取自 `newg3.py`，它对投影后的 per-episode arm 直接与 B+unguided
做配对 bootstrap，不是标量倍数关系，判决有意义。

## 2026-08-24（同日第二次）：候选 1（scale 0.45）全量运行结果 —— FAIL，且失败方向与预测相反

`p1-hsi-b-v2-eval-epoch222-guided-dose045-shard8-s42-20260824`。8/8 shard `fail=0`，merge exit 0，
**5 h 38 m**（00:10:45–05:48:36），22 条 merge 锚全过（分片切分、key 集合、checkpoint sha
`5daaf813`、`excluded_as_warmup` 5、375 个 motion 导出），门槛与预测在 manifest 创建之前
冻结于 commit `f40cbe6`。

**判定：holdout355（主要）FAIL —— G1 FAIL、G2 FAIL、G3 FAIL、G4 FAIL、G5 PASS；
full375 FAIL —— G1 FAIL、G2 FAIL、G3 PASS、G4 PASS、G5 PASS。**

### A. 穿模不是问题，而且从来不是（在这个 scale 上）

| holdout355 | B+guided | **arm 0.45** | B+unguided | 保留率 | delta vs B+guided | 原显著性形式 |
|---|---:|---:|---:|---:|---|---|
| `pen_ratio` | 0.02286 | **0.02292** | 0.03241 | **99.3%** | +0.00006 [−0.00202,+0.00238] **ns** | **也通过** |
| `pene_pct_scene` | 0.04938 | **0.05058** | 0.05716 | **84.6%** | +0.00120 [−0.00106,+0.00373] **ns** | **也通过** |
| `min_dist` | 0.00480 | 0.00490 | 0.00306 | — | +0.00010 ns | 通过 |
| `contact_count` | 847.48 | 855.02 | 885.09 | — | +7.55 ns | 通过 |

G5 双项显著优于 B+unguided（−0.00949 与 −0.00658，都 SIG）。

**两项穿模指标不但远超新的 50% 保留率下界，连 CI 上界形式也过，而且按 §C 原来的显著性形式
（"不得显著升高"）同样通过。**所以 2026-08-24 那次门槛修订**在这个 scale 上是无关的**——
它既没有帮候选 1 通过，也不是失败的原因。这一点必须写清，避免日后误读为"放宽门槛还是没过"。

**同时它证伪了 2026-08-23 第七次记录 §E 的一个结论。**那里论证"任何均匀削弱都过不了显著性形式的
G3，因为 |delta|/半宽在剂量上不变"。该论证的前提是响应对已测向量**仿射**；实测响应强烈非仿射，
在 s=0.45 处穿模 delta 已经基本归零。**该结论作废**，§E 的方法学部分（仿射投影无法翻转显著性
判决）仍然成立，被推翻的是把它当成机制结论的那一步。

### B. jerk 没有修好，这是失败的实质

| holdout355 | 值 | 门槛 | 判定 |
|---|---:|---|---|
| `boundary_jerk` | **174.911 = 2.077× GT** | ≤2.0×GT = 168.434 | **FAIL** |
| vs C+guided | 174.911 对 155.101 | 必须更低 | **FAIL** |
| delta vs B+guided | **−9.153 [−20.801,+3.464] ns** | 必须显著为负 | **FAIL** |
| `jerk_ratio` | **2.2594**（B+guided 2.2105） | 必须下降 | **FAIL（G4c）** |
| 有 >5g 帧的 episode | **9/355** | ≤8 | **FAIL** |
| >5g 帧总数 | 37 | ≤38 | 通过 |
| walk `h_min`<0.6 | **6/126** | ≤2 | **FAIL** |

**delta vs B+guided 不显著**——即 0.45 的剂量在平滑度上与 released 路径**统计上无法区分**。
G4 的四条幅度下限全部有余量通过，而且幅度实际**上升**（`mean_speed` arm/Bg 1.032、
`pelvis_path_horiz` 1.066），所以 G4c 的失败不是低通，是 jerk 没降。
holdout 的 G3 只有一条子句失败：`goal_planar_err_m` +0.00732 [+0.00268,+0.01205] SIG，
在 full375 上该项为 ns。

### C. 现在有四个剂量点了，机制清楚了

| s | `boundary_jerk` | ×GT | `pen_ratio` | 保留率 | `pene_pct_scene` | 保留率 | >5g eps | walk<0.6 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0（B+unguided） | 126.050 | 1.497 | 0.03241 | 0.0% | 0.05716 | 0.0% | 0 | 0 |
| 1/23.8（dose1） | 129.547 | 1.538 | 0.02921 | 33.5% | 0.05610 | 13.6% | 0 | 1 |
| **0.45（本次）** | **174.911** | **2.077** | **0.02292** | **99.3%** | **0.05058** | **84.6%** | **9** | **6** |
| 1（B+guided） | 184.064 | 2.186 | 0.02286 | 100.0% | 0.04938 | 100.0% | 15 | 10 |

**1/23.8 → 0.45 这一段承担了全部 jerk 伤害的 78.2%，只买到 66–71% 的穿模收益。**
两者在剂量轴上几乎同时被买走，所以**常数 scale 无法把它们分开**。

在两个**已测**内部点之间插值（比之前跨 23.8 倍的外推可靠得多，但仍是模型）：

| 门槛 | 边界 |
|---|---|
| G2b `boundary_jerk` < 155.101 | s ≤ 0.272 |
| G2a ≤ 168.434 | s ≤ 0.392 |
| G1 >5g episode ≤ 8 | s ≤ 0.405 |
| **G1 walk `h_min`<0.6 ≤ 2** | **s ≤ 0.124 ← 最紧** |
| G3 pene 保留率 ≥ 50% | s ≥ 0.251 |
| G3 `pen_ratio` 保留率 ≥ 50% | s ≥ 0.144 |

**上界 0.124 对下界 0.251 → 空。**只有在去掉 walk 子句后才有 **[0.251, 0.272]**，宽 0.021。
**绑定约束现在是 walk 的地面穿透子句，不是穿模指标。**

### D. 预测记分卡（诚实记账）

| 条款 | LINEAR 预测 | POWER 预测 | 实测 | 误差方向 |
|---|---:|---:|---:|---|
| `boundary_jerk` | 152.765 | 154.641 | **174.911** | 两者都低估 13–14% |
| >5g episode | 4 | 4 | **9** | 都低估 5 |
| >5g 帧 | 22.5 | 22.5 | **37** | 都低估 15 |
| walk `h_min`<0.6 | 0 | 0 | **6** | 都低估 6 |
| `pen_ratio` 保留率 | 61.8% | 75.9% | **99.3%** | 都低估 23–38 pp |
| `pene` 保留率 | 50.4% | 60.4% | **84.6%** | 都低估 24–34 pp |
| `jerk_ratio` | 2.1205 | 2.1452 | **2.2594** | 都低估，且实际**上升** |

**两个模型在两条轴上都朝同一方向错**：穿模饱和得比模型快得多（好消息），jerk 伤害积累得
比模型快得多（坏消息），净效果是这笔交易比任何模型说的都差。scale 从 0.40 移到 0.45 的推导
本身没有错——错的是它所依赖的两个外推模型。

### E. worst20 故障修复（仅诊断，不参与判定）

`boundary_jerk` 904.709 → **241.419**（dose1 是 160.015），>5g **16 episode / 85 帧 → 2 / 5**
（dose1 是 0 / 0），`pen_ratio` 0.07225 → 0.06795。0.45 连最差集也没有完全修好。

### F. 未做

按用户指令完成即停。**没有尝试其他 scale，没有实施候选 2 的状态相关控制器，没有修改或重训 C，
没有顺带修 batch-size compensation。**`hsi_guidance_dose_scale` 恢复默认 null，
`hsi_guidance_norm_cap` 为 null，`hsi_guidance_alpha_decay` 为 false，continuous-w 继续暂缓。
常数 scale 家族现在有四个实测点，冻结门槛下的可行内部为空。下一步交用户决定。


## 2026-08-24（同日第三次）：walk h_min 门槛的无 GPU 审计 —— 门槛为真，但**限值**不是判别器

用户批准：不删除也不放宽 `walk h_min<0.6` 门槛，先用现有 motion export 审计 s=0.45 那 6 个
失败 walk episode，逐个确认是真实骨盆下沉、还是地面高度／动作语义／指标定义造成的误报。
全部证据在 `.claude/scratch/walkaudit_20260824/`（`AUDIT.md`、`audit.json`、6 个 episode
图 + 1 张汇总图、`verify.py`）与 `.claude/scratch/walkroot_20260824/NOTES.md`。
没有改动任何仓库文件。

**先验证同一性**：`verify.py` 给出 `max |审计 h_min − 已封存 rebuild h_min| = 0.0`，覆盖全部
130 个 walk episode，同时对 arm 与 B guided 两个 cell 成立。所以下面这些就是 G1 当初判定所用
的同一批数字，不是另一次重新推导。

### A. 六个 episode 全部是**真实下沉**（category 1），四条排除依据

| sequence | arm world | arm 离地 | GT world | GT 离地 | 最深穿地（SDF） | 最小值所在 win/off |
|---|---:|---:|---:|---:|---:|---|
| `061-new_loco:009720` | 0.4812 | 0.4756 | 0.8586 | 0.3824 | −0.0701 m | 22/31 |
| `024:001754` | 0.4978 | 0.4933 | 0.9085 | 0.9031 | −0.0413 m | 3/41 |
| `024:001781` | 0.5239 | 0.4319 | 0.8447 | 0.3459 | −0.1522 m* | 4/41 |
| `024:001749` | 0.5242 | 0.5188 | 0.9054 | 0.8997 | −0.0185 m | 3/35 |
| `061:006111` | 0.5413 | 0.5361 | 0.9154 | 0.9106 | −0.0115 m | 3/41 |
| `031:002608` | 0.5522 | 0.5461 | 0.9361 | 0.9300 | −0.0532 m | 3/38 |

\* 该 episode 每个 cell 都是 ≈−0.15（B unguided −0.1518，C guided −0.1581），是场景 artifact。

1. **不是动作语义。** 从标注恢复逐窗动作标签：全部 130 个 walk episode 中**非均匀标签 0 个**，
   这六个每一窗都是 `walk`。
2. **不是下坡／下台阶。** 骨盆下降 0.40–0.47 m，同时**脚只移动 0.033–0.095 m**（GT 0.015–0.045 m），
   最小值处脚在 −0.036…+0.008 m，骨盆下方局部地面在 0.0045–0.0065 m（平地）。脚是站住的。
3. **不是穿地。** 最深 1.2–15.2 cm，而 GT 自己是 1.5–5.3 cm；最深那个每个 cell 都一样深。
   穿地深度不区分下沉者，骨盆高度才区分。
4. **不是指标定义 —— 而且方向与我的猜测相反。** 我怀疑"用世界 Y 而非离地高度"是弱点。实测：

| 度量 | GT min | GT p5 | GT n<0.6 | arm n<0.6 |
|---|---:|---:|---:|---:|
| world-Y `h_min` | 0.8447 | 0.8637 | **0** | 6 |
| 骨盆离局部地面 | 0.1258 | 0.3388 | **17** | 36 |

world-Y 才是稳健的那个：它在真实运动上从不触发，离 GT 最差值还有 0.24 m 余量。所谓"修正"的
离地形式会在**17 个 ground-truth walk episode 上误报**，因为 `floor_max` 高到 0.806 m ——
地面查询打到了床和平台，人走过家具旁边就被读成零间隙。**不要采用离地形式。**

### B. 三个没预期到的发现

1. **与 >5g 完全不相交。** 六个全部 **0 个 >5g 帧**，峰值加速度 0.96–1.54 g，最小高度处
   0.00–0.31 g。骨盆是**平滑**下沉的。这是两个独立缺陷，意味着基于加速度的信号无法预测下沉。
2. **末窗塌陷，且是吸收态。** 逐窗骨盆 Y 均值（arm）：`024:001754` 0.937/0.939/0.936/**0.689**；
   `024:001749` 0.946/0.958/0.933/**0.591**；`031:002608` 0.943/0.952/0.952/**0.717**。
   审计测得 **12/12 次塌陷再也没回到 0.85 m 以上**；41 窗那个在 frame 422（第 9 窗）塌陷，
   之后 1306/1728 帧（43.5 s）一直趴着。最小值离任何 seam 有 11–41 帧，离 history 帧 130+ 帧，
   所以不是 seam 也不是 warm-up artifact。
3. **`goal_height_err_m` 是个错标的指标，而它独立复现了这个下沉。** `pelvis_goal[1]` 在全部
   375 个 episode 上恒等于 0.0，看着像铁证：目标在地面高度，任何 3-D 拉力都会把骨盆拽下去。
   **被 GT 自己否证**：GT `goal_planar_err_m` 恰好等于 0.0000，而 GT `goal_height_err_m` 在
   walk 上是 0.9406。目标是**平面**的，Y 是占位符。于是在 walk episode 上这个指标就是
   **末帧骨盆高度**，**数值越低运动越差**，与直觉读法相反。它给出同样的剂量响应：

| cell | dose s | `goal_height_err_m` n<0.6 | `h_min` n<0.6 |
|---|---:|---:|---:|
| GT | — | **0** | 0 |
| B unguided | 0 | **0** | 0 |
| dose1 | 1/23.8 | 1 | 1 |
| **arm** | **0.45** | **6** | **6** |
| B guided | 1 | 11 | 13 |
| C guided | 1 | 1 | 1 |

（`h_min` 在 B guided 多抓 2 个，因为它对全部帧取最小，而 `goal_height_err_m` 只读末帧。
walk 上 spearman(两者) 在 arm 是 0.55、B guided 是 0.51，而不下沉的 cell 只有 0.09–0.19。）

### C. 两个被否证的机制假设，都记下来

1. **不是 guidance loss 的 free-limb 缺陷。** `apply_hsi_guidance_loss`（`code/guidance_loss.py:96-99`）
   丢弃 `is_penetrating` 并取无 mask 的 `F.mse_loss(...)*20000`，看起来与 HOI 的"几何项给自由手
   记账"同类。**并不是**：`_get_nearest_free_voxel_direct`（`datasets/infbagel.py:781-819`）把
   target 初始化为 `points_flat.clone()`，只在 `penetrating_mask` 处覆写，所以非穿模关节的
   target 就是它自己，损失与梯度**恰好为 0**。mask 在 target 里，不在 reduction 里。
2. **不是"缺了 HOI 的 feet-floor 项"。** `apply_feet_floor_contact_guidance` 把支撑脚高度拉向
   **0.02 m**。这六个 episode 在骨盆最低处脚在 −0.036…+0.008 m —— 蹲姿**本来就满足**它，而
   GT 自己的脚在 −0.011…−0.021 m，所以该项对 GT 和对塌陷几乎一样不满意。**它不判别，加上它
   也修不了这个。** 正确的、更窄的说法是：HSI 的 guidance 目标只有一项
   `mse(joints, nearest_free_voxel)*20000`，里面**完全没有任何关于骨盆高度或姿态的函数**，
   所以"脚站住的蹲姿"对该目标是隐形的。

### D. 最重要的发现是关于**我的限值**，不是关于运动

126 个 holdout walk episode 上的配对精确 McNemar：

| 比较 | 计数 | p |
|---|---|---:|
| arm 0.45 vs **B unguided** | 6 vs 0 | **0.0312 SIG** |
| B guided vs **B unguided** | 10 vs 0 | **0.0020 SIG** |
| B guided vs **C guided** | 10 vs 1 | **0.0039 SIG** |
| **arm 0.45 vs B guided** | **6 vs 10** | **0.3438 ns** |

guidance 确实是成因，且 B 在这一项上显著差于 C。但 **s=0.45 与 released 路径统计上不可区分**。
我的 G1 限值 2 是 released 参考自己就超了 5 倍的限值 —— 在这个限值下，该子句检验的是整条
guided 路径的既有缺陷，而不是候选的任何性质。这与 dose1 那轮 B unguided 也过不了
`pelvis_path_horiz` 幅度下限是同一类标定错误。

这直接改变 2026-08-24 §C 的读数：walk 子句在限值 2 下把 s 压到 ≤0.124，对上穿模保留率的
≥0.251，这才是可行内部为**空**的原因。任何非劣形式下候选都以 p=0.34 通过该子句，绑定约束
退回到 `boundary_jerk` 的 s ≤ 0.272 对 s ≥ 0.251 —— **非空的 [0.251, 0.272]**，宽 0.021。
**这里只做记录，不做动作**：用户指令是暂不删除或放宽该门槛，所以门槛照旧，常数剂量家族保持关闭。

根因**未确定**。已确定：真实、guidance 必要（unguided 130/130 全不下沉）、剂量单调调制计数、
末窗、吸收态；但**具体哪些 episode 下沉近乎随机**（B guided 的 13 与 arm 的 6 只重叠 3，
1.0→0.45 修好 10 个又新造 3 个）。宁可这么说，也不猜。

### E. 一个遗留风险

场景 `031` 非水密，用六射线符号回退，其地面高度是四个场景里最不确定的 —— 不过它在塌陷处
测得的地面 0.0062 m 与三个水密场景一致。

## 2026-08-24（同日第四次）：候选 2 只读探针的启动前预注册 —— 判据、null、特征集与惰性证据全部冻结

用户批准 R0（纯 CPU 重做最新版仪器的只读惰性验证）与 R1（把预注册冻结进本文档并提交），
**并明确 R1 完成即停，不启动 60 集 GPU 探针**。本节在探针产生任何 60 集数据**之前**写定，
此后不依结果调整。判据草稿来自 `.claude/scratch/probecrit_20260824/CRITERIA_DRAFT.md`
（2026-08-24 12:28），本节在冻结时订正了其中两处错误，逐条记在 §B 与 §C，不做静默修改。

探针只读、不产生 motion、不产生选择，因此**不烧 holdout、不分配 run id、不进 registry、
不调用 `tools/experiment.py start`**。provenance 由启动器记录的 HEAD sha 与仪器 sha256 承担。

### A. 最新版仪器的惰性证据（阻塞性判据，已通过）

2026-08-24 12:03 仪器改版（`probe_hook2.py`，npz 记录键从 20 增至 25）后跑的 round-3 验证，
其输出在旧 session 中断时丢失，**只有 round-2 版本的证据留在磁盘上**。R0 已补齐：

- 被验对象：`.claude/scratch/probe_20260824/probe_hook2.py`，
  sha256 `6a4d8daa9660d851dd56f4d9ce623525b3b482469728d6eabf74a169cd200af9`
- 完整输出：`results/lingo_hsi/probe3_smoke_20260824/verification.txt`（179 行，自足）
- 验证脚本改动**仅两行**：`verify.py:14` 加轮次选择器、`:78` 随之切换 npz 目录。
  **断言语义一字未改。**惰性回归证明：以默认档（round 2）运行，逐字节复现磁盘上已有的
  `probe2_smoke_20260824/verification.txt`（diff 为空）。

两个互不依赖的检查器，同一结论：

| 关系 | motion 数组 | 不一致 | 指标比较 | 不一致 |
|---|---:|---:|---:|---:|
| ON vs 封存 B guided | 44 | **0** | 见下表 | **0** |
| OFF vs 封存 B guided | 44 | **0** | 见下表 | **0** |
| ON vs OFF | 44 | **0** | 见下表 | **0** |

即 2 个 episode × 22 数组 = 44，三个关系各 0 不一致；`verify.py` 与独立写的
`independent_check3.py` 分别得到同一结果。**阻塞性判据通过。**

顺带实测（都在 artifact 里）：探针开销 **1.389 MB/窗**（round-2 为 1.006），
ON/OFF 墙钟比 **0.973 与 1.0005**（ON 更快的那一档），即探针成本低于本机 run-to-run 方差，
**不可引用为一个正数开销**。60 集 364 窗的存储估计 **0.51 GB**。

### B. 零梯度 = 无穿模状态（非阻塞；订正草稿的第一处错误）

`verify.py:359` 断言 `grad_norm > 0`，因此 round-2 与 round-3 的 artifact 结尾都是
`VERDICT: FAIL failures=2`，两处 failure 都是 `nonpositive/nonfinite grad_norm`。
**草稿只写了"0 mismatches"，没有提这个 FAIL verdict，这会让后来者误读。**实测结论：

| episode | calls | `grad_norm==0` | `loss_value==0` | 二者同时为零 | grad=0 而 loss>0 | grad>0 而 loss=0 |
|---|---:|---:|---:|---:|---:|---:|
| `045:004307` | 1497 | 212 (14.16%) | 212 | 212 | **0** | **0** |
| `062:006249` | 1996 | 126 (6.31%) | 126 | 126 | **0** | **0** |
| 合计 | 3493 | 338 | 338 | 338 | **0** | **0** |

无负值、无非有限值；非零段 `grad_norm` 落在 1.409–27.64（045）与 0.532–386.7（062）。
即 **`grad_norm == 0` 与 `loss_value == 0` 逐样本恰好等价**，这正是
`_get_nearest_free_voxel_direct` 把 mask 做进 target 的结果（`datasets/infbagel.py:781-819`：
非穿模关节的 target 就是它自己 ⇒ 损失与梯度恰好为 0，已记于本文档 §C-1 与 walk 审计 §C-1）。

**判定：`verify.py:359` 的谓词写错了，正确谓词是 `isfinite(grad_norm) & (grad_norm >= 0)`。
那两条 CHECK_FAIL 是断言的产物，不是仪器缺陷。`blocking_motion_failures=0` 才是阻塞项，
它在两轮都是 0。**本轮**不修改该断言**，以保持 round-2 与 round-3 artifact 逐行可比；
将来若修，必须同时重跑两轮并说明。

这条对特征集有一个直接后果，写进 §J：**F5（`max loss_value`）在无穿模窗口上恒为 0**，
所以 F5 是"本窗是否发生过穿模"的指示量，不是强度量；且按本文档 §G3-2 的记录，
`loss` 因包含 2 帧被 `set_fixed_points` 覆盖的历史帧而**系统性高报不可纠正的穿模**，
控制器若读 `loss` 必须先掩掉那 2 帧。

### C. 指标叶子数量的唯一正确说法（订正草稿的第二处错误）

草稿与 `launch.sh` 写的是"177 metric leaves per sequence"。**177 不是叶子数，是比较次数，
而且出自另一个检查器。**逐项核准（`.claude/scratch/r0_20260824/adjudicate_r0.out`）：

每个 sequence 的 `metrics[sequence_id]` 展平后 **59 个叶子**，构成为：

| 类别 | 个数 | 具体 |
|---|---:|---|
| 数值（int/float，不含 bool） | **49** | 参与逐值比较 |
| `None` | 5 | `goal_orientation_err_rad`、`per_window_wall_seconds`、`rds`、`rds_max`、`sampling_seconds` |
| `bool` | 4 | `excluded_as_warmup`、`is_object`、`non_watertight`、`rds_available` |
| `str` | 1 | `scene_name` |

于是两个数各有出处，**都对**：

- `verify.py` 只比数值叶子：49 × 3 个关系 = **147** `pairwise_numeric_comparisons`／sequence，
  两个 sequence 合计 98 叶子 / **294** 次比较，0 不一致。
- `independent_check3.py` 比全部叶子：59 × 3 = **177** `pairwise_leaf_comparisons`／sequence，
  两个 sequence 合计 **354** 次比较，0 不一致。

**冻结的表述**：本轮惰性证据覆盖 **每 sequence 59 个指标叶子（其中 49 个数值）**，
在 ON/OFF/SEALED 三个关系上比较，**两个 sequence 合计 354 次叶子比较，0 不一致**。
今后不得再把 147 或 177 称作"叶子数"。

同时补一个草稿没有覆盖的洞：`verify.py` 打印的 `exclusions=0` 统计的是**被跳过的数值叶子**，
那 10 个非数值叶子是**按构造完全不参与比较**的。R0 已单独把它们比过
（`.claude/scratch/r0_20260824/nonnumeric_leaves.out`）：10 叶 × 2 sequence × 3 关系 = 60 次比较，
**0 不一致**，其中 `sampling_seconds` 与 `per_window_wall_seconds` 在三条臂上都是 `None`
（分片运行 timing 按设计作废），`scene_name` 分别为 `045`/`062`。所以"59 of 59 全等"是实测，
不是推断。

### D. 分析单元与富集率

| 单元 | n | 阳性（>5g） | 召回 0.6 处的 95% 半宽 |
|---|---:|---:|---:|
| **窗口（主单元）** | **364** | **38** | **±0.156** |
| episode | 60 | 16 | ±0.240 |
| 每 episode 首个阳性窗 | 60 | 16 | ±0.240 |
| 窗口 0（首峰在 0 的 episode） | 60 | 8 | ±0.339 |

窗口是唯一在本集上分辨率足够把有用规则与 null 分开的单元。episode 级与窗口-0 级**明确欠功效**，
落在半宽内的差异报 INCONCLUSIVE，不报 PASS。

**富集不是总体率**：冻结 60 集按分层设计带 26.7% 的 >5g episode（总体 375 上是 8.3%，3.2×），
阳性窗占 10.44%（总体 2.95%）。**只有固定操作点上的 recall/precision 可外推**；
误触发率必须先重加权回总体，才能与 §I 的 P4 穿模预算比较。

### E. 目标

**T1（主）—— >5g 窗口。**episode e 的窗口 w 在封存的 B guided 30 Hz FK 轨迹里至少有一帧
骨盆 |加速度| > 5g。364 窗中 38 个，60 集中 16 个。来源：`windows.json` 的 `over5g` 字段，Bg 格。

**T2（次，不作为门）—— walk 骨盆塌陷。**2026-08-24 walk 审计已判定该子句为真（见同日第三次
§A）。但它在本集上不能当门，三个实测理由：42 个塌陷窗只落在 **4** 个 episode 里（其中两个长 50
和 41 窗，伪重复）；塌陷是**吸收态**（12/12 不回升），所以唯一可预测的事件是**起始窗**，本集
恰好 **4** 个起始，n=4 支撑不了阈值；且塌陷**不含任何 >5g 帧**（峰值 0.96–1.54 g）。
**T2 只在高度特征 F6 上报诊断，绝不在加速度特征上评分。**本集上 4 个塌陷 episode 有 3 个同时带 >5g。

### F. 可用输入 —— "异常之前"的准确定义

在切换步 T* 对 episode e 的窗口 w 触发一次 flag，规则**只准**读：

- 窗口 w 自己 schedule 中 `t_index >= T*` 的探针记录（`t_index` 从 499 递减到 1，**大即早**；
  guidance 块结构性跳过 `t_index 0`），即 w 自己的前 (500 − T*) 步；
- 窗口 0..w−1 的任何东西，包括它们已提交的 motion。

**禁止**：窗口 w 中 `t_index < T*` 的步；任何 w 之后的窗口；窗口 w 的导出 motion；
任何 evaluator 指标；ground truth；该 episode 自己的结局。

### G. 探针无关的 null —— 在探针存在之前就测好

`.claude/scratch/probecrit_20260824/null_model.py`，同 60 集按 episode 留一，目标 T1，
最佳单特征 `pre_acc_max`（上一窗已提交 motion 的加速度）：

| 格 | 阳性数 | recall | precision | FTR |
|---|---:|---:|---:|---:|
| 全部阳性窗 | 38 | **0.605** | 0.397 | **0.107** |
| 每 episode 首个阳性窗 | 16 | **0.188** | 0.079 | 0.107 |
| 仅窗口 0 | 8 | **0.000** | 0.000 | 0.000 |
| episode 级 | 16 | **0.188** | — | — |

（`pre_speed` 全窗召回可达 0.658，但 FTR 0.175；`pre_acc_max` 操作点更好，取它为冻结 null。
首峰召回 0.188 复现了 2026-08-23 在 full375 上测得的 0.226。）

**这次测量改变了门槛设计，记在这里而不是藏起来**：初稿把主门设为"全窗召回 ≥ 0.60"，
而探针无关的 null 自己就是 0.605，直接过。全窗召回被"跟在先前尖峰之后的窗口"主导，
那里上一窗已提交的加速度本来就管用。探针只需要在 null 瞎掉的格里存在。
null 的窗口-0 召回是 **0.000 且是结构性的**：`pre_*` 在窗口 0 上是 NaN，任何基于上一窗的规则
在那里根本无法触发。

### H. 已提交信号参照 —— 不是可达上限

`.claude/scratch/probecrit_20260824/ceiling.py`，同样按 episode 留一重拟阈值：全窗 0.974、
首窗 0.938、窗口 0 为 8/8，FTR 0.043。

**这不是探针的 oracle 上限。**它测在导出的 `global_jpos`（**已提交**的粗轨迹）上，而那正是
控制器要预测、却读不到的量。探针自己的主信号是 `x_start`（模型的**前置 guidance** 预测），
两者不可互换：在 2 个验证 episode 上，末步 `x_start` 骨盆对已提交骨盆的偏差在良性窗上是
**0.0036–0.0187 m**，在两个异常窗上是 **0.0857 与 0.1326 m** —— 恰好在需要它准的地方劣化 15–20 倍。
7 个验证窗上，窗口的 free-frame gap 与其 `grad_norm` 之和的 Pearson **r = 0.9705**
（与 F3 族的 r = 0.9287；F3 与 F4 之间 r = 0.9825）。**订正**：12:28 草稿在此处写的 0.943
在 round-2 与 round-3 的数据上都无法复现，两轮都恰好给出 0.9705 —— 两轮给出同一个值，
本身又是一次逐位一致的旁证。以 0.9705 为准，0.943 作废。

所以该表**只**确立一件事：T1 基本上是已提交粗骨盆轨迹的确定性函数，因此 T1 没有任何内在不可预测性，
失败会是在线信号的失败，而不是目标不可学。它**不**给在线信号定上限。

### I. 冻结的判据 P1–P5

**P1（无退化，明确不是重点）—— 全窗召回 ≥ 0.605 且 FTR ≤ 0.107**，即对齐 null 的操作点。
单独通过 P1 不说明任何事；**没通过则说明探针不如免费信息**。

**P2（主）—— 首个阳性窗召回 ≥ 0.50**（n=16，null 0.188）。两条独立推导给出同一个 bar：
(i) G1 的上限是 holdout355 上 8 个 >5g episode 对 released 路径的 15 个，而一个 episode 只有在它
**第一个**尖峰窗被 flag 时才被救下，`(1−r)×15 ≤ 8` 给 `r ≥ 0.467`；
(ii) 要显著优于 null 需清过 `0.188 + 0.19`（null 自己的半宽）= 0.38。0.50 同时清过两者。
原则上可达：同一粗信号读在**末步**给出召回 0.791，所以 0.50 严格落在 null 与该参照之间。
**欠功效，半宽 ±0.245**：报 CI，落在半宽内的差异记 INCONCLUSIVE，既不算 PASS 也不算 FAIL。

**P3（主，筛选性）—— 窗口 0 召回 ≥ 4/8**，且**只准用窗口 0 自己 `t_index ≥ 250` 的步**。
null 在此结构性为 0.000，所以任何非零都优于 null，bar 由**有用性**而非 null 定。n=8，不是显著性检验。
读法：**0–1/8 杀死该路线；≥4/8 让它活；2–3/8 不确定**。这是决定性的一格：窗口 0 在本集上占首峰的
50%，在 full375 上 58.1%，在 holdout355 上 66.7%，而这正是被否掉的历史键控规则无物可读之处。

**P4 —— 误触发率 ≤ 0.20**，在 326 个非阳性窗上（p=0.2 处半宽 ±0.043，功效充足）。
推导：唯一实测能消除 >5g 的剂量是 1/23.8，它在被施加的窗口上要付 66.5% 的 `pen_ratio` 收益。
触发总体比例 f 大致留下 `1 − 0.665f` 的收益，要求 ≥85% 保留（明显高于 G3 的 50% 地板）
给出 `f ≤ 0.226`。**比较前必须先从富集的 60 集重加权回总体。**

**P5 —— 前置时间。**`T* = 250` 是判定点：在此触发仍留下 500 步中的后 250 步可用降剂量。
推导自唯一一个"消除全部 >5g 帧同时保留 0.600× 总施加剂量"的臂 —— `alpha_decay`，
其 `(1 − alpha_cumprod)` 因子恰好压制**晚期低噪声步**。T*=250 给控制器的晚期压制范围严格大于该臂用到的。
报告 `T* ∈ {450,400,350,300,250,200,150,100,50}` 上完整的 recall/FTR 曲线，
以及 P2 与 P4 同时成立的最大 T*。

### J. 冻结的特征集（closed set，数据出现之前声明）

采样器速率 10 Hz，dt = 0.1 s，故加速度用 `/0.01`。**不得按 `interp_scale` 重缩放**：
细/粗峰值比在 p05/p50/p95 上确实是 1.561/2.428/6.830，因为 `linspace(0, T-1, T*scale)` 会错过粗节点。

| id | 特征 | 类别 |
|---|---|---|
| **F1** | `x_start` 上 `max_t peak_f accel(pelvis_xyz[t])` | 运动学 —— **唯一有效的那个** |
| F3 | `max_t ‖xnext_pelvis[t] − xprev_pelvis[t]‖` | **潜变量**剂量步长，**不是世界位移** |
| F4 | `sum_t grad_norm[t]` | 累计施加 guidance 幅度 |
| F5 | `max_t loss_value[t]` | guidance 损失；无穿模时恒为 0（见 §B） |
| F7 | `max_t ‖xprev_pelvis[t]‖` | 潜变量激励幅度 |
| F6 | `min over t, f, j of jointY` | 高度/塌陷信号，**仅 T2** |

**F2 已撤回**（原为 `peak accel(xnext_pelvis)`，即一个潜变量的加速度，不是有意义的量）。
撤回依据是 round-3 的实测：`x` 与 `x_prev` 是噪声级 t 上的**潜变量**，去归一化后并不产生位置。
末步对已提交粗骨盆的距离（自由帧，均值）：`x_start` 0.0036–0.1326 m，而 `xnext_pelvis`
15.7–146.0 m、`x_pelvis` 25.0–255.7 m —— 差 400 至 1700 倍。**`x_start` 是唯一的运动学代理。**

（同一轮的副产物，记录但不作为判据：499 次未归一化增量把潜变量推到去归一化单位下约
**1.1 km 与 3.3 km** 之外，而模型每一步都重新投影出 1.8–2.4 m 的合理 `x_start`；输出之所以正常
是因为 `posterior_mean_coef2` 在 t→0 时收缩 `x` 的贡献。带两个异常窗的 episode 达 3266 对良性的
1085，这是 F7 入选的理由。**n=2 episode，是线索不是已确立的关系。**）

T1 的主特征在**每个交叉验证折内**、**只用训练 episode**、从 {F1, F3, F4, F5, F7} 中选。
预期强共线（7 个验证窗上 r(F3 族, F4) = 0.9825），折内选择直接挑更好的那个即可。
T2 的特征固定为 F6。可选扩展，封顶：至多 3 个特征的 logistic 回归，同样折内选择。**不得更宽。**

### K. 选择与推断

按 **episode** 留一交叉验证，分组以防 50 窗序列泄漏进自己的评估。特征选择与阈值**都在折内重拟**。
所有报告的 recall/precision/FTR 均为 out-of-fold。Bootstrap CI **重采样 episode，绝不重采样窗口**
——两条序列各贡献 50 与 41 窗，按窗重采样会把它们当独立样本从而低估不确定性。
可复现性已验：喂同样数据，`adjudicate.py` 的 CV 逐位复现独立写的 `ceiling.py`（0.974 / 0.938 / 8-of-8）。

### L. 报告项（运行前写定）

T1 在 T* 网格上每点的 recall / precision / FTR / 前置时间，窗口级与 episode 级并列，
CI 按 episode 重采样；P3 的窗口-0 格；对 §G null 的比较；T2 诊断连同其 episode 数与起始数。
另加：探针开销，以及对封存 B guided 导出的逐位一致结果。

### M. 判定规则

PASS 要求在 `T* = 250` 上**同时**满足：P1（在对齐 FTR 上不劣于 null）、P2 ≥ 0.50、
P3 ≥ 4/8、P4 ≤ 0.20。P2 落在其 0.50 半宽内记 INCONCLUSIVE，本身不构成 FAIL；
P3 落在 2–3/8 同样不确定；**P3 落在 0–1/8 无论其他条款如何都是硬 FAIL**，
因为窗口 0 承载多数首峰，在那里无法动作的控制器修不好该指标。

**FAIL：按用户指令停止 guidance 控制路线。**

**PASS：提出最小控制器** —— 普通窗满剂量、被 flag 的窗降剂量 —— 低剂量按 4 点实测曲线重新推导。
**不是 0.40（已失败），也不是 0.45（穿模全保留但 jerk 与 released 路径统计不可区分）。**
实测证据把低剂量放在 **[0.042, 0.124]**。

**上限不可过读。**在 T1 上验证过的控制器**只**处理 >5g 子句。walk 塌陷**完全没有加速度特征**
（峰值 0.96–1.54 g），任何加速度键控的 flag 都抓不到它；它需要在 F6 上另设一个 flag，
而那个 flag 在本集上无法验证（只有 4 个塌陷起始）。所以即使 T1 干净 PASS，walk 子句仍未解决。

### N. 明确不在本次预注册范围内

- **不启动 60 集 GPU 探针**（R2）。本节冻结后即停，等用户批准。
- **不实施任何状态相关控制器**，无论探针结果如何——实施需另行批准。
- **不修改 walk `h_min` 门槛**，限值维持 2，本轮不动。
- **"绝对上限 ≤2 是否合理"作为独立待决事项单独挂账**，见同日第三次 §D 的 McNemar 证据
  （arm 0.45 vs B guided 为 6 vs 10，p=0.3438 ns，即 released 参考自己超该限值 5 倍）。
  **该事项与本节的探针预注册相互独立**：P1–P5 全部定义在 T1（>5g）上，T2 只报诊断不设门，
  因此探针的 PASS/FAIL 判定不依赖 ≤2 的任何取值，反之亦然。两者不得合并讨论。
- **不修改 C、不训练模型、不修 §G3 记录的 batch-size compensation、continuous-w 继续暂缓。**
- `hsi_guidance_dose_scale` / `hsi_guidance_norm_cap` 保持 null，`hsi_guidance_alpha_decay` 保持 false。

### O. 启动 R2 时的既定配置（冻结，供批准后直接执行，不在本轮执行）

启动器 `results/lingo_hsi/probe_smoke60_20260824/launch.sh` 已就位，任务表已生成：
**60 episode / 364 窗**，按窗数配平为 50/44/44/44/44/46/46/46。
选择方式 `shard_count=375 shard_index=<每 episode 槽位>`，槽位取自 `smoke60.json` 的
`shard_index_of_375`；`plan_episode_shards` 在 375 上每片恰好一个 episode，
且逐 episode RNG 无条件为 `seed_everything(42 + canonical_ordinal)`，
所以单 episode 运行能从封存的 8 分片运行里逐位复现该 episode —— 这正是惰性比较得以成立的前提。
成本按 round-3 实测 58.3–67.4 s/窗计约 **6.4 GPU-hour**；8 卡最重一卡 50 窗约 **0.9 h** 墙钟。
启动前需重新确认 GPU 占用情况（2026-08-24 14:22 检查时 0–3 卡被另一位用户占用，4–7 空闲）。

## 2026-08-24（同日第五次）：候选 2 只读探针的 60 集运行结果 —— 冻结门槛判 FAIL，两条 PRIMARY 都过，失败全在误触发率

用户批准 R2。按 14f2e32 冻结的判据执行，运行完成后**未修改任何门槛、未重拟任何 bar**，
判定由冻结的 `adjudicate.py` 直接给出。

### A. 运行记录

| 项 | 值 |
|---|---|
| 启动／完成 | 2026-08-24 16:17:39 / 17:23:44，墙钟 **1 h 06 m 05 s** |
| HEAD | `14f2e32`，`worktree_at_launch.txt` 为 0 字节（干净） |
| checkpoint | `hsi_b_lingo_full_v2_epoch222.pth`，sha 硬校验 `5daaf813ca828788` 通过 |
| 仪器 | `probe_hook2.py` sha256 `6a4d8daa…cd200af9` |
| 覆盖项 | `use_guidance=true seed=42 export_motion=true hsi_progress_fix=true shard_index/shard_count=375` |
| guidance 开关 | 三个都未覆盖 → config 默认 `null / null / false`，released B guided 原样 |
| 结果 | 60/60 job **rc=0**，`failures=0`，60 个 npz 共 482 MB（1.32 MB/窗） |
| 卡 | 0–7 八路，0–3 与另一用户并行；我方每进程约 6.6 GB，0–3 合计 19.2–19.5 GB / 24 GB |

**硬件分配未改变任何冻结内容，这一点是证明出来的而不是声明的。**按窗口数确定性重新装箱后，
8 个任务文件与 12:03 冻结的那份**逐字节相同**（哈希全部一致）；9 项校验全过：60 个 job、
sequence_id 集合等于冻结集、无重复、`canonical_ordinal` 与 `shard_index_of_375` 各 60 个互异、
窗口总数 364 且等于冻结值。清单 `.claude/scratch/r2_20260824/task_manifest.json`
（sha256 `90049efc…6595ee5a`），冻结样本集 `smoke60.json` sha256 `d291e9d3…8cdcb7d3`。
每卡负载（窗口）50/44/44/44/44/46/46/46。

### B. 惰性 —— 从 2 集扩到 60 集，仍然 0 不一致

本轮只有 ON 臂：**封存的 B v2 guided 8 分片导出即对照组**（同 checkpoint、同 seed、同 config，
完全未加载 hook）。`.claude/scratch/r2_20260824/verify60.out`：

| 比较对象 | 数量 | 不一致 |
|---|---:|---:|
| motion 数组（60 × 22） | 1320 | **0** |
| 指标叶子（60 × 59，数值 2940 + 非数值 600） | 3540 | **0** |

每 episode 恒为 22 个数组、59 个叶子，无缺失、无未解析。**探针改变的位数为零，
所以它没有烧掉任何 holdout：被打分的输出本来就是已封存、已评分的那一批。**

`cudnn.deterministic=True` / `benchmark=False`（`test_infbagel_hosi.py:43-45`）且逐 episode
`seed_everything(42 + canonical_ordinal)`，所以与另一用户共卡只影响耗时，不进入数值路径 ——
上表就是这一点的实测确认。

### C. 冻结门槛的判定：**FAIL**

`.claude/scratch/r2_20260824/adjudicate60.out`。样本与冻结值完全吻合：364 窗 / 60 集，
>5g 窗 38、首峰 16、首峰在窗口 0 者 8、塌陷窗 42 在 4 个 episode 里、4 个起始。

在冻结的判定点 **T\* = 250**（折内特征投票 F5:52 / F3:8）：

| 条款 | 实测 | bar | null | 已提交信号参照 | 判定 |
|---|---|---|---|---|---|
| P1 全窗召回 @ 对齐 FTR | 0.737 [0.528, 0.905] @ FTR 0.279 | ≥0.605 且 FTR ≤0.107 | 0.605 @ 0.107 | 0.974 | **FAIL** |
| **P2（主）首峰召回** | **0.625 [0.375, 0.875]**（n=16） | ≥0.50 | 0.188 | 0.938 | **PASS** |
| **P3（主）窗口 0 首峰** | **5/8** | ≥4/8 | 0/8（结构性） | 8/8 | **PASS** |
| P4 误触发率 | 0.279 | ≤0.20 | — | 0.043 | **FAIL** |
| P5 提前量 | P2 与 P4 同时成立的最大 T\* = **450** | — | — | — | 报告项 |

**总判定：FAIL。**P3 是 5/8，不在 0–1/8 的硬 FAIL 带内，所以这是经由 P1 与 P4 的普通 FAIL。

**失败的结构必须说清楚，因为它不是"没有信号"。**两条 PRIMARY 条款都过了，而且过在 null
结构性做不到的地方：窗口 0 上 null 恒为 0/8（`pre_*` 在窗口 0 是 NaN，基于上一窗的规则根本无法触发），
探针拿到 5/8；首峰召回 0.625 对 null 的 0.188。**失败集中在同一个轴上 —— 误触发率。**
P4 直接超线（0.279 对 0.20），而 P1 也是败在它自己的 FTR 上限而非召回：召回 0.737 高于 null 的 0.605，
但操作点的 FTR 是 null 的 2.6 倍，所以它不构成"在 null 自己的操作点上支配 null"。

### D. 冻结文本明确要求、但 `adjudicate.py` 没做的两项补算

`.claude/scratch/r2_20260824/p1p4_forms.out`。**两项都不推翻 §C 的判定，只说明它是哪一种失败。**

**(a) P4 重加权回总体**（§I 原文：「比较前必须先从富集的 60 集重加权回总体」）。
`adjudicate.py` 拿富集集的原始 FTR 直接比 0.20。按总体阳性窗占比 0.0295 重加权，
触发总体比例 `f = recall×0.0295 + FTR×0.9705`：

| T\* | 富集 FTR | 召回 | 精确 | 总体 f | bar f ≤ 0.226 | 隐含穿模保留率 1−0.665f |
|---:|---:|---:|---:|---:|---|---:|
| **250（判定点）** | 0.279 | 0.737 | 0.235 | **0.2926** | **FAIL** | 0.805 |
| 450 | 0.160 | 0.789 | 0.366 | **0.1781** | PASS | 0.882 |

即 **P4 在两种形式下都在判定点失败**。隐含保留率 80.5% 低于 P4 当初设定所依据的 85%，
但高于 G3 的 50% 地板 —— 记录下来，不据此调整任何门槛。

**(b) P1 按原文形式**（「全窗召回 ≥ 0.605 **且 FTR ≤ 0.107**」，即对齐 null 的操作点）。
`adjudicate.py` 把它实现成对折内 CV 自选操作点的合取检验，那是另一个问题：它无法把
「没有信号优势」与「CV 选了一个更高 FTR 的操作点」分开。改为折内在 `FTR ≤ 0.107` 约束下
重拟特征与阈值：

| T\* | 约束后召回 @ FTR | bar | 判定 | 该点上的首峰召回 |
|---:|---|---|---|---|
| 250 | 0.711 @ **0.110** | ≥0.605 @ ≤0.107 | **FAIL** | 10/16 = 0.625（仍过 P2） |
| 450 | 0.632 @ **0.123** | 同 | **FAIL** | 7/16 = **0.438（跌破 P2）** |

两点都 FAIL，且失败都在 FTR 上限而非召回（0.711 与 0.632 都高于 0.605）。
out-of-fold 的 FTR 无法精确控到 0.107，判定点上超出 **0.003**。
**并且 T\*=450 上不存在同时满足 P1 与 P2 的操作点**：把 FTR 压到 0.107 会把首峰召回压到 0.438。

### E. 完整 T\* 曲线（P5 报告项）

| T\* | 全窗召回 | 全窗精确 | FTR | 首峰召回 | 首峰精确 | 窗口 0 | 折内特征 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 450 | 0.789 | 0.366 | 0.160 | 0.625 | 0.161 | 5/8 | F4:58 F3:2 |
| 400 | 0.789 | 0.316 | 0.199 | 0.688 | 0.145 | 6/8 | F3:60 |
| 350 | 0.737 | 0.292 | 0.209 | 0.625 | 0.128 | 5/8 | F3:58 F5:2 |
| 300 | 0.737 | 0.235 | 0.279 | 0.625 | 0.099 | 5/8 | F5:52 F3:8 |
| **250** | **0.737** | **0.235** | **0.279** | **0.625** | **0.099** | **5/8** | **F5:52 F3:8** |
| 200 | 0.921 | 0.292 | 0.261 | 0.875 | 0.141 | 7/8 | F5:60 |
| 150 | 0.921 | 0.289 | 0.264 | 0.875 | 0.140 | 7/8 | F5:60 |
| 100 | 0.921 | 0.285 | 0.270 | 0.875 | 0.137 | 7/8 | F5:60 |
| 50 | 0.868 | 0.270 | 0.273 | 0.812 | 0.127 | 6/8 | F5:59 F3:1 |

两点观察，都只作记录：召回在 T\* 上**非单调**（200 以后跳到 0.921 / 7-of-8），因为晚期步
读到更多信息；而折内选出的特征随 T\* 从 F4（累计 grad_norm）经 F3（潜变量剂量步长）
换到 F5（`max loss_value`）—— 早期是剂量幅度在说话，晚期是穿模损失在说话。
**精确度在所有 T\* 上都低（首峰精确 0.099–0.161）**，这与 P4 的失败是同一件事。

### F. T2 诊断（不是门）

42 个塌陷窗在 4 个 episode 里，4 个起始；**F6（`min jointY`）抓到 4/4 个起始**。
按 §E 冻结的读法，**n=4 支撑不了任何阈值**，这只是一个提示：高度特征在塌陷上不是盲的，
而加速度特征对它完全无能（峰值 0.96–1.54 g）。不据此提出任何东西。

### G. 冻结规则的后果，以及本轮**没有**做的事

§M 原文：**「FAIL：按用户指令停止 guidance 控制路线。」**本轮据此停止。

按用户指令完成即停：**没有实现状态相关控制器**（无论 P2/P3 通过与否）、**没有启动任何新的
scale 实验**、**没有修改 walk `h_min` 门槛**、**没有修改或重训 C**、**没有重拟任何 bar 或
判定点**、continuous-w 继续暂缓。`hsi_guidance_dose_scale` / `hsi_guidance_norm_cap` 仍为 null，
`hsi_guidance_alpha_decay` 仍为 false。「绝对上限 ≤2 是否合理」仍是与本轮无关的独立待决事项。

探针不是 registry run，没有分配 run id，没有调用 `tools/experiment.py start`，
没有烧任何 holdout（§B 的 0 不一致就是这一点的证明）。下一步交用户决定。

## 2026-08-24（同日第六次）：阶段收口 —— guidance-control 路线正式关闭，共同 16 帧 rollout 边界地板独立立项

本节的数字不是从本文档既有散文转抄的。主 session 于本日直接从封存产物重测，
并把结果与引用陷阱固化在 `.claude/scratch/r3_20260824/MEASURED_FACTS.md`；归因计划固化在
`.claude/scratch/r3_20260824/PLAN_DESIGN.md`。下面的表格与标量均以这两份记录为准。

**阶段判定分成两件事：guidance-induced spike 已关闭；共同的 16 帧 rollout
边界地板仍未解决，但它不再归入 guidance-control 路线。**

### A. 匹配分析集，以及八格的完整基表

所有跨格均值都在**同一批 episode**上计算：375 个共有 episode 减去各格
`excluded_as_warmup` 的并集 5 个，得 **n=370**。GT 自己的 metrics 文件标记的排除数是
0；如果不做匹配，就会拿 370 去比 375。这一步不是格式整理，而是跨格判定成立的前提。

| cell | artifact dir | `boundary_jerk` | `pen_ratio` | `pene_pct_scene` | `fs_nemf` | `goal_planar_err_m` | `min_dist` | `contact_count` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GT v3 | `ground-truth-v3` | 84.50508 | 0.02796 | 0.05050 | 0.26102 | 0.00000 | 0.00000 | 789.89801 |
| B unguided | `b_v2_unguided_shard8` | 127.92255 | 0.03545 | 0.06095 | 0.32479 | 0.05187 | 0.00305 | 946.21615 |
| B guided | `b_v2_guided_shard8` | 222.75034 | 0.02569 | 0.05247 | 0.31717 | 0.08473 | 0.00484 | 895.87580 |
| B guided dose1 (`s=1/23.8`) | `b_v2_guided_dose1_shard8` | 131.17277 | 0.03174 | 0.05928 | 0.32222 | 0.05521 | 0.00345 | 937.59211 |
| B guided `s=0.45` | `b_v2_guided_dose045_shard8` | 178.32869 | 0.02536 | 0.05362 | 0.32333 | 0.08874 | 0.00485 | 907.05669 |
| C unguided | `c_v2_unguided` | 128.96413 | 0.03389 | 0.06024 | 0.30289 | 0.05402 | 0.00384 | 930.18216 |
| C guided v4（progressfix OFF） | `c_guided_v4` | 135.43425 | 0.02183 | 0.04957 | 0.26061 | 0.04693 | 0.00444 | 816.12541 |
| C guided v5（progressfix ON） | `c_guided_v5_baseline` | 159.43782 | 0.02529 | 0.05332 | 0.29298 | 0.06069 | 0.00396 | 879.33961 |

`boundary_jerk` 相对 GT 依次为 1.514 / 2.636 / 1.552 / 2.110 / 1.526 / 1.603 / 1.887；
`pen_ratio` 相对 GT 依次为 1.268 / 0.919 / 1.135 / 0.907 / 1.212 / 0.781 / 0.904；
`fs_nemf` 相对 GT 依次为 1.244 / 1.215 / 1.234 / 1.239 / 1.160 / 0.998 / 1.122。

### B. guidance-control 证据链：从原路径到探针的每一个出口都已走完

异常尖峰的硬计数来自 `windows.json`，口径是 full375：

| cell | episodes | windows | `>5g` frames | `>5g` windows | `>5g` episodes |
|---|---:|---:|---:|---:|---:|
| B guided | 375 | 2271 | 132 | 67 (2.95%) | 31 (8.3%) |
| B guided dose1 | 375 | 2271 | **0** | **0** | **0** |
| B unguided | 375 | 2271 | **0** | **0** | **0** |
| C guided v5 | 375 | 2271 | 41 | 21 (0.92%) | 10 (2.7%) |
| GT | 375 | 2271 | **0** | **0** | **0** |
| **C unguided** | 375 | 2271 | **0** | **0** (0.00%) | **0** (0.0%) |
| C guided v4（progressfix OFF） | 375 | 2271 | 44 | 25 (1.10%) | 12 (3.2%) |
| B guided s=0.45 | 375 | 2271 | 42 | 19 (0.84%) | 11 (2.9%) |

**表中后三行是本节新测的，并且要说明它们为什么原先不在。**`windows.json` 只覆盖
Bg / d1 / Bu / Cg(v5) / GT 五格，**C unguided 从未被测过 `>5g`**；本节的初稿曾在没有测量的
情况下断言它是 0。该断言现已被实测取代，而不是被沿用。

测法与它为什么是零 GPU：`>5g` 判据作用在**骨盆**加速度上，而 SMPL-X 的骨盆关节等于
`transl` 加一个与姿态无关的静止位姿偏移，常量在二阶差分中消掉，所以 |accel| 可以只用
`transl` 以 numpy 算出——不需要 FK、不需要 torch、不需要 GPU。窗口边界与
`dt = 1/(fps*interp_scale) = 1/30` 逐字复用 `windows.py`。

**先验证后使用**：该捷径在 `windows.json` 覆盖的全部四格上逐项精确复现——Bg 132/67/31、
d1 0/0/0、Bu 0/0/0、Cg 41/21/10，帧数、窗口数、episode 数与 `nw=2271` 全部一致。只有在
复现精确之后，新的三格才被报告。证据在 `.claude/scratch/r3_20260824/cu_over5g.out`。

于是尖峰列现在是**完整实测**而非部分假定：B guided 132、C guided v5 41、C guided v4 44、
s=0.45 42，而 **B unguided、C unguided、dose1、GT 全部恰好为 0**。两个 unguided 格与 GT
都不含任何 `>5g` 帧；凡是含有 `>5g` 帧的格，guidance 都是开着的。

norm-cap 那一轮的三档 tau **全部失败**。这不是结果出来后才补的解释：只读步增量分布
已经表明，B 的问题在于增量被重复施加形成的总剂量，norm cap 却只截单步幅度；最高档
在运行前就能证明为逐位惰性，非平凡档随后也全部未过门。**失败原因在跑之前就可测。**

dose1 把 B 的 guidance 引致 `boundary_jerk` 超出量移除了 **96.6%**，并移除了
**100%** 的 `>5g` 帧（132 → 0）；但它在已冻结门槛上因穿模保留率失败。`s=0.45`
保住了穿模收益，但 jerk 与 released 路径统计上不可区分。两个固定剂量点分别失败在
冻结门槛的两侧，不再追加固定 scale。

最后的 in-loop 只读探针（commit `a574618`）覆盖 60 episode / 364 windows，在 released
B guided 上依冻结规则判定：

| 条款 | `T*=250` 实测 | 对照／bar | 判定 |
|---|---|---|---|
| P1 | 0.737 @ FTR 0.279 | null 0.605 @ 0.107 | **FAIL** |
| P2 | 0.625 [0.375,0.875] | null 0.188 | **PASS** |
| P3 | 5/8 | null 0/8（结构性） | **PASS** |
| P4 | FTR 0.279；重加权后总体 `f=0.2926` | 0.20；`f` bar 0.226 | **FAIL** |
| P5 | P2 与 P4 同时成立的最大 `T*=450` | 报告项 | — |

全窗精确率为 0.235，首峰精确率为 0.099。60 个 episode 的惰性比对覆盖 1320 个
motion arrays 与 3540 个 metric leaves，不一致数都是 0。**探针不是没有主信号；它按
预注册规则因误触发率失败，总判定为 FAIL。**

### C. 从此保留的 baseline/default 与引用口径

| 对象 | 保留口径 |
|---|---|
| guidance defaults | `hsi_guidance_dose_scale: null`；`hsi_guidance_norm_cap: null`；`hsi_guidance_alpha_decay: false` |
| B 侧参考 | `b_v2_unguided_shard8` 与 `b_v2_guided_shard8` |
| C 侧 gate | `c_guided_v5_baseline`（progressfix ON） |
| C 侧 fix-OFF 对照 | `c_guided_v4`，**只作对照，不作 gate** |
| GT 参考 | **只用** `ground-truth-v3` |

**三个 guidance 开关保持默认关闭。**

### D. 从当前决策中退役的八条结论

1. **修复前的 model-side geometry 全部作废，且不可重算。**表示修复是 `3ded4eb` +
   `a4c979c`（`2026-08-18`）。修复前的 `b_guided_shard8`、`b_unguided_shard8`、
   `c_guided{,_v2,_v3}`、`c_unguided`、`latency_*`、`smoke-*`、`rds_gate_smoke`、
   `shard_bitwise` 中的模型侧几何数字都无效。那些 checkpoint 拟合的 132-dim 通道已不存在，
   只有重训能替代；数据资产未被修改，输入 hash 不变。
2. **显著引用陷阱：`results/lingo_hsi/c_unguided` 与
   `results/lingo_hsi/c_v2_unguided` 同时在磁盘上。**前者是修复前退役行，后者才是有效的
   修复后行。本节表格全部使用 `c_v2_unguided`；误引 `c_unguided` 会无声地生成作废数字。
3. **`2026-08-16` 的两条 B registry 行对当前决策作废。**
   `p1-hsi-b-eval-epoch222-shard8-s42-20260816`（guided，4.99 h）与
   `...-unguided-shard8-s42-20260816`（unguided，2.68 h）都在 `101ac84` 上评估，早于表示修复。
4. **修复前的蒸馏 `2×2` factorial 与它的 seam 结论作废。**修复后
   `boundary_jerk` 由 6831 变为 134.9，推翻幅度约 `50×`；旧主效应
   `boundary_jerk -965`（student better）与 `fs_nemf +0.178`（worse）都不得再引用。
   **`fs_nemf +0.178` 在修复后符号翻转。**匹配 `n=370` 上，unguided 为
   `0.32479 → 0.30289`（C better by `0.0219`），guided 为 `0.31717 → 0.29298`
   （C better by `0.0242`）；即修复后蒸馏使 `fs_nemf` 改善 `0.022–0.024`。
   **这些只是均值，未做显著性检验。**对应的 project memory 已据此更正。不得把这个退役
   factorial 与有效的修复后 `2026-08-23` guidance-vs-sampler `2×2` 混为一件事。
5. **显著引用陷阱：C+guided 的 `1.60×` 是 progressfix-OFF 行，不是 gate 行。**
   `c_guided_v4` 本日实测为 `1.603×`；progressfix ON 的 gate 行是
   `c_guided_v5_baseline`，本日实测为 **`1.887×`**。不得把 `1.60×` 引为 gate reading。
6. **被取代的标量不得再引。**seam/spike correlation 是 **0.9705**，不是 0.943；
   探针的 per-window overhead 不得引为一个正数；`2026-08-24` 的
   “`s=0.45` 上 penetration delta essentially zero”结论作废，但“affine projection 不能翻转显著性
   判定”的方法学部分仍成立。
7. **三个旧 GT 标量是 artifact。**其中包括 `min_dist ~1 mm`、`fs_nemf 0.568`
   与另一项旧标量；`ground-truth-v3` 是唯一有效的 GT 参考行。它的 `quaternion_slerp`
   weight bug 按设计仍未修；修它会使 `ground-truth-v1/v2/v3` 全部作废。
8. **一项旧的插值问题仍然开放，且与本次收口直接相关。**`interpolate_joints` 使用
   `linspace(0,T-1,T*scale)`，`interp_jrot` 却按 `1/scale` 步进；在 `interp_scale=3` 时，
   pelvis 单独就漂移 4.86 mm mean / 43.3 mm max。它直接位于 fine-rate `boundary_jerk`
   上游，是下面零 GPU 计划的首要嫌疑项。

### E. 已关闭的问题：guidance-induced spike

B guided 的 132 个 `>5g` 帧与 C guided 的 41 个 `>5g` 帧，在各自 unguided、GT
和 dose1 上都消失。B 的 guidance-induced `boundary_jerk` 超出量是 **`+1.122× GT`**
（相对自身 floor 为 `+74.1%`），C 是 **`+0.361× GT`**（相对自身 floor 为
`+23.6%`）。这是 guidance 额外叠加的组件，不是下一节的共同地板。

dose1 消除了 B 的 96.6% guidance 引致 `boundary_jerk` 超出量和 100% `>5g`
帧，但在冻结门槛上因穿模保留率失败；`s=0.45` 保住穿模，但 jerk 与 released
路径不可区分；norm-cap 三档全败；in-loop 探针按预注册规则判 FAIL。本文档同日第四次
§M 的冻结原文是：**「FAIL：按用户指令停止 guidance 控制路线。」**

**现在执行该规则：guidance-control 路线正式关闭。**不因探针的 P2/P3 通过而改写总判定，
也不从已失败的固定剂量、norm-cap 或探针中再抽出一个新 guidance 方案。

### F. 仍未解决的问题：共同的 16 帧 rollout 边界地板

这是本节最重要的新实测发现。`boundary_jerk_samples` 每 episode 为 15.2，
`interior_jerk_samples` 每 episode 为 242.5，所有 cell 完全相同，因为 seam 相同。

| cell | `boundary_jerk` | `interior_jerk` | `jerk_ratio` | boundary ×GT | interior ×GT | ratio ×GT |
|---|---:|---:|---:|---:|---:|---:|
| GT v3 | 84.505 | 70.440 | 1.1943 | 1.000 | 1.000 | 1.000 |
| B unguided | 127.923 | 63.956 | 2.0205 | 1.514 | **0.908** | 1.692 |
| B guided | 222.750 | 92.807 | 2.2829 | 2.636 | 1.318 | 1.912 |
| B g dose1 | 131.173 | 64.754 | 2.0565 | 1.552 | 0.919 | 1.722 |
| B g `s=0.45` | 178.329 | 78.422 | 2.2478 | 2.110 | 1.113 | 1.882 |
| C unguided | 128.964 | 63.088 | 2.0435 | 1.526 | **0.896** | 1.711 |
| C guided v4 | 135.434 | 63.070 | 2.0252 | 1.603 | 0.895 | 1.696 |
| C guided v5 | 159.438 | 68.834 | 2.1715 | 1.887 | 0.977 | 1.818 |

**两个 unguided cell 的内部 jerk 比 GT 还低**：B 为 `0.908×`，C 为 `0.896×`；
它们只在 seam 上高，分别为 `1.514×` 与 `1.526×`，两者只差 **0.8%**。因此这不是整体
平滑度不足，而是定位在窗口边界的缺陷；它与专家身份、蒸馏与 guidance 都无关。
这个共同 floor 占 B guided 相对 GT 全部 `boundary_jerk` 超出量的 **31.4%**。

同时必须记下一个限制：GT 本身的 `jerk_ratio` 是 **1.1943**。GT 是连续运动，却也在
seam 索引处被抬高。这为“模型超出量中有多少是真实 rollout 缺陷”设了上界，也是下面 C1
必须第一个做的原因。

`boundary_jerk` 的精确定义是三阶差分
`(p[3:] - 3p[2:-1] + 3p[1:-2] - p[:-3]) / dt^3`，对 xyz 取 L2，再对关节取均值，
得到 `[T-3]` stencils。stencil `t0` 跨 `[t0,t0+3]`，seam `s` 是 `s-1` 与 `s`
之间的切口；边界 stencil 满足 `seam-3 <= t0 <= seam-1`，每个 seam 有 3 个。该指标要求
`StitchedSequence`，并在 coarse-to-fine 插值之后的 **FINE rate** 上计算。

### G. 为什么新归因可以真正做到零 GPU、且不需要 torch

模型 cell 的 motion npz 已经包含运动学结果，不需要再把姿态送回模型：

| 字段 | 已验证的 schema | 用途 |
|---|---|---|
| `global_jpos` | `float32 [44,28,3]` | coarse 10 Hz joints，**已做 FK** |
| `transl` | `float32 [132,3]` | fine 30 Hz 原始 SMPL-X 平移参数 |
| `global_orient` | `float32 [132,3]` | fine 30 Hz 原始 SMPL-X 全局方向参数 |
| `body_pose` | `[132,21,3]` | fine 30 Hz SMPL-X 姿态参数 |
| stitch 结构 | `seams [16,30]`，`window_lengths [16,14,14]` | `history_frames=2`，`interp_scale=3`，`fps=10.0` |

这一 schema 共 22 个 key；表中形状以 `045:004307` 为已验证例。在 B unguided、C unguided、
C guided 与 B guided 之间，`betas` 逐位相同（`max|diff|=0`），窗口结构也相同。

GT 没有 motion export：`results/lingo_hsi/ground-truth-v3/` 恰好只有一个
`evaluation/per_sequence_metrics.json`。但 GT 轨迹不需要 GPU 重建：

| raw array | dtype | shape |
|---|---|---|
| `data/dataset/transl_aligned.npy` | `float64` | `(2915752,3)` |
| `data/dataset/human_joints_aligned.npy` | `float64` | `(2915752,28,3)` |
| `data/dataset/human_orient.npy` | `float64` | `(2915752,3)` |
| `data/dataset/start_idx.npy` / `end_idx.npy` | `int32` | `(19450,)` |

`human_joints_aligned.npy` 已是与 `global_jpos` 相同的 28 关节布局，GT arm **不需要
SMPL-X forward pass**。episode 到 raw index 的重建也只是整数 numpy：`WINDOW_FRAMES=16`、
`HISTORY_FRAMES=2`、`DATA_STEP=3`、`WINDOW_STRIDE_RAW=(16-2)*3=42`，即
`raw = start + window_index*42 + arange(16)*3`，并 clamp 到 `end-1`。

**这里主动订正我先前的说法。**既有离线 harness
`.claude/scratch/walkaudit_20260824/common.py` 设的是 `DEVICE=cuda:0`，并经
`ev.ground_truth_motion(...)` 取 GT，因此当时的 walk 审计是“没有新采样运行”，
不是字面意义的零 GPU。新计划不继承这一点；直接 mmap `human_joints_aligned.npy`，
使整个归因真正做到零 GPU 且 torch-free。

### H. 零 GPU 归因计划：C1–C5，按最早杀死无效工作的顺序

**C1 —— 地板是否由插值造成。这项第一个做。**`boundary_jerk` 在 fine rate、
coarse-to-fine 插值之后计算；已知 `interpolate_joints` 与 `interp_jrot` 的取点不一致，
在 `interp_scale=3` 时 pelvis 漂移 4.86 mm mean / 43.3 mm max。GT 的 `jerk_ratio=1.1943`
说明连续运动也会在 seam 索引处被该路径抬高。在 GT、B unguided、C unguided 和 C guided 上，
用 10 Hz coarse `global_jpos` 重算同一 metric stencil，移除插值。**deliverable：报告去掉插值后
仍然存活的 boundary excess 比例。**如果大部分不存活，目标就是 evaluator/exporter 缺陷，
而不是 rollout 缺陷。

**C2 —— root translation 与 local pose 拆分。**在同一 stencil 上按三种口径重算：
root-only（`global_jpos[:,0,:]`，并独立用 `transl`）、root-relative joints
（`global_jpos - global_jpos[:,0:1,:]`）和 full。每个 cell 都报 boundary 与 interior jerk。
**deliverable：每个 cell 的 floor root/pose 拆分。**

**C3 —— seam-offset profile。**对每个 stencil，用中心 `t0+1.5` 到最近 seam
`s-0.5` 的 signed distance 分箱，分 cell、coarse/fine 绘图或列表，同 offset 的 GT 作 null。
**deliverable：判断超出是精确集中在 offset 0，还是宽范围抬高。**后者意味着 `boundary_jerk`
被误命名，当前 3-stencil 窗只是抽到了更宽的缺陷。

**C4 —— history-frame coverage 与 state reset。**后一窗重生成 2 个 history frames，
`set_fixed_points` 用已提交值覆写它们。离线先验证每个 seam 上，后一窗声明的历史 coarse frames
与前一窗已提交 frames 相等；如果不等，就是 stitch 缺陷。然后测量每个 seam 后**第一个新生成帧**
的 position / velocity / acceleration 相对 interior 分布是否异常，以及随后帧的衰减。
**deliverable：reset transient 的 per-offset decay curve，以及回到 interior 分布内所需的帧数。**

**C5 —— velocity / acceleration continuity，GT 作 index-matched null。**GT 没有窗口结构，
所以在相同帧索引 `16,30,44,...` 上评估 GT，可以隔离这些索引本身是否特殊。每个 cell 与 GT 都报
seam-crossing 相对 interior 的 `|dv|` 与 `|da|` 分布，coarse/fine 并列，CI 按 episode 重采样。
**deliverable：分开 velocity excess 与 acceleration excess，判定三阶差分是被真实位置不连续的
velocity step 驱动，还是被曲率不匹配的 acceleration step 驱动。**

### I. 横切要求，以及计划自己的停止门

- **首要自检门：**逐字复制 `code/priors/hsi/metrics.py:955-1015` 的 evaluator stencil，
  从 export 在 fine rate 重算 `boundary_jerk`，必须逐 episode 匹配封存
  `per_sequence_metrics.json` 值。**任何 episode 不匹配，立即停止并报告；在此之前不信任任何拆分。**
- 分析集固定为匹配 `n=370`，即 375 个共有 episode 减去 5 个 `excluded_as_warmup` 并集，
  与本节收口表格完全相同。
- C unguided 只用 `c_v2_unguided`，不得读退役的 `c_unguided`；C guided 用
  `c_guided_v5_baseline`（progressfix ON）作主行，`c_guided_v4` 只报 fix-OFF 对照。
- 所有 CI 都重采样 **episode**，不重采样 window 或 stencil。
- 所有 intermediate 只写入 `.claude/scratch/`；每个报告数字都必须有保存的 artifact 支撑，
  不能由散文承担 provenance。

### J. 预计成本与边界

整个计划是对磁盘现有数据的 pure-numpy 扫描。每 episode 负载是
`global_jpos [<=~800,28,3]` 与 `transl [<=~2400,3]`；GT 是从两个 mmap array 中 gather，
375 个 episode 合计约 `~100k` raw frames。计划成本为：

| 范围 | GPU | CPU／内存 | 预计墙钟 |
|---|---:|---|---:|
| 4 cells + GT × 370 episodes，C1–C5 含重建门 | **0 GPU-hour** | single CPU process，well under 1 GB resident（mmap） | **~10–20 min** |

成本足够低，不需要 staging 或 sharding。**但它产出的是 attribution，不是 fix。**
明确范围外包括：新 sampling、training、controller、guidance-knob change、walk-gate change、
C modification 与 continuous-w work；任何 fix 都是归因之后的另一个决策。

### K. 本节没有做的事

- **未实施状态相关控制器**；也未用本轮 60 样本调整 `T*`、特征、分类器或误报门槛。
- 未追加任何固定 scale，未回改任何既有实验结论，也未用新分解复活已失败方案。
- `hsi_guidance_dose_scale: null`、`hsi_guidance_norm_cap: null`、
  `hsi_guidance_alpha_decay: false`；**三个 guidance 开关保持默认关闭。**
- 未修改或重训 C；continuous-w 继续暂缓。
- 未删除或放宽 walk `h_min<0.6` 绝对门槛，限值仍为 2。
- **「绝对上限 `<=2` 是否适合作为 baseline 迭代门槛」登记为独立待决事项。**
  它与 rollout continuity 归因和已关闭的 guidance-control 路线分开；**不得用它复活任何
  已经失败的 guidance 方案。**

**本节只提交分析计划与预计成本，未启动任何新 GPU 实验，也未执行 C1–C5。**
等用户决定是否开启独立的 rollout continuity 研究线。

### L. 归因问题的范围补记

新计划只读磁盘上已有的 B unguided、C unguided、C guided 与 GT，不引入新采样。
既有 `walkaudit_20260824/common.py` 只复用 cell globs 与 `window_bounds` / `seam_frames`
helpers，不复用它的 GT arm。

C2 不重复旧问题：`2026-08-23` 的 seam 归因已测得 B-vs-C **guided** 超出量有 69%
来自 root translation；C2 要回答的是不同问题，即**共同 unguided floor 究竟由 root 还是
local pose 承载**。旧结果不能替代该 deliverable。

**本节仍然只提交分析计划与预计成本，未启动任何新 GPU 实验，也未执行 C1–C5。**
等用户决定是否开启独立的 rollout continuity 研究线。

## 2026-08-24（同日第七次）

### A. 本轮授权范围与结论提要

**C1 的注册假设——地板主要是插值／evaluator artifact——被 FALSIFIED。**

本轮只执行了计划中的**首要自检门**与 C1；C2–C5 未执行。全程 zero GPU、zero
torch、pure numpy，只读已有 export 与已有数据，不启动采样、训练或评测工作负载。
分析集仍是 closure 使用的 matched `n=370`，但本节所有数字都直接来自
`.claude/scratch/c1_20260824/MEASURED_FACTS_C1.md` 及其列出的脚本产物。

C1 的结果不是“插值问题被证实”或“插值问题暂时无法判断”：去掉插值后，三个模型格的
boundary excess 在每一种定义下都保留并增大；GT 自己的 `1.1943` 也不是由插值抬高的。
唯一找到的真实 evaluator artifact 是 GT 末端被 clamp 后重复的冻结帧，它反而抬高了
GT null 的 denominator，使模型对 GT 的 excess 被低估。

### B. 无 torch 的 FK，以及它为什么是精确的

`SMPLX_JOINTS_28 = [0..21, 23, 24, 25, 28, 40, 43]` -- all < 55，都是
kinematic-tree joints，从不是 smplx 在 54 之后追加的 vertex-derived landmarks。
smplx 通过
`batch_rigid_transform(rot_mats, J_regressor @ v_shaped, parents)` 计算这些 55 个
关节，然后加上 `transl`；pose blendshapes 不会触碰 `J`。六个非 body slot 是 head /
left_wrist / right_wrist 的 children；一个 joint 的位置只依赖 parent 的 global
transform 与自己的 rest offset，所以 hand、eye、jaw pose 值与这些 slot 无关。

对 `data/dataset/human_joints_aligned.npy` 的验证覆盖 6 条随机 sequence、886 个 frame：

| slots | mean error |
|---|---:|
| 26 of 28 | **0.000 mm** |
| slot 25 (SMPL-X 28, middle1) | 28.018 mm |
| slot 27 (SMPL-X 43, middle1) | 28.018 mm |

slot 25 / 27 的差异正是文档已经登记的那一对：
`code/test_infbagel_lingo_hsi.py:869` 指出 evaluator 的 `SMPLX_JOINTS_28` 带的是
middle1（28/43），而 dataset bundle 带的是 ring1（34/49），两者“about 2.3 cm apart”。
本轮实测为 28.0 mm。因此这里的 numpy FK 对 26 of 28 slots 是精确复现，另两个误差是
已知的 slot-layout mismatch，不是 torch 被移除后产生的近似。

### C. 自检门结果

自检门逐字重算 evaluator stencil：`code/priors/hsi/metrics.py:955-1015`，在
`recon.py:jerk` 中转录，从 export 的 fine rate 重算，并逐 episode 对照封存的
`per_sequence_metrics.json`。`boundary_jerk` / `interior_jerk` / `jerk_ratio` 的
容差是 **1e-4 relative**；`boundary_jerk_samples`、`interior_jerk_samples` 与
`frame_count` 要求 exact。

| cell | max rel `boundary_jerk` | max rel `interior_jerk` | max rel `jerk_ratio` | episodes over tolerance |
|---|---:|---:|---:|---:|
| B unguided | 1.554e-05 | 7.449e-06 | 1.886e-05 | **0 / 370** |
| C unguided | 2.079e-05 | 1.139e-05 | 2.382e-05 | **0 / 370** |
| C guided v5 | 1.414e-05 | 8.126e-06 | 1.953e-05 | **0 / 370** |
| GT v3 (no export) | 4.087e-01 | 5.793e-02 | 4.056e-01 | 211 / 370 |

三个 export-backed cells 都以 0 / 370 exceedances 通过。四个 cell × 370 个 episode
的 sample counts 与 `frame_count` 均 exact match。模型格残留的约 `2e-5` 是 float32-GPU
与 float64-CPU 算术的差异。

GT 行**不是 export recompute**。`results/lingo_hsi/ground-truth-v3/` 没有 motion
export，只有一个文件 `evaluation/per_sequence_metrics.json`；这一事实已在本轮之前的
closure section G 记录。因此 GT 行是完整重实现 GT arm 的结果，而不是从 export 重算，且
这个 GT reimplementation does not reproduce。

GT relative-error distribution 如下：`boundary_jerk` median `1e-5`，48 episodes
`>1e-4`，44 `>1e-3`，29 `>1e-2`，4 `>1e-1`；`interior_jerk` median `2.4e-4`，
209 `>1e-4`，141 `>1e-3`，31 `>1e-2`，0 `>1e-1`。

### D. GT arm 为什么在设备外不可复现

`code/utils.py:quaternion_slerp` 有如下两个 arm：

```text
35:    use_lerp = dot > (1.0 - eps)              # eps = 1e-6
43:    lerped = q1 * step + q2 * (1 - step)      # endpoint weights REVERSED
```

在 `step=0` 时，LERP arm 返回 `q2` 而不是 `q1`。因此两个 arm 在它们自己的 switch
threshold 处并不一致，而 switch 又由 float32 dot product 的最后几位决定。

| episode | as reconstructed | after flipping borderline branch decisions | flips needed |
|---|---:|---:|---:|
| `061:006110` `boundary_jerk` rel | 4.087e-01 | **3.530e-05** | 2 (coarse frames 15, 16; joint 0) |
| `045:004275` `boundary_jerk` rel | 2.323e-01 | **3.220e-05** | 2 |
| `010:000335` `interior_jerk` rel | 3.980e-03 | **2.427e-07** | 1 (coarse frame 48, joint 0, `dot-thr = +5.96e-08` = half a float32 ULP) |

把 pose chain 改为 float32 而不是 float64，已经消除了最初 6 个 bad episode 中的 5 个
残差。例如 `010:000322` 的 `interior_jerk` relative error 从 4.055e-03 变成
5.338e-07；216 个 fine frame 中有 3-8 个发生差异，最大为 1.478 mm。

`lerpfix.py` 在 seed 7 的 40 个随机 episode 上显示，LERP arm 在连续
(frame, joint) pair 中的 firing rate 是 **16.0%**，范围为 [min 0.2%, max 39.6%]。
修正 endpoint weights 后，joint position 的 mean shift 是 0.0556 mm，largest single
frame 是 3.683 mm；metric 变化为：

| | as written | corrected | delta |
|---|---:|---:|---:|
| `boundary_jerk` | 79.8277 | 77.4889 | -2.93% |
| `interior_jerk` | 67.1174 | 62.2754 | **-7.21%** |
| `jerk_ratio` | 1.1638 | 1.2648 | **+8.68%** |

这不是对 tie 的 isolation，因为 branch decisions 仍来自 numpy dots。修正 weights 会使
GT ratio 更高；该 defect 是 masking seam elevation，而不是造成 seam elevation。

**C1 的 GT fine column 使用 sealed values，绝不使用这个 rebuild。**GT 的其他 view
`coarse`、`coarse_fk`、`raw30` 不含 interpolation，也不含 slerp，因此不受这一点影响。

### E. C1 主表

以下是 n=370 的均值；`[...]` 是 95% CI，bootstrap 使用 10000 reps、以 episode
重采样、seed 42。`aggB/aggI` 是 mean(boundary) / mean(interior)，即在 aggregate
上计算而不是逐 episode 计算 ratio。

views 定义为：`fine_sealed` 是 sealed value；`fine` 是我的 recompute（30 Hz、对
exported fine params 做 FK）；`fine@knots` 是相同 FK joints 在 interpolation knots
取 subsample、按 10 Hz 计分；`coarse` 是没有 interpolation、没有 FK 的 native 10 Hz
joint channel（model 使用 `global_jpos`，GT gather `human_joints_aligned`）；
`coarse_fk` 只对 GT，是未插值 coarse pose 的 FK；`raw30` 只对 GT，是同一 span、同一
seam indices 上 dataset 的 true 30 Hz motion（pure gather，不做 interpolation、FK 或
slerp）。

| cell | view | boundary_jerk | interior_jerk | jerk_ratio | aggB/aggI |
|---|---|---:|---:|---:|---:|
| GT v3 | fine_sealed | 84.5051 [80.7042, 88.3123] | 70.4399 [67.8579, 72.9919] | 1.1943 [1.1667, 1.2234] | 1.1997 |
| GT v3 | coarse | 11.9233 [11.2976, 12.5780] | 9.9646 [9.4834, 10.4532] | 1.2204 [1.1879, 1.2539] | 1.1966 |
| GT v3 | coarse_fk | 11.9641 [11.3340, 12.6244] | 9.9980 [9.5135, 10.4896] | 1.2202 [1.1876, 1.2537] | 1.1966 |
| GT v3 | raw30 | 71.8933 [62.3554, 82.8698] | 61.0106 [54.9611, 67.7692] | **1.1543** [1.0905, 1.2242] | 1.1784 |
| B unguided | fine_sealed | 127.9225 [122.4131, 133.8731] | 63.9562 [62.0851, 65.8430] | 2.0205 [1.9534, 2.0898] | 2.0002 |
| B unguided | fine | 127.9225 | 63.9561 | 2.0205 | 2.0002 |
| B unguided | fine@knots | 24.5114 [23.4714, 25.6516] | 8.2119 [7.9118, 8.5184] | 3.1893 [3.0637, 3.3203] | 2.9849 |
| B unguided | coarse | 34.0737 [32.5801, 35.7022] | 9.1493 [8.8172, 9.4813] | **3.9019** [3.7639, 4.0462] | 3.7242 |
| C unguided | fine_sealed | 128.9641 [123.6137, 134.5941] | 63.0876 [61.2590, 64.9574] | 2.0435 [1.9886, 2.1001] | 2.0442 |
| C unguided | fine | 128.9641 | 63.0876 | 2.0435 | 2.0442 |
| C unguided | fine@knots | 24.4276 [23.4115, 25.5319] | 9.2569 [8.9847, 9.5347] | 2.6659 [2.5900, 2.7469] | 2.6388 |
| C unguided | coarse | 33.4686 [31.9853, 35.0475] | 9.8852 [9.5983, 10.1806] | **3.3793** [3.2794, 3.4813] | 3.3857 |
| C guided v5 | fine_sealed | 159.4378 [144.4321, 177.0289] | 68.8335 [65.9032, 72.1294] | 2.1715 [2.0931, 2.2556] | 2.3163 |
| C guided v5 | fine | 159.4379 | 68.8335 | 2.1715 | 2.3163 |
| C guided v5 | fine@knots | 29.4305 [26.8259, 32.3985] | 10.2417 [9.6945, 10.8501] | 2.8119 [2.7062, 2.9248] | 2.8736 |
| C guided v5 | coarse | 40.3555 [36.8566, 44.3173] | 11.3073 [10.6668, 12.0387] | **3.5221** [3.3849, 3.6676] | 3.5690 |

绝对 jerk 不能跨 sampling rate 比较：third difference 在 10 Hz 跨 0.3 s，在 30 Hz
跨 0.1 s，而 `dt^-3` 相差 27x。可比较的是 `jerk_ratio`，以及同一 view 中的
cell-vs-GT comparison。

GT 的 `coarse_fk` 对 `coarse` 只差 0.34% 的 `boundary_jerk` 与 0.0002 的
`jerk_ratio`；joint-channel choice negligible。这正是 model cells 的 coarse view
可以使用 `global_jpos` 的 licensing evidence。

### F. 去掉插值后边界超出量保留了多少

三种 definition 均以 fine 对 coarse，`retained = coarse / fine`：

| cell | measure | fine | coarse | retained |
|---|---|---:|---:|---:|
| B unguided | `jerk_ratio - 1` | +1.0205 [+0.9534, +1.0898] | +2.9019 [+2.7639, +3.0462] | **284.4%** |
| B unguided | `ratio - ratio_GT` (paired) | +0.8262 [+0.7544, +0.8995] | +2.6816 [+2.5411, +2.8278] | **324.6%** |
| B unguided | `boundary/boundary_GT - 1` | +0.7382 [+0.6457, +0.8353] | +2.4928 [+2.2770, +2.7226] | **337.7%** |
| C unguided | `jerk_ratio - 1` | +1.0435 [+0.9886, +1.1001] | +2.3793 [+2.2794, +2.4813] | 228.0% |
| C unguided | `ratio - ratio_GT` (paired) | +0.8492 [+0.7877, +0.9106] | +2.1589 [+2.0560, +2.2635] | 254.2% |
| C unguided | `boundary/boundary_GT - 1` | +0.7480 [+0.6606, +0.8412] | +2.4205 [+2.2156, +2.6350] | 323.6% |
| C guided v5 | `jerk_ratio - 1` | +1.1715 [+1.0931, +1.2556] | +2.5221 [+2.3849, +2.6676] | 215.3% |
| C guided v5 | `ratio - ratio_GT` (paired) | +0.9772 [+0.8937, +1.0649] | +2.3018 [+2.1624, +2.4479] | 235.6% |
| C guided v5 | `boundary/boundary_GT - 1` | +1.1545 [+0.9466, +1.3919] | +3.1034 [+2.7356, +3.5110] | 268.8% |

FK channel cross-check：同样在 10 Hz，以 `fine@knots` 对 GT `coarse_fk = 1.2202`，
B unguided 为 +1.9691，C unguided 为 +1.4457，C guided v5 为 +1.5917；对应 fine-view
的 +0.8262 / +0.8492 / +0.9772。

**所有 definition 与两个 channel 都给出同一结论：去掉 interpolation 后，excess 变大。**

### F2. 「这只是采样率」这一质疑的实测反驳

从 30 Hz -> 10 Hz 会同时改变 stencil 的 physical aperture（一个 4-frame span 覆盖
0.1 s 对 0.3 s），并把 interpolated frames 从 interior set 中移除。如果 F 节的 elevation
只是 metric 在这一采样率改变下的性质，而不是 cell 的性质，那么 GT 应该和 models 一样
移动。下面是 paired per episode 的结果，CI over episodes：

| cell | paired change in `jerk_ratio`, fine -> coarse | multiple of the GT change |
|---|---:|---:|
| GT v3 | **+0.0261 [-0.0052, +0.0582]** (interval contains 0) | 1.0x |
| B unguided | +1.8814 [+1.7920, +1.9741] | **72.1x** |
| C unguided | +1.3358 [+1.2585, +1.4162] | 51.2x |
| C guided v5 | +1.3507 [+1.2555, +1.4491] | 51.8x |

在 FK channel 而不是 native channel 中测同一变化（`fine@knots - fine`，同样是
30 Hz -> 10 Hz）：B unguided +1.1688 [+1.0925, +1.2478]，C unguided +0.6224
[+0.5794, +0.6670]，C guided v5 +0.6404 [+0.5875, +0.6938]。GT 没有
`fine@knots` view，因为构造它需要那条不可复现的 interpolation。

因此，rate change 不是产生 model elevation 的原因。C1 **没有**解决两个同向机制之间的
拆分：interpolation 把 seam discontinuity smear 到 3 个 fine frames 上，以及
interpolation 抬高 interior denominator（在 GT 上，它把 `interior_jerk` 从 61.0106
抬到 70.4399，把 `boundary_jerk` 从 71.8933 抬到 84.5051，即两者都按相近 factor
变化）。区分这两种机制要靠 C3 的 per-offset profile；本轮未执行。

### G. GT 自己的 1.1943 也不是插值造成的

`GT fine (sealed) - GT raw30` 在 episode-paired 比较下为 **+0.0399 [-0.0268,
+0.1021]**。该 interval contains 0，因此 interpolation+FK path 没有 measurable raise
GT 的 `jerk_ratio`。coarse↔raw index map（`coarse frame c <-> raw start + 3c`）在
**370/370** 个 episode 上验证 exact。

所以，GT 自己的 `1.1943` 不能归因于 interpolation；本轮不能用 interpolation 解释
model boundary floor，也不能用 GT fine 的这条路径把该 floor 改写成 evaluator artifact。

### H. 唯一找到的真实 evaluator artifact：被 clamp 的冻结帧

`GroundTruthSource.episode_indices` 使用
`raw = min(start + 42w + 3k, end - 1)`。当 source sequence 没有填满最后一个 window
时，tail 会重复 final frame；repeated frame 的 jerk 恰好为 zero。

- **331 / 370** 个 GT episode 至少带一个 repeated coarse frame；平均是 **5.89** 个
  frozen frame（共 **86.9** 个 coarse frame）= 全部 GT coarse frame 的 **6.78%**，最大
  **13** 个。
- model cells 的 zero-motion coarse step 为 **zero**：三个 cell 的 mean `0.000`、
  max `0`，全部为零；rollout 不论 source length 如何，每个 window 都生成 16 frame。

排除所有 4-frame span touch frozen frame 的 stencil 后：

| GT view | `jerk_ratio` as scored | frozen-excluded | paired delta | interior_jerk |
|---|---:|---:|---:|---|
| coarse | 1.2204 | **1.1324** [1.1018, 1.1639] | -0.0880 [-0.0989, -0.0772] | 9.9646 -> 10.7208 (x1.0759) |
| raw30 | 1.1543 | **1.0716** [1.0117, 1.1347] | -0.0827 [-0.1010, -0.0670] | 61.0106 -> 65.5343 (x1.0741) |

Boundary stencils lost to exclusion 的平均数是 0.11；34/370 episode 至少失去一个。
所以 correction 几乎完全是 denominator 的 correction，而不是 boundary numerator 的
correction。在 genuine continuous 30 Hz motion 上移除 clamped frame 后，ratio 是
**1.0716**；seam indices 接近 neutral，正如 continuous motion 所应有。

对 frozen-excluded GT coarse null `1.1324` 的 10 Hz excess 为：B unguided **+2.7695**，
C unguided **+2.2469**，C guided v5 **+2.3897**。

### I. 派生量（明确标注为派生）

冻结帧 correction 将 GT 的 interior_jerk 提高 x1.0759（coarse）／x1.0741（raw30）。
它不能直接应用到 GT 的 fine view，因为 fine view 是 sealed value，且其中没有保存
frozen mask。下面把实测 +7.4% 做 **projection**：

| cell | fine interior_jerk | / GT as sealed | / GT projected frozen-corrected |
|---|---:|---:|---:|
| B unguided | 63.956 | 0.9080 | 0.8454 |
| C unguided | 63.088 | 0.8956 | 0.8339 |
| C guided v5 | 68.834 | 0.9772 | 0.9099 |

closure 中“unguided cells are smoother than GT in the interior（0.908x / 0.896x）”的
reading 因此是 **understated**；真实 gap 按该 projection 会更大。该 row 是 projection，
不是 measurement，不能把 projected frozen-corrected column 当作 GT fine 的实测值。

### J. 判定

C1 的 registered hypothesis——boundary floor 主要是 interpolation / evaluator artifact
——**FALSIFIED**。去掉 interpolation 后，measured excess 不但没有消失，反而在每种
definition 与两个 channel 中都增大；GT 自身 ratio elevation 也不是 interpolation
造成的（+0.04，ns），而主要是 clamped-frame artifact（排除后 -0.083）。这个 artifact
抬高 null，因此 model-vs-GT excess 是 understated 而非 overstated。剩下的是 real
rollout-continuity residual，且在 10 Hz generation rate 大于 sealed metric 所报告的
30 Hz rate。

本轮明确没有做以下事项：

- 未执行 C2–C5。
- 未修复发现的两个 evaluator defect，只记录：`quaternion_slerp` 的 reversed LERP
  weights 加 discontinuous branch，以及 GT arm 的 clamped frozen tail。
- 未启动任何 sampling 或 training；未修改 model、guidance defaults、C 或 walk `h_min`
  门槛。
- 三个 guidance switches 仍默认关闭；continuous-w 继续暂缓。
- 未使用上一轮样本做任何重新调参。
- guidance-control 路线保持正式关闭。
- C3 的 per-offset seam profile 是区分上述两种机制的检查，但本轮未执行。

这两项 defect 的处理状态是 recorded, not fixed；本轮只完成首要自检门与 C1 归因，
C2–C5 remain unexecuted。

### K. 产物清单

以下文件均位于 `.claude/scratch/c1_20260824/`，分别支撑本节各 block；scratch 目录
被 git-ignored，因此本节本身必须携带这些数字：

- B、C 的 numpy FK 与自检：`fk_numpy.py`、`check_fk.py`、`recon.py`、`run_c1.py`、
  `run_c1.out`、`gate.json`、`per_episode.json`。
- D 的 GT dtype、branch 与 LERP 修正探针：`gt_dtype.py`、`gt_branch.py`、
  `gt_branch2.py`、`lerpfix.py`、`lerpfix.json`。
- E、F、G 的聚合、raw30 与汇总：`raw30.py`、`raw30.json`、`aggregate.py`、
  `aggregate.out`、`c1_summary.json`。
- F2 的 paired rate-change 实测：`rate_delta.out`。
- H 的冻结帧 census 与排除结果：`frozen.py`、`frozen.json`。
- 本节所有 block 的唯一数字事实记录：`MEASURED_FACTS_C1.md`。

---

## 2026-08-24（同日第八次）

### A. 授权范围与本轮结论提要

本轮执行预注册的 C2–C5，范围仅为 attribution；未实现任何 fix。全程 zero GPU、zero torch、
pure numpy，使用与 C1 相同的 matched n=370。每个 CI 均按 episode 重采样 10000 reps、seed 42。

本轮把未解决的 seam floor 拆成 root translation/local pose、seam offset、history coverage/reset
decay 与 derivative order 四个归因问题；结论均以实际生成的 10 Hz 视图为准。共同结论是：
unguided floor 同时由 root translation 与 local pose 承载，超出集中在 seam 的 {-1,0,+1} 三个
stencil，window restart 的主要缺口是 acceleration/curvature continuity，而不是单纯的速度或
插值造成的宽泛 interior elevation。

### B. 通道与视图口径

`global_jpos`（native joint head）不是 FK-derived：它在同一 knot 与 FK(exported fine params)
相差 4.3–7.4 cm，native-pelvis-minus-transl offset 还存在非恒定的约 4.4 mm 漂移
（`root_channels.json`）。因此在 MODEL cell 上，native joint head 与 pose/transl head 是两个
独立的 network outputs；GT 上两个 channel 一致（C1：0.34%）。

| model view | GT partner |
|---|---|
| `fk30` | `raw30`（true 30 Hz gather） |
| `fk10` | `fk10`（FK of un-interpolated coarse pose） |
| `nat10` | `nat10`（dataset joint gather） |

GT interpolated fine view **FORBIDDEN**：它在设备外不可复现，不承载本轮结论；只能引用 sealed
GT fine scalars。

`fk30` 对 profile/ordering 工作是 interpolation-contaminated：model 的 `fk30` 是 10 Hz
generation 经 `interp_scale=3` 推到 30 Hz，interior 带有 period-3 的 piecewise-linear
interpolation kink 与 phase structure；同一 profile 在 phase 0 读到 30.58、phase 1 读到
205.47（`c3.out`）。其 GT partner `raw30` 是 genuine un-interpolated motion，因此 phase-flat。
cross-rate ordering 与 root/pose split 必须从 10 Hz views 读取；`fk30` 只作 phase-stratified
或 scalar 报告，不能作 like-for-like profile。

### C. C2 root translation 与 local pose 拆分

在同一 stencil 上分解为 full、`root_only = joints[:,0:1,:]`（并独立使用 transl channel）以及
`local pose = joints - joints[:,0:1,:]` over 27 non-zero slots。

FK-channel identity 已确认：由 `joints[:,0]` 得到的 root_only 与由 transl 得到的 root_only，
在全部 370 个 episode 上相差仅 6.26e-16 m（`joint0 = J_rest[0]+transl`；third difference
会消掉 constant）。nat10 的两个 root source 则确实不同。

paired excess over GT partner（`jerk_ratio_cell - jerk_ratio_GT`，CI over episodes）：

| view | cell | full | root_only | local pose (27) |
|---|---|---:|---:|---:|
| fk10 (clean) | B unguided | +1.9691 [1.841,2.104] | +1.2822 [1.162,1.404] | **+2.0539 [1.913,2.202]** |
| fk10 (clean) | C unguided | +1.4457 | +1.2237 | +1.3160 |
| fk10 (clean) | C guided v5 | +1.5917 | +1.3921 | +1.5214 |
| fk30 (CONFOUNDED) | B unguided | +0.8662 | +1.2311 | +0.6836 |
| fk30 (CONFOUNDED) | C unguided | +0.8891 | +1.2911 | +0.6235 |
| fk30 (CONFOUNDED) | C guided v5 | +1.0171 | +1.4204 | +0.7406 |

jerk_ratio by decomposition, fk10（`anchor_c2_fk10.json` / `c2c5.json` agree）：

| cell | full | root_only | local pose |
|---|---:|---:|---:|
| B unguided | 3.1893 | 2.5050 | 3.2884 |
| GT v3 | 1.2202 | 1.2228 | 1.2345 |

native 10 Hz 中，从 `global_jpos[:,0]` 取 root 与从 transl 取 root 的 two-head 对照
（`root_channels.json`）：

| cell | native-pelvis ratio | transl ratio | non-const drift |
|---|---:|---:|---:|
| B unguided | 3.745 [3.561,3.938] | 2.505 [2.392,2.622] | 4.4 mm |
| GT v3 | 1.223 | 1.223 | 0.000 mm |

**DELIVERABLE (C2)：common unguided floor 由 BOTH root translation 与 local pose 承载，而非
root alone。** 在模型实际生成的 interpolation-free 10 Hz rate 上，B 的 local pose excess
+2.0539 至少与 root 的 +1.2822 一样大；fk30 中 ordering 反转，但该 view 被 interpolation
confounded，不用于 verdict。两部分都明显高于 GT 的 flat ~1.22。这个结论**不复现**旧的
2026-08-23 “69% root translation”结果：旧结果是 guided-only 的 B-minus-C contrast，本轮
回答的是 unguided floor 自身的 composition，问题不同。

native joint head 在 seam 上比 transl head 有更尖的 spike（3.745 vs 2.505），所以两个 model
head 的最大分歧正好出现在 seam。

### D. C3 seam-offset profile

`offset = t0 - s + 2`；boundary stencils 为 {-1,0,+1}，其中 0 是 centred on seam 的
stencil。profile 在 10 Hz 以 bounded far band（3<=|off|<=6）归一化；unbounded tail 含有
GT 的 clamped frozen frames，会使 denominator 被压低。nat10 full-body normalized profile：

| off | -3 | -2 | -1 | 0 | +1 | +2 | +3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B unguided | 1.22 | 1.27 | **3.94** | **6.22** | **3.06** | 1.15 | 0.98 |
| GT noFz | 0.98 | 0.97 | 1.005 | 1.006 | 1.014 | 1.05 | 1.05 |

peak 在 offset 0：B 为 6.22x，C unguided 为 5.55x，C guided v5 为 5.83x（均为 nat10 band）。
在 {-1,0,+1} 的 concentration 为总 positive excess 的：

| cell | nat10 | fk10 |
|---|---:|---:|
| B unguided | 92.3% | 90.0% |
| C unguided | 94.3% | 90.2% |
| C guided v5 | 91.3% | 88.6% |

GT 的 total excess 为 0.026（noFz），约为 models 的 1/240。half-width（回到 <=1.10x band）
在 30 Hz 上 B/Cu 为 5，缩到 10 Hz 后正好是 {-1,0,+1} triple；GT half-width 为 0。
fk30 phase-stratified 读数显示，在单一 interpolation phase 内 model 仍只在 {-1,0,+1}
spike；phase 1 的 offsets -3/0/3 为 2.73/2.22/1.15，GT 为 1.08/1.03/1.10。因此
30 Hz seam excess 是真实的，period-3 oscillation 是叠加其上的 interpolation effect。

**DELIVERABLE (C3)：excess exact concentrated at {-1,0,+1}，约 90–94%，不是 broad
elevation。** `boundary_jerk` 命名正确；3-stencil window 没有在采样更宽的 defect。这也
解决了 C1 留下的 mechanism split：seam discontinuity 是 genuinely local event，peak 在
offset 0，并非 interpolation 抬高了 broad interior denominator；GT 在全部 offsets 上保持
flat（0.97–1.05），不存在可把 model spike 归因于的 broad denominator effect。

### E. C4 history coverage 与 reset decay

Part 1 structural：1110/1110 个 `(episode,cell)` 在 `seams_ok`、`len_ok`、`fine_ok`、
`gt_seams_ok` 上 PASS。每条 record 的 `history_frames = 2`；windows/episode 为 2..55。

**LIMIT（不是 PASS）：** `stitch()` 在 export 前丢弃 regenerated history frames，因此
“later window's declared history == previous window's committed frames” 在 offline 不可测量；
本项不能宣称通过。

**ASYMMETRY：** `history_frames=2` 让 new window 以 2 个 past coarse frames 为条件，但
offset -1 的 boundary stencil（`t0=s-3`）会跨 seam 回看 3 frames。

Part 2 reset transient：nat10，按 interior band `4<=|foff|<=10` 归一化（`foff=0` 是 first
new frame）：

| foff | -2 | -1 | 0 | +1 | +2 | decay |
|---|---:|---:|---:|---:|---:|---:|---:|
| speed B | 1.07 | 1.28 | 0.88 | 0.83 | 0.81 | 0 |
| accel B | 1.08 | **3.45** | **2.76** | 1.08 | 0.90 | 1 |
| accel GT noFz | 1.25 | 1.26 | 1.28 | 1.30 | 1.31 | (flat) |

**DELIVERABLE (C4)：reset transient 宽度为 2 frames（foff -1 与 0，即 last committed frame
和 first newly generated frame），是 ACCELERATION transient 而非 speed transient。** speed
仍接近 interior（1.28x），而 acceleration spike 到 3.45x；到 foff +1 已回到 interior
band 内。GT 在相同 indices 16,30,44,... 上 flat（无 window structure），decay N/A。

### F. C5 速度与加速度连续性

代数 identity 已 VERIFIED：
`max|third_diff(t0) - (a[t0+2]-a[t0+1])| = 2.220e-15 m`。因此该 metric IS the
acceleration step。

| view | cell | order1 speed | order2 vel-step | order3 metric (accel step) |
|---|---|---:|---:|---:|
| nat10 | B unguided | 1.75 [1.65,1.86] | 3.81 [3.60,4.04] | **4.84 [4.63,5.07]** |
| nat10 | C unguided | 1.84 | 3.63 | 3.84 |
| nat10 | C guided v5 | 2.07 | 3.94 | 4.07 |
| fk10 | B unguided | 1.58 | 3.05 | 3.95 |
| any | GT v3 | 1.13 | 1.14 | 1.14 |

**DELIVERABLE (C5)：excess 随 derivative order MONOTONICALLY 增长（1.75 -> 3.81 ->
4.84），所以 third difference 由 ACCELERATION/curvature step 驱动，而不是 position/velocity
discontinuity。** position 跨 seam 几乎连续（order1 仅 1.75x），velocity 有中等程度的 step，
acceleration step 最大。new window 在接近正确位置和 speed 处 restart，却没有延续 departing
window 的 curvature。GT 在相同 frame indices 16,30,44,... 的每个 order 都 flat（约 1.14x），
说明这些 indices 本身并不特殊。

诚实记录：较早的 audit anchor 曾从 confounded `fk30` view 把 ordering 读反；`fk30` 的
order3 会落到 2.47，因为 interpolation flatten 了 knots 之间的 metric。本轮已纠正，verdict
取 interpolation-free 的 10 Hz views。

### G. GT 口径与 clamp 敏感性

每一个 GT comparison 都同时报告 all 与 nofrozen，即同时包含和排除 clamped duplicate GT
frames。这里 frozen 是 clamped duplicate GT frames，占 GT coarse frames 的 6.78%；model
cell 中为 0。frozen correction 将 GT ratios 移动 <=0.03，且不改变任何 verdict，只会进一步
tighten GT 已经 flat 的 null。

横向看，C guided v5 在每个 stage 都略高于两个 unguided cells，但呈现相同的 shape
（concentration、decay、order monotonicity），符合 guidance 在已有 unguided floor 上增加
magnitude；这不是新的 guidance finding，guidance route 仍保持关闭。

### H. 最可能原因排序、候选修复方向与预计验证成本

1. **window restart 处的 curvature（acceleration）discontinuity，由 root 与 local pose
   共同承载。** Evidence：C5 order-monotonic `1.75->3.81->4.84`；C4 accel transient
   `3.45x` 而 speed 为 `1.28x`；C3 在 {-1,0,+1} 集中约 `92%`。Candidate fix：给
   autoregressive history 增加 acceleration/curvature constraint（`>=3 history frames`，或
   seam 处显式的 `2nd-order continuity term`），因为 2 个 position frames 能 pin position
   与 velocity，却不能 pin curvature。Estimated verification cost：retrain/finetune B，使用
   `history_frames>=3` 或 seam-continuity loss，再对相同 n=370 re-score；one training run +
   one 5 h sharded eval，GPU。
2. **native joint head 与 transl head 在 seam 处分歧最大。** Evidence：C2 nat10 为
   `3.745 vs 2.505`，有 `4.4 mm` drift。Candidate fix：在 seam 对 `global_jpos` 与
   FK(transl,pose) 加 consistency loss，或从 metric path 移除 native head。Estimated
   verification cost：只使用 FK joints 的 zero-GPU re-score 已经 available；training-side tie
   为 one run。
3. **interpolation 在真实 seam 之上增加 period-3 kink。** 这是 exporter/eval-representation
   issue，不是 rollout defect，且与 floor 分离。Candidate fix：在 10 Hz score，或把
   piecewise-linear joint interpolation 换成 C2 interpolant。Estimated verification cost：
   zero-GPU。

### I. 本节没有做的事

- 未实现 fix；C2–C5 仅作 attribution。
- 三个 guidance switches 仍为 default-off；C 未修改或 retrained；continuous-w 仍 deferred；
  walk `h_min` gate unchanged。
- 两个 evaluator defects（clamped GT tail、slerp/LERP branch）按用户指示保持
  recorded-not-fixed，以免扰动 existing GT convention。
- `c_guided_v4` 本轮未作为一个 cell 运行。

### J. 产物清单

本轮产物均位于 `.claude/scratch/c25_20260824/`：

- `MEASURED_FACTS_C25.md`
- `c25.py`
- `c2c5.json`
- `c3.json`
- `c3.out`
- `c4.json`
- `c4.out`
- `collect.py`
- `collected.json`
- `concentration.json`
- `concentration.py`
- `order_table.json`
- `order_table.out`
- `order_table.py`
- `reduce_c2c5.py`
- `reduce_c3.py`
- `reduce_c4.py`
- `root_channels.json`
- `root_channels.py`

## 2026-08-25（零GPU 修复可行性审计与 pilot 计划）

### A. 授权范围与本轮边界

用户批准的是**零 GPU 的修复可行性审计**，不含训练、finetune 或全量 GPU 评估。本轮
实际执行：纯 numpy + `c1_20260824/fk_numpy.py` 的 torch-free FK，匹配 **n=370**
（`r3_20260824/cells_matched.json`），零采样、零训练、零 GPU。未修改模型、C、guidance
默认值、walk 门槛、evaluator、continuous-w，未触碰 `code/priors/core/`。evaluator 的
clamp 与 slerp 缺陷继续只登记不修改。所有 GT 比较同时给出包含与排除 clamp 重复帧
（frozen）两套结果。判据一律读 **10 Hz**（模型自身生成率），`fk30` 视图受插值污染。

本轮只交付审计与计划，**不启动任何东西**。

### B. history_frames 的全部作用点

穷尽 grep：33 个文件 114 处。同一个量存在**三处平行定义**，互不派生：

| 位置 | 值 | 层 |
|---|---|---|
| `code/priors/core/contracts.py:19` `DatasetContract.history_frames` | 2 | **冻结契约** |
| `code/priors/core/representation.py:28` `RepresentationSchema.history_frames` | 2 | **冻结契约** |
| `tests/core/test_expert_contract.py:50` | 断言 == 2 | **冻结契约** |
| `code/test_infbagel_lingo_hsi.py:36` `HISTORY_FRAMES` | 2 | eval harness |
| 11 个 `code/config/*.yaml` 的 `auto_regre_num` | 2 | 训练 + 采样 |

随 `history_frames` 移动的派生量：

- `WINDOW_STRIDE_RAW = (WINDOW_FRAMES - HISTORY_FRAMES) * DATA_STEP`（`:36`），h=2 时 42
- `expected_windows = ceil((end - start - HISTORY_FRAMES*DATA_STEP)/WINDOW_STRIDE_RAW)`（`:205`）
- `pi = step * (WINDOW_FRAMES - HISTORY_FRAMES) * DATA_STEP`（`:1561`）——进度条件
- `sequence_length = episode_num * WINDOW_STRIDE_RAW + HISTORY_FRAMES * DATA_STEP`（`:1567`）
- `mat_step = get_mat(cfg, points, -HISTORY_FRAMES)`（`:1524`）与
  `initial = points[..., -HISTORY_FRAMES, 0]`（`:1541`）——窗口局部原点与 yaw 锚点是
  `-HISTORY_FRAMES` 帧。自洽（窗口第 0 帧永远是锚点），但换了源帧。
- `stitch_windows(..., history_frames=h)`：窗口 0 整存，窗口 1.. 贡献 `w[h:]`
  （`code/priors/hsi/metrics.py:253`）——决定输出长度与 **seam 索引**

训练侧（生产路径）：`train_infbagel.py:571`
`get_mask(x_start, -1, p=1., fixed_frame=cfg.auto_regre_num)`，**p=1.0** 恒定；随后
`p_losses`/`consistency_loss` 执行 `x_noisy[mask] = x_start[mask]`，故前 `auto_regre_num`
帧在每一步都是干净 GT，五个基础 loss 及 `loss_fk`/`mask_points` 全部排除它们
（`infbagel.py:431,448,875`）。

**生产/审计路径分离（已验证）**：B 的 trainer 是 `config/sampler/pelvis.yaml` 指向的
`models.infbagel.Sampler`。而硬编码读取 `REPRESENTATION.history_frames` 的
`code/priors/core/ddpm.py`（`:75,:125,:135`）属于 smoke/audit 路径
（`train_prior_smoke.py`、`config_prior_audit.yaml`），**不在 B 的训练路径上**。

### C. B checkpoint 能否仅在推理侧用 history_frames=3

两个必须分开的答案。

**架构上可以。** `hsi_b_lingo_full_v2_epoch222.pth` 共 218 个张量：含维度 16 的
**0 个**；含维度 2 的 **1 个**，即 `embedding_pelvis_goal.embedding_input.0.weight
(512,2)`（pelvis 目标的 XZ 输入，与历史无关）；所有位置编码是长度 5000 的正弦缓冲
（`scene_embedding.pos_embedding (1,17,512)` 是场景体素 token，不是运动窗口）。历史是
带内的——模型调用为 `student_model(x_noisy, occ, t, ...)`，**不传 mask**，仅凭噪声水平
推断哪些帧是条件。故钉 3 帧历史**不需要权重手术、不需要改形状**。

RNG：`p_sample_loop` 每窗口抽 1 次 `torch.randn(shape)`，`p_sample` 内每个时间步抽 1 次
`torch.randn_like(x)`，形状恒为 `[B,16,232]`，**每窗口与 h 无关**；重置按 episode
（`seed_everything(seed + canonical_ordinal)`，`:1818`），故 h=3 的第 *w* 个窗口与 h=2 的
第 *w* 个窗口**拿到相同噪声**，差异只来自窗口总数与条件内容。

**口径上不安全。** stride 42→39 的后果（`geom_h3.py`）：

| h | stride | 窗口数 | T(coarse) | stencils | boundary | interior | GT frozen |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 42 | 6.065 | 86.908 | 83.908 | 15.195 | 68.714 | 5.889 |
| 3 | 39 | 6.370 | 85.814 | 82.814 | 16.111 | 66.703 | 4.816 |
| 4 | 36 | 6.803 | 85.632 | 82.632 | 17.408 | 65.224 | 4.668 |

h=2 → h=3 逐 episode：拼接长度在 **368/370** 上改变（均值 −1.095 coarse 帧）；seam 集合
仅 **2/370** 相同；boundary stencil 占比 **16.8% → 17.9%**（stride 变短 ⇒ 每 episode
seam 更多）；源跨度守住，240.146 → 240.119 raw 帧，**334/370 完全相同**。

即"同 episode"成立，但 evaluator 的 boundary/interior 划分不成立。且
`c25.iter_episodes` 断言 model seams == GT seams，`recon.episode_indices` 按 h=2 构造 GT
臂，故 **GT 参考行也必须按 h=3 重建**。

### D. 候选 A（history_frames ≥ 3）被否决

这是本轮的头条结论，两条**互相独立**的证据，都在零 GPU 下取得。

**D-1 多给的那一帧不携带可用信息。** 受审机制是：h=2 钉住位置与一个速度差，但不钉任何
加速度；新窗口要匹配的 `a[s-2] = p[s-1]-2p[s-2]+p[s-3]` 需要 `p[s-3]`，只有 h≥3 提供。
`p[s-3]` 是否携带信息是**数据的性质**，可在 GT 上无模型测量（`headroom.py`）：

| view | stencil | E2 (h=2 信息) | E3 (h=3 信息) | E4 (h=4) | E3/E2 | E4/E3 |
|---|---|---:|---:|---:|---:|---:|
| nat10 | all | 0.01008 | 0.01035 | 0.01554 | 1.0257 [1.0118,1.0400] | 1.4980 |
| nat10 | nofrozen | 0.01092 | 0.01102 | 0.01633 | **1.0063 [0.9923,1.0209]** | 1.4764 |
| fk10 | all | 0.01011 | 0.01038 | 0.01559 | 1.0266 [1.0127,1.0409] | 1.4980 |
| fk10 | nofrozen | 0.01095 | 0.01106 | 0.01639 | **1.0071 [0.9932,1.0218]** | 1.4764 |

`E2 = |p[k+1] - (2p[k]-p[k-1])|`（匀速，h=2 信息集）；
`E3 = |p[k+1] - (3p[k]-3p[k-1]+p[k-2])|`（匀加速，加入 h=3 所给）。
**E3/E2 在每个 cell 都 ≥ 1，且排除 frozen 的 CI 含 1。** 第三帧历史不降低单步续推误差，
第四帧则差 48%。

加速度自相关（`seam_accel.py`）：

| 序列 | rho | sigma_a | rho² |
|---|---:|---:|---:|
| GT 10 Hz（模型实际粒度） | 0.3386 | 0.01126 | 11.5% |
| GT 10 Hz，排除 frozen | 0.3414 | 0.01155 | 11.7% |
| GT 30 Hz（连续运动） | **−0.2868** | 0.00311 | 8.2% |
| GT 30 Hz，排除 frozen | **−0.2910** | 0.00320 | 8.5% |

在模型自身粒度上仅约 **11.7%** 的加速度方差可由前一个加速度线性回收，而 30 Hz 上相关性
**为负**——**不存在某个采样率让这多出来的一帧变得有信息量**。"≥3 历史帧"的直觉是在平滑
连续运动上校准的，本数据在 0.1 s 间隔上不具备该平滑性。

同一机制的量级检验：若纯属信息损失，模型从正确边缘分布独立抽取接缝加速度，
`Var(j) = 2σ²` 对 GT 的 `2σ²(1-ρ)`，即**约 1.23× GT**；实测 `boundary_jerk` 是
**2.86× GT**（C1：Bu coarse 34.0737 对 GT coarse 11.9233）。信息损失差了 2.3 倍。

**D-2 缺陷不是猜错而是过冲，且 h=3 只会把它挪后一帧。** 见 E 节：超出量是窗口内瞬变，
window 0 在**精确 GT 条件**下同样出现（C4），故为生成物而非继承物。h=3 改变的是"哪一帧
算第一个生成帧"，不改变"前几个生成帧会过冲"，代价是一次重训。

**预期反驳及答复。** "E3/E2 测在内部帧上，接缝处未必成立。" 不成立：E 节 profile 显示
d ≤ −2（正含 h=3 会补上的 `p[s-3]` 邻域）位于 0.99–1.00× interior，即 h=3 新增的那一帧
本身就处在加速度正常区，对尖峰不具信息量——与 E3/E2≈1 是同一事实的两次独立印证。

**登记：** "≥3 历史帧"这一候选出自本项目 2026-08-24 C2–C5 章节自身的 "Candidate fix"
一行，现由测量**撤回**。A 的翻案条件：在模型自身 10 Hz 生成率（非 30 Hz 插值视图）上
测得 E3/E2 显著小于 1。

### E. 缺陷的真实形态

`|a|` 在新窗口首个受控索引处、对其自身 interior 水平之比（10 Hz，nat 通道，
`seam_accel.py`）：

| cell | \|a\| first | \|a\| last | \|a\| interior | first/int | step seam/step int |
|---|---:|---:|---:|---:|---:|
| B unguided | 0.02889 | 0.00904 | 0.00979 | **3.156 [3.04,3.27]** | 2.503 [2.45,2.56] |
| C unguided | 0.02742 | 0.00852 | 0.00929 | **3.039 [2.94,3.14]** | 2.214 [2.17,2.26] |
| GT v3 | 0.01173 | 0.01160 | 0.00978 | 1.229 [1.20,1.26] | 1.189 [1.16,1.23] |
| GT 排除 frozen | 0.01173 | 0.01160 | 0.01066 | **1.119 [1.09,1.15]** | 1.113 [1.08,1.15] |

上一窗口的尾端是**正常的**（B 的 `|a| last / interior` = 0.92）。纯信息损失预测
first/int ≈ 1.0，实测 3.16：模型不是挑了个**错的**加速度，而是挑了个**过大的**。

`|a[s+d]|` / interior 的逐 seam 偏移 profile（`accel_profile.py`、`accel_profile_fk.py`）：

| cell | d=−4 | d=−3 | d=−2 | **d=−1** | **d=0** | d=+1 | d=+2 | d=+3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B unguided, nat10 | 0.977 | 1.005 | 0.992 | **3.389** | **2.670** | 0.981 | 0.828 | 0.822 |
| C unguided, nat10 | 0.928 | 0.933 | 0.978 | **3.170** | **2.728** | 0.770 | 0.788 | 0.774 |
| B unguided, **fk10** | 0.946 | 0.977 | 0.947 | **2.868** | **2.064** | 1.045 | 0.826 | 0.814 |
| C unguided, **fk10** | 0.962 | 0.985 | 1.003 | **2.695** | **1.841** | 1.008 | 0.813 | 0.818 |
| GT v3, nat10 | 1.524 | 1.533 | 1.557 | 1.577 | 1.579 | 1.585 | 1.578 | 1.520 |
| GT 排除 frozen | 1.288 | 1.294 | 1.318 | 1.335 | 1.339 | 1.355 | 1.361 | 1.345 |

三条各自承重的读法：

1. **恰好 2 个加速度宽**（d=−1、d=0），到 d=+1 已回到 interior。GT 在同样偏移上是平的
   ——它没有窗口结构。
2. **属于新窗口内部，不是拼接产物。** 在 seam s 展开索引：
   `a[s-1] = w1[2] - 2·w1[1] + w1[0]`、`a[s] = w1[3] - 2·w1[2] + w1[1]`，两者完全在
   窗口 1 之内。并由上一轮 C4 的 window-0 reset 探针独立复现（foff −1/0/+1 处加速度
   3.45× / 2.76× / 1.08×，`c25_20260824/c4.out`），而 window 0 的条件帧是**精确 GT**
   ——故既非误差累积，也非自身输出的再规范化。
3. **是真实运动，不是关节头假象。** FK 通道是与 `global_jpos` 完全独立的量（记录在案：
   后者离 FK 4–7 cm 且在接缝处分歧最大），它承载同样的 2 帧形状，幅度为 85%
   （2.868 对 3.389）。头间分歧衰减了该现象但不产生它。

推得幅度：d=−1 处超出量为 (3.389 − 0.98) × 0.00979 m ≈ **2.4 cm** 的首生成帧位置误差。
（此行为推算，标注为推算。）

**LIMIT：** 逐 episode 的 "d=0 超出量 / d=−1 超出量" 配对比值在分母趋零时数值不稳定，
B/fk10 给出 −1.686 [−6.189, 0.618]。因此只读聚合 profile。B/nat10 的 0.730
[0.695,0.771] 与"孤立单帧误差"（预测 2.0）不符，而与"新窗口整体相对自身条件帧存在常量
偏移"（预测 (+δ, −δ, 0, 0…) 签名）相符。

### F. 候选 B 的代数事实

`p_losses` 中 `predicted_noise` 实为 x_start 预测（`loss_jpos = mse(x_start[...],
predicted_noise[...])`）。历史帧是 GT 常量，故接缝二阶残差

    â[1] − a*[1] = (p̂[2] − 2p[1] + p[0]) − (p[2] − 2p[1] + p[0]) = p̂[2] − p[2]

**恒等于首个生成帧的位置残差**。所以"匹配 GT 接缝加速度"这个 loss 在代数上就是把既有
`loss_jpos` 在前两个生成帧上**加权**，不引入任何新信号。两个变体必须分开：

- **B-match**（匹配 GT 加速度）≡ 逐帧加权。无偏，但上限只是"14 帧中的 2 帧能变准多少"。
- **B-smooth**（惩罚加速度幅度）≢ 加权；它把运动偏向匀速。**测量上反对采用**：窗口体
  已经比 GT 更平滑（profile d=+2..+5 为自身 interior 的 0.77–0.83；C1 的 interior 为
  0.918× GT，coarse 10 Hz：9.1493/9.9646），再往平滑推是反方向。

挂载点（若推进）：`code/models/infbagel.py` 的 `p_losses`
（`loss = loss_jpos + loss_jrot + ...` 一带的 loss 组装，经
`dict(loss=..., loss_object=..., loss_fk=...)` 返回），加上 `train_infbagel.py` loss 组装
处的一个权重。纯窗口内；不改数据管线，不改几何、契约或评测口径。

### G. 候选 A 与 B 的对照

| 维度 | A：history_frames ≥ 3 | B-match：接缝帧加权 | B-smooth：惩罚加速度 |
|---|---|---|---|
| 所需代码修改 | 3 处冻结契约 + 11 个 config + stride/pi/seq_len/锚点 + GT 臂重建 | `p_losses` 约 6 行 + `train_infbagel.py` 一个权重 | 同左 |
| 是否必须重训 | 必须，全量 29.6 h 或 finetune | finetune（约 2.7 h / 20 epoch） | finetune |
| 是否改变评测口径 | **是**：368/370 seam 不同，boundary 占比 16.8→17.9%，GT 需重建 | 否 | 否 |
| 是否触及冻结契约 | **是**（contracts.py、representation.py、契约测试） | 否 | 否 |
| 主要风险 | 一无所获（E3/E2≈1）且把瞬变挪后一帧 | 上限受限：仅在 2/14 帧变准时才有效 | 过平滑；已被测量反对 |
| 预计 GPU 成本 | 全量重训 + 全量 GT 重建评估 | 2.7 h 训练 + 分级评估 | 同左 |

成本锚点：B 全量训练 223 epoch / 146,255 updates / **29.6 h** wall clock / 4 GPU；分片
评估约 **5 h**。

### H. 推荐：先做零 GPU 的 oracle（step 0）

在任何 GPU 开销之前，在**现有导出**上对每窗口前 2 个生成帧做事后重混（post-hoc
reblend），重算整套指标。理由：瞬变恰好 2 帧宽且窗口体已过平滑，故 2 帧混合定位精准；
它同时给出**任何**接缝干预在 `boundary_jerk` 上的上界，并直接为护栏指标定价。若增益很小
或护栏退化，则 **A 与 B 同时以零 GPU 成本被否决**。

诚实前提：事后平滑器降低 `boundary_jerk` 有一部分是**构造性的**，故其正当角色是
**oracle / 上界**，不是被主张的修复；事后处理算不算修复是用户的决定（有些流程禁止改动
封存输出）。成本：仅 CPU。

**本轮已设计但未执行。**

### I. 最小 GPU pilot 设计（本轮不启动）

全部臂：B unguided，相同 episode，相同逐 episode seed。

- **Arm C0 惰性复现**：现有 `b_v2_unguided_shard8` 导出**本身**就是 h=2 对照——候选 B
  只改训练侧，eval harness 不动，故 C0 的 GPU 成本为 **0**。通过判据：逐 episode
  `boundary_jerk` 复现封存值，容差取已建立的 float32-GPU 量级（C1 在 Bu 上实测最大相对
  误差 1.554e-05）。
- **Arm T0 预算对照**：自 epoch222 起 finetune 约 20 epoch，**loss 不变**（约 2.7 h，按
  29.6 h / 223 epoch 折算）。必需，否则加权效应与多训的 20 epoch 混淆。
- **Arm T1 加权臂**：同样 20 epoch，加上逐帧加权（约 2.7 h）。

分级：

- **Stage 1（只看 jerk）**：在封存的分层 60 上评估 T0 与 T1。jerk 效应量很大，n=60 足以
  分辨；约 1.6 h。若 T1 相对 T0 不动，**停止**——候选 B 以约 6 h 总代价被否决。
- **Stage 2（护栏，仅在 Stage 1 通过后）**：两臂全量 n=370，约 10 h。因为 penetration
  需要 n≈266，在 60 上无法分辨。

### J. 冻结门槛（四条全部必需）

- **G-A 主判据（jerk）**：10 Hz 下 `|a| first / |a| interior` 须从 3.156 向 GT 的 1.119
  下降，闭合差距 ≥70%（即 ≤1.60），且 95% CI 排除 3.156。
- **G-B 护栏（不得变坏）**：penetration 须对 T0 非劣——配对 CI 不得在变差方向排除 0。
  这正是 guidance-dose 路线已经踩过的坑（dose 1/23.8 消除了全部 >5g 帧，却让
  penetration 显著变差；s=0.45 是退守点）。
- **G-C 护栏（反过平滑）**：interior `|a|` 不得进一步下降。模型已处于 0.918× GT interior（coarse 10 Hz），再降即说明 jerk 增益是用过平滑买来的。**只看 jerk 的门槛看不见这一条**，
  故单列。
- **G-D 口径**：T1 的 `seams`、`window_lengths`、`history_frames`、`frame_count` 须与 T0
  逐位相同，且 episode 集合同为 n=370。候选 B 不改几何，故任何偏离都意味着干预泄漏进了
  harness。

### K. 未做的事

- 未运行任何 GPU 工作负载；未启动训练或 finetune；未采样。
- H 节的 step-0 oracle **已设计但未执行**。
- evaluator 的 clamp 与 slerp 缺陷仍为只登记；guidance 保持关闭；未触碰 C、walk 门槛、
  continuous-w；未修改 `code/priors/core/`。
- 候选 A 建议**放弃**而非推迟，翻案条件见 D 节末。
- 本轮未测量事后重混的实际增益与护栏代价——那正是 step 0 的内容。

### L. 产物清单

`.claude/scratch/h3_audit_20260825/`：

- `geom_h3.py` / `geom_h3.json`
- `headroom.py` / `headroom.json`
- `seam_accel.py` / `seam_accel.json`
- `accel_profile.py` / `accel_profile.json`
- `accel_profile_fk.py` / `accel_profile_fk.json`
- `MEASURED_FACTS_H3_AUDIT.md`（本节全部数字的唯一出处）
- `DECISION_RECORD.md`（主 session 的判断记录与对照表）

## 2026-08-25（第二次：step-0 oracle 执行结果与 pilot 判断）

### A. 授权范围与本轮边界

用户批准执行零 GPU 的 step-0 oracle，仍不批准训练、finetune、采样或 GPU 评估。执行前
已按要求**先冻结定义**（`.claude/scratch/oracle_20260825/FROZEN_ORACLE_DEFINITION.md`，
9 节；其 §10 的 pilot 判据在任何结果产生之前追加登记）。全部脚本设
`CUDA_VISIBLE_DEVICES=''`。所有被修改的 motion 只写入 `.claude/scratch/oracle_20260825/`，
未覆盖任何封存产物、未登记为正式实验输出、未分配 run id。未触碰模型、C、guidance
默认值、walk 门槛、evaluator 源码、continuous-w、`code/priors/core/`。evaluator 的 clamp
与 slerp 缺陷仍只登记。

两类干预严格分开报告，且 **GT oracle 从未被称为可用修复**。

### B. 可行性与自检门

penetration 的主口径是 **SMPL-X 10475 顶点**，故护栏需要 CPU 重算顶点。冻结前已验证：
单 episode CPU 端到端 **1.69 s**，复现封存 `pen_ratio` 0.030906、`pen_depth_mean`
0.041674、`pene_pct_scene` 0.057432（封存 0.057431）。

**自检门 PASS**（n=12，容差 1e-4 相对）：`boundary_jerk` 最大相对 1.524e-05、
`interior_jerk` 4.874e-06、`jerk_ratio` 1.475e-05、`pen_ratio` 3.875e-05、
`pene_pct_scene` 7.696e-06、`contact_count` 2.180e-16、`fs_nemf` 4.341e-06、
`skate_ratio` 1.960e-16、`goal_planar_err_m` 5.584e-07、`min_dist` 7.057e-05。
**0 项超差**，`frame_count` 精确。1.524e-05 与 C1 独立测得的 1.554e-05 同级。

### C. 类别 1（GT oracle）：构造失效，**测不到上界**

v1 §3 的构造是"在受影响帧上代入 GT 参数"。结果：

| 量 | BEFORE | AFTER | 配对差 |
|---|---:|---:|---:|
| `\|a\|`first/int（10 Hz, fk） | 2.8684 | **61.3081** | +58.4397 SIG |
| `boundary_jerk`（30 Hz） | 127.9226 | **16941.99** | +16814.07 SIG |
| `jerk_ratio`（30 Hz） | 2.0205 | **16.3700** | +14.3495 SIG |
| `fs_nemf` | 0.32479 | 1.59393 | +1.26914 变差 |
| 最坏逐帧关节位移 | — | **411.97 cm** | 均值-max 107.90 cm |

因为 rollout 已偏离 GT，代入 GT **绝对**位姿等于把身体瞬移最多 **4.1 m**。G-E 正确
FLAG。**这是 v1 §3 的设计错误，如实登记**：它测的是瞬移，不是可达平滑度，不能作为上界。

预先登记的 §3 警告成立：GT oracle 的 penetration 列**全部变好**
（`pene_pct_scene` −0.00124），正如事先量化的混淆所预测（GT 在该列比 B 干净 1.207×），
而 `fs_nemf` 暴涨 4.9×。即其护栏列是"GT 是更好的动作"与"瞬移毁掉脚部接触"的混合，
**按定义事先声明的那样不可用于护栏预测**。

### D. 类别 2（可部署 reblend），fine 率：把封存指标弄**坏**了

| 量 | BEFORE | AFTER | 配对差 |
|---|---:|---:|---:|
| `\|a\|`first/int（10 Hz, fk） | 2.8684 | 2.4038 | −0.4646 SIG |
| `boundary_jerk`（30 Hz） | 127.9226 | **148.3538** | **+20.4313 SIG（变差）** |
| `interior_jerk`（30 Hz） | 63.9562 | 64.2022 | +0.2460 SIG |
| `jerk_ratio`（30 Hz） | 2.0205 | **2.3082** | **+0.2877 SIG（变差）** |
| `pene_pct_scene` | 0.06095 | 0.06099 | +0.00004 SIG |
| `fs_nemf` | 0.32479 | 0.33029 | +0.00550 变差 |
| `contact_count` | 946.216 | 946.417 | +0.201 变好 |
| `skate_ratio` | 0.13022 | 0.12914 | −0.00108 变好 |
| `goal_planar_err_m` / `last_dist` | 0.05187 / 0.02367 | 不变 | ns |
| 最坏逐帧关节位移 | — | 9.72 cm | 均值-max 3.75 cm |

门：G-A FAIL、G-B FAIL、G-C PASS、G-D PASS（370/370）、G-E PASS。

**机制诊断。** 封存 `boundary_jerk` 是 30 Hz 插值序列上的三阶差，而该序列的微结构
**按构造是 period-3** 的（单 episode `transl` 相位分层实测平均 `|三阶差|`
7.247e-04 / 9.036e-04 / 1.013e-03，相位间差 40%）。fine 率上的局部平滑替换不复现这个
微结构，于是跨越"已改/未改"边界的 stencil 反而变大。这是已登记的
fk30-插值混淆作用在**干预**上而非作用在**读数**上。

**v1 门算术更正：** `reduce_oracle.py` 打印的 "closed 36.9%" 误用了审计的 **nat10**
基线 3.156 去比 **fk10** 通道的实测 2.8684（后者与审计自身的 fk10 值 2.868 一致）。
正确闭合率为 (2.8684−2.4038)/(2.8684−1.119) = **26.6%**。FAIL 判定不变。

### E. 现有导出无法支撑忠实的 coarse 率事后修复

`code/utils.py:interpolate_joints` 用 `interp1d` 在
`np.linspace(0, in_len−1, in_len*scale)` 上升采样，步长 **0.330233**，不是 1/3。
单 episode 实测（T_fine 216, k 3, n_coarse 72）：只有 fine 帧 **0 与 215** 精确落在节点上
（2/216）；最近-fine-索引到节点最大偏 **0.163 帧**；由最近索引节点重建 fine 数组代价
**2.057e-03 m**。

即 `fine[::k]` 是漂移重采样而非生成网格，且**导出从未包含 coarse 的 SMPL-X 参数**，
只有其插值结果。因此参数级的 coarse 率修复在现有导出上不可重构，这也是 coarse 路线
**无法给出 penetration**（顶点 ← 参数）的原因。

对已提交工作的影响：审计的头条数字全部在 `nat10` = `global_jpos` = 真 coarse 网格上，
不受影响；审计的 `fk10` 列带此重采样假象，但它仅作对照，且 0.163 帧的重采样无法制造
2.9× 的两帧瞬变，该对照的结论成立。

### F. v2：真 coarse 网格上的修复 —— 瞬变**确实可移除**

在 `global_jpos`（真 10 Hz 生成网格）上做两侧 Hermite，仅用模型自身输出，无 GT：

| 量 | BEFORE | AFTER | 配对差 |
|---|---:|---:|---:|
| `\|a\|` first / interior | 3.3890 [3.2404,3.5453] | **1.4202** [1.3695,1.4729] | −1.9688 SIG |
| `\|a\|` first | 0.02890 | 0.01240 | −0.01650 SIG |
| `\|a\|` interior | 0.00950 | 0.00950 | **+0.000000（不变）** |
| `boundary_jerk` 10 Hz | 34.0737 | **12.1563** | −21.9174 SIG |
| `interior_jerk` 10 Hz | 9.1493 | 9.0965 | −0.0528 SIG |
| `jerk_ratio` 10 Hz | 3.9019 | **1.3794** | −2.5226 SIG |
| foot-slide 比例（10 Hz 代理） | 0.7769 | 0.7855 | +0.0086 SIG（变差） |

动作改变量：关节均值 **0.139 cm**、p95 **1.001 cm**、max **6.739 cm**
[6.367,7.135]、**最坏 episode 25.622 cm**；触及 10.65% 的 coarse 帧；跳过 0/1874 seam。

门：**G-A PASS**（1.4202 ≤ 1.800，闭合到 GT 1.119 的 **86.7%**）；
**G-B 未评估**（penetration 需顶点 ← coarse 参数，见 E 节）；**G-C PASS**（interior `|a|`
不变）；**G-D PASS**（几何按构造不变）；**G-E FLAG** —— 最坏逐帧位移 **25.62 cm**，
而瞬变幅度仅约 2.4 cm。

修复后 10 Hz `jerk_ratio` 1.3794 略高于 GT 的 coarse 1.2204，也高于排除 clamp 的 1.1324。

### G. 上界的正确陈述（替换失效的 GT oracle）

GT oracle 未能给出上界（C 节）。可用的上界陈述改为**构造性**的：解析天花板是 GT 平价，
即 10 Hz `jerk_ratio` 3.9019 → 1.2204；v2 在关节通道上实际移除了 coarse 超出量的
**94.1%**（2.6815 → 0.1590）。也就是说**天花板在关节通道上基本可达**，且这是构造证明
而非代入 GT。这比原计划的 GT oracle 更强，但它只覆盖关节通道，不覆盖 penetration。

**投影（标注为投影，非测量）：** C1 的配对视图给出插值稀释比——B 对 GT 的超出量在
fine 30 Hz 是 **+0.826**，在 coarse 10 Hz 是 **+2.6815**，比值 **3.25**。若同样移除
94.1%，fine 超出量 0.826 → 0.049，封存 `jerk_ratio` 投影为 **2.0205 → ~1.24**
（GT 1.1943）。该投影假设稀释比与干预无关，而 D 节证明对 fine 率干预**不成立**，
故它只适用于 coarse 率干预（如 B-match）。

### H. B-match pilot 是否仍值得运行

按 §10 **事先登记**的三情形判据裁决，并如实说明一处证据替换：规则的情形 3 原文要求
"GT oracle 显示有大量可移除 jerk"，而 GT oracle 失效；该证据由 F 节的构造性结果替代
（94.1% 可移除），这正是 GT oracle 本应提供的内容。

- **情形 1（可部署方案全门通过 ⇒ 不跑 pilot）：不成立。** fine 率臂 G-A/G-B FAIL；
  coarse 率臂 G-A PASS 但 G-E FLAG、G-B 未评估。
- **情形 2（可部署方案 G-B 或 G-C 失败 ⇒ 不跑 pilot）：不能据此成立。** fine 率臂确实
  G-B FAIL，但那是 `pene_pct_scene` +0.00004、相对 **+0.07%** 的变化被 n=370 的近确定性
  配对检验判为显著（见 J 节的门设计缺陷）；以此杀掉候选是错误的。coarse 率臂的 G-B
  **未评估**。
- **情形 3（可移除量大，但可部署路线偏偏在 G-E 位移上失败 ⇒ pilot 成立）：这是实际落点。**
  86.7% 的连续性差距可闭合，而代价是最坏 **25.62 cm** 位移，远超 2.4 cm 的瞬变幅度。
  按事先登记的表述，这正是"接缝需要在生成时做对、而非事后修补"的图样。

**结论：B-match pilot 仍然值得运行**，理由是事后修补要么破坏封存指标口径（fine 率），
要么以远超缺陷幅度的位移换取改善（coarse 率），而 B-match 天然是 coarse 率干预，
不受 D 节的微结构问题影响。

**但必须同时记录未解决的风险**，且按 §10 的升级条款交由用户决定：coarse 路线的
**penetration 未被评估**，所以杀死 guidance-dose 路线的那类护栏代价对本路线仍是未检验的。
Stage 2 的存在正是为此。

### I. T0 / T1 / Stage 1 / Stage 2 成本重算

锚点（实测，非估计）：B 全量 223 epoch / 146,255 updates / **29.6 h 墙钟** /
0.7305 s/update / **4 GPU** × 512 × accum 1 ⇒ 655.9 updates/epoch，**118.4 GPU-hour**；
HSI 硬件池 **8 GPU**（`training_resource_protocol.json`）；全量 eval **8 shards** ~5 h。

| 臂 | 规模 | 墙钟 | GPU-hour |
|---|---|---:|---:|
| T0 / T1 各自（20 epoch finetune） | 13,117 updates | 2.66 h（4 GPU） | 10.6 |
| Stage 1（frozen-60，两臂） | 60 episode ×2 | 1.62 h | 13.0 |
| Stage 2（全量 370，两臂） | 370 ×2 | 10.0 h | 80.0 |

**用户的 6.94 h 顺序读数经重算完全正确**：2.66 + 2.66 + 1.62 = **6.94 h 墙钟，
34.3 GPU-hour**。

**T0 与 T1 可以并行**：HSI 拥有 8 GPU，每臂用 4，两臂同时装得下（一个 run 允许只用其
expert 池的子集）。并行后 max(2.66, 2.66) + 1.62 = **4.28 h 墙钟**，而
**GPU-hour 不变，仍是 34.3** —— 并行买的是墙钟，不是算力。

并行的保留意见：两臂共享主机内存与 dataloader 带宽，4.28 h 是下界。该代码库对 layout
敏感（4×512 实测 0.71223 s/update 对 8×256 的 0.96221），两个并发 4-GPU 作业不保证维持
0.7305 s/update，并行变体建议预算 **4.3–5.5 h** 墙钟。Stage 1 的评估无法与它所评分的
训练重叠。

| 路径 | 顺序墙钟 | 并行墙钟 | GPU-hour |
|---|---:|---:|---:|
| kill path（Stage 1 失败即停） | 6.94 h | **4.28 h** | **34.3** |
| 全路径（Stage 1 + 2） | 16.94 h | **14.28 h** | **114.3** |

**必须直说的一点：** 全路径 **114.3 GPU-hour** 对 B 全量重训的 **118.4** —— 相差 4%。
分级 pilot 在全路径上**并不省算力**；它买到的是（a）Stage 1 失败时 **34.3 GPU-hour**
的止损点（重训的 29%），以及（b）墙钟 14.3 h 对 29.6 h。全路径成本的 **70%（80/114.3）
是护栏评估而非训练**，而这由 penetration 需要 n≈266 的功效要求决定，不是可选项。

### J. 门设计缺陷（登记，不追溯改判）

D 节中 `pene_pct_scene` 变化 +0.00004、基数 0.06095，**相对 +0.07%**，而配对 CI
[+0.00002,+0.00005] 排除 0，于是 G-B FAIL。在 n=370 且配对差近确定性的条件下，
"CI 不得在变差方向排除 0" 会把任何系统性变化判为失败，无论其多么微小。**G-B 在被用于
杀死候选之前需要配一个实际显著性下限**。此缺陷登记在案，不用于追溯推翻 D 节判定。

### K. 未做的事

- 未运行任何 GPU 工作负载；未训练、未 finetune、未采样、未做 GPU 评估。
- **未启动 pilot**，且不会自动启动；H 节的结论是建议，等用户决定。
- coarse 路线的 penetration / contact **未评估**（E 节的结构性原因），未以任何近似替代。
- 探索性 span sweep **未运行**；分层 SELECT 185 / CONFIRM 185（walk 各 65，seed 42，
  `split.json`）已就绪但未使用，因 headline 的 span 是先验冻结、不含选择，故全量 370
  用于确认以保住 penetration 功效。
- v1 §3 的 GT oracle 构造错误、v1 fine 率干预的微结构失败、G-B 的门设计缺陷、
  `interpolate_joints` 的节点漂移，均如实登记而非修补掩盖。
- 未修改 C、walk 门槛、continuous-w、evaluator、`code/priors/core/`。

### L. 产物清单

`.claude/scratch/oracle_20260825/`：`FROZEN_ORACLE_DEFINITION.md`（含事先登记的 §10 判据）、
`MEASURED_FACTS_ORACLE.md`（本节全部数字的唯一出处）、`PILOT_COST_RECOMPUTE.md`、
`oracle.py`、`selfcheck.py` / `selfcheck.json`、`run_oracle.py` / `oracle_rows.json`、
`reduce_oracle.py` / `reduce.out` / `oracle_summary.json`、`coarse_repair.py` /
`coarse_repair.json`、`split.json`。

## 2026-08-25（第三次：B-match Stage 1 预注册 —— 启动前冻结，含硬件调度修订 1）

### A. 授权范围与本轮边界

用户批准：零 GPU 补齐并冻结规格后，**无需再次确认**即可实施并运行 **Stage 1**；但无论
Stage 1 结果如何，**禁止自动进入 Stage 2**。启动任何 GPU 任务前必须完成：单一 T1 权重与
归一化方式的冻结（不扫描）、T0/T1 共同 checkpoint 与 optimizer/LR/resume/预算/seed/唯一
10 Hz 主视图与 G-A..G-D 的冻结、Stage 2 有实际意义的 penetration 非劣界限、**先提交预注册**、
default-off 开关加 CPU 惰性测试后**单独提交代码**。

随后用户批准**一次**硬件调度修订（在任何 GPU 结果产生前），并要求先提交修订记录。

guidance 保持关闭；不修改 C、evaluator、walk 门槛、continuous-w、`code/priors/core/`；
不运行其他实验。evaluator 的 clamp 与 slerp 缺陷继续只登记。

唯一出处：`.claude/scratch/bmatch_20260825/FROZEN_PILOT_SPEC.md`（10 节，443 行）、
`seamstat.py`（冻结统计量与估计量）、`baseline.json`、`gt_baseline.json`。

### B. 干预的四个自由度，全部由已有测量定死

**哪两帧。** 抬升的加速度恰好两个，位于 seam 偏移 d=−1 与 d=0（`accel_profile.json`：
B unguided nat10 在 d=−4..+2 为 0.977 / 1.005 / 0.992 / **3.389** / **2.670** / 0.981 /
0.828）。在窗口局部索引下展开这两个 stencil：

    a[s-1] = w[2] - 2w[1] + w[0]      仅受生成帧 w[2] 控制
    a[s]   = w[3] - 2w[2] + w[1]      仅受生成帧 w[2], w[3] 控制

故两者都只是**窗口局部第 2、3 帧**的函数。**冻结：`auto_regre_num : auto_regre_num+2`，即 2:4。**

**哪个通道。** 缺陷测在 `global_jpos`（0:84）。独立的 FK 通道以 85% 幅度承载同样的两帧形状，
故旋转通道也带这个缺陷；但 `loss_jrot` 是 L1、单位不同，单一权重无法良定义。
**冻结：仅位置 `x_start[:, 2:4, :84]`。** 旋转侧接缝项登记为后续候选，**不属于本臂**。

**哪种函数形式。** 历史帧是 GT 常量，故接缝二阶残差恒等于首生成帧的位置残差
`â[1] − a*[1] = p̂[2] − p[2]`。两种候选（δ_k = p̂[k] − p[k]）：

    (i)  逐帧加权         惩罚 δ₂² + δ₃²
    (ii) 显式加速度匹配   惩罚 δ₂² + (δ₃ − 2δ₂)²

由**实测缺陷形状**裁决：逐 episode 的"d=0 超出量 / d=−1 超出量"为 **0.730 [0.695,0.771]**，
与"孤立单帧误差"（预测 2.0）不符，与"新窗口整体常量偏移"（δ₂ ≈ δ₃ ≈ δ）相符。在该形状下
(i) 给 2δ²，(ii) 给 δ² + δ² = 2δ²，**两者相同**。**冻结：形式 (i)**，不引入新函数形式，
从而使"关掉即不变"的证明是结构性的而非数值性的。

**归一化。** **冻结：`F.mse_loss` 默认 `reduction='mean'`，只在接缝切片上取逐元素均值**，
分母 `B×2×84`。理由：现有五个基础项全是逐元素均值；`get_mask` 在 `p=1.0` 下恒定遮蔽前 2 帧
故分母确定；该形式对 batch 与帧数不变。

### C. 权重：唯一值 1.0，及其取值依据

    N_gen  = 14 × 84 = 1176（loss_jpos 分母）    N_seam = 2 × 84 = 168（loss_seam 分母）
    逐元素权重比 ρ = 1 + w·(N_gen/N_seam) = 1 + 7w

**冻结 `loss_w_seam = 1.0`。** 纯计数推得的后果：

| 量 | 值 |
|---|---|
| 逐元素权重，接缝 : 内部 | **8×** |
| 接缝两帧占位置目标的份额 | 16/28 = **57.1%** |
| 内部十二帧份额 | 12/28 = **42.9%** |
| 内部帧的**绝对**系数 | **不变**（该项是加上去的，不是重新分配） |

为什么是 1.0：

- **它是该项的自然单位**，没有可反推的小数，不像调出来的；`loss_w_fk` 同样是裸整数。
- **风险不对称，且指向"宁大勿小"。** Stage 1 是**止损**门。权重过小会返回 null，无法区分
  "B-match 无效"与"权重太小"，那 28 GPU-h 就白花了；权重过大会返回 G-A 过而 G-C 或护栏不过，
  这是**有信息**的，且后续动作（降权重）现成。
- **8× 是仍让内部保持多数（42.9%）的最大加权。** 内部已处于 0.918× GT interior，不需要帮助，
  也不该被饿着。
- **不扫描**，按指示只登记一个值。

被拒的候选权重，以便审计：ρ = 3.156（"按实测抬升加权"）——加速度**幅度**比不是误差比，
从前者到 MSE 权重没有推导，用它等于把任意选择装成测量；梯度范数配平（`loss_w_fk=3` 的先例）
——需要 GPU 前后向测量，本阶段在冻结前禁止；19:10 的逐帧 profile（两帧对三个 boundary
stencil 的平方杠杆）——可辩护，但指示要求"前两帧一个权重"，profile 需要第二个数。

### D. 代码契约：default-off 开关

单一旋钮挂在 sampler 上并自带权重，trainer 的 loss 算术**零改动**：

    # Sampler.__init__
    self.seam_loss_weight = float(kwargs.get('seam_loss_weight', 0.0) or 0.0)
    # p_losses，紧接五个基础项之后
    loss = loss_jpos + loss_jrot + loss_otrans + loss_orot + loss_contact
    loss_seam = None
    if self.seam_loss_weight > 0.0:
        n = int(self.auto_regre_num)
        loss_seam = F.mse_loss(x_start[:, n:n+2, :84], predicted_noise[:, n:n+2, :84])
        loss = loss + self.seam_loss_weight * loss_seam

`consistency_loss`（C 的目标）**不动**，C 从不设置该键，也不加接缝项。
**惰性要求（GPU 前必须在 CPU 上证明）**：`seam_loss_weight` 缺省或 0.0 时走完全相同分支，
`loss` 由完全相同表达式构成，`loss_seam is None`；测试须在固定合成 batch 上断言每个返回
loss 与改动前**逐位相同**。

### E. 冻结的启动状态（两臂共同）

| 项 | 值 | 来源 / 原因 |
|---|---|---|
| 起点 | `hsi_b_lingo_full_v2_resume.pth` | `epoch=222`、`next_epoch=223`、`micro_steps=optimizer_updates=146255`、`epoch_completed=False` |
| 权重身份 | 同 `..._epoch222.pth`，sha256 `5daaf813ca82…` | 所有封存 B v2 cell 用的同一 checkpoint |
| optimizer | 从同一文件恢复的 Adam（210 项），`lr=2e-4` | resume 恢复 `optimizer.state_dict()` |
| LR | 2000 update 线性 warmup 后恒定 2e-4，位置恢复在 `last_epoch=146255` ⇒ **全程平的** | `LambdaLR(min((u+1)/2000,1))` |
| RNG | 逐 rank `numpy/python/torch_cpu/torch_cuda` 全部恢复 | **故两臂看到逐位相同的 batch 与 timestep，是配对的** |
| 预算 | **20 epoch**，`epochs: 243`（223..242），`max_optimizer_updates: 159375` = 146255 + 20×656 | 两者同时到界，互不截断 |
| seed | 42 | 不变 |
| layout | world_size **4** × 512 × accum 1，effective 2048，bf16_tf32 | `RESUME_GEOMETRY_FIELDS` **拒绝**其他 world_size |
| 梯度裁剪 | **无** | 不变；该字段参与 resume 几何校验 |
| `OMP_NUM_THREADS` | **4** | 逐位相同，本 layout 上值 1.38× |
| 落盘 | epoch240、epoch242 与滚动 resume | `epoch % 20 == 0 or epoch == epochs-1` |
| T0 | `loss_w_seam` **关** —— loss 表达式与原始 run 逐字节相同 | 预算对照 |
| T1 | `seam_loss_weight = 1.0` | 两臂**唯一**差异 |

### F. 硬件调度修订 1（在任何 GPU 结果产生前提交）

**固定映射，不替代：T0 → GPU 0–3；T1 → GPU 4–7。** 每臂拿到自己那四张连续卡才启动，
否则**等待**；不得跨组、不得非连续、不得降 `world_size` / batch size。

B 在 4×512 下实测每卡需 **19.6 GiB**（`sweep_mb512.json`：`peak_reserved_gib` 18.797、
`driver_used_gib_after` 19.556 / 23.570）。修订时 GPU 0–3 载有他人作业各 12.9 GiB，
故 T0 **等待**而不迁移。

**双队列、彼此独立、机会式启动。** 每组一个持久监控器，自己那四张卡同时空闲即刻启动自己那臂，
不等另一组；先空出的先启动；两组同时空闲则**两臂并行，占满 8 卡**；一臂启动或完成后，
另一组监控器继续等待直到对应臂也完成。

**"空闲"由显存与 compute process 共同判定，绝不看利用率。** 仅当 `memory.used ≤ 1024 MiB`
**且** compute process 数为 **0**（任何用户），并连续 **3 次轮询（间隔 20 s）**都成立，才算空闲。
1024 MiB 已用留下 ≥ 23.0 GiB 对 19.6 GiB 需求，余量 3.4 GiB。**0% 利用率但有常驻进程不算空闲。**

**不干扰他人**：监控器只读 `nvidia-smi`，从不发信号、挂起或降权任何进程。

**重启安全**：每臂一个原子 `mkdir` 锁，加 `started` / `done` / `exit_code` 标记、`pid` 文件与独立日志。
启动前若锁存在、标记存在、PID 存活或最终 checkpoint 已在盘上，则跳过。故 Claude 断线、
监控器重启或两个监控器竞争都不会重复启动同一臂。基础设施失败只允许以完全相同配置恢复。

**修订 1 改了什么、没改什么。** 改的只有调度：placement（两臂同在 4–7 → T0 0–3 / T1 4–7 固定）、
顺序（串行 → 按组机会式，可并行）、启动条件（立即 → 等待本组空闲）、墙钟（7.12 h →
两组同空 3.56 h / 单组 7.12 h，另加排队等待）。**没改**：§B–§C 的干预、起点 checkpoint 与
sha256、optimizer 及其 Adam 状态、LR 及其恢复位置、逐 rank RNG 及由此而来的配对、
world_size 4、micro-batch 512、accum 1、effective 2048、bf16_tf32、seed 42、20 epoch /
13,120 update 预算、无裁剪、`OMP_NUM_THREADS=4`、冻结 60 及其总体权重、主统计量与唯一
10 Hz 视图、`A_GT = 1.2368`、§H 全部门槛、§I 的 Stage 2 界限，以及 **28.5 GPU-hour**。

**修订 1 放弃的东西，如实记录而非藏起来。** 修订 0 让两臂同卡的理由是"两臂只差 loss，
不差 PCIe/NUMA 位置"。本修订放弃该性质。本机上 0–3 与 4–7 两组并非可互换：同 effective batch
的 layout 变更 4×512 → 8×256 实测把 update 1 的全局梯度范数移动 **4.60%**。那是 **rank 数**
效应、在此不适用（两臂都保持 world_size 4），但两个 NUMA 组之间的残余位置差异并未被本分支上
任何测量排除。**读 Stage 1 的后果：观测到的 `A_T1 − A_T0` 与分组位置存在未测量程度的混淆。**
对**止损**门可接受——其登记通过线是闭合一个实测大小为 1.9 的差距的 40%，而位置效应在该单位上
根本没有被测到过的存在；对**发布**主张不可接受，Stage 2 不得在未测量的情况下继承这种拆分布局。

首个 GPU 进程启动后，不再允许任何修订。

### G. 主统计量与唯一视图

**冻结：`A = first_over_int`，在 `global_jpos` 上、10 Hz**（模型自身的粗生成网格 nat10）。
实现于 `bmatch_20260825/seamstat.py`，**写在两臂存在之前**；Stage 1 用该模块原封不动地归约。

逐 episode，`a[k] = p[k+1] − 2p[k] + p[k-1]`，`|a[k]|` 为逐关节 L2 再对 28 个关节取均值：

    a_first    = 对各 seam s 的 |a[s-1]| 取均值
    a_interior = 对既非 first 也非 last 的 k 的 |a[k]| 取均值
    A_ep       = a_first / a_interior      A = 对 episode 取 A_ep 的均值

**A 是比值的均值，不是均值的比值**（0.028886/0.0097907 = 2.951 而 A = 3.1565）；已登记的门槛
就是按比值的均值陈述的，故冻结此形式。不用另两个视图：`fk10` 带 `interpolate_joints` 的节点
漂移（步长 0.330233；216 个 fine 帧只有 2 个落在节点上），`fk30` 受插值污染（period-3 微结构）。

**冻结 60 上的估计量。** 该 60 是**分层**样本，故点估计为总体权重 `N_s/375` 的分层加权均值，
配对 bootstrap **在层内**重采样再按同权重合并（`hetero_20260823/stratify.py:weighted/wboot`，
10000 reps，seed 42）。对该集取未加权均值是有偏的。

**预注册基线（在两臂存在之前算出）**，冻结 60（364 窗口）加权：

| cell | A（加权） | 95% CI | A（未加权） | a_first | a_interior |
|---|---:|---|---:|---:|---:|
| B unguided epoch222 | **3.1441** | [2.8592, 3.4488] | 3.2592 | 0.028316 | 0.009823 |
| C unguided | 2.9976 | [2.7581, 3.2409] | 3.1136 | 0.027321 | 0.009438 |
| GT v3 | 1.3210 | [1.2105, 1.4374] | 1.2690 | 0.013646 | 0.010514 |
| **GT v3 排除 clamp** | **1.2368** | [1.1376, 1.3363] | 1.1864 | 0.013646 | 0.011094 |

层总体 31/46/195/58/45，权重 0.08267/0.12267/0.52/0.15467/0.12，和为 1.0。GT 无 motion NPZ，
其 nat10 按审计同一路径（`c25_20260824/recon.py`）从数据集重建，与模型导出 **0/60 seam 不一致**。
自检：加权的 3.1441 复现匹配-370 的 3.1565 到 **0.4%**，故该集对此统计量具代表性。
**冻结参考常量 `A_GT = 1.2368`**（排除 clamp，因 6.78% 的 GT 粗帧是零运动重复帧而模型没有）。

### H. 门槛

登记**两个**阈值，因为它们回答不同问题；这不是把先前登记的 70% 放宽——那是**发布**判据，
而 Stage 1 是**止损**门。

- **G-A / kill（Stage 1，只决定是否值得做 Stage 2）**：同时满足 (1) `A_T1 − A_T0` 的分层配对
  bootstrap **显著为负**；(2) 闭合率 `(A_T0 − A_T1)/(A_T0 − 1.2368) ≥ 0.40`。
- **G-A / ship（Stage 2，n=370，未加权，`A_GT = 1.1193`）**：闭合率 **≥ 0.70**，与前节一致不变。
- **G-C 反过平滑（Stage 1，与 G-A 并列为 PRIMARY）**：若配对
  `(a_interior,T1 − a_interior,T0)/a_interior,T0` 的**上**界低于 **−0.05** 则 FAIL。单侧，
  因为登记的担忧是"jerk 增益由过平滑买来"，且模型已在 0.918× GT interior。双侧值照报。
  **刻意不用裸 CI 形式**——见 §I 的门设计缺陷。
- **G-D 口径同一（Stage 1，PRIMARY）**：60 个 episode 上，T1 与 T0 的 `seams`、
  `window_lengths`、`history_frames`、`interp_scale`、`frame_count` 必须**逐位相同**，
  且 episode 集合相同；B-match 不改几何，任何偏离都意味着干预泄漏进了 harness。
  另外 T0 的几何须与封存 epoch222 导出在同 60 上一致。
- **G-B penetration —— Stage 1 不作判定，且这是算出来的而非假设的。** Stage 2 界限为
  Δ(`pen_ratio`) = 0.0030、Δ(`pene_pct_scene`) = 0.0042（§I），而冻结 60 自身的半宽是
  **0.00653** 与 **0.00745**（`smoke60.json:mde`），是界限的 **2.2×** 与 **1.8×**。
  n=60 上**任何方向的结论都不可得**。Stage 1 只把这些数作为**探索性**结果连同半宽一并报出。

**Stage 1 探索性（报半宽、不下判定）**：`pen_ratio`、`pene_pct_scene`、`pen_depth_mean`、
`min_dist`、`contact_count`、`fs_nemf`、`skate_ratio`、`goal_planar_err_m`，以及封存 30 Hz 的
`boundary_jerk` / `interior_jerk` / `jerk_ratio`。

### I. Stage 2 的 penetration 非劣界限 —— 推导得出，不是挑的

指示要求实际有意义的界限，不得用"任何非零变化都失败"。后者是已登记的缺陷：它把
**相对 +0.07%** 的变化判为失败，因为 n=370 的配对差近乎确定。

可用的界限 Δ 必须同时满足两条**实测**约束：

1. **真 null 能过。** 检验为"配对差的 97.5% 上界 < Δ"，真效应为 0 时上界即半宽 h，
   故 **Δ > h**。n=370 实测（同族封存配对）：h(`pen_ratio`) = 0.002514、
   h(`pene_pct_scene`) = 0.003014。
2. **仍能抓住项目已经否决的那个失败。** dose1 对 B guided 是封存的 FAIL：
   `pen_ratio` **+0.006046** [+0.003929,+0.008436]、`pene_pct_scene` **+0.006811**
   [+0.004392,+0.009503]。要稳健抓住（整条 CI 都在界限之上）需 **Δ < 其 CI 下界**。

| metric | h（下限） | dose1 CI 下界（上限） | 宽度 |
|---|---:|---:|---:|
| `pen_ratio` | 0.002514 | 0.003929 | 1.56× |
| `pene_pct_scene` | 0.003014 | 0.004392 | 1.46× |

非空但**很窄**，须直说：n=370 上这个门只有约 **1.5×** 的窗口介于"null 能过"与"仍能抓住已知失败"
之间。第三个锚点在其中定点——B unguided 到 GT 的剩余距离：`pen_ratio` **+0.007485**、
`pene_pct_scene` **+0.010448**（n=370 配对，均 SIG）。

**冻结：Δ = 该剩余距离的 40%**

    Δ(pen_ratio)      = 0.40 × 0.007485 = 0.002994  →  +0.0030
    Δ(pene_pct_scene) = 0.40 × 0.010448 = 0.004179  →  +0.0042

两者都落在各自可行区间内；40% 是唯一能让**两个**指标同时落入的、对某个项目内有意义量的整分数
（1/3 会让 `pen_ratio` 以 0.002495 < 0.002514 失格）。读法：T1 最多可消耗到 GT 剩余距离的 40%，
超过即算 penetration 退化。

**登记的局限**：`pene_pct_scene` 的界限只低于 dose1 CI 下界 **4.4%**（0.0042 对 0.004392），
故该界限对自身的"抓住"要求是紧的；落在 0.0042–0.0044 之间的候选会被判失败，而这个区分本设计
无法自信地分辨。

Stage 2 另冻结一个已被接受的先例锚点：s=0.45 对 B guided **未**被判为 penetration 失败
（它失在 walk h_min）：`pen_ratio` −0.000337 [−0.002492,+0.001984] ns、
`pene_pct_scene` +0.001153 [−0.001350,+0.003768] ns。上述两个界限都必须容纳它，且确实容纳。

### J. 成本重算，以及对已提交表格的一处更正

**先前提交的 §I 用错了锚点**：它取 29.6 h / 223 epoch。而该 run 自己的记录写着
**`wall_clock_hours = 20.98`**、`seconds_per_update_sustained = 0.5164`、`omp_num_threads = 4`
（`results/hsi_b_lingo_full_v2/metrics.json`）。29.6 h 是 **OMP 未设限**的口径，而该 run 并未
使用它——config 注释与 launch 脚本都写明设了限。正确的每 epoch 成本是 0.09407 h，不是 0.13274 h。

| 项 | 值 |
|---|---|
| 20 epoch = 13,120 update × 0.5164 s | 每臂 **1.882 h** 墙钟，4 GPU ⇒ **7.53 GPU-h** |
| Stage 1 评估，冻结 60，每臂 | **6.71 GPU-h**（`smoke60.json:gpu_hours`），4 GPU 上 **1.678 h** |
| **Stage 1 GPU-hour 合计** | **28.5**（与布局无关） |
| 墙钟，两组同时空闲（8 卡） | **3.56 h** = 1.882（两臂并行）+ 1.678（两评估并行） |
| 墙钟，仅一组可用、两臂串行 | **7.12 h** |

排队等待不计入以上任何一项：它不消耗 GPU-hour，其长度由他人作业决定而非本设计。
对已提交的 6.94 h / 34.3 GPU-h：**算力少 17%**。Stage 2 未重新计价，因为它未获批。

### K. 本阶段无法决定的事

- **penetration，任何方向**（§H，已量化）。
- 接缝修复能否在 n=370 上或在 guidance 下存活。
- 换一个权重会不会更好。按指示只有一个权重、不扫描。G-A 过而 G-C 不过将指示权重偏大，
  那是新实验而非微调。
- 旋转通道是否需要自己的接缝项（FK 通道 85% 的幅度提示很可能需要）。

### L. 现行禁令

Stage 1 无论 PASS 或 FAIL 都**不得**自动进入 Stage 2，须用户明示批准。guidance 保持关闭。
不修改 C、evaluator、walk 门槛、continuous-w、`code/priors/core/`。不运行其他实验。
evaluator 的 clamp 与 slerp 缺陷继续只登记不修。除 `.claude/scratch/` 与 run 目录外不产生
未跟踪文件。

## 2026-08-25（第四次：B-match 实现落地与两处被迫更正 —— 仍在任何 GPU 结果之前）

### A. 本节地位

`519caba` 冻结了预注册，`4bb972e` 提交了 default-off 开关与惰性证明。本节记录**实现阶段
被迫做出的两处更正**，两处都不是自由裁量，都是因为按指示"先做完再启动"而被发现的，
且都发生在**任何 GPU 结果产生之前**。唯一出处：
`.claude/scratch/bmatch_20260825/FROZEN_PILOT_SPEC.md` §10 修订 2。

### B. 更正 2a：起点是"仅权重续训"，不是 `resume_from`

由 AGENTS.md 要求的实数据功能 smoke 发现——这正是 smoke 要跑在正式臂之前的理由。原始
run 的滚动 `_resume.pth` 被拒绝：

    resume state was written after epoch 222 stopped mid-epoch at the optimizer-update
    budget; the run is finished and there is nothing to resume

（`train_infbagel.py:282`）。**该守卫是对的，且未被削弱**：原始 run 是因为达到自己预注册的
`max_optimizer_updates` 而结束的，继续它等于悄悄超出那个 schedule。`test_seam_loss.py`
现在断言该守卫仍在原处，因为"把守卫改掉"是显然的错误修法。本 pilot 是另行预注册的预算，
守卫无法知道这一点。

于是两臂改为从 `epoch222.pth` 的**权重**起步：`start_epoch: 223`、`load_state_dict: true`、
`resume_from: ""`。因 `utils.init_model` 用 `strict=False` 加载（会静默接受部分加载），
先在 CPU 上验证：**218/218 张量，0 missing，0 unexpected，加载后逐位相等**。

| §E 原先写的 | 现在为真的 |
|---|---|
| 从同一文件恢复 Adam 状态 | **全新 Adam 矩**，两臂相同 |
| 从同一文件恢复逐 rank RNG | **全新 RNG**，`seed + rank`，两臂相同 |
| 两臂配对 | **仍然配对**，由构造而非由恢复保证 |

配对——T1 减 T0 真正需要的性质——得以保留：`train_ddp` 在 `seed + rank` 上给
random/numpy/torch/cuda 播种（`:374-377`），数据顺序是
`DistributedSampler(seed=42).set_epoch(epoch)`，故两臂从相同的全新 Adam 状态出发，看到
**逐位相同的 batch 与 timestep**。`1/(1−β₂) ≈ 1000` update 的 Adam 瞬态因此**同时存在于
两臂**，在配对对比中抵消。

**必须直说的后果：T0 现在是 T1 的唯一合法基线。** 两臂都不是原始轨迹的纯粹延续，故封存的
epoch222 cell 对任何一臂都不是合法比较对象。门槛结构本来就以 T0 为对照，故门槛不变——但
下面那行 frozen-59 的 epoch222 数值只是**健全性参考**，不是基线。

`start_epoch: 223` 同时修好了 schedule：`optimizer_updates` 初始化为 223×656 = **146,288**，
已过 2000 update 的 warmup，故 lr 全程平在 2e-4；146,288 + 13,120 = **159,408** 使 epoch
界与 update 界继续同时到达。`ckpt_interval` 由 20 降为 **5**，使基础设施重启最多损失
约 0.47 h 而非约 1.8 h；保存块位于 epoch 之间，不消耗 RNG、不触碰 optimizer 状态，
故该节奏对轨迹中立，且两臂相同。

### C. 更正 2b：分析集是 59，不是 60

`010:000433` 的 canonical ordinal 是 **2**，而 `timing_warmup_sequences` 是 **5**，故
evaluator 将其标记 `excluded_as_warmup`。该规则是**canonical** 的（`ordinal < 5`）而非
"最先执行的几个"，所以它在封存的 8-shard run、在 375-shard run、在两臂中都被同样标记。
被标记的 episode 仍会写出 motion NPZ，但**不产生 metrics 记录**——若保留它，主统计量会算在
60 个 episode 上而护栏算在 59 个上。审计的匹配集是 375 − 5 = 370，正是同一个原因。

**冻结：唯一分析集，即 evaluator 自己的那个 —— 59 个 episode、359 个计分窗口。**
总体权重不变（它们是总体量而非样本量）；只有 S4_pene_tight 少一个成员，10 → 9。
逐层 n：16 / 10 / 16 / 9 / 8。评估仍然**运行**全部 60 个 episode，故每臂 6.71 GPU-h 不变。

重述预注册基线，frozen 59，加权：

| cell | A（加权） | 95% CI | A（未加权） | a_first | a_interior |
|---|---:|---|---:|---:|---:|
| B unguided epoch222（健全性参考） | **3.1042** | [2.8169, 3.4035] | 3.2207 | 0.028380 | 0.009923 |
| C unguided | 2.9814 | [2.7354, 3.2300] | 3.0988 | 0.027343 | 0.009492 |
| GT v3 | 1.3218 | [1.2119, 1.4394] | 1.2691 | 0.013532 | 0.010421 |
| **GT v3 排除 clamp** | **1.2362** | [1.1359, 1.3362] | 1.1850 | 0.013532 | 0.011007 |

**冻结参考常量 `A_GT`：1.2368 → 1.2362**（移动 0.05%）。§H 其余门槛全部不变。仅供定位
（A_T0 尚不存在）：若 T0 落在 epoch222 的数值上，40% 闭合线会在 A ≤ 2.3570、70% 在
A ≤ 1.7966。门槛按实测的 A_T0 计算，绝不按这两个数。

### D. 归约器与其自检

`reduce_stage1.py` 计分 G-D、G-A、G-C 与探索性护栏，**在两臂存在之前**已冻结并自检：
把封存导出同时喂给两臂，它在全部 11 个护栏与两个主统计量上返回**恰好 0.0** 的配对差、
闭合率 0.0000、G-A FAIL、G-C PASS、G-D PASS。空对照必须过不了 G-A，它确实过不了。

### E. 硬件与监控现状

监控器 `results/bmatch_stage1_20260825/monitor.sh` 已按修订 1 启动（HEAD `4bb972e`，
worktree 干净），双队列 T0→GPU 0–3、T1→GPU 4–7，只读 `nvidia-smi` 判定空闲
（`memory.used ≤ 1024 MiB` 且 compute process 为 0，连续 3 次轮询间隔 20 s）。
启动时全 8 卡被他人两个作业占用（每卡约 12.6 GiB），其中 GPU 4–7 **利用率为 0% 但有常驻
进程**——正是规则要求判为"非空闲"的情形，两组均正确等待。锁为原子 `mkdir` + PID，
另有第二层保护：即便陈旧锁被误回收，真在跑的 trainer 占着那四张卡，`gpufree.py` 会报 BUSY，
watcher 也无法启动。

### F. 现行禁令（不变）

Stage 1 无论 PASS 或 FAIL 都不得自动进入 Stage 2。guidance 保持关闭。不修改 C、evaluator、
walk 门槛、continuous-w、`code/priors/core/`。不干扰他人 GPU 进程。首个 GPU 进程启动后不再
允许任何修订。

## 2026-08-25（第五次：B-match Stage 1 执行与判定 —— G-A FAIL，且该 FAIL 对机制无判别力）

Stage 1 按 `519caba` 冻结的预注册执行完毕，两臂并行，全部退出码 0。判定为 **FAIL**。
唯一证据文件：`.claude/scratch/bmatch_20260825/FROZEN_PILOT_SPEC.md` §11–§14 与
`stage1_verdict.txt`。本节只记录结论与其效力边界，不提出 Stage 2。

### A. 执行事实

| 项 | T0（对照，`seam_loss_weight=0.0`） | T1（处理，`=1.0`） |
|---|---|---|
| GPU | 0–3 | 4–7 |
| 起点 | epoch222 权重，sha256 `5daaf813ca828788` | 同一份 |
| 训练 | exit 0，2h00m12s，ckpt `1f20c942b9399fc2` | exit 0，2h00m12s，ckpt `72fccf0d6eeeae11` |
| 评估 | 60/60 导出，全部 rc=0 | 60/60 导出，全部 rc=0 |

两臂 14:38:57 同时启动，17:36:47 全部完成。实测成本约 **20 GPU-h**（冻结估计 28.5），
其中训练比预测高 6.5%（两臂同机竞争），评估比预测快约 3.7 倍。窗口核算：每臂 364 个，
每队列 91 个；364 − 359 = 5，正好等于 warmup 排除样本 `010:000433` 的窗口数，
导出集与冻结统计量完全对齐。

### B. 判定

| 门 | 结果 |
|---|---|
| G-D 协议一致性 | **PASS** — 59 个 episode，0 个字段不匹配 |
| G-A / kill | **FAIL** — 两个条件均未满足 |
| G-C 反过平滑 | **PASS** — 上界 +1.86%，下限为 −5% |

主统计量 A（`global_jpos`，10 Hz）：T0 **3.4445** [3.1085, 3.8002]，
T1 **3.4333** [3.0911, 3.7804]，GT 1.2362。配对差 **−0.0112 [−0.1266, +0.1080] ns**，
closure **0.0051**，对 0.40 的门槛差 **78 倍**。方向符合预期，量级不符合。

### C. 为什么这个 FAIL 不能判别机制

零 GPU 阶段已测得该干预的实际剂量远低于冻结规格的描述：加入项占总 loss 的
**0.156%**，在权重相同的第 146,289 次更新上对梯度范数的扰动为 **1.9e-6**（约 16 个
float32 ULP）。冻结规格 §1 声称的"占位置目标 57.1%"是在"各帧逐元素误差均匀"这一
**假设**下的上界，实测该假设不成立；8× 的逐元素系数结论不受影响，份额结论受影响。

运行本身还提供了一个更关键的量。预注册的封存基线（B unguided，epoch222，同 59 个
episode、同统计量、同视图）为 **A = 3.1042**；而 **T0——即用完全相同的已发布 loss
多训练 20 个 epoch 的对照臂——测得 3.4445，漂移 +0.3403**。

| 量 | 值 |
|---|---|
| 处理效应 T1−T0 | −0.0112 |
| 延训漂移 T0 − epoch222 | **+0.3403** |
| 比值 | **30.4×** |

即：本设计的噪声地板是其信号的 30 倍，且方向有害。任何与 w=1.0 同量级的剂量都不可能
被该设计分辨出来。

补充一处必须自我更正的规格缺陷：§4 声称两臂"by construction 配对"。这对数据顺序与初始
噪声抽样成立，对训练轨迹不成立——`train_infbagel.py` 只设种子与 TF32 开关，
**既未设 `cudnn.deterministic` 也未设 `use_deterministic_algorithms`**，
反向传播不可逐位复现。参数空间实测 d(T0,T1) = 26.145，是两臂各自行程的 **0.439**；
1.9e-6 的梯度扰动不可能线性产生 0.439 的位移，故该分离是混沌放大，无复制臂可归因。

### D. 唯一显著的探索性指标不成立

`jerk_ratio` −0.06717（半宽 0.06067）为 SIG，但仅为自身半宽的 **1.107 倍**，且属于
**11 个未做多重比较校正**的探索性指标（5% 下期望假阳约 0.55 个）。按 ratio-of-means
分解：总变化 −0.1208，其中仅分子（boundary）贡献 −0.0514，仅分母（interior）贡献
**−0.0711**；`boundary_jerk` 本身 ns（−3.31，半宽 5.42）。§13 已指定"seam 局域性"为本
设计唯一可用的归因证据，而它不存在。**`jerk_ratio` 不得作为正面结果报告。**

### E. 一个非计划内的发现（不在本次授权范围）

延训造成的退化是 **seam 局域的**：epoch222 → T0，`a_first` 上升 11.3%
（0.028380 → 0.031593），而 `a_interior` 变化 −0.14%（0.009923 → 0.009909）。
在已发布 loss 上把 B 训得更久会专门恶化 seam 而不动内部。这是关于已发布目标本身的线索，
与 B-match 无关，**本次未获授权，不在此提出方案**。

### F. 结论

rho=8 的加速度匹配不能移动 seam。它在"成为该目标一阶项"的剂量下是否有效，**未被检验**；
本设计也无法检验——噪声地板是信号的 30 倍。后续若重试，需要同时具备远大的 w
**与**一条复制臂来标定地板。

**Stage 2 未启动**，且无论判定如何都不得在无用户明确批准的情况下启动。
