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
