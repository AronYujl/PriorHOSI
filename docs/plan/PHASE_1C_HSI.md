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


