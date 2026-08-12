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

