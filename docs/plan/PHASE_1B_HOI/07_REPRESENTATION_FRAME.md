# Phase 1B-07：表示帧缺陷修复与基线重建（P12）

本文件于 2026-08-19 新建，用于记录发布代码的坐标表示帧缺陷在 `phase/01b-hoi` 上的修复
与随之而来的基线重建。缺陷由 `phase/01c-hsi` 于 2026-08-18 定位（提交 `3ded4eb`），
交接文档 `/data/yujinlun/report/蔚金伦_260818_坐标表示缺陷修复与HOIPrior重训交接.md`；
本文件记录的是**在 OMOMO 上独立复测后的 HOI 侧结论**，其中若干条与交接文档正文相反，
每一条都附实测数字。
导航：[总览](../OVERVIEW.md) · [Phase 1B 索引](README.md)

#### 2026-08-19 Phase 1B P12 表示帧缺陷修复与基线重建（用户批准）

### 口径声明：这不是一个单因子实验

本仓惯例是"一个实验只动一个因子"。**P12 不满足该惯例，且不应被当作满足。**
它是一次**发布代码的缺陷修复 + 基线重建**：修复项按构造捆绑，实测任何真子集都比现状更糟
（见"为什么必须一次打完"），因此无法拆成单因子序列。预注册在此明确该偏离，
并把可归因的科学内容收缩到一个**同 checkpoint 内**的对照上（见门控 (ii)）。

| 项 | 固定值 |
|---|---|
| 子阶段 | `1B-P12` |
| 训练 run id（拟） | `p1-hoi-p12-frame-repair-baseline-s42-20260819` |
| 臂配置 | `code/config/config_train_hoi_prior_p12.yaml`——**只含配方组合与 run id，无任何被操纵因子** |
| 配置组合 | `defaults: [config_train_hoi_prior, recipe: d2ai, _self_]`；`hand_object_contact_weight` 保持 0.0 默认（即 D2-AI 的目标函数，非 W3/P10/P11 的） |
| 评测 run id（拟） | 由 `tools/hoi_chain.py` 派生，主 cell 与对照 cell 各一 |
| 主机 | worker `node01` / `infbagel-4gpu`（10.181.9.214），4× RTX 3090 |
| 载体 | `tools/hoi_chain.py` 链式 train → evaluate → bootstrap，只回传结果 |
| 预算 | 299,520,000 窗口 = 146,250 次更新（= D2-AI 全预算，未改） |
| 有效 batch | 2048 = micro 512 × 4 GPU × accum 1（已注册常规档） |
| LR / warmup / clip | 1e-4 / 无 / 无（沿用 `recipe/d2ai.yaml`，冻结未改） |
| seed / 初始化 | 42 / 随机（不从发布 checkpoint 初始化） |
| 参考成本 | D2-AI 实测 21.74 h @ 3827 windows·s⁻¹ |

### 单一变更：表示帧修复（6 文件，一次打完）

| 文件 | 变更 | 路径归属 |
|---|---|---|
| `code/priors/hoi/data.py` | 世界校正只左乘 root；模板反共轭回 y-up；不再共轭 `human_pose` | **训练** |
| `code/priors/core/window_codec.py` | heading 由内旋 `"ZXY"[…,2]` 改为外旋等价的 `"YXZ"[…,0]` | 训练 + 评测 |
| `code/datasets/utils.py` | 新增按语料功能性判定世界系的探针与校正工具 | 共同上游 |
| `code/datasets/infbagel.py` | 同上约定；`transl` y-up 恢复改为**逐布局** | **评测** |
| `code/test_infbagel_hoi.py` | 删除 `:186-191` 的补偿性 SMPL-X 三明治 | 评测 |
| `code/utils.py` | `interpolate_joints` 移到 `interp_jrot` 的 `1/scale` 基准 | 评测 |
| `tests/core/test_contract_freeze.py` | 冻结哈希重钉 `74ed3353…` → `d545359b…` | 治理 |
| `code/test_infbagel_hosi.py` | 与 HSI 分支 parity（本分支零 importer） | 治理 |

`code/config/recipe/d2ai.yaml` **逐字节未改**。`fk_weight` 保持 `0.3569973401779424`。

### 为什么必须一次打完（实测，不是判断）

| 部分修复 | 后果 |
|---|---|
| 只打 `window_codec.py` | heading 误差 137.94° → 137.36°，约定错误单独只值 **0.623°**，近似 no-op |
| 只打评测侧（不动 `priors/hoi/data.py`） | 制造一个今天不存在的约 82° 训练/评测帧错配 |
| 删 `transl` 恢复 + 删三明治 | pelvis 绝对位置误差 0.0001 m → **0.5378 m** |
| 保留三明治 + 修源头 | 顶点移 **1.87 m**，不崩，但 SDF 查询出盒、`hand_pen`/`human_pen` 静默归零 |
| 修旋转但不反共轭模板 | `data/test` 上 FK 闭合 7.06e-07 m → **1.111 m** |

### 预期收益：采样效率假设 + 评测一致性

**不主张"修好了 HOIPrior 迭代无效的根因"。** 该主张在 OMOMO 上已被测量否证：
FK 对 `human_joints_aligned.npy` 的误差修复前 **2.80e-07 m** / 修复后 2.83e-07 m（3000 帧 / 60 序列，
两个消融分别失败于 0.371 m 与 0.601 m）；`losses["fk"]` 在 GT 窗口上 1.10e-13 → 1.07e-13。
交接报告的 0.56 m 位移是 **LINGO 独有**，OMOMO 上共轭逐级相消
（`G_yup·Mᵀ · (M Pᵢ Mᵀ)… · M o_true = G_yup·P_chain·o_true`）。

**(A) 采样效率**（训练侧，OMOMO `data/train`，训练路径，1200 窗口 seed 42）：

| 量 | 完全不正则化 | 修复前 | 修复后 |
|---|---|---|---|
| 髋方位角对圆均值 p50 | 22.04° | **46.28°** | **2.13°**（p95 3.94） |
| 被去掉的 \|shift\| | — | 93.19° / p50 90.40 | 42.206 / p50 22.755（== 真实 heading，偏差 ≤1.03e-05°） |
| 旋转通道（132 维）总方差 | — | 24.399 | **11.539**（2.11×） |
| 关节通道（84 维）总方差 | — | 1.7097 | 1.5432 |
| 根 6-D 首帧总方差 | — | 0.8942 | **0.0579**（15.4×） |
| 根 6-D 首帧测地散布 | — | mean 63.40° / p50 46.93 / max 179.87 | **mean 11.12° / p50 9.67 / max 46.02** |

（1200 窗口 seed 42；总方差定义 = 逐维方差在 pooled `n_windows×16` 帧上求和，无偏。
400 窗口口径给出 24.294 → 11.398 与 0.8881 → 0.0543，两个口径一致。
修复后根 6-D 逐维 std `[.0066,.075,0.0,.0722,.0377,.2134]`——第三维**恰好为 0**，
即 heading 自由度被正则化收掉，这是该量应有的结构性签名。）

**2×2 分解（根 6-D 首帧，1200 窗口 seed 42）——本臂最硬的一条实测：**

| 约定 × 输入 | 总方差 | 测地散布 mean / p50 / max |
|---|---|---|
| 旧约定 `ZXY[2]` × 共轭输入（= 修复前） | 0.8942 | 63.40° / 46.93 / 179.87 |
| 新约定 `YXZ[0]` × y-up 输入（= 修复后） | **0.0579** | 11.12° / 9.67 / 46.02 |
| **旧**约定 × y-up 输入（只修输入） | 0.0577 | 11.13° / 9.68 / 46.76 |
| **新**约定 × 共轭输入（只修 codec） | **1.0364** | **79.85°** / 57.36 / 179.82 |

结论：收益几乎全部来自**修输入**（0.0577 ≈ 0.0579）；而**只修 `window_codec.py`
比什么都不做更糟**——方差 0.8942 → 1.0364（+16%），散布 63.40° → 79.85°（+26%），
同时作废全部已封存 checkpoint。交接报告第 6 节第 1 条建议"先把 `window_codec.py`
的同款改动落到 `phase/01b-hoi`"，按此实测**不成立**：它不是无害的第一步，是有害的第一步。

旧代码不是"未做 heading 正则化"，而是**在注入朝向噪声**（22.04 → 46.28），修复后 2.13，
优于完全不做。这是本臂唯一可主张的训练侧机制，且它是**假设**而非已证结论：
"更集中的窗口流形提升样本效率"未在 HOI 上直接测量过。

**(B) 评测一致性**（`data/test`，438 个评测窗口）：训练走 codec，评测 step ≥1 也走 codec
故本来精确一致（0.000°）；**step 0 走 `datasets/infbagel.py:439-457`，修复前错配 50.12° 均值 /
179.89° 最大，修复后约 1e-06°**。符号是**条件化**而非错误解码（`mat` 在 encode/decode 对称使用），
故修复前模型显得**比实际更差**。`obj_rot` 帧不变，故坏掉的是人-物朝向**配对**。

### 比较对象：没有可比的 before，本臂重建基线

所有 D2-* 模型侧几何行**与** released 基线行（`p0-hoi-table5-baseline-s42-20260712`）
同时被表示变更与评测器变更作废，且**不可重算**：旧 checkpoint 拟合的是 `G_yup·Mᵀ` 通道，
再叠加一个逐窗口、且是该窗口自身 root 的函数的 y 旋转，在输出上不可逆。
**禁止**报一个跨越该边界的 before/after 差值。

可归因的科学内容收缩为**同 checkpoint 内**的两个评测 cell：

| cell | step-0 帧规则 |
|---|---|
| **PRIMARY** | 修复后的规则（外旋 y，与训练一致） |
| **CONTROL** | **复算历史 step-0 帧规则**：把已修好的 root 临时重新共轭，再取 scipy 外旋 yaw 建帧 |

对照 cell **不采用**"翻回 codec 约定"：实测约定错误单独只值 0.623°，是 no-op，
会把全部差异记到模型上。成本：多一趟 438 序列 unguided。

### 数据层前置闸门（已全部通过，CPU，非结果门控）

| 闸门 | 结果 |
|---|---|
| 世界系探针 | `data/train`→'z'、`data/test`→'z'、`data/dataset`→'y'；两假设残差差 6 个数量级 |
| A 通道竖直 vs 关节竖直 | 修复后 22.5412357 **== 原始资产** 22.5412352；修复前 68.6050731 **== 被共轭的原始资产** 68.6050723（各 7 位）；FK 闭合两侧同为 2.6245e-07 m |
| B heading | 见上表；shift 前后直立度逐位不变 |
| E 旧 dataset vs `core/` codec | 旋转测地 50.12° → 2e-06°；关节通道 0.943 → 1.79e-07 |
| 四格帧 | step 0 50.12° → 9.75e-07°；step ≥1 两侧精确 0.000° |
| 绝对位置 | pelvis 0.1347 mm；对 GT 的绝对精度改前改后不变到 2.6e-08 m；pre→post 为刚体平移（0.0008 mm） |
| 插值 | 新网格与 `interp_jrot` 逐位相同；`trans_dist` 伪影 −1.23 cm；GT FS 0.35632 → 0.31654；**MPJPE Δ 恰好 0.000e+00** |
| 全仓测试 | 320 passed（默认）／317 passed 3 skipped（`INFBAGEL_WORKER_EXPERT=hoi`），与改前完全一致 |

### 结果门控（在看到结果之前写死）

- **(i) 运行与体检**：训练稳定完成 146,250 次更新，损失有限；checkpoint 通过体检四量
  （阈值见 `docs/plan/PHASE_1B_HOI/` 同节体检小节，源自 OMOMO 真值，**不得沿用 HSI/LINGO 的数**）。
  体检不通过则不花完整评测预算。
- **(ii) PRIMARY——评测一致性**：PRIMARY cell 必须在 `trans_dist` 与 `end_obj_trans_err` 上
  **显著优于** CONTROL cell，且 `contact_f1` 不得显著更差。这是本臂唯一的对照式判据。
- **(iii) 真值地板**：438 协议的 GT 参照行必须先产出并封存为 HOI 常驻真值参照行。
  **若 `foot_sliding` 的 GT 地板 ≥ released 行的 0.33336，则 FS 降级为诊断量、不作门控指标**
  （已知 GT FS 在插值修复后为 0.31654，需以完整 438 协议确认）。
- **(iv) 基线交付**：产出含全部预注册指标的新基线行，供 P13+ 比较。

### 停止分类

- `frame-repair-baseline-established`：(i)(iii)(iv) 成立，且 (ii) 成立；
- `eval-consistency-null`：(ii) 失败——step-0 帧一致性对指标无实质贡献，
  则"评测一致性"这一半收益判负，采样效率假设仍未被检验，下一入口是训练侧对照而非评测侧；
- `health-check-fail-stop`：(i) 的体检不通过——表示修复未能产出可用 checkpoint，
  回到数据层排查，不进入评测预算；
- `baseline-unstable-stop`：训练未稳定完成——按操作性失败保留，不复用 run id。

### 文件范围（锁定）

上表 8 个文件，加 `tests/hoi/` 下一个新增表示帧回归测试，加 `config_train_hoi_prior_p12.yaml`
（`tools/hoi_chain.py:70` 要求可上报的臂自己声明 run id，故该 fragment 必需；它不含任何被操纵因子），
加本节与一行 registry。**不改 `recipe/d2ai.yaml`，不改任何既有测试或 validator，
不改 `experiments/training_resource_protocol.json`。**

### 已知局限（预先声明，不等结果出来再解释）

1. **heading 正则化对水平身体退化，且 OMOMO 比 LINGO 严重。** OMOMO 躯干倾角
   p50 11.5° / p75 **51.5°** / p95 76.4° / max 94.1°。修复后髋方位角 p50 2.13 / p95 3.94 说明主体良好，
   但任何单轴 heading 对接近水平的躯干都退化。闸门 A 修复后的 22.54° 大于报告在 LINGO 上的 8.98°，
   **是语料差异**（它精确等于原始资产本身），不得当作修复不到位。
2. **骨长刚性必须在粗帧上算。** 粗帧 max 3.6e-06、0 根超 1e-2；经 `interp_s=3` 插值后
   max 0.036、**120 根超 1e-2**。在导出的插值 motion 上跑这条会把完好真值判坏。
3. **`fk_weight` 未重新推导。** 它是梯度范数均衡的产物（`auxiliary_balancing.py:57-62`，
   target 0.6279429736100133 / raw_fk 1.7589570087469566），不是损失量级校准。
   保持不动的依据是匹配扰动（σ=0.02/0.05/0.10）下的敏感度比 **1.021**——
   **1.021 是敏感度比，是梯度范数均衡的代理量，不是重新推导。** 忠实的重新推导需要在新表示下
   从随机初始化重测逐场梯度范数，属另一个预注册诊断。
4. **`code/priors/hsi/data.py` 在本分支仍共轭 root 与 pose**，两个专家的 dataset 因此暂不一致。
   归 `phase/01c-hsi` 所有；`core/` 已逐字节一致（`d545359b…`），graft 前提未破。
5. 四处 `window_codec.py:251→:260` 与 `hoi/data.py:97/200/206` 的行号引用过时，
   其中两处在 append-only 记录里不可更正。codec 必须与 `3ded4eb` 逐字节相同，行移不可避免。

### 不做什么

- **不调整几何项权重。** 交接报告正文建议下调，附录 A.1 已自我订正为"保持"；
  在 HOI 上更强：`losses["fk"]` 本来就在精确极小点（1.10e-13），无标度变化可言。
- **不新增臂配置或新配方。** 本臂就是 D2-AI 的配方在修复后的表示上重跑。
- **不维护 legacy 表示代码路径。** 旧行作为"修复前"留档；为拿到表示侧的 before/after
  需要同时翻回 step-0 帧、插值基准与三明治三个开关，那等于维护死代码。
- **不为了让检查通过而改测试或 validator。** 唯一的测试改动是冻结哈希重钉，
  那是冻结契约自身规定的授权变更程序。

---

## 实现追记：CONTROL 对照 cell（2026-08-20，用户批准）

追加节，不改上文。上文 `### 已知局限` 第 5 条与 `### 不做什么` 末条仍然成立：本节的开关
**只翻 step-0 帧一个开关**，不翻插值基准与三明治，所以它不是"legacy 表示代码路径"，
而是一个评测侧的诊断 cell。

### 机制

`InfBaGelDataset` 新增构造参数 `step0_frame_rule`，取值 `repaired`（默认）|
`historical_conjugated`，经 `+dataset.step0_frame_rule=…` 传入，与 `asset_world_up` 同路径。
**故意不写进 `config/dataset/omomo_test.yaml`**：那会在每个 PRIMARY 的 resolved config 里
多出一行；已实测 PRIMARY 的 resolved config 逐字节不变。

CONTROL 下**只有 `init_global_orient_euler` 的来源变了**：历史 root 由修复后的 root 代数重建为
`M C^T R_rep M^T`（OMOMO 上化简为 `R_rep M^T`），再取 scipy 外旋 `'zxy'[2]`。
`shift_euler`、`shift_rot_matrix`、`mat`、`joints @ S.T` 四处**是发布版原文，未改一字**。
所以两个 cell 是**同一份数据集的两个函数**，差别只在窗口帧，别无其他。

### 关键发现：CONTROL 打开的帧误差是它要复现的缺陷的 1.55 倍

438 个 step-0 窗口实测（deg）：

| 量 | 含义 | mean | p50 | p95 | max | >5° |
|---|---|--:|--:|--:|--:|--:|
| `\|C−P\|` | 只翻 codec 约定（已否决路线） | **0.410** | 0.128 | 1.79 | 4.62 | 0.0% |
| `\|A−B\|` | **发布版真实的训练/评测错配** | **50.12** | 38.98 | 145.14 | 179.89 | 90.9% |
| `\|A−P\|` | **本 CONTROL 实际打开的** | **77.83** | 51.89 | 175.60 | 179.81 | 91.6% |

（P = `C R_stored` 的外旋 `'zxy'[2]`；A = `M R_stored M^T` 的同一读法；B = 同输入的
pytorch3d 内旋 `"ZXY"[…,2]`；C = `C R_stored` 的内旋读法。）

`|A−B|` 复现上文 `:102` 记录的 50.12° 到 4 位有效数字（50.122），这是"A 确为发布版评测规则"
的最强可得识别，并已写成测试断言。已否决的 codec-flip 路线实测 0.410°，确认是 no-op，
CONTROL 是它的 **190 倍**。

**这改变 gate (ii) 的读法，必须在解释结果前声明。** 50.12° 是**同一输入上的两种约定**，
而 CONTROL 是**同一约定作用在两种输入上**，量的结构不同，所以过冲到 1.55 倍。
后果：在 77.83° 平均帧误差、52% 窗口过 45° 之下，PRIMARY 打赢 CONTROL 接近必然，
`eval-consistency-null` 从"有信息量"变成"会很意外"。

**它仍然可用，但只作上界。** D2-AI 的条件是"评测用 A、模型训练用 B"（错配 50.12°），
CONTROL 是"评测用 A、模型训练用修正规则"（错配 77.83°）——结构同类、错配更大，
故 CONTROL 给出评测器贡献份额的**上界**，`77.83 ≥ 50.12` 正是上界成立的理由。
**忠实的 `A` vs `B` 用本 checkpoint 做不出来**：它要求模型在 `B` 下训练，而 P12 训练在修正规则下。
这不是 `core/` 改动能解决的问题。

### 安全性证据（默认路径逐位不变）

两条互相独立的测量：

1. 编辑前在干净 HEAD `2593162` 上导出全部 **1314 个窗口**的 14 个数组
   （`mat, global_rot_6d, global_rot_6d_gt, joints, joints_gt, pelvis_goal, scene_goal,
   object_goal, object_trans, object_rot_mat, obj_rot_mat_ref, rest_human_offsets, transl,
   starts`），编辑后不带覆盖、以及显式 `=repaired` 各导一次：**14/14 逐字节相同，
   `maxabsdiff` 恰为 0.000e+00**。
2. 针对"可能导错了键"这一残余风险：还原 HEAD 的两个源文件，把 `__getitem__` 的**全部 35 个键**
   在 64 个窗口上逐键 SHA256，再换回：**35/35 哈希相同，0 键有别**。

`code/datasets/utils.py` 是**纯追加**（`current[:221]` 与 HEAD 逐字节相同，追加 46 行），
这一性质是承重的：训练用的 `code/priors/hoi/data.py:16` 从 `datasets.utils` import。
`code/train_hoi_prior.py` 不 import `datasets.infbagel`，已写成测试断言。
8 处对抗式源码回退（默认翻转、丢 `C^T`、CONTROL 换成 codec-flip、丢 shift 符号、
改用内旋 `'ZXY'`、删分支、`mat` 与通道脱同步、euler 索引取 0）**全部被测试捕获**。
其中"删分支"是被 (b) 的 repaired 半边捕获，**不是**被逐位测试捕获——逐位测试比较两个实例，
删分支会让两者一起移动。该局限已写进测试 docstring 并指明绝对锚点。

**GT 参照行可证明不动**：`joints_gt`、`object_rot_mat`、`obj_rot_mat_ref`、
`rest_human_offsets`、`transl` 在两个 cell 间逐字节相同，绝对全局关节一致到 **7.57e-07 m**，
且各自复现 `human_joints_aligned.npy` 到 3.97e-07 / 4.06e-07 m。

测试：`pytest tests` **340 passed**（HEAD 为 334，新增 6）；
`INFBAGEL_WORKER_EXPERT=hoi` **337 passed / 3 skipped**；零失败。
评测侧开销：1314 窗口含构造 9.39 s（repaired）vs 9.11 s（historical），无可测差异。

### 启动路线：绕开 hoi_chain 的阶段守卫

`tools/hoi_chain.py` 的 `stage_completed` 按**阶段名**取状态文件
（`results/experiments/<train_run_id>/chain/evaluate.json`），**不按 eval run id**，
所以第二次 `--stages evaluate` 会打印 "evaluate already completed; not rerunning" 而静默跳过。
CONTROL 因此**直接调用评测器**，命令与 `hoi_chain.evaluate_command` 构造的逐字节相同，
只多一个 `+dataset.step0_frame_rule=historical_conjugated` 覆盖。
不改 `tools/hoi_chain.py`——把状态文件改按 eval run id 取键是可行的修法，但那会动到
所有历史臂共用的编排代码，超出本次范围。

### 行号锚点订正

上文 `:102` 引 `datasets/infbagel.py:439-457`，该处代码现在在 `:564-574`。
本次插入使其再下移 20 行；此前已过时。append-only 记录，不原地更正。

---

## 结果订正：门控 (iii) 里的 GT `foot_sliding` 是 0.26346，不是 0.31654（2026-08-20）

追加节，不原地改上文门控行。**若与上文冲突，以本节为准。**

上文门控 (iii) 的括注写"已知 GT FS 在插值修复后为 0.31654"。**这是一处归类错误，是我写的。**

`foot_sliding` **不经过** `interpolate_joints`。代码链：GT 走 `test_infbagel_hoi.py` 的
`data_dict['joints_gt'] = self.joints[start:end]`（原生 48 帧）加未 `[::step]` 的
`global_rot_6d_gt`，直接 FK 成 `points_fk_all_gt_48`；`interpolate_joints` 只作用于**模型**的
16 帧输出。**真值行与插值修复正交。**

0.31654 的真实来处是早期评测审计里三个**并列**的数：
`shipped 0.35632 / desync-corrected 0.31654 / un-interpolated GT 0.24808`，
我把中间那个标成了 "GT FS" 并写进门控。

数值反证：预测 root 在插值三元组内的二阶差为 **2.4e-07 m**（float32 噪声，证明它就是 1/scale
线性插值），GT root 同一量为 **7.8e-03 m**（证明未插值）。

### 438 协议实测的真值参照行（封存）

| | GT 地板 | P12 模型行 | |
|---|--:|--:|--:|
| `foot_sliding` | **0.26346464114890705**（p50 0.24276 / p95 0.56484） | 0.34159 | **1.30×** |
| `feet_height` | **0.034326739609241486** | 0.06110 | +2.68 cm |
| `gt_contact_percent` | **0.6618830180474017** | 同值 | 跨 47 个 438 序列 run 逐位相同 |

探针正确性：同一脚本对 predictions 复算得 `0.3415924303475854` / `0.06109542399644852`，
与 `aggregate_metrics.json` **逐位相同**。

12 个指标解析已定（GT 自比恒为 0 的六项、接触四项恒为 1.0、`contact_percent` 恒等于
`gt_contact_percent`）；**4 个穿透指标未测**——需 SMPL-X 顶点与物体 rest SDF，导出 npz 只有
`global_jpos`。HOI 侧**没有**现成的真值评测通路：`save_chois_eval_npz` 只导出关节给 CHOIS 的
FID/R-Precision 评测器，不参与那 18 个原生指标。

**门控后果与误标所暗示的相反**：GT 地板 0.26346 **低于** released 行的 0.33336，
故 `foot_sliding` **不降级**，仍是门控指标。

### 两处执行偏离，如实记录

1. 门控 (iii) 写的是 GT 参照行须**先**产出；实际是 PRIMARY 评测先跑（chain 自动执行），
   GT 行随后建立。无科学后果：GT 行对 step-0 帧规则可证明不变（两 cell 间 7.57e-07 m），
   且它是对未变数据资产的纯 CPU 复算。
2. 若把真值行做成"config 开关在模型输出层替换 GT"，会**必然重现 0.31654 那个错数**，
   因为 `compute_metrics` 会对被替换的 16 帧再跑一遍插值。故本行由只读探针在 post-FK 层复算。

### 三条必须随行的口径限制

1. 两行不在同一平滑度基线上（模型 126 帧是 3 帧线性/slerp 段，GT 原生 30 Hz）。弦长插值只会
   **低估**滑动，故不是偏袒 GT；但不能靠"给 GT 也插值"消除，那会污染真值。
2. `feet_height` 是 DBSCAN 从运动自身估的地板高度、非绝对地面，2.68 cm 是垂直偏置的诊断量。
3. `compute_foot_sliding_for_smpl` 原地修改输入且缺权重钳制（低于地板权重可达 2.0），
   故 FS 列部分在测穿透；对两行同向。

---

## 结果订正：真值参照行的接触指标只有 `contact_acc` 是 1.0，另三项上限 0.906392694063927（2026-08-21）

追加节，不原地改上文。**若与上文冲突，以本节为准。**
封存的紧凑结果 `experiments/results/p1_hoi_p12_frame_repair_baseline_s42_20260820.json`
**保持逐字节不变**（sha256 `08bae281fa15…` 被 P12 完成行与 phase summary 第 6、240 行钉住，
按 `docs/EXPERIMENT_CONVENTIONS.md` §5 封存结果只追加不重写），其中的错值 1.0 原样留存。
详见 `docs/HOIPRIOR_EVIDENCE_INDEX.md` 结论 20 与
`docs/phase_summaries/PHASE_1B_P12_REPRESENTATION_FRAME.md` 2026-08-21 追记。

上文 `:314` 写"接触四项恒为 1.0"。**这是错的，是我写的。** 正确的是：

| 指标 | 上文写的（错） | 正确值 |
|---|--:|--:|
| `contact_precision` / `contact_recall` / `contact_f1` | 1.0 | **0.906392694063927** = 397/438 |
| `contact_acc` | 1.0 | **1.0（正确，不订正）** |

成因：438 条官方测试序列里有 **41 条没有任何真值接触帧**，而
`code/eval_metrics.py:311-323` 在 `TP + FP == 0` / `TP + FN == 0` 时把 precision / recall
直接置 0（两者皆 0 时 f1 也置 0），**真值自比也不例外**——那时 `FP = FN = 0`、
`TP = gt_contact_cnt = 0`。`code/test_infbagel_hoi.py:340` 按逐序列均值聚合，
故这 41 条以精确 0 进入**任何模型**的这三列，可达上限就是 397/438。
`contact_acc` 不受影响（零接触序列 `TP = FP = FN = 0`、`TN = 126`，准确率确实是 1.0）；
`contact_percent` 恒等于 `gt_contact_percent` = 0.6618830180474017 也仍然正确。
"12 个指标解析已定"这个计数不变，只有其中三项的值变了。

证据：这 41 条不是阈值边缘个案——最近的一条最小手-物距离 **0.07432 m**（比 0.05 m 阈值还远
2.43 cm），最远 **0.62411 m**，全 438 条里只有 **1** 条落在阈值 1 cm 以内。用评测器自己的规则
（手关节 22/23、0.05 m、126 帧跨度）从数据集加 rest 物体网格独立复算，计数逐位复现，
平均 `gt_contact_percent` 与封存值差 **0.0**；更紧的跨模型独立上界——全部 47 个已封存 438 序列 run 的
"三项同时精确为 0"集合取交集（24 个互异集合）——给出 **42**，只比 41 多 `sub17_woodchair_029`
（126 帧里 1 帧真值接触、最小距离 0.04549 m，全协议唯一的阈值边缘序列，从无模型命中）。只在有定义的 397 条上做真值往返参照得
precision **0.9968411711880822** / recall **0.9954630506006943** / f1 **0.9960322331956971**。

连带覆盖率事实：四个穿透指标只在 **181/438** 条上计算（`code/test_infbagel_hoi.py:276` 硬编码排除
woodchair、whitechair、largebox、largetable、plasticbox、trashcan），而**这 41 条里有 32 条同时被
穿透排除**——既无接触读数也无穿透读数，却仍占 438 的分母；只有 9 条保留穿透读数。

后果：本仓库任何把这三列读作"离 1.0 还差多少"的说法都**高估约 10 个百分点**。
门控 (iii) 的判定不变——它比的是与 released 行的配对差，不是与 1.0 的距离。
