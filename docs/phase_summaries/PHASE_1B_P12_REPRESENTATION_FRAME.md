# Phase 1B P12 — 表示帧缺陷修复与基线重建

- 分支 `phase/01b-hoi`；预注册 `docs/plan/PHASE_1B_HOI/07_REPRESENTATION_FRAME.md`
- 提交 `89411c4`（预注册）、`729e973`（实现）、`2593162`（臂配置）、`ebae4bb`（CONTROL 开关）
- 紧凑结果 `experiments/results/p1_hoi_p12_frame_repair_baseline_s42_20260820.json`
  sha256 `08bae281fa15576d7f5bdc14eb1eaa865b8498f3b5eb5b4013170c16d4c02fab`
- 停止分类 **`eval-consistency-null`**

## 结论先行

作者发布的代码把人体**旋转通道**和**关节通道**放在相差 90° 的两个世界系里，`rest_human_offsets_aligned.npy`
被 `zup_to_yup` 烘过，于是窗口 heading 正则化实际上什么都没规范化。本臂修好这三层，在 D2-AI 的配方与
预算上重训，并用一个只改评测侧 step-0 窗口帧的 CONTROL cell 去拆 M9 混淆。

**门控 (ii) 判负**，落在 `end_obj_trans_err`：配对差 +0.124、CI [−0.025, +0.282]、不显著，
而门控要求它与 `trans_dist` 同时显著优于 CONTROL。故按预注册字面分类为 `eval-consistency-null`。

**但该标签附带的理由被测量否证。** 它写的是"step-0 帧一致性对指标无实质贡献"；实测帧规则把
**7/14 个指标显著移动，全部利于修好的规则、0 项利于历史规则**，含 `mpjpe` −4.43 cm、
`obj_trans_dist` −6.92。分类不因看到结果而改写，矛盾与分类并列记录。

**PRIMARY 那一行必须作为新基线独立存在，不能读作对 D2-AI 的改进。** 帧因子的效应是观测到的
PRIMARY 减 D2-AI 差值的 **7–14 倍**，该比较已被混淆到无法分解。

## 缺陷与修复

三层，必须一起修，单独修任何一层都更糟（已实测）：

1. `human_orient.npy` 的世界系**按语料不同**（OMOMO 是 z-up、LINGO 是 y-up），代码按同一种处理。
2. 发布代码用一次 `zup_to_yup` **共轭**去凑，只在 OMOMO 上凑对；且 `human_pose` 是 21 个父相对
   局部旋转、不承载世界系，对它做世界变换没有意义。
3. 于是 `datasets/infbagel.py` 的 heading 正则化在一个 y 是水平轴的帧里取 y 角，
   实际去掉的 |shift| 中位数 3.56° 而真实 heading 中位数 92.53°。

修复：按语料探针判定世界系（`resolve_asset_world_up`），**只对 root 左乘**世界校正，
模板反共轭回 y-up，`human_pose` 不再被变换；`core/window_codec.py` 的 heading 从内旋
`"ZXY"[...,2]` 改为 `"YXZ"[...,0]`。

**同批修掉的两处**：`code/test_infbagel_hoi.py:186-191` 的补偿三明治（源头修好后再套一次会把误差加回来），
以及 `code/utils.py` 的 `interpolate_joints` 与 `interp_jrot` 网格失同步。

**取过来的与自己发现的**：授权的四个文件里 `datasets/infbagel.py` 只在 HOI 评测路径上、
`test_infbagel_hosi.py` 在本分支零 importer。**HOI 训练走 `code/priors/hoi/data.py`**
（`train_hoi_prior.py:32` 只从那里选），它不在授权集内——单独打那四个文件会留下训练侧继续共轭。
该文件按 HSI 侧同款 diff 补齐（四处修复点逐字对应，偏移一致为 −14）。

## 交接报告中被本臂测量否证的三条

1. **"平均错位 0.56 m 导致 `loss_fk` 一直在错的骨架上算"在 OMOMO 上不成立。** FK 误差旧 2.80e-07 m
   vs 新 2.83e-07 m，`losses["fk"]` 1.10e-13 → 1.07e-13。那 0.56 m 是 LINGO-only 现象，在 OMOMO 上
   与被共轭的模板恰好相消。故本臂的预期收益写作"采样效率假设 + 评测一致性"，不是"修好了迭代无效的根因"。
2. **`loss_w_fk` 不能由 `losses["fk"]` 的量级重新推导。** `fk_weight` 是梯度范数均衡
   （`auxiliary_balancing.py:57-62`，`sqrt(||g_human||·||g_object||)`），不是损失幅度校准。
   `recipe/d2ai.yaml` 冻结，权重原值沿用；1.021 是敏感度比、是均衡量的代理，不是重新推导。
3. **报告建议的"先只改 codec 约定"是有害的第一步。** 2×2 分解：新约定 × 被共轭输入 = 1.0364
   vs 修复前 0.8942，散布 79.85° vs 63.40°。

## 原生判据逐条裁决

| 门控 | 要求 | 裁决 |
|---|---|---|
| (i) 运行与体检 | 稳定完成 146,250 更新、损失有限、checkpoint 过体检四量 | **通过**，但第四量记为"不可测"而非"通过" |
| (ii) PRIMARY 评测一致性 | `trans_dist` **与** `end_obj_trans_err` 均显著优于 CONTROL，`contact_f1` 不得显著更差 | **判负**，见下 |
| (iii) 真值地板 | 438 协议 GT 参照行须先产出并封存 | **通过**，含两项偏离 |
| (iv) 基线交付 | 含全部预注册指标的新基线行 | **通过**，438/438 |

门控 (ii) 逐项：

| 指标 | 配对差 | 95% CI | 显著 | 裁决 |
|---|--:|---|---|---|
| `trans_dist` | −2.94284 | [−3.31002, −2.59399] | 是 | PASS |
| **`end_obj_trans_err`** | **+0.12419** | **[−0.02538, +0.28236]** | **否** | **FAIL** |
| `contact_f1` | +0.06789 | [+0.04423, +0.09169] | 是 | PASS |

门控在看到结果之前按字面写死，事后未被重新解释。

## 配对结果（B = PRIMARY 修复后帧，A = CONTROL 历史帧）

`tools/paired_bootstrap.py`，438/438 配对，10,000 次重采样，seed 42，按序列重采样。
**7/14 显著，全部利于 PRIMARY，0 项利于 CONTROL。**

| 指标 | CONTROL | PRIMARY | 配对差 | 95% CI |
|---|--:|--:|--:|---|
| `mpjpe` | 15.71071 | 11.28390 | **−4.42681** | [−5.03786, −3.82698] |
| `obj_trans_dist` | 21.02608 | 14.10532 | **−6.92076** | [−8.01832, −5.82218] |
| `trans_dist` | 10.88064 | 7.93780 | **−2.94284** | [−3.31002, −2.59399] |
| `pelvis_goal_error_cm` | 4.96294 | 3.61873 | **−1.34421** | [−1.57349, −1.11831] |
| `obj_rot_dist` | 1.31455 | 0.96753 | **−0.34702** | [−0.41017, −0.28592] |
| `contact_recall` | 0.55288 | 0.63314 | **+0.08026** | [+0.05380, +0.10729] |
| `contact_f1` | 0.60776 | 0.67565 | **+0.06789** | [+0.04423, +0.09169] |

不显著：`contact_precision`、`end_obj_trans_err`、`foot_sliding`、四个穿透项。

### CONTROL 能与不能确立什么

**能**：step-0 帧规则是一阶因子，不是可忽略项。

**不能**：分解 PRIMARY 减 D2-AI 的差值。CONTROL 与 D2-AI 的错配**性质不同**——D2-AI 是训练用 `B`、
评测用 `A`，两者都在历史世界，模型在那个世界里学过；CONTROL 是模型训练在修正世界却被喂历史帧，
是纯 OOD 侮辱、无补偿性训练。三个帧误差量（438 个 step-0 窗口，deg）：

| 量 | 含义 | mean | p50 |
|---|---|--:|--:|
| `\|C−P\|` | 只翻 codec 约定（已否决路线） | **0.410** | 0.128 |
| `\|A−B\|` | 发布版真实的训练/评测错配 | **50.12** | 38.98 |
| `\|A−P\|` | 本 CONTROL 实际打开的 | **77.83** | 51.89 |

过冲 **1.55 倍**，而帧因子效应是观测差值的 7–14 倍，故上界松了约一个数量级。
`|A−B|` 复现预注册记录的 50.12° 到 4 位有效数字（50.122），这是"A 确为发布版评测规则"的最强可得识别。
忠实的 `A` vs `B` 要求模型在 `B` 下训练，**用本 checkpoint 做不出来**，不是 `core/` 改动能解决的。

四个穿透项与 `foot_sliding` 在帧因子下不显著，但 CI 宽到足以容纳观测差值
（`hand_pen_ratio` 的 [−0.02054, +0.03151] 包含观测的 +0.01614），故对它们**既不能证实也不能排除**。
穿透那约 +13% 的回退仍然未解释。

## 体检（四个共轭不变或有独立参照的量）

读数路径先自证：真值经**同一条导出管线**复算与直读 `data/test` 逐位相符
（分母 3.4330 vs 3.433、池化 max 27.1346 vs 27.13、接缝 4.1562/3.2066/11.1770、
(ii) 8.7329/6.6981/22.4934/43.9058）。骨盆旋转由骨盆三子关节精确 Procrustes 反解
（cond 3.92、残差 ≤1.1e-7 m），对真值验证到 1e-4°。

| 检查 | 生成侧 | HOI 真值 | 旧坏 checkpoint | 判定 |
|---|--:|--:|--:|---|
| (i) 根旋转增量 步1→2 | p50 4.745°，r 1.31–1.61 | p50 3.723°，r 0.879 | 122.68° | 健康（阈值 r≤2.0 且 p50≤15°） |
| (ii) 骨盆 up 轴 | mean 9.78°，>45° **0%** | mean 8.73°，>45° 0% | 90.58° | 健康，落在真值支撑内 |
| (iii) 关节通道骨长 CV | 0 | 0 | 0.101 | **不可测，非通过** |
| (iv) FK pelvis→neck | p50 10.15°，>45° 31.4% | p50 11.26°，>45° 27.9% | 95.15° | 健康 |

**表示帧修复在生成侧生效**：(ii) 从 90.58° 回到 9.78° 且零帧过 45°，(iv) 中位数落在真值之下，
两者都没有任何质量靠近 90°。

**(iv) 的重尾是 OMOMO 自己的性质**，真值本身 27.9% 过 45°、18.2% 过 60°；搬运任务弯腰取物本来如此。
留账不门控：生成侧 `frac>80°` 0.0551 是真值 0.0217 的 **2.53 倍**，max 超出真值 11.5°。

**(iii) 是规格缺陷，不是通过。** 导出是 `human_jnts_48`（由旋转通道 + rest offsets 做 FK），
骨长按构造刚性，pred 与 GT 都读 0，比真值关节数组读数还低 4 倍，是 float32 往返噪声。
要真测它需要 `points_orig_seg`（`test_infbagel_hoi.py:137-138`，前 24 通道是人体关节），
它存在于 `compute_metrics` 内但从不落盘。**一个真实的关节通道刚性缺陷会从这个闸门下漏过去。**

**意外的正面结果**：模型的接缝连续性**优于真值**（p50 3.1337 vs 3.2066、p95 9.162 vs 11.177），
同时窗口内系统性比真值平滑（池化 p50 2.8622 vs 3.3610 = 85%）。HOI 上不存在 HSI 侧那种接缝不连续。

**一个未解现象**：17,082 个样本里恰好 1 个超过 90°（157.888°，`sub16_plasticbox_010`、window step 1、
粗帧 12→13，真值同位置 11.792°，Procrustes 残差 5.9e-08 故是真实内容）。分解后是**绕近竖直骨盆轴的
偏航翻转、不是倾倒**（总测地角 157.888° 但骨盆 up 轴只走 18.456°）。不是孤例：12 个 >20° 的样本里
10 个是 plasticbox、10 个在 window step 1、10 个在窗口后段（粗帧 ≥11），涉及 4 个序列；真值任何位置不超 30°。

## 真值参照行（封存）

HOI 侧**没有**现成的真值评测通路：`save_chois_eval_npz` 只导出关节给 CHOIS 的 FID/R-Precision
评测器，不参与那 18 个原生指标。本行由只读探针用 `eval_metrics` 原函数从导出 npz 复算，
零源码改动、零 GPU。**探针正确性**：同一脚本对 predictions 复算得 0.3415924303475854 /
0.06109542399644852，与 `aggregate_metrics.json` 逐位相同。

| | GT 地板 | P12 模型 | |
|---|--:|--:|--:|
| `foot_sliding` | **0.26346464114890705**（p50 0.24276 / p95 0.56484） | 0.34159 | **1.30×** |
| `feet_height` | **0.034326739609241486** | 0.06110 | +2.68 cm |

12 个指标解析已定：`mpjpe`/`trans_dist`/`obj_trans_dist`/`obj_rot_dist`/`end_obj_trans_err`/
`xy_points_err` 为 0（GT 自比），四个接触指标为 1.0，`contact_percent` 恒等于
`gt_contact_percent` = 0.6618830180474017（该值跨 47 个 438 序列 run 逐位相同）。
**4 个穿透指标未测**：需 SMPL-X 顶点（pose+betas+gender+transl）+ 物体 rest SDF，导出 npz 只有 `global_jpos`。

### 订正：GT `foot_sliding` 是 0.26346，不是 0.31654

`foot_sliding` **不经过** `interpolate_joints`——GT 走 `data_dict['joints_gt']`（原生 48 帧）直接 FK，
`interpolate_joints` 只作用于模型的 16 帧输出。0.31654 是早期审计里三个并列数
（`shipped 0.35632 / desync-corrected 0.31654 / un-interpolated GT 0.24808`）的**中间那个**，
被误标为 "GT FS" 并写进本臂门控。数值反证：预测 root 三元组内二阶差 2.4e-07 m（float32 噪声，
证明是 1/scale 线性插值），GT root 7.8e-03 m（证明未插值）。**真值行与插值修复正交。**

**门控后果与误标所暗示的相反**：GT 地板 0.26346 **低于** released 行的 0.33336，
故 `foot_sliding` **不降级**，仍是门控指标。

### 三条必须随行的口径限制

1. 两行不在同一平滑度基线上：模型的 126 帧是 3 帧线性/slerp 段，GT 是原生 30 Hz。
   弦长插值只会**低估**滑动，故这不是偏袒 GT；但也**不能**靠"给 GT 也插值"消除，那会污染真值。
2. `feet_height` 是 DBSCAN 从运动自身估的地板高度、不是绝对地面，故 2.68 cm 是垂直偏置的**诊断量**而非误差。
3. `compute_foot_sliding_for_smpl` 原地修改输入且缺权重钳制（低于地板的帧权重可达 2.0），
   故 FS 列**部分在测穿透**；该缺陷对两行同向存在。

## 训练与评测记录

| | |
|---|---|
| 主机 | worker `node01`，训练 `CUDA_VISIBLE_DEVICES=0,1,2,3`，评测 1 卡 |
| 预算 | 299,520,000 窗口（精确打满）/ 146,250 次更新 / 526.87 epoch 等效 |
| 有效 batch | 2048（注册档位），梯度累积 1，`OMP_NUM_THREADS=4` |
| 墙钟 / 吞吐 | 91,693 s = 25.47 h / 3,266 窗口每秒；参考估计 21.74 h，实际慢 17% |
| 节律 | 937 s/3,072,000 窗口（前 10 个 927 s → 后 10 个 982 s），82 个模型 checkpoint + 终末 1 个 |
| 训练 HEAD | `2593162`，完成时 `git_head` 相同（无中途提交改写 provenance） |
| 终末 checkpoint | `_windows299520000.pth` sha256 `722d83ee7755b051e2095ccd01d4094bacce99589e679f89379f54661fb43704` |
| 权重口径 | `ema_decays: []`，`primary_weight_variant: online`，`gradient_clipping: false`（配方注册选择） |

**Spike 扫描**：`metrics.json["validation"]` 是 98 条逐节律记录（每条逐 loss 键，online 权重，
最敏感口径），全部 finite，`loss_finite: True`。9 处 >6× median|Δ| 的跳变里 8 处在前 11 条
（早期陡降段，其中 `contact_accuracy` 4 处是变好）；唯一中段候选是第 42 条（w=132,096,000），
`joint_rotation` +6.4× 且同为 `fk`/`joint_position`/`velocity` 各自最大正跳——四键同点是抗巧合信号，
但量级只有 **+2.7%**（对比已封存的 B 型 +266%、C 型 +11.6%）。
**分辨率上限**：1,500 次更新/点，B 型事件（约 15,000 更新恢复期）可见，C 型（约 85 更新）**不可见**；
`fk_weight` 0.357 离 C 更近，故更可能发生的正是漏掉的那一型。**准确说法是"在该分辨率下未检出"，不是"无 spike"。**
尾部 153.6M→299.5M 的 `total` +3.3% 是已知的反相关特征，不得读作过训练。

**两条通过的不变性检查**：`gt_contact_percent` 与 D2-AI 行**逐位相同**（0.6618830180474017），
说明表示修复未触及真值侧接触计算；归一化审计 4,877,568 个生成值中 0 非有限、0 越界（两路越界率 0.0）。

## 失败记录

### 评测第一次尝试的操作性失败

`p1-hoi-p12-frame-repair-baseline-eval-guided-s42-20260819`，`status: failed`，chain 退出码 2。
启动 chain 时未给 `--eval-override checkpoint_weight_variant=online`，而
`config_eval_hoi_prior.yaml:46` 默认 `ema_0.9999`，配方 `ema_decays: []` 从不保存 EMA 权重，
于是 `code/priors/hoi/models.py:591` 抛 `ValueError: HOIPrior checkpoint is missing ema_model weights`。
registry 里 38 行历史 HOI 评测全部用 `online`——先例摆在那里。

**未花 GPU 预算**：在加载模型时失败、生成之前退出，run 目录只有 4 个 hydra 配置文件、零结果。
run id 保留、不复用；由 `-r1` 重试后缀取代。失败记录另存为
`chain/evaluate.attempt1-failed-ema_0.9999.{json,log}`——**这是必要的**，因为
`write_stage_status` 按 train_run_id 写 `chain/evaluate.json`，一次 dry-run 就会把它覆写成 `skipped`。

### CONTROL cell 绕开 chain 的阶段守卫

`tools/hoi_chain.py` 的 `stage_completed` 按**阶段名**取状态文件、**不按 eval run id**，
故第二次 `--stages evaluate` 会打印 "evaluate already completed; not rerunning" 而静默跳过。
CONTROL 因此直接调用评测器，命令与 `hoi_chain.evaluate_command` 构造的逐字节相同，
只多 `+dataset.step0_frame_rule=historical_conjugated`。未改 `tools/hoi_chain.py`。

### 五次 subagent 因网关流中断死亡

每次以增量落盘指令续跑，进度保住。不影响任何科学结论，记录以说明为何 scratch 里有分段 JSON。

## 关键产物与哈希

| 产物 | 标识 |
|---|---|
| 紧凑结果 | `experiments/results/p1_hoi_p12_frame_repair_baseline_s42_20260820.json` sha256 `08bae281fa15576d7f5bdc14eb1eaa865b8498f3b5eb5b4013170c16d4c02fab` |
| 终末 checkpoint | sha256 `722d83ee7755b051e2095ccd01d4094bacce99589e679f89379f54661fb43704` |
| PRIMARY 评测 | `results/experiments/p1-hoi-p12-frame-repair-baseline-eval-guided-r1-s42-20260820/` |
| CONTROL 评测 | `results/experiments/p1-hoi-p12-frame-repair-baseline-eval-guided-control-s42-20260820/` |
| 配对 bootstrap | `.../p1-hoi-p12-frame-repair-baseline-s42-20260819/chain/bootstrap_primary_vs_control.json` |
| 冻结契约 | `code/priors/core/window_codec.py` sha256 `d545359ba124bce0…`，与 `3ded4eb` 逐字节相同 |
| 数据契约 | `a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf`（未变） |

回收：训练 33 GB / 98 checkpoint 两阶段非破坏性回传，终末 checkpoint sha256 与 worker 侧一致（一次校验通过）；
CONTROL 与 bootstrap 单独回传，`rsync --checksum --dry-run` 零差异。

## 验证命令

```
pytest tests -q --tb=short --no-header                      # 340 passed
INFBAGEL_WORKER_EXPERT=hoi pytest tests -q --tb=short       # 337 passed, 3 skipped
pytest tests -q -k "registry or governance or provenance"   # 73 passed
```

CONTROL 开关的默认路径安全性由两条互相独立的测量确立：全部 1314 个窗口的 14 个数组
`maxabsdiff` 恰为 0.000e+00；还原 HEAD 源文件后 `__getitem__` 全部 35 个键在 64 窗口上逐键
SHA256 → 35/35 相同。8 处对抗式源码回退全部被测试捕获。`code/datasets/utils.py` 是纯追加
（`current[:221]` 与 HEAD 逐字节相同），这一性质承重：训练用的 `code/priors/hoi/data.py:16`
从它 import；`code/train_hoi_prior.py` 不 import `datasets.infbagel`，已写成测试断言。

## 局限

1. **门控 (ii) 的裁决与其标签理由互相矛盾**，两者都记录在案。分类未因看到结果而改写。
2. **CONTROL 只给上界，且松约一个数量级。** 忠实的 `A` vs `B` 用本 checkpoint 做不出来。
3. **(iii) 闸门未测。** 一个真实的关节通道刚性缺陷会从这里漏过去。
4. **4 个穿透指标在 GT 侧未测。** 补它需约 15 行改动（`pose_all_gt`、`points_all_gt_48` 算了但从未被消费）
   加一次 438 序列重跑。
5. **穿透约 +13% 的回退未解释**，且 CONTROL 无法证实或排除帧规则的作用。
6. **plasticbox 后段偏航翻转聚簇未解释。**
7. **Spike 检测分辨率 1,500 更新/点**，C 型事件不可见；每步梯度范数在 HOI 上**根本没被计算**
   （唯一的 `clip_grad_norm_` 在被关掉的裁剪分支内），`log_grad_norm` 在本分支零命中。
8. **所有 D2-* 封存的模型侧几何数字作废且不可重算**，只能靠重训取代。旧行仅作"修复前"留档。
9. 训练慢于参考估计 17%，无争用证据；漂移与 checkpoint 目录增长相关，量级不重要。

## 下一入口

按预注册对 `eval-consistency-null` 写死的后果：**"评测一致性"这一半收益判负，采样效率假设仍未被检验，
下一入口是训练侧对照而非评测侧。**

预注册的诊断入口是 `end_obj_trans_err` 在 CONTROL 下的行为——它是唯一在帧规则退化下**不变差**
（点估计还略好）的目标类指标，与其余七项显著项方向相反，机制未知。

可选、需单独批准：(iii) 的导出改动、4 个穿透指标的 GT 值、plasticbox 聚簇追查。
