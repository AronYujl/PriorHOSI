# Phase 1B HOIPrior 计划索引

本目录是 `docs/EXPERIMENT_PLAN.md` 于 2026-08-10 拆分后的 Phase 1B 分册。
六个编号文件是**预注册与逐次 amendment 的原文逐字节副本**，按主题而非时间归并；
它们记录"当时决定了什么"，不是结论摘要。

导航：[总览](../OVERVIEW.md) · [计划入口](../../EXPERIMENT_PLAN.md)

## 怎么读这个目录

1. **研究上下文的第一入口仍然是 `docs/HOIPRIOR_EVIDENCE_INDEX.md`**，不是本目录。
   证据索引给出已经收敛的结论、锁定对照点和"下一步不要再试什么"。
2. **精确数字以 `experiments/results/` 下的紧凑 JSON 为准。** 计划正文里的数值是当时写下的，
   已经出现过至少一次 prose 误引（见证据索引结论 16 的 P10 更正）；JSON 与 registry 是权威。
3. 只有在需要知道**某个实验当时被允许做什么、禁止做什么**时，才回到本目录读原文小节。
4. 新的 dated amendment 追加到本目录中对应主题的文件，并同步在 `experiments/registry.jsonl`
   追加一条 hypothesis（`AGENTS.md` "Experiment lifecycle"）。

## 六个分册

| 文件 | 内容 | 原 `EXPERIMENT_PLAN.md` 行范围 | 复制行数 |
|---|---|---|---:|
| [`01_GATE_AND_EARLY_DIAGNOSIS.md`](01_GATE_AND_EARLY_DIAGNOSIS.md) | Phase 1B 立项与门槛定义、2026-07-14 首次 seed-42 正式训练与 95% 原生 gate 失败、修复重试预注册，以及 D0–D2-S 的表示/坐标/采样器/优化诊断谱系 | 133-2027 | 1895 |
| [`02_FROM_RANDOM_TRAINING.md`](02_FROM_RANDOM_TRAINING.md) | 从随机初始化建立强训练谱系：D2-T 作者 update-rule 对齐、D2-U balanced objective、D2-V 十倍预算、D2-W checkpoint 前沿、D2-X FK-foot 路由（封存对照）、D2-Y/D2-Z 放大与近地门控、D2-AA Table-5 补全 | 2028-3545 | 1518 |
| [`03_INTERACTION_REPRESENTATION.md`](03_INTERACTION_REPRESENTATION.md) | 交互表示与关系场谱系：D2-AB no-slip、D2-AC 局部物体 token、D2-AD human-local BPS 坐标修复、D2-AE GPU-native 稀疏关系场、D2-AF 可靠性路由、D2-AG self-conditioned relation source，以及 2026-07-30 的精简迭代工作流转折 | 3546-7434 | 3889 |
| [`04_BUDGET_AND_LONG_ARMS.md`](04_BUDGET_AND_LONG_ARMS.md) | 预算杠杆：P4 预算-指标曲线（61.44M 处远未饱和）与 D2-AI/D2-AJ 长预算双臂（4.875×），含目标条件通路被判为第十次模型侧失败 | 8479-8882 | 404 |
| [`05_INFERENCE_GUIDANCE.md`](05_INFERENCE_GUIDANCE.md) | 基线协议分解与推理期接触引导：P1 released 协议归因、P2 引导协议对齐（Arm A/B）、P3 关系场 × 引导 2×3、P5 接触 mask 剂量-响应与 GT 上界、P6 手部子项重加权 | 7435-7677、7934-8270、8883-9107 | 805 |
| [`06_GEOMETRY_TERM.md`](06_GEOMETRY_TERM.md) | 训练侧手-物几何项：D2-AH 度量几何权重恢复（前置诊断判负）、P8/P9/P9b/P9c 权重八点剂量扫描、P10 接触铰链 × 物体 detach 的 2×2 公式修复 | 7678-7933、9108-9519 | 668 |

Phase 1B 之外：[`../PHASE_0.md`](../PHASE_0.md)、[`../PHASE_1A_DATA.md`](../PHASE_1A_DATA.md)、
[`../PHASE_1C_HSI.md`](../PHASE_1C_HSI.md)、[`../PHASE_1D_GATE.md`](../PHASE_1D_GATE.md)。

## 当前状态（截至 2026-08-10）

**1. 95% 原生 gate 于 2026-07-14 失败，此后从未重新通过。**
`p1-hoi-eval-native-r1-s42-20260714` 完成 438 序列 × 3 窗口，全部指标有限，但 object/pelvis
goal error、FS、contact P/R/F1 与 CHOIS FID/R-Precision 均远低于门槛，只有 human-object
penetration 一项过线（`01_GATE_AND_EARLY_DIAGNOSIS.md`，原 `EP:223-234`；
`experiments/results/p1_hoi_phase1b_gate_s42_20260714.json`）。此后的全部工作都在
"失败 gate 只授权其预注册诊断/fallback"这一约束下进行，Phase 1B 未 merge、未 tag
`exp/p1b-hoi-v1`。

**2. 模型侧干预连续为受控阴性。**
计划把 D2-AJ 记为**第十次**模型侧失败（`04_BUDGET_AND_LONG_ARMS.md`，原 `EP:8782`；证据索引
结论 13），把 D2-X 之外的九个 61.44M 配置记为"九次失败的 model-side 实验"（`06_GEOMETRY_TERM.md`，
原 `EP:7687`）；连同 D2-AH 的前置诊断判负，到 P6 为止**十一次干预**没有一次真正移动接触参与度
（`docs/phase_summaries/PHASE_1B_P6_GUIDANCE_SUBTERM.md:13`）。反复出现的机制是：
**加在网络上的东西会被联合训练吸收成通用残差**。任何新的模型侧提案必须先说明它为什么不会被这样吸收。

**3. 预算是唯一被测量到的有效杠杆。**
D2-AI 在 299,520,000 windows（4.875× 正式预算，预算是唯一被操纵因子）上对封存对照 D2-X
做 438 序列 paired bootstrap：**18 项指标中 9 项显著更好、0 项显著更差**；相对 released
的两个最大真实缺口 `hand_pen` `+41.3% → +2.7%`、`human_pen` `+40.0% → +1.8%`
（`docs/phase_summaries/PHASE_1B_D2AI_D2AJ.md`，
`experiments/results/p1_hoi_d2ai_d2aj_long_budget_arms_s42_20260804.json`）。
代价是接触：`contact_percent` 对 GT 的偏离从 `+45.2%` 扩大到 `+140.5%`，且对推理引导的响应变弱。

**4. HOIPrior v1 = D2-AI（用户 2026-08-10 决定）。**
选定 run `p1-hoi-d2ai-full-budget-s42-20260803`，299.52M windows / 4.875×，训练分类
`budget-positive-goal-pathway-null`。这是**用户在 2026-08-10 做出的选择性决定**，
不是任一子阶段预注册自动产生的结论——P8–P9c 的 W3（`hand_object_contact_weight=3`）
仍是封存的接触配置，P10 未选取任何新 checkpoint。该决定的正式登记（registry 行、
版本 tag、以及是否把 W3 作为 v1 的接触变体）由主 session 在用户批准后写入，本索引只作记录。

**5. 两条已经走完的路。** 推理期引导已触顶（P5/P6：即使给完美 GT 接触标签，参与度也不会
被引导创造出来）；训练侧几何项的剂量（P8–P9c 八点扫描）与公式（P10 铰链 × detach）
两条路也都走完，P10 分类 `geometry-term-repair-negative-stop`。证据现在指向
"目标函数里只有吸引子、没有排斥子"和"穿透可能由手的姿态/朝向而非掌关节距离驱动"
（`docs/phase_summaries/PHASE_1B_P10_GEOMETRY_REPAIR.md` 的"下一入口"一节，陈述为指向、不作为推荐）。

## 锁定对照点与不可越线的事实

读任何一个分册前先记住这几条，否则很容易把跨协议的数字当成模型差距
（全部出自 `docs/HOIPRIOR_EVIDENCE_INDEX.md` 第 1 节）：

- **D2-X 是封存的自主扩散对照**（`p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723`）。
  除 P2/P3 明确标注的引导格外，全部 D2-* 数字都来自未改动的官方 438 序列、三窗口、
  500 步**无引导**原生协议。
- **released InfBaGel 只是 baseline，永远不得用于初始化 HOIPrior / HSIPrior / mixer**
  （`AGENTS.md` "Locked provenance"）。它那一行是 16 步 consistency 采样 + 引导 + CFG +
  scene/object-voxel 条件，与 D2 协议在至少六个轴上不同：**不要当作协议对齐的对比**。
  2026-08-01 的 P1 分解把 `0.1331` 的 contact recall 差拆成引导 `0.0788`(59.2%)、
  16-vs-1 步 `0.0085`(6.4%)、真实模型差 `0.0458`(34.4%)。
- 推理期引导**默认关闭**，P2/P3/P5/P6 都没有把它改成生产默认，也没有选取任何 checkpoint。
- 留出（held-out）去噪验证损失与原生 rollout 指标**反相关**，不得用来决定预算、
  early stopping 或 checkpoint 选择（证据索引结论 10）。

## 权威来源顺序

1. `docs/HOIPRIOR_EVIDENCE_INDEX.md` — 研究上下文入口，锁定对照点与结论。
2. `experiments/results/*.json` — 精确数值、CI、协议与哈希的权威来源。
3. `experiments/registry.jsonl` — 假设、run、完成记录与分类；只追加。
4. `docs/phase_summaries/PHASE_1B_*.md` — 每个子阶段的实现、失败与下一入口。
5. 本目录 — 预注册原文与 amendment，用于确认"当时允许做什么"。

计划正文与本索引都**不能**覆盖 JSON/registry；出现不一致时以 JSON/registry 为准，
并把更正写进证据索引（P10 收尾时的 `end_obj_trans_err` 更正就是这样处理的）。

## 下一入口：4 卡 worker 上的 HOI 迭代

执行侧的事实是固定的，不需要每次重新发现：

- 执行机为 4×RTX 3090 worker `10.181.9.214`（SSH 别名 `infbagel-4gpu`，node01），
  仓库/环境/数据/结果都在 `/home/yujinlun/data` 下；权威机是 8 卡 `10.184.17.253`。
- worker 校验时设 `INFBAGEL_WORKER_EXPERT=hoi`；`smpl_models` 是必需的运动学资产，必须哈希校验。
  不要把 LINGO `data/dataset` 或合成的 OMOMO `Scene*` 复制进 HOI worker 快照。
- 代码只从权威机以 Git fast-forward 发布到 worker，数据快照单独传输；所有 server-to-server
  传输由 worker 发起。reportable run 运行期间 worker 不得改动源码。
- 详见 `docs/MULTI_SERVER_TRAINING.md` 与 `AGENTS.md` "Execution environment"。

流程入口是 `docs/HOIPRIOR_ITERATION_WORKFLOW.md` 的 Stage A–F，逐条如下：

1. **Stage A（只读）**：核对 checkout/branch/HEAD/clean/日期/`INFBAGEL_PYTHON`，读 `AGENTS.md`、
   该 workflow、`docs/HOIPRIOR_EVIDENCE_INDEX.md`、最近的 phase summary 与紧凑结果，然后**只**给出
   一个推荐实验（被操纵因子、预期收益、主要风险、因果诊断、训练/评测契约、被现有证据否掉的备选）。
   停下来等用户明确批准。Stage A 不得改文件、不得分配 run id、不得加载 checkpoint、不得起 GPU 负载。
2. **Stage B（一次预注册）**：批准后，向本目录中对应主题的文件追加一条 dated amendment，
   向 `experiments/registry.jsonl` 追加一条 hypothesis，锁死单一被操纵因子、对照、性能条件、
   内部诊断、native gate、停止分类与允许改动的文件范围。
3. **Stage C（一次实现）**：源码 + config + 测试 + 文档一个逻辑提交；按改动路径跑对应检查
   （targeted tests、config 解析、registry validation、`git diff --check`；共享
   model/diffusion/training/data/evaluator 改动加跑一次完整 authority suite；运行时改动加一次
   真实数据 functional smoke；影响每步计算/通信/形状/显存时加一次 full-micro-batch 基准）。
4. **Stage D–F**：`tools/experiment.py start` 起一次正式训练（run id 形如
   `p1-hoi-<variant>-s42-<YYYYMMDD>`，永不复用、永不覆盖），固定评测，一次非破坏性回收与校验，
   一条完成记录 + 一份 phase summary + 一个紧凑 JSON 结果。

**下一个科学方向由用户决定。** 按 `AGENTS.md`，失败 gate 只授权其预注册的诊断/fallback；
任何新方向都必须先有 dated plan amendment 与 registry hypothesis，才能改代码或起 GPU 负载。
