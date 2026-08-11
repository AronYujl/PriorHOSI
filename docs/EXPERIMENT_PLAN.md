# 状态条件 HOI/HSI Prior 组合的 HOSI 实验计划（导航页）

**2026-08-10：本计划已按渐进披露拆分。** 原来的单文件共 9,519 行，一次性载入会挤占
上下文却只有一小部分与手头的 phase 有关。现在原文被**逐字节**切进 `docs/plan/` 下的
分册（未改写、未重排、未修正笔误），本文件只保留导航。路径保持不变，因为大量已提交的
引用指向它。

拆分前的单文件版本可从 tag `archive/p1b-preclean-20260810` 取回：

```
git show archive/p1b-preclean-20260810:docs/EXPERIMENT_PLAN.md
```

## 阅读规则

> **总是读 `docs/plan/OVERVIEW.md`；再读你正在做的那个 phase 的分册。其他分册不必读。**

Phase 1B 有六个分册，先读 `docs/plan/PHASE_1B_HOI/README.md` 挑分册，不要顺序通读。
研究上下文的第一入口仍然是 `docs/HOIPRIOR_EVIDENCE_INDEX.md`；精确数值以
`experiments/results/` 下的紧凑 JSON 与 `experiments/registry.jsonl` 为准。

## 分册

| 文件 | 内容 | 原行范围 | 复制行数 |
|---|---|---|---:|
| [`plan/OVERVIEW.md`](plan/OVERVIEW.md) | 研究主张与边界、`TaskSpec`/`TaskPlan` 接口、两专家与 mixer 组合、状态化能量；阶段粒度与交接约定；Phase 1 正式训练资源协议；Phase 2–6 的范围与门槛；评测/基线/统计协议；变更日志与 fallback | 14-74、97-121、8300-8478 | 265 |
| [`plan/PHASE_0.md`](plan/PHASE_0.md) | Phase 0：治理、数据与评测闭环，及其已通过的 gate 决定 | 75-96 | 22 |
| [`plan/PHASE_1A_DATA.md`](plan/PHASE_1A_DATA.md) | Phase 1A：数据契约、232 维表示与专家脚手架，及其 gate | 122-132 | 11 |
| [`plan/PHASE_1B_HOI/README.md`](plan/PHASE_1B_HOI/README.md) | **Phase 1B 索引**：六个分册的用途、当前状态、权威来源顺序、4 卡 worker 上的下一入口 | 新写 | — |
| [`plan/PHASE_1B_HOI/01_GATE_AND_EARLY_DIAGNOSIS.md`](plan/PHASE_1B_HOI/01_GATE_AND_EARLY_DIAGNOSIS.md) | Phase 1B 立项与门槛、2026-07-14 原生 gate 失败、修复重试预注册、D0–D2-S 诊断谱系 | 133-2027 | 1895 |
| [`plan/PHASE_1B_HOI/02_FROM_RANDOM_TRAINING.md`](plan/PHASE_1B_HOI/02_FROM_RANDOM_TRAINING.md) | 从随机初始化的强训练谱系 D2-T → D2-AA（含封存对照 D2-X） | 2028-3545 | 1518 |
| [`plan/PHASE_1B_HOI/03_INTERACTION_REPRESENTATION.md`](plan/PHASE_1B_HOI/03_INTERACTION_REPRESENTATION.md) | 交互表示与关系场谱系 D2-AB → D2-AG，以及精简迭代工作流的转折 | 3546-7434 | 3889 |
| [`plan/PHASE_1B_HOI/04_BUDGET_AND_LONG_ARMS.md`](plan/PHASE_1B_HOI/04_BUDGET_AND_LONG_ARMS.md) | 预算杠杆：P4 预算-指标曲线与 D2-AI/D2-AJ 长预算双臂（4.875×） | 8479-8882 | 404 |
| [`plan/PHASE_1B_HOI/05_INFERENCE_GUIDANCE.md`](plan/PHASE_1B_HOI/05_INFERENCE_GUIDANCE.md) | 基线协议分解 P1，推理期接触引导 P2/P3/P5/P6 | 7435-7677、7934-8270、8883-9107 | 805 |
| [`plan/PHASE_1B_HOI/06_GEOMETRY_TERM.md`](plan/PHASE_1B_HOI/06_GEOMETRY_TERM.md) | 训练侧手-物几何项：D2-AH 权重恢复、P8–P9c 剂量扫描、P10 公式修复 | 7678-7933、9108-9519 | 668 |
| [`plan/PHASE_1C_HSI.md`](plan/PHASE_1C_HSI.md) | Phase 1C：HSIPrior 从零训练与原生域评测，及其 gate | 8271-8286 | 16 |
| [`plan/PHASE_1D_GATE.md`](plan/PHASE_1D_GATE.md) | Phase 1D：独立专家联合审计与 Phase 1 gate | 8287-8299 | 13 |

复制行数合计 9,506；加上被本导航页取代的原文件头部 13 行，正是原来的 9,519 行。
原文件第 1-13 行是当时的标题与状态块，是本次唯一未被复制的内容。

## 新增内容写到哪里

- 新的 dated amendment 追加到对应 phase 的分册（Phase 1B 见其 README 的主题划分），
  并在 `experiments/registry.jsonl` 追加一条 hypothesis。
- 跨 phase 的协议、门槛与统计约定改在 `plan/OVERVIEW.md`。
- 结论与"下一步别再试什么"写进 `docs/HOIPRIOR_EVIDENCE_INDEX.md`，不写在计划正文里。
