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

**本分支（`phase/01c-hsi`）只保留 HSI 侧。** Phase 1B 的六个 HOI 分册、
`docs/HOIPRIOR_EVIDENCE_INDEX.md`、`docs/HOIPRIOR_ITERATION_WORKFLOW.md` 和
`docs/phase_summaries/PHASE_1B_*.md` 都只存在于 `phase/01b-hoi`，本分支已按
`AGENTS.md`「Concurrent expert branches」删除，以免 HOI 的迭代史挤占 HSI 的上下文。

做 HSIPrior 时**不要**去另一条分支取那些文件：可迁移的结论已经一次性写进
`docs/HSIPRIOR_DESIGN_PRIORS.md`，取别的内容属于跨分支通信，需先获得用户批准。
若确有必要查阅原始记录，它们在 `git show phase/01b-hoi:<path>` 与
tag `archive/p1b-preclean-20260810` 下都可读。

精确数值仍以 `experiments/results/` 下的紧凑 JSON 与 `experiments/registry.jsonl`
为准——这两者是共享历史，两条分支都完整保留。

## 分册

| 文件 | 内容 | 原行范围 | 复制行数 |
|---|---|---|---:|
| [`plan/OVERVIEW.md`](plan/OVERVIEW.md) | 研究主张与边界、`TaskSpec`/`TaskPlan` 接口、两专家与 mixer 组合、状态化能量；阶段粒度与交接约定；Phase 1 正式训练资源协议；Phase 2–6 的范围与门槛；评测/基线/统计协议；变更日志与 fallback | 14-74、97-121、8300-8478 | 265 |
| [`plan/PHASE_0.md`](plan/PHASE_0.md) | Phase 0：治理、数据与评测闭环，及其已通过的 gate 决定 | 75-96 | 22 |
| [`plan/PHASE_1A_DATA.md`](plan/PHASE_1A_DATA.md) | Phase 1A：数据契约、232 维表示与专家脚手架，及其 gate | 122-132 | 11 |
| `plan/PHASE_1B_HOI/` | **不在本分支**：Phase 1B HOIPrior 迭代史（六个分册）。见 `phase/01b-hoi` | — | — |
| [`plan/PHASE_1C_HSI.md`](plan/PHASE_1C_HSI.md) | Phase 1C：HSIPrior 从零训练与原生域评测，及其 gate | 8271-8286 | 16 |
| [`plan/PHASE_1D_GATE.md`](plan/PHASE_1D_GATE.md) | Phase 1D：独立专家联合审计与 Phase 1 gate | 8287-8299 | 13 |

复制行数合计 9,506；加上被本导航页取代的原文件头部 13 行，正是原来的 9,519 行。
原文件第 1-13 行是当时的标题与状态块，是本次唯一未被复制的内容。

## 新增内容写到哪里

- 新的 dated amendment 追加到对应 phase 的分册（Phase 1B 见其 README 的主题划分），
  并在 `experiments/registry.jsonl` 追加一条 hypothesis。
- 跨 phase 的协议、门槛与统计约定改在 `plan/OVERVIEW.md`。
- 结论与"下一步别再试什么"写进 `docs/HSIPRIOR_DESIGN_PRIORS.md`，不写在计划正文里。
