# Phase 1D：独立专家联合审计与 Phase 1 gate

本文件于 2026-08-10 从 `docs/EXPERIMENT_PLAN.md` 第 8287-8299 行原样切出（逐字节复制，未改写、未重排、未修正任何笔误）。
导航：[总览](OVERVIEW.md)

#### Phase 1D：独立专家联合审计与 Phase 1 gate

在 `phase/01d-gate` 上不新增模型方向，仅汇总 single-seed-42 的最终专家结果，验证 checkpoint
provenance、参数不共享、各专家内部训练预算/effective batch 一致性、processed-window/frame
预算、完整 hash 和统计协议；补做预注册的
失败分层与专家不确定性对比，形成进入组合前的不可变 expert contract。联合 contract 还必须
证明两专家接受同一 232-D history、输出同一当前窗口坐标下的 clean x0、使用同一 codec 完成
global/local round-trip，并且组合不需要任何可学习或 expert-specific coordinate adapter。

门槛：1B/1C 均通过各自 95% 原生域门槛，且不存在系统性 contact/penetration/FID 退化；否则
Phase 1 不合入，不进入 Phase 2。通过后写 `PHASE_1D.md`，合入研究分支并 tag
`exp/p1-priors-v1`。

