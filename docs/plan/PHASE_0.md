# Phase 0：治理、数据与评测闭环

本文件于 2026-08-10 从 `docs/EXPERIMENT_PLAN.md` 第 75-96 行原样切出（逐字节复制，未改写、未重排、未修正任何笔误）。
导航：[总览](OVERVIEW.md)

### Phase 0：治理、数据与评测闭环

- 从锁定提交建 `research/state-compositional-priors`；禁止旧 feature。
- 跟踪本计划、`AGENTS.md`、registry、split/task manifests 与 artifact hashes。
- 固定 469 条 Atomic-HOSI 参考：82.09% completion、4.57/8.17 cm pelvis/object error、
  0.14 FS、0.781 contact、23.34 FPS。该值须重新运行后才算通过。
- 锁定 CHOIS 官方 evaluator，补 Table 5 FID 与 Top-1/2/3 R-Precision，记录 upstream、
  checkpoint、输入转换哈希。
- 分报 warm generation、LLM planning、end-to-end latency；CUDA 计时显式同步。
- LINGO 按 scene family 分组，mirror/new-loco/action 变体同侧，seed 42 固定 80/20。
- 8×3090 上从 micro-batch `{32,64,128}` 选最大稳定值，以累积固定 effective batch。

通过：Atomic-HOSI 在论文/复现容差内；HOI 全指标可重复；data/evaluator/checkpoint hash
完整。缺数据、官方 evaluator checkpoint 或真实 GPU 运行时不得宣称通过。

Phase 0 gate 决定：通过。469-case Atomic-HOSI、完整 HOI 原生/CHOIS 指标、batch=1 timing、
数据/evaluator/checkpoint hashes、LINGO split 与 8 卡 micro-batch 决策均已形成可复现闭环。
Phase 0 当时锁定的 smoke 配置为每卡 micro-batch 128、8 卡、accumulation 1、global
effective batch 1024。该结果继续作为历史 smoke/容量证据；2026-07-13 的训练资源协议修订
不追溯修改 Phase 0，但覆盖其对 Phase 1 正式训练的跨专家约束。完整历史证据见
`docs/phase_summaries/PHASE_0.md`。

