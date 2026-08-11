# Phase 1A：数据契约、表示与专家脚手架

本文件于 2026-08-10 从 `docs/EXPERIMENT_PLAN.md` 第 122-132 行原样切出（逐字节复制，未改写、未重排、未修正任何笔误）。
导航：[总览](OVERVIEW.md)

#### Phase 1A：数据契约、表示与专家脚手架

在 `phase/01a-data` 上固化 OMOMO-only HOI 与过滤后的 LINGO HSI dataset/config contract，
实现通道 mask、相同 232 维表示、normalization/坐标审计和从零初始化断言；HSI 排除手持动态
物体窗口，只保留 locomotion、静态交互和无物体动作，并严格使用 scene-disjoint split。
执行 CPU/unit test、单卡最小 batch 和 8 卡各一次 smoke update，不做筛选性完整训练。

门槛：数据计数/filter/split hashes 固定，无 scene-family leakage；两专家无共享可学习参数且
均拒绝 released InfBaGel checkpoint 初始化；smoke loss 有限、通道 mask 正确、resolved config
和 manifest 完整。通过后总结并 tag `exp/p1a-data-v1`。

