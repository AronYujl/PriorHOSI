# Phase 1B D2-AA：Table-5 completeness evaluation

## Scope and outcome

D2-AA 是一个对称、non-selection 的补充评估 subphase。它没有训练、resume、optimizer/EMA
恢复、checkpoint 写入或选择，也没有改变 D2-V/X/Y/Z 已完成的 success/negative gates。唯一
目标是用同一固定协议补齐四个 from-random diffusion HOIPrior checkpoint 缺失的 FID、
R-Precision@1/2/3、Matching Score、Diversity 与本地 batch-1 timing，并把论文 Table 5 列和
额外诊断严格分开。

最终 classification 为 `table5-completion-pass-nonselection-stop`。D2-V/X/Y/Z 的 native
aggregate/per-sequence 结果均精确复现；四组补充指标、uncertainty 和 timing 全部 finite。
没有发现 production sampler、native evaluator、CHOIS point-estimate formula 或任一 checkpoint
的确定 scientific implementation defect。发现并补足的是 CHOIS `drop_last=true` 的 effective
sample-count reporting omission。

## Implementation and fixed protocol

- Plan commit：`b4087c502c0c5d6345ce6af95011925f96f7f367`。
- Implementation/run commit：`82ef8f212e77042abd4d6cedfc03fff16d9756eb`。
- Umbrella run：`p1-hoi-d2aa-table5-completion-s42-20260724`，seed 42。
- Worker：`infbagel-4gpu/node01`，clean exact commit；顺序使用单张 RTX 3090。
- Candidate checkpoints：D2-V `e0705681...01a4`、D2-X `b0fa6bdd...3d51`、
  D2-Y `8734431f...b7a7`、D2-Z `44c1ff8c...2b6`，全部 final online。
- Native：official 438 sequences × 3 windows、500-step unguided diffusion；只新增
  `save_chois_eval_npz=true`，不改变生成质量路径。
- CHOIS：commit `8ec585aa...9463`、text-to-motion `72df96ec...1636`、feature checkpoint
  `a125bc15...8775`；additive metrics 10,000-replicate sequence bootstrap，FID 200-replicate
  paired sequence bootstrap，均为 seed 42。
- Timing：同一确定首序列、batch 1、3 windows/126 frames、一次 warmup 不计、CUDA synchronized。

`tools/run_hoi_d2aa_table5.py` SHA-256 为
`5727c2a8c3e262e7de133258cb46427938025f4178cbefb2f0e293df160f1fb8`；
更新后的 opt-in CHOIS adapter SHA-256 为
`1038e7e1e7dc2882a2a199396b972ef1eb05335da0877c8f05310ff5ad738b4b`；
新增测试 SHA-256 为
`ef70c1375d949a70652f3133514435bcf6dfd71f679e6707a051397ae6992827`。
新选项默认 disabled，Phase-0 point-estimate 公式与 RNG 顺序未改变。

## Results

Table-5-aligned native/embedding point estimates如下。`↓` 越低越好，`↑` 越高越好。本地固定
evaluator 只定义 R-Precision@1/2/3，不能 post-hoc 映射为论文未进一步定义的单一 `Rprec`。

| Row | Te↓ | Txy↓ | FS↓ | FID↓ | Cprec↑ | Crec↑ | Cf1↑ | C%↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Released baseline | 3.0372 | 3.9231 | 0.3334 | 0.9334 | 0.7908 | 0.7276 | 0.7273 | 0.5983 |
| D2-V | 3.6807 | 4.0103 | 0.3783 | 1.5781 | 0.7891 | 0.5853 | 0.6286 | 0.4693 |
| D2-X | 3.7402 | 4.0505 | 0.3630 | 1.7755 | 0.7881 | 0.5945 | 0.6374 | 0.4766 |
| D2-Y | 4.8506 | 3.9676 | 0.3572 | 1.9414 | 0.7973 | 0.5895 | 0.6351 | 0.4774 |
| D2-Z | 4.4567 | 3.8216 | 0.3634 | 1.9356 | 0.7993 | 0.5856 | 0.6308 | 0.4716 |

| Row | Pbody↓ | MPJPE↓ | Troot↓ | Tobj↓ | Oobj↓ | local batch-1 FPS↑ |
|---|---:|---:|---:|---:|---:|---:|
| Released baseline | 2.5893 | 11.9976 | 8.2088 | 15.7256 | 1.0202 | 17.7788 |
| D2-V | 4.1712 | 12.1224 | 8.2456 | 16.1082 | 1.0245 | 15.2809 |
| D2-X | 3.8691 | 12.0508 | 8.1701 | 15.9940 | 1.0309 | 18.1699 |
| D2-Y | 3.4353 | 12.1246 | 8.3380 | 16.3290 | 1.0266 | 18.0095 |
| D2-Z | 3.4479 | 12.2655 | 8.2608 | 16.3945 | 1.0158 | 18.0700 |

| Row | Match↓ | R@1↑ | R@2↑ | R@3↑ | Diversity |
|---|---:|---:|---:|---:|---:|
| Released baseline | 3.8229 | 0.1731 | 0.3101 | 0.4351 | 9.1489 |
| D2-V | 3.8586 | 0.1490 | 0.2837 | 0.4135 | 8.7407 |
| D2-X | 3.8798 | 0.1514 | 0.2837 | 0.4207 | 8.7810 |
| D2-Y | 3.9511 | 0.1514 | 0.2861 | 0.4159 | 8.3977 |
| D2-Z | 3.9180 | 0.1538 | 0.2933 | 0.4062 | 8.4481 |

D2-V/X/Y/Z 的 FID bootstrap 95% CI 分别为
`[1.1886,2.2535]`、`[1.3039,2.4942]`、`[1.5093,2.5745]`、
`[1.5077,2.5672]`。完整 Matching/R@1/2/3 CI、论文 Table 5 原始 7 行×15 列、额外
feet height/contact accuracy/hand penetration/penetration ratios/batch-438 throughput 均在
compact aggregate 中，避免把非论文列混进主表。

四组 native regeneration 各比较 18 个 aggregate scalars 与 5,104 个 per-sequence scalars，
最大绝对差 `0.0`、mismatch count 0。CHOIS 每组导出 438 predictions/438 GT，但 pinned
`batch_size=32, drop_last=true` 只把 416 条送入 embedding metrics；相同 22 个 dropped IDs 的
ordered SHA-256 为
`b7ddcb96dae95814e44d1df8f4fe1791c2c7930ed3ddfca55c3ea3fcde31bd15`。

科学解释不变：released checkpoint 只作 baseline，绝不能初始化 prior；四个自主训练 checkpoint
都没有成为正式 selectable HOIPrior。D2-X 仍是较平衡的描述性候选，但其预注册 paired
foot-sliding gate 失败；D2-Y 只有最低 FS 点估计，却违反 D2-X protection；D2-Z 同样为 joint
negative。D2-AA 的补充 FID 不重开或改写这些结论。

## Verification and retained failures

Authority 在实现提交前完成：

- `py_compile`；
- 7 项 D2-AA tests；
- 22 项 governance tests；
- 全量 293 tests；
- registry validation；
- `git diff --check`。

Worker fast-forward 到 exact Git object 后完成 29 项 targeted tests、registry validation、完整
resolved-config 与 same-context GPU preflight。Umbrella exit code 0，training updates 0，
optimizer/checkpoint write/checkpoint selection/consistency/HSI/Mixer 均为 false。

首次 preflight flatten 因 worker 缺少 `jq` 在 scientific workload 前失败，空
`preflight.json` SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
永久保留；有效替代为 Python 生成的 `preflight_r1.json`。原异步 start 已成功创建 manifest
后的一次重复 start 被 overwrite guard 拒绝，没有覆盖 manifest、重启 workload 或复用 run id。
这两项均是 operational evidence，不影响结果。

## Artifacts and handoff

- Compact aggregate：
  `experiments/results/p1_hoi_phase1b_d2aa_table5_completion_s42_20260724.json`，
  SHA-256 `d791c04bf1a896f4230a55e77518368cf4c5cb5c691c6ce98de65c18a87914d8`。
- Authority staging：
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2aa-table5-completion-s42-20260724`。
- Full tree：3,601 files / 132,167,442 bytes，worker/authority SHA-256 均为
  `1fa7f570f935d58adaf1baddd8db2367ae49aa05804310b0a8c2d1cc7febeb77`。
- Manifest/metrics/resolved/preflight-r1/run-local-registry SHA-256：
  `a98f6aea...50a` / `9512467b...989` / `c70286c6...495` /
  `70bee275...097` / `082f77f7...6c1`。

D2-AA 到此停止。不得选择 D2-V/X/Y/Z checkpoint、继续训练、启动 consistency、进入 HSIPrior
或 Mixer。任何下一 HOIPrior 机制仍需新 session、只读审计、核验下一个未占用 Phase 1B ID，
再作 dated plan/registry preregistration。

## Reporting appendix：论文、released、本地作者代码复现与自主 prior

用户在 D2-AA completion 后授权一项只读 reporting amendment：把论文 Table 5 的 InfBaGel
1/8/16、作者提供 released checkpoint 的本地评估、历史本地复现作者
diffusion→consistency 训练代码所得 CM e200，以及 D2-V/X/Y/Z 放入一份来源显式的整合表。
该 amendment 没有重新加载 checkpoint、运行 evaluator/GPU、训练、选择模型或改变任何 gate。

| Row / source | Te↓ | Txy↓ | FS↓ | Rprec↑ | FID↓ | Cprec↑ | Crec↑ | Cf1↑ | C%↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| InfBaGel 1 / paper | 2.93 | 4.19 | 0.36 | 0.63 | 3.41 | 0.79 | 0.64 | 0.66 | 0.52 |
| InfBaGel 8 / paper | 2.99 | 3.89 | 0.34 | 0.67 | 1.78 | 0.78 | 0.68 | 0.70 | 0.56 |
| InfBaGel 16 / paper | 3.06 | 3.70 | 0.32 | 0.67 | 0.68 | 0.78 | 0.73 | 0.73 | 0.60 |
| Released checkpoint / local | 3.0372 | 3.9231 | 0.3334 | — | 0.9334 | 0.7908 | 0.7276 | 0.7273 | 0.5983 |
| Author-code CM e200 / historical local | 3.5553 | 3.7640 | 0.3320 | — | — | 0.7901 | 0.7499 | 0.7453 | 0.6238 |
| D2-V / autonomous local | 3.6807 | 4.0103 | 0.3783 | — | 1.5781 | 0.7891 | 0.5853 | 0.6286 | 0.4693 |
| D2-X / autonomous local | 3.7402 | 4.0505 | 0.3630 | — | 1.7755 | 0.7881 | 0.5945 | 0.6374 | 0.4766 |
| D2-Y / autonomous local | 4.8506 | 3.9676 | 0.3572 | — | 1.9414 | 0.7973 | 0.5895 | 0.6351 | 0.4774 |
| D2-Z / autonomous local | 4.4567 | 3.8216 | 0.3634 | — | 1.9356 | 0.7993 | 0.5856 | 0.6308 | 0.4716 |

| Row / source | Pbody↓ | MPJPE↓ | Troot↓ | Tobj↓ | Oobj↓ | FPS↑ |
|---|---:|---:|---:|---:|---:|---:|
| InfBaGel 1 / paper | 2.83 | 11.92 | 8.88 | 15.40 | 1.02 | 1566.71 |
| InfBaGel 8 / paper | 2.61 | 11.97 | 8.47 | 15.59 | 1.02 | 60.94 |
| InfBaGel 16 / paper | 2.49 | 12.11 | 7.93 | 15.93 | 1.02 | 29.31 |
| Released checkpoint / local | 2.5893 | 11.9976 | 8.2088 | 15.7256 | 1.0202 | 17.7788 |
| Author-code CM e200 / historical local | 2.7775 | 11.8195 | 7.7883 | 15.8828 | 1.0186 | — |
| D2-V / autonomous local | 4.1712 | 12.1224 | 8.2456 | 16.1082 | 1.0245 | 15.2809 |
| D2-X / autonomous local | 3.8691 | 12.0508 | 8.1701 | 15.9940 | 1.0309 | 18.1699 |
| D2-Y / autonomous local | 3.4353 | 12.1246 | 8.3380 | 16.3290 | 1.0266 | 18.0095 |
| D2-Z / autonomous local | 3.4479 | 12.2655 | 8.2608 | 16.3945 | 1.0158 | 18.0700 |

`Author-code CM e200` 是 seed-42、8×RTX 3090、FP32、effective batch 2048 的历史作者代码
复现，最终评估为 official 438 × 3 windows、16-step consistency；但它是完整
scene-conditioned InfBaGel，不是独立 HOIPrior。原 run 被登记为 exploratory /
`reportable=false`，没有 `tools/experiment.py` manifest、per-sequence uncertainty 或 CHOIS
export，因此 FID/Rprec 不得补值；它的 full-438 descriptive throughput `321.6566 FPS` 也不能
冒充 batch-1 或论文 FPS。论文 `Rprec` 与本地 R@1/2/3 不作映射，论文 FPS 与本地 batch-1
FPS 也不作横向排名。

整合结果保存在
`experiments/results/p1_hoi_phase1b_d2aa_integrated_table_s42_20260724.json`。它逐项绑定
immutable source 与缺失原因；原 D2-AA compact SHA-256
`d791c04bf1a896f4230a55e77518368cf4c5cb5c691c6ce98de65c18a87914d8` 保持不变。
