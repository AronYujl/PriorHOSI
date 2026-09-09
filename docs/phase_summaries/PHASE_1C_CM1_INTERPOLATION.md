# Phase1C-CM1.5：部署插值修正与固定权重复评

2026-09-09完成；Phase1C保持开放。零训练更新，后继教师训练尚未启动。

## Scope and implementation

用户批准修正LERP方向和共同时间网格，对GT、R2 DDPM500 U/G、CM16 U/G固定权重复评。
code/utils.py用正确normalized LERP及向量化Torch插值；所有通道使用min(k/3,T−1)，
保持3T长度和末帧。evaluator保存粗帧local pose/transl并核对历史global_jpos；metrics5、
motion export4，interpolation_version=fixed_rate_endpoint_hold_v1。core及其他expert分支未改。
一个配置片段config_sample_hsi_interpolation.yaml复用既有入口，全部52份job配置已解析。

## Results and scientific limits

五组各375条，四模型每组2271窗；1875条coarse replay逐值一致、max difference0。
原生FID/MM-Dist/R@3输入及分数保持，教师FID U38.41515/G40.04968，CM16 U20.19524/G22.90380。
Diversity从封存embeddings按原生函数/默认245对/完整顺序RNG补算，标明为新补算的缓存统计。

body21 root-relative mean（cm，旧→新）：GT0.43238→0.01644；R2 U0.95248→0.79425，
G1.04784→0.88703；CM16 U1.72761→1.64894，G1.72400→1.65606。五个变化的Bonferroni
同时区间均排除0。修正后R2−GT为U+0.77781 CI[0.71471,0.84782]、G+0.87059
[0.80487,0.94276]；CM16−R2为U+0.85469[0.82212,0.89028]、G+0.76903[0.73004,0.80930]。
GT floor保持非零，teacher与student额外分歧仍明确。插值两项联合干预的单独贡献未分离，
CM/R2仍有权重及sampler family混杂；数据不支持把全部student gap归于单一训练机制。

五组FS和boundary jerk下降，pen_ratio上升，20项物理主对照的对应同时区间均排除0。
exterior contact均值均下降，其中R2 U/G、CM16 G同时区间排除0。total contact均值略增。
修正后CM16 G−R2 G：pen +0.006248，FS +0.015929，exterior contact −45.6692，逐项CI
均明确退化；boundary jerk +3.0602 CI[−2.7869,8.9657]不确定，interior jerk改善。
完整13项、额外接触、全部7项表示量和28关节分解、36份配对报告保留于报告/compact。

绝对holdout355守卫均通过。R2 G full375有11条/16帧>5g，holdout6条/8帧；CM16 G
旧4条/11帧→新7条/15帧，holdout旧1/2→新4/8。CM16 U/G各保留1条低骨盆walk。
全部高加速度、低骨盆及最大增减案例保留；从root translation定位的高加速度帧均不在末帧
保持区。平滑均值改善不等于尾部风险同步下降。

## Validation, failures and resources

Authority472 passed/3 skipped（86.70s）；CUDA初始化恢复后定向41 passed（7.04s）。
运行入口为tools/experiment.py start → test_infbagel_lingo_hsi.py → tools/paired_bootstrap.py。
Bootstrap10000、seed42、sequence-paired；主量分别以family5/20报告同时CI。

首个GT p1-hsi-interpolation-gt-s42-20260909在显存统计早于CUDA初始化时失败，2秒、
0个episode；失败manifest已封存，恢复使用gt-r1新id。正式成功六项耗费62.224167 GPU-h，
含该失败总62.224722；低于教师结果完成前按实测修订的64上限（初始48估计偏低）。
GT单卡batch128 warm forward5.838ms/21927frames/s；GT插值1.233秒包含初始调用。
最终八卡表示重建30秒，起始manifest已记录7卡有其他进程；记录其竞争下耗时，保持独占
latency结论与之分开。R2 G启动时无外部进程，缺少连续遥测以精确分摊后续竞争成本。

提交：d8b42e1 prereg →97bda1c implementation →359ae11 CUDA恢复 →96a6cbe资源治理。
359ae11与96a6cbe只在计划/registry不同；R2 U按既有explicit transition完成封存。
六个成功manifest均已在clean source上终结，再统一登记。完成提交可通过本文件Git历史定位；
本轮诊断闭环，未合并整个Phase1C、未创建expert sealed tag。

## Artifacts and exact next entry

- 报告：experiments/results/p1_hsi_cm1_interpolation_s42_20260909.md及同名.json。
- 运行索引：results/cm1_interpolation_setup_20260909/runs_r1.json；原失败GT索引runs.json。
- 完整统计、失败案例、预算transition及日志：results/cm1_interpolation_setup_20260909/。
- 表示记录：results/hsi_interpolation_position_fk_s42_20260909/，五组各375。
- 每个run目录含manifest、resolved configs、preflight、日志、completion；checkpoint及输入身份
  沿用封存manifest引用，原始motion/per-sequence不入Git。

新session先读本总结、docs/plan/OVERVIEW.md及PHASE_1C_HSI.md的CM1.5节。
后继只审阅一个全未来14帧的GT锚定body21 FK位置目标：预测/GT各自去root后xyz MSE，
保留现有position/rotation/手足FK/seam约束，并保留首个未来帧可学习。目标覆盖缺口已定位，
训练对物理及FID的效果未验证；先做单次量级/梯度校准，再提出固定预算从随机训练方案。
R3 hard-rebase和大输出相减负结果继续约束后继，纯两预测头self-consistency不能替代GT锚定。
本轮没有选择新系数、分配新训练run id或开始训练；具体训练实验需另行批准。
