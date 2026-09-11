# R4：全未来帧 GT 身体 FK 教师验收

**R4未通过，保留修正插值后的R2+CG。** 十个表示/物理主比较中，7项经多重比较校正后明确退化、0项明确改善、3项不确定；G的FID也明确退化。U/G低骨盆行走均超过绝对守卫。

## 实验与对照

新增目标覆盖全部14个未来帧、21个非root身体关节，使用GT直接位置锚定root-relative FK，权重0.24812712291771177。位置MSE、旋转L1、手足FK权重3和seam0.5494500113254572保持R2值。GPU4–7从随机初始化完成146255更新、299530240窗口；最终EMA为唯一验收权重。旧656更新运行因用户要求从头训练而中止并保留。

U/G均为DDPM500、CFG w1，G使用posterior-coefficient CG。各375条/2271窗口，seed42，fixed_rate_endpoint_hold_v1插值。物理与表示对照使用CM1.5修正后的R2；R2在插值修正前后的粗帧输出一致，因此可复用其封存Table3。两模型245个FID样本ID、GT文本/运动embedding逐值一致。FID配对bootstrap2000次，其余10000次。R@3单位为冻结gallery的224次query occurrence（148个唯一序列）。这些区间不表示跨训练seed置信度。

用户取消教师单卡时延，本轮没有该项GPU评估。

## 联合验收

| arm | 表示 | 物理主判据 | exterior接触≥95% | FID/语义保真 | 绝对安全 | 联合 |
|---|---|---|---|---|---|---|
| unguided | FAIL | FAIL | PASS | FAIL | FAIL | FAIL |
| guided | FAIL | FAIL | PASS | FAIL | FAIL | FAIL |

## 表示与物理主比较

差值为R4−R2。body使用cm，其他保持原生单位；family10区间控制十项主比较。

| arm | metric | R2 | R4 | 差值 | 95%CI | family10同时CI |
|---|---|---:|---:|---:|---|---|
| unguided | pen_ratio | 0.0308178 | 0.0349004 | 0.00408268 | [0.00252148, 0.00579112] | [0.00187805, 0.00665882] |
| unguided | fs_nemf | 0.286728 | 0.310904 | 0.0241762 | [0.0172889, 0.0314028] | [0.0141781, 0.0346111] |
| unguided | boundary_jerk | 108.045 | 119.348 | 11.3026 | [7.71958, 14.9023] | [6.21999, 16.3546] |
| unguided | contact_count_exterior | 314.685 | 303.117 | -11.5677 | [-23.2028, 0.360701] | [-27.7579, 5.13075] |
| unguided | body21位置/FK分歧 cm | 0.794252 | 0.934823 | 0.140571 | [0.117226, 0.164108] | [0.107905, 0.17322] |
| guided | pen_ratio | 0.0218843 | 0.0257017 | 0.00381742 | [0.00248308, 0.00526896] | [0.00194872, 0.00591785] |
| guided | fs_nemf | 0.272221 | 0.330189 | 0.0579683 | [0.0500562, 0.0659092] | [0.0467796, 0.0695805] |
| guided | boundary_jerk | 142.686 | 137.586 | -5.09945 | [-11.5469, 0.90045] | [-14.5658, 3.15646] |
| guided | contact_count_exterior | 365.631 | 365.255 | -0.376269 | [-12.0818, 11.9835] | [-17.4152, 17.5917] |
| guided | body21位置/FK分歧 cm | 0.887031 | 1.03561 | 0.148583 | [0.104456, 0.197156] | [0.0864824, 0.21779] |

G的boundary jerk点估计下降但区间跨0；U/G exterior接触差值也不确定，均值达到R2的95%。同时，U/G穿透、FS和身体表示明确退化，U boundary jerk也明确退化。

## FID与语义

| arm | metric | R2 | R4 | R4/R2 | 差值95%CI | family6同时CI |
|---|---|---:|---:|---:|---|---|
| unguided | FID | 38.4151 | 47.3355 | 1.2322 | [-2.4603, 21.2512] | [-6.49551, 25.9082] |
| unguided | MM-Dist | 9.23861 | 9.78619 | 1.0593 | [-0.19699, 1.29743] | [-0.468059, 1.57387] |
| unguided | R-Precision@3 | 0.415179 | 0.361607 | 0.8710 | [-0.120536, 0.0133929] | [-0.142857, 0.0357143] |
| guided | FID | 40.0497 | 58.2128 | 1.4535 | [8.27034, 29.1822] | [5.64912, 32.1925] |
| guided | MM-Dist | 8.90005 | 9.76487 | 1.0972 | [0.232139, 1.51797] | [-0.00895555, 1.74733] |
| guided | R-Precision@3 | 0.433036 | 0.397321 | 0.9175 | [-0.107143, 0.0357143] | [-0.129464, 0.0625] |

G FID差值+18.1632，同时CI[5.64912,32.1925]；U FID区间跨0。G MM-Dist逐项95%CI大于0，但同时区间跨0；两组R@3下降的区间均跨0。两组FID/MM均值都超过R2的105%、R@3均低于95%，因此预注册点估计保真门均FAIL；并非每项都有显著退化证据。

| arm | model | R@1 | R@2 | R@3 | Diversity | MultiModality |
|---|---|---:|---:|---:|---:|---:|
| unguided | r2 | 0.223214 | 0.348214 | 0.415179 | 12.1312 | 6.70588 |
| unguided | r4 | 0.160714 | 0.28125 | 0.361607 | 11.3473 | 6.60063 |
| guided | r2 | 0.258929 | 0.357143 | 0.433036 | 12.0838 | 5.92187 |
| guided | r4 | 0.209821 | 0.308036 | 0.397321 | 10.9461 | 5.50713 |

GT Diversity14.8947802、MultiModality3.88313985。分布读数使用原生函数及“GT行先消耗随机数、再计算生成行”顺序。Diversity朝GT的距离扩大；MultiModality有反向改善的点估计，完整保留。以上为冻结内部LINGO编码器指标，与其他论文编码器的FID不可混比。

## 全部原生分组指标

以下为逐项95%CI。全375、holdout355及所有可分析字段均包含在compact的19份配对报告中。

| arm / cohort | metric | R2 | R4 | 差值95%CI |
|---|---|---:|---:|---|
| unguided/locomotion | fs_nemf | 0.366907 | 0.392094 | [0.012506, 0.0385499] |
| unguided/locomotion | pene_pct_scene | 0.0351521 | 0.0373559 | [0.00153896, 0.00293001] |
| unguided/locomotion | pene_sum_max_floorexcl | 10.2206 | 9.48855 | [-3.00299, 1.84666] |
| unguided/locomotion | pene_sum_mean_floorexcl | 0.853586 | 0.938348 | [-0.223789, 0.475397] |
| unguided/interactive | MM-Dist | 9.23861 | 9.78619 | [-0.19699, 1.29743] |
| unguided/interactive | boundary_jerk | 110.862 | 124.549 | [8.76315, 18.7554] |
| unguided/interactive | contact_count | 1111.4 | 1124.02 | [-23.9547, 52.2196] |
| unguided/interactive | contact_count_exterior | 380.216 | 371.64 | [-25.805, 8.03147] |
| unguided/interactive | interior_jerk | 51.0095 | 53.2331 | [1.03319, 3.43357] |
| unguided/interactive | last_dist | 0.0271587 | 0.0240883 | [-0.00561775, -0.000482467] |
| unguided/interactive | pene_sum_max_floorexcl | 48.7205 | 53.7721 | [-3.32239, 15.9582] |
| unguided/interactive | pene_sum_mean_floorexcl | 18.5469 | 19.6418 | [-1.28451, 3.72332] |
| unguided/interactive | success_last_5cm | 0.877551 | 0.930612 | [0.00408163, 0.102041] |
| guided/locomotion | fs_nemf | 0.322867 | 0.39397 | [0.0570626, 0.0850738] |
| guided/locomotion | pene_pct_scene | 0.029226 | 0.0332153 | [0.00295934, 0.00516617] |
| guided/locomotion | pene_sum_max_floorexcl | 9.74066 | 10.0577 | [-1.43485, 2.03658] |
| guided/locomotion | pene_sum_mean_floorexcl | 0.827463 | 0.984279 | [-0.052667, 0.377632] |
| guided/interactive | MM-Dist | 8.90005 | 9.76487 | [0.232139, 1.51797] |
| guided/interactive | boundary_jerk | 151.171 | 142.882 | [-17.4776, 0.253422] |
| guided/interactive | contact_count | 1079.11 | 1134.73 | [20.5504, 93.6172] |
| guided/interactive | contact_count_exterior | 429.565 | 432.501 | [-14.0928, 20.5937] |
| guided/interactive | interior_jerk | 65.2786 | 68.1605 | [1.08171, 4.65176] |
| guided/interactive | last_dist | 0.0293282 | 0.0415725 | [0.00862332, 0.0157311] |
| guided/interactive | pene_sum_max_floorexcl | 49.7206 | 57.4831 | [-0.704997, 17.4738] |
| guided/interactive | pene_sum_mean_floorexcl | 16.3247 | 17.5707 | [-0.806242, 3.48515] |
| guided/interactive | success_last_5cm | 0.877551 | 0.677551 | [-0.269388, -0.130612] |

U交互组末端距离和5cm到达率改善；G总接触量增加，同时G的5cm到达率0.87755→0.67755，下降20个百分点。相反方向的次级结果完整保留，主判据和安全门仍失败。

## 表示分解与中途诊断

| cohort / arm | n | body21 cm | root m | seam区域 cm | 内部区域 cm |
|---|---:|---:|---:|---:|---:|
| epoch19/gt | 60 | 0.018835 | 2.027e-08 | 0.022518 | 0.018398 |
| epoch19/r4_epoch19 | 60 | 61.020745 | 4.271e-08 | 63.717408 | 60.700326 |
| final/gt | 375 | 0.016441 | 2.132e-08 | 0.020756 | 0.015952 |
| final/r4_guided | 375 | 1.035614 | 3.699e-08 | 0.847449 | 1.058458 |
| final/r4_unguided | 375 | 0.934823 | 3.825e-08 | 0.827569 | 0.948060 |

最终GT分歧0.01644cm；R4 U/G为0.93482/1.03561cm，root误差约4e−8m，内部同样有分歧。该量衡量直接位置与部署FK的一致性；对GT的目标误差与此量含义不同。七项表示指标、28关节均值及最差案例均保留。

epoch19 EMA为13120更新/26869760窗口的固定中途快照。60条已暴露开发序列/364窗口的body分歧61.0207cm；GT60为0.01884cm。按注册预算继续到终点，诊断没有用于checkpoint选择或早停。

| cohort | t | GT首两帧FK加速度 | clamp后首两帧 | 内部首两帧 | 首未来帧pelvis误差 m |
|---|---:|---:|---:|---:|---:|
| holdout | 498 | 1.15548 | 48.1098 | 26.4291 | 0.408969 |
| holdout | 250 | 1.15548 | 43.1812 | 18.4816 | 0.379893 |
| holdout | 50 | 1.15548 | 35.2128 | 5.82475 | 0.307025 |
| train | 498 | 1.05375 | 49.8148 | 27.1177 | 0.424397 |
| train | 250 | 1.05375 | 44.7176 | 19.1614 | 0.392398 |
| train | 50 | 1.05375 | 36.3105 | 5.7967 | 0.324975 |

单步诊断含352个有效test窗和364个train窗；12个终端补齐窗按既定规则排除。既有判定为single-forward seam supported、history-clamp mechanism inconclusive、generalization not established。它描述epoch19 EMA及其预算/EMA成熟度；最终模型由最终rollout独立验收。冻结strata加权的60条诊断及全部区间见compact。

## 绝对安全与失败案例

| arm | cohort | >5g episode/frame | 低骨盆walk |
|---|---|---:|---:|
| unguided | full375 | 2/3 | 8 |
| unguided | holdout355 | 2/3 | 8 |
| guided | full375 | 6/11 | 14 |
| guided | holdout355 | 3/4 | 14 |

holdout上限为>5g最多8条/38帧、低骨盆walk最多2条（pelvis最低高度<0.6m）。加速度计数通过；U/G低骨盆walk为8/14，均失败。G最严重案例045-new_loco:009708最低pelvis0.127153m。

| arm | 低骨盆walk ID | 最低pelvis m |
|---|---|---:|
| unguided | 018-1:001041 | 0.465068 |
| unguided | 024:001759 | 0.499370 |
| unguided | 024:001763 | 0.439028 |
| unguided | 024:001795 | 0.492501 |
| unguided | 031:002577 | 0.510982 |
| unguided | 031:002608 | 0.511353 |
| unguided | 056:005755 | 0.492406 |
| unguided | 061-new_loco:009720 | 0.465283 |
| guided | 018-1:001011 | 0.518884 |
| guided | 018-1:001041 | 0.520842 |
| guided | 024:001749 | 0.519347 |
| guided | 024:001759 | 0.506594 |
| guided | 024:001763 | 0.506547 |
| guided | 024:001795 | 0.523891 |
| guided | 031:002577 | 0.505764 |
| guided | 038-bed:003178 | 0.537696 |
| guided | 044:004180 | 0.496600 |
| guided | 045-new_loco:009708 | 0.127153 |
| guided | 056-new_loco:009715 | 0.425022 |
| guided | 056-new_loco:009716 | 0.384782 |
| guided | 056:005750 | 0.502278 |
| guided | 056:005755 | 0.510259 |

全部>5g IDs/帧数/峰值、物理每项最差10例、表示最差10例在compact中。最终U/G各375条finite_motion=true、nonfinite_ratio最大值0；异常来自有限轨迹的质量退化。

## 执行、恢复与成本

| workload | GPU-h |
|---|---:|
| epoch19-teacher-forced | 0.027222 |
| epoch19-rollout | 2.551111 |
| epoch19-position-fk | 0.024444 |
| final-unguided | 14.364444 |
| final-guided | 43.720000 |
| table3 | 0.276389 |
| position-fk | 0.031111 |

七项GPU均成功，合计60.994722GPU-h<80。正式训练96.068889、用户中止原运行含暂停0.494249、校准/满批测量0.218889；R4全流程157.776749GPU-h。每项preflight记录GPU0–3外部推理占用；本轮没有教师单卡时延。

初次统计缺少冻结Diversity脚本所在目录的sys.path，出现ModuleNotFoundError: build_train_bundle。七项GPU及17份配对报告已完成；恢复按生产loader补齐同级模块路径，复用这17份报告，完成剩余2份语义配对及缺失汇总。保留原analysis.log与pipeline_exit_status=1；analysis_resume_exit_status=0及final_completion.json记录恢复后的最终状态。

此前authority476 passed/3 skipped、定向56 passed；本次运行代码保持，复用验证。另核对全部任务clean source4fa797d、完整覆盖、final EMA214参数一致、GT embedding逐值一致，主量和MM/R@3汇总CI与paired_bootstrap一致。完成登记后统一执行registry validation。

## 解释边界与后继

本次否定这条固定权重、固定预算、seed42的R4配方的晋级条件。补齐GT目标覆盖后，部署表示、物理及FID仍未改善。第79步有限巨梯度91848576已记录；本轮没有干预对照把终点退化归因于该峰值。初始单批次校准没有测出新项的全程相对梯度贡献，因此不能据此直接指定新权重或保证其他监督方案成功。

保留R2+CG，R4作为完整负结果封存。Phase1C保持开放，本轮完成R4验收。后继先审阅本报告和训练审计，任何新实验需单独具体批准；本轮不启动新训练或蒸馏。

Artifacts：同名compact JSON；experiments/results/p1_hsi_r4_fk_training_s42_20260911.json；results/r4_evaluation_setup_20260911/；各run的results/experiments/p1-hsi-r4-*-s42-20260911/；交接docs/phase_summaries/PHASE_1C_R4_FK.md。
