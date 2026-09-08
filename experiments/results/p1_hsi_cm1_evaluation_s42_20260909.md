# HSIPrior CM1：R2 固定 w=1 蒸馏评估

**结论：未通过保真门和有引导 20 FPS 目标，继续保留 R2+CG 工作基线。**

CM1 有引导生成加速 25.40×，无引导加速 49.12×。有引导 FID 从 40.0497 降至 22.9038，但穿透比例上升 28.3%、脚滑上升 7.8%、非穿透接触计数下降 12.8%。有引导平均 FPS 为 19.2587，目标为 20。

## 固定协议

R2 final EMA 为 teacher；student/EMA target 从 R2 初始化，teacher/target 保持既有 train-mode dropout。固定 CFG w=1，8×256、有效 batch2048、seed42、58,678 updates /120,172,544 windows。全部469,424条rank-update诊断有限，11个update触发既有clipping。最终epoch089是唯一正式质量终点；epoch004仅用于预注册的中途诊断。

Teacher 使用500-step diffusion，student使用16-step consistency；两者有引导格的几何scale均为1，分别使用对应状态更新系数。固定LINGO v3 scene-family split、两帧历史、native goal heading、同一scene/goal输入与canonical-ordinal seed；无best-of-N。质量4格共870条、5270窗口全部成功。

## 单卡计时

单RTX3090、batch1、CUDA同步。latency70选择器实际选中19条、69窗口；前5条预热，14条参与汇总。四格输入ID与窗口数一致，启动/结束GPU进程快照均为空，运行检查中其余GPU空闲。

| 模型 | 采样步数 | 平均生成 FPS | 总帧/总生成时间 FPS | 每序列生成秒 | 每序列端到端秒 |
|---|---:|---:|---:|---:|---:|
| R2 U | 500 | 2.6897 | 2.6841 | 55.8837 | 55.9780 |
| R2 + CG | 500 | 0.7622 | 0.7606 | 197.2047 | 197.3333 |
| CM1 U | 16 | 132.0633 | 131.8531 | 1.1376 | 1.2087 |
| CM1 + CG | 16 | 19.2587 | 19.3182 | 7.7647 | 7.8486 |

加速比采用同一计时集合的总生成秒之比。以上是warm generation与端到端读数；本任务无planning阶段。质量8卡分片的时间读数保持无效。

## 原生 Table3 与接触量

全部13项原生指标及接触量如下。走路130条；到达/交互245条。表面穿透为比例，保持原生量纲；floor-excluded两项保持原评估定义。

| 分组/指标 | R2 U | R2 + CG | CM1 U | CM1 + CG |
|---|---:|---:|---:|---:|
| 走路 / pene_pct_scene | 0.0351821 | 0.0292586 | 0.0366744 | 0.0349252 |
| 走路 / pene_sum_mean_floorexcl | 0.862994 | 0.835353 | 1.06373 | 0.923393 |
| 走路 / pene_sum_max_floorexcl | 10.2188 | 9.79527 | 10.7703 | 10.3498 |
| 走路 / fs_nemf | 0.379114 | 0.331667 | 0.400131 | 0.365149 |
| 走路 / contact_count | 559.664 | 551.398 | 570.438 | 562.683 |
| 走路 / contact_count_exterior | 191.131 | 244.914 | 186.273 | 196.842 |
| 交互/到达 / last_dist | 0.0271587 | 0.0293282 | 0.0287533 | 0.0289914 |
| 交互/到达 / success_last_5cm | 0.877551 | 0.877551 | 0.869388 | 0.902041 |
| 交互/到达 / pene_sum_mean_floorexcl | 18.4322 | 16.2713 | 19.9529 | 15.3212 |
| 交互/到达 / pene_sum_max_floorexcl | 48.1232 | 49.2295 | 52.9241 | 43.6298 |
| 交互/到达 / interior_jerk | 57.8375 | 68.2326 | 60.5127 | 62.3275 |
| 交互/到达 / boundary_jerk | 126.342 | 165.616 | 160.731 | 173.923 |
| 交互/到达 / contact_count | 1106.35 | 1074.39 | 1119.16 | 1083.45 |
| 交互/到达 / contact_count_exterior | 381.715 | 433.083 | 370.283 | 386.72 |
| 交互/到达 / MM-Dist | 9.23861 | 8.90005 | 7.56155 | 8.25482 |
| 交互/到达 / FID | 38.4151 | 40.0497 | 20.1952 | 22.9038 |
| 交互/到达 / R-Precision@3 | 0.415179 | 0.433036 | 0.517857 | 0.450893 |

`last_dist`为终帧28关节的最小平面到达距离，成功阈值为含边界5cm；不是三维手部接触误差。FID/MM-Dist用245条，R@3沿用224个query occurrence/148个独立sequence的冻结gallery与重采样单位。特征指标采用内部LINGO encoder。

## 有引导学生相对 R2+CG 的完整保真门

25项相对质量检查：12通过、6失败、7不确定。每项单独应用预注册5%边界：越小越好的比值95%CI上界≤1.05；越大越好的比值下界≥0.95。所有质量门、安全门与FPS门同时满足才晋级。

| 分组/指标 | student/teacher | 配对95%CI | 判定 |
|---|---:|---|---|
| full375/pen_ratio | 1.2831 | [1.21947, 1.35266] | FAIL |
| full375/pene_pct_scene | 1.10811 | [1.07975, 1.13727] | FAIL |
| full375/pene_sum_mean_floorexcl | 0.945953 | [0.86158, 1.03558] | PASS |
| full375/pene_sum_max_floorexcl | 0.902521 | [0.813468, 0.998802] | PASS |
| full375/fs_nemf | 1.07783 | [1.04748, 1.11057] | INCONCLUSIVE |
| full375/boundary_jerk | 1.04054 | [1.00183, 1.08048] | INCONCLUSIVE |
| full375/interior_jerk | 0.911039 | [0.889759, 0.933703] | PASS |
| full375/goal_planar_err_m | 0.873426 | [0.814339, 0.937574] | PASS |
| full375/contact_count | 1.01101 | [0.990708, 1.03291] | PASS |
| full375/contact_count_exterior | 0.872352 | [0.842814, 0.902611] | FAIL |
| locomotion/pene_pct_scene | 1.19367 | [1.16358, 1.22731] | FAIL |
| locomotion/pene_sum_mean_floorexcl | 1.10539 | [0.947238, 1.28978] | INCONCLUSIVE |
| locomotion/pene_sum_max_floorexcl | 1.05661 | [0.967619, 1.17298] | INCONCLUSIVE |
| locomotion/fs_nemf | 1.10095 | [1.0533, 1.15347] | FAIL |
| interactive/last_dist | 0.988514 | [0.894714, 1.08841] | INCONCLUSIVE |
| interactive/success_last_5cm | 1.02791 | [0.972973, 1.08696] | PASS |
| interactive/pene_sum_mean_floorexcl | 0.94161 | [0.855588, 1.03462] | PASS |
| interactive/pene_sum_max_floorexcl | 0.886253 | [0.789433, 0.989155] | PASS |
| interactive/interior_jerk | 0.913456 | [0.884739, 0.942097] | PASS |
| interactive/boundary_jerk | 1.05015 | [0.994503, 1.10827] | INCONCLUSIVE |
| interactive/contact_count | 1.00843 | [0.983161, 1.03672] | PASS |
| interactive/contact_count_exterior | 0.892947 | [0.857748, 0.929376] | FAIL |
| interactive/MM-Dist | 0.927502 | [0.860315, 0.998206] | PASS |
| interactive/FID | 0.571885 | [0.459767, 0.718007] | PASS |
| interactive/R-Precision@3 | 1.04124 | [0.910714, 1.19356] | INCONCLUSIVE |

均值比值区间采用GPU上的10,000次seed42配对重采样；FID采用同一245条顺序和2,000次配对重采样。普通差值由既有paired_bootstrap工具完整输出，原始逐序列结果和所有区间均保留。

## 全375条关键几何差值

| 指标 | R2+CG | CM1+CG | 差值95%CI |
|---|---:|---:|---|
| pen_ratio | 0.0214163 | 0.0274793 | [0.00502509, 0.0071216] |
| pene_pct_scene | 0.0501421 | 0.0555631 | [0.00413698, 0.00671207] |
| pene_sum_mean_floorexcl | 10.9202 | 10.33 | [-1.58347, 0.349918] |
| pene_sum_max_floorexcl | 35.559 | 32.0927 | [-7.23011, -0.148277] |
| fs_nemf | 0.276482 | 0.298 | [0.0131718, 0.0299367] |
| boundary_jerk | 159.083 | 165.533 | [0.147699, 12.5987] |
| interior_jerk | 72.4152 | 65.9731 | [-8.03684, -4.7993] |
| goal_planar_err_m | 0.0583062 | 0.0509261 | [-0.0111878, -0.00352793] |
| contact_count | 893.089 | 902.918 | [-8.19234, 29.404] |
| contact_count_exterior | 367.851 | 320.895 | [-56.9246, -36.1714] |

总contact_count包含穿透点（SDF≤5cm）；contact_count_exterior只统计0≤SDF≤5cm。因此总计数近似持平不能代表非穿透接触保持。本次后者明确下降。走路表面穿透和脚滑变差；交互最大floor-excluded穿透及窗口内部jerk改善，完整列出各方向。

## 安全统计

| 学生 | 队列 | >5g序列 | >5g帧 | 走路骨盆最低<0.6m序列 |
|---|---|---:|---:|---:|
| unguided | full375 | 0 | 0 | 1 |
| unguided | holdout355 | 0 | 0 | 1 |
| guided | full375 | 4 | 11 | 1 |
| guided | holdout355 | 1 | 2 | 1 |

冻结守卫在holdout355上使用≤8个>5g序列、≤38帧、≤2个低骨盆走路序列，U/G均通过。完整375条的失败也保留：有引导4条11帧超5g，1条低骨盆走路。holdout排除的是既有worst20清单；没有根据本次结果重新选子集。

## 固定中途诊断

epoch004，60条/364窗口，以冻结B_n60五个stratum的总体权重汇总。该读数不用于checkpoint选择。

| 指标 | R2 U | R2+CG | epoch004 U | epoch004 G |
|---|---:|---:|---:|---:|
| pen_ratio | 0.0333673 | 0.0230891 | 0.0333221 | 0.02544 |
| pene_pct_scene | 0.0607343 | 0.0521358 | 0.0608744 | 0.0542842 |
| pene_sum_mean_floorexcl | 13.438 | 11.6323 | 14.2405 | 11.6129 |
| fs_nemf | 0.293138 | 0.270862 | 0.280929 | 0.258506 |
| boundary_jerk | 126.525 | 154.463 | 139.374 | 157.56 |
| interior_jerk | 63.7029 | 72.6603 | 64.2729 | 66.1584 |
| goal_planar_err_m | 0.0472303 | 0.0582849 | 0.0513153 | 0.0588152 |
| contact_count | 983.217 | 954.986 | 983.363 | 939.904 |
| contact_count_exterior | 347.025 | 408.864 | 345.703 | 371.277 |

## 失败案例与执行记录

下列案例按有引导学生相对teacher的最大退化排序用于定位，完整375条均已保留。它们是指标读数，不作未经视觉或干预验证的机制归因。

| 指标 | sequence | teacher | student | delta |
|---|---|---:|---:|---:|
| pen_ratio | 031:002600 | 0.174551 | 0.243761 | 0.0692093 |
| pen_ratio | 031:002589 | 0.189295 | 0.24967 | 0.0603748 |
| pen_ratio | 015:000944 | 0.203577 | 0.253565 | 0.049988 |
| pen_ratio | 010:000361 | 0.0025191 | 0.0393144 | 0.0367952 |
| pen_ratio | 027:002210 | 0.0206864 | 0.0549228 | 0.0342364 |
| fs_nemf | 044:004170 | 0.671397 | 0.979084 | 0.307688 |
| fs_nemf | 031:002588 | 0.101117 | 0.38451 | 0.283393 |
| fs_nemf | 027:002236 | 0.173295 | 0.447867 | 0.274572 |
| fs_nemf | 027:002214 | 0.246431 | 0.510253 | 0.263823 |
| fs_nemf | 044:004181 | 0.179822 | 0.439911 | 0.26009 |
| boundary_jerk | 044:004219 | 221.662 | 570.591 | 348.929 |
| boundary_jerk | 044:004231 | 256.783 | 600.622 | 343.839 |
| boundary_jerk | 044:004146 | 197.154 | 524.266 | 327.112 |
| boundary_jerk | 015:000961 | 446.448 | 642.932 | 196.484 |
| boundary_jerk | 044:004128 | 174.075 | 360.534 | 186.459 |

全部GPU任务exit0。第一次internal统计将375条teacher与60条student直接配对，工具拒绝集合不一致；按预注册60条身份显式投影teacher后完成r1统计，错误日志保留，动作生成保持一次。

资源上界合计66.5238 GPU-h（包含正式训练、benchmark和本轮全部评估），低于160 GPU-h预算。验证：组件72项通过；authority450项通过、3项跳过；registry有效。

## 工件与后续

- Compact: `experiments/results/p1_hsi_cm1_evaluation_s42_20260909.json`。
- 完整比值门与计时: `results/hsi_cm1_gate_s42_20260909/summary.json`。
- 全13项及individual CI: `results/hsi_cm1_table3_s42_20260909/summary.json`。
- 差值/分层统计/失败记录: `results/cm1_evaluation_setup_20260909/`。
- 各GPU任务manifest/config/log: `results/experiments/p1-hsi-cm1-*-s42-20260909/`。
- 训练终点: `results/p1-hsi-cm1-r2-fixed-w-s42-20260908/checkpoints/p1-hsi-cm1-r2-fixed-w-s42-20260908_epoch089.pth`。

本轮以负的晋级结果结束：CM1保留为未晋级的速度/语义候选，R2+CG继续承担质量基线。一个训练seed的sequence/occurrence区间不代表跨训练seed置信度；500-step teacher与16-step student的比较同时包含权重和sampler变化。几何引导仍是外部计算，本轮不验证把CG本身蒸入无引导学生。任何新训练方向从单独的具体方案进入；本轮不合并Phase1C、不启动下一phase。
