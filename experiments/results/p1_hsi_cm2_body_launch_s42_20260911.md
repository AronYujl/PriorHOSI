# CM2.2：低噪声身体终点蒸馏稳定启动

2026-09-11。稳定启动门已完成，正式训练仍在后台运行，质量尚待固定验收。Phase1C保持开放，
修正后R2+CG继续作为质量基线。

## 唯一训练改动

在原CM1配方上增加低噪声19/39/59的身体终点保真：从相同噪声状态、历史和首步场景条件，
由独立冻结eval R2构造1–3步DDIM终点，约束学生未来14帧body22（含root）的绝对FK位置。
误差单位m²，按整个batch平均。原有手足GT FK、consistency MSE、train-mode teacher/
EMA target、EMA0.95与随机流保持；新教师前向fp32并沿用TF32设置，所有几何fp32。
首步crop按原batch的temporal块索引，后续复用原生occupancy和DDIMSolver。推理CM16保持。

## 一次校准与资源准入

八卡8×256第一批真实数据，零更新，按预注册梯度范数比固定lambda=1.2518757249916617。
trunk10%上限1.373943740373，rotation-head25%上限1.251875724992，取后者。
新项与consistency的trunk梯度cosine中位数−0.1873、与原CM1总目标+0.1312；这是初始
局部耦合读数，不能据此宣称长期优化相容或质量改善。完整表见单独校准报告。

160更新满批测量完成，前32预热、后128 CUDA同步计时，1.125550秒/更新，
1819.56窗口/秒。1280条同步梯度有限且各rank逐值相同，trunk和cfg
必需梯度存在，128条新分项loss记录有限。峰值allocated 6.0820GiB，
reserved 6.9551GiB；含外部GPU0占用的最小实测余量9301MiB。

校准+测量成本0.735828GPU-h；按测量预测正式训练146.766697GPU-h，
加8GPU-h评估预留，合计155.502525<160。当前启动段wall速度对应
约156.861GPU-h的同口径总估计；初始化、checkpoint IO及后续竞争会改变实际成本。
本轮没有教师单卡时延测量。

## 正式训练与恢复验证

Run：`p1-hsi-cm2-body-train-s42-20260911`。8×RTX3090、micro256、accum1、effective2048，seed42、
Adam/lr2e-4/warmup2000/clip1/bf16_tf32，58,678更新/120,172,544窗口，90epoch上限。
student/teacher/EMA target均从已封存R2重新初始化，optimizer/RNG冷启动。配置差分核对
CM1的dataset/model/sampler/优化/预算等主字段，除新目标外仅新增无效的旧body_fk默认0。
路径按checkout根与旧../写法规范化后比较，数据链接指向相同权威输入。

前160更新的1280条逐rank诊断与独立测量全部逐值一致，最大差0。首个完整epoch0恢复点
写出于656更新：218个student状态张量、218个EMA target张量、102组optimizer参数状态、
8个rank RNG状态。学生/target严格载入、optimizer载入和恢复契约检查全部通过；epoch000
导出与恢复点student逐值一致，全部student/target张量有限。新系数、噪声阈值和教师路径
已纳入恢复契约。正式进程持续运行。

稳定审计固定前640更新，处于2000-update warmup内：5120条梯度、5120条cfg诊断、512条新项日志均有限；各rank
同步梯度一致，trunk/cfg必需梯度为正，梯度min/median/max为
0.00725411/0.14420460/0.43926346，该前缀没有触发clip。
训练段最小实测余量9301MiB。wall参考速度
1.135965秒/更新，快照预计剩余18.23小时，
预计完成2026-09-12T06:17:28+08:00（北京时间，受后续竞争影响）。

## 运行所有权与验证

训练固定在`/data/yujinlun/InfBaGel-hsi-cm2-training`，执行提交e6185a1；该工作树保持干净且固定。
数据/结果/kinematic资产通过本机链接复用；权威主工作树完成总结时，运行源码保持原值。
持久launcher位于本run目录的launcher.log/launcher.pid；实际命令与执行脚本由
`results/cm2_body_setup_20260911/formal_job.json`及`execute_job.sh`记录。
训练结束时launcher在执行工作树自动finish本地manifest，后续评估沿已批准CM2.3继续。

定向66通过，完整authority487 passed/3 skipped（86.94秒）。校准、满批测量和正式任务
各自通过实际Hydra入口完全解析，配置归档在对应run目录；registry当前423行有效。
校准/测量均exit0，校准退出的CUDA context shutdown warnings保留；正式任务仍running。
未出现reportable负载失败。工作树配置时一个额外scratch链接被Git视为未跟踪，正式
start前移除了该不需要的链接；训练输入与数据链接保持正确。

预注册edf0c6b；逻辑实现985e5be；实测系数落地e6185a1；本报告与交接由启动completion
提交记录。核心契约及其他expert路径保持原值，无Phase1C合并或sealed expert标签。

## 精确下一入口

固定内部权重：`/data/yujinlun/InfBaGel-hsi/results/p1-hsi-cm2-body-train-s42-20260911/checkpoints/p1-hsi-cm2-body-train-s42-20260911_epoch004.pth`。
最终权重：`/data/yujinlun/InfBaGel-hsi/results/p1-hsi-cm2-body-train-s42-20260911/checkpoints/p1-hsi-cm2-body-train-s42-20260911_epoch089.pth`。前者只读epoch004，后者只读epoch089；依完整预算
和既定数据/采样/统计规则执行，不能用中途loss选择checkpoint或改变预算。
下一次用户要求继续时，先检查manifest/exit_status/终点更新及上述两权重，再执行已批准
CM2.3：内部60条U/G；最终U/G各375条2271窗；完整物理、有效接触、安全、表示、FID/
Diversity/MM-Dist/R@1/2/3与配对区间；学生单3090 latency70。教师时延复用已有记录。
训练目标的有效性、生成历史迁移、语义耦合及有限CG修正的缺口，都留待这些完整验收。

工件：本报告同名JSON；run目录的manifest.json、config_resolved_job.yaml、preflight.json、
initial_reproducibility.json、initial_resume_validation.json、stable_launch.json、next_entry.json；
原始loss/梯度/RNG恢复状态在训练输出目录。训练manifest保持running，最终质量记录待完成。
