# Phase 1B-03：交互表示与关系场谱系（D2-AB → D2-AG）

本文件于 2026-08-10 从 `docs/EXPERIMENT_PLAN.md` 第 3546-7434 行原样切出（逐字节复制，未改写、未重排、未修正任何笔误）。
导航：[总览](../OVERVIEW.md) · [Phase 1B 索引](README.md)

#### 2026-07-25 Phase 1B D2-AB predicted-support no-slip objective 预注册

D2-V 已证明固定的 232-D representation、architecture、conditions、balanced losses 与
61,440,000-window budget 能从随机初始化学到强 diffusion HOIPrior；D2-W 排除了通过中间
checkpoint selection 解决 foot sliding。D2-X 将足部 x/z temporal gradient 路由到
predicted root/rotations→FK 后，official foot sliding 点估计改善但 paired 95% CI 跨 0。
D2-Y 的固定 1,024 倍 amplification 显著降低 internal routed residual，却没有显著改善
official foot sliding，并破坏 end-object/contact protection；D2-Z 的 immutable-GT binary
near-ground gate 保留 D2-Y 约 78--86% gradient norm、cosine 约 0.95--0.99，仍未改善
official foot sliding。D2-Y/Z 的 routed-foot 与 FK gradient cosine 在 `t=0` 为正、在
`t=249/499` 为负，支持“当前 teacher-forced FK temporal target 在 noisy/mid timesteps
与 reconstruction/FK/object capability 冲突”，但尚未检验由 predicted state 本身定义的
物理支撑与 no-slip target。

用户在严格只读审计后明确批准执行本 subphase。registry 全量扫描确认 D2-AB 是下一个未使用的
Phase 1B identifier。本 campaign 最多允许两次新的完整 from-random training；D2-AB 占用第一
次且当前只授权这一次。第二次只可能是下文严格条件触发、另行 dated amendment 和用户明确授权的
fallback；不得自动启动。

1. **唯一 manipulated factor 与运行身份。** 正式训练 run id 固定为
   `p1-hoi-d2ab-predicted-support-no-slip-s42-20260725`、subphase `1B-D2-AB0`、seed 42。
   唯一改变是用下述 predicted-state/contact-aware differentiable no-slip residual 替换
   D2-X velocity tensor 中 joints `[7,8,10,11]` 的 8 个 x/z routed residual slots。其余
   79 slots、87-slot global mean、global velocity weight `0.1`、reconstruction/FK/object
   surface/terminal-goal losses、data、conditions、model、diffusion 与 production sampler
   全部不变。该机制不调用 official floor helper、DBSCAN、official ankle/toe thresholds、
   official foot-sliding reduction 或 test-set statistics，因此不是 evaluator-threshold trick。
2. **固定 from-random training contract。** 保持 232-D、16 frames、2 history frames、
   500-step clean-x0 diffusion、512-wide/16-head/8-layer scene-free HOIPrior、固定
   seed-42 split、4×RTX 3090 `infbagel-4gpu/node01`、per-GPU batch 512、effective batch
   2,048、accumulation 1、61,440,000 processed windows、30,000 optimizer updates、FP32
   Adam betas `(0.9,0.999)`、LR `1e-4`、无 warmup/scheduler/weight decay/gradient clipping/
   AMP/EMA。loss weights 固定为 FK `0.3569973401779424`、object surface
   `0.4772322188400037`、velocity `0.1`、terminal object goal `1.0`。released、author、
   D2-V/X/Y/Z、consistency 或其他 prior checkpoint 均不得加载；`init_checkpoint`、
   `weight_init_checkpoint` 和首次正式启动的 `resume_checkpoint` 必须为空。训练必须由
   `tools/experiment.py start` 在 clean worker exact committed Git object 上创建 manifest。
3. **train-only support metadata。** 对固定 train split 的 4,088 条 raw 30-Hz immutable
   aligned sequences，令
   \[
   f_s=Q_{0.05}^{linear}\{y^{GT}_{s,t,10},y^{GT}_{s,t,11}\}.
   \]
   pooled strictly-positive clearance 是所有 train sequences、四个 joints
   `[7,8,10,11]` 的 `y-f_s>0` 值；其固定 median 为
   \[
   \ell=0.03925712490454316\ {\rm m}.
   \]
   train floor min/median/max 必须为
   `-0.004783304338343441 / 0.0353932767175138 / 0.06221588589251041 m`。
   metadata 必须绑定 split、`human_joints_aligned.npy`、`start_idx.npy`、`end_idx.npy`、
   `norm.npy` hashes、ordered train sequence indices、per-sequence floors、positive count、
   quantiles 和算法；不得读取 official test。internal-validation loss 可对其自身 immutable
   sequence 仅用相同公式即时计算 floor，但 `ell/kappa` 只能来自 train metadata。
4. **公式级 intervention。** predicted foot positions 来自 model predicted clean x0：
   denormalize root/direct positions，decode rotations，经同一 24-joint FK 得到
   \(p^\theta\)；GT foot positions \(p^{GT}\) 来自 denormalized clean direct-position channels。
   左/右 foot pairs 分别为 \(J_L=\{7,10\}\)、\(J_R=\{8,11\}\)。对作为 residual previous
   state 的位置定义
   \[
   d_{t,q}=-\ell\log\left({1\over |J_q|}
   \sum_{j\in J_q}\exp(-(p^\theta_{t,j,y}-f_s)/\ell)\right),\qquad
   s_{t,q}=\sigma(-d_{t,q}/\ell).
   \]
   第一个 future residual 的 previous state 使用 immutable GT history frame；后续 previous
   state 使用前一 predicted FK state。固定 sampled-frame interval \(\Delta t=0.1s\)：
   \[
   v^\theta_{t,j}={\Pi_{xz}(p^\theta_{t,j}-\bar p^\theta_{t-1,j})\over0.1},
   \qquad
   v^{GT}_{t,j}={\Pi_{xz}(p^{GT}_{t,j}-p^{GT}_{t-1,j})\over0.1},
   \]
   \[
   r^{AB}_{t,j}=v^\theta_{t,j}-
   (1-s_{t-1,q(j)})v^{GT}_{t,j}.
   \]
   OMOMO position ranges are locked as
   `Rx=6.658331632614136 m`、`Rz=6.975271224975586 m`，固定 scalar
   \(\kappa=0.029363068377844033\,s/m\)。8 个 routed slots 使用
   \(L^{AB}_{foot}=\operatorname{mean}\|\kappa r^{AB}\|_2^2\) 对应 element errors；
   其余 79 element errors沿用 D2-X，最后仍对全部 87 slots 和 14 residual frames作一次
   unchanged mean。没有 multiplier、threshold sweep、contact-label gate、SNR weighting、
   gradient projection、rollout exposure 或 sampler intervention。
5. **cheap pre-training diagnostic 与 fail-closed tests。** 在任何 GPU workload 前，CPU-only
   metadata builder 必须复现上述 4,088 sequence/count/floor/clearance/range/constants，且
   synthetic 与真实 train-batch tests 必须证明：first-future previous state 为 immutable GT；
   later previous state 为 predicted FK；support 对 predicted foot height 可微、floor/GT
   stop-gradient；只有 8 个 routed x/z slots 被替换；direct foot x/z 不获得该分量梯度而
   root/rotation 获得有限非零梯度；`s→0` 时 routed residual 退化到 scaled physical
   D2-X target、`s→1` 时退化到 zero-slip target；无 checkpoint load。任一 metadata/hash/
   shape/gradient/contract 失败即停止，不启动 GPU。
6. **registered GPU smoke。** run id
   `p1-hoi-d2ab-gpu-smoke-s42-20260725`，只在 clean exact worker commit 的 `cuda:0`
   上读取固定 real-data batch 8，覆盖 timesteps `0/249/499`，执行一次 random-initialized
   forward/backward。不得创建 optimizer、optimizer update 或 checkpoint；必须记录全部
   losses、support occupancy/quantiles、关键 root/rotation/model gradients、CUDA-synchronized
   peak memory/headroom、initial model hash 和 0 checkpoint load/write。四卡必须可见且无
   compute contention。smoke 任一非有限/零关键 gradient、support collapse、hash/host contract
   失败即保留 artifacts 并停止正式训练。
7. **训练稳定性与停止轮询。** 正式 detached run 必须先通过 resolved-config、same-context
   machine preflight、finite losses/required gradients 的初始稳定区间、注册显存 headroom 和
   至少一个可实际 resume 的 checkpoint。通过后报告 measured throughput、ETA 和 checkpoint
   hash，停止连续轮询并等待用户通知；tunnel interruption 不允许重启、复用 run id 或覆盖。
   正式预算无论结果如何不得延长、选择中间 checkpoint 或 resume D2-V/X/Y/Z。
8. **fixed internal diagnostic。** 训练完成后 run id
   `p1-hoi-d2ab-predicted-support-no-slip-internal-s42-20260725`，使用 sealed
   32-sequence/96-window internal cohort、相同 clean windows/noise/dropout、final-online
   D2-X control 与 D2-AB target，在 timesteps `249`、`499` 报告每 sequence
   \[
   M=\operatorname{mean}_{t,j}s_{t-1,q(j)}\|v^\theta_{t,j}\|_2^2
   \]
   及 support mass/height/no-slip residual。每个 timestep 对 `D2-X minus D2-AB M` 做
   seed-42、10,000-replicate sequence bootstrap；internal mechanism gate 要求两个 CI
   下界均 `>0`。support sanity 还要求 target/control mean support-mass ratio 的 paired
   95% CI 完全落在 `[0.80,1.20]`，避免通过抬脚关闭 support。该 diagnostic 不选择 checkpoint。
9. **固定 native evaluation 与 registered uncertainty。** run id
   `p1-hoi-d2ab-native-eval-s42-20260725`，只加载 D2-AB fixed final-online checkpoint，
   执行一次 official 438 sequences×3 windows、500-step unguided production diffusion；
   CFG/dynamic perception/guidance/scene/consistency 均关闭。primary paired control 是封存
   D2-X final-online aggregate/per-sequence records，checkpoint/aggregate/per-sequence
   SHA-256 分别为
   `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51` /
   `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
   `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`，
   不重新生成 control。所有 paired metrics 使用 sequence 为单位、seed 42、10,000 bootstrap；
   penetration 固定使用 D2-X evaluator 已封存的同一 181-sequence finite mask。FID、
   Matching、R-Precision@1/2/3、Diversity 和 timing 若 evaluator 正常产生必须原样保留和
   报告，但 FID/R-Precision 当前不参与选择，不得删除或反向调 evaluator。
10. **gates 与 classification。** official mechanism gate 要求 `D2-X minus D2-AB`
    foot-sliding paired-difference 95% CI 下界 `>0`。protection gate 要求 D2-AB/D2-X 的
    MPJPE、end-object、Txy、object translation、hand penetration、human penetration paired
    ratio CI 上界均 `<=1.10`，contact-F1 difference CI 下界 `>=-0.02`。absolute released
    baseline gate 要求 MPJPE/end-object/Txy/object translation/FS ratios
    `<=1.30/2.00/1.50/1.50/1.10` 且 contact F1 `>=0.60`。internal、official foot、
    protection 和 absolute gates 全通过才分类
    `predicted-support-no-slip-positive-candidate-stop`；internal 失败为
    `predicted-support-no-slip-optimization-negative-stop`；internal 通过但 official foot
    失败为 `predicted-support-no-slip-transfer-negative-stop`；official foot 通过但 protection
    失败为 `predicted-support-no-slip-conflict-negative-stop`；mechanism/protection 通过但
    absolute gate 失败为 `predicted-support-no-slip-positive-but-not-effective-stop`；
    lifecycle/hash/support contract 失败为
    `predicted-support-no-slip-contract-failure-stop`。无论正负均停止，不选择 checkpoint、
    不延长预算、不自动授权 consistency。
11. **artifact contract。** 保留并双端 hash 验证：dated plan/registry、support metadata 与
    builder resolved record、logical implementation commit、authority CPU/test logs、worker
    pull/preflight/resolved configs、smoke manifest/log/metrics、formal manifest/train logs/
    training state、全部 checkpoints 与 per-rank RNG、validation/training metrics、resume
    demonstration、internal diagnostic manifest/per-sequence bootstrap、native evaluation
    manifest/aggregate/per-sequence/resolved config/logs、optional FID/R@/timing、run-local
    registry、dependency/hardware/data/evaluator hashes、完整 recovered artifact tree 和所有
    operational/scientific failures。大 artifacts 不进 Git；Git 只记录 code/config/tests、
    metadata、compact result、phase summary 与 hashes。
12. **最多一次条件性 fallback，当前不授权。** 只有 D2-AB internal gate 与 support sanity
    均通过、native transfer/protection 结果支持 late-timestep conflict、且 D2-AB final
    no-slip/protection gradient cosine 在 `t=249/499 <= -0.2`、`t=0 >= +0.2`，同时不存在
    support collapse 或 contract failure，才可提出 local objective-gradient projection：
    只投影 no-slip gradient，其他 objective gradients 不变，并必须再次从随机初始化。
    触发时须重新扫描 unused Phase 1B identifier、追加 dated plan/registry、取得用户新的明确
    授权；不得 resume D2-AB。本次批准不包含该第二次训练、consistency、HSIPrior 或 Mixer。

#### 2026-07-25 D2-AB CPU smoke-contract amendment（未启动 workload）

在 GPU/worker workload 之前的 authority CPU 审计中，精确复现 seed-42、batch-8、随机
初始化的 D2-AB smoke forward 得到 support mean `0.0266431123`，但 support 分布并未
collapse（约 `7.6%` entries `>0.05`，且同时存在低/高 support）。原 smoke 工具把
`mean >= 0.05` 当作 collapse gate，会错误拒绝该合法随机初始化；这是 operational
gate 实现缺陷，不是 scientific negative，也未消耗 smoke run id、创建 manifest、加载
checkpoint 或启动 GPU。

修正仅将 smoke 的 support 检查改为方向中性的非退化 contract：所有值有限，分布宽度
大于 `1e-3`，至少 `5%` entries `>0.05` 且至少 `5%` entries `<0.95`；同时记录
`q05/median/q95` 与两侧 occupancy。训练公式、数据、预算、native/internal gates、
run id 和 fallback 授权均不变。修正后仍必须重新运行专项/全量 CPU tests，再在 clean
exact commit 上执行原注册的 `p1-hoi-d2ab-gpu-smoke-s42-20260725`。

同一 CPU import 审计还发现 internal diagnostic 错把不存在的
`priors.optimizer_reset.paired_ratio` 作为 support-sanity bootstrap；在任何 checkpoint
load/evaluation 前已改为同模块既有的 `paired_mean_ratio`（ratio of paired-resampled
means，seed 42、10,000 replicates），与预注册统计量一致。该修正不改变 cohort、metric、
gate 或训练，并须由 CLI import test 覆盖。进一步的 CPU schema test 固定 96 windows
按每窗口全 residual/joint 维均值后再按三窗 sequence 聚合，并验证 aggregate 输入使用
`{"timesteps": ...}` 包装，避免把 per-window tensor 当作 per-sequence scalar。
native-evaluation wrapper 的 CPU resolver test 还必须在 monkey-patch shared D2-X
evaluator 后成功生成 D2-AB resolved record；wrapper 必须调用预先保存的 shared resolver，
不得递归调用已被替换的自身函数。该项仍是 pre-workload operational contract，不改变
任何 evaluator formula、metric 或 gate。

#### 2026-07-25 D2-AB continuous detached training lifecycle amendment（用户确认）

此前 D2-AB 首段使用 `pause_after_windows=3072000` 的目的仅是形成并审计一个可恢复
checkpoint；它不是科学停止点、不是第二次训练预算，也不是要求 Codex 等待后再决定是否
继续。该 checkpoint、paused state、resume evidence 和原始日志均为不可变 operational
证据，不能覆盖或删除。

本 amendment 将 D2-AB 的执行约束改为连续 detached training：

1. **同一 lineage 继续。** 用户已明确确认从
   `p1-hoi-d2ab-predicted-support-no-slip-s42-20260725` 的
   `3072000`-window checkpoint 继续。resume 必须使用同一 Git object
   `3fce4767111f7b4c01b5c2af252f6c3ef362cf43`、同一 run id、同一 seed 42、同一
   232-D representation、architecture、conditions、loss、optimizer、effective batch
   和固定 `61440000` processed-window budget；这不是新的 from-random training，也不占用
   第二机制预算。
2. **唯一 continuation config。** continuation 的 `resume_checkpoint` 必须绑定
   `ceb73ebc3a72d6290fc63e2546533c1565912b905a980381d406ea71b39a2ecc`，且
   `pause_after_windows: null`、`max_processed_windows: 61440000`。不得加载 released、
   author、D2-V/X/Y/Z、其他 prior、EMA 或任何不同 run 的 state；不得修改预算、改变
   checkpoint cadence，或选择中间 checkpoint。
3. **manifest 与 artifact 不覆盖。** 原始 `tools/experiment.py start` manifest、
   初始 resolved config、首段 train log、paused state 和 checkpoint 保持原样。因为 run id
   已经被使用，continuation 不创建第二个 manifest，也不复用/覆盖文件；必须新增
   `resolved_config_resume.yaml`、resume command、same-context resume preflight、
   `resume_initial_stability` 和 `resume` log/exit artifacts。完整训练结束后才对原 manifest
   执行一次 `finish/register`，同时绑定初始与 continuation 两套配置和完整 artifact tree。
4. **Codex 监测边界。** resume 启动后，Codex 只检查一个初始稳定区间：resolved config
   无 unresolved interpolation、四卡无 CUDA compute contention、进程持续运行、trainer
   的 finite loss/required-gradient fail-closed 检查未触发，以及注册的显存余量保持。形成
   稳定快照后立即报告已测 resume 吞吐、总训练耗时估计和 ETA，并停止主动轮询；训练必须在
   worker-owned persistent session 中继续运行。停止轮询不等于停止、暂停、杀进程或创建
   新 checkpoint。tunnel/access interruption 也不授权重启、复用 run id 或覆盖结果。
5. **完成后的固定动作。** 训练仍须跑满 `61440000` windows/`30000` updates；无论中途
   观测到的 validation/loss 如何，均不得提前选择 checkpoint、延长预算或启动
 consistency。只有完整训练结束并完成 artifact recovery 后，才运行已注册的一次 internal
 diagnostic 和一次 native evaluation；本 amendment 不授权 fallback、consistency、
 HSIPrior 或 Mixer，也不改变任何 scientific gate、uncertainty 或 FID/R-Precision
 保留规则。

#### 2026-07-25 D2-AB resume provenance guard amendment（operational）

在 continuation workload 启动前发现：正式 trainer 原本要求
`checkpoint.git_commit == current HEAD`。首段 checkpoint 绑定
`3fce4767111f7b4c01b5c2af252f6c3ef362cf43`，而本 lifecycle amendment 的治理 commit
已经是不同 object；若直接 resume，trainer 会在 GPU DDP 初始化后 fail closed。这是可确定的
provenance/lifecycle 不兼容，不是模型、loss、数据或 evaluator defect；该次未启动
resume GPU workload，旧 checkpoint 未被改写。

允许的修复仅是一个 hash-bound continuation provenance guard：resume config 必须显式绑定
checkpoint source commit、worker target HEAD、两者 `git diff --binary` SHA-256 和固定的
changed-file allowlist（仅本次治理/配置/guard/test 文件）；trainer 仍拒绝所有未显式绑定或
包含其他 source/config/model 变化的 commit transition。exact-commit resume 及所有新训练
仍保持原有 fail-closed 规则。该 guard 不改变 forward、loss、optimizer、sampling、预算或
任何 scientific gate；其 source/target/diff hash 必须进入 continuation resolved config、
 resume contract artifact 和最终 manifest。

#### 2026-07-26 D2-AB completion manifest transition amendment（operational）

完整续训在 `0db60d82e454dd722320832e9f7b3f228a90ef72` 正常结束，但原始
`tools/experiment.py start` manifest 绑定的是首段 workload commit
`3fce4767111f7b4c01b5c2af252f6c3ef362cf43`。因此普通 `experiment.py finish` 的
exact-HEAD guard 会在完成阶段拒绝一个已经通过 resume provenance guard、且 metrics 明确记录
source/target/diff 的合法同一 lineage。这是确定的 lifecycle implementation defect，不是
训练或 scientific defect；训练已完成、checkpoint 未修改、evaluation 尚未启动。

允许的最小修复仅扩展 `tools/experiment.py finish/register` 的 manifest provenance contract：
在显式提供原始 manifest commit、当前 workload commit、`git diff --binary` SHA-256 和固定
changed-file allowlist 时，验证这些值、metrics 的 `resume_commit_provenance` 与最终
`metrics.git_commit`，并把 transition 作为不可变 manifest/registry 字段记录；未显式绑定或
路径/hash 不匹配仍 fail closed。不得放宽 dirty-worktree、run-id、artifact overwrite 或
普通 exact-HEAD checks，也不改变训练、评估、gate、uncertainty、checkpoint selection、
consistency 或 fallback 规则。此修复不启动任何 GPU workload，随后仍只执行已注册的 D2-AB
finish/recovery、internal diagnostic 和 native evaluation。

#### 2026-07-26 Phase 1B D2-AB completion / optimization-negative stop

D2-AB 已完成全部预注册 lifecycle，并严格分类为
`predicted-support-no-slip-optimization-negative-stop`。本 completion 只封存已完成的 smoke、
training、internal diagnostic 和 native evaluation；没有启动第二次完整训练、conditional
fallback、checkpoint selection、consistency、HSIPrior 或 Mixer。

1. **GPU smoke 与固定训练 contract。** `p1-hoi-d2ab-gpu-smoke-s42-20260725` 在
   `cuda:0` 覆盖 timesteps `0/249/499`，random-initialized batch 8 的 loss 与关键
   root/rotation/model gradients 均 finite/nonzero；support 分布通过方向中性的
   non-collapse contract。smoke 没有创建 optimizer、update 或 checkpoint，也没有加载任何
   checkpoint。随后 formal run
   `p1-hoi-d2ab-predicted-support-no-slip-s42-20260725` 在
   `infbagel-4gpu/node01` 完成固定的 `61,440,000` processed windows /
   `30,000` updates、effective batch `2,048`、seed 42、232-D、4×RTX 3090、
   final-online contract。初始 model-state SHA-256 为
   `ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e`，
   final checkpoint SHA-256 为
   `3eb68cc55cae15fd4bd3ff5279131ffd9a35ba0399e8e90557e89cb301631d8e`。
   训练总 wall time `18,382.995 s`，吞吐 `3,342.219 windows/s`，每 rank 最小显存余量
   `21,261,123,584 bytes`；final validation total 为 `0.0488587866`。全部 20 个
   cadence checkpoints 与 80 个 RNG sidecars 已保留。released、author、D2-V/X/Y/Z、
   prior 或 EMA 均未作为初始化来源。
2. **连续 resume provenance。** 首个 `3,072,000`-window checkpoint
   `ceb73ebc3a72d6290fc63e2546533c1565912b905a980381d406ea71b39a2ecc`
   只作为 resumability evidence；训练在 worker-owned persistent session 中持续到完整预算，
   没有被 Codex 停止、重启或换 run id。source commit
   `3fce4767111f7b4c01b5c2af252f6c3ef362cf43` 到 workload target
   `0db60d82e454dd722320832e9f7b3f228a90ef72` 的 binary diff SHA-256
   `9c777e1058ddc78ffdf2455141870e3d08eee37b51621bae4ffa45b32448ec86`
   已进入 continuation config、metrics、manifest transition 和 run-local registry。
3. **internal mechanism gate 失败。**
   `p1-hoi-d2ab-predicted-support-no-slip-internal-s42-20260725` 使用 sealed
   32-sequence/96-window internal cohort、D2-X/D2-AB final-online checkpoints 和
   timesteps `249/499`。对主要量 `D2-X minus D2-AB supported velocity`，sequence bootstrap
   结果分别为：
   - `t=249`：mean `-0.0013696933`，95% CI
     `[-0.0021679224,-0.0006211924]`；
   - `t=499`：mean `-0.0017119791`，95% CI
     `[-0.0029663018,-0.0008443758]`。

   两个 CI 都在 0 以下，说明 D2-AB 的 predicted-support 区域水平足速反而显著更高。
   target/control support-mass ratio CI 分别为
   `[1.005099,1.012231]`、`[1.004567,1.013073]`，完全位于预注册
   `[0.80,1.20]`，因此 support sanity 通过，负结果不能解释为通过抬脚或关闭 support 获得。
   internal lifecycle contract/finite checks 均通过，但 mechanism gate 失败；该 diagnostic
   没有 optimizer、update、checkpoint write/selection 或 official-test 使用。
4. **固定 native evaluation。** `p1-hoi-d2ab-native-eval-s42-20260725` 只加载上述
   D2-AB fixed final-online checkpoint，执行 official 438 sequences×3 windows、
   500-step unguided diffusion；sealed D2-X control records 原样复用。D2-AB point estimates：

   | Te | Txy | FS | Cprec | Crec | Cf1 | C% | Pbody | MPJPE | Troot | Tobj | Oobj |
   |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
   | 3.6840 | 4.0892 | 0.3661 | 0.7896 | 0.5953 | 0.6383 | 0.4775 | 3.7714 | 12.0639 | 8.1101 | 15.9248 | 1.0244 |

   primary `D2-X minus D2-AB` foot-sliding paired difference mean 为
   `-0.0030536116`，95% CI `[-0.0175441732,0.0110958477]`，official improvement
   gate 失败。D2-AB minus D2-X contact-F1 difference CI
   `[-0.0046342095,0.0066400986]` 通过；MPJPE、end-object、Txy、object translation、
   hand penetration 与 human penetration 的 target/control ratio CI 上界全部 `<=1.10`，
   因而 protection gate 全通过。全部 released-baseline absolute diffusion checks 也通过，
   其中 FS ratio `1.0980924 <= 1.10`。固定 181/438 penetration finite-mask contract
   完全匹配。
5. **timing 与 optional metrics。** native runtime `392.837 s`；generation
   `62.556 s`、end-to-end generation `386.303 s`、55,188 frames、CUDA-synchronized
   descriptive throughput `882.212 FPS`。本 evaluator 没有生成 FID 或
   R-Precision（`fid_rprecision_used=false`）；该缺失已原样保留，不删除字段、不以
   D2-AA 或其他 checkpoint 的值代填，也不参与选择。
6. **科学结论。** D2-AB 的 support sanity、protection、absolute、provenance、normalization
   和 artifact contracts 全通过，但 internal optimization direction 与 official FS gate
   均失败。故 predicted-state/contact-aware no-slip objective 在当前固定 objective mixture
   下不是充分机制；这不是 evaluator trick 的失败，也不是确定 training loss/model/math/
   official-evaluator implementation defect。因为预注册 fallback 明确要求 internal
   mechanism gate 先通过，local gradient-projection fallback 的触发条件不成立，剩余第二次
   full-training budget 不得使用。
7. **artifact seal。** compact result 为
   `experiments/results/p1_hoi_phase1b_d2ab_predicted_support_no_slip_s42_20260726.json`。
   smoke tree SHA-256
   `654733afafcde8ffed20f41d0812e46cd4a62d91f3f8ccedb4d9d1c837823bd6`
   （15 files / 91,202 bytes）；training tree
   `e357b0c6e8ed3fdd2d8a0ed8a1ca1ac8dbff461892066bc4bd151928e50063ab`
   （149 files / 7,127,317,639 bytes）；internal tree
   `605cfd89381ceb9eb5e35adc4d274feccd102b06f26fbde6570499d6db9cdd55`
   （18 files / 258,193 bytes）；native-eval tree
   `d56bff74b8982a6f63efd62e784c2759233d9f5591222dbe9deb9922e17b5d42`
   （22 files / 379,944 bytes）。worker/authority tree hashes 均一致。

D2-AB 到此停止且 checkpoint 不可选择。任何未来第二机制必须重新执行 `date`、全量扫描下一
unused Phase 1B identifier、添加新的 dated plan/registry hypothesis，并获得用户新的明确授权；
不得 resume D2-AB，不得自动启动 consistency、HSIPrior 或 Mixer。

#### 2026-07-26 Phase 1B D2-AC part-aware local-object cross-attention interaction adapter 预注册（plan-only）

D2-AB 已按预注册分类为 `predicted-support-no-slip-optimization-negative-stop`：训练稳定且
protection 通过，但 predicted-support 区域足速在 `t=249/499` 均显著向错误方向移动，
official foot sliding 也未改善。因此 D2-AB 的 objective fallback 保持关闭，checkpoint 不可
resume、不可选择、不可初始化任何后续 prior。全量扫描 plan、registry、code、tests 与 tools
确认 D2-AA、D2-AB 已占用而 D2-AC/`d2ac` 尚未出现；D2-AC 是下一个 unused Phase 1B
identifier。

当前最强且最安全的 diffusion control 固定为 D2-X final-online。相对 released local baseline，
D2-X 的 MPJPE、Troot、Tobj、Oobj 已分别只差约 `+0.44%/-0.47%/+1.71%/+1.05%`，但
end-object、FS、contact recall、contact F1、contact coverage 与 Pbody 的缺口仍分别约为
`+23.15%/+8.88%/-18.30%/-12.36%/-20.35%/+49.43%`。本地代码审计同时确认当前
HOIPrior 把完整 `1024×3` BPS flatten 后压成单个 global condition token；16 个 motion
tokens 虽能经 self-attention 间接访问物体条件，但没有显式的 left-hand/right-hand/object-motion
到局部 object geometry 的关系通路。这不是确定 implementation defect，而是与现有 contact/
penetration 缺口一致、可由实验否证的 capacity/routing 假设。

用户已确认采用小型 part-aware、object-aware cross-attention interaction adapter，并放宽原
“最后一次完整训练”上限；但该放宽不构成开放式 architecture search。本 amendment 只锁定一个
adapter 机制、一次 primary full training，以及最多一次严格条件触发的同机制 longer-budget
training。当前 session 只写 plan/registry，不修改 source/config/tests，不创建 workload，不启动
GPU、训练、evaluation、checkpoint selection 或 consistency。实际 lifecycle id 必须在新的
implementation session 首先执行真实 `date` 后，以 dated implementation amendment 绑定，不能
预占未来日期。

1. **假设与可区分的竞争解释。** Primary hypothesis 是：当前 contact recall/coverage、
   hand-object geometry 与 penetration 缺口的一部分来自单一 global-BPS token 无法为不同身体
   role 提供稳定的局部物体关系；在主干中段加入小型 part-to-local-object cross-attention，应在
   不改变 loss、representation 或 sampler 的条件下改善 contact F1/recall 与物理 hand-object
   alignment。竞争解释至少包括：
   - `unused-capacity`：adapter gate 或内部路径没有被优化，full 与 gate-ablated 输出无差异；
   - `unstructured-capacity`：adapter 有贡献，但打乱局部几何对应后效果不变，增益只来自额外
     参数或一般非线性；
   - `objective/distribution-limited`：adapter 使用了正确局部几何，但固定训练 objective 或
     train/rollout gap 仍使 native contact/penetration 不改善。
   Internal causal ablation、local-correspondence permutation 和 native transfer gate 必须分别
   区分这三类解释，attention map 本身不得替代 causal gate。
2. **唯一 manipulated factor 与明确排除项。** D2-AC0 相对 D2-X 只加入下述一个
   interaction adapter。保持现有 global BPS token、D2-X FK-foot temporal routing、全部 loss
   tensor/reduction/weight、optimizer、data、conditions、diffusion、production sampler 和
   evaluator 不变；D2-AB predicted-support objective 必须关闭。不加入 contact/no-slip/
   penetration 新 loss，不改变 contact label supervision，不做 SNR/timestep weighting、
   gradient projection、predicted-history exposure、CFG/guidance、future-GT conditioning、
   threshold/multiplier sweep、token-count sweep、adapter-depth/placement sweep 或中间
   checkpoint selection。adapter 不调用 official FS floor/near-ground helper、contact threshold、
   penetration mask 或 test statistics，因此不是 evaluator trick。
3. **固定 local-object tokenization。** 只读取现有 `code/bps.pt` 的 immutable BPS basis，
   file SHA-256 固定为
   `fdff7204b4697e105457cb7e39267b9555bc0d8d854dbc92cd67e2d8c3e77042`。
   对 basis 坐标执行 deterministic lexicographic-first、farthest-point sampling，距离相等时按
   最小原始 index 决定，固定 16 个 centers：
   `[328,903,503,817,474,1023,382,864,640,431,445,960,547,829,545,756]`。
   每个 basis point 分配给最近 center，tie 按 center 顺序决定；cluster sizes 固定为
   `[39,40,57,61,65,68,70,134,77,64,59,79,43,46,84,38]`，assignment canonical
   SHA-256 固定为
   `b62f91f4eb6c4bf2a9211f0187cd1eb97c25394ee45de155f336079fddeecd`。
   clustering 只依赖固定 basis，不读取 train/validation/test motion、contact 或 evaluator。
   令 \(b_i\in\mathbb R^3\) 为 basis、\(d_i\in\mathbb R^3\) 为当前既有 BPS delta，
   对 cluster \(C_k\) 构造固定 10-D feature
   \[
   u_k=\left[
   {1\over |C_k|}\sum_{i\in C_k}b_i,\;
   {1\over |C_k|}\sum_{i\in C_k}d_i,\;
   \sqrt{{1\over |C_k|}\sum_{i\in C_k}d_i\odot d_i},\;
   {1\over |C_k|}\sum_{i\in C_k}\|d_i\|_2
   \right].
   \]
   不做 train-stat normalization、learned clustering、object-category embedding 或额外 mesh/
   point-cloud encoder。
4. **公式级 adapter。** 保持 4 个 condition tokens 和 16 个 motion tokens 的原顺序，先通过
   原 8-layer Transformer 的前 4 层；只取 contextualized motion token
   \(H_t\in\mathbb R^{512}\)。local object tokens 与三个 role queries 固定为
   \[
   O_k=\operatorname{LN}_o(E_o(u_k)+e_k^{obj}),\qquad
   E_o:10\rightarrow128\rightarrow128,
   \]
   \[
   Q_{t,p}=\operatorname{LN}_q(W_qH_t+e_p^{part}),\qquad
   p\in\{\mathrm{left\ hand},\mathrm{right\ hand},\mathrm{object\ motion}\},
   \]
   \[
   A_{t,p}=\operatorname{MHA}_{d=128,h=4,\mathrm{dropout}=0}
   (Q_{t,p},O,O),
   \]
   \[
   R_t=W_r[A_{t,L};A_{t,R};A_{t,O}],\qquad
   H'_t=H_t+\tanh(\alpha)R_t.
   \]
   \(W_r:384\rightarrow512\)，单个 scalar \(\alpha\) 初始严格为 `0`；这是固定 ReZero
   identity gate，不是 checkpoint-derived prior。所有 Linear/MHA/embedding matrices 与原主干
   一起由 seed 42 从随机初始化，released、author、D2-V/X/Y/Z/AB、prior、EMA 或
   consistency weights 均不得加载。adapter 写回后，完整 token sequence 继续通过原第 5--8
   层；输出仍为 `[B,16,232]`。512-wide locked model 的当前参数量为 `29,673,448`，
   adapter 固定增加 `349,697`，总计 `30,023,145`，增量约 `1.1785%`，不得通过额外 hidden
   layer、第二 adapter 或 enlarged token set 超出 `1.25%` 预注册上限。
5. **BPS 与 production provenance。** Training 只使用当前 loader 已提供的单个
   `object_bps` condition；adapter 不读取 per-frame future BPS、future GT object pose、
   rest-mesh vertices 或 contact labels。Autoregressive production sampling 继续由
   `WindowStateCodec.recompute_bps()` 根据当前生成 object rotation 和 hash-verified rest
   geometry 重算下一窗口 BPS；不得回读 stored per-frame BPS。现有 global BPS token 在 full、
   ablated 与 permuted variants 中始终 byte-matched，确保 causal comparison 只改变新的 local
   relation path。
6. **Mixer/HSIPrior compatibility contract。** HSIPrior architecture、parameters、
   checkpoint schema 和 forward path 均不改变；adapter 只属于 HOIPrior。未来 Mixer 仍只接收
   两个专家在同一 timestep、同一 `WindowStateCodec` frame 下的 clean `[B,16,232]`
   prediction，不读取 adapter token、attention map 或 expert-specific latent。因此 Mixer 不要求
   两专家逐层同构，只要求 232-D field semantics、history、normalization、coordinate frame 和
   clean-x0 API 一致。CPU tests 必须继续证明 HOI/HSI parameter/storage independence 和 codec
   round-trip；若 adapter 迫使 Mixer 增加 HOI-specific coordinate/latent adapter，则 D2-AC
   contract 失败。
7. **cheap pre-training CPU diagnostic。** 在任何 GPU workload 前，authority Python 必须：
   - 复现 BPS file hash、16 centers、cluster sizes、assignment hash、feature shape/dtype/
     finiteness；
   - 证明 adapter-disabled/base model 在共享 trunk state、`eval()`、\(\alpha=0\) 时输出
     max-abs difference `<=1e-6`，并保持 `[B,16,232]` API；
   - 证明初始 backward 时 \(\alpha\) gradient finite/nonzero；在 test-only
     \(\tanh(\alpha)=0.1\) probe 下，object encoder、object/part embeddings、Q/K/V/out
     projections 与 writeback gradients 全部 finite/nonzero。正式 initialization 仍为
     \(\alpha=0\)，probe 不得保存或进入训练；
   - 证明 local feature permutation 在 gate nonzero 时改变 adapter contribution，而完整 token
     reorder 不被错误当作 locality test；固定 causal permutation 必须把
     cluster-delta statistics \(k\leftarrow(k+8)\bmod16\)，同时保留 \(\bar b_k\)、
     \(e_k^{obj}\) 和 global BPS token；
   - 验证 exact parameter count/增量上限、zero/constant/extreme BPS finiteness、role query
     separation、batch/device/dtype propagation、checkpoint variant rejection、HSIPrior
     independence、Mixer clean-output contract，以及 source/static path 中没有 future GT/
     stored per-frame BPS/evaluator threshold。
   任一 hash、parity、gradient、shape、parameter-count、provenance 或 interface check 失败即
   `interaction-adapter-contract-failure-stop`，不得启动 GPU。
8. **registered GPU smoke 与连续 detached lifecycle。** 新 session 以真实日期绑定唯一 smoke
   run id；只在 clean exact authority commit fast-forward 后的 `infbagel-4gpu/node01`
   `cuda:0`，对 fixed real-data batch 8、timesteps `0/249/499` 做 random-initialized
   forward/backward。不得创建 optimizer、update、checkpoint load/write；必须记录 losses、
   initial model hash、\(\alpha\) gradient、test-only nonzero-gate adapter gradients、
   CUDA-synchronized peak memory/headroom 和 4-GPU contention。按每 rank micro-batch 512，
   16 frames、3 roles、4 heads、16 local tokens 估算的 cross-attention score elements 为
   `1,572,864`；真实 smoke 仍必须满足注册 headroom，估算不能替代测量。
   Formal detached training 启动后必须持续运行到固定预算，不设置人为 pause node。通过
   resolved-config/preflight、finite loss/required gradients 初始稳定区间、显存余量和至少一个
   可实际 resume checkpoint 后，Codex 报告 measured throughput、总耗时估计与 ETA，并停止
   主动轮询；worker-owned persistent training 继续运行。停止轮询不等于停止、暂停、kill、
   restart 或 checkpoint selection。
9. **D2-AC0 primary full-training contract。** D2-AC0 占用本 amendment 的第一次完整训练：
   seed 42、固定 split
   `experiments/splits/omomo_hoi_train_validation_seed42.json`
   （SHA-256
   `019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e`）、
   232-D、16/2 frames、500-step clean-x0 diffusion、512-wide/16-head/8-layer trunk、
   4×RTX 3090、per-GPU batch 512、effective batch 2,048、accumulation 1、
   `61,440,000` processed windows / `983,040,000` frames / `30,000` updates。
   固定 FP32 Adam、LR `1e-4`、betas `(0.9,0.999)`、无 warmup/scheduler/weight decay/
   gradient clipping/AMP/EMA；loss weights 固定为 FK
   `0.3569973401779424`、object surface `0.4772322188400037`、velocity `0.1`、
   terminal object goal `1.0`，velocity tensor/reduction 完全沿用 D2-X。首次 start 的
   init/weight-init/resume checkpoint 必须为空；final-online fixed-budget checkpoint 是唯一
   target，不按 validation、internal 或 official test 选择中间 checkpoint。
10. **sealed interaction mechanism diagnostic。** 训练完成后，只加载 D2-AC0 fixed
    final-online checkpoint，在 D2-O 已封存的 internal-validation cohort
    `64 sequences × 3 windows`、phase offsets `(14,56,98)`、selection SHA-256
    `1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`
    上运行三条 500-step paired rollout：
    - `full`：正常 adapter；
    - `gate_ablated`：每一步强制 \(\tanh(\alpha)=0\)，其余 state/weights 不变；
    - `local_correspondence_permuted`：只执行第 7 项固定 \(k\leftarrow(k+8)\bmod16\)
      delta-stat permutation，global BPS 与其余 condition 不变。
    三条 path 的 initial latent、每一步 posterior noise、window/chunk ordering、conditions 与
    history restoration 必须 byte-matched；不得 optimizer、checkpoint write/selection、
    CFG、guidance 或 official-test use。按 sequence、seed 42、10,000 bootstrap 报告
    left/right/union semantic contact P/R/F1/coverage、direct-hand indices `24/26` 与
    FK-palm indices `22/23` 的 `2/5/7.5/10 cm` physical contact P/R/F1/coverage、
    contact run length、GT-contact-frame hand-object distance、penetration、MPJPE、
    object/pelvis goals、FS、learned gate 和 per-role attention entropy。
    Primary internal mechanism gate 使用 direct-hand union 5-cm physical contact F1：
    `full - gate_ablated` 与 `full - local_correspondence_permuted` 的 paired 95% CI
    下界都必须 `>0`；同时两组 `comparator - full` GT-contact-frame mean hand-object
    distance CI 下界都必须 `>0`。任一 ablation gate 失败说明 adapter 未被有效使用；只有
    gate-ablation 通过而 locality permutation 失败，则说明是 unstructured extra capacity，
    不能进入 positive classification。其他阈值/representation 与 attention map 只作完整
    诊断，不得替代 primary causal gate。
11. **固定 native evaluation 与 uncertainty。** Internal 完成后，无论正负都执行一次完整
    reporting evaluation：official 438 sequences×3 windows、500-step unguided production
    diffusion、D2-AC0 fixed final-online weights；CFG/dynamic perception/guidance/scene/
    consistency 均关闭。Locked paired control 是 D2-X final-online，checkpoint/aggregate/
    per-sequence SHA-256 分别为
    `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51` /
    `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
    `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`，
    原样复用且不重新生成。Released aggregate 文件 SHA-256 为
    `76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6`。
    所有 paired metrics 用 sequence unit、seed 42、10,000 bootstrap；penetration 继续使用
    sealed D2-X evaluator 的相同 181-sequence finite mask。FID、Matching、
    R-Precision@1/2/3、Diversity 与 timing 若 evaluator 正常产生必须原样保留和报告；
    FID/R-Precision 当前不参与 selection，不得删除、代填或反向调 evaluator。
12. **native transfer、protection、absolute gate 与 classification。**
    - Native transfer gate 要求 D2-AC0 minus D2-X 的 contact-F1 与 contact-recall paired
      95% CI 下界均 `>0`，且 contact-F1 point estimate 至少关闭 released--D2-X 缺口的
      `25%`：
      \[
      {C_{F1}^{AC}-C_{F1}^{X}\over C_{F1}^{released}-C_{F1}^{X}}\ge0.25,
      \]
      即按当前 sealed points 至少达到约 `0.6598838781`。
    - Protection gate 要求 D2-AC0/D2-X 的 end-object、Txy、FS、Pbody、hand penetration、
      MPJPE、Troot、Tobj、Oobj paired mean-ratio 95% CI 上界全部 `<=1.10`，且 contact
      precision difference CI 下界 `>=-0.02`。固定 penetration finite-mask contract 也必须
      通过。
    - Final effectiveness gate 仍使用 Phase 1B section-wide released-baseline 95% point
      gate：lower-is-better 的 end-object、Txy、FS、Pbody、MPJPE、Troot、Tobj、Oobj
      均 `<=baseline/0.95`，higher-is-better 的 contact P/R/F1 均
      `>=0.95×baseline`。Contact coverage 必报但不单独当 monotone selection metric；
      FID/R-Precision 当前只报告、不选择。
    Contract 失败分类 `interaction-adapter-contract-failure-stop`；full 对 gate ablation 失败为
    `interaction-adapter-unused-optimization-negative-stop`；gate ablation 通过但 locality
    permutation 失败为 `interaction-adapter-locality-negative-stop`；internal 通过但 native
    transfer 失败为 `interaction-adapter-transfer-negative-stop`；transfer 通过但 protection
    失败为 `interaction-adapter-conflict-negative-stop`；mechanism/transfer/protection 通过但
    released 95% gate 失败为
    `interaction-adapter-positive-but-not-effective-stop`；全部通过才为
    `interaction-adapter-positive-candidate-stop`。只有最后一类可把 fixed final-online
    checkpoint 标记为 selectable autonomous diffusion HOIPrior candidate；这不是中间
    checkpoint selection，也不自动授权 consistency。
13. **artifact contract。** 必须保留并在 worker/authority 双端 hash 验证：dated
    plan/registry 与 lifecycle amendment、BPS cluster metadata/builder resolved record、
    logical implementation commit、source/config/tests、authority CPU/parity/gradient logs、
    worker pull/preflight/resolved configs、GPU smoke manifest/log/metrics、formal training
    manifest/log/state、全部 cadence checkpoints 与 per-rank RNG、initial/final model hashes、
    validation/training metrics、resumability evidence、measured throughput/ETA、internal
    three-variant manifest/per-sequence/paired-noise/attention appendix、native manifest/
    aggregate/per-sequence/bootstrap/penetration-mask、optional FID/R@/timing、run-local
    registry、dependency/hardware/data/evaluator hashes、recovered artifact tree、compact
    result、`docs/phase_summaries/PHASE_1B_D2AC.md` 及所有 operational/scientific
    failures。大 artifacts 不进 Git，不覆盖结果、不复用 run id。
14. **最多一次同机制 conditional longer-budget training。** D2-AC1 只有在 D2-AC0 严格
    分类为 `interaction-adapter-positive-but-not-effective-stop` 时才 eligible：即 causal
    gate/locality、native contact transfer 与 protection 全通过，只因 released 95% magnitude
    未达标而停止。触发后仍须真实 `date`、新的非复用 run id、dated plan/registry binding 和
    用户再次明确确认；不得自动启动。D2-AC1 必须以相同 seed 42 和相同 adapter/trunk/loss/
    optimizer/data/evaluator 从随机初始化开始，不得 resume 或加载 D2-AC0/D2-X/D2-AB；
    唯一变化是预算固定为 `122,880,000` processed windows /
    `1,966,080,000` frames / `60,000` updates。D2-AC1 重跑同一 internal/native gates，
    D2-AC0 全部结果仍保留并完整报告。若 D2-AC0 属于 contract/unused/locality/transfer/
    conflict negative，或 D2-AC1 未通过全部 candidate gates，则本 amendment 到此停止；
    不允许继续增加预算、改变 token 数/width/layers/placement、换 role、加入新 loss、做
    parameter sweep 或 checkpoint selection。任何进一步方向须使用下一个 unused Phase 1B
    identifier 和新的用户授权。

本 plan-only amendment 不授权当前 session 的 implementation、GPU smoke、training、
evaluation、checkpoint selection、consistency、HSIPrior 或 Mixer。新的 implementation session
必须首先重新执行 `date`、path/branch/HEAD/status 核验，完整读取 `AGENTS.md`、本计划和
`docs/phase_summaries/PHASE_1B_D2AB.md`，再以真实 lifecycle identities 添加
implementation amendment；source change、CPU tests、worker publication 和 D2-AC0 workload
只能在该入口之后进行。

#### 2026-07-26 Phase 1B D2-AC0 implementation/lifecycle binding amendment

本 session 已在真实日期 `2026-07-26` 重新核验 authority path、`phase/01b-hoi`、
handoff HEAD `61a989adab2f3053230bfcd0ebb702601fcdaab2` 与 clean worktree，并获用户明确
授权连续完成 D2-AC0 的 implementation、CPU contract、worker publication、注册 GPU smoke、
一次完整 from-random training、固定 internal diagnostic、一次 native evaluation 与 artifact
recovery。本 amendment 只绑定已预注册机制和本日未使用 lifecycle identities，不改变 D2-AC0
的 architecture、representation、loss、optimizer、data、budget、gates 或 stop rules：

- implementation logical change：`p1-hoi-d2ac-interaction-adapter-implementation-s42-20260726`；
- authority CPU contract：`p1-hoi-d2ac-cpu-contract-s42-20260726`；
- registered GPU smoke：`p1-hoi-d2ac-gpu-smoke-s42-20260726`；
- formal training：`p1-hoi-d2ac-interaction-adapter-s42-20260726`；
- internal diagnostic：`p1-hoi-d2ac-interaction-adapter-internal-s42-20260726`；
- native evaluation：`p1-hoi-d2ac-native-eval-s42-20260726`。

Implementation must remain a single logical commit containing source, config, CPU tests, this dated
binding and its registry record. The worker may execute only that exact committed Git object. The
first-start init/weight-init/resume checkpoints remain empty; all D2-AC0 workloads use seed 42 and
the fixed final-online target. If the date changes before an unstarted lifecycle, that lifecycle must
receive a new identity-only dated amendment before its manifest/workload; an old identity is never
reused. D2-AC1 remains unauthorized and cannot be started automatically.

The implementation-session preflight also found a transcription defect in the plan-only assignment
hash, before any source or workload was created. The preregistration calculation serialized
`{"algorithm":"lexicographic-seed-farthest-point-16-v1","centers":...,"assignments":...}`
with sorted JSON keys and compact separators and produced the valid 64-hex SHA-256
`b62f91f4eb6c4bf2a9211f0187cd1eb97c25394ee45de155f33607959fddeecd`;
the plan-only text and registry line accidentally omitted the two hex characters `59` and therefore
recorded a 62-character value that cannot be a SHA-256. D2-AC0 binds the valid canonical 64-hex
value. This correction changes no center, assignment, cluster size, token feature, model parameter,
training/evaluation protocol, gate, or authorized scope; CPU validation must reproduce the canonical
payload and corrected hash exactly and fail closed otherwise.

#### 2026-07-26 Phase 1B D2-AC0 evaluator closure 与 GPU-smoke retry amendment

在 implementation tree 的 authority 全套 CPU gate 中，323 项测试首次运行有 321 项通过；
仅 `tests/test_hoi_d2t.py` 与 `tests/test_hoi_d2u.py` 的旧整文件冻结断言仍要求 D2-AC 之前的
`code/priors/models.py` SHA-256。失败只涉及 approved D2-AC architecture-variant extension
所在的 shared file；D2-AC exact shared-trunk、`eval()`、`alpha=0` parity 的 measured
max-abs difference 为 `0.0`，representation/data/loss/diffusion hashes 均未改变。CPU gate
closure 因此只允许把这两个 historical freeze 更新到 approved post-D2-AC model source，
同时保留 exact base-path parity test；不得借此接受任何旧 HOIPrior 行为变化。

本 amendment 也在任何 formal training 前封存剩余 evaluator implementation：

- internal runner 只加载 fixed final-online D2-AC0 checkpoint，并在 sealed D2-O
  `64×3` cohort 上运行 `full`、`gate_ablated`、`local_correspondence_permuted` 三路
  production 500-step rollout；
- primary direct-hand union 5-cm F1 与 GT-contact-frame distance 严格使用
  indices `24/26`。GT-contact-frame 定义为 fixed target direct-hand union distance
  `<5 cm`；无 GT-contact frame 的 sequence 不做零值代填，paired bootstrap 使用三路共享的
  target-derived finite sequence mask，并在 artifact 中记录 count 与 identities；
- 本 authority-only derivation fixes that mask at `57` sequences with ordered-name
  SHA-256 `2fa79d30ab6dd6a915098344c4aa7267cb6c3323c6d2a762b4b704f8757cebaa`;
- FK-palm `22/23`、semantic contact、2/5/7.5/10-cm physical contact、run length、
  MPJPE、object/pelvis goal、FS、learned gate、per-role attention entropy 全部报告；
  internal penetration 复用 official SDF formulas，在 internal native 10-Hz frames 上作为
  descriptive metric，且沿用 official excluded object categories/finite mask，不参与
  internal causal gate；
- native wrapper 只生成 D2-AC target，复用 sealed D2-X aggregate/per-sequence hashes 与
  released aggregate，执行既定 contact transfer、九项 protection、precision、penetration
  finite-mask 和 released 95% gates；不重跑 control、不选择中间 checkpoint；
- runner 接受既有 date-transition rule 下的 dated internal/native lifecycle identity，
  但 scientific protocol、selection、seed、threshold、bootstrap 与 gates 不随日期变化。

原 `p1-hoi-d2ac-gpu-smoke-s42-20260726` 已在 model/trainer implementation commit 上产生
稳定 no-update artifact，但它先于 authority full-suite closure，且旧 smoke metadata 没有
正确区分 batch-8 measured attention tensor 与 formal micro-batch-512 的 registered
`1,572,864` score-element estimate。该 artifact 必须保留，不能覆盖或复用；它不作为
formal-launch 的最终 smoke gate。完成本 amendment、authority full tests 和 clean commit
后，唯一允许的 operational retry 是
`p1-hoi-d2ac-gpu-smoke-r1-s42-20260726`（若未启动前跨日则按 date-transition rule 换新日期）。
Retry 的 architecture/data/timesteps/batch/no-update/no-checkpoint protocol 不变，只绑定最终
committed source tree并同时记录 batch-8 actual score-element shape、formal estimate 与实际
CUDA peak/headroom。Formal training run id、预算、随机初始化与全部 scientific gates 不变。

#### 2026-07-26 Phase 1B D2-AC0 authority CPU lifecycle retry amendment

原 `p1-hoi-d2ac-cpu-contract-s42-20260726` 已在任何 CUDA/optimizer/checkpoint workload
前因 resolved-config helper 错把 checkout root 推导为 `/data/yujinlun/code` 而中止；其
manifest、operational failure、CPU log 和 resolved config 已以 `aborted` 状态封存并注册，
不得覆盖或复用。该事件不构成 scientific contract failure，也没有产生模型、梯度或评测
结果。为满足 append-only lifecycle，唯一的 CPU retry identity 是
`p1-hoi-d2ac-cpu-contract-r1-s42-20260726`。Retry 只修复 run-root/path binding，保持
同一 D2-AC0 source/config/test/seed/BPS hashes、CPU contract、无 optimizer/CUDA/checkpoint
语义；必须在最终 logical implementation commit 的 clean tree 上重新创建 manifest，并在
CPU gate 失败时于 GPU 前停止。该 retry 不改变 GPU smoke r1、formal training、internal/
native run ids 或任何 scientific gate，也不授权 D2-AC1。

CPU retry `p1-hoi-d2ac-cpu-contract-r1-s42-20260726` 已于
`a32707047014abb2618b0b2c0ca5a23f55bfcc69` 完成并封存：329 项 authority tests 全部通过，
contract diagnostic 为 `cpu-contract-passed`，manifest/metrics SHA-256 分别为
`151b49d01c0b16980b1607a8b32e5e2fff24752cb6f5b744b07cddff75d5ddea` /
`b152ff16d90492a9010bab916035b8bd1c9179de38fef5965d05d5019f9d01ec`。该结果确认
interaction-adapter contract 可进入 worker publication；它没有创建 optimizer、执行 CUDA、
加载/写入 checkpoint 或进行 scientific selection。

#### 2026-07-27 Phase 1B D2-AC0 post-training evaluation lifecycle date-transition amendment

D2-AC0 formal training lifecycle 已于注册日期 `2026-07-26` 使用
`p1-hoi-d2ac-interaction-adapter-s42-20260726` 启动，并在 clean worker commit
`273e6d7e693f6664b3cd9d0c45b31b6b20c58496` 上完成固定的 61,440,000-window /
983,040,000-frame / 30,000-update from-random contract。训练 exit code 为 0；fixed
final-online checkpoint SHA-256 为
`fede1c2b2f331407ceba7db16e3a4b30ccc6ffb6c8fc252861662bdcc96c7b96`。完整 immutable
training tree 已由 worker 主动回收到 authority
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ac-interaction-adapter-s42-20260726`，worker 与
authority canonical tree SHA-256 均为
`d3784f0b01b8762ab1e6dcc7b0343ef2aa2147c1ca9672f516ae2f672cd92d98`
（115 files / 7,211,816,400 bytes）。

在真实日期 `2026-07-27` 核验时，authority 与 worker 均为 clean
`phase/01b-hoi@273e6d7e693f6664b3cd9d0c45b31b6b20c58496`；原绑定的 internal/native
identities 尚未创建 manifest、加载 checkpoint 或启动 workload。依照既有 date-transition
rule，只允许以下 identity-only replacement：

1. internal diagnostic 改为
   `p1-hoi-d2ac-interaction-adapter-internal-s42-20260727`，subphase 保持
   `1B-D2-AC0-internal`；
2. native evaluation 改为 `p1-hoi-d2ac-native-eval-s42-20260727`，subphase 保持
   `1B-D2-AC0-native`；
3. 未使用的 `p1-hoi-d2ac-interaction-adapter-internal-s42-20260726` 与
   `p1-hoi-d2ac-native-eval-s42-20260726` 在任何 workload 前 supersede，永不创建、复用或
   绑定 checkpoint；这不是 scientific/operational failure，也不产生 retry entitlement。

本 amendment 不改变 training run/checkpoint、seed 42、sealed D2-O 64×3 cohort、三路 paired
500-step rollout、10,000 次 sequence bootstrap、primary causal/locality gates、official
438×3 evaluator、sealed D2-X/released controls、penetration finite mask、native
transfer/protection/released-95% gates、classification 或 artifact contract。Internal 完成后仍
必须无条件执行一次 fixed native evaluation。D2-AC1、checkpoint selection、consistency、
HSIPrior、Mixer、任何 sweep 或新机制仍未授权。

#### 2026-07-27 Phase 1B D2-AC0 internal penetration-asset loader retry amendment

第一次 replacement internal lifecycle
`p1-hoi-d2ac-interaction-adapter-internal-s42-20260727` 在 clean
`655930f0d9b6bb47fbe116c1d779650cfd3dff63` 上通过 resolved-config、manifest、fixed
checkpoint hash 与 same-context GPU preflight 后启动，但在 21.504 秒后 fail-closed：
`load_penetration_assets()` 对 official asset `floorlamp.ply.npy` 使用 `Path.stem`，错误地产生
key `floorlamp.ply`；随后 fixed cohort 的 object category `floorlamp` 无法查到同一 SDF。Official
evaluator 已封存的实现使用 `file.split('.')[0]`，对该文件产生正确 key `floorlamp`。因此这是
一个确定的 internal evaluator asset-key transcription defect，不是 interaction adapter、
checkpoint、cohort、penetration formula/mask、threshold 或 gate 的 scientific failure。

失败 lifecycle 已以 `failed` finish/register，manifest/metrics/run-local registry SHA-256 分别为
`6c17c6e3b73664927c2f8c432c90e8df0e69a025c0a63a6c818765c1be0ab574` /
`2a1be5639174e7d85b46e7f6fd8fc1082077297d803584e1fd4bf207c7325e11` /
`e59caad3066042779b1b80394d3649d17cfa5ab2db0df3b5c5217fcdf8a2bfd7`，并由 worker 主动
回收到 authority；双端 canonical tree SHA-256 为
`e9eaea941d71624152d7d4d05d4ffc162f8fcd02f9c4e0cdf28ce139ce31d7ce`
（10 files / 46,938 bytes）。该 attempt 没有 optimizer、training update、checkpoint write/
selection、official-test use 或 consistency。

只允许以下最小 closure：

1. penetration SDF key extraction 改为与 immutable official evaluator 完全相同的首个 `.` 前
   basename；不改变 asset bytes、SDF/SMPL-X 公式、excluded categories、finite handling 或任何
   reported metric；
2. 加入 double-suffix regression test，并仅让 internal/native provenance regex 接受 append-only
   retry identity `p1-hoi-d2ac-interaction-adapter-internal-r1-s42-20260727`；
3. r1 必须重新生成 resolved config、same-context preflight 和 manifest，从头运行同一 64×3
   full/gate-ablated/locality-permuted paired rollout；不得复用失败 attempt 的 partial output；
4. native lifecycle 仍为 `p1-hoi-d2ac-native-eval-s42-20260727`，且仅在 r1 完成后运行一次。

除上述 filename-key parity 外，source/config/checkpoint/seed/batch/cohort/noise/bootstrap/
metrics/gates/classification/artifact contract 全部不变。若 r1 再次 contract-fail，则封存并停止；
不得继续 retry、改变 mask、删除 penetration、选择 checkpoint 或启动 D2-AC1/consistency。

#### 2026-07-27 Phase 1B D2-AC0 internal zero-denominator summary r2/native continuation amendment

Internal r1
`p1-hoi-d2ac-interaction-adapter-internal-r1-s42-20260727` 已在 clean
`7481c5ee2465725a857fd961876d8f1b997a0eed` 上从头完成 full、gate-ablated 与
local-correspondence-permuted 三条 fixed 64-sequence paired rollout；三份 raw variant 均为
64/64 sequences、finite、all-fields-reported，并共享 28-sequence official penetration
finite cohort。最终汇总在计算 descriptive
`full_mean / gate_ablated_mean` hand-penetration comparison 时失败，因为
gate-ablated 的合法 `hand_pen_loss_omomo` mean 为严格 `0`，而继承的 D2-M ratio helper
要求 denominator mean 严格大于 `0`。零 penetration 是合法且 lower-is-better 的 evaluator
结果；该失败只说明比值在零 denominator 下数学上未定义，不得将它解释为 evaluator metric、
mask、checkpoint 或 adapter contract 失败。

r1 已以 `interaction-adapter-contract-failure-stop` 封存并由 worker 主动回收。Manifest、
metrics、run-local registry 与 canonical artifact-tree SHA-256 分别为
`073bdb39605e518fd08124ad5175380dab2e4afedb455f1fe39ad624b724f28c` /
`ec79022477e5fb02cafaeb8329e0ee439c9d2612802aecda17607464fb582fa4` /
`89221db4a4cf7195c5a652472f9039f262c21642e9b39f7dc5b7941a93f0eca8` /
`2f7c013814fb294bedd5abe0d9c503e3ff1282369daa396ea1e3ac45ec5f9dd8`
（11 files / 15,375,787 bytes）。Full/gate-ablated/locality-permuted raw SHA-256 分别为
`7dc60b213777334d5e6d4a09bb78cf920a60c81858eedec384f3b5992370e472` /
`761844422a42f5738d1611e4d68bee9963ba7c18b639dcb312bd0a1c84c5f192` /
`f2abed08aa3eecf59179c03958f2e5a3e6671a0cb9409a9250ae727cb8a8ee9a`。
没有 optimizer、training update、checkpoint write/selection、official-test use 或
consistency。

用户在检查该证据后明确授权修复并完成最终评估。只允许以下 deterministic evaluator-summary
closure：

1. 仅为 D2-AC internal descriptive nonnegative penetration comparison 增加显式
   zero-denominator contract。原始 `hand_pen_loss_omomo` /
   `human_pen_loss_infbagel`、official finite mask、per-sequence values、excluded categories、
   SDF/SMPL-X 公式与 aggregate metric 均不变；
2. denominator mean `>0` 时必须调用原 `paired_mean_ratio`，保持 seed 42、10,000 paired
   sequence bootstrap、point estimate、CI 与 per-unit 数据逐字段一致；
3. denominator mean `==0` 时不得加 epsilon/pseudocount、不得 clamp 成有限 ratio、不得用
   infinity 或改变 metric。必须记录 `ratio_defined=false`、`mean_ratio=null`、
   `bootstrap_95_ci=null` 与原因 `zero_denominator_mean`，同时用同一 paired sequence values
   调用原 `paired_difference`，报告绝对 difference point estimate/95% CI；
4. mismatched、empty、non-finite 或 negative penetration vectors 继续 fail-closed；
5. 新 internal identity 唯一为
   `p1-hoi-d2ac-interaction-adapter-internal-r2-s42-20260727`。r2 必须重新生成 resolved
   config、same-context preflight/manifest，并从头运行完整三路 rollout；不得复用 r1 raw
   output；
6. primary internal mechanism/locality gates 仍只使用 registered direct-hand union 5-cm F1
   paired differences 与 GT-contact-frame distance paired differences；descriptive penetration
   ratio/difference 不参与 gate；
7. r2 contract 完成后，无论 mechanism gate 正负，执行一次仍未启动的
   `p1-hoi-d2ac-native-eval-s42-20260727`。Native 必须继续使用未修改的 official 438×3
   production evaluator、sealed D2-X control、released aggregate、181-sequence penetration
   mask、原 paired ratio/difference helpers、阈值与 gates；本 amendment 不改变 native
   evaluator 数学或既有实验口径。

Authority implementation closure 在 plan-only commit
`ef39e62c2d30c9dd0d2575121a7806375d53e23b` 之后完成：新增 helper 只由 internal penetration
summary 调用，native wrapper 明确不引用它；既有 `paired_ratio_fixed` 函数 source SHA-256
在修改前后均为
`2d1e58aab9d340250eb90e3ea176132380e4d27fea069ec3c51d33ff90fe9b08`。
Authority targeted tests 为 67 passed，full suite 为 333 passed，registry validation 为
178 records valid。对 immutable r1 raw artifacts 的只读 summary replay 成功产生
`ratio_defined=false` / null ratio 与原 paired-difference CI，没有写回或提升 r1；正式
mechanism classification 仍只由新 r2 lifecycle 决定。

若 r2 仍发生 contract failure，则封存并停止，不得继续 retry。D2-AC1、checkpoint
selection、consistency、任何 architecture/token/parameter/placement sweep、新 loss、
HSIPrior、Mixer 或其他 HOIPrior 搜索仍未授权。

#### 2026-07-27 Phase 1B D2-AC0 native serialized-field parity r1 amendment

Internal r2 已完成全部 contract，并按预注册规则无条件启动 fixed native lifecycle
`p1-hoi-d2ac-native-eval-s42-20260727`。该 lifecycle 在 clean
`cc931b8b6272e323e25be6cc6c6a6e3a49076558` 上完整运行 immutable official
438-sequence × 3-window、500-step production evaluator，生成了 438 条 target
per-sequence records 和 finite aggregate metrics；official aggregate 中 Troot/Tobj/Oobj
分别为 `8.722931146621704` / `16.810938650698322` / `1.0305662157312816`。Target 与
sealed D2-X 的 `trans_dist`、`obj_trans_dist`、`obj_rot_dist` per-sequence fields 均为
438/438 finite，target penetration fields 也严格保持 sealed 181-sequence finite mask。

Official evaluator 完成后，D2-AC post-evaluator paired summary 在第一个 Troot protection
comparison fail-closed，错误为
`native metric trans_dist contains missing/nonfinite values`。根因是 D2-AC wrapper 的
`PER_SEQUENCE_KEYS` 将 evaluator 已封存的 serialized keys `trans_dist` /
`obj_trans_dist` / `obj_rot_dist` 错误映射成不存在的
`translation_difference` / `object_translation_difference` /
`object_rotation_difference`；这使 `_metric_array` 读到 `None`，并不表示 evaluator raw
records 非有限。D2-X sealed wrapper 与 official records 始终直接使用短字段名，因此该问题是
D2-AC paired-summary field routing defect，不是 checkpoint、sampler、official metric、
finite mask 或 native gate 的科学结果。

失败 lifecycle 已以 status `failed` 原样封存并由 worker 主动回收。Manifest、metrics、
run-local registry、aggregate、per-sequence 与 canonical artifact-tree SHA-256 分别为
`e3134b5567eac018a6b99c49c05276d802db3a4d2b6c7864adbbaac419bbd6d6` /
`0ab818a018cd7465b8e25b831992d5bd2ba5f76a0b89af73c30c34394035218e` /
`dead4f9ebbcb639da24b8629272daa7f9a82eded272c94bbf603cfe4b0433262` /
`995acb311187a1f0bfd8abe2f74358da70998deb9ce5b8c98a99e9e36b99e6c3` /
`dd8803c8efe4b836a09d31dc8c86b6f8230d3de6cb92aa2f30c574b96cb4ad6a` /
`ae7f0e3d9975a2bc4d96058dc1f3c5a965a4fb9870634d10889db97f2a0e1b27`
（14 files / 339,245 bytes）。Authority staging 为
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ac-native-eval-s42-20260727`；
worker/authority tree hash 完全一致。该 attempt 没有训练、optimizer/update、
checkpoint write/selection 或 consistency，也没有形成 native gate classification。

用户已明确要求修复并完成与既有实验一致的最终评估。只允许以下 deterministic native
closure：

1. 将上述三个 D2-AC `PER_SEQUENCE_KEYS` value 改为 official evaluator 与 sealed D2-X
   records 的原 serialized short keys；其他 key mapping、metric formula、aggregate、
   finite handling、penetration mask、bootstrap helper、threshold 与 gate 全部不变；
2. 增加基于 official per-sequence schema 的回归测试，证明 Troot/Tobj/Oobj 从 short keys
   读取，旧的不存在 alias 不被要求；继续 fail-closed 于真实 missing/non-finite values；
3. 用 source hash/test 证明 `code/test_infbagel_hoi.py`、`code/eval_metrics.py`、
   `code/config/config_eval_hoi_prior.yaml`、shared D2-X wrapper、sealed control/baseline 与
   internal zero-denominator helper 均未改变；
4. 新 identity 唯一为 `p1-hoi-d2ac-native-eval-r1-s42-20260727`。必须重新生成 resolved
   target/config、same-context preflight 和 manifest，并从头运行完整 official 438×3
   evaluator；不得复用失败 attempt 的 aggregate/per-sequence/partial output；
5. retry 仍只加载 D2-AC0 fixed final-online checkpoint，复用 sealed D2-X control，不重新生成
   control；paired unit、seed 42、10,000 bootstrap、181-sequence penetration mask、
   transfer/protection/released-95% gates 和 classification precedence 全部不变；
6. 无论 native transfer 结果，最终 D2-AC0 classification 仍受已封存 internal locality
   failure 约束，不得 selectable，也不 eligible for D2-AC1。

若 native r1 仍 contract-fail，则封存并停止，不得继续 retry、修改 evaluator/mask/gate、
选择 checkpoint、启动 D2-AC1/consistency、HSIPrior、Mixer 或任何新 HOIPrior 搜索。

#### 2026-07-27 Phase 1B D2-AC0 native serialized-field parity scope correction

在上述 plan-only amendment 后、worker publication 或 native r1 启动前，authority 使用失败
attempt 已回收的 immutable 438-sequence target records 与 sealed D2-X records 对修正后的
paired-summary 做了只读 replay。Troot/Tobj/Oobj 已能完整计算，但 penetration mask contract
显示 0 finite sequences。原始 records 的
`hand_pen_loss_omomo` / `human_pen_loss_infbagel` 实际仍为两侧相同的 181 finite
sequences；0-mask 同样来自 `PER_SEQUENCE_KEYS` 中不存在的
`hand_object_penetration` / `human_object_penetration` aliases。

因此 native field-parity fix 的完整范围校正为五个、且仅五个 official serialized short-key
identity mappings：

- `trans_dist -> trans_dist`；
- `obj_trans_dist -> obj_trans_dist`；
- `obj_rot_dist -> obj_rot_dist`；
- `hand_pen_loss_omomo -> hand_pen_loss_omomo`；
- `human_pen_loss_infbagel -> human_pen_loss_infbagel`。

回归测试必须同时锁定五个 mappings、真实 missing/non-finite fail-closed behavior 与 sealed
181-sequence penetration mask replay。该 scope correction 不改变 SDF/evaluator formulas、
excluded categories、finite mask、paired statistics、native gates、retry identity 或从头重跑
要求；前一 amendment 中“其他 key mapping 不变”应解释为除这五个已证实 alias defect 外
全部不变。若五-key replay 不能通过完整 comparison contract，则不得发布 worker 或启动
native r1。

Authority implementation closure 在 plan-only commit
`376950ea03652306e448bd8c7e7f27362860dd54` 后完成。D2-AC wrapper 只将五个 confirmed
aliases 改为 official short-key identity mappings，并只扩展 lifecycle regex 以接受已登记的
`p1-hoi-d2ac-native-eval-r1-s42-20260727`；没有改变 official evaluator、shared D2-X
wrapper、internal diagnostic、paired statistics 或 gate code。修正后 native wrapper 与
D2-AC tests SHA-256 分别为
`04b49c17602d13da2f45f2ae47dba191c4a21a5e914ada560994cdde3c0c827c` /
`1c51204f5f8140d95bc5a1abbd5e76cab4812759b349c8e3211385c2707a2c3f`。

Authority targeted D2-AC tests 为 25 passed，full suite 为 335 passed，registry validation
为 180 records valid，`py_compile` 与 `git diff --check` 通过；full-suite log SHA-256 为
`528c707aa2413a23e007dc92580c954cf4832cf5dba7dc0d5bd452ae49264619`。对失败 native
attempt 的 immutable raw records 做只读 replay 后，全部 9 个 protection ratios 与三条
contact paired differences 均成功产生，penetration contract 恢复为精确 181 sequences /
`2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec`，没有写回 artifact
或用于正式 selection。Locked source SHA-256 仍为：official test
`22886f8797ceb04a892487393dea9f80e19877bc02dd7a6f39127e7319119524`、eval metrics
`445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547`、eval config
`89c702d96b98289924225c4b163d3b29eb22efe27c50ac799ddd0c71c515aa73`、shared D2-X
wrapper `b6753a66207492e6ee4addb8f450cb38c5d021401d43430faa9e5c9ed77c6e31`、internal
diagnostic `e9a0157f80695469a53a5333b20685cb3c66d042b0ccd621b86164238764bcc5`。

#### 2026-07-27 Phase 1B D2-AC0 completion record

D2-AC0 的全部已批准 lifecycle 已完成。最终 tracked compact result 为
`experiments/results/p1_hoi_phase1b_d2ac_interaction_adapter_s42_20260727.json`，
phase summary 为 `docs/phase_summaries/PHASE_1B_D2AC.md`。本 completion record 只封存
已经执行的固定机制、失败与结果，不新增方向、训练、selection 或 fallback。

1. **CPU、smoke 与 training。** Authority CPU retry
   `p1-hoi-d2ac-cpu-contract-r1-s42-20260726` 通过 exact BPS/assignment、参数量、
   `[B,16,232]` API、`alpha=0` base parity、初始/activated gradients、local permutation、
   extreme-input、provenance、HSIPrior/Mixer independence 与 static path contract。
   Registered final smoke
   `p1-hoi-d2ac-gpu-smoke-r1-s42-20260726` 在 `cuda:0`、real-data batch 8 上通过，
   measured attention score shape/elements 为 `[8,16,3,4,16] / 24,576`，formal
   micro-batch-512 estimate 为 `1,572,864`，peak allocated/reserved/headroom 为
   `252,510,720 / 304,087,040 / 24,991,956,992` bytes。Formal training
   `p1-hoi-d2ac-interaction-adapter-s42-20260726` 从随机初始化完成
   `61,440,000` windows / `983,040,000` frames / `30,000` updates，wall time
   `19,157.121 s`，throughput `3,207.162 windows/s`，20 cadence checkpoints 与
   80 rank RNG sidecars 完整；fixed final-online SHA-256 为
   `fede1c2b2f331407ceba7db16e3a4b30ccc6ffb6c8fc252861662bdcc96c7b96`。
   Learned alpha/gate 为 `0.0907876045 / 0.0905389935`。
2. **Internal r2。**
   `p1-hoi-d2ac-interaction-adapter-internal-r2-s42-20260727` 在 sealed
   64-sequence/192-window cohort 上从头完成三路 paired 500-step rollout。Full minus
   gate-ablated direct-hand union 5-cm F1 为 `+0.6215448246`，95% CI
   `[0.5397640759,0.7003120412]`；gate-ablated minus full GT-contact distance 为
   `+90.978005 cm`，CI `[81.0602569,100.8264305]`，证明 adapter 被使用。Full minus
   locality-permuted F1 为 `+0.0103920517`，CI
   `[-0.0177715936,0.0375934559]`；permuted minus full distance 为
   `+0.013819838 cm`，CI `[-0.3039546465,0.3092713829]`，两项 locality gate 均失败。
   合法零 hand-penetration denominator 显式记录为 `ratio_defined=false`、null ratio/CI，
   并保留同一 paired values 的 difference `9.308420447e-07`、CI
   `[0,2.792526134e-06]`；没有 epsilon、pseudocount、clamp 或 infinity encoding，
   且该 helper 不进入 native evaluator。Internal classification 为
   `interaction-adapter-locality-negative-stop`。
3. **Official native parity 与结果。** Final native lifecycle
   `p1-hoi-d2ac-native-eval-r1-s42-20260727` 在 clean
   `e6ee3fd9611ede9ee8e0cad20b94bd81e9c13366` 上从头运行 official 438×3、
   500-step unguided evaluator。D2-AC wrapper 的修复仅把五个 paired-summary
   serialized-field aliases 路由到 official short keys：
   `trans_dist`、`obj_trans_dist`、`obj_rot_dist`、`hand_pen_loss_omomo`、
   `human_pen_loss_infbagel`。Official evaluator、eval metrics、eval config 与 shared
   D2-X wrapper SHA-256 仍分别为
   `22886f8797ceb04a892487393dea9f80e19877bc02dd7a6f39127e7319119524` /
   `445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547` /
   `89c702d96b98289924225c4b163d3b29eb22efe27c50ac799ddd0c71c515aa73` /
   `b6753a66207492e6ee4addb8f450cb38c5d021401d43430faa9e5c9ed77c6e31`。
   与 sealed D2-X resolved target 对比，在排除 run/output/checkpoint identity 后无
   semantic config difference；control aggregate/per-sequence 原 hash 复用且未重生成。
4. **Native gates。** D2-AC target 的 end-object/Txy/FS/contact
   precision/recall/F1/coverage/Pbody/hand penetration/MPJPE/Troot/Tobj/Oobj 为
   `5.6473 / 4.2379 / 0.3986 / 0.7876 / 0.6042 / 0.6480 / 0.4913 /
   4.0121 / 0.2518 / 12.4268 / 8.7229 / 16.8110 / 1.0306`。
   Contact F1/recall paired differences 为 `+0.0105639`、`+0.0097080`，但 CI
   `[-0.0088320,0.0303036]`、`[-0.0124421,0.0322983]` 均包含零；released
   contact-F1 gap closure 仅 `0.1175963 < 0.25`。End-object、FS、Pbody 与 hand
   penetration protection CI upper bounds 分别为
   `1.58993 / 1.17165 / 1.19812 / 1.18712`，超过 `1.10`；181-sequence
   penetration mask contract 通过。Released-95% effectiveness gate 也失败。Evaluator
   未生成 FID、Matching、R-Precision 或 Diversity，缺失原样保留且
   `fid_rprecision_used=false`。
5. **Final decision 与 artifact recovery。** Classification precedence 由已经失败的
   internal locality gate 决定，最终严格分类为
   `interaction-adapter-locality-negative-stop`；即使 native transfer/protection/
   effectiveness 也失败，不重新命名 classification。Training/internal/native recovered
   tree SHA-256 分别为
   `d3784f0b01b8762ab1e6dcc7b0343ef2aa2147c1ca9672f516ae2f672cd92d98` /
   `62225323d8a5d3d252d34587165bd2da0ade4ed469ddae1c644e848cd391e753` /
   `83b6a811eab7e519f5f15ce2cfeb36d12bb8814625905ac7f2378caeb8fefa34`。
   Internal initial SDF failure、internal r1 zero-denominator failure、native initial
   serialized-field failure 与原 smoke 均按 append-only contract 保留，未覆盖或删除。

D2-AC0 fixed final-online checkpoint 不可选择、不可 resume、不可初始化后续 prior。
D2-AC1 只有 `interaction-adapter-positive-but-not-effective-stop` 才 eligible；当前
locality-negative classification 不满足该条件，因此 D2-AC1 严格 ineligible 且未授权。
不得自动启动 consistency、HSIPrior、Mixer、checkpoint selection、任何 adapter/token/
parameter/placement sweep、新 loss、SNR weighting、gradient projection、rollout exposure、
CFG/guidance 或新的 HOIPrior 搜索。

#### 2026-07-27 Phase 1B D2-AD0 human-local full-mesh BPS coordinate-contract repair 预注册（plan-only）

本 amendment 只注册一个由 D2-AC0 封存结果直接触发、已经得到用户确认的单变量
coordinate-contract repair，不重新开放 HOIPrior 搜索。Identifier audit 确认 D2-AD/`d2ad`
在本 amendment 前未出现在 plan、registry、source、tests 或 lifecycle id 中，因此 D2-AD
是下一个 unused Phase 1B identifier。当前 plan-only source HEAD 为
`dcf871644b6a1b72116dbab03dcc4fafc755dc28`，branch 为 `phase/01b-hoi`，authority
worktree 在修改前 clean。

1. **封存证据与可证伪假设。** D2-AC0 已证明 adapter 被优化器强烈使用：full minus
   gate-ablated direct-hand union 5-cm physical-contact F1 为 `+0.621545`，且
   gate-ablated minus full GT-contact-frame hand-object distance 为 `+90.978 cm`；
   但 full minus local-correspondence-permuted F1 仅 `+0.010392` 且 CI 包含零，
   permutation minus full distance 仅 `+0.01382 cm` 且 CI 包含零。Native contact
   F1/recall 相对 sealed D2-X 只增加约 `1.66%/1.63%`，同时 end-object 与 FS 分别退化
   `50.99%/9.81%`。因此 D2-AC0 严格保持
   `interaction-adapter-locality-negative-stop`，其 checkpoint 不得选择、resume 或初始化
   D2-AD。

   后续 code audit 发现一个更具体、可独立修复的 coordinate contract：

   - 232-D human/object state 使用 Y-up、window-local XZ origin、initial-root-yaw aligned
     frame；
   - dataset 的 author BPS delta 已执行 `zup_to_yup`；
   - D2-AC local token 的 cluster basis mean 却直接来自 raw `code/bps.pt`，仍是原
     Z-up convention；
   - 即使只把 raw basis mean 或 stored/global delta 做轴变换/旋转，也无法恢复正确
     locality，因为 fixed global queries 的 nearest-point correspondence 会随共同 global
     yaw 改变，component-wise RMS 也不是任意旋转下的不变量。

   D2-AD0 的唯一科学假设是：若 D2-AC locality failure 的主要原因是 adapter-only local
   geometry 没有与 human window-local frame 建立一致的 nearest-point correspondence，则
   在同一 full rest mesh 上直接重算 human-local BPS，应使 causal locality gate 与 native
   contact transfer 改善；这不预先声称该 coordinate mismatch 足以解释全部 HOIPrior
   baseline gap。

2. **只读 authority-CPU 原型证据。** Prototype 只读取 locked split、immutable PLY、
   `code/bps.pt` 和当前首帧 pose，没有创建 checkpoint、optimizer、CUDA workload 或
   per-window condition artifact。

   - BPS file SHA-256：
     `fdff7204b4697e105457cb7e39267b9555bc0d8d854dbc92cd67e2d8c3e77042`；
   - fixed Y-up basis float32 tensor SHA-256：
     `02b4f8f3510e723174010a823630f663ddda9875ad82a2f8de807d2bdccebd7d`；
   - raw-versus-Y-up basis/cluster-mean max abs：
     `1.3970013 / 1.1064382`；
   - sealed D2-O 64-sequence × 3-window cohort selection SHA-256：
     `1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`；
   - 192 windows cover all 13 object classes；在共同 global yaw
     `{-179,-90,-37,53,120,179}` degrees 下，full-mesh local BPS max abs
     `1.4901161e-7`，其 `[B,16,10]` cluster feature max abs
     `2.3841858e-7`；
   - exact query 的 worker-count `1/3/all` 输出逐位一致；
   - 相同 real-window probe 中，只旋转旧/global BPS 的共同 37-degree yaw max-abs
     error 平均 `0.6791 m`、最坏 `1.1257 m`；只修 basis 但保留 global delta 的
     cluster-feature max-abs error 平均 `0.1541`、最坏 `0.5767`；
   - full-local 与 rotate-old delta 的逐点 L2 差异平均 `0.5958 m`；
   - 对 192-window cohort 滚动 relative object pose 后，local BPS 逐点 L2 平均改变
     `0.1691 m`，证明 condition 不是 constant；
   - batch-grouped exact query 在 authority CPU 上对 192 windows/13 objects、
     3 query workers 耗时 `1.4742 s`；real DataLoader prototype 保留
     `batch_size=512,num_workers=4` 时测得约 `308.84 windows/s/rank` 的 condition
     delivery。该值只用于 wall-time planning；registered worker smoke 必须重新实测。

3. **唯一 manipulated factor 与精确 local-BPS 方程。** D2-AD0 相对 D2-AC0 只修复
   adapter local geometry 的 coordinate/query contract。Global BPS condition token、
   232-D state、loss、trunk、adapter 参数、placement、sampler 和 evaluator 全部保持。
   令固定 axis conversion 为 \(C_{Z\rightarrow Y}\)，raw BPS basis point 为
   \(b_i^Z\)，则 human-local fixed query 为

   \[
   b_i^L=C_{Z\rightarrow Y}b_i^Z.
   \]

   对每个 current window，令 \(W\) 为 current human frame 的 world-to-local rotation，
   \(R_O\) 为 current/global object rotation reference，immutable Y-up rest-mesh vertex
   为 \(v_j\)。定义

   \[
   L_O=WR_O,\qquad
   j^*(i)=\arg\min_j\lVert L_Ov_j-b_i^L\rVert_2^2,\qquad
   d_i^L=L_Ov_{j^*(i)}-b_i^L .
   \]

   实现允许利用旋转保距性把 query 送回 rest-object frame 后做同一 exact nearest-vertex
   查询，但输出必须与上述定义一致。共同 global yaw \(G\) 下
   \(W'=WG^{-1},R'_O=GR_O\)，所以 \(W'R'_O=WR_O\)；local BPS 及其
   component-wise RMS 必须保持不变。

4. **Immutable geometry 与 exact builder contract。** 只读取
   `data/object/rest_object_geo/*.ply` 的全部原始 vertices；禁止 100-point/1024-point/
   任何新 mesh subsample、SDF/voxel approximation、mesh encoder、category embedding、
   train-stat normalization 或 per-window local-BPS file/cache。13-file canonical
   PLY manifest SHA-256 为
   `ce8328ef2bf873a79d74fb5fd20cc488551a20d56fe5c5ecabf609824b0654d1`；
   sorted object mapping 为
   `[clothesstand,floorlamp,largebox,largetable,monitor,plasticbox,smallbox,smalltable,`
   `suitcase,trashcan,tripod,whitechair,woodchair]`，mapping SHA-256 为
   `424fc96102c576a1d11b0824cc0ee616d52cd9e39524819f49b207d1598fe41b`。

   Builder 固定使用 `scipy.spatial.cKDTree.query(k=1,eps=0,p=2)`；tree 只缓存每类
   immutable rest mesh 的 spatial index，不缓存任何 window condition。Training collate
   按 object 分组并固定 `local_bps_query_workers=3`，继续使用
   `num_workers=4`；query worker count 是 48-CPU worker 上的 operational ownership，
   不是 scientific sweep。Authority/worker 必须记录 SciPy/dependency hash，并验证
   worker-count 不改变 indices/output。输出固定 float32 `[B,1024,3]`，随后继续使用
   D2-AC 同一 16-way assignment、cluster identities、cluster sizes 与
   `[mean basis,mean delta,RMS delta,mean norm]` 10-D statistics。Cluster basis mean
   改为同一 assignment 上的 \(b_i^L\) mean；assignment 不重新按 Y-up lexicographic
   seed 派生，仍锁定：

   - centers：
     `[328,903,503,817,474,1023,382,864,640,431,445,960,547,829,545,756]`；
   - sizes：
     `[39,40,57,61,65,68,70,134,77,64,59,79,43,46,84,38]`；
   - assignment SHA-256：
     `b62f91f4eb6c4bf2a9211f0187cd1eb97c25394ee45de155f33607959fddeecd`。

5. **Training/rollout causal availability。** Training 只能从 current window 第一帧的
   human frame \(W\)、current object rotation reference \(R_O\) 和 immutable rest mesh
   构造 adapter-only local BPS。不得读取 future pose/contact、stored future/per-frame
   local BPS 或 evaluator statistics。Autoregressive rollout 的第一窗口使用 evaluator
   已提供的 current history frame；后续窗口必须从 generated two-frame history 建立新的
   `WindowFrame` 和 generated object reference，再重算 local BPS。Global BPS token 保持
   D2-X/D2-AC author semantics：第一窗口读取既有 current BPS，后续窗口沿用既有
   current-generated BPS replay。Local BPS 只送入 interaction adapter，不能进入 global
   BPS encoder、loss、evaluator threshold、HSIPrior 或 Mixer。

6. **Architecture 与 parameter lock。** 新 checkpoint variant 固定为
   `d2ad_local_frame_interaction_adapter`。它复用 D2-AC：

   - 512-wide、16-head、8-layer trunk；
   - 4 condition tokens、16 motion tokens；
   - layer 4 后、layer 5 前的单个 adapter；
   - 16 object tokens、3 roles、128 adapter width、4 attention heads、dropout 0；
   - `10→128→128` object encoder、`512→128` query、`384→512` writeback；
   - single scalar `tanh(alpha)` ReZero gate，alpha 严格从 0 初始化；
   - exact adapter/base/total parameters
     `349,697 / 29,673,448 / 30,023,145`，增量 `1.1785% <=1.25%`。

   不改变 role query、token 数、width/depth/placement、global BPS token 或任何 parameter。
   D2-AD0 全部矩阵、embedding、gate 以 seed 42 从随机初始化；不得加载 released、author、
   D2-V/X/Y/Z/AB/AC、prior、EMA、consistency 或任何 weight-init/resume checkpoint。
   D2-AC checkpoint schema 必须被 D2-AD loader 拒绝，反之亦然。

7. **保持不变的 optimization contract。** Fixed split 仍为
   `experiments/splits/omomo_hoi_train_validation_seed42.json`，SHA-256
   `019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e`。
   D2-AD0 只在 `infbagel-4gpu/node01`、4×RTX 3090 上训练；per-GPU batch 512、
   effective batch 2,048、accumulation 1；总预算
   `61,440,000 windows / 983,040,000 frames / 30,000 updates`。Optimizer 仍为
   FP32 Adam、LR `1e-4`、betas `(0.9,0.999)`、weight decay 0、无 warmup/scheduler、
   AMP、gradient clipping、EMA；primary 为 fixed final-online。FK/object-surface/
   velocity/terminal-goal weights 仍为
   `0.3569973401779424 / 0.4772322188400037 / 0.1 / 1.0`；D2-X FK-foot routing
   enabled，D2-AB support objective disabled。Formal run 必须从随机初始化持续完整预算，
   不人为 pause、不选择中间 checkpoint。

8. **Authority CPU fail-fast contract。** 任何 GPU 前必须以 authority Python 完成并归档：

   - BPS、Y-up basis tensor、split、13 PLY、mapping 与 assignment hashes；
   - raw-to-Y-up conversion、same assignment/cluster sizes、`[B,1024,3]` local BPS 与
     `[B,16,10]` features 的 shape/dtype/finiteness；
   - sealed 64×3 cohort、上述 6 yaw 的 local-BPS max abs `<=1e-6`，cluster-feature
     max abs `<=1e-6`；
   - query workers `1/3/all` indices/output exact parity；
   - repeated-call determinism、batch ordering、all 13 object coverage、relative-pose
     sensitivity、zero/constant/extreme input finiteness；
   - dataset-collate 与 evaluation helper exact parity；
   - first training window current-pose parity，以及 generated-history rollout 不读取
     future GT/stored local BPS 的 static and runtime audit；
   - exact parameter count、`[B,16,232]` output、alpha-zero shared-trunk parity
     `<=1e-6`；
   - initial alpha gradient finite/nonzero；test-only `tanh(alpha)=0.1` probe 下
     object encoder/identity/query/QKV/out/writeback gradients finite/nonzero，probe
     不保存、不训练；
   - local correspondence permutation causal effect、role separation、dtype/device/
     batch propagation；
   - base/D2-AC/D2-AD checkpoint provenance rejection；
   - HSIPrior parameter/storage/forward unchanged；Mixer 只消费 clean
     `[B,16,232]` output；
   - static scan 无 future GT、stored per-window local BPS、mesh subsample、evaluator
     threshold/helper、new loss/guidance 进入 D2-AD model path。

   任一失败在 CUDA 前分类并停止：
   `local-frame-interaction-adapter-contract-failure-stop`。

9. **Registered GPU smoke。** Implementation lifecycle 必须按真实日期绑定未使用 id。
   Smoke 只在 worker `cuda:0`，real-data batch 8、timesteps `0/249/499`、random init、
   no optimizer/update/checkpoint load/write；必须使用与 formal training 相同 exact
   full-mesh collator，记录 local-BPS construction wall time、initial model/local-BPS
   hashes、coordinate contract replay、alpha gradient、test-only nonzero-gate adapter
   gradients、CUDA-synchronized peak allocated/reserved/headroom、四卡 visibility/
   contention。Cross-attention score shape/element count不变；formal throughput/ETA 以
   实测为准，不以 authority CPU prototype 代替。

10. **Fixed internal causal diagnostic。** Training 完成后只加载 D2-AD0 fixed
    final-online checkpoint，在 sealed D2-O 64×3 cohort、phase offsets
    `(14,56,98)`、selection SHA-256
    `1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`
    上运行与 D2-AC 相同的三条 paired 500-step rollout：

    - `full`；
    - `gate_ablated`：每一步强制 `tanh(alpha)=0`；
    - `local_correspondence_permuted`：只将 human-local cluster delta statistics
      `k<-(k+8) mod 16`，保留 local basis mean、learned object identity、global BPS
      与其余 condition。

    三条 path 共享 initial latent、每步 posterior noise、condition、window ordering 与
    history restoration。指标、sequence-unit seed-42 10,000 bootstrap、attention entropy
    appendix 和 primary mechanism/locality gates 与 D2-AC 完全相同；official test 禁止，
    no optimizer/update/checkpoint write/selection。Primary gate 仍要求：

    - full minus ablated direct-hand union 5-cm physical-contact F1 CI lower `>0`；
    - full minus permuted 同一 F1 CI lower `>0`；
    - ablated minus full GT-contact-frame hand-object distance CI lower `>0`；
    - permuted minus full 同一 distance CI lower `>0`。

11. **Fixed native evaluation 与 gates。** 无论 internal 正负，都必须执行一次与 sealed
    D2-X/D2-AC protocol-identical 的 official 438 sequences × 3 windows、500-step、
    unguided production evaluation；只改变 target run/checkpoint/architecture identity 和
    adapter-only local-BPS construction。Official evaluator、metric keys、181-sequence
    penetration finite mask、bootstrap seed/replicates 不得调整。Sealed D2-X checkpoint/
    aggregate/per-sequence hashes仍为
    `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51` /
    `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
    `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`，
    不重新生成。Released aggregate hash 仍为
    `76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6`。

    Selection gates 完全复用 D2-AC：

    - D2-AD minus D2-X contact F1 与 recall paired CI lower 均 `>0`；
    - contact-F1 released-gap closure `>=0.25`，对应 point estimate 最低约
      `0.6598838781`；
    - end-object、Txy、FS、Pbody、hand penetration、MPJPE、Troot、Tobj、Oobj
      paired mean-ratio CI upper 全部 `<=1.10`；
    - contact precision difference CI lower `>=-0.02`，penetration finite-mask
      contract 通过；
    - released 95% point-effectiveness gate 保持原 lower/higher-is-better 公式。

    Native output 还必须以 sealed artifacts 对 D2-AC0 作相同 sequence-paired、仅描述性
    comparison，以量化 coordinate repair 相对唯一前驱的改变；该 secondary comparison 不
    参与 checkpoint selection，不触发额外 generation。FID/Matching/R-Precision/Diversity/
    timing 若 evaluator 生成必须原样保留和报告，FID/R-Precision 不参与 selection。

12. **分类、授权边界与 lifecycle。** Classification precedence 固定为：

    - contract failure：
      `local-frame-interaction-adapter-contract-failure-stop`；
    - adapter unused：
      `local-frame-interaction-adapter-unused-optimization-negative-stop`；
    - locality negative：
      `local-frame-interaction-adapter-locality-negative-stop`；
    - native transfer negative：
      `local-frame-interaction-adapter-transfer-negative-stop`；
    - protection conflict：
      `local-frame-interaction-adapter-conflict-negative-stop`；
    - mechanism/transfer/protection 通过但 released-95% 失败：
      `local-frame-interaction-adapter-positive-but-not-effective-stop`；
    - 全部通过：
      `local-frame-interaction-adapter-positive-candidate-stop`。

    只有最后一类允许把 fixed final-online checkpoint 标为 selectable autonomous
    HOIPrior candidate；不得选择中间 checkpoint。D2-AD0 没有自动 longer-budget
    extension 或 fallback；任何 D2-AD1、budget/LR/token/width/depth/placement/role/
    query-worker scientific sweep、新 loss、SNR weighting、gradient projection、
    rollout exposure、CFG/guidance、consistency、HSIPrior 或 Mixer 都需新的 dated plan、
    append-only registry 和用户再次明确确认。

    用户已授权在上述固定 D2-AD0 范围内连续完成 implementation、CPU tests、worker
    publication、registered smoke、from-random full training、fixed internal/native
    evaluation、artifact recovery、compact result、phase summary 与 completion record。
    本 plan-only commit 不改 source、不创建 lifecycle run、不启动 CPU contract/GPU/
    training/evaluation；implementation session 必须先重新读取真实 date，并以未使用的
    dated lifecycle ids 写 implementation binding amendment。跨日的尚未启动 lifecycle
    必须在 workload 前 append identity-only amendment，绝不复用或覆盖 id。

13. **Artifact 与 closure。** 必须保留/hash verify resolved configs、authority CPU logs、
    PLY/BPS/mapping manifests、worker preflight、smoke manifest/log/metrics、formal
    training manifest/log/state、all cadence checkpoints/per-rank RNG、initial/final model
    hashes、local-BPS construction throughput、wall time/ETA、internal full/ablated/permuted
    artifacts、paired-noise/attention appendix、native aggregate/per-sequence/bootstrap/
    penetration mask/optional metrics、run-local registry、dependency/hardware/data/evaluator
    hashes、complete recovered tree、compact result、
    `docs/phase_summaries/PHASE_1B_D2AD.md` 和全部 operational/scientific failures。
    大 artifact 不进入 Git。Logical implementation commit 必须同时含 source、config、
    tests、dated implementation amendment、registry binding 与必要 documentation。

#### 2026-07-27 Phase 1B D2-AD0 implementation/lifecycle binding amendment

Authority 在 plan-only commit
`ccc023f44056a056131c730ff39a2dfae447505b`、clean `phase/01b-hoi` 和真实日期
`2026-07-27` 上进入已授权的 D2-AD0 implementation。以下 identities 在创建本
amendment 前均未使用：

- implementation logical change：
  `p1-hoi-d2ad-local-frame-interaction-adapter-implementation-s42-20260727`；
- authority CPU contract：
  `p1-hoi-d2ad-cpu-contract-s42-20260727`；
- registered GPU smoke：
  `p1-hoi-d2ad-gpu-smoke-s42-20260727`；
- formal from-random training：
  `p1-hoi-d2ad-local-frame-interaction-adapter-s42-20260727`；
- fixed internal diagnostic：
  `p1-hoi-d2ad-local-frame-interaction-adapter-internal-s42-20260727`；
- fixed native evaluation：
  `p1-hoi-d2ad-native-eval-s42-20260727`。

Implementation logical commit 必须包含 source、config、tests、本 amendment、registry
binding 和必要 documentation；在该 committed Git object 通过 authority CPU contract 前
不得发布 worker 或启动 CUDA。若任一尚未启动 lifecycle 跨到新的真实日期，必须先追加
identity-only date-transition amendment 并 supersede 旧 identity；不得创建、复用或覆盖旧
run directory。Scope、single manipulated factor、training budget、random-init provenance、
internal/native gates 和所有 forbidden items 完全继承 D2-AD0 plan-only preregistration。
D2-AD1、checkpoint selection、consistency、HSIPrior 与 Mixer 仍未授权。

#### 2026-07-27 Phase 1B D2-AD0 implementation pre-CUDA verification

在进入正式 authority CPU lifecycle 前，D2-AD0 implementation 已完成并保持
single-factor scope。提交内容包括 local-frame full-mesh builder、D2-AD architecture/config、
training/sampler wiring、fixed internal/native wrappers、CPU tests、registry binding 与本
dated amendment；没有启动 worker publication、CUDA、optimizer、training、checkpoint
load/write、internal/native evaluation 或 selection。

Authority verification 使用指定 `infbagel` Python 完成：

- targeted D2-AD + D2-AC：42 tests passed；
- full CPU suite：352 tests passed；
- `tools/experiment.py validate`：192 registry records、2 splits、2 evaluators、
  1 training protocol valid；
- `py_compile`、`git diff --check` 与 internal/native `--resolve-only` 均通过；
- read-only CPU contract prototype 通过 sealed D2-O 64-sequence × 3-window cohort
  （selection SHA-256
  `1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`）：13/13 object classes、six common-yaw
  checks、query workers 1/3/all、repeated/batch-order determinism、training/evaluator
  parity、generated-history recomputation、parameter/gradient/provenance/static gates；
  local-BPS 与 `[B,16,10]` feature common-yaw max abs 分别为
  `1.1920928955078125e-7` 与 `2.384185791015625e-7`。

该 prototype 未使用 official test、未创建 checkpoint/optimizer、未写入 lifecycle
artifact，也不替代后续正式 CPU manifest。任何正式 GPU workload 仍须在 committed clean
object、same-context preflight 与 registered smoke 通过后才可开始。

#### 2026-07-28 Phase 1B D2-AD0 unstarted lifecycle date-transition amendment

真实日期已跨至 `2026-07-28`。07-27 implementation identity
`p1-hoi-d2ad-local-frame-interaction-adapter-implementation-s42-20260727`
继续记录实际 implementation-start，不改名也不覆盖。跨日前尚未启动的 authority CPU、
GPU smoke、formal training、internal 与 native lifecycle 均未创建 manifest、run directory
或 workload，因此按既定 date-transition rule 将以下旧 identity 标为 superseded：

- `p1-hoi-d2ad-cpu-contract-s42-20260727`；
- `p1-hoi-d2ad-gpu-smoke-s42-20260727`；
- `p1-hoi-d2ad-local-frame-interaction-adapter-s42-20260727`；
- `p1-hoi-d2ad-local-frame-interaction-adapter-internal-s42-20260727`；
- `p1-hoi-d2ad-native-eval-s42-20260727`。

后续唯一有效且此前未使用的 lifecycle identities 为：

- authority CPU contract：
  `p1-hoi-d2ad-cpu-contract-s42-20260728`；
- registered GPU smoke：
  `p1-hoi-d2ad-gpu-smoke-s42-20260728`；
- formal from-random training：
  `p1-hoi-d2ad-local-frame-interaction-adapter-s42-20260728`；
- fixed internal diagnostic：
  `p1-hoi-d2ad-local-frame-interaction-adapter-internal-s42-20260728`；
- fixed native evaluation：
  `p1-hoi-d2ad-native-eval-s42-20260728`。

本 amendment 只改变未启动 lifecycle 的日期 identity。D2-AD0 的 single manipulated
factor、source implementation、random initialization、training budget、loss/optimizer、
internal/native evaluator、gates、classification precedence、artifact contract 与全部
forbidden items 均不变；不授权 D2-AD1、checkpoint selection、consistency、HSIPrior、
Mixer 或任何 sweep。

#### 2026-07-28 Phase 1B D2-AD0 completion record

D2-AD0 的全部已批准 lifecycle 已完成。Tracked compact result 为
`experiments/results/p1_hoi_phase1b_d2ad_local_frame_interaction_adapter_s42_20260728.json`，
phase summary 为 `docs/phase_summaries/PHASE_1B_D2AD.md`。本 record 只封存固定
coordinate-contract repair、运维失败、科学结果与 artifact；不新增 fallback 或研究方向。

1. **CPU、smoke 与 formal training。** Authority CPU contract
   `p1-hoi-d2ad-cpu-contract-s42-20260728` 以 352 tests、exact
   parameter/API/base-parity、coordinate equivariance、query-worker/dataset/evaluator
   parity、activated gradients、provenance、HSIPrior/Mixer independence 和 static scan
   全部通过。Registered smoke `p1-hoi-d2ad-gpu-smoke-s42-20260728` 在
   `cuda:0`、real-data batch 8、timesteps `0/249/499` 上通过，peak
   allocated/reserved/headroom 为
   `252,609,024 / 304,087,040 / 24,991,956,992` bytes，且没有 optimizer 或
   checkpoint activity。Formal training
   `p1-hoi-d2ad-local-frame-interaction-adapter-s42-20260728` 从随机初始化完成
   `61,440,000` windows / `983,040,000` frames / `30,000` updates，wall time
   `47,890.633 s`、throughput `1,282.923 windows/s`；20 cadence checkpoints 与
   80 rank RNG sidecars 完整。Learned alpha/gate 为
   `0.10238598 / 0.10202970`，fixed final-online SHA-256 为
   `f527d970243a42a1534b8db4437cd09dbc25334c832c3a13eb011f81db101c06`。
2. **Fixed internal diagnostic。** 原 internal identity 因 preflight 接收到错误的
   CHOIS asset directory 而在 manifest/workload 前停止，1-file failure tree
   `d0eda6ede4e692acb2ca52ed8286ba4e122b0fc1e4edc2845946d03714898a47`
   原样保留。Corrected retry
   `p1-hoi-d2ad-local-frame-interaction-adapter-internal-r1-s42-20260728`
   从头完成 sealed 64×3 cohort 的三路 paired 500-step rollout。Full minus
   gate-ablated direct-hand union 5-cm F1 为 `+0.6274720`，95% CI
   `[0.5403343,0.7116109]`；gate-ablated minus full GT-contact distance 为
   `+97.93614 cm`，CI `[87.31619,108.58261]`，证明 adapter 被使用。Full minus
   locality-permuted F1 为 `+0.0135183`，CI
   `[-0.0062331,0.0342150]`；permuted minus full distance 为
   `+0.143274 cm`，CI `[-0.057880,0.354987]`，两个 locality gate 均失败。
3. **Fixed official native evaluation。**
   `p1-hoi-d2ad-native-eval-s42-20260728` 完成 official 438×3、500-step
   unguided evaluator，复用 sealed D2-X control，未重生成 control，D2-AC 只作 sealed
   descriptive comparison。D2-AD end-object/Txy/FS/contact
   precision/recall/F1/coverage/Pbody/hand penetration/MPJPE/Troot/Tobj/Oobj 为
   `4.2373 / 4.8036 / 0.42539 / 0.76795 / 0.53300 / 0.58687 /
   0.43497 / 3.4625 / 0.21656 / 12.3847 / 9.2747 / 16.4076 / 1.01478`。
   相对 D2-X，contact F1/recall differences 为
   `-0.0505537 / -0.0614576`，95% CI 分别为
   `[-0.0713216,-0.0293031] / [-0.0852768,-0.0377657]`；released contact-F1
   gap closure 为 `-0.562760`。End-object/Txy/FS/Troot protection ratio CI upper
   为 `1.19830 / 1.23695 / 1.25273 / 1.16017`，均超过 `1.10`；
   precision difference CI lower `-0.0418402 < -0.02`。Native transfer、
   protection 和 released-95% effectiveness gates 全部失败；181-sequence
   penetration mask contract 通过。Evaluator 未生成 FID、Matching、R-Precision 或
   Diversity，未代填。
4. **Final decision 与 recovery。** Classification precedence 由固定 internal
   locality failure 决定，最终为
   `local-frame-interaction-adapter-locality-negative-stop`。CPU/smoke/training/
   internal/native recovered tree SHA-256 分别为
   `514163cc45801253f19dbb6e1789464e791f59a00aa6f1b44cdadf9f348eb7ce` /
   `85ef57f3874ab113d4cac75b813259fb61ae5cff5d1b24ed9078b924223c621a` /
   `d694962309735ecae12f4480d4dcb52c8d191a9a453603fefd8e5f4bbd18b656` /
   `4b80a78745de4d3fecc23399f023d736d4b5ff1f9e7d12e043e70e6bf27055e3` /
   `6d0bcf47eac49aaf1a10341d81bc8d4f1a518ed86344fd145283b17c236c7d0c`，
   worker/authority 一致。

D2-AD0 fixed final-online checkpoint 不可选择、不可 resume、不可初始化后续 prior。
本 D2-AD0 计划没有 D2-AD1/longer-budget fallback；任何新机制、预算或参数方向都必须先有
新的 dated plan、append-only registry hypothesis 和用户明确授权。不得自动启动
checkpoint selection、consistency、HSIPrior、Mixer 或任何 sweep。

#### 2026-07-28 Phase 1B D2-AE0 GPU-native sparse current-state role-relative object-field routing 预注册（plan-only）

本 amendment 只注册用户明确授权的一个新单变量 HOIPrior 实验，不重新开放 HOIPrior
search。修改前的 authority 为 clean `phase/01b-hoi`、HEAD
`45b59330f6d09da9050cedb01e5edb7fa5deefda`（`Close Phase 1B D2-AD0`）。Identifier audit
对 tracked/untracked operational text、全部 reachable Git history/diffs/refs/reflogs、registry、
authority/worker staging 文件名和现有 lifecycle identities 做了大小写不敏感扫描，确认
D2-AE、`d2ae`、`D2-AE0`、`p1-hoi-d2ae-*` 与 `sparse-relation-field` 尚未使用。Locked baseline
`b9a158f75ab0740c91c9cfc8863a65fa381b014c` 是 HEAD ancestor；
`feature/independent-hoi-hsi-priors` 既不是 ancestor，也没有 patch-equivalent cherry-pick。

1. **封存证据与可证伪假设。** D2-AC0/D2-AD0 的 adapter whole-gate ablation 很强，但
   correspondence permutation 基本无效，且 native contact transfer 变差；D2-AD0 formal
   throughput 还因 CPU full-mesh KD-tree path 从 D2-X 的
   `3,243.0357134915853 windows/s` 降到 `1,282.923 windows/s`。作者 released InfBaGel 的
   `occ_temp` 路径说明 current-state spatial relation 配合固定 temporal routing 可能是有用
   归纳偏置，但其 training 使用 noisy object pose 与 clean future `x_start` human/grid anchor，
   sampling 又使用 current `x` object pose 与 previous `x0` human anchor，并读取 synthesized
   `Scene*`。D2-AE0 的唯一假设是：若每个 diffusion step 只从同一当前 `x_t` 构造结构性绑定
   left hand/right hand/pelvis 与当前 sparse object surface 的明确相对场，并在 trunk 前按固定
   temporal segments 写回，则 relation path 的 temporal correspondence 与 role identity 会在
   paired causal diagnostic 中成为必要信息，并在不引入 scene/future leakage 或 CPU dynamic
   geometry 的情况下改善 D2-X native contact transfer。

2. **唯一 manipulated factor 与保持项。** D2-AE0 相对 sealed D2-X 只增加一个
   GPU-native sparse current-state role-relative object-field residual。保持 `[B,16,232]`
   clean-output API、232-D field semantics、16-frame window、2-frame history restoration、
   500-step clean-x0 diffusion、512-wide/16-head/8-layer trunk、原四个 condition tokens、
   global BPS token、D2-X FK-foot temporal routing、全部既有 losses/reductions/weights、
   optimizer/LR/batch/split/budget/sampler、official evaluator，以及 HSIPrior/Mixer clean-output
   contract 不变。不得扩大 D2-AC/D2-AD adapter，也不得改变 point count、width、depth、role、
   placement、anchor、batch、LR、loss、threshold 或训练预算。

3. **Current-state relation source 与 immutable sparse asset。** Relation builder 只接收当前
   diffusion state `x_t [B,16,232]`、现有 `rest_object_points [B,100,3]`、
   `world_to_local_rotation [B,3,3]`、`object_rotation_reference [B,3,3]`，以及 locked
   position/object normalization tensors。D2-X data path 对每个 immutable rest mesh 使用
   `trimesh.load_mesh(process=False)`、float32 Z-up→Y-up，再以
   `linspace(0,N-1,100).round()` 选择固定 vertices；13 个对象在 real D2-X batch 中均为
   100 points，并与该重建 byte-exact。以下 canonical hashes 在本 plan 中锁定：

   - object-name mapping（`sequence-name-second-underscore-field-v1`）：
     `1af35119c1dd54e2ad44c99f3cb91b62c1b88f62ca80cddcc96f4b201ffe0f5b`；
   - per-object source/count/index/point manifest
     （`d2x-rest-object-points-100-yup-linspace-vertex-v1`）：
     `e88d74a7ee434f3e6320c95d1ebb74efdc8fe4740b70ff596e502666a096f7a7`；
   - stacked tensor `[13,100,3]` in fixed object-name order：
     `793dad6a805d0a908087b273590bf171e7bce4c026297cf94d40f8c651fe4cab`。

   Training 已直接提供这 100 points；native evaluator 已加载同一 immutable rest meshes，
   sampler 只允许在 diffusion loop 前以相同固定 indices 建立/缓存 13 个 100-point tensors，
   随后每个 batch 只传 `[B,100,3]`。这不是 full-mesh nearest query、per-window relation cache
   或 dynamic CPU geometry。Relation math 本身必须是 train/sample 共用的 pure PyTorch
   function；不得使用 SciPy、NumPy、trimesh、KD-tree、full-mesh `cdist`、dense occupancy、
   stored future relation 或 collator-side dynamic geometry。

4. **精确几何方程。** 固定 temporal anchors `F=(0,5,10,15)`；固定 roles 按顺序为
   `(left_hand_direct_joint_24,right_hand_direct_joint_26,pelvis_joint_0)`。对每个 anchor
   `tau`，从当前 `x_t` 反归一化 28 个 local joints `J_tau` 与 object translation `o_tau`，
   并计算

   `R_local_tau = world_to_local_rotation @ project_to_so3(R_relative_tau @ object_rotation_reference)`。

   当前 sparse object surface 为
   `S_tau,n = P_rest,n @ R_local_tau^T + o_tau`。实现顺序必须与现有
   `hoi_training_losses()` object-surface transform 逐点 parity：先投影 relative，随后
   `project_to_so3(relative @ reference)`，再左乘 `world_to_local`，最后
   `einsum("bpc,btdc->btpd") + translation`。Role anchor 分别为
   `J_tau,24/J_tau,26/J_tau,0`；每点 feature 固定为
   `[delta_x,delta_y,delta_z,||delta||_2]`，其中 `delta=S-h`。不得加入 contact label、
   SDF、penetration label、scene/category embedding 或任何 threshold。

5. **固定 sparse field encoder。** 所有 temporal/role sets 共享 point encoder
   `phi: 4 -> 128 -> 128`，每层后使用 SiLU。每个 role 只做 point-set mean/max pooling，
   得到 256-D；三个 role 按 left/right/pelvis 顺序结构性拼接为 768-D。Relation vector 为
   `r_tau = LN(W_r g_tau + e_tau)`，其中 `W_r:768->512`，`e_tau` 是四个 learned
   temporal-slot embeddings。不得复用 D2-AC 的“同一 motion token 加 additive role
   embedding”设计。Point order permutation invariance 是预期性质，不得作为 locality
   ablation。

6. **固定 temporal writeback 与参数预算。** 保持原 20-token sequence，不插入 occupancy
   tokens。先算 `H_t=motion_input(x_t)`，再以固定 mapping
   `0..4->0, 5..9->5, 10..14->10, 15->15` 写回
   `H'_t = H_t + tanh(alpha) * r_a(t)`。`alpha` 为单 scalar 且严格初始化为 0；writeback
   位于 condition concat/position embedding 与完整 8-layer trunk 之前。`alpha=0` 时必须与
   共享 D2-X trunk `eval()` max-abs parity `<=1e-6`。所有 sparse-field 和 trunk 参数均由
   seed 42 随机初始化；不得加载 released/author/D2-V/X/Y/Z/AB/AC/AD/prior/EMA/
   consistency checkpoint。Exact parameter contract 为：point encoder `17,152`、projection
   `393,728`、temporal embeddings `2,048`、LayerNorm `1,024`、alpha `1`，increment
   `413,953`；base `29,673,448`；total `30,087,401`；increase `1.3950283%`，硬上限
   `1.50%`。若 CPU 实测不一致或超限，GPU 前分类
   `sparse-relation-field-contract-failure-stop`，不得改 width。

7. **独立 architecture/provenance 与 train/sample symmetry。** D2-AE 使用独立
   architecture variant `d2ae_sparse_relation_field` 与独立 checkpoint contract；released、
   D2-X/base、D2-AC、D2-AD schema 必须 fail-closed。Training 的 relation 只从
   `GaussianDiffusion.q_sample()` 产生的 current `noisy` 构造，绝不读取 clean target。
   Sampling 的每个 500-step model call 只从当步 current `x_t` 构造同一 relation，不得使用
   previous predicted clean `x0` 作为专有 condition。两条路径共享同一 builder、normalization、
   100-point tensor 和 frame/reference contract。HSIPrior 不接受该 variant、不共享参数或
   storage；未来 Mixer 仍只接收 clean `[B,16,232]`。

8. **Authority CPU hard gate。** 任何 GPU workload 前，registered authority CPU lifecycle
   必须完成并归档：path/branch/commit/clean/date；identifier/provenance；100-point
   asset/mapping/tensor hashes；loss surface parity；common global-yaw invariance；relative
   translation/rotation sensitivity；left/right swap 精确 block exchange；nonzero-gate temporal
   anchor permutation sensitivity；point-order invariance；zero/constant/extreme noisy-state 与
   SO(3) finiteness；dtype/device/batch propagation；exact parameter/API/base parity；initial
   alpha finite/nonzero gradient；test-only `tanh(alpha)=0.1` 下 point encoder/projection/
   temporal embeddings/relevant trunk gradients finite/nonzero；probe 不保存且不进入训练；
   train/sampler builder parity；relation source 无 clean/future/Scene/contact；checkpoint
   rejection；HSIPrior independence；Mixer clean API；forbidden-path static scan；full authority
   suite、registry validation 和 `git diff --check`。任一失败立即停止为
   `sparse-relation-field-contract-failure-stop`。

9. **固定 lifecycle identities。** 本 plan-only commit 不实施 source、不启动 CPU/GPU
   workload。Implementation commit 后只允许绑定以下本日未使用 IDs；跨日或 unstarted
   preflight failure 必须 append-only supersede/`-r1`，不得覆盖：

   - plan：`p1-hoi-d2ae-sparse-relation-field-preregister-s42-20260728`；
   - implementation：`p1-hoi-d2ae-sparse-relation-field-implementation-s42-20260728`；
   - CPU：`p1-hoi-d2ae-cpu-contract-s42-20260728`；
   - functional smoke：`p1-hoi-d2ae-gpu-functional-smoke-s42-20260728`；
   - performance：`p1-hoi-d2ae-performance-benchmark-s42-20260728`；
   - formal：`p1-hoi-d2ae-sparse-relation-field-s42-20260728`；
   - internal：`p1-hoi-d2ae-sparse-relation-field-internal-s42-20260728`；
   - native：`p1-hoi-d2ae-native-eval-s42-20260728`；
   - completion：`p1-hoi-d2ae-completion-s42-20260728`。

10. **Single-GPU functional smoke。** 在 exact committed clean worker object 上，以 verified
    worker Python、`INFBAGEL_WORKER_EXPERT=hoi`、real-data batch 8、timesteps `0/249/499`、
    seed 42、random initialization 执行。不得创建 optimizer、update 或 checkpoint；必须记录
    relation values/shapes、alpha gradient、test-only activated gradients、loss/model finiteness、
    peak allocated/reserved/headroom、visible GPUs、resolved config、same-context preflight 与
    manifest。失败按 contract failure 停止。

11. **4-GPU full-micro-batch performance hard gate。** Formal training 前必须在
    `infbagel-4gpu/node01`、4×RTX 3090、per-GPU batch 512/effective 2048、FP32 Adam、seed 42、
    random initialization 上完成独立 sacrificial benchmark：64 warm-up + 256 measured = 320
    updates；measured windows `524,288`；不加载/保存 checkpoint，benchmark weights 不复用。
    CUDA timing必须同步，分别报告 loader wait、H2D、GPU relation build、forward、backward、
    optimizer、DDP、peak allocated/reserved/headroom、CPU/GPU utilization、contention 和
    intermediate shapes。Measured throughput 必须
    `>=2756.580356467847 windows/s`（sealed D2-X 的 85%），完整预算 ETA 必须
    `<=6.20 h`；每卡 headroom 必须 `>=max(2 GiB,10% device memory)`，loss/gradients finite，
    且无 CPU dynamic geometry。失败即
    `sparse-relation-field-performance-negative-stop`；保留全部 benchmark artifacts，不得通过
    point/width/depth/role/anchor/placement/batch/loss/budget 或 workers/threads sweep 重试。

12. **Formal from-random training（仅 performance pass）。** 固定 split
    `experiments/splits/omomo_hoi_train_validation_seed42.json`，SHA-256
    `019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e`；只在
    `infbagel-4gpu/node01` 4×RTX 3090 运行。Per-GPU batch 512、effective 2048、accumulation 1；
    `61,440,000` windows / `983,040,000` frames / `30,000` updates；FP32 Adam、LR `1e-4`、
    betas `(0.9,0.999)`、weight decay 0、no warmup/scheduler/AMP/clipping/EMA；primary
    final-online；FK/object-surface/velocity/terminal-goal weights
    `0.3569973401779424/0.4772322188400037/0.1/1.0`；D2-X FK-foot routing on，D2-AB/new
    losses off。First start 的 init/weight-init/resume 全空，所有旧 model/optimizer/RNG/EMA/
    scaler/scheduler load count 为 0。必须完整跑完预算，不得选择中间 checkpoint。通过 initial
    stability、memory headroom、finite required gradients 和 resumable checkpoint 后记录实测
    throughput/ETA/checkpoint hash，并按 multi-server policy 让 worker-owned persistent session
    独立完成；不得因 control tunnel 中断 restart、复用 run id 或覆盖。

13. **Fixed internal causal diagnostic。** 只加载 fixed final-online，在 sealed D2-O
    64 sequences×3 windows、phase offsets `(14,56,98)`、selection SHA-256
    `1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a` 上运行四条 paired
    500-step rollouts：`full`；每步 gate 强制 0 的 `relation_gate_ablated`；geometry anchor
    block `k<-(k+2) mod 4` 但 target temporal embedding/routing slot 不变的
    `temporal_correspondence_permuted`；projection 前交换 left/right pooled geometry blocks 的
    `left_right_role_swapped`。四路共享 initial latent、每步 posterior noise、condition、ordering
    与 history restoration；official test 禁止，无 optimizer/update/checkpoint write/selection。
    统计 unit 为 sequence、seed 42、10,000 paired bootstrap。必须报告 semantic/direct/FK-palm
    contact、多阈值、coverage/run length、GT-contact distance、penetration、MPJPE、object/pelvis
    goal、FS、alpha/gate、temporal/role block norm/variance/permutation sensitivity 与 paired
    uncertainty。Primary gates 固定为：full-gate-ablated direct union 5-cm F1 CI lower `>0`；
    full-temporal-permuted 同指标 CI lower `>0`；full-role-swapped direct left/right macro-F1
    CI lower `>0`；gate-ablated-full 与 temporal-permuted-full GT-contact mean distance CI lower
    均 `>0`。Classification precedence 依次为 unused、temporal negative、role negative；无论
    internal 正负都继续一次 fixed native evaluation。

14. **Fixed native evaluation 与 selection gates。** 完全复用 D2-AC/D2-AD protocol：official
    438 sequences×3 windows、500-step unguided production diffusion、final-online、seed 42、
    10,000 paired sequence bootstrap；CFG/guidance/scene/dynamic perception/consistency 全 off；
    不重新生成 sealed D2-X。Control checkpoint/aggregate/per-sequence SHA-256 分别为
    `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51` /
    `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
    `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`；released aggregate
    `76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6`；penetration 使用
    sealed 181-sequence finite mask SHA-256
    `2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec`。
    Native transfer 要求 contact F1/recall difference CI lower `>0` 且 released gap closure
    `>=25%`（F1 point estimate 约 `>=0.6598838781`）。Protection 继续要求 end-object/Txy/FS/
    Pbody/hand penetration/MPJPE/Troot/Tobj/Oobj paired mean-ratio CI upper `<=1.10`，contact
    precision difference CI lower `>=-0.02`，penetration mask contract 通过；released-95%
    effectiveness gate不变。D2-AC/D2-AD 只作 sealed descriptive evidence；FID/R-Precision
    即使生成也不参与 selection；evaluator 生成什么就保留什么，不删除、补值或改 metric math。

15. **最终分类、artifact 与停止边界。** Classification precedence 固定为：
    `sparse-relation-field-contract-failure-stop`；
    `sparse-relation-field-performance-negative-stop`；
    `sparse-relation-field-unused-optimization-negative-stop`；
    `sparse-relation-field-temporal-routing-negative-stop`；
    `sparse-relation-field-role-binding-negative-stop`；
    `sparse-relation-field-transfer-negative-stop`；
    `sparse-relation-field-conflict-negative-stop`；
    `sparse-relation-field-positive-but-not-effective-stop`；
    `sparse-relation-field-positive-candidate-stop`。只有最后一类可把 fixed final-online 标为
    selectable autonomous diffusion HOIPrior candidate。所有 operational/scientific failures、
    resolved configs、same-context preflights、manifests/logs/profile、checkpoints/RNG、paired noise、
    internal/native raw/summary/bootstrap、mask、optional evaluator outputs、run-local registry、
    dependency/hardware/data/evaluator hashes 与 recovered trees 必须保留并做 worker/authority
    unified `sha256_path` 核验；worker 发起 non-destructive rsync，禁止 `--delete`。最终写 compact
    result、`docs/phase_summaries/PHASE_1B_D2AE.md` 与 append-only completion record。
    D2-AE1、longer budget、任何 sweep、D2-AC/D2-AD retrain/resume/selection、new loss、SNR/
    timestep weighting、gradient projection、rollout exposure、CFG/guidance、distillation、
    HSIPrior、Mixer、scene encoder/Scene*/occupancy、future clean/GT/stored relation 均未授权，
    不得自动启动。

#### 2026-07-28 Phase 1B D2-AE0 implementation / pre-GPU lifecycle binding amendment

本 amendment 实现且只实现上节 plan-only 已锁定的 D2-AE0 机制，并在任何 reportable CPU/GPU
workload 前封存 source、config、tests 与 lifecycle hard binding。Implementation source head 为
`eded185f7e5ba075ba83fde97282cb1464ddb08f`（`Preregister D2-AE sparse relation routing`）；
截至本记录，authority 未创建 optimizer、未加载/写入 checkpoint，worker publication、CPU
contract、functional smoke、performance benchmark、formal training、internal 与 native 均未启动。

1. **实现边界。** 新 architecture variant `d2ae_sparse_relation_field` 使用 train/sample 共用的
   pure-PyTorch builder，只从当前 `x_t`、现有 immutable `[B,100,3]` rest-object points、
   history-derived window/reference rotations 与 locked normalization 构造
   `[B,4,3,100,4]` role-relative point field。共享 `4->128->128` encoder、mean/max pooling、
   fixed left/right/pelvis concatenation、`768->512` projection、four temporal embeddings、LN 与
   zero-init scalar ReZero gate 按 `0/5/10/15` segments 在全部八层 trunk 前写回。普通
   `PriorWindowDataset` 保持不变；没有 D2-AD collator、SciPy/KD-tree/full-mesh query、CPU dynamic
   geometry、dense occupancy、Scene、contact/clean/future/stored relation 或 evaluator change。
2. **参数、API 与 provenance。** CPU recomputation 锁定 base/increment/total 为
   `29,673,448 / 413,953 / 30,087,401`，increase `1.3950283% <= 1.50%`；输出仍为
   `[B,16,232]`。Seed-42 shared D2-X state 共 119 keys byte-exact，只有 10 个 sparse-field keys
   新增，alpha 初始精确为 0。Released、D2-X、D2-AC、D2-AD schemas 均 fail-closed；resume 还必须
   将 checkpoint 自报的 random/no-source/no-old-state provenance 与当前 fresh seed-42 D2-AE
   initial state hash 精确匹配。HSIPrior 参数/storage independence 与 Mixer clean-output API 不变。
3. **Train/sample 与 causal diagnostic binding。** Training relation 只从 `q_sample()` 返回的
   current noisy state 建立；500-step sampler 每步只用当步 current state，并通过真实
   `HOIPriorSampler` metadata reconstruction 与 real `PriorWindowDataset` window 对七项 metadata、
   surface/features 做 exact parity。Internal runner 固定四路、共享 initial latent 与 499 次
   posterior draws；首窗 exogenous condition/history 共享，分叉后 frame/BPS/local-goal/relation
   metadata 保持各 path-local。Sealed `(14,56,98)` cohort runtime proof 锁定 source starts
   `(0,42,84)`，前窗 sampled tail `[start+42,start+45]` 精确成为下一窗 history。
4. **Performance hard gate 现在 fail-closed。** Registered 4-GPU benchmark 保持
   `4x512`、64 warm-up、256 measured、FP32 Adam，并记录 loader/H2D/relation/forward/backward/
   optimizer/DDP、四 rank relation shapes/device、memory、utilization 与 compute contention。
   Benchmark CLI 必须显式接收 actual-date performance run id 与 intended formal run id（含可选
   严格 `-rN`），summary 中的 `formal_run_id` 必须与 formal config `run_id` 精确一致且两者
   与 benchmark run id 使用同一实际日期。Formal config 新增必填 benchmark summary absolute
   path 与 SHA-256；trainer 必须验证 passing
   classification、`>=2756.580356467847 windows/s`、ETA `<=6.20 h`、headroom、finite losses/
   gradients、四 rank GPU-only relation、无外部 compute contention、零 checkpoint activity，
   并验证 benchmark commit 是 current commit ancestor 且 benchmark/formal tracked runtime source-tree
   hash 完全一致。缺失、tamper 或任何 gate failure 都在 optimizer/GPU training 前拒绝。
5. **Lifecycle identity 与 verification。** CPU/smoke/performance/internal/native IDs 使用 locked stem、
   actual start date 和可选严格 `-rN`；fresh formal start 也要求 actual date，same-run resume 则保留
   checkpoint-bound 原 run id，允许跨午夜而不伪造新 identity。Authority 已通过 D2-AE targeted
   `26/26`、D2-AC/D2-AD/independent/remediation/D2-T/D2-U regressions `115/115`、full suite
   `378/378`（authority 未启用 worker-only LINGO skip）、`py_compile`、registry validation
   （implementation record 前 200 records；包含该 record 后 201 records）与
   `git diff --check`。HOI worker 环境预期在同一 378 项中 skip 2 个 real-LINGO-only tests；
   Official evaluator 与 locked
   metric/helper sources未修改。

下一步只能先提交本 logical implementation，使 authority clean；随后以
`p1-hoi-d2ae-cpu-contract[-rN]-s42-<actual-date>` 注册并执行 authority CPU hard gate。只有 CPU、
single-GPU functional smoke 和 hash-bound 4-GPU performance gate 全部通过，formal training 才能
启动；performance negative 时必须立即按已注册分类停止，不得 sweep 或修改机制。

#### 2026-07-29 Phase 1B D2-AE0 unstarted lifecycle date-transition amendment

真实日期已跨至 `2026-07-29`。Authority 在追加本 amendment 前为 clean
`phase/01b-hoi@993934cb1d27a2fb406b4d3640eda90d8737767a`。已经启动或完成的
2026-07-28 lifecycle identities 全部保持不变，特别是 formal retry
`p1-hoi-d2ae-sparse-relation-field-r1-s42-20260728`：它于
`2026-07-28T19:43:17+08:00` 从随机初始化启动，于
`2026-07-29T00:49:32+08:00` 在同一 run id 下完成，exit code `0`，没有 resume、重训或
checkpoint selection。Formal 完成 `61,440,000` windows / `983,040,000` frames /
`30,000` updates，final-online SHA-256 为
`b7d49046504e9f8367bfd2bce0aeefb1c8590bf9c542b6eed637f05bdfcdd840`。

完整 formal tree 已由 worker 发起 non-destructive recovery 到
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ae-sparse-relation-field-r1-s42-20260728-recovery-r1`；
worker/authority 统一 `sha256_path` 均为
`3c8a987d54dfb63e89d7ec243fb065dc4f84c95808d92eee13b46ab621959428`
（119 files / 7,226,999,632 bytes），随后 checksum dry-run 为零传输。较早的
118-file pre-run-local-registry snapshot
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ae-sparse-relation-field-r1-s42-20260728`
也原样保留，tree SHA-256
`420e2f89d8059e4d9b5d0249001fbb9dbaffd5e591990f8ba7d6fbcdf6e44ae6`；
不得删除或覆盖任一 recovery evidence。

以下已经实际发生的 identities 继续作为唯一历史记录，不得改名、复用或 supersede：

- plan `p1-hoi-d2ae-sparse-relation-field-preregister-s42-20260728`；
- implementation `p1-hoi-d2ae-sparse-relation-field-implementation-s42-20260728`；
- CPU `p1-hoi-d2ae-cpu-contract-s42-20260728`；
- failed functional smoke base 与 completed retry
  `p1-hoi-d2ae-gpu-functional-smoke[-r1]-s42-20260728`；
- completed performance base
  `p1-hoi-d2ae-performance-benchmark-s42-20260728`；
- failed formal base
  `p1-hoi-d2ae-sparse-relation-field-s42-20260728`；
- completed performance retry
  `p1-hoi-d2ae-performance-benchmark-r1-s42-20260728`；
- completed formal retry
  `p1-hoi-d2ae-sparse-relation-field-r1-s42-20260728`。

跨日前从未创建 manifest、run directory 或 workload 的旧 internal、native 和 completion
identities 现被永久 supersede：

- `p1-hoi-d2ae-sparse-relation-field-internal-s42-20260728`；
- `p1-hoi-d2ae-native-eval-s42-20260728`；
- `p1-hoi-d2ae-completion-s42-20260728`。

后续唯一有效且 identifier audit 确认未使用的 identities 为：

- fixed internal：
  `p1-hoi-d2ae-sparse-relation-field-internal-s42-20260729`；
- fixed native：
  `p1-hoi-d2ae-native-eval-s42-20260729`；
- completion：
  `p1-hoi-d2ae-completion-s42-20260729`。

本 amendment 只改变尚未启动 lifecycle 的日期 identity。D2-AE0 mechanism、final-online
checkpoint、sealed cohort、四条 internal paths、500-step sampler、native evaluator、bootstrap、
threshold、gate、classification precedence 与 artifact contract 全部不变；不授权新的
performance benchmark、fresh formal training、resume、checkpoint selection、D2-AE1、
longer budget、sweep、consistency、HSIPrior 或 Mixer。

#### 2026-07-29 Phase 1B D2-AE0 fixed internal causal diagnostic completion

Fixed internal lifecycle 在 clean worker `phase/01b-hoi@190d95d1c634299407b398946b2a01d5737b45d7`
上执行。Base identity
`p1-hoi-d2ae-sparse-relation-field-internal-s42-20260729` 与 retry `-r1`
均在 manifest 和 GPU workload 前停止：两次 preflight 均确认四卡显存、利用率、compute
process、Git、Python、数据、CHOIS checkout/checkpoint 与 NTP contract 正常，但单次快照分别
观察到 GPU 1 和 GPU 3 为瞬时 `P5`，利用率仍为 `0%`、compute process 为空。两个目录不覆盖、
不复用，分别以 2 files / 10,311 bytes /
`015e180d5aa21f093fe7f712d576150f12d47203aac26269f28f56c0015336e3` 和
2 files / 10,320 bytes /
`88f20c8ba3f0c013ba475e04551706ce2194c1904d33db2738dde497175de8bd`
原样保留。随后 20 次一秒间隔只读采样均为四卡 `P8`、`0%`，未改变 idle 判据；`-r2`
从头生成 resolved config 和 preflight，未复用任何 partial output。

成功 run
`p1-hoi-d2ae-sparse-relation-field-internal-r2-s42-20260729` 只加载 fixed final-online
checkpoint SHA-256
`b7d49046504e9f8367bfd2bce0aeefb1c8590bf9c542b6eed637f05bdfcdd840`，
在 sealed D2-O 64-sequence / 192-window cohort
`1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`
上完成四条 paired 500-step causal rollout。Exit code 为 `0`，runtime
`332.670974 s`；29 项 runtime contract 全真，四路径各 64 sequences、24 causal
batch-windows、每 window 500 次 relation forward，paired noise/exogenous condition/initial
history、57-sequence GT-contact finite mask、causal overlap、history restoration、model-state
unchanged 与 GPU current-state relation capture 全部通过。未创建 optimizer，training update、
checkpoint write、checkpoint selection 与 official-test use 均为零。

五个 primary mechanism gates 全部通过：

| comparison | paired point | sequence-bootstrap 95% CI |
|---|---:|---:|
| full − gate-ablated direct-hand union 5-cm F1 | +0.236691 | [0.148411, 0.326983] |
| full − temporal-permuted direct-hand union 5-cm F1 | +0.153893 | [0.081493, 0.226123] |
| full − left/right-swapped direct-hand macro-F1 | +0.178708 | [0.122784, 0.232256] |
| gate-ablated − full GT-contact-frame distance (cm) | +3.509101 | [2.090270, 4.957889] |
| temporal-permuted − full GT-contact-frame distance (cm) | +4.010482 | [2.072867, 6.222641] |

因此 internal classification 为
`sparse-relation-field-internal-positive-continue`：relation path 被使用，固定 temporal
correspondence 与结构性 left/right role binding 均有正 causal evidence。Learned
`alpha=-0.1493664682`、`tanh(alpha)=-0.1482654959`。Full frame-aggregate direct-hand
union 5-cm F1 为 `0.778771`，MPJPE `11.985653 cm`、object goal `94.125465 cm`、
pelvis goal `5.238491 cm`、FS `0.799586`；这些 internal descriptive values 不替代 sealed
D2-X native control，也不用于 checkpoint selection。

Worker 发起 non-destructive recovery 后，成功树在两端均为 17 files /
37,798,242 bytes，tree SHA-256
`044f98f78d52347af0c3120a1a5ca4df25c5e4773256c89c2fd5e6bd77fd0b21`，
checksum dry-run 为零差异。Metrics/manifest/paired-noise/paired-conditioning/
sparse-relation-appendix SHA-256 分别为
`0d1e422386bd181e86ef5d77be80d05972bea92411cbb716644ff0a5f2811ba9` /
`811cb1be8e3383295b7d60b8d8f488a2ddafd5d9d829a652e2a5e894325c80b2` /
`1f4123945ae576b8a12ed83fa115dc32d9bea6df81b67c759f1cf482f088988c` /
`1eaa2380f26368ae8dd754c3be5452949f44492eba9f480ee71078a188434b9d` /
`a38693f5743be3b06b05097c1dc0129f6968eff0ea6e22ce70985c6df6a60815`。

下一步只允许将本 append-only record 提交并 fast-forward 到 worker，然后运行一次固定
`p1-hoi-d2ae-native-eval-s42-20260729`。Internal 正结果不授权 checkpoint selection、
D2-AE1、consistency、longer budget、任何 sweep、HSIPrior 或 Mixer。

#### 2026-07-29 Phase 1B D2-AE0 fixed native evaluation completion

Fixed native run `p1-hoi-d2ae-native-eval-s42-20260729` 在 clean worker
`phase/01b-hoi@5a167347ec4761ec8427b518a36da9157b8fe033` 上只加载 fixed final-online
checkpoint SHA-256
`b7d49046504e9f8367bfd2bce0aeefb1c8590bf9c542b6eed637f05bdfcdd840`。
协议为 official 438 sequences × 3 windows、500-step unguided production diffusion、
online/final-online weights、seed 42 与 10,000 次 paired sequence bootstrap；CFG、guidance、
scene conditioning、dynamic perception 与 consistency 均关闭。Sealed D2-X aggregate /
per-sequence SHA-256
`3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
`69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`
直接复用且未重新生成，sealed D2-X checkpoint SHA-256 为
`b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51`；
released aggregate SHA-256 为
`76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6`。
D2-AC/D2-AD 只作 sealed descriptive evidence。Run local start/end 为
`2026-07-29T01:54:02+08:00` / `2026-07-29T02:07:31+08:00`，exit status 为 completed，
runtime/end-to-end 为 `383.200603 / 375.213926 s`，55,188 frames 的 synchronized
generation 为 `71.085844 s`。未进行 optimizer update、checkpoint write、checkpoint
selection 或训练；FID、Matching、R-Precision 与 Diversity 均未生成。

Internal path/temporal/role mechanism gates 保持全通过，但 native transfer 未通过：

- D2-AE contact F1 `0.64194385`，sealed D2-X 为 `0.63742594`；paired difference
  `+0.00451791`，95% CI `[-0.01809406, 0.02684589]`；
- contact recall difference `+0.00168393`，95% CI
  `[-0.02313870, 0.02638117]`；
- contact precision difference `+0.01557008`，95% CI
  `[-0.00519356, 0.03679141]`；
- released contact-F1 gap closure `0.05029306 < 0.25`；
- target contact F1 `0.64194385 < 0.6598838781`。

Protection gate 也未通过：end-object target/control mean-ratio CI 为
`[1.08425, 1.21382]`，FS ratio CI 为 `[1.03233, 1.17347]`，其 upper bounds 均超过
`1.10`。181-sequence penetration finite-mask contract 通过，finite-mask sequence-ID
SHA-256 为
`2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec`。
Classification precedence 因而首先停在
`sparse-relation-field-transfer-negative-stop`；后续 protection 与 released-95%
failures 保留为证据但不覆盖该 classification。Checkpoint selected/selectable 均为 false。

Metrics/manifest/aggregate/per-sequence/resolved-config/resolved-target/preflight/
run-local-registry SHA-256 分别为
`55927debc01eba5a2a07484695b62aed9cb1f7c29e30d289e84e4371229d60f8` /
`419be60fc35c747c27d585270ff0f504921c79f8993004a48fbebd66b2f4d8db` /
`157acda463036bdf787618c217262c14c77a09a3f409cbeada03de06e9b902a1` /
`8533b66ea3c1fb0928b8a7581bb79c0cc14d594970314a3b7619659daddfb95c` /
`d747b549190c1e3fd8e5f91b12ae8c51db405e0a7e4495d556e94ad63fa7a378` /
`4ee5916806c3aafb600054641a0b7baaa17db8d9479c949a1d7fd4e7f7530ad8` /
`5572d7f53913e50b763da5772dfb4bb2d336bcd42f925916a960e2c64833487e` /
`8fb263138e5bc2f429630dc8e5c57b93fe4019ab818cf062a1dddaf82cc1e972`。
Worker 发起 non-destructive recovery 后，两端完整 native tree 均为 18 files /
3,474,559 bytes，SHA-256
`4f31bb8f61bd40eb4604a25a0802a970686092306faf86efa0b289c856cd34b5`，
checksum dry-run 为零差异。

Detached wrapper 的 `exit_code` 原始 bytes 为 literal `0n`，SHA-256
`3ad4ee182e21c25db763cda6359ecc441b8ea32ea4d6631c012aac7fa7d362dc`；
文件未覆盖，postflight 按 leading return code 解析为 `0`。首次 postflight verifier
错误要求 aggregate-only aliases 出现在 per-sequence rows，失败 artifact SHA-256
`05133fc6afc981ec8b28d7b3ede5c938da9110fc0146a704858045289ed50e15`
原样保留；只修 serialized schema mapping 的 append-only r1 verifier 通过，SHA-256
`c4f2f86ccc341e835fdfe6f87f11fb9ec3d7dfa5db8c1bb4d4abba073ba28d18`。
未改变或重跑任何 metric、mask、reduction、threshold、evaluator 或 native workload。

#### 2026-07-29 Phase 1B D2-AE0 completion record

Completion identity 为 `p1-hoi-d2ae-completion-s42-20260729`。D2-AE0 的
identifier/source/provenance audit、plan-only registration、implementation、
authority CPU contract、functional smoke、4-GPU full-micro-batch performance benchmark、
一次从 seed-42 random initialization 完整运行的 61,440,000-window formal training、
fixed internal causal diagnostic、fixed native evaluation、non-destructive artifact recovery
与 hash verification 均已完成。所有 operational failures、完整 formal tree、20 cadence
checkpoints、80 per-rank RNG sidecars、paired internal artifacts、native raw outputs 与
postflight schema failure/r1 correction均保留且未覆盖。

Performance gate 通过，formal throughput 为 `3,347.042 windows/s`；internal path、
temporal correspondence 与 left/right role-binding gates 全通过。但 native contact F1/
recall improvement 无显著 paired evidence、contact-F1 point/gap-closure gates失败，因此
最终 classification 锁定为
`sparse-relation-field-transfer-negative-stop`。Fixed final-online checkpoint
`b7d49046504e9f8367bfd2bce0aeefb1c8590bf9c542b6eed637f05bdfcdd840`
不可选择、不可 resume、不可初始化后续 prior。

Lifecycle tree SHA-256 完整绑定如下：

- authority CPU：
  `662cf1fa37121d24b660334fa22c5fec1d5114e980271d6e1df58aa67973fae5`；
- functional preflight failure / successful r1：
  `d2bd049d7688c8f5493c0698066f79dcfceeb90f8ff34530da4b4035db4170b5` /
  `c2eed8eef78c720db46fd4064d78bad07fb85f1462e25d113a99a69cea474259`；
- performance base / formal-bound r1：
  `0d62d1c5e1da2272c309cbd1882ebbd785897690f2f5ed02750ee87542ba59bb` /
  `b7042d965a8483afd8b1306e7a81d2a30d067f54f1094dfc8910d88fcb4882c7`；
- formal preworkload failure / complete r1 / earlier preserved snapshot：
  `620c4cd5d6361036d15e0adac58a40adb503e7196a2946c7c25ebc4cd43c0136` /
  `3c8a987d54dfb63e89d7ec243fb065dc4f84c95808d92eee13b46ab621959428` /
  `420e2f89d8059e4d9b5d0249001fbb9dbaffd5e591990f8ba7d6fbcdf6e44ae6`；
- internal base / r1 preflight failures / successful r2：
  `015e180d5aa21f093fe7f712d576150f12d47203aac26269f28f56c0015336e3` /
  `88f20c8ba3f0c013ba475e04551706ce2194c1904d33db2738dde497175de8bd` /
  `044f98f78d52347af0c3120a1a5ca4df25c5e4773256c89c2fd5e6bd77fd0b21`；
- fixed native：
  `4f31bb8f61bd40eb4604a25a0802a970686092306faf86efa0b289c856cd34b5`。

Final decision booleans 为：contract/performance/internal mechanism/path/temporal/role
均 `true`；native transfer/protection/released-95% effectiveness 均 `false`；
checkpoint selected/selectable 均 `false`。

Compact result：
`experiments/results/p1_hoi_phase1b_d2ae_sparse_relation_field_s42_20260729.json`
（SHA-256
`13311ea2cea311904225d22bb20fd88f652f32c5612a84d66c7d2b93b96a4036`）；
phase summary：
`docs/phase_summaries/PHASE_1B_D2AE.md`
（SHA-256
`54bd808a8f01c3d1d538c4c5f9f0e0932e078ef431c14b30a9acc19ee8e0c206`）。

Phase 1B D2-AE0 在此停止。未启动 D2-AE1、longer-budget extension、consistency、
任何 point/width/depth/role/placement/LR/batch/threshold sweep、D2-AC/D2-AD retrain/
resume/selection、新 loss、timestep weighting、rollout exposure、CFG、HSIPrior 或 Mixer；
也不 merge/tag。任何后续 HOIPrior direction 必须重新获得授权并先做 dated plan 与
append-only registry hypothesis。

#### 2026-07-29 Phase 1B D2-AF0 sqrt-alpha-bar current-state reliability routing 预注册（plan-only）

用户在完整审阅 D2-AE0 后，只授权最后一次 HOIPrior 方向预算；该方向结束后，无论结果
正负，Phase 1B 均停止继续搜索，下一次独立 session 从 Phase 1C HSIPrior 的 dated
preregistration 开始。本 amendment 执行前，authority checkout 为
`/data/yujinlun/InfBaGel-release`、branch `phase/01b-hoi`、HEAD
`8c4f731846645a4b0a422c6a1bd0405552b831a9`
（`Close Phase 1B D2-AE0`），worktree clean；核验时间为
`2026-07-29T15:31:12+08:00`。重新扫描 working tree、全部 Git objects/refs/reflogs、
append-only registry、authority/worker staging names 与 worker checkout 后，D2-AF、
`d2af`、`D2-AF0`、`p1-hoi-d2af-*` 和 `sqrt-alpha-bar-reliability` 均未被用作
identifier；历史 JSON SHA 中偶然出现的 `d2af` 字节子串不构成 identifier。Integration
baseline `b9a158f75ab0740c91c9cfc8863a65fa381b014c` 是当前 HEAD ancestor，禁止分支
`feature/independent-hoi-hsi-priors` 不是 ancestor。Source audit 证明下述机制可由现有
train/sample 共用 timestep API 实现，没有结构性阻塞。本 commit 只允许修改本 plan 和
registry；不得包含 source change、checkpoint load、GPU workload、训练或评测。

1. **Sealed evidence 与唯一假设。** D2-AE0 的 fixed internal diagnostic 已证明 sparse
   relation path 被使用、固定 temporal correspondence 有因果作用、left/right role binding
   是结构性的；但 native contact F1/recall 相对 D2-X 只有
   `+0.004518/+0.001684`，paired 95% CI 均跨零，contact F1 仅 `0.641944`，released
   gap closure 仅 `5.03%`，并出现 end-object 与 FS protection failure。D2-AF0 的唯一
   假设是：D2-AE 在高 diffusion noise 的 early reverse steps 对当前 `x_t` 几何给予了与
   late clean steps 相同的 residual scale，因而把不可靠 relation 当作强条件写入 trunk；
   使用同一 diffusion schedule 的 clean-signal reliability
   `sqrt(alpha_bar_d)` 衰减该 residual，可能保留 D2-AE 已证实的 relation/temporal/role
   结构，同时修复 official rollout 的 contact transfer、end-object 和 FS。不得把本
   hypothesis 扩展为 learned timestep gate、SNR/loss weighting、exposure、guidance、
   consistency 或任何新 loss。

2. **唯一 manipulated factor。** D2-AF0 相对 D2-AE0 只将 writeback 从

   \[
   H'_t=H_t+\tanh(\alpha)r_{a(t)}
   \]

   改为

   \[
   \rho(d)=\sqrt{\bar\alpha_d},\qquad
   H'_t=H_t+\rho(d)\tanh(\alpha)r_{a(t)}.
   \]

   `d` 是 model 当前收到的逐样本 diffusion timestep `[B]`；training 必须使用生成当前
   `x_t` 的同一个 `d`，sampling 必须使用 reverse loop 当前的同一个 `d`，mixed-timestep
   batch 必须逐样本 index。`d=499` 是最 noisy/最早 reverse step，`d=0` 是最 clean/最后
   reverse step。不得使用 clean target、predicted `x0`、previous `x0`、future GT、
   contact、Scene、stored relation 或 sampler-only source 计算 `rho`。`rho=1` 只允许作为
   注册的 test/internal counterfactual，不允许用于训练、checkpoint selection 或第二次
   formal run。

3. **Canonical schedule contract。** `rho` 必须直接来自与 `GaussianDiffusion` 共用的
   PyTorch-1.13.1 float32、500-step linear-beta schedule：
   `betas=torch.linspace(0.0001,0.02,500,float32)`，
   `alpha_bar=cumprod(1-betas)`。必须只有一个 project canonical schedule constructor；
   released author utility 不得改动。Canonical raw float32 tensor hashes 为：

   - beta SHA-256：
     `496ec54f35af6fe7b92417f7da8b442f31c9c0070bfdd62dbb16fefc426c8f3e`；
   - alpha-bar SHA-256：
     `55f162cebbe109c67a75b00a10a1d23ea85fb1d18df9a372a3e237df5a8f48d4`；
   - sqrt-alpha-bar SHA-256：
     `5d25c63d6618c77cc31976ee9e2c5645aa41653030fca210594a05254323b440`。

   `rho[0/100/249/400/499]` 必须分别为
   `0.9999499917030334 / 0.8995221257209778 / 0.5297974348068237 /
   0.19632703065872192 / 0.0797039046883583`，并严格单调递减。Model-side schedule
   buffer 必须是 canonical、non-persistent、无参数、无 optimizer state；checkpoint
   metadata 必须记录完整 schedule contract/hash，loader 必须在 `load_state_dict` 前验证
   独立 D2-AF architecture/provenance。D2-AE 与 D2-AF learned `state_dict` schema 可以
   相同，但 D2-AE/released/base/D2-AC/D2-AD checkpoint 即使 tensor shapes 相容也必须被
   D2-AF loader 拒绝；D2-AE loader 也必须反向拒绝 D2-AF。

4. **全部保持项。** D2-AE 的 current-state relation builder、100 immutable rest-object
   points、surface transform、roles `(joint 24,joint 26,joint 0)`、temporal anchors
   `(0,5,10,15)`、`4→128→128` point encoder、mean/max pooling、role concat order、
   `768→512` projection、four temporal embeddings、LayerNorm、single scalar
   `tanh(alpha)`、alpha exact-zero initialization、fixed segment routing、full-trunk
   placement、20 tokens、4 condition tokens、global BPS、D2-X FK-foot routing、
   `[B,16,232]` clean output、2-frame history restoration、500-step clean-x0 diffusion、
   losses/reductions/weights、optimizer、LR、batch、split、budget、sampler 和 official
   evaluator 全部不变。Global `rho(d)` 会同时衰减 anchor 0 的 clean-history relation；
   这是本单变量设计预先承认的代价，不得事后改为 per-anchor gate。参数必须仍为
   base `29,673,448`、relation `413,953`、total `30,087,401`，增量
   `1.3950283%`；不得新增 learnable parameter。Seed-42 fresh initialization 的完整
   non-persistent-buffer model-state SHA-256 必须继续精确为
   `b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c`，
   否则 GPU 前停止。

5. **Authority CPU hard gate。** D2-AF 必须继承 D2-AE 全部 geometry、asset、SO(3)、
   invariance/sensitivity、point permutation、finite、dtype/device/batch、parameter/API、
   train/sample builder parity、checkpoint provenance、HSIPrior/Mixer independence、
   forbidden-source static scan、full suite 与 registry validation contracts，并新增：

   - canonical helper、`GaussianDiffusion.sqrt_alpha_bar` 与 field buffer byte-exact；
   - timestep 必须是与 batch 同 device 的 `torch.long[B]`，负值、`>=500`、错误 shape/
     dtype/device 全部 fail closed；
   - mixed batch `[0,249,499]` 必须逐样本得到注册的三个 `rho`；
   - test-only `tanh(alpha)=0.1` 时，field-level
     `delta_AF(d)=rho(d)*delta_unit_rho`，float32 max abs `<=1e-6`；
   - `alpha=0`、shared D2-X trunk、`eval()` output max abs `<=1e-6`，并要求实际 exact
     zero where representable；
   - initial alpha gradient与 activated point-encoder/projection/temporal-embedding/
     relevant-trunk gradients在 `d=0/249/499` 均 finite/nonzero；
   - training q-sample 的 timestep 与 model timestep exact same；sampler capture 必须
     exact 为 `499,498,...,0`；
   - D2-AE/D2-AF resolved configs 除 identity、mechanism flag/variant、eligibility/
     performance binding 外 exact equivalent；
   - four-rank schedule hash一致，raw relation norm、`rho` 和 attenuated writeback norm
     分开记录；
   - 无 loss/SNR/timestep weighting、gamma/exponent/threshold/clamp、learned schedule、
     per-anchor reliability 或第二 writeback。

   任一失败分类为 `diffusion-reliability-contract-failure-stop`，不得开始任何 GPU
   workload。

6. **One-GPU functional smoke。** 注册 stem 为
   `p1-hoi-d2af-gpu-functional-smoke[-rN]-s42-<actual-date>`，worker 固定
   infbagel-4gpu/node01、1×RTX 3090、real-data batch 8、timesteps
   `0/249/499`、seed 42 random initialization、无 optimizer、zero updates、
   zero checkpoint writes。除 D2-AE smoke 内容外，必须记录三个 `rho`、mixed-batch
   per-sample scaling、raw/attenuated relation values、initial alpha gradient、
   activated gradients、peak allocated/reserved/headroom 和 model/schedule hashes。
   Operational preflight failure保留原目录并使用新 run id；scientific contract failure
   立即停止。

7. **No-training clean-signal premise gate。** Functional smoke 通过后、performance/
   formal 前，运行唯一注册的
   `p1-hoi-d2af-clean-signal-eligibility[-rN]-s42-<actual-date>`。该 diagnostic 不加载
   任何 checkpoint，不创建 model/optimizer，不做 update、rollout、official test 或
   downstream contact/goal/FS 指标；它只检验本方向的输入前提，不能选择旧 checkpoint
   或从多个可训练方向中择优。

   - 数据为完整 internal-validation split：216 sequences、29,382 windows，canonical
     non-shuffle global-index SHA-256
     `eab0bde2dc2ddad7ce2cc1817973ca46b9adaf24b1c906307f865930aeb11eb9`，
     sorted sequence-name SHA-256
     `472768c85c6d6c5b682a31a4d40a879d7a1e3d0b16085923c153db1045223fd8`；
     `num_workers=0`、batch 128。
   - timesteps 固定为 `0/249/499`。每个 timestep 使用 CPU float32
     `torch.Generator`，seed 为 `42 + 1,000,003*d`，按上述 canonical order/batch
     产生 Gaussian noise并记录完整 stream hash；`q_sample` 必须保持原 2-frame history。
   - 对 clean `x0` 和相应 `x_d` 运行完全相同的 pre-encoder pure-PyTorch relation
     builder。每个 window 的 mutable-anchor corruption 定义为 anchors `5/10/15` 上
     `[role,point,delta_xyz+distance]` 的
     `C_d=sqrt(mean((feature_d-feature_clean)^2))`。先在 sequence 内平均，再以 sequence
     为 paired unit、seed 42、10,000 bootstrap。
   - `C249-C0` 与 `C499-C249` 的 paired 95% CI lower 必须都 `>0`；anchor 0 因属于
     immutable history，其 scaling 前 feature 在三 timestep 间 max abs 必须 `<=1e-6`。
     Eligibility manifest 必须绑定已通过的 CPU/smoke summary absolute path、SHA-256
     与相同 source-tree hash；任何相关 code change 都使该 eligibility失效并要求新 id
     从头执行，但不得因此改变科学条件。

   Contract/implementation failure 分类为 `clean-signal-contract-failure-stop`；corruption
   单调 premise gate失败分类为 `clean-signal-premise-negative-stop`。任一失败均停止
   D2-AF0，不运行 performance/formal，也不得换 timestep、subset、noise、metric、anchor
   或改成 per-anchor scaling。

8. **4-GPU full-micro-batch performance hard gate。** 只有上述三层 gate全部通过，才在
   clean、identical committed worker object 上运行
   `p1-hoi-d2af-performance-benchmark[-rN]-s42-<actual-date>`：4×RTX 3090、
   per-GPU batch 512、effective batch 2048、FP32 Adam、64 warm-up + 256 measured
   updates、524,288 measured windows、CUDA synchronized timing、random initialization、
   checkpoint load/write均为零、benchmark weights禁止复用。必须记录 loader wait、H2D、
   GPU relation build、rho lookup/writeback、forward、backward、optimizer、DDP、CPU/GPU
   utilization、contention、intermediate shapes、四 rank schedule hash和 peak/headroom。
   Idle hard gate使用多次 compute-process/memory/utilization采样；P-state单独记录但不因
   单次非 P8 独立否决。

   Sealed D2-AE formal throughput 为 `3,347.0419610997483 windows/s`；D2-AF0 的
   预注册门槛固定为其 95%：

   \[
   throughput\ge 3,179.689863044761\ {\rm windows/s},
   \]

   对应 61.44M-window ETA `<=5.367399778519349 h`。同时 memory headroom 必须
   `>=max(2 GiB,10% device memory)`、losses/gradients finite、无 CPU dynamic
   geometry、无 contention。Benchmark必须一对一绑定 intended formal run id/source
   hashes；formal pre-workload retry 需要新的 benchmark retry identity。Completed
   scientific benchmark未过即分类
   `diffusion-reliability-performance-negative-stop`，不得通过 batch/worker/thread/
   architecture/point/width/role/routing或任何 sweep重试。

9. **唯一 formal training。** 只有 contract、smoke、clean-signal premise 和 performance
   全通过，才运行
   `p1-hoi-d2af-sqrt-alpha-bar-reliability[-rN]-s42-<actual-date>`。固定 seed 42、
   split `experiments/splits/omomo_hoi_train_validation_seed42.json`（SHA-256
   `019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e`）、
   4×RTX 3090、batch 512/GPU、effective 2048、accumulation 1、61,440,000 windows、
   983,040,000 frames、30,000 updates、FP32 Adam、LR `1e-4`、betas
   `(0.9,0.999)`、weight decay 0、no warmup/scheduler/AMP/clipping/EMA。
   FK/object-surface/velocity/terminal-goal weights继续为
   `0.3569973401779424 / 0.4772322188400037 / 0.1 / 1.0`，D2-X FK-foot routing
   enabled，全部新 loss disabled。必须从 seed-42 random initialization 开始；
   init/weight-init/resume均为空，released/D2-X/D2-AC/D2-AD/D2-AE/任何 prior/
   EMA/consistency checkpoint load count全为零。完整运行 fixed budget，只使用
   online/final-online；不得选择 cadence/best-validation checkpoint。稳定区间和至少一个
   resumable checkpoint通过后，按 worker-owned persistent-session规则报告 throughput/
   ETA/hash并停止主动轮询。

10. **Fixed five-path internal causal diagnostic。** Formal完成后只加载 fixed
    final-online，复用 sealed D2-O 64 sequences × 3 windows、phase offsets
    `(14,56,98)`、selection SHA-256
    `1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`。
    五条 paired 500-step rollout固定为：

    - `full_rho`；
    - `unit_rho`：每一步只将 `rho(d)` 强制为 1；
    - `relation_gate_ablated`：每一步 `tanh(alpha)=0`；
    - `temporal_correspondence_permuted`：沿用 D2-AE `k<-(k+2) mod 4`；
    - `left_right_role_swapped`：projection前只交换 left/right pooled blocks。

    除被操纵因子外，五路共享 initial latent、每一步 posterior noise、condition、history、
    ordering 和 restoration；permuted/swapped paths继续使用 canonical `rho(d)`。统计、全部
    contact/distance/penetration/goal/FS/uncertainty和描述性 relation appendix沿用 D2-AE，
    并额外报告每个 timestep/anchor/role 的 raw relation、rho、attenuated writeback norm/
    variance/sensitivity。Primary gates全部 conjunctive：

    - `full_rho-unit_rho` direct-hand union 5-cm F1 CI lower `>0`；
    - `unit_rho-full_rho` GT-contact-frame mean distance CI lower `>0`；
    - D2-AE 原五个 path/temporal/role/distance gates原阈值全部通过。

    `unit_rho` 只证明同一 D2-AF trained model 是否依赖 schedule，不等同于 D2-AF 与
    separately trained D2-AE 的模型比较。结果必须分别保存
    `internal_status={unused|schedule-negative|temporal-negative|role-negative|passed}`；
    internal无论正负都执行下面唯一一次 fixed native，不得以 internal cohort过滤 official
    result。

11. **Fixed native evaluation 与双重比较。** 协议严格沿用 D2-AE：official 438
    sequences × 3 windows、500-step unguided production diffusion、final-online、
    CFG/guidance/scene/dynamic perception/consistency全部 off，paired sequence unit、
    seed 42、10,000 bootstrap、sealed D2-X 181-sequence penetration mask、official
    evaluator/hash/helper/threshold不变。D2-X checkpoint/aggregate/per-sequence继续为
    `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51` /
    `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
    `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`，
    不重新生成。D2-AE 只复用
    sealed aggregate
    `157acda463036bdf787618c217262c14c77a09a3f409cbeada03de06e9b902a1`
    和 per-sequence
    `8533b66ea3c1fb0928b8a7581bb79c0cc14d594970314a3b7619659daddfb95c`；
    不加载/重跑其 checkpoint，不把它作为 initializer、resume、checkpoint selector 或
    candidate。

    Native必须同时通过两层 gate：

    - **D2-AE single-factor repair：** AF−AE contact F1 和 recall paired CI lower
      均 `>0`；AF/AE end-object 与 FS paired mean-ratio CI upper均 `<1.0`；
    - **D2-X candidate：** AF−X contact F1/recall CI lower均 `>0`，contact F1
      `>=0.6598838781`，released–D2-X contact-F1 gap closure `>=25%`。

    D2-AC/D2-AD protection contract原样继承：AF/X end-object、Txy、FS、Pbody、
    hand penetration、MPJPE、Troot、Tobj、Oobj mean-ratio CI upper均 `<=1.10`，
    contact precision difference CI lower `>=-0.02`，penetration finite mask exact；
    released-baseline 95% effectiveness gate原样继承。所有条件均为 AND，不允许
    composite、best-of、metric替换或阈值修改。

12. **Decision、lifecycle 与 stop rule。** Post-training同时保存 `internal_status` 和
    `native_status`；native negative是 headline，internal positive不能救回。单线终态顺序为：

    - `diffusion-reliability-contract-failure-stop`；
    - `clean-signal-contract-failure-stop`；
    - `clean-signal-premise-negative-stop`；
    - `diffusion-reliability-performance-negative-stop`；
    - `diffusion-reliability-ae-repair-negative-stop`；
    - `diffusion-reliability-d2x-transfer-negative-stop`；
    - `diffusion-reliability-conflict-negative-stop`；
    - `diffusion-reliability-positive-but-not-effective-stop`；
    - native全部通过但任一internal gate失败：
      `diffusion-reliability-native-positive-mechanism-unverified-stop`；
    - 全部通过：
      `diffusion-reliability-positive-candidate-stop`。

    只有最后一类可将 fixed final-online 标为 selectable autonomous HOIPrior candidate。
    Lifecycle stems固定为
    `p1-hoi-d2af-{cpu-contract|gpu-functional-smoke|clean-signal-eligibility|
    performance-benchmark|sqrt-alpha-bar-reliability|
    sqrt-alpha-bar-reliability-internal|native-eval|completion}[-rN]-s42-<actual-date>`；
    config default `run_id=null`，每次实际 date现场生成，失败目录保留且retry使用新 id。
    所有 resolved config、same-context manifest/preflight、logs/profile、failure trees、
    checkpoints/RNG、internal five paths/paired noise、native raw/optional outputs、run-local
    registry、hardware/data/dependency/evaluator hashes必须由worker发起non-destructive
    recovery，双端统一 `sha256_path` 和 checksum dry-run；不得 `--delete`。

    本方向禁止任何 gamma/exponent/threshold、learned schedule、per-anchor gate、LR/batch/
    budget/point/width/depth/role/placement sweep，禁止第二次 formal run、longer budget、
    D2-AF1、D2-AE/D2-AC/D2-AD resume/retrain/selection、新 loss、SNR/timestep loss
    weighting、gradient projection、rollout exposure、CFG、consistency、scene、HSIPrior 或
    Mixer。若 pretraining gate失败，formal budget不消耗但本次最后 HOIPrior direction仍
    结束；若formal启动则只允许完整运行该一次预算。最终必须写 compact result、
    `docs/phase_summaries/PHASE_1B_D2AF.md` 和 append-only completion。无论最终分类，
    本 session 均在关闭 Phase 1B 后停止，不自动开始 Phase 1C；下一独立 session 的唯一
    entry point 是 HSIPrior plan-only preregistration。

#### 2026-07-29 Phase 1B D2-AF0 implementation / pre-GPU lifecycle binding amendment

本 amendment 在 plan-only commit
`cbf55ef2c5d667d28698597127767e0b14151f06` 上实现且只实现已经预注册的
D2-AF0。核验时间为 `2026-07-29T17:01:41+08:00`；authority path、branch 和 dirty
状态分别为 `/data/yujinlun/InfBaGel-release`、`phase/01b-hoi` 和仅包含本 logical
implementation 的预期修改。当前没有启动 CPU reportable lifecycle、worker publication、
CUDA workload、optimizer update、checkpoint load/write、训练或评测。

1. **Single-factor source implementation。** 新 architecture variant
   `d2af_sqrt_alpha_bar_reliability` 完整复用 D2-AE 的 current-state geometry、100-point
   sparse assets、roles、anchors、point encoder、pooling、projection、temporal embeddings、
   LayerNorm、single alpha、routing 和 D2-X trunk，只将 field writeback 固定为

   \[
   H'_t=H_t+\sqrt{\bar\alpha_d}\tanh(\alpha)r_{a(t)}.
   \]

   Model 对逐样本 `torch.long[B]` timestep fail closed；training 中同一个 tensor object
   同时交给 `q_sample` 和 model，production sampler 按 `499,...,0` 将当前 reverse
   timestep 交给同一 field。没有 clean/future `x0`、previous predicted `x0`、Scene、
   contact、stored relation、sampler-only relation source、gamma/exponent/threshold、
   per-anchor/learned gate、loss/SNR weighting或第二 writeback。

2. **Canonical schedule 与 model/checkpoint contract。** 新增唯一 project helper
   `code/priors/diffusion_schedule.py`；`GaussianDiffusion` 与 D2-AF field byte-exact
   复用注册的 500-step float32 linear schedule。Field schedule 是 non-persistent buffer，
   不增加 parameter、learned state 或 optimizer state。D2-AF parameter count仍严格为
   base/relation/total `29,673,448 / 413,953 / 30,087,401`，seed-42 initial model-state
   SHA-256仍为
   `b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c`。
   Checkpoint metadata使用独立 `diffusion_reliability_contract`；D2-AE 与 D2-AF loader
   双向拒绝对方 provenance，base/released/D2-X/D2-AC/D2-AD 继续被拒绝。Formal resume
   还绑定 random origin、same-run identity、eligibility/performance path与SHA、source
   contract、schedule/assets以及每 rank RNG sidecar；checkpoint及任一 rank sidecar已存在
   时一律拒绝覆盖。

3. **Pre-training lifecycle implementation。** 新增 authority CPU runner、single-GPU
   functional smoke、no-checkpoint clean-signal eligibility和4-GPU full-micro-batch
   benchmark。所有工具要求现场 actual-date run id、clean exact Git object、resolved config
   先落盘且无 interpolation、exclusive output，并支持 `--resolve-only`。Eligibility不创建
   model/optimizer、不加载 checkpoint，只遍历完整216-sequence internal validation、
   canonical 29,382 windows，以注册的CPU noise streams检验 anchor `5/10/15` relation
   corruption单调性和anchor 0 history exactness；其summary同时绑定已通过的authority CPU
   与worker smoke artifact、SHA及相同 formal source-tree hash。Benchmark固定
   4×512、64 warm-up + 256 measured updates、FP32 Adam、zero checkpoint I/O，并将passing
   eligibility SHA和唯一 intended formal id写入summary；formal trainer在启动前重新验证
   eligibility和performance每个字段。

4. **Performance/preflight hardening。** Worker idle preflight改为3次、间隔1秒的GPU/
   compute-process采样；memory、utilization和external CUDA process仍是硬门禁，P-state只作
   描述性记录，避免D2-AE internal曾发生的瞬时P5误停。Benchmark仍要求无contention、
   CUDA synchronized timing、loss/gradient finite、GPU-only relation build、4-rank schedule/
   source/model identity、memory headroom和
   `throughput >= 3179.689863044761 windows/s`、ETA
   `<=5.367399778519349 h`。失败后禁止任何architecture、batch、worker、thread或科学条件
   sweep。

5. **Fixed post-training tools。** Internal runner固定五路
   `full_rho / unit_rho / relation_gate_ablated /
   temporal_correspondence_permuted / left_right_role_swapped`，共享initial latent、
   每步posterior noise、conditions、history与window ordering；另外保存raw relation、
   canonical rho、raw/attenuated writeback的timestep/anchor/role appendix。Native runner
   即使internal mechanism为负也继续唯一一次official evaluation，只复用sealed D2-AE与
   D2-X aggregate/per-sequence artifacts，不加载/重跑D2-AE checkpoint；AE-repair、D2-X
   transfer、protection、released effectiveness和最终classification precedence均逐项
   fail closed。Official evaluator source、threshold、mask和reduction未修改。

6. **Implemented paths and hashes。** Logical implementation覆盖：

   - base/D2-AF configs：
     `fe7619fbaa8256d664d5f68247ef9ebd56738db05e942bafab659b8eac5186e2` /
     `f248bdd118b1d14275867670f32e5973271c93a9d5a2a991df6c36cb4dc73876`；
   - schedule/diffusion/models/sparse relation：
     `b4d9cf74174d63de30f75acb3f687e87f824e75b147f3a2efcfd3d76befd5b09` /
     `fd8d05c34689cf4697920097bd330e6a25e3424c7460eb3a4e7ef12f45ed17a2` /
     `f7d464e48629a5e6420ea6a21f1ff8130980223cb6f59944a6226a83a952dd12` /
     `d86b49dc4030c5621510e3c66345e592235b03350fca767216de56ab78350ba3`；
   - trainer：
     `0f07d4d0060b4394bbe75e2f86bca385f6c720fdd06abc42fa7099339bba3e2d`；
   - CPU/eligibility/smoke/benchmark：
     `5d741c3f863b577ce3f8eba32b77d11fe4ffdd556a98fdf24ca84fba52c2b3c3` /
     `7c75db0ff38786b240cda39d6c95335ff35d6cac6f542a3ffd7a82b8ac26378d` /
     `71754e302215e8d5dcf37e76ef044ea504a8308d2dfaf3015b63d2d885e6b681` /
     `309fd5c0d4556ce902757b56ee92abed184f83e41fed8670c38e79d3bd69ca4f`；
   - internal/native：
     `29f542ba999c3d00b7a0b0d08814114f1803ca4a7e297bb75bd6c3f2b363109c` /
     `581efaac439e51721a4ada83ff6c852ab2f61be7d7039159539eecd81275ea6a`。

7. **Authority verification before this record。** 使用锁定authority Python完成：

   - D2-AF core/eval/CPU-lifecycle/GPU-lifecycle定向测试 `36/36`；
   - 完整 `unittest discover`：`414/414` pass，0 failure；
   - 全部新增/修改Python文件 `py_compile`；
   - `tools/experiment.py validate`：implementation record前213条registry记录有效；
   - `git diff --check`。

   本 implementation record加入后必须重新运行完整suite、registry validation、
   `py_compile`与diff check，然后提交一个clean logical implementation object。只有该
   object上的reportable authority CPU hard gate通过，才允许worker从authority发起Git
   fast-forward；之后依次为functional smoke、authority eligibility、4-GPU benchmark。
   Formal training仍严格条件化于所有pretraining gates，且无论D2-AF0最终结果如何都不再
   启动下一次HOIPrior实验。

#### 2026-07-29 Phase 1B D2-AF0 authority CPU contract completion record

Reportable authority run `p1-hoi-d2af-cpu-contract-s42-20260729` 在 clean
implementation commit
`cae7d4ed64fbc6c15b046c0d17b0cbdefd365b41` 上完成，classification为
`cpu-contract-passed`，runtime `9.5093920920 s`。Manifest在workload前由
`tools/experiment.py start`创建，resolved config先独占落盘且SHA-256为
`4377122c4f8abbba1c175f15f97f61e7e4034cac0cae3e6908c9aba01da21c45`；
manifest完成后SHA-256为
`9c7a05305a71f7907f1f056c3163037ebaa0daf0781753474e0da55d4fe50476`。

CPU hard gate结果：

- canonical schedule、`GaussianDiffusion`和四个模拟rank field buffer byte-exact，
  sqrt-alpha-bar SHA-256为
  `5d25c63d6618c77cc31976ee9e2c5645aa41653030fca210594a05254323b440`，
  buffer均不进入`state_dict`；
- mixed timesteps `(0,249,499)` 的
  `delta_AF-rho*delta_unit` max abs为 `2.384185791015625e-07`，低于注册的
  `1e-6`；错误shape/dtype/device/range全部被拒绝；
- base/relation/total参数严格为
  `29,673,448 / 413,953 / 30,087,401`，seed-42 initial state SHA-256严格为
  `b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c`；
- alpha=0与shared D2-X trunk output max abs为exact `0.0`；timestep
  `0/249/499` 的initial alpha及test-only activated point/projection/norm/temporal/trunk
  gradients全部finite/nonzero，probe未保存且optimizer update为0；
- training中`q_sample`和model收到同一个timestep tensor object；sampler trace严格为
  `499,...,0`，SHA-256为
  `a3b41318496c448ebc2cfe9a9c2b727b777e00188ac8672160b6b51de2817661`；
- D2-AF拒绝released/base/D2-X/D2-AC/D2-AD/D2-AE及缺失contract的伪D2-AF，
  D2-AE反向拒绝D2-AF；scientific checkpoint load为0；
- 全部继承的D2-AE sparse asset、surface-loss parity、yaw invariance、relative-pose
  sensitivity、left/right exchange、temporal permutation、point-set invariance、
  SO(3)/finite、train/sample builder、HSIPrior storage与Mixer clean-output contracts通过；
- source static scan确认无Scene、future/clean target、previous `x0`、contact、
  stored relation、NumPy/SciPy/trimesh/KD-tree、loss/SNR weighting、learned/per-anchor
  schedule或第二writeback；
- D2-AE与D2-AF resolved formal configs除注册identity/mechanism/eligibility/performance
  bindings外无差异；formal source-tree contract为91 files，SHA-256
  `68269a2cac8eaf6fd2b55b139bb2be5b5dbafde6e7f22496f5a894f18b843145`。

Metrics和完整stdout log均为66,188 bytes、SHA-256
`8726ad247b4b9b3828bbdef444426fa197cdb1b2f4333bfcd663fe6e4308eb7f`。
Authority staging tree包含3 files / 135,456 bytes，统一`sha256_path`为
`df730afb3685171099a7296fee87538e41cc64ae3ea61d50056eb87632221cd2`。
本 lifecycle没有CUDA、optimizer、update、checkpoint load/write、official test、
selection、formal training、consistency、HSIPrior或Mixer。下一步只允许提交本append-only
record，然后由worker发起Git fast-forward并执行same-context preflight与注册的single-GPU
functional smoke。

#### 2026-07-29 Phase 1B D2-AF0 single-GPU functional smoke completion record

Reportable worker run `p1-hoi-d2af-gpu-functional-smoke-s42-20260729` 在
infbagel-4gpu/node01 的 clean commit
`758d54897640e93cc60ac76050b9e769ddf4afbc` 上完成，status/classification为
`stable / functional-smoke-passed`。Manifest在任何CUDA workload前由
`tools/experiment.py start`从与workload相同的worker execution context创建；live
preflight连续三次确认4×RTX 3090无compute process且通过全部host/Python/data/evaluator/
clock/tunnel checks。归档的worker完整suite为`414/414` pass、按HOI-only contract仅skip
2个真实LINGO asset测试；registry validation为216条记录前的215条全部有效。

固定real-data batch 8、mixed timesteps
`(0,249,499,0,249,499,0,499)` 的结果：

- relation surface/features/point encodings/pooled blocks/relation vectors及raw/attenuated
  writeback全部finite、`torch.float32`且位于`cuda:0`，没有collator/CPU dynamic geometry；
- canonical rho严格为
  `0.9999499917030334 / 0.5297974348068237 / 0.0797039046883583`，
  `attenuated-rho*unit` max abs为
  `1.2153759598731995e-07`，低于注册的`1e-6`；
- seed-42 initial model-state SHA-256严格为
  `b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c`，
  field/diffusion schedule SHA-256均严格为
  `5d25c63d6618c77cc31976ee9e2c5645aa41653030fca210594a05254323b440`；
- loss、initial alpha gradient，以及`t=0/249/499` test-only activated
  point-encoder/projection/temporal-embedding/relevant-trunk gradients全部
  finite/nonzero；probe未保存；
- peak allocated/reserved为`270,197,248 / 325,058,560` bytes，device headroom为
  `24,970,985,472` bytes；
- optimizer未创建，update、checkpoint load/write、selection均为0，formal training未启动。

Resolved config、metrics、completed manifest与worker preflight SHA-256分别为
`d3c6865611a258e76a0c22306ab56686c8ac1543f1014ed702ec08b0b5354dec`、
`43862309e7758af25b99c7ad7f45d5882d2010912d48d54b8dabe0877fe9c8af`、
`47cff55236bf2f8698b6db74e17001419c0e058e11188e0a681e441f938f7e1a`和
`86638b0992c16a69ae1da70c4bc37912dbaa171bf7db1ad2f88f7d583dd8e369`。
Worker发起无`--delete` recovery后，checksum dry-run传输0 files；worker与authority的
10-file / 149,421-byte tree统一`sha256_path`为
`61fc8d844c68637e7cf34af4bb9e9b4dc969b71bb001af003fe22309247c0747`，
authority路径为
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2af-gpu-functional-smoke-s42-20260729`。

下一步只允许提交本append-only lifecycle record，然后在authority运行唯一注册的
216-sequence、29,382-window no-model/no-optimizer clean-signal eligibility。该premise gate
通过并恢复/注册前，不得运行4-GPU performance benchmark或formal training。

#### 2026-07-29 Phase 1B D2-AF0 clean-signal eligibility completion record

Reportable authority run
`p1-hoi-d2af-clean-signal-eligibility-s42-20260729` 在clean commit
`d12036e5e79d0e7142e8d163fc9a80a62fea317c` 上完成，status/classification为
`passed / clean-signal-premise-passed`，runtime为`105.1356934551 s`。它绑定已通过的
authority CPU metrics
`8726ad247b4b9b3828bbdef444426fa197cdb1b2f4333bfcd663fe6e4308eb7f`
和recovered worker smoke metrics
`43862309e7758af25b99c7ad7f45d5882d2010912d48d54b8dabe0877fe9c8af`，
三者formal source-tree contract均严格为91 files /
`68269a2cac8eaf6fd2b55b139bb2be5b5dbafde6e7f22496f5a894f18b843145`。
Smoke JSON未被改写；其worker绝对resolved-config引用通过authority只读symlink解析到已恢复、
SHA-256相同的staging artifact，最终prerequisite record保存的是该canonical recovered path。

Diagnostic完整遍历locked internal-validation的216 sequences / 29,382 windows，selection
hash严格为：

- global indices：
  `eab0bde2dc2ddad7ce2cc1817973ca46b9adaf24b1c906307f865930aeb11eb9`；
- sorted sequence names：
  `472768c85c6d6c5b682a31a4d40a879d7a1e3d0b16085923c153db1045223fd8`。

固定CPU noise streams与pre-encoder relation corruption结果：

- `C0 / C249 / C499` sequence mean分别为
  `0.037421006676433013 / 3.7462720501557163 / 4.458573468406191`；
- `C249-C0` paired mean为`3.7088510434792834`，10,000-replicate paired
  bootstrap 95% CI为`[3.6977504341815073,3.720119281574443]`；
- `C499-C249` paired mean为`0.7123014182504745`，95% CI为
  `[0.698195667507825,0.7267265609093563]`；
- 两个CI lower均严格`>0`；frame-0 immutable history在全部timestep与cross-timestep的
  max abs均exact `0.0`，低于`1e-6`。

因此全部三个premise gates通过，performance benchmark与formal training的premise flag为
true。该run没有创建model/optimizer，没有update、checkpoint load/write、rollout、official
test、downstream metric或checkpoint selection。

Metrics、resolved config与completed manifest SHA-256分别为
`c52c0536423d7a17101829cb2b020316b9c6e0f7aa2cf39f33b984ffb39896b4`、
`bf0646a3ec69453a17f54de78a5c7b477a6c0334bab8924e5afdad1cd39a1173`和
`83d1b5b1a6db9b4d1cea8052abadebc6d901fd08229057de4ea1d408ea78b763`。
Authority staging包含4 files / 190,598 bytes，统一`sha256_path`为
`1a9b9a2c6779d9971046ded5bc5ac23639aa73926690a8076b3c018b638bef52`。

下一步只允许提交本append-only lifecycle record，将完全相同的clean Git object和immutable
eligibility summary发布到worker，然后运行唯一注册的4×512、64 warm-up + 256 measured
performance benchmark。若其任一hard gate失败，D2-AF0与Phase 1B立即以
`diffusion-reliability-performance-negative-stop`关闭，不得启动formal training或调整条件。

#### 2026-07-29 Phase 1B D2-AF0 performance benchmark failure record

Reportable worker run
`p1-hoi-d2af-performance-benchmark-s42-20260729` 在clean commit
`1c6c3058478411361bf3e73830f900f660ae516b` 上完成。Process return code为0，固定
64 warm-up + 256 measured updates、4×512 effective batch 2048、524,288 measured
windows全部执行；scientific status/classification为
`failed / diffusion-reliability-performance-negative-stop`。

Hard-gate结果：

- measured synchronized wall：`250.8741843551 s`；
- throughput：`2089.8443630127094 windows/s`，低于注册门槛
  `3179.689863044761`；
- sealed D2-AE throughput fraction：`0.6243854685126333`，低于`0.95`；
- full 61.44M-window ETA：`8.166477355310539 h`，高于上限
  `5.367399778519349 h`；
- minimum memory headroom：`18,993,577,984` bytes，高于要求
  `2,529,604,403`；
- losses/gradients finite、relation GPU-only、四rank relation shapes、initial-state/
  schedule hashes、memory和无external contention contracts全部通过；
- checkpoint load/write为0，320-update sacrificial weights未保存且不可复用。

Mean measured profile totals across ranks为：

- loader wait/H2D：`56.2303234 / 0.3658839 s`；
- relation geometry/point encoder/projection/norm/derived pool-route-rho-writeback/
  complete module：
  `0.5566526 / 1.1547405 / 0.0265346 / 0.0036046 / 0.2908488 /
  2.0323810 s`；
- forward+loss/backward-inclusive-DDP/gradient validation/optimizer：
  `16.0291985 / 170.9401413 / 3.9957207 / 3.0369855 s`。

Relation module和完整forward与sealed D2-AE benchmark的
`1.9570280 / 15.9230519 s`接近；本次固定run的主要descriptive异常是rank-1 loader wait
`154.4085614 s`，其他rank为`53.4023972 / 9.5764329 / 7.5339022 s`，并在其他rank形成
inclusive backward/DDP等待。该证据不能事后授权retry：preregistration明确禁止第二次
benchmark、num-worker/thread/architecture sweep或改变任何科学/执行条件，因此不在
transient rank stall和可复现固定stack bottleneck之间做post-hoc选择。

Benchmark summary、completed manifest、resolved config和preflight SHA-256分别为
`53e9842d0522cf456a86eedc25d2a972cd00db3fb067113ff25f31f6117e1f33`、
`97d13c60dd0e073fdd649aec7b76bde4dad23fdf0fd7e8cef9b9ca04b6a04e54`、
`04f747890fd9e7ad3d40a580223783da849ac80a5eae1d826c9bc9af2f4b45a9`和
`e238a8242f31b5a08b083f1e11044834d922babe5c29681754d0a755396613d1`。
Worker发起无`--delete` recovery后，checksum dry-run传输0 files；worker/authority的
14-file / 1,914,984-byte tree统一`sha256_path`为
`076ed5e3ee80bd5325c661f9a3adbe225e45be963cc2166128cdc5c0faadf895`，
authority路径为
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2af-performance-benchmark-s42-20260729`。

Performance gate失败后已现场验证intended formal目录
`p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729`不存在。Formal training、
formal optimizer/checkpoint、internal和native均未启动。不得retry、调参、运行D2-AF1或
启动任何新的HOIPrior方向。

#### 2026-07-29 Phase 1B D2-AF0 one-time user-authorized performance waiver（plan-only）

在上述失败被完整保留并报告后，用户明确接受已测完整预算ETA
`8.166477355310539 h`，并授权：若没有确定、简单且不改变科学条件的训练时间优化，则直接
运行现有D2-AF0唯一formal budget。该新授权覆盖原先“performance失败即不训练”的执行
stop rule，但不回写历史、不把benchmark改成passed，也不改变其
`diffusion-reliability-performance-negative-stop`分类。

1. **ETA与根因解释锁定。** ETA只由固定预算和实测端到端吞吐外推：

   \[
   61{,}440{,}000 / 2{,}089.8443630127094 / 3600
   = 8.166477355310539\ {\rm h}.
   \]

   即每个2048-window update约`0.9800 s`。新增
   `sqrt(alpha_bar)`/rho并非主要计算开销：256个measured updates中，D2-AF relation
   module为`2.0323810 s`（约`7.94 ms/update`），sealed D2-AE为`1.9570280 s`
   （约`7.64 ms/update`）；完整forward分别为`16.0291985 / 15.9230519 s`，仅增加约
   `0.41 ms/update`。主要wall增长来自rank-skewed DataLoader/DDP critical-path wait：
   rank-1 loader wait为`154.4085614 s`，其他rank为
   `53.4023972 / 9.5764329 / 7.5339022 s`，其余rank相应在inclusive backward/DDP中等待。

2. **不做post-hoc execution sweep。** 当前没有已证实能够消除上述rank skew的单一安全
   toggle。`num_workers`、CPU affinity、prefetch/pinning、线程或I/O布局变更都需要新的
   full-micro-batch比较才能证明有效；第二次benchmark和这些sweep继续禁止。
   `profile_every_update=true`也保持不变：其同步不是已测rank-1 loader stall的根因，事后
   改动会改变注册execution contract而没有可报告的同条件证据。因此本waiver选择用户授权的
   “直接训练”分支，不改模型数学、训练循环计算、data loader配置或instrumentation。

3. **Waiver的精确范围。** 只允许启动一次原intended formal identity
   `p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729`；启动时仍须满足actual-date规则，
   且该目录必须此前不存在。原benchmark不重跑，320-update sacrificial weights仍不可复用。
   Formal仍从seed-42随机初始化，4×512/effective 2048、30,000 updates、61.44M windows、
   FP32 Adam、LR/loss/budget/split/checkpoint cadence和全部D2-AF0科学条件完全不变。
   不允许第二次formal、resume旧方向、checkpoint selection、D2-AF1、longer budget、
   consistency、HSIPrior或Mixer。

4. **Fail-closed implementation。** Formal trainer不得简单删除performance检查或伪造
   passing summary。它必须同时绑定：

   - 原failed benchmark JSON的absolute path、SHA-256、run id、failed status/
     classification、实测throughput/ETA及全部non-speed contracts；
   - 一份tracked、immutable、SHA-bound waiver JSON；
   - waiver中的唯一formal run id、用户授权事实、benchmark SHA、原/目标Git commit、
     exact transition diff SHA、允许改变的governance/validator/config/test路径和目标
     formal source-tree contract；
   - `formal_runs_maximum=1`、benchmark retry/sweep=false、training conditions
     unchanged=true、random initialization=true。

   原benchmark的throughput/ETA checks必须在formal lifecycle中继续保存为false；
   新状态只能表示为`failed-waived / user-authorized-performance-waiver`，不得表示为
   `performance-gate-passed`。Benchmark中memory、finite loss/gradient、GPU-only relation、
   optimizer/checkpoint I/O、four-rank identity、contention、eligibility和schedule等任一
   non-speed contract不通过时，waiver无效并停止。

5. **Source transition与重新验证。** 为接受waiver所需的source修改只允许涉及performance
   validator、base/D2-AF config binding及对应tests/documentation；不得修改models、
   diffusion schedule、relation builder/encoder/routing、loss、optimizer或training-loop
   数学。由于原CPU/smoke/eligibility/benchmark是在旧formal source hash上完成，waiver必须
   以source/target commit和exact Git diff hash显式授权这次validator-only transition，
   而不是重写旧artifact。目标commit上必须重新通过完整authority suite、registry
   validation、static source/diff audit和resolved-config fail-closed测试；不重跑scientific
   performance benchmark。

6. **Formal后的原评测不变。** 训练完成后仍只使用fixed final-online，依次执行已注册的
   five-path internal和一次fixed native evaluation。Internal/native gates、统计、sealed
   controls、最终科学分类和selectability条件全部不变；compact result和phase summary必须
   同时报告原performance failure、用户waiver、实际formal wall/throughput及最终结果。

下一步仅允许提交本plan-only waiver和append-only registry hypothesis；随后实现上述最小
hash-bound validator/config/tests，创建并提交immutable waiver contract，通过authority
verification后由worker fast-forward相同clean Git object并启动唯一formal run。

#### 2026-07-29 Phase 1B D2-AF0 performance waiver implementation / contract record

一次性waiver已按上述plan实现，且未改写原benchmark。Validator/config/tests logical
implementation commit为
`9c908ad87dce8806eb052b2a2627160b0a1bbe72`；tracked immutable contract commit为
`69d8cb025c89c0e776d0a4c03a8c158bbd0a3265`，文件
`experiments/contracts/p1_hoi_d2af_performance_waiver_s42_20260729.json` 的SHA-256为
`8a2d11c0febea603ac74328fbcd51622982740c4bef48597a0af71de7a53da97`。

实现保持两条分离路径：

- 原passing benchmark仍要求`status/classification/throughput/eta/formal_authorized`
  全部通过，且waiver fields必须为空；
- 原failed benchmark只允许exact五项
  `classification / eta / formal_authorized / status / throughput`为false，其他
  memory、finite、GPU-only、timing、optimizer/checkpoint I/O、four-rank identity、
  contention、schedule、eligibility、source identity和sweep contracts必须全通过。
  之后才读取tracked waiver，并返回
  `status=failed-waived`、
  `classification=user-authorized-performance-waiver`、
  `formal_authorization=explicit-single-run-waiver`和
  `original_gate_passed=false`。

Waiver exact绑定：

- benchmark summary：
  `53e9842d0522cf456a86eedc25d2a972cd00db3fb067113ff25f31f6117e1f33`；
- eligibility：
  `c52c0536423d7a17101829cb2b020316b9c6e0f7aa2cf39f33b984ffb39896b4`；
- source commit/contract：
  `1c6c3058478411361bf3e73830f900f660ae516b` /
  `68269a2cac8eaf6fd2b55b139bb2be5b5dbafde6e7f22496f5a894f18b843145`；
- target implementation commit/contract：
  `9c908ad87dce8806eb052b2a2627160b0a1bbe72` /
  `299d7a900c6a96264dd698c50ef476ea78d2b2efdfbb3b0e375d27d99101cc3e`；
- exact binary Git diff：
  `24d0dbb8abd96b56f6e745b0f08fcabeb0a50792a4737d8332ca6158aadec7c3`；
- changed paths严格为base/D2-AF config、trainer performance validator、
  `tests/test_hoi_d2af.py`和既有plan/registry六项；models、diffusion、relation、
  loss、optimizer、training loop、DataLoader和profiling均未修改；
- 唯一formal id：
  `p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729`，formal runs maximum 1，
  random initialization，benchmark retry/sweep/reclassification均false。

Authority在clean contract commit上使用真实recovered eligibility/benchmark和tracked
waiver直接调用formal validator，全部authorization与waiver checks为true；原五项
benchmark failed checks继续原样保存。Target source contract由current worktree与target
Git object两种算法独立重算，均为91 files /
`299d7a900c6a96264dd698c50ef476ea78d2b2efdfbb3b0e375d27d99101cc3e`。

Source implementation提交前的authority verification为：

- 完整`unittest discover`：`419/419` pass；
- D2-AF waiver定向测试：passing path、exact failure+waiver、missing waiver、
  extra non-speed failure、benchmark/waiver tamper全部通过；
- `py_compile`、registry validation（219 records）和`git diff --check`通过；
- formal output、metrics、state和checkpoint均不存在；CUDA workload、optimizer update、
  checkpoint load/write、internal和native均未启动。

下一步只允许提交本append-only lifecycle record，随后由worker发起Git fast-forward到相同
clean object；现场验证worker Python/data/artifacts、actual date、formal目录不存在、
resolved config无interpolation、same-context manifest/preflight和waiver validator后，
启动唯一formal run。不得运行第二次performance benchmark或任何优化sweep。

#### 2026-07-29 Phase 1B D2-AF0 checkpoint-race operational continuation（plan-only）

用户报告GPU workload早于ETA结束后，authority通过worker的loopback-only control channel
完成只读取证。核验时authority为`/data/yujinlun/InfBaGel-release`、branch
`phase/01b-hoi`、HEAD
`7202d32a7375e7197886c4f873688fd472e2c803`、worktree clean；worker checkout为
`/home/yujinlun/data/work/InfBaGel-release`、相同branch/HEAD且clean；核验时间为
`2026-07-29T23:15:09+08:00`至`2026-07-29T23:22:30+08:00`。重新扫描authority
working tree、Git history/refs/reflogs、registry、authority staging、worker checkout与
worker artifacts后，
`p1-hoi-d2af-checkpoint-race-continuation-s42-20260729`及stem
`d2af-checkpoint-race-continuation`均未使用。本commit只允许追加本plan与registry，
不得修改source、config或tests，不得加载checkpoint或启动GPU workload。

1. **失败事实与分类。** 唯一formal run
   `p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729`并未完成。它从seed-42随机初始化
   正常运行到第三次cadence save，在attempted `9,216,000` windows /
   `4,500` updates处以return code `1`退出；没有`training_state.json`或
   `metrics.json`，manifest继续保留`running`，internal/native均未启动。失败记录
   `operational_checkpoint_race_failure.json`的SHA-256为
   `a66fec685afb5cbb4079619de9417b7171af7e29244723f1deac9d4ba306d1b1`。
   根因是`code/train_hoi_prior.py::_save_checkpoint()`中每个rank在写自己的RNG
   sidecar前检查了主checkpoint和全部rank sidecar；rank 0/1/3先写后，rank 2把这些合法
   peer文件误判成overwrite collision。该错误不涉及D2-AF relation数学、loss、gradient、
   CUDA OOM、数据、磁盘容量或scientific gate，分类固定为
   `ddp-checkpoint-sidecar-existence-race-operational-failure`。

2. **失败与可恢复状态均不可改写。** 最后完整checkpoint固定为
   `6,144,000` windows / `3,000` updates，SHA-256
   `3c94f7344991cb38aab37fd8356cabe83a84b449d10505e0e46341490605287e`，
   四个rank RNG sidecar均存在并已逐文件hash。第三次cadence已写出的rank 0/1/3
   partial sidecar已无损移动到
   `operational_failures/checkpoint_race_windows009216000`；3 files /
   45,977 bytes，tree SHA-256
   `b5573764eceb388f6a28f10b4ed89b44bbbcdd430213dad490f6c8b5caa7f9dd`，
   内容未修改、未删除。原`train.log`、`returncode.txt`、resolved config、manifest、
   preflight、initial-stability和checkpoint artifacts全部保留，不得覆盖或伪装成成功。

3. **同一scientific lineage，而非第二次实验预算。** 唯一允许动作是从上述exact
   `6,144,000`-window checkpoint在同一formal run id内继续到原固定
   `61,440,000` windows / `30,000` updates。seed、split、model、sqrt-alpha-bar
   routing、100-point relation、loss/reduction/weight、optimizer及其state、LR、batch、
   DataLoader、profiling、checkpoint/validation cadence、budget和final-online规则全部
   不变；不得从`9,216,000` partial state恢复，不得重启from-random、创建第二manifest/
   formal id、延长预算或选择中间checkpoint。崩溃前未保存的1,500 updates丢弃并由exact
   RNG/optimizer state重放；accepted lineage仍为30,000 updates，但实际GPU总成本须另报
   已失败的4,500加continuation的27,000 updates，不得把重放隐藏为正常30,000-update wall。
   该continuation不授权D2-AF1、任何sweep、consistency、HSIPrior或Mixer。

4. **唯一source修复。** `_save_checkpoint()`只允许：

   - 每个rank在任何write前检查自己的RNG sidecar是否已存在；
   - 只有rank 0检查共享主checkpoint是否已存在；
   - 通过collective collision flag使任一rank发现collision时全部rank同步fail closed；
   - 全部rank通过collision preflight后增加barrier，再写各自sidecar；
   - 保留sidecar写完后的barrier、rank-0主checkpoint atomic write和最终barrier。

   peer sidecar绝不再作为collision；own sidecar或主checkpoint仍必须fail closed。不得修改
   checkpoint cadence/schema/value、RNG内容、optimizer/model state、训练循环数学或任何
   D2-AF source。必须增加collective顺序、peer-sidecar、own-sidecar和main-checkpoint
   collision regression tests。

5. **Hash-bound operational resume contract。** 修复后HEAD与checkpoint commit不同，
   因此resume只能通过tracked immutable continuation JSON和既有generic commit-transition
   guard共同授权。continuation contract必须精确绑定：

   - same formal run id与seed 42；
   - source/checkpoint commit `7202d32a7375e7197886c4f873688fd472e2c803`；
   - exact checkpoint path/basename/SHA和`6,144,000` windows / `3,000` updates；
   - 四个rank RNG sidecar SHA；
   - exact failure-record path/SHA/classification/return code；
   - exact partial archive path/tree SHA、文件数/bytes及原checkpoint目录已只清除该组
     partial files；
   - 原performance waiver target formal-source contract
     `299d7a900c6a96264dd698c50ef476ea78d2b2efdfbb3b0e375d27d99101cc3e`；
   - source commit到唯一implementation target commit的
     `git diff --binary` SHA、changed paths与target formal-source contract；
   - science/config/budget unchanged、same-run only、new formal budget false。

   允许的source-transition paths仅为base/D2-AF config、trainer checkpoint/provenance
   guards、D2-AF regression tests及本plan/registry。任何checkpoint/RNG/failure/contract/
   diff tamper、额外source path或current formal-source drift必须在GPU前停止。原failed
   performance benchmark和one-time waiver仍原样保留；不得重分类为performance passed。

6. **Resume execution与artifact规则。** Worker必须由authority committed clean object
   fast-forward到完全相同的target commit，现场验证machine-local Python、四卡空闲、
   data/assets/hash、checkpoint/failure/partial archive和source-transition。原manifest不
   重建；新增且不覆盖
   `resolved_hydra_config_resume.yaml`、`resume_preflight.json`、
   `resume_contract.json`、`resume.log`、`resume_returncode.txt`和
   `resume_initial_stability.json`。Resolved config必须无interpolation，并包含exact
   checkpoint、continuation contract及source/target/diff binding。续训在worker-owned
   persistent session运行；初始稳定、finite loss/gradient、显存和下一个完整cadence
   checkpoint验证后停止主动轮询。

7. **完成路径不变。** 训练完整结束后，`metrics.json`和`training_state.json`必须记录
   resumed-from checkpoint、source/target/diff provenance及完整30,000-update结果。
   Resume进程的内置wall/loss/validation/checkpoint-hash accumulators会从continuation
   启动点重新累计，且raw throughput以累计processed windows为分子，因此不得把该raw值
   报作完整run throughput；必须分别报告continuation wall/throughput、accepted-lineage
   active wall/throughput和包含失败重放的总GPU cost。`tools/experiment.py finish`使用
   既有hash-bound manifest transition，不修改该工具、不放宽dirty/overwrite检查。随后才按
   原注册顺序执行fixed five-path internal、fixed native、non-destructive worker-initiated
   recovery、双端tree/hash验证、compact result、`PHASE_1B_D2AF.md`和append-only
   completion record。任何新的operational failure均保留，不自动改run id、删artifact或
   改变科学协议。

下一步仅允许提交本plan-only amendment与append-only registry hypothesis；随后实现上述
最小checkpoint-race fix、D2-AF专用continuation validator/config/tests，并创建绑定唯一
implementation commit的immutable continuation contract。所有authority/worker CPU与
preflight contracts通过前不得恢复GPU训练。

#### 2026-07-29 Phase 1B D2-AF0 checkpoint-race continuation implementation record

上述plan-only amendment已提交为
`3b9c0c53bff1e09ec880a6795fd1fad550bc2495`。本logical implementation严格限制为checkpoint
save同步、D2-AF resume provenance config/validator与regression tests；未修改model、
diffusion schedule、relation builder/encoder/routing、loss、optimizer、DataLoader、
batch、budget、sampler或evaluator。

- `_save_checkpoint()`不再由每个rank扫描全部peer sidecar。每个rank只形成own-sidecar
  collision flag，rank 0额外形成main-checkpoint flag；NCCL collective MAX使任一collision
  在全部rank同步失败。collective通过后另有pre-write barrier，随后保持原own-sidecar
  atomic write、post-sidecar barrier、rank-0 main atomic write和final barrier。
- Base/D2-AF config只新增默认null的tracked continuation path/SHA字段；这些字段不进入
  scientific `_resume_contract`，因此不会改变旧checkpoint的model/optimizer/data contract。
- 原performance waiver仍要求其旧target formal-source contract。只有same-run resume且
  current source因本次registered fix变化时，validator才进一步要求一份exact
  checkpoint-race continuation contract；fresh run、其他checkpoint或无contract的
  changed-source resume继续fail closed。
- Continuation validator绑定原manifest、6,144,000-window checkpoint及四rank RNG、
  failure JSON、9,216,000 partial archive、原waiver target source contract、
  source-to-implementation binary diff、current execution diff和scientific-unchanged
  booleans。Generic resume commit-transition allowlist只增加本次D2-AF config/tests/contract
  paths；D2-AB exact/guard behavior保持通过。
- Regression覆盖peer sidecar不构成collision、own sidecar/main checkpoint仍拒绝、
  remote collision collective传播、changed-source waiver必须有continuation，以及
  checkpoint/RNG/failure/partial/source/execution/science全部绑定的positive fixture。

Implementation-stage定向verification为：D2-AF core `22/22`、D2-AF CPU/GPU lifecycle
`15/15`、D2-AB resume regression `16/16`通过，`py_compile`通过。此时尚未创建immutable
continuation JSON，未加载worker checkpoint，未启动GPU、optimizer、internal或native。
下一步先运行完整authority suite、registry validation、static diff audit并提交本logical
implementation；随后以该commit为唯一implementation target创建tracked contract和
append-only binding record。Contract commit不得修改formal runtime source。

完整authority verification随后通过：`unittest discover`为`424/424`，registry
validation为222 records，`py_compile`和`git diff --check`通过。Source commit
`7202d32a7375e7197886c4f873688fd472e2c803`的D2-AF formal-source contract重算为
91 files /
`299d7a900c6a96264dd698c50ef476ea78d2b2efdfbb3b0e375d27d99101cc3e`；
implementation worktree同算法为91 files /
`daa57294f4d25db4591a2ef6bcbe8157ca812b99b3b1dfe4c6c01aaf23c2ffd4`。
Changed paths严格为base/D2-AF config、trainer、D2-AF tests和既有plan/registry六项；
models、diffusion、relation、loss、DataLoader、sampler、evaluator、`tools/experiment.py`
均无diff。额外4-process CPU Gloo contract证明无collision时四rank全部通过、rank 2
own-sidecar collision时四rank全部同步拒绝。上述verification仍未读取scientific
checkpoint或启动GPU workload。

#### 2026-07-29 Phase 1B D2-AF0 checkpoint-race continuation contract binding

唯一immutable continuation contract已在implementation commit
`b7248bba3e77234c8f2a5993d8bf3ee8a1db2757`之后创建为
`experiments/contracts/p1_hoi_d2af_checkpoint_race_continuation_s42_20260729.json`，
SHA-256为
`1a4ddf3b220b96f7aea0f1de7c0b8fd3fd9458eb913d284aaacc85a7fa226424`。
该contract只授权原run
`p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729`从唯一完整的
6,144,000-window / 3,000-update checkpoint继续同一manifest；不创建新formal run、
不从随机初始化重启、不使用不完整的9,216,000-window保存点，也不改变模型、relation、
loss、optimizer、batch、budget、DataLoader或evaluation协议。

Contract精确绑定：

- 原running manifest SHA-256
  `985192f686de2d4330cb82c826b648a08d12b7ed55c0bd4c8d196951d05b589b`；
- resume checkpoint SHA-256
  `3c94f7344991cb38aab37fd8356cabe83a84b449d10505e0e46341490605287e`
  及四rank RNG sidecar hashes；
- preserved operational failure SHA-256
  `a66fec685afb5cbb4079619de9417b7171af7e29244723f1deac9d4ba306d1b1`
  与partial archive tree SHA-256
  `b5573764eceb388f6a28f10b4ed89b44bbbcdd430213dad490f6c8b5caa7f9dd`；
- source commit `7202d32a7375e7197886c4f873688fd472e2c803`、implementation target
  `b7248bba3e77234c8f2a5993d8bf3ee8a1db2757`及binary diff SHA-256
  `19778c2dac54ae080b241dabb1215dd55d6defa0e231c301ccee2ed48d43498a`；
- source/target formal-source contract SHA-256分别为
  `299d7a900c6a96264dd698c50ef476ea78d2b2efdfbb3b0e375d27d99101cc3e`
  和
  `daa57294f4d25db4591a2ef6bcbe8157ca812b99b3b1dfe4c6c01aaf23c2ffd4`。

JSON syntax、contract SHA、source transition diff与changed paths均在authority重算通过。
Continuation最终accepted lineage仍固定为61,440,000 windows / 30,000 updates；失败尝试
执行4,500 updates，continuation从3,000继续27,000 updates，因此实际GPU执行成本必须显式
报告为31,500 updates，不能将重放隐藏在raw resumed throughput中。本binding commit之后，
formal workload完成并关闭manifest前不得再改变execution HEAD。下一步仅允许worker发起
Git fast-forward、验证相同clean object与machine-local Python，生成不覆盖原文件的resolved
resume config/preflight/contract/log artifacts，并恢复原persistent lineage。

#### 2026-07-30 Phase 1B D2-AF0 formal training completion and recovery record

唯一授权的formal run
`p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729`已在worker
`phase/01b-hoi@044227fe512a9ee6d1c2a1bc898d3b8a2c6ca706`完成，并于
`2026-07-30T04:56:20+08:00`闭合。`resume_returncode.txt=0`；accepted lineage严格为
61,440,000 windows / 983,040,000 frames / 30,000 optimizer updates。原失败尝试执行
4,500 updates，continuation执行27,000 updates，因此实际GPU updates为31,500；该重放成本
保留为operational accounting，不改变固定科学预算。训练仍从seed-42随机初始化，released、
D2-X/D2-AE/D2-AC/D2-AD、prior、EMA或consistency checkpoint加载数均为0。

Fixed final-online checkpoint为
`p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729_windows061440000.pth`：

- checkpoint SHA-256：
  `483c63ecaeb6dbf5a0a54400e0eecec722ff6df6d72226ce263e7fe053e412e2`；
- model-state SHA-256：
  `7b6e333724f21490c96a0599103cc7eb087b9452e64a8d3c2b9a5ce85ae704bb`；
- 参数量：30,087,401；optimizer state为129项，step min/max均为30,000；
- learned alpha / `tanh(alpha)`：
  `-0.0925037190 / -0.0922407731`；
- final validation total：`0.0502382831`；loss、model、optimizer与required gradients均
  finite，AMP overflow为0；
- 每rank最小显存headroom为19,369,492,480 bytes。

Cadence完整性为20个main checkpoints与80个rank RNG sidecars，missing/extra均为0。
必须保留原checkpoint-race failure
`a66fec685afb5cbb4079619de9417b7171af7e29244723f1deac9d4ba306d1b1`、
partial archive tree
`b5573764eceb388f6a28f10b4ed89b44bbbcdd430213dad490f6c8b5caa7f9dd`
以及两个未启动GPU的旁路operational failure目录。

Timing只允许报告一致口径：

- continuation-only throughput：`3232.575359 windows/s`；
- accepted-lineage artifact throughput：`3218.215477 windows/s`；
- serialized raw `3591.750399 windows/s`使用全预算windows除以continuation-only wall，
  不得作为formal完整训练吞吐。

Worker lifecycle已执行`finish`与run-local `register`。关键SHA-256为：

- manifest：
  `49371a577a037444aef47fd5fda64f5d147ecd712247308b99b675d1edee55d3`；
- run-local registry：
  `7d1ba1bd99cf2c4dcf95a2300522f828e832d70b0f12afe3213c547471dedf50`；
- metrics：
  `25b172f21d78d97412cb4eeeb79b43566d7e488286c383127a4edf0272c11903`；
- training state：
  `8dcb3ea4e1e39d661bcef138de6ff347731db8eeb88213fe0b4e0ba83204f8a4`；
- resolved resume config：
  `6845d032e48027a35ccbd20169d118cc81429ac652c31f8c615e5474d45fe870`；
- resume preflight：
  `5f404f407d022c7db80fb1c781f69828d7abccbcecbd3f995cdd2049cace4e51`；
- resume contract：
  `35240fb486b891a520ad3f08c9e557594349adc77b4868eca9547b845f540f2f`；
- formal completion verification：
  `a6263835cf79c6b803275c3d9c96c269aa1c2e75b1c8fea3fce4b4b56f7f1ec1`。

Worker随后发起non-destructive `rsync -aH --partial`，未使用`--delete`，将完整156-file /
7,227,356,886-byte formal tree恢复到
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729-recovery-r1`。
Worker与authority的`sha256_path`均为
`9bff3d9a182138ee30ca586b10d71f689e6aa0c7345d2b1052fe0ea15251dc6c`，
重复`rsync --checksum --dry-run`无差异。Fixed final-online不可选择、不可resume、不可用于
任何后续prior初始化；下一步仍只允许预注册的five-path internal和一次fixed native。

#### 2026-07-30 Phase 1B D2-AF0 evaluation provenance hardening（plan-only）

在任何internal/native GPU workload前，对现有D2-AF evaluator做一次纯provenance
hardening。核验时authority为clean
`phase/01b-hoi@d51057c35485d9b5e1abc846a55dc2f4324f9659`，worker仍为clean
`phase/01b-hoi@044227fe512a9ee6d1c2a1bc898d3b8a2c6ca706`。Identifier audit覆盖authority
tracked worktree、Git history/refs/reflogs、staging、authority results及worker checkout/
artifacts；以下identifier均未使用：

- `p1-hoi-d2af-evaluation-provenance-hardening-s42-20260730`；
- `d2af-evaluation-provenance-hardening`；
- `p1-hoi-d2af-sqrt-alpha-bar-reliability-internal-s42-20260730`；
- `p1-hoi-d2af-native-eval-s42-20260730`。

本amendment不改变scientific protocol、checkpoint、cohort、sampler、intervention、metric、
threshold、bootstrap、gate、classification precedence或native evaluator，只使已注册事实
fail closed：

1. **Internal formal-lineage binding。** Internal CLI/resolved config/manifest必须显式接收并
   hash-bind completed formal `manifest.json`、`metrics.json`、`training_state.json`、
   `resume_contract.json`与fixed final-online。Validator必须同时证明：

   - manifest已completed并绑定唯一formal run；
   - metrics/state均为61,440,000 windows、983,040,000 frames、30,000 accepted updates；
   - final checkpoint basename/SHA、model-state SHA、architecture、seed、random origin、
     20 main checkpoints与80 RNG sidecars一致；
   - checkpoint source commit为
     `7202d32a7375e7197886c4f873688fd472e2c803`，execution target为
     `044227fe512a9ee6d1c2a1bc898d3b8a2c6ca706`，binary diff为
     `f0cba48ae5d1ba271750ef5d7c042d1b04e8ec6b5e60df00fca5f19c1db8f609`；
   - operational continuation只来自registered 6,144,000-window checkpoint，原failure/
     partial archive bindings保持一致；
   - fixed final-online checkpoint SHA仍为
     `483c63ecaeb6dbf5a0a54400e0eecec722ff6df6d72226ce263e7fe053e412e2`。

2. **Internal RNG/input identity。** Batch size从“64的任意因数”收紧为严格`8`，因为seed label
   使用chunk index，改变batch会改变实际随机流。五条paths的first window必须比较完整
   `path_local_model_inputs` identity，而不只比较fixed history；至少覆盖history、global
   BPS、local goals、object/world rotation references及全部exogenous model inputs。
   Later windows仍按各自causal generated history分叉，不能被错误要求相同。

3. **Internal raw-artifact closure。** Summary必须记录并hash-bind五个raw variants：
   `full_rho`、`unit_rho`、`relation_gate_ablated`、
   `temporal_correspondence_permuted`、`left_right_role_swapped`，以及paired noise、
   paired conditioning、causal overlap和reliability appendix。Existing seven internal
   decision/gate booleans、paired sequence uncertainty、selection hash与schedule hash必须从
   raw artifacts重算一致；不得新增诊断路径或改变gate math。

4. **Native upstream closure。** Native CLI/resolved config/manifest必须显式hash-bind同一formal
   manifest/metrics/state/resume contract与internal summary。Native preflight必须重新读取并
   验证五个internal raw variants、supporting paired artifacts、seven decision/gate booleans及
   它们的SHA；只接受上述唯一resumed lineage。Internal mechanism无论正负仍执行native，
   只有contract failure停止。

5. **Regression gate。** GPU前至少覆盖：

   - completed resumed-lineage positive fixture；
   - missing/tampered formal manifest、metrics、state或continuation contract；
   - wrong source commit、execution target或binary diff；
   - checkpoint/final lineage mismatch；
   - internal batch size非8；
   - first-window完整input identity mismatch；
   - missing/tampered internal raw variant、gate或hash；
   - native对formal/internal binding的positive与negative fixtures。

6. **Allowed source scope与stop rule。** Logical implementation只允许修改
   `tools/run_hoi_d2af_internal.py`、`tools/run_hoi_d2af_native_evaluation.py`、
   `tests/test_hoi_d2af_eval.py`及本plan/registry；若测试需要，可在同一D2-AF eval test文件
   内增加fixture/helper。不得修改models、diffusion、relation、loss、training、dataset、
   official evaluator或sealed artifacts。任一hardening contract失败分类为
   `diffusion-reliability-contract-failure-stop`，不得启动GPU workload。

Hardening实现与full authority suite通过后，worker只允许Git fast-forward到完全相同的clean
commit，现场验证Python/worker expert/data/evaluator hashes，按实际日期创建未使用run id，
先registered fixed five-path internal，再无条件执行唯一一次fixed 438×3 native。不得启动
D2-AF1、second training、sweep、consistency、HSIPrior或Mixer；Phase 1B completion record
提交前不进入Phase 1C。

#### 2026-07-30 Phase 1B D2-AF0 evaluation provenance hardening implementation verification

上述plan-only hardening已按注册范围实现，未修改model、diffusion、relation builder、
training/loss/data、scientific sampler、metric、threshold、bootstrap或gate math。实现只涉及
两个D2-AF evaluation runner、同一D2-AF eval regression test，以及本append-only
plan/registry：

- internal现在显式绑定completed formal manifest、metrics、training state、resume contract、
  final-online checkpoint与`formal_completion_verification.json`；
- recovered formal cadence必须恰好包含20个main checkpoints与80个rank RNG sidecars，
  并逐文件验证regular/non-symlink、basename、kind/rank/window、bytes、SHA、checkpoint
  schema/commit/progress/architecture/RNG pattern及RNG exact state schema；
- 6,144,000-window resume checkpoint与四个sidecars、checkpoint-race continuation、
  source/target commit及binary diff均交叉绑定；
- internal batch严格锁定为8，五条paths固定24个`chunk × window` stream coordinates；
  first-window完整model-input hashes必须跨paths相同，later windows只允许同一路径causal
  history；
- native不再信任internal summary/support自报布尔，而是从五个raw variants、paired noise、
  paired conditioning、causal overlap与reliability appendix重新验证cohort/order、
  selection、500-step schedule、relation trace、comparisons、seven gates及decision evidence；
- artifact closure只接受metrics run root内的regular files，拒绝absolute path、escape与最终
  symlink；
- 最终审查识别的`rho_*_max_abs`/schedule sentinel NaN比较绕过已在GPU前修复；所有这些
  scalar现在必须为finite real number，并新增self-consistent NaN raw-artifact regression。

真实recovered formal tree验证通过：100个cadence files、7,226,924,444 bytes，全部16项
formal-lineage checks为true。Authority验证为：

- `tests.test_hoi_d2af_eval`：17/17 passed；
- full `unittest discover -s tests`：432/432 passed；
- existing registry：226 records validated；
- project Python 3.8.20 `py_compile` passed；
- `git diff --check` passed；
- adversarial regressions覆盖different noise、future-GT/exogenous forgery、missing window、
  forged cohort、empty relation、zero schedule、nonfinite relation scalar/sentinel、
  nonfinite summary与excessive history。

本verification期间checkpoint load、optimizer update与GPU workload均为0；fixed internal/
native run id仍未使用。下一步先提交该logical implementation，再追加只绑定实际implementation
Git object的governance-only record；随后worker仅可fast-forward到相同clean execution
object并执行fixed internal。Internal contract失败立即停止；internal科学结果正负均不改变
随后唯一native的执行要求。

#### 2026-07-30 Phase 1B D2-AF0 evaluation hardening implementation binding

Logical implementation已提交为
`3d4ff1eb5c57b1b08537859dca8e895bc428a26d`，tree为
`c46b6b4dfad7ecd3b8b90af6cf71e2ed8fc7ecf7`。相对plan-only commit
`7a484fe18dc28e29e30b2966d966825823130c0b`的binary diff SHA-256为
`b07123b8086cd523cbe3c89006ce7264e13f08468ba6ae13844b48bb6ecf8b34`，changed paths严格
为两个D2-AF evaluation runners、一个D2-AF eval test及plan/registry五项。该commit之后
尚未启动checkpoint load或GPU workload。Worker必须fast-forward到包含本binding的最终clean
HEAD；其source代码必须与上述implementation object相同，随后才可创建registered internal
manifest。

#### 2026-07-30 Phase 1B D2-AF0 fixed internal causal diagnostic completion

Reportable worker run
`p1-hoi-d2af-sqrt-alpha-bar-reliability-internal-s42-20260730` 在
infbagel-4gpu/node01 的clean evaluation execution commit
`a4cdcf09f84553159be10c555ff8a6773b65d3aa` 上完成，exit code为0，runtime为
`538.3266074708663 s`。Manifest在GPU workload前由相同execution context创建；resolved
config无unresolved interpolation，worker source/Python/data/evaluator/formal-lineage
preflight全部通过。Run只加载fixed final-online checkpoint
`483c63ecaeb6dbf5a0a54400e0eecec722ff6df6d72226ce263e7fe053e412e2`，
没有optimizer、training update、checkpoint write/selection或official test。

固定D2-O cohort严格为64 sequences / 192 windows、phase offsets `(14,56,98)`、selection
SHA-256
`1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`。五条500-step
paths为`full_rho / unit_rho / relation_gate_ablated /
temporal_correspondence_permuted / left_right_role_swapped`；initial latent、每一步posterior
noise、condition、history、window ordering和restoration按注册协议配对。全部formal
lineage、schedule、current-state/current-timestep、first-window input identity、later-window
path-local causality、raw artifact closure、finite、mask和no-write contracts通过。

Learned alpha/gate为
`-0.09250371903181076 / -0.09224077314138412`，但七个primary mechanism gates全部失败：

- full-rho − unit-rho direct-hand union 5-cm F1：
  `-0.0017534958725371544`，CI
  `[-0.008731448896887319,0.005308128210568779]`；
- unit-rho − full-rho GT-contact-frame distance：
  `+0.026833171113742785 cm`，CI
  `[-0.08000149312555654,0.15226497063856367]`；
- full-rho − gate-ablated direct-hand union 5-cm F1：
  `+0.012897639098904238`，CI
  `[-0.027088665174081348,0.05341855160295159]`；
- gate-ablated − full-rho distance：
  `+0.4767763515114704 cm`，CI
  `[-0.05841018885359712,1.0529788793694062]`；
- full-rho − temporal-permuted direct-hand union 5-cm F1：
  `+0.002165564488603878`，CI
  `[-0.031244155095278628,0.043877801226154534]`；
- temporal-permuted − full-rho distance：
  `+0.23349129590381712 cm`，CI
  `[-0.31420466831855565,0.8945656086402927]`；
- full-rho − role-swapped direct-hand left/right macro-F1：
  `+0.013202214669035858`，CI
  `[-0.008038313502508903,0.035691421866214094]`。

因此`relation_path_used / schedule_reliability_passed /
temporal_routing_passed / role_binding_passed / mechanism_passed`均为false，
`internal_status=unused`，classification严格为
`diffusion-reliability-internal-unused-continue-native`。非零learned gate不替代causal
evidence；internal结果不选择checkpoint，也不取消已注册的唯一native evaluation。

Metrics、manifest、resolved config、preflight和run-local registry SHA-256分别为
`38e5a641707cff9a880fea5d4c90b7d936290e912237e60baa2ecf25bde8ff52`、
`1b3ad9792aa5df897982a482453561d2dfb7e56571404b19b060d9c16013ed96`、
`c05c69adeb171dd018b814df6d72197b59e4c6c70bbe0a6bd9a5eb0243fb1e21`、
`b1eb1aee4c3b2c0f67cc4a9bd810e7a3b7c570dd6db37028c4d44b610cb7ca7c`和
`dc1b0e8526ff636a9642be8262b126a908ed0efed18ecec175a480eefd754efd`。
五条raw paths、paired noise/conditioning、causal overlap和diffusion-reliability appendix
均逐项hash闭合。Worker发起non-destructive recovery后，17 files / 224,452,243 bytes的
worker/authority tree SHA-256均为
`5d28e3abc02dcf62f781270fd0391e44f64f4172b7ff705257995be63faffeee`，
checksum dry-run为0 differences。

Internal正式run之前，一个launch-control wrapper因未canonicalize Python symlink而在创建
run目录/manifest/artifact/GPU workload之前退出，正式run id未被占用；完成后一次tree-hash
wrapper因错误cwd出现`ModuleNotFoundError: tools`，但artifact已成功传输且随后在正确repo
root重算通过。两者均为operational wrapper noise，不是scientific/run failure，也不授权retry。

#### 2026-07-30 Phase 1B D2-AF0 fixed native evaluation completion

Reportable worker run `p1-hoi-d2af-native-eval-s42-20260730` 在相同clean commit
`a4cdcf09f84553159be10c555ff8a6773b65d3aa` 上完成，exit code为0，runtime为
`378.46187578188255 s`。Official evaluator执行438 sequences × 3 windows、500-step
unguided production diffusion、final-online、seed 42和10,000 paired sequence bootstrap；
CFG、guidance、scene conditioning、dynamic perception和consistency全部off。Sealed D2-X与
D2-AE aggregate/per-sequence artifacts只读复用，没有重生成；D2-AE checkpoint未加载，也未
用于initialization、resume、selection或candidate。

D2-AF0 target point estimates为：

- end-object `5.5734798312187195 cm`，Txy `4.607994854450226 cm`，
  FS `0.3595817663400341`；
- contact precision/recall/F1/coverage
  `0.7909342632567542 / 0.5990408567110541 /
  0.6410550040393033 / 0.4834384286439081`；
- Pbody `3.5893259727196254`，hand penetration `0.2268892808331516`；
- MPJPE `12.422095239162445 cm`，Troot/Tobj
  `8.447693288326263 / 17.303804395245994 cm`，Oobj
  `1.0007245875630288`。

第一优先级D2-AE single-factor repair gate失败：

- AF−AE contact F1：
  `-0.0008888491232585847`，CI
  `[-0.02435946905539561,0.022823955375799218]`；
- AF−AE recall：
  `+0.0029022340549778826`，CI
  `[-0.023672364012952776,0.02978128354401244]`；
- AF/AE end-object mean-ratio：
  `1.2920819122177505`，CI
  `[1.224156033921259,1.3639746920239342]`；
- AF/AE FS ratio：
  `0.9013030413462073`，CI
  `[0.8455771344455049,0.960204522985284]`，仅该subgate通过。

D2-X transfer也未通过：AF−X contact F1为
`+0.003629064933324414`，CI
`[-0.016673086777549785,0.024194001573353836]`；recall为
`+0.004586162666655739`，CI
`[-0.018407816634206476,0.02811076350108559]`；released gap closure仅
`0.040398463734960754`，contact F1点估计低于`0.6598838781`。

D2-X protection gate对end-object、Txy和Tobj失败，ratio CI upper分别为
`1.5730099778138071 / 1.1801967402975144 / 1.1099017284465105`；FS、Pbody、
hand penetration、MPJPE、Troot、Oobj和contact-precision checks通过。固定181-sequence
penetration finite mask通过，sequence-ID SHA-256为
`2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec`。
Released-95%-effectiveness gate失败。

按预注册classification priority，headline classification严格为
`diffusion-reliability-ae-repair-negative-stop`；`checkpoint_selected=false`、
`selectable_autonomous_diffusion_candidate=false`、`D2-AF1/consistency=false`，
HOIPrior search closed。

Metrics、manifest、resolved config、resolved target、preflight、run-local registry、
aggregate和per-sequence SHA-256分别为
`94fc71cd3d3fbbe87ac6ec38246e39fb0c965d630fd7626c604f0983a1248f56`、
`10498fb42a02501859cfd0aaab484a7a606ee5f711134d7f009519723655de06`、
`1aacaccee84eddaab141e3f4b31cfd3770db247d887512307ef279b3c549e4a9`、
`d956821b273fd70dec1aa2b5f58db4b16b40ebf8f5ba71895e2ae9b382d593fe`、
`8d9e8d2139e145487ea4193ca4a8b4b2fe89fc9afa8e0e6b5409b3238ff32fe9`、
`31c1571a599350036e9b5c57ee5f0f77ed34f3dea2aae4657188b8e94d792fcc`、
`417c245df047e4fd7724c7ddcc7f0884fffd5bda934fefe465fb904da400f488`和
`7252931861dd2d4e60476a05cd7dd35d67aa7369995687de2dc9bcbc67c8acd5`。
Worker发起non-destructive recovery后，16 files / 2,514,430 bytes的双端tree SHA-256均为
`40a9925468e54966f726b2cccec4f55aa53caa92f2a0da188dccc435ebc5bd21`，
checksum dry-run为0 differences。一次recovery wrapper最初只创建空staging目录而未传文件；
同一空目录随后通过无`--delete` rsync完整恢复并hash闭合，保守分类为post-completion
transfer-wrapper no-op，不改变任何workload/artifact/scientific result。

#### 2026-07-30 Phase 1B D2-AF0 recovery and final closure

D2-AF0的CPU、functional smoke、clean-signal eligibility、failed performance benchmark、
user-authorized waiver、formal checkpoint-race failure/continuation、fixed internal和fixed
native全部按append-only lifecycle保留。Formal/internal/native recovered trees分别为：

- 156 files / 7,227,356,886 bytes /
  `9bff3d9a182138ee30ca586b10d71f689e6aa0c7345d2b1052fe0ea15251dc6c`；
- 17 files / 224,452,243 bytes /
  `5d28e3abc02dcf62f781270fd0391e44f64f4172b7ff705257995be63faffeee`；
- 16 files / 2,514,430 bytes /
  `40a9925468e54966f726b2cccec4f55aa53caa92f2a0da188dccc435ebc5bd21`。

三棵树worker/authority完全一致；worker发起的
`rsync -aH --checksum --dry-run --itemize-changes`均exit 0、无itemized output。
Formal manifest/metrics/state/checkpoint cadence和80个rank RNG sidecars、internal九项raw
artifact closure、native aggregate/per-sequence/author log/resolved target均逐项hash验证。
所有operational wrapper/checkpoint-race记录保留，没有覆盖、删除或伪装为scientific pass。

Tracked compact result为
`experiments/results/p1_hoi_phase1b_d2af_sqrt_alpha_bar_reliability_s42_20260730.json`，
phase summary为`docs/phase_summaries/PHASE_1B_D2AF.md`。Final classification为
`diffusion-reliability-ae-repair-negative-stop`，internal classification为
`diffusion-reliability-internal-unused-continue-native`。Fixed final-online不可选择；
不merge/tag，不启动D2-AF1、第二次training、resume、checkpoint selection、performance/
architecture/worker sweep、consistency或任何新HOIPrior方向。

Phase 1B HOIPrior search在此关闭。本session不启动Phase 1C。下一独立session唯一entry point
是Phase 1C HSIPrior的dated plan-only preregistration；HSIPrior必须从seed-42随机初始化，
不得加载released/author/D2-X/D2-AE/D2-AF或任何HOIPrior checkpoint。

#### 2026-07-30 Phase 1B user-directed reopening and lean iteration workflow

用户在阅读D2-AF0结果后明确决定暂不关闭Phase 1B，并授权为后续HOIPrior迭代建立一套
精简、固定的执行流程。上述D2-AF0 closure、negative classification、artifact和checkpoint
non-selection仍是不可变历史事实；本amendment只向前覆盖“不得再做任何HOIPrior方向”和
“下一入口只能是Phase 1C”的治理决定，不改写任何既有科学结果。

Phase 1B现重新开放，但当前只授权两个非GPU交付物：

1. 形成`docs/HOIPRIOR_ITERATION_WORKFLOW.md`，将未来单次HOIPrior迭代固定为read-only
   evidence review、一次preregistration、一次logical implementation、必要且按变更范围触发的
   CPU/functional/performance gate、一个from-random formal lineage、一个internal、一个native、
   一次recovery和一个completion；
2. 形成`docs/HOIPRIOR_EVIDENCE_INDEX.md`与
   `docs/prompts/CLAUDE_CODE_OPUS5_HOIPRIOR.md`，让Claude Code Opus 5在不默认读取
   `AGENTS.md`的前提下仍显式加载仓库规则、实验事实和双服务器约束。

固定瘦身原则是：不减少formal budget、模型/梯度/显存检查、paired causal rollout、official
438x3 native coverage、metric、bootstrap或failure retention；只删除重复的date-transition、
binding/hardening/recovery-binding记录、unchanged artifact重复hash、documentation-only full-suite
rerun、reportable start前本地wrapper typo登记和稳定detached run持续轮询。完整规则以
`AGENTS.md`和`docs/HOIPRIOR_ITERATION_WORKFLOW.md`为准。

本amendment本身不选择新机制，不创建candidate run id，不修改model/data/loss/diffusion/
sampler/evaluator，不加载checkpoint，也不启动GPU。Claude Stage A必须只读分析全部Phase 1B
证据，给出恰好一个最值得做的实验并等待用户明确批准。只有批准后，才能为该具体机制另行
追加dated scientific hypothesis并按精简workflow一次性完成implementation、training和
evaluation。剩余formal预算只有一次，不允许sweep或多个候选训练。

Phase 1C HSIPrior因此延后，直到该用户批准的单个HOIPrior实验完成或停止。不得自动启动
HSIPrior、Mixer或consistency。

#### 2026-07-31 Phase 1B D2-AG self-conditioned relation source 预注册（plan-only）

用户在阅读 D2-AF0 negative 结果并完成 read-only Stage A 审阅后，明确批准 Phase 1B
`EP:6872` 所剩的唯一一次 formal 预算用于本方向 D2-AG。本 amendment 执行前，authority
checkout 为 `/data/yujinlun/InfBaGel-release`、branch `phase/01b-hoi`、HEAD
`fa2c73e48830d9545d5624eb72260ba40c3ce4cb`
（`Add tiered subagent types and standing dispatch authorization`），worktree clean；
核验时间为 `2026-07-31T02:08:08+08:00`。重新扫描 working tree、全部 Git
objects/refs/reflogs、append-only registry（232 行）、authority/worker staging names 与
worker checkout 后，`D2-AG`、`d2ag`、`p1-hoi-d2ag-*`、`selfcond`、`self_cond` 和
`self-cond` 均未被用作 identifier；历史 JSON SHA 中偶然出现的字节子串不构成 identifier。
Integration baseline `b9a158f75ab0740c91c9cfc8863a65fa381b014c` 是当前 HEAD ancestor
（`git merge-base --is-ancestor` 通过），禁止分支 `feature/independent-hoi-hsi-priors`
（`860ec8ca10cb5d6bed9d901560d3eb3d811a8143`）不是 ancestor。Source audit 证明
`SparseCurrentStateRelationField.forward` 已把 relation 几何来源 `current` 与 trunk
embedding `motion` 作为两个独立入参接收，因此"改变 relation 几何来源"可以在不改动
relation builder、encoder、routing 或 writeback 数学的前提下实现，没有结构性阻塞。
本 commit 只允许修改本 plan 和 registry；不得包含 source change、checkpoint load、
GPU workload、训练或评测。

1. **Sealed evidence 与唯一假设。** D2-AE0 已证明 sparse relation path 被使用、固定
   temporal correspondence 有因果作用、left/right role binding 是结构性的，但 native
   contact F1/recall 相对 D2-X 只有 `+0.004518/+0.001684` 且 CI 跨零。D2-AF0 在其上
   施加 canonical `sqrt(alpha_bar)` 衰减后，internal `full_rho − unit_rho` direct-hand
   union 5-cm F1 为 `-0.0017534958725371544`、`internal_status=unused`；native contact
   F1 为 `0.6410550040393033`、AF−AE F1 为 `-0.0008888491232585847`、AF−X F1 为
   `+0.003629064933324414`（CI 跨零）、released gap closure 仅
   `0.040398463734960754`，并在 end-object/Txy/Tobj 上 protection 失败，classification
   为 `diffusion-reliability-ae-repair-negative-stop`。把两者放在一起，未被检验的共同
   前提是：D2-AE/D2-AF 的可变锚点 relation 一直建立在 `x_t` 上，而在 500-step
   production rollout 的绝大多数步里 `x_t` 的 frames 5/10/15 是高噪声几何，因此
   internal paired 干预看得见的"因果使用"并不等于 rollout 中存在可用的几何信号。
   D2-AG 的唯一可证伪假设是：把可变锚点的几何来源换成模型自身 detached `x0_hat`
   （train/sample 两侧对称），可以在不改变 relation 几何、参数量、loss、budget、
   sampler 协议和 scene-free provenance 的前提下，让同一条 relation path 在 rollout
   中获得低噪声几何，从而修复 train-to-rollout 的 contact transfer gap。本 hypothesis
   不得被扩展为 scene conditioning、HSIPrior、Mixer、consistency、old-checkpoint
   init、guidance/CFG、SNR/timestep loss weighting、新 loss、rollout exposure 或任何
   sweep。

2. **唯一 manipulated factor。** 相对 D2-AE/D2-AF 基线，只改变 sparse relation field
   在可变 temporal anchors `5/10/15` 上读取几何的来源张量；writeback 数学保持 D2-AE
   原式且不启用 D2-AF 的 `sqrt(alpha_bar)` 衰减：

   \[
   H'_t=H_t+\tanh(\alpha)\,r_{a(t)}(s),
   \]

   其中 `s` 为 relation source state。训练侧对每个样本独立抽取
   `m_i\sim\mathrm{Bernoulli}(p=0.5)`：

   \[
   \hat x_0^{(i)}=\bigl[f_\theta(x_t^{(i)},t_i)\bigr]_{\text{no-grad},\,r\equiv 0},\qquad
   s^{(i)}=\mathrm{sg}\bigl[\hat x_0^{(i)}\bigr],\qquad
   s^{(i)}[:,{:}2]\leftarrow x_t^{(i)}[:,{:}2].
   \]

   被选中的样本先在**同一 timestep** `t_i` 上做一次 `torch.no_grad()` 估计前向，该前向
   本身使用现行 D2-AE 式 `x_t` source（即与 D2-AE 前向数值等价）；该 no-grad 前向只在
   被选中的子集上运行。未被选中的样本（`m_i=0`）取 `s^{(i)}=x_t^{(i)}`，**即完全等同于
   D2-AE 的现行行为**。两种情况下 relation field 都**照常激活**，不存在 relation 置零
   分支。采样侧维护 `prev_x0`：

   \[
   s=\begin{cases}x_t,& \text{prev}=\varnothing\ (t=499)\\
   \text{prev},&\text{otherwise}\end{cases},\qquad
   s[:,{:}2]\leftarrow x_t[:,{:}2],\qquad \text{prev}\leftarrow \hat x_0 .
   \]

   `prev_x0` 固定为该步 model 的**原始（raw）** `x0_hat`，即 `prepare_clean_x0` **之前**
   的值：`prepare_clean_x0` 的 history restoration 与可选 SO(3) 投影只作用于 sampler
   自身的 `clean`，不作用于 `prev_x0`；两侧均**不**对 `s` 做 SO(3) 投影，且都只用同一处
   手工 pin `s[:,:2]=x_t[:,:2]` 恢复前两帧，以保证两侧 source 构造逐字节对称。锚点 0 与
   2-frame history 在两侧都完全保持现状。`p=0.5` 是注册的
   固定值，不是可扫参数；不得改为 schedule、learned gate、per-anchor 或 per-timestep
   概率。不得使用 clean target、future GT、stored per-frame BPS、stored relation、
   contact label、Scene 资产或任何 sampler-only source 构造 `s`。

   **随机源与 eval-mode 契约（注册为硬性实现条件）。** Bernoulli mask 必须从一个
   **独立的 `torch.Generator`** 抽取，其 seed 由
   `cfg.seed * 1_000_003 + processed_windows + rank` 导出（与
   `code/train_hoi_prior.py:3869-3870` 的既有 per-rank 派生式同构）；no-grad 估计前向
   必须在 DDP-unwrapped inner module 的 `inner.eval()` 下执行，并以 `try/finally`
   恢复原 train 模式。`timesteps` / `noise` 的抽取点
   （`code/train_hoi_prior.py:3758-3761`）保持逐字节不动。理由必须一并记录：正式
   training 的 `_forward_losses` 调用点不传 `generator`
   （`code/train_hoi_prior.py:4630-4634`），因此 `timesteps`/`noise` 实际取自**全局**
   RNG；而 trunk 的 `TransformerEncoderLayer` 带 active `dropout=0.1`
   （`code/priors/models.py:79`、`:119-127`）。若 mask 走全局 generator，或估计前向在
   train 模式下运行，则该 step 的全局 RNG 消费量将依赖 `mask.sum()`，从而破坏 seed-42
   与既有 sealed 运行的 `(t, eps)` 对齐以及 resume 的位精确性。两项均为契约条件，
   任一违反归入第 6 条的 contract-failure。

   **无 sampler 首步语义改变（正面声明）。** 首个 reverse step `t=499` 没有 `prev_x0`，
   此时 `s=x_t`，因此 D2-AG 在 `t=499` 的行为与 D2-AE/D2-AF **完全一致**（三者同为
   `x_t` source，relation field 同样激活）。本方向**不引入任何 sampler 首步语义改变**，
   也不存在任何 timestep 区间的 relation 置零分支。该首步状态与训练中未被选中样本
   （`m_i=0`，同样为 `x_t` source）同分布，train/sample 对称因此在首步同样成立。

3. **被解锁的 direction-scoped 禁令（唯一一条）。** D2-AE0 §7（`EP:5109-5115`）写明
   "Sampling 的每个 500-step model call 只从当步 current `x_t` 构造同一 relation，
   不得使用 previous predicted clean `x0` 作为专有 condition"。用户已明确批准解锁**且
   仅解锁**该条 direction-scoped 禁令。解锁理由必须记录为：该条款的立法目的是阻止
   作者式的 train/sample **不对称**（sampling 独有、training 无法复现的 condition），
   而 D2-AG 以两侧对称的方式达到同一目的——训练侧用同一模型、同一 timestep 的
   detached `x0_hat` 复现 sampling 侧的 `prev_x0`，因此 "专有 condition" 的构成要件
   不成立。该解锁不得被推及 `EP:3993-3999`/`EP:5582-5583` 中的其余任何条目：
   clean-target、future-GT、stored per-frame BPS、stored relation 仍然全部禁止；
   scene conditioning、HSIPrior、Mixer、consistency、old-checkpoint init、guidance、
   SNR/timestep weighting、新 loss 和任何 sweep 仍然全部禁止。本次解锁一次性绑定
   D2-AG，不构成后续方向的先例。

4. **全部保持项。** 除第 2 条的 source 替换外，其余一切与 D2-AE/D2-AF 基线逐字节
   一致：current-state relation builder、100 immutable rest-object points 与 asset
   contract、surface transform、roles `(joint 24, joint 26, joint 0)`、temporal anchors
   `(0,5,10,15)`、routing `5/5/5/1`、`4→128→128` point encoder、mean/max pooling、
   role concat order、`768→512` projection、four temporal embeddings、LayerNorm、
   single scalar `tanh(alpha)`、alpha exact-zero initialization、full-trunk placement、
   20 tokens、4 condition tokens、global BPS、window frame 与 object reference、
   normalization、D2-X FK-foot routing、`[B,16,232]` clean output、2-frame history
   restoration、500-step clean-x0 diffusion、losses/reductions/weights、optimizer、LR、
   batch、split、budget、sampler 和 official evaluator 全部不变。D2-AF 的
   `sqrt(alpha_bar)` 衰减在本方向**不启用**，D2-AG field 不注册 `sqrt_alpha_bar`
   buffer。**relation 曝光率保持 100%，与 D2-AE/D2-AF 逐字节一致**：未被选中的样本取
   `s=x_t`（D2-AE 的现行行为），被选中的样本取 `s=\mathrm{sg}[\hat x_0]`，两者的
   relation field 都照常激活，不存在 relation 置零分支，每个 step 每个样本都有 relation
   写回。**唯一被操纵因子是可变锚点 `5/10/15` 在一半样本上的 geometry provenance**，
   relation 几何、曝光率、写回数学与训练信号规模均不变；`t=499` 的 sampler 行为亦与
   D2-AE/D2-AF 完全一致。`p` 不得改为可调值、schedule 或 per-anchor 概率。
   **零新增参数**：base `29,673,448`、relation `413,953`、total
   `30,087,401`，增量 `1.3950283%`，硬上限 `1.50%`；不得新增任何 learnable parameter。
   Seed-42 fresh initialization 的完整 non-persistent-buffer model-state SHA-256 必须
   精确为 `b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c`
   （本 amendment 已在 authority CPU 上以 `dim_model=512 / num_heads=16 /
   num_layers=8`、`torch.manual_seed(42)` 独立复算 D2-AE 与 D2-AF variant，两者均为
   该值，且 total parameters 均为 `30,087,401`），否则 GPU 前停止。

5. **Architecture 与 checkpoint provenance。** D2-AG 使用独立 architecture variant
   `d2ag_selfcond_relation_source`（常量 `HOI_ARCHITECTURE_D2AG`）与独立 checkpoint
   contract。released、author、base/D2-X、D2-AC、D2-AD、D2-AE、D2-AF schema 即使
   tensor shapes 相容也必须被 D2-AG loader fail-closed 拒绝；D2-AE/D2-AF loader 也
   必须反向拒绝 D2-AG。Checkpoint metadata 必须记录完整 self-conditioning contract
   （`p=0.5`、`x_t`-source no-grad 估计前向、未选中样本回落 `s=x_t`、
   `s[:, :2]=x_t[:, :2]`、`prev_x0=raw x0_hat`、无 relation 置零分支、无 `rho`）。
   HSIPrior 不接受该 variant、不共享参数或 storage；
   未来 Mixer 仍只接收 clean `[B,16,232]`。

6. **Authority CPU hard gate。** D2-AG 必须继承 D2-AE/D2-AF 全部 geometry、asset、
   SO(3)、invariance/sensitivity、point permutation、finite、dtype/device/batch、
   parameter/API、train/sample builder parity、checkpoint provenance、HSIPrior/Mixer
   independence、forbidden-source static scan、full authority suite 与 registry
   validation contracts，并新增：

   - training 与 sampling 的 relation source 构造必须共享同一实现路径，且在同一
     `(x_t, t, prev/x0_hat)` 输入下 field 输入张量逐元素 max abs `<=1e-6`；
   - 被选中样本的 no-grad 估计前向必须与 **D2-AE 式 `x_t`-source 前向**数值等价
     （同一权重、同一 `t`、同一 `x_t`，relation field 照常激活，max abs `<=1e-6`），
     且不产生梯度、不进入 autograd graph、不改变 optimizer state；
   - 未被选中样本（`m_i=0`）的完整前向必须与 D2-AE 式 `x_t`-source 前向逐元素等价
     （max abs `<=1e-6`）；实现中不得存在任何 relation 置零分支；
   - `s[:, :2]` 必须在两侧都精确等于当前 `x_t[:, :2]`（float32 exact where
     representable）；
   - `x0_hat` 必须 detached：对 `s` 的任意扰动不得回传到第一次前向的参数；
   - sampler `prev_x0` 首步必须为 `None`，且 `t=499` 的整个 model call 必须与 D2-AE 式
     `x_t`-source call 逐元素等价（max abs `<=1e-6`，field 照常激活，无置零）；此后
     每步 `prev_x0` 必须来自上一步的 raw `x0_hat`（`prepare_clean_x0` 之前，不做 SO(3)
     投影），sampler timestep trace 必须精确为 `499,498,...,0`；
   - Bernoulli mask 必须来自第 2 条注册的**独立** `torch.Generator`
     （seed `cfg.seed*1_000_003 + processed_windows + rank`），`p` 必须精确为 `0.5`，
     并进入既有 training audit digest；
   - 全局 RNG 消费量必须与 `mask.sum()` 无关：在同一 `(batch, processed_windows,
     rank)` 下改变 mask 内容后，`timesteps`/`noise` 必须逐字节不变；
   - 估计前向必须在 `inner.eval()` 下执行且以 `try/finally` 恢复原 `training` 标志，
     调用前后 `model.training` 必须一致；
   - `alpha=0`、shared D2-X trunk、`eval()` output max abs `<=1e-6`，并要求实际
     exact zero where representable；
   - initial alpha gradient 与 activated point-encoder/projection/temporal-embedding/
     relevant-trunk gradients 在 `t=0/249/499` 均 finite/nonzero；
   - exact parameter counts 与 initial-state hash 与第 4 条一致；
   - D2-AE/D2-AF/D2-AG resolved configs 除 identity、mechanism flag/variant 外
     exact equivalent；
   - 无 `rho`/schedule buffer、无 loss/SNR/timestep weighting、无 learned 或
     per-anchor 概率、无第二 writeback、无 stored relation 或 clean-target 读取。

   任一失败分类为 `selfcond-relation-source-contract-failure-stop`，不得开始任何 GPU
   workload。

7. **One-GPU functional smoke。** 注册 stem 为
   `p1-hoi-d2ag-gpu-functional-smoke[-rN]-s42-<actual-date>`，worker 固定
   infbagel-4gpu/node01、1×RTX 3090、real-data batch 8、timesteps `0/249/499`、
   seed 42 random initialization、无 optimizer、zero updates、zero checkpoint writes。
   除 D2-AE/D2-AF smoke 内容外，必须记录 Bernoulli mask 与其独立 generator seed、
   被选中子集大小、估计前向的 `eval()` 进入/恢复与前后 `model.training` 一致性、
   两次前向的 finite 性、`x0_hat` 与 `s` 的统计、`s[:, :2]` 与 `x_t[:, :2]` 的一致性、
   detach 检查、raw relation 与 writeback norm、initial alpha gradient、activated
   gradients、peak allocated/reserved/headroom 和 model hashes。Operational preflight
   failure 保留原目录并使用新 run id；scientific contract failure 立即停止。

8. **4-GPU full-micro-batch performance hard gate（必需，不得跳过）。** self-conditioning
   的 no-grad 估计前向会改变 per-step 计算量与峰值显存，因此按
   `docs/HOIPRIOR_ITERATION_WORKFLOW.md:68-70` 与 `AGENTS.md:48-51` 的
   "compute/data/communication/tensor-shape 或 memory 路径改变"条款，必须运行一次
   full-micro-batch benchmark，并与 formal run id 一对一绑定；本方向**不适用**
   "executed path unchanged" 豁免，也不得以 sealed 执行 profile 代替。只有 CPU gate
   与 smoke 通过后，才在
   clean、identical committed worker object 上运行
   `p1-hoi-d2ag-performance-benchmark[-rN]-s42-<actual-date>`：infbagel-4gpu/node01、
   4×RTX 3090、per-GPU batch 512、effective batch 2048、FP32 Adam、seed 42、
   random initialization、64 warm-up + 256 measured = 320 updates、`524,288` measured
   windows、CUDA synchronized timing、checkpoint load/write 均为零、benchmark weights
   为一次性且禁止复用。必须记录 loader wait、H2D、GPU relation build、no-grad 前向
   耗时与被选中子集大小的逐 rank 分布、forward、backward、optimizer、DDP、CPU/GPU
   utilization、contention、intermediate shapes、per-rank hashes 和 peak/headroom。
   本方向采用**通用先例形式**——sealed D2-X formal throughput `3243.0357134915853
   windows/s` 的 85%：

   \[
   throughput\ge 2756.580356467847\ {\rm windows/s},\qquad
   {\rm ETA}\le 6.20\ {\rm h}
   \]

   （`61,440,000 / 2756.580356467847 = 6.191245840747081 h <= 6.20 h`）。该两个数值与
   `code/train_hoi_prior.py:89-90` 的 `D2AE_MINIMUM_THROUGHPUT` /
   `D2AE_MAXIMUM_ETA_HOURS` **数值相同**，但在此注册为 D2-AG 自己的门槛，须以
   D2-AG 专属常量与 run-id 校验绑定，不得复用 D2-AE 的 gate 校验路径。**D2-AF 的
   `3179.689863044761 windows/s` / `5.367399778519349 h`（95%-of-immediate-predecessor
   形式，`code/train_hoi_prior.py:110-111`）不适用于本实验**，不得作为 D2-AG 的门槛
   或参照。同时 memory headroom 必须 `>=max(2 GiB, 10% device memory)`、
   losses/gradients finite、无 CPU dynamic geometry、无外部
   contention。Benchmark 必须一对一绑定 intended formal run id 与 source hashes。
   Completed scientific benchmark 未过即分类
   `selfcond-relation-source-performance-negative-stop`，**不得**通过
   batch/worker/thread/architecture/point/width/role/routing/`p` 或任何 sweep 重试。
   本方向不继承 2026-07-29 D2-AF0 的一次性 performance waiver（`EP:6137-6209`），
   该 waiver 是 run-id-bound 的一次性用户覆盖，不得推广。

9. **唯一 formal training。** 只有 contract、smoke 和 performance 全通过，才运行
   `p1-hoi-d2ag-selfcond-relation-source-s42-20260731`（若实际启动日期晚于本预注册
   日期，则只按 `<actual-date>` 更新日期后缀，科学配置不变）。固定 seed 42、split
   `experiments/splits/omomo_hoi_train_validation_seed42.json`（SHA-256
   `019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e`）、
   infbagel-4gpu/node01、4×RTX 3090、batch 512/GPU、effective 2048、accumulation 1、
   61,440,000 windows、983,040,000 frames、30,000 updates、FP32 Adam、LR `1e-4`、
   betas `(0.9,0.999)`、weight decay 0、no warmup/scheduler/AMP/clipping/EMA。
   FK/object-surface/velocity/terminal-goal weights 继续为
   `0.3569973401779424 / 0.4772322188400037 / 0.1 / 1.0`，D2-X FK-foot routing
   enabled，全部新 loss disabled。必须从 seed-42 random initialization 开始；
   init/weight-init/resume 均为空，released/author/D2-X/D2-AC/D2-AD/D2-AE/D2-AF/
   任何 prior/EMA/consistency checkpoint load count 全为零。完整运行 fixed budget，
   只使用 online/final-online；不得选择 cadence/best-validation checkpoint。稳定区间
   和至少一个 resumable checkpoint 通过后，按 worker-owned persistent-session 规则
   报告 throughput/ETA/hash 并停止主动轮询。`formal_runs_maximum=1`。

10. **Fixed five-path internal causal diagnostic。** Formal 完成后只加载 fixed
    final-online，复用 sealed D2-O internal-validation cohort：64 sequences × 3
    windows、phase offsets `(14,56,98)`、selection SHA-256
    `1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`、batch 恰好
    `8`。五条 paired 500-step rollout 固定为：

    - **source substitution**：在同一训练好的模型内把可变锚点的 relation source 由
      `x0_hat` 换回当步 `x_t`（self-conditioning 路径保持开启，只改来源）。该扰动
      **就是** source substitution，**不是**对 `prev_x0` 做时间打乱、延迟或置换；
    - **high-t restriction**（本方向的判别性关键 gate）：self-conditioned source 只在
      `t<250` 生效；`t>=250` 时**回落到 `x_t` source**（不是置零，与首步 `t=499` 的
      注册行为一致）。它与 gate 1（全 schedule 都用 `x_t`）构成对照，用于分辨收益是否
      来自高 `t` 段的 source provenance；
    - **counterfactual object displacement**：只对 `s` 的 object translation 通道施加
      固定平移 `\delta`，其余通道与全部 condition 不变，手必须表现出方向性跟随。
      `\delta` 注册为 `0.10 m`，在 **metric space、denormalize 之后**施加。依据：该量级
      与实测生成手-物相对偏移（约 `10 cm`；GT×GT 为 `1.70 cm`）同阶，且远高于 `5 cm`
      contact 阈值，因此方向性跟随可判别；
    - **temporal routing permutation**：沿用 D2-AE `k\leftarrow(k+2)\bmod 4`；
    - **role swap**：沿用 D2-AE，projection 前只交换 left/right pooled blocks。

    除被操纵因子外，五路共享 initial latent、每一步 posterior noise、condition、
    history、ordering 和 history restoration；seed 42、10,000 paired **sequence**
    bootstrap、gate 判据统一为 `CI_lower>0`。主指标为 GT-contact-frame mean
    hand-object distance（方向相反）与 direct-hand union 5-cm contact F1。另注册
    **仅报告、不参与判定**的量：per-sequence（**不得跨序列拼接**）within-sequence
    contact run 结构与 coverage；以及 field 锚定 joints `24/26` 与 official FK palms
    `22/23` 的对照检查。结果保存为 `internal_status`；**internal 无论正负，下面唯一
    一次 fixed native 都必须执行**，不得以 internal cohort 过滤 official result，也不得
    用 internal 结果选择 checkpoint。

11. **Fixed native evaluation 与 gates。** 协议严格沿用 D2-AE/D2-AF：official 438
    sequences × 3 windows、500-step unguided production diffusion、fixed final-online、
    CFG/guidance/scene/dynamic perception/consistency 全部 off、paired sequence unit、
    seed 42、10,000 bootstrap、sealed D2-X 181-sequence penetration finite mask、
    official evaluator/hash/helper/threshold 不变。Sealed D2-X checkpoint/aggregate/
    per-sequence 继续为
    `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51` /
    `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
    `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`，released
    aggregate 为
    `76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6`，
    penetration mask 为
    `2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec`；全部按引用
    复用，**禁止重算或重新生成**。本方向不设 D2-AF 式的 predecessor-specific
    single-factor repair gate，只使用标准 D2-X native gates：

    - **transfer：** AG−X contact F1 与 contact recall 的 paired CI lower 均 `>0`；
      contact F1 点估计 `>=0.6598838781`；released–D2-X contact-F1 gap closure
      `>=0.25`；
    - **protection：** AG/X end-object、Txy、FS、Pbody、hand penetration、MPJPE、
      Troot、Tobj、Oobj 九项 mean-ratio CI upper 均 `<=1.10`；contact precision
      difference CI lower `>=-0.02`；181-sequence penetration finite mask 契约必须
      exact；
    - **released 95% effectiveness floor：** 八项 lower-is-better 指标
      `target/released<=1/0.95`，三项 contact 指标 `target/released>=0.95`，
      seed-42 point estimate，无 CI。

    FID / Matching Score / R-Precision / Diversity / timing 若 evaluator 产出则必须
    保留并报告，但不参与选择；contact coverage 报告但不作独立单调选择指标。所有条件
    均为 AND，不允许 composite、best-of、metric 替换或阈值修改。

12. **Decision、lifecycle 与 stop rule。** Post-training 同时保存 `internal_status`
    和 `native_status`，两者无论正负都必须完整报告。单线终态顺序为：

    - `selfcond-relation-source-contract-failure-stop`；
    - `selfcond-relation-source-performance-negative-stop`；
    - `selfcond-relation-source-transfer-negative-stop`：native transfer 三项
      （AG−X contact F1 CI lower、AG−X recall CI lower、contact F1 点估计与
      released gap closure）中任一失败；
    - `selfcond-relation-source-conflict-negative-stop`：native transfer 全过，但九项
      protection ratio、contact precision difference、penetration finite mask 或
      released-95% effectiveness floor 中任一失败；
    - `selfcond-relation-source-native-positive-mechanism-unverified-stop`：native
      transfer、protection 与 released-95% **全部**通过，但任一 internal gate 失败；
    - `selfcond-relation-source-mechanism-negative-stop`：internal gate 失败**且**
      native 亦未全部通过（即 native 已落入上面某一 negative 层）时，除记录该 native
      headline 外一并记录本 internal 标签；
    - 全部通过：`selfcond-relation-source-positive-candidate-stop`。

    与 D2-AF 小节同一原则：`native_status` 是 headline，positive internal 不能救回
    negative native；上表按顺序求值并在第一处失败停止，`internal_status` 与
    `native_status` 始终各自完整记录，internal 结果不改变 native 是否执行。

    只有最后一类可将 fixed final-online 标为 selectable autonomous-diffusion HOIPrior
    candidate，且即使如此也不授权 consistency。Lifecycle stems 固定为
    `p1-hoi-d2ag-{cpu-contract|gpu-functional-smoke|performance-benchmark|
    selfcond-relation-source|selfcond-relation-source-internal|native-eval|
    completion}[-rN]-s42-<actual-date>`；config default `run_id=null`，实际 date 在
    workload 启动时现场生成，失败目录保留且 retry 使用新 id，已存在的 run id 或
    manifest 一律不得复用或覆盖。所有 resolved config、same-context manifest/preflight、
    logs/profile、failure trees、checkpoints/RNG、internal five paths/paired noise、
    native raw/optional outputs、run-local registry、hardware/data/dependency/evaluator
    hashes 必须由 worker 发起 non-destructive recovery，双端统一 `sha256_path` 与
    checksum dry-run；不得 `--delete`。

    **允许改动的文件范围（本预注册锁定，用户已批准含扩展部分）：**
    `code/priors/models.py`、`code/priors/diffusion.py`、
    `code/priors/sparse_relation.py`、`code/train_hoi_prior.py`、
    `code/config/config_train_hoi_prior_d2ag.yaml`（新建）、D2-AG 专属
    smoke/benchmark/internal/native-evaluation 工具链及其 tests、
    `docs/EXPERIMENT_PLAN.md`、`experiments/registry.jsonl` 和一处简短文档说明。
    扩展理由记录为：已批准机制无法在不触及
    `code/train_hoi_prior.py::_forward_losses`（训练侧 Bernoulli 与 no-grad 估计前向的
    唯一调用点）、variant 判定、`_model_config`、`_resume_contract`、run-id 校验与
    config 解析的前提下实现，且 D2-AE/D2-AF 的
    diagnostic/smoke/benchmark/internal/native 工具链均为 variant-bound、无法直接复用。
    此范围之外的改动仍须停下并另行请示，不得由实现者自行扩大。

    本方向禁止：第二次 formal run、longer budget、D2-AG1、`p` 或任何
    LR/batch/budget/point/width/depth/role/placement/anchor/threshold/num-workers/
    threads sweep、checkpoint selection、best-of-N 不对称、D2-X/D2-AC/D2-AD/D2-AE/
    D2-AF/released/author checkpoint 的 init/resume/retrain/selection、新 loss、
    SNR/timestep loss weighting、gradient projection、rollout exposure、CFG/guidance、
    consistency、scene 资产或 occupancy、clean-target/future-GT/stored-relation/
    stored per-frame BPS 来源、以及在本实验完成或停止前启动 HSIPrior 或 Mixer。
    若 pretraining gate 失败，formal budget 不消耗但本方向结束；若 formal 启动则只
    允许完整运行该一次预算。最终必须写 compact result、
    `docs/phase_summaries/PHASE_1B_D2AG.md` 和 append-only completion record。

#### 2026-07-31 Phase 1B D2-AG0 one-time user-authorized performance waiver（plan-only）

D2-AG 的 4-GPU full-micro-batch performance hard gate（`EP:7075-7111`）已在
`p1-hoi-d2ag-performance-benchmark-r2-s42-20260731` 上完成并**失败**。Worker 为
infbagel-4gpu/node01、4×RTX 3090，commit
`ada2d84223ecbf76f5ed9bbd313f5ac6dfce2cbb`、`worktree_clean=true`，64 warm-up + 256
measured = 320 updates、`524,288` measured windows、wall `241.36226116283797 s`。实测
end-to-end throughput 为 `2172.2037135137825 windows/s`，低于注册门槛
`2756.580356467847 windows/s`（缺口 `-21.199332774135327%`）；
`throughput_fraction_of_sealed_d2x = 0.6698056714198497`（sealed D2-X formal
`3243.0357134915853`）；ETA `7.8568444388944645 h` 超过注册上限 `6.20 h`。Status 为
`failed`，classification 为 `selfcond-relation-source-performance-negative-stop`，
`formal_training_authorized=false`、`sweep_authorized_on_failure=false`、
`performance_waiver=null`。Failed checks 恰为 `classification`、`eta`、
`formal_authorized`、`status`、`throughput` 五项，与 D2-AF0 waiver 的五项同构；
**其余全部科学与执行契约均通过**：`all_rank_contract_pass`、`memory_headroom_pass`、
`contention_pass`、`losses_finite`、`gradients_finite`、
`selfcond_estimate_forward_measured`、`selfcond_graph_pass_instrumentation_pass`
均为 true，`external_contention_before` 与 `external_contention_after` 均为空。
Benchmark 的 `formal_source_contract` SHA-256 为
`55ff307986e1c1e0ff94286b1fadec681b7cb3fe478da7b8ce5da670de84ee88`
（`tracked_file_count=92`），与 authority checkout 在 HEAD `ada2d84` 上重算的值逐字段
一致。同日先行的 `p1-hoi-d2ag-performance-benchmark-s42-20260731` 目录为 operational
preflight 中止（只含 `resolved_hydra_config.yaml`，无 summary），按 `EP:7073` 保留原
目录并以新 run id `-r2-` 重跑，不得复用或覆盖。

在上述失败被**完整保留并报告**后，用户明确接受已测完整预算 ETA
`7.8568444388944645 h`，并授权直接运行 `EP:7113-7128` 所注册的 D2-AG 唯一 formal
budget。该新授权**覆盖**原先"performance 失败即不训练"的执行 stop rule，但**不回写
历史、不把 benchmark 改成 passed，也不改变其
`selfcond-relation-source-performance-negative-stop` 分类**。

1. **ETA 与根因解释锁定。** ETA 只由固定预算和实测端到端吞吐外推，不使用任何模型化
   或乐观假设：

   \[
   61{,}440{,}000 / 2{,}172.2037135137825 / 3600
   = 7.8568444388944645\ {\rm h}.
   \]

   即每个 2048-window update 约 `0.9428213326673357 s`，30,000 updates 与
   `61,440,000` windows 的注册预算完全一致（`30,000 × 2,048 / 2,172.2037135137825
   / 3600` 复算得同一值）。根因与 2026-07-29 D2-AF0 属**同一 pathology**，且**不是
   self-conditioning 机制本身**：per-rank inclusive `backward`（含 DDP critical-path
   wait）为 rank0 `203.87` / rank1 `120.11` / rank2 `105.08` / rank3 `206.66 s`，
   跨度约 2 倍（`1.966598891801045`）；而被操纵机制逐 rank 均匀——
   `estimate_trunk_forward` 为 `6.53–6.94 s`（跨 rank 均值 `6.744397082805633 s`），
   仅占 wall 的 `2.794304731117614%`；`forward_and_loss` 为 `22.87–24.21 s`，同样
   均匀。跨 rank 均值 timing 汇总（秒）为 backward `158.932`、loader_wait `50.161`、
   forward_and_loss `23.418`、estimate_trunk_forward `6.744`、gradient_validation
   `4.978`、optimizer `3.186`、gpu_relation_module `1.903`。

   **关键算术（本 waiver 的判别性依据）。** 把 estimate forward 的成本完全去掉，wall
   约为 `234.61786408003235 s`，throughput 约为 `2234.6465477204924 windows/s`，
   **仍远低于 `2756.580356467847` 门槛**。即该机制只解释
   `584.3766429540647 windows/s` 总缺口中的约 `10.685374742401925%`，其余约
   `89.31%` 来自 baseline harness 的 rank skew；换言之**一个零成本的 D2-AG 同样会
   失败此 benchmark**，该 gate 在当前 harness 下并不能判别本方向的机制开销。达到门槛
   需要把 wall 压到 `190.19507222775036 s`，远超去掉整个机制所能获得的余量。先例
   佐证：D2-AF benchmark 实测 `2089.8443630127094 windows/s`（fraction of sealed
   D2-AE `0.6243854685126333`）被 waive 后，其 formal continuation 实测
   `3232.575359023025 windows/s`，说明 benchmark harness 的 rank skew 并未在 formal
   执行中复现。该佐证只用于解释缺口归因，**不构成对 D2-AG formal 吞吐的承诺或预测**。

2. **不做 post-hoc execution sweep。** 当前没有已证实能够消除上述 rank skew 的单一
   安全 toggle，而第 1 条的算术表明即使把机制成本归零也无法通过该门槛，因此任何
   execution 调参都不可能把本 benchmark 变成 passed。据此本 waiver 选择用户授权的
   "直接训练"分支，**不改 batch、micro-batch、`num_workers`、CPU affinity、
   prefetch/pinning、线程或 I/O 布局、architecture、point/width/role/routing、`p` 或
   budget**，也不改模型数学、训练循环计算与 instrumentation。第二次 benchmark、
   `-r3` 重跑与上述任何 sweep 继续禁止；benchmark 中的
   `sweep_authorized_on_failure` 保持 `false`，不得被 waiver 改写。

3. **Waiver 的精确范围（一次性、run-id 绑定、不可继承）。** 本 waiver 一次性绑定且
   仅绑定两个 identity：被 waive 的 benchmark
   `p1-hoi-d2ag-performance-benchmark-r2-s42-20260731`，以及唯一 formal run
   `p1-hoi-d2ag-selfcond-relation-source-s42-20260731`。只允许启动一次该 formal
   identity；启动时仍须满足 actual-date 规则，且该目录必须此前不存在。原 benchmark
   不重跑，其 320-update sacrificial weights 仍不可复用
   （`benchmark_weights_reusable=false`）。不允许第二次 formal、resume 旧方向、
   checkpoint selection、D2-AG1、longer budget、consistency、HSIPrior 或 Mixer。
   **本 waiver 不构成后续方向的先例**：与 `EP:7110-7111` 对 D2-AF0 waiver 所作的
   声明同构，它不得被 D2-AH 或任何后续实验继承、援引或推广，后续方向若再次未过
   performance gate，必须重新取得用户对该次具体实验的明确授权。

4. **Fail-closed implementation 与状态表述。** Formal trainer 不得简单删除
   performance 检查或伪造 passing summary。它必须同时绑定：原 failed benchmark JSON
   的 absolute path、SHA-256、run id、failed status/classification、实测
   throughput/ETA 及全部 non-speed contracts；一份 tracked、immutable、SHA-bound
   waiver JSON；waiver 中的唯一 formal run id、用户授权事实、benchmark SHA、原/目标
   Git commit、exact transition diff SHA、允许改变的 governance/validator/config/test
   路径与目标 formal source-tree contract；以及 `formal_runs_maximum=1`、benchmark
   retry/sweep=false、training conditions unchanged=true、random initialization=true。

   原 benchmark 的 throughput/ETA checks 必须在 formal lifecycle 中继续保存为
   `false`；新状态只能表示为 `failed-waived` / `user-authorized-performance-waiver`，
   **不得表示为 `performance-gate-passed`**（与 `EP:6187-6189` 同一措辞与同一约束）。
   Benchmark 中 memory、finite loss/gradient、GPU-only relation、optimizer/checkpoint
   I/O、four-rank identity、contention、selfcond estimate/graph instrumentation 与
   schedule 等任一 non-speed contract 不通过时，waiver 无效并停止。

5. **Source transition、artifact recovery 与重新验证。** 为接受 waiver 所需的 source
   修改只允许涉及 D2-AG performance waiver validator、base/D2-AG config binding 及
   对应 tests/documentation；不得修改 models、diffusion schedule、relation
   builder/encoder/routing、self-conditioning 机制、loss、optimizer 或 training-loop
   数学。由于原 CPU gate / smoke / benchmark 是在 formal source contract
   `55ff307986e1c1e0ff94286b1fadec681b7cb3fe478da7b8ce5da670de84ee88`（commit
   `ada2d84`）上完成，waiver 必须以 source/target commit 和 exact Git diff hash 显式
   授权这次 validator-only transition，而不是重写旧 artifact。目标 commit 上必须重新
   通过完整 authority suite、registry validation、static source/diff audit 和
   resolved-config fail-closed 测试；**不重跑 scientific performance benchmark**。
   在 formal 启动前，benchmark run 目录必须由 worker 发起 non-destructive recovery
   （无 `--delete`、checksum dry-run、双端统一 `sha256_path`）落到 authority staging，
   并把 summary/resolved-config/rank-metrics SHA-256 记入 completion 记录。

6. **科学契约与 formal 后评测完全不变。** 除上述 validator-only transition 外，
   `EP:7113-7128` 注册的科学条件逐条不动：seed 42、`experiments/splits/
   omomo_hoi_train_validation_seed42.json`、seed-42 random initialization（全部
   released/author/D2-X/D2-AC/D2-AD/D2-AE/D2-AF/prior/EMA/consistency checkpoint load
   count 为零）、infbagel-4gpu/node01 4×RTX 3090、batch 512/GPU、effective batch
   2048、gradient accumulation 1、`61,440,000` windows、`983,040,000` frames、30,000
   updates、FP32 Adam、LR `1e-4`、betas `(0.9,0.999)`、weight decay 0、无
   warmup/scheduler/AMP/clipping/EMA、loss weights
   `0.3569973401779424 / 0.4772322188400037 / 0.1 / 1.0`、D2-X FK-foot routing
   enabled、`p=0.5` 与全部 self-conditioning contract 均不变。训练完成后仍只使用
   fixed final-online、**不得做 checkpoint selection**，依次执行 `EP:7130-7159` 的
   five-path internal causal diagnostic 与 `EP:7161-7190` 的一次 fixed native
   evaluation；internal 五 gate、native transfer/protection/released-95%
   effectiveness floor、统计协议、sealed controls、`EP:7192-7208` 的终态顺序与
   selectability 条件全部不变。Compact result 与 phase summary 必须同时报告原
   performance failure、本用户 waiver、实际 formal wall/throughput 及最终结果。

下一步仅允许提交本 plan-only waiver 与 append-only registry hypothesis；随后实现上述
最小 hash-bound validator/config/tests，创建并提交 immutable waiver contract，通过
authority verification 后由 worker fast-forward 相同 clean Git object 并启动唯一
formal run。

#### 2026-08-01 Phase 1B D2-AG0 评估工具链 provenance 绑定改为语义契约（用户批准）

D2-AG formal run `p1-hoi-d2ag-selfcond-relation-source-s42-20260731` 已完整跑满注册
预算并 `completed`（61,440,000 windows / 30,000 updates / wall `17988.12 s` /
`3415.59 windows/s` / `amp_overflow_skips=0` / 全部 20 个 validation 点 finite）。其
internal diagnostic 与 native evaluation 原被两个**无法满足**的门禁阻塞：

1. `tools/run_hoi_d2ag_internal.py::FORMAL_LINEAGE_SEALED` 是一张 9 键全 `None` 的
   pre-committed hash 表，`sealed_lineage_contract()` 在 `--resolve-only` 分支**之前**
   抛错，必须由人工在 formal run 结束后把哈希填入源码并另开一次治理提交才能解锁。
2. 两个 runner 均 `required=True` 要求 `--resume-contract`，但直通训练不产生
   `resume_contract.json`（`training_state.json` 记录 `resume_checkpoint: null`）；该
   工件类只存在于被中断后 resume 的运行。

根因是 D2-AG 照搬了 D2-AF 加固提交 `3d4ff1e` 的外壳，而 D2-AF 是一次**被 resume** 的
运行；D2-AG 与 D2-AE 同为直通训练。故按 `EP:7226-7237` 已锁定的可改文件范围（D2-AG
专属 internal/native 工具链及其 tests 与一处简短文档说明），将 provenance 绑定改为
`tools/run_hoi_d2ae_internal.py:100-197` 已验证的语义契约形式：对目标 checkpoint 按
CLI 传入值做**一次**字节校验并核对固定最终 basename，其余身份一律从 checkpoint 内部
读取绑定——`run_id`、`seed=42`、`processed_windows=61,440,000`、
`processed_frames=983,040,000`、`optimizer_updates=30,000`、`world_size=4`、
`effective_batch_size=2048`、`architecture_variant`、`data_contract_sha256`、
`split_sha256`、`weight_initialization`（from-random）与
`selfcond_relation_source_contract`。

该形式**严于**原表：pre-committed 表只能证明"字节等于某个人手打的常量"，语义契约证明
"该文件就是此 run 跑满 61.44M windows 的最终 checkpoint"，且不需要事后治理提交。

同时删除的仅为无科学内容的空转校验：同一进程内对同一文件的重复哈希、常量与自身比较
（`cadence_main_checkpoints`/`cadence_rng_sidecars` 两侧同为 `len(FORMAL_CADENCE_WINDOWS)`，
从不统计磁盘文件）、写死的 `"asset_hashes_exact": True`、仅校验十六进制格式的
`terminal_model_state_sha256`，以及未被引用的 `EXPECTED_INITIAL_MODEL_STATE_SHA256`。

**科学协议与全部实质门禁不变**：fixed final-online checkpoint 选择规则、sampler、
metrics、uncertainty、cohort 定义、failure rules 一律未动；目标 checkpoint 字节校验
（移入 `checkpoint_contract()`）、`metrics.checkpoint_hashes` 的
`processed_windows==61,440,000` ∧ sha ∧ basename 行、`initial_model_state_sha256`
（唯一证明 from-random 初始化的门禁）、waiver↔benchmark 绑定、11 件产物 closure 校验、
control/baseline/evaluator 静态资产与 worker 数据资产哈希、cohort selection、GT-contact
mask、sampler schedule 哈希全部保留。外部与传输输入的哈希按 `AGENTS.md` lean profile
保留，本次精简只针对"未改动、本地产出或已由 manifest 标识"的重复校验。

验证：新增 7 项否定测试（外来 `run_id`、中间 cadence checkpoint、非 D2-AG
`architecture_variant`、被篡改的 selfcond contract、非随机初始化、错误 shape/budget、
hash 或 basename 不匹配）；并以 mutation testing 证伪其非空洞——将 `run_id` 与
`processed_windows` 判定分别改为恒真后，对应否定测试确实失败。D2-AG 三个测试文件共
121 tests OK，`tools/experiment.py validate` 通过（234 registry records）。

