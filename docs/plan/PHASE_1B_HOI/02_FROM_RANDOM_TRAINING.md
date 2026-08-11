# Phase 1B-02：从随机初始化的强训练谱系（D2-T → D2-AA）

本文件于 2026-08-10 从 `docs/EXPERIMENT_PLAN.md` 第 2028-3545 行原样切出（逐字节复制，未改写、未重排、未修正任何笔误）。
导航：[总览](../OVERVIEW.md) · [Phase 1B 索引](README.md)

#### 2026-07-21 Phase 1B D2-T author-DDPM update-rule parity screen 预注册

作者自主训练的 InfBaGel diffusion epoch500 已在 official 438-sequence HOI evaluator 上形成有效
HOI 能力，而 D2-H 已证明当前 HOIPrior 存在 model-parent reverse-state exposure 放大。两项证据共同
否定“一致性蒸馏是基础能力的唯一来源”，但尚未区分当前 HOIPrior 的失败主要来自 model/condition/
loss 机制，还是此前与作者 DDPM 明显不同的 update-rule contract。D2-T0 因而只把训练更新规则对齐
到作者 DDPM；保持 HOIPrior 的独立 232-D prior、network、conditions、diffusion、loss、data 与 sampler
不变。它不是作者网络移植，不加载 released 或自主训练作者 checkpoint，也不授权 consistency stage。

1. **唯一 manipulated factor。** control 为 sealed R1024 HOIPrior；intervention 固定为
   `p1-hoi-d2t-author-update-rule-s42-20260721`、subphase `1B-D2-T0`、seed 42。D2-T 将一组不可拆分的
   optimizer/update-rule contract 作为唯一 factor：4×RTX 3090、per-GPU batch 512、gradient
   accumulation 1、effective batch 2048、Adam (`betas=(0.9,0.999)`、weight decay 0)、constant
   LR `1e-4`、FP32、无 warmup/cosine、无 global gradient clipping、无 EMA selection，只保存/评估
   online model。该组合同来自作者 DDPM 的实际训练更新规则；不得只挑其中观察后有利的子项。
2. **fixed controls。** 保持 `WindowStateCodec` 与 232-D representation、16-frame window、2-frame
   immutable history、scene-free independent HOIPrior、512-wide/16-head/8-layer network、现有 timestep/
   text/BPS/goal/progress/history condition routing、500-step x0 DDPM、训练 split、seed 42、random
   initialization、FK/surface/velocity/terminal-goal weights `50/50/0.1/1`、normalization、dataset sampler
   与 processed-window budget `6,144,000` 不变；对应 3,000 optimizer updates。validation 固定
   32,768 windows，在 3,072,000 与 6,144,000 processed windows 评估；checkpoint 同步保留 midpoint
   与 terminal online weights。禁止 CFG、dynamic perception、contact/bump guidance、任何 sampler
   intervention、released/author checkpoint、旧 HOIPrior initialization/resume、CM/teacher/student/target。
3. **execution ownership 与 lifecycle。** authority/integration host `10.184.17.253` 只做开发、CPU/static
   tests、commit、manifest recovery 与未来 HSI 工作，明确禁止运行任何 HOIPrior CUDA workload。
   D2-T 只能在 `infbagel-4gpu`（worker `10.181.9.214`）执行；worker 必须主动 fetch 精确 commit，使用
   `/home/yujinlun/data/envs/infbagel/bin/python`，显式设置 `INFBAGEL_PYTHON` 与
   `INFBAGEL_WORKER_EXPERT=hoi`。先归档 fully resolved Hydra config，再由 clean worker checkout 运行
   `tools/experiment.py start`，核验四张 RTX 3090、无 contention、commit/config/assets hashes 与 worker
   role 后，才可在 worker-owned persistent session 启动。旧 run id、脏 checkout、authority CUDA、
   非四卡 topology 或缺少 lifecycle manifest 均为 contract failure；不得静默降 batch、改 accumulation
   或在当前服务器代跑。
4. **mechanism gate。** 使用与 sealed R1024 相同的 author-native HOI evaluation protocol、official
   438 sequences、500-step unguided diffusion loop、online weights、sequence-paired 10,000-replicate
   bootstrap（seed 42）。除 finite、checkpoint/config/hash/lifecycle、official-438、paired-unit 和禁止
   checkpoint-load contracts 全部通过外，D2-T 相对 sealed R1024 必须同时满足：MPJPE、end-object
   error、xy error、object-translation error 的 paired improvement 95% CI 下界均 `>0`；contact F1
   difference CI 下界 `>=-0.02`；foot-sliding D2-T/R1024 paired mean-ratio CI 上界 `<=1.10`。
   任一失败分类 `author-update-rule-negative-stop`，不选择 checkpoint、不进入 CM。
5. **effective-diffusion gate。** mechanism gate 通过后，再相对 released Phase-0 author-native baseline
   检查 MPJPE ratio `<=1.30`、end-object ratio `<=2.00`、xy ratio `<=1.50`、object-translation ratio
   `<=1.50`、foot-sliding ratio `<=1.10` 且 contact F1 `>=0.60`。mechanism pass 但该绝对 gate 失败，
   分类 `positive-but-not-effective-diffusion-stop`；两 gate 全通过才可称为 usable diffusion HOIPrior。
   单 training seed 只支持该 preregistered mechanism screen，不得外推训练方差或 main-table 稳健性。
6. **实现、测试、artifact 与停止规则。** 新增独立 D2-T config，并对 optimizer class、constant LR、
   FP32、no clipping、no EMA、online validation/checkpoint、3,000 updates、random-only initialization、
   worker host/role 和 model/data/diffusion/loss source invariance 加 fail-closed guards/tests。checkpoint
   必须记录 optimizer/scheduler/clipping/AMP/primary-weight metadata，且 `ema_models={}`、不写 legacy
   `ema_model`。artifact 至少包括 manifest、resolved config、preflight/hardware snapshot、training log、
   metrics/state、midpoint/terminal checkpoint 与 hashes、native evaluation aggregate/per-sequence records、
   bootstrap/gate summary 和 append-only registry completion。任何确定实现 defect、OOM、NaN、hash/
   lifecycle failure 或 gate negative 均登记全部负结果并停止；不得观察结果后改变 update-rule factor、
   训练预算、checkpoint variant、evaluator 或阈值。只有 effective-diffusion gate 通过后，才可另行提出
   并预注册从自主训练 HOIPrior diffusion checkpoint 开始的 consistency stage。

#### 2026-07-21 D2-T display-only Xorg idle preflight clarification

D2-T implementation commit `baa01bf73b6e151693f949fe943505bf8eda3410` 已在 authority 与
`infbagel-4gpu` clean checkout 通过 207 项测试（worker 按 HOI-only contract 跳过 2 项真实 LINGO
资产测试），且尚未执行 `tools/experiment.py start`、GPU smoke 或 training。首次与第二次 immutable
preflight 分别为 `preflight.json` 和 `preflight_r1.json`，SHA-256
`b0f35ae89511eb2dc1e88d04c56fd244bab8a75a9078fa8820bf1bebdd5b8047` 与
`9626a12af3c08f7def479e75dec51b969108517ef30a0101e61242909f320527`；二者除
`four_gpu_idle` 外所有 checks 均通过。GPU 0 只有 Xorg graphics PIDs `2552/3224`，无 CUDA compute
process，P8、约 96--100 MiB used、memory utilization 0%；14 秒 71 samples 的 GPU utilization
min/avg/max 为 `0/0/6%`，但 `nvidia-smi` 单点值稳定显示 `1%`。因此失败来自 preflight 将瞬时
utilization 写死为精确 0%，不是训练 contention。

用户明确确认该长期存在的 Xorg 占用可忽略。仅作 operational clarification：四卡 idle 仍要求
4×RTX 3090、CUDA compute-process 列表为空、每卡 memory used `<=128 MiB`，并把瞬时 GPU
utilization 上限从 `0%` 调整为 `1%`，用于容纳 display-only Xorg driver floor。不得放宽到用户提及
的 10 GiB 训练峰值，不得容纳任何 compute process、P2/P0 训练状态或 `>128 MiB` 的外部 allocation。
preflight 输出必须记录该 tolerance 和逐卡 checks；新增 tests 后提交新的 exact commit，worker 主动
fast-forward，再写入不覆盖旧失败证据的 `preflight_r2.json`。只有 r2 全部通过，才可创建原 D2-T
run id 的 lifecycle manifest 并启动原预注册 workload；scientific config、run id、seed、budget、gates
和所有禁止 checkpoint-load 项均不变。

#### 2026-07-21 D2-T author-native evaluation lifecycle amendment

D2-T training 已在 commit `6d57d34e27aeb748039896052a861fdda386873e` 完整结束：returncode
数值 0、3,000 updates、6,144,000 windows、final online checkpoint SHA-256
`1543af304acf76f385dbd3656a1ca82ea25dcd504ee120f7f63e821d71483647`。training manifest 已完成，
因此不得重新打开或复用其 run id 来承载 GPU evaluation。原预注册 scientific gate 不变，新增唯一
evaluation lifecycle run id `p1-hoi-d2t-native-eval-s42-20260721`，subphase `1B-D2-T0-eval`。

评估只加载 D2-T final checkpoint 的 `model`/online weights，不加载 optimizer、scheduler、RNG、EMA、
released 或 author checkpoint。使用现有 `code/test_infbagel_hoi.py` official 438 author-native protocol、
3 windows/sample、500-step unguided diffusion、seed 42、无 CFG/dynamic perception/guidance、无 CHOIS
export、无 FID/R-precision gate。evaluator static hashes 锁定为 test script
`22886f8797ceb04a892487393dea9f80e19877bc02dd7a6f39127e7319119524`、metrics
`445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547`、config
`89c702d96b98289924225c4b163d3b29eb22efe27c50ac799ddd0c71c515aa73`。

paired control 不重复生成，复用 D2-N 已完成且同 evaluator/seed/order/protocol 的 sealed R1024 online
records：aggregate SHA-256 `d95d3090455e763159a4cac793301f9f4744837bf60b4ed21eaef4a141c9ad2b`、
per-sequence SHA-256 `11c11fcd90c0ce2e67d705bb64c3a78bbe2b0e9f84aff7fcb57cab25087e2a1f`，
438 sequences / 3 windows，checkpoint SHA-256
`ded9a12d4e85179c37e2457475649ccc614ef364b97eaebade0629b2c11d4ed8`。任何 hash、sequence identity、
metric set、normalization、online-weight 或 official-count mismatch 均分类
`author-update-rule-contract-failure-stop`。

mechanism gate 逐 sequence、10,000 bootstrap、seed 42：control-minus-D2-T 的 MPJPE、end-object、xy、
object-translation improvement CI 下界均 `>0`；D2-T-minus-control contact F1 CI 下界 `>=-0.02`；
D2-T/control foot-sliding ratio CI 上界 `<=1.10`。失败分类 `author-update-rule-negative-stop`。
若通过，再相对 Phase-0 released aggregate（JSON SHA-256
`76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6`）检查原锁定 absolute ratios 与
contact F1 `>=0.60`；absolute gate 失败分类 `positive-but-not-effective-diffusion-stop`，两 gate 均通过
分类 `effective-diffusion-hoi-prior-stop`。评估完成后 finish/register/recover 全部 artifacts 并停止；
无论结果如何都不在本 amendment 启动 CM、第二次训练、checkpoint selection 或 sampler intervention。

#### 2026-07-21 D2-T native evaluation display-only preflight clarification

evaluation lifecycle 在任何 GPU workload 启动前保留了两个失败 preflight：`preflight.json` SHA-256
`9435c8cfc3e31e4198263331ebb6b7e9d4403c3349e4a171a1e32cd003a5c3f7` 与 `preflight_r1.json`
SHA-256 `3215d8ae53379ac2ffa00c9d01812b8a522da75aa427ae746e9a8411af73189a`。两次均只有 GPU0
display Xorg floor：memory 100 MiB、P8、compute-process list 为空；其余全部 checks 通过，但瞬时
utilization 分别为 10% 和 7%。随后 30 次连续只读采样显示 GPU0 为 7--10%，其余 GPU 为 0%，
说明原 training-oriented `<=1%` 瞬时门槛会稳定误拒绝单卡 evaluation。

按用户明确授权忽略 Xorg 占用，只为本次 evaluation preflight 新增显式 opt-in
`--allow-gpu0-display-utilization`：仍要求四张卡均为 RTX 3090、compute-process list 为空、每卡
memory `<=128 MiB`、P8，且 GPU1--3 utilization `<=1%`；仅 GPU0 的 display utilization 不进入
idle 判定。默认 preflight 和训练仍使用原 `<=1%` 规则，绝不把该 opt-in 用于训练。下一次只写
`preflight_r2.json`，不得覆盖前两次失败。resolved target/config、run id、device `cuda:0`、checkpoint、
evaluator、seed、438 sequences、scientific gates 均不变；不授权训练、CM 或 sampler intervention。

#### 2026-07-21 D2-T author update-rule parity completion

D2-T 训练与独立 evaluation lifecycle 均已完成并回收到 authority。训练从随机初始化开始，只把
HOIPrior 的 optimizer/update-rule contract 对齐作者 DDPM：4×RTX 3090、per-GPU batch 512、effective
batch 2048、Adam、LR `1e-4`、无 scheduler/warmup/weight decay/clipping/AMP/EMA，固定 3,000 updates、
6,144,000 windows；232-D representation、HOIPrior architecture/conditions/losses/data/sampler 均保持不变。
final online checkpoint SHA-256 为
`1543af304acf76f385dbd3656a1ca82ea25dcd504ee120f7f63e821d71483647`，训练 finite 且 final validation
total loss 2.954167。

official-438 author-native、3 windows/sequence、500-step unguided diffusion 评估得到：MPJPE 34.7367、
end-object 38.5563、xy 17.6001、object translation 57.4467、foot sliding 0.1761、contact F1 0.2764。
相对 sealed R1024 control，四项 lower-is-better paired improvement CI 全为负，contact F1 difference
95% CI 为 `[-0.0957,-0.0294]`；只有 foot-sliding ratio gate 通过。因此 mechanism/effective-diffusion
gate 均失败，分类 `author-update-rule-negative-stop`，checkpoint 不选择，CM 不授权。

compact aggregate 为
`experiments/results/p1_hoi_phase1b_d2t_author_update_rule_s42_20260721.json`，SHA-256
`0d211dd59bff79addcf1848da6ffaa5f4076a66d666e1c9d04cc086611933529`。training/evaluation authority
staging tree SHA-256 分别为
`9173d340ab66dffb08b73ba04b177bdaf171377a7e2cd484e361821d20729387` 与
`e77bb8e6d9bfb112a6bf34e142baeab672ab34a9a6ad5ede4073f199692572dc`。该结果否定“仅 update-rule parity
即可得到有效 diffusion HOIPrior”，但不否定 HOIPrior；下一训练方向必须重新预注册并针对仍未隔离的
architecture/condition/loss mechanism。不得从 D2-T checkpoint 启动 CM 或进行观察后 checkpoint selection。

#### 2026-07-21 Phase 1B D2-U from-random balanced-objective screen 预注册

D2-T 已用随机初始化、作者 DDPM update-rule contract 和 6,144,000-window 固定预算排除“只要把
Adam/LR/effective batch/FP32/无 EMA 等更新规则对齐即可学成 HOIPrior”：其 official-438
MPJPE/end-object/xy/object-translation 均显著差于 sealed balanced control。另一方面，D2-I 的
parameter-gradient audit 显示原 `50×FK + 50×object-surface` 在高噪声时令 auxiliary/reconstruction
gradient-norm ratio 约为 90--126；D2-L 证明锁定的 balanced 权重能直接修正 raw gradient routing，
D2-M/D2-N 则显示 balanced objective 可大幅改善已有模型的 kinematics/object-goal，但 D2-M 只有
64 updates 且受 AMP overflow divergence 与旧训练 lineage 限制。D2-U 因而只检验 H3 loss geometry：
在 D2-T 的 clean from-random update contract 上替换这两个已预先锁定的权重，不改变 architecture、
condition、representation、data、diffusion、sampler 或 optimization。

1. **唯一 manipulated factor。** intervention run id 固定为
   `p1-hoi-d2u-balanced-author-update-s42-20260721`、subphase `1B-D2-U0`、seed 42。相对已完成
   D2-T，唯一改变为 `fk_weight: 50.0 -> 0.3569973401779424` 与
   `object_surface_weight: 50.0 -> 0.4772322188400037`；`velocity_weight=0.1`、
   `goal_weight=1.0` 不变。权重来自 D2-I locked aggregate，不根据 D2-U 结果重新调参。
2. **固定训练 contract。** 必须从随机初始化开始，保留 232-D state、16-frame window、2-frame
   history、512-wide/16-head/8-layer Transformer、500-step x0 diffusion、原 data/split/condition/
   sampler。只在四卡 `infbagel-4gpu/node01` 运行：4×RTX 3090、batch/GPU 512、accumulation 1、
   effective batch 2048、Adam `(0.9,0.999)`、constant LR `1e-4`、FP32、无 warmup/scheduler/
   weight decay/clipping/AMP/EMA，3,000 updates、6,144,000 processed windows，midpoint/final
   checkpoint 与 validation cadence 完全复用 D2-T。禁止加载 released、author、source、current、
   balanced、D2-T、prior、resume 或任何 EMA checkpoint；不得设置 `d2m_candidate`。
3. **实现和 lifecycle。** 新增独立 D2-U Hydra config 与 fail-closed mode；D2-T exact contract
   必须保持不变。代码、config、tests、plan、registry 和 target-only evaluator/gate runner 组成一个
   logical commit。authority 完成 registry validation、targeted/full tests、compile 与 diff check 后，
   worker 只能 worker-initiated fast-forward 到 exact commit。训练前必须归档无 unresolved
   interpolation 的 resolved config、同一 escalated context 的四卡 preflight，并通过
   `tools/experiment.py start` 创建不覆盖旧结果的 manifest；detached workload 不依赖 SSH 存活。
4. **mechanism gate。** D2-U final online checkpoint 使用与 D2-T 完全相同的 author-native HOI
   official-438、每序列三窗口、500-step unguided diffusion evaluator；独立 evaluation lifecycle
   run id 固定为 `p1-hoi-d2u-native-eval-s42-20260721`、subphase `1B-D2-U0-eval`。逐 sequence、10,000 paired
   bootstrap、seed 42：`D2-T minus D2-U` 的 MPJPE、end-object、xy、object-translation improvement
   CI 下界均须 `>0`；`D2-U minus D2-T` contact-F1 CI 下界须 `>=-0.02`；D2-U/D2-T
   foot-sliding ratio CI 上界须 `<=1.10`。D2-T per-sequence records 只读复用，不重新生成。
5. **absolute effective-diffusion gate。** 同时相对 Phase-0 released aggregate 满足 MPJPE ratio
   `<=1.30`、end-object `<=2.00`、xy `<=1.50`、object-translation `<=1.50`、foot-sliding
   `<=1.10` 且 contact F1 `>=0.60`，才分类为 `effective-diffusion-hoi-prior-stop`。mechanism
   gate 通过但 absolute gate 失败，分类 `balanced-objective-positive-but-not-effective-stop`；任一
   mechanism 条件失败分类 `balanced-objective-negative-stop`；contract/hash/lifecycle 失败分类
   `balanced-objective-contract-failure-stop`。
6. **停止规则和 artifact contract。** 无论结果如何，D2-U 训练与 official evaluation 后停止；不得
   自动延长预算、选择 midpoint、修改 contact loss、启动 architecture/condition intervention 或 CM。
   必须保留 resolved config、preflight、manifest、日志、loss/validation、midpoint/final checkpoint
   hashes、RNG/checkpoint-resume 证据、official aggregate/per-sequence/bootstrap/gate JSON、artifact-tree
   hash 和 negative result。只有 absolute gate 通过后，用户才可另行授权选择 final checkpoint，并以
   新 dated amendment 讨论从自主训练 HOIPrior diffusion checkpoint 开始的 consistency stage。

#### 2026-07-21 D2-U balanced-objective completion

D2-U 已在 clean worker commit `c4293735b7a144ddf7a1190e8ecf6e43b9698d18` 完成。训练严格从
随机初始化开始，未加载 released、author、source、current、balanced、D2-T 或任何 prior
checkpoint；固定 D2-T 的 4×RTX 3090、effective batch 2048、Adam、LR `1e-4`、FP32、无
scheduler/warmup/clipping/AMP/EMA 和 6,144,000-window 预算，只把 FK/object-surface 权重改为
`0.3569973401779424/0.4772322188400037`。3,000 updates 全部 finite，无 overflow；final validation
total 为 `0.0822169`，final checkpoint SHA-256 为
`7cb379263f8a72e7f9017e4ada9d521a9e25f7c160c061305a92b9822bda2cad`。

唯一一次 official-438、3 windows/sequence、500-step unguided diffusion evaluation 得到：MPJPE
`17.0285`、end-object `10.0201`、xy `9.5509`、object translation `27.1250`、foot sliding
`0.3101`、contact F1 `0.3391`。相对 D2-T，MPJPE/end-object/xy/object-translation 的 paired
improvement 95% CI 分别为 `[17.2038,18.2254]`、`[26.8360,30.3090]`、
`[7.6452,8.4718]`、`[28.6225,32.0440]`，contact-F1 difference CI 为
`[0.0323,0.0943]`；这直接支持 H3 loss geometry 是主要机制之一。但 foot-sliding ratio CI 为
`[1.6229,1.9071]`，超过 `<=1.10` 保护门槛，且相对 Phase-0 released baseline 的 absolute
MPJPE/end-object/xy/object-translation/contact-F1 gates 仍失败。因此按预注册分类为
`balanced-objective-negative-stop`，不选择 checkpoint、不启动 CM。

D2-U 与此前 sealed balanced checkpoint 的 descriptive comparison 为：MPJPE `17.0285 vs
18.7640`、end-object `10.0201 vs 10.8458`、object translation `27.1250 vs 29.7522` 更好；
xy `9.5509 vs 7.9996`、foot sliding `0.3101 vs 0.2934` 更差；contact F1 `0.3391 vs 0.3386`
基本相同。故从随机训练的 balanced objective 已复现并略推进既有 balanced checkpoint 的主要
kinematic/object 能力，但尚未形成可用 HOIPrior，剩余缺口集中为 foot/contact 与 trajectory
质量的训练期 Pareto，而不是 update rule 或 sampler-only contact heuristic。

训练/evaluation artifacts 已由 worker 主动回收到 authority staging；固定 `LC_ALL=C` tree
SHA-256 分别为
`f480ab0223beb998821932f495f882d3067c2d6ab21ddb5fa7578c09ff43a670` 与
`67704f86e69c12f38d6a0c9382d4ac383396c384c955d81f9264450afc1ce7c3`。compact aggregate 为
`experiments/results/p1_hoi_phase1b_d2u_balanced_author_update_s42_20260721.json`。D2-U 至此停止；
任何下一训练机制必须另行 dated plan/registry amendment，且不得从本 checkpoint 自动启动 CM。

#### 2026-07-22 Phase 1B D2-V from-random balanced long-budget screen 预注册

D2-U 已直接证明 balanced loss geometry 能把 clean from-random HOIPrior 的主要 kinematic/object
指标推进到或略优于此前 sealed balanced checkpoint，但在 6,144,000 processed windows 终止时
internal validation 仍明显下降，且作者自主训练 DDPM 使用了远长于该 screen 的训练周期。D2-V
因此只检验 H4 training-budget insufficiency：在 D2-U 完全相同的 232-D diffusion HOIPrior、data、
condition、architecture、loss、optimization 和 evaluation contract 下，把固定预算扩大十倍。它不
检验 consistency、dynamic perception、sampler guidance、architecture 或 condition routing。

1. **唯一 manipulated factor。** training run id 固定为
   `p1-hoi-d2v-balanced-long-budget-s42-20260722`、subphase `1B-D2-V0`、seed 42。相对 D2-U，
   唯一变化为 `max_processed_windows: 6,144,000 -> 61,440,000`，对应 optimizer updates
   `3,000 -> 30,000`。D2-V 必须重新随机初始化；禁止 resume D2-U 或加载其 checkpoint，因此
   “更长预算”不与 checkpoint lineage 混杂。
2. **固定训练 contract。** 完全保留 D2-U 的 232-D state、16-frame window、2-frame history、
   512-wide/16-head/8-layer Transformer、500-step clean-x0 diffusion、固定 OMOMO split、全部
   conditions、FK/object-surface/velocity/goal 权重
   `0.3569973401779424/0.4772322188400037/0.1/1.0`。只在四卡
   `infbagel-4gpu/node01` 运行：4×RTX 3090、batch/GPU 512、accumulation 1、effective batch
   2048、Adam `(0.9,0.999)`、constant LR `1e-4`、FP32、无 warmup/scheduler/weight decay/
   clipping/AMP/EMA。validation windows 32,768，validation/checkpoint cadence 均保持 D2-U 的
   3,072,000 windows。禁止加载 released、author、source、current、balanced、D2-T、D2-U、
   prior、resume 或任何 EMA checkpoint；不得设置 `d2m_candidate`。
3. **实现和 lifecycle。** 新增独立 D2-V Hydra config、fail-closed mode、tests 和 target-only
   evaluator；D2-T/D2-U exact contracts 必须保持不变。代码、config、tests、plan、registry 和
   evaluator 组成一个 logical commit。authority 通过 registry validation、targeted/full tests、
   compile、resolved-config 和 semantic diff 后，worker 只能主动 fast-forward 到 exact commit。
   workload 前必须归档 fully-resolved config、同一 escalated context 的四卡 preflight，并用
   `tools/experiment.py start` 建立新 manifest；detached workload 不依赖 SSH 存活。
4. **budget mechanism gate。** final online checkpoint 只评估一次，evaluation run id 固定为
   `p1-hoi-d2v-native-eval-s42-20260722`、subphase `1B-D2-V0-eval`。复用 D2-U official-438
   per-sequence records 作为只读 control，使用相同每序列三窗口、500-step unguided diffusion、
   10,000 paired bootstrap、seed 42。`D2-U minus D2-V` 的 MPJPE、end-object、xy、
   object-translation improvement CI 下界均须 `>0`；`D2-V minus D2-U` contact-F1 CI 下界须
   `>=-0.02`；D2-V/D2-U foot-sliding ratio CI 上界须 `<=1.10`。
5. **absolute effective-diffusion gate。** 同时相对 Phase-0 released aggregate满足 MPJPE ratio
   `<=1.30`、end-object `<=2.00`、xy `<=1.50`、object-translation `<=1.50`、foot-sliding
   `<=1.10` 且 contact F1 `>=0.60`，才分类为 `effective-diffusion-hoi-prior-stop`。budget
   mechanism gate 通过但 absolute gate 失败，分类
   `long-budget-positive-but-not-effective-stop`；任一 mechanism 条件失败分类
   `long-budget-negative-stop`；contract/hash/lifecycle 失败分类
   `long-budget-contract-failure-stop`。
6. **停止规则和 artifacts。** 无论结果如何，完成 D2-V fixed-budget training 与一次 official
   evaluation 后停止；不得观察中间 checkpoint 后提前选择、延长预算、改变 loss/contact、启动
   architecture/condition intervention 或 CM。保留 resolved config、preflight、manifest、完整
   log/metrics、全部 cadence checkpoint hashes、resume evidence、official aggregate/per-sequence/
   bootstrap/gate JSON、artifact-tree hash 和任何 negative result。只有 absolute gate 通过后，
   才能等待用户另行授权选择 final checkpoint；任何 consistency stage 都必须以新的 dated
   amendment 从自主训练的 D2-V HOIPrior diffusion checkpoint 开始，本 subphase 不授权 CM。

#### 2026-07-22 D2-V balanced long-budget completion

D2-V 已在 clean worker commit `91e54654b6d9aa05e7cfca384ea7ab488018c298` 完成。D2-U 与
D2-V 的 initial model-state SHA-256 均为
`ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e`；data、model、condition、
loss、optimizer、seed 和 sampling 均相同，唯一变化是从 6,144,000/3,000 增至
61,440,000 processed windows/30,000 updates。因此这是严格 budget-only comparison，而不是从
D2-U resume。训练约 108.08 epochs，全部 loss/gradient finite，无 AMP/EMA；20 个固定 cadence
checkpoints 全部保留。final validation total 为 `0.0489330`，固定 final checkpoint SHA-256 为
`e0705681bbaeed40d353494852494d8b7bdaf4d32da92368c0d2ceedea4c01a4`，吞吐
`3,200.83 windows/s`。internal validation 最低点在 24,576,000 windows (`0.0449481`)，仅作
descriptive evidence；没有选择该中间 checkpoint，official evaluation 仍只使用预注册 final。

唯一一次 official-438、3 windows/sequence、500-step unguided diffusion evaluation 得到：MPJPE
`12.1224`、end-object `3.6807`、xy `4.0103`、object translation `16.1082`、object rotation
`1.0245`、foot sliding `0.3783`、contact recall/F1 `0.5853/0.6286`。相对 D2-U，四个
lower-is-better 指标的 paired improvement 95% CI 分别为 `[4.4763,5.3445]`、
`[5.8280,6.8378]`、`[5.1792,5.9114]`、`[10.1110,11.9449]`，contact-F1 difference CI 为
`[0.2576,0.3218]`。这直接支持 H4 training-budget insufficiency 是主要机制之一：当前
architecture/condition/232-D representation 能在不经过 CM 的情况下学到接近 released baseline 的
kinematic、object-goal 和 contact 能力。

但 D2-V/D2-U foot-sliding ratio CI 为 `[1.1232,1.3275]`，超过预注册 `<=1.10`；相对 released
baseline 的 foot-sliding ratio 为 `1.1347`，也超过 absolute `<=1.10`。其余 absolute
MPJPE/end-object/xy/object-translation gates 均通过且 contact F1 `>=0.60`。此外非 gate 的
descriptive protection 指标也恶化：hand/human penetration loss 为 `0.2641/4.1712`，分别是
released baseline 的 `1.6259×/1.6110×`。因此必须按预注册分类
`long-budget-negative-stop`，不选择 checkpoint、不启动 CM，也不以 favorable subset 或中间
checkpoint 改判。科学上它是当前最强的 from-random HOIPrior capability evidence，但尚不是 sealed
usable expert；剩余缺口已收缩为强 denoiser 上的 foot-sliding/contact/penetration protection，而非
基本 denoiser 无法学习。

training/evaluation artifacts 已由 worker 主动回收到 authority
`/data/yujinlun/InfBaGel-p1b-staging/`；标准 directory tree SHA-256 分别为
`fe1d9ec82369864ab6b3dfd334d01d07c58832027b56f25435e3d8074a15c6f3`（113 files / 7,127,146,664
bytes）与 `05b8a9c6fc0c1021c3fbc2a09308b80246eadee6efb9e4221eb0baf4477c6bb8`（16 files /
362,375 bytes），worker/authority 完全一致。compact aggregate 为
`experiments/results/p1_hoi_phase1b_d2v_balanced_long_budget_s42_20260722.json`。D2-V 至此停止；
任何下一 foot-sliding/penetration training intervention 必须另行 dated plan/registry amendment，
且 consistency 仍不授权。

#### 2026-07-22 Phase 1B D2-W fixed-checkpoint FK foot-sliding frontier diagnostic 预注册

D2-V 已证明 10x budget 能把独立 232-D diffusion HOIPrior 的 MPJPE、object-goal、object-translation
和 contact 提升到接近 released baseline，但 fixed final 的 official foot-sliding 仍为 `0.3783`，相对
D2-U 的 paired ratio CI 为 `[1.1232,1.3275]`。训练期 direct normalized velocity validation loss 同期
从 `0.0010924` 降至 `0.0001033`，而 official evaluator 通过 predicted rotations、root translation 和
rest offsets 做 FK 后再计算近地面 ankle/toe 水平位移；production velocity loss 则只读取 direct
joint-position 与 object-translation channels。D2-W 在任何新训练前，只判断 foot-sliding 缺口主要是
`24.576M -> 61.44M` constant-LR 后半程退化，还是在能力形成时已经存在的 rotation/FK trajectory
loss-geometry 缺口。

1. **唯一 reportable diagnostic。** run id 固定为
   `p1-hoi-d2w-checkpoint-frontier-s42-20260722`、subphase `1B-D2-W0`、seed 42，只能在
   `infbagel-4gpu/node01` 运行。加载同一 D2-V run 的三个 online model checkpoints，仅用于 inference：
   `6,144,000` windows（file SHA-256
   `be8233c0a4c013d973c4140ba5c1f472332f1fdd6be8efa21585deeb250506d3`，model-state SHA-256
   `cfcb5836129d177bf57c60ffd8669ee4516fad77f52b58afd037d063e9aaa0c7`）、预先由 teacher-forced
   internal validation total 唯一确定的 `24,576,000` windows（file SHA-256
   `efab7f55d6a719ac85659de0aa66c2f94235e1875ae5e6951e9c4334017ee9a3`，model-state SHA-256
   `1ee340962d158e12a31d3ad081da37886cd8bbc3eddd80b523de4eb236ba2735`）和 fixed final
   `61,440,000` windows（file SHA-256
   `e0705681bbaeed40d353494852494d8b7bdaf4d32da92368c0d2ceedea4c01a4`，model-state SHA-256
   `f7d134ac98ede806abae322c77816ef21ace427e3905a4cb5e1d4a2a2b4b89fc`）。D2-V 6.144M
   model state 已只读验证与 D2-U final 完全相同；不得加载 optimizer/scheduler/scaler/RNG/EMA，
   不得写 checkpoint 或执行 optimizer update。
2. **固定 internal-only rollout。** 复用 D2-M 在 D2-V 前已经 sealed 的 internal-validation
   three-window native holdout：eligible sequence ranks `128--159`、32 sequences、96 windows、selection
   SHA-256 `30524c88481f6cb81e8063073d510ad01543be92d91eb4ef9b2b8a376cc4fbae`。三个 checkpoint
   必须使用逐 step 完全相同的 500-step diffusion noise、matched text/BPS/goal/progress、generated-history
   handoff 和 current generated-object BPS；只用 online weights，无 CFG、dynamic perception、guidance、
   CHOIS、official-438 或 released/author checkpoint。
3. **指标与 parity。** 每个 checkpoint 必报 per-sequence/aggregate：direct-joint foot sliding、
   rotation-to-FK foot sliding、FK MPJPE、pelvis/object goal error、object translation/rotation error、physical
   contact precision/recall/F1，以及 direct-vs-FK foot trajectory disagreement。FK 使用与 production loss
   相同的 24-joint parents/rest offsets，脚滑使用 official evaluator 相同 ankle/toe indices、Y-up、近地面
   height thresholds 与水平 displacement 公式。D2-W 还必须以 synthetic parity test 证明 torch 实现与
   `code/eval_metrics.py::compute_foot_sliding_for_smpl` 一致，并证明三个 checkpoint 的 q-noise hashes
   完全相同。
4. **唯一 gate。** 10,000 次 paired sequence bootstrap、seed 42。`24.576M` 只有同时满足以下条件才分类
   `midbudget-protection-supported-stop`：`61.44M minus 24.576M` 的 FK foot-sliding difference CI 下界
   `>0`；`24.576M / 61.44M` 的 FK-MPJPE、pelvis-goal、object-goal、object-translation ratio CI 上界均
   `<=1.10`；`24.576M minus 61.44M` physical-contact-F1 CI 下界 `>=-0.02`；且相对 6.144M，
   FK-MPJPE、object-goal 和 object-translation 的 improvement CI 下界均 `>0`。任一失败分类
   `midbudget-protection-negative-stop`；contract/hash/parity/lifecycle 失败分类
   `midbudget-protection-contract-failure-stop`。direct/FK divergence 和 penetration coupling 只作机制解释，
   不得改写 gate。
5. **停止与后续。** 本 subphase 不选择任何 D2-V checkpoint，不使用 internal 或 official 指标追认
   favorable intermediate，不修改 production loss/model/condition/sampler，不训练，不启动 CM。若 gate
   通过，下一 dated proposal 只能测试 from-random fixed mid-budget/learning-rate schedule；若 gate 失败，
   下一 dated proposal 才能测试一个 evaluator-aligned FK-foot temporal loss，并保持 D2-V 的 61.44M
   budget。两条路径都必须重新获得用户确认；penetration 的 smallbox/suitcase contact-collision tradeoff
   保留为后续独立假设，禁止与 foot intervention 捆绑。

#### 2026-07-22 Phase 1B D2-W worker interpreter symlink implementation amendment

D2-W implementation commit `21580b66b4af0c79ca54940afb908df34ff4a4a4` 发布后、任何 run directory、
manifest、checkpoint load 或 GPU workload 创建前，只读 worker preflight 发现 lifecycle guard 的确定缺陷：
规范解释器 `/home/yujinlun/data/envs/infbagel/bin/python` 是指向 `python3.8` 的符号链接，
`sys.executable` 保留前一路径，而 `Path(sys.executable).resolve()` 得到后一路径；原 guard 将这个 canonical
路径与未 canonicalize 的 `EXPECTED_PYTHON` 字符串比较，故合法 worker 环境也必然触发
`D2-W interpreter mismatch`。实测两侧分别为
`.../bin/python3.8` 与 `.../bin/python`，比较结果为 false。用户已确认授权以下最小 amendment。

1. **唯一实现修改。** interpreter identity guard 只改为比较
   `Path(sys.executable).resolve()` 与 `Path(EXPECTED_PYTHON).resolve()`；环境变量仍必须逐字等于规范
   `INFBAGEL_PYTHON`，worker hostname、clean worktree、CUDA、checkpoint/data/hash/parity 等 guard 均不变。
2. **回归要求。** 单元测试必须构造 `python -> python3.8` 符号链接，证明 alias/target 双向 canonical
   比较通过、不同文件失败；D2-W 全部测试、完整 authority suite、worker role-applicable suite 与 registry
   validation 必须通过后才能创建 lifecycle artifact。
3. **科学协议不变。** run id、三个 checkpoint 及 hash、internal selection、paired noise、500-step sampling、
   指标、bootstrap gate 和停止规则完全不变；不训练、不选择 checkpoint、不运行 official-438/CHOIS/CM，
   不修改 production model、loss、condition、sampler 或 evaluator。
4. **失败处理。** 修复后若 worker guard、resolved-config、preflight、checkpoint contract、noise parity 或 finite
   contract 任一失败，保留 artifact 并分类 contract failure；不得通过放宽 guard 或更换解释器继续运行。

#### 2026-07-23 Phase 1B D2-W date-rollover operational retry amendment

原 lifecycle `p1-hoi-d2w-checkpoint-frontier-s42-20260722` 的 manifest 于
`2026-07-22T23:59:37+08:00` 创建，但等待受控 SSH/Git 操作后，worker-owned workload 实际于
`2026-07-23T00:10:32+08:00` 才启动，`metrics.json` 于 `00:11:18+08:00` 写成。由于实际 GPU workload
日期已经跨日，原 `20260722` run id 违反真实日期绑定，即使进程 return code 为 0、推理合约通过，也不得作为
reportable D2-W 科学结果。该 lifecycle 已以 `aborted` finish/register 到 run-local registry，全部文件保留、
run id 永不复用；其已观察数值不得用于改动阈值、checkpoint、selection、指标、gate 或任何后续配置。

授权一次且仅一次严格 operational retry：
`p1-hoi-d2w-checkpoint-frontier-r1-s42-20260723`，subphase `1B-D2-W0-r1`。r1 完整继承
2026-07-22 D2-W 预注册的三个 checkpoint/file/model-state hash、32-sequence/96-window internal selection、
逐 step paired noise、generated-history/current-BPS、500-step online diffusion、全部 metric、10,000 次 paired
bootstrap、唯一 gate、分类和停止规则。唯一允许差异是 run id、subphase、真实日期及由新 commit/lifecycle
自然产生的 manifest/preflight/runtime hash；不得因已观察 aborted output 改变任何科学变量。

r1 必须使用新结果目录和新 resolved config/preflight/manifest，重新绑定全部 input hash；原 failed/passed
preflight、manifest、log、metrics、exit code 与 aborted registry 均不覆盖。r1 无论 gate 正负都 finish/register、
回收并停止，不选择 checkpoint、不训练、不运行 official-438/CHOIS/CM，也不开始 FK-foot loss intervention。

D2-W r1 已在 worker commit `e8f4599d1f4d67b1b90f319c850ed2230280b1ad` 完成，return code 0，
runtime `32.128 s`。selection、三个 checkpoint file/model-state hash、逐 step noise、official foot-sliding
torch/numpy parity 与 finite contract 全部通过；r1 与跨日期 aborted lifecycle 的 selection、checkpoint
contracts/metadata、完整 per-sequence results、comparison、decision 和 contract 逐字段完全一致。两个 immutable
目录均已由 worker 主动回收到 authority staging，worker/authority 标准 tree hash 一致：aborted 为
`224c9cb6837ef08ebaa46c2c3c140bb4cbf3abd30c140c2400c4dfea6a98f36b`（8 files / 281,739
bytes），completed r1 为
`bdbd4d66c7304e5d8f80624bd66b99b6088fbda68b898ea12fa95c3b92cfad4d`（7 files / 277,106
bytes）。compact aggregate 为
`experiments/results/p1_hoi_phase1b_d2w_checkpoint_frontier_r1_s42_20260723.json`。

| checkpoint | FK foot sliding | direct foot sliding | FK MPJPE cm | pelvis goal cm | object goal cm | object trans cm | direct/FK foot disagreement cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6.144M control | 1.0184 | 1.4574 | 24.5116 | 8.9756 | 63.2352 | 45.4860 | 11.2901 |
| 24.576M midpoint | 1.3803 | 1.3443 | 15.5517 | 4.5868 | 56.6646 | 20.6455 | 5.2632 |
| 61.44M final | 0.9332 | 1.0083 | 13.9775 | 3.9919 | 58.4418 | 20.3566 | 3.2849 |

核心 gate 的 `final - midpoint` FK foot-sliding paired 95% CI 为
`[-0.7963,-0.1099]`，方向与“后半程退化”假设相反；midpoint/final 的 FK-MPJPE ratio CI 为
`[1.0286,1.2009]`，pelvis-goal 与 object-translation ratio CI 上界也分别为 `1.4025/1.2015`，且
midpoint 相对 control 的 object-goal improvement CI `[-4.0275,17.6240]` 未严格大于 0。因此分类固定为
`midbudget-protection-negative-stop`，不选择 midpoint。该 32-sequence internal holdout 上三个 checkpoint
physical-contact precision/recall/F1 均为 0，contact-preservation 项只能视为无信息，不能外推到 official-438。

科学上，D2-W 排除了“24.576M 后继续 constant LR 是 D2-V official foot-sliding 缺口主要来源”这一首选
解释；后半程反而降低 direct 与 rotation-to-FK foot sliding 并缩小 direct/FK trajectory disagreement。
结合 production velocity loss 不读取 rotation channels、official metric 完全依赖 rotation-to-FK 的确定
目标错位，下一候选应是单一 evaluator-aligned FK-foot temporal training loss，而不是 mid-budget checkpoint
选择或 schedule-only retry。这仍是基于证据的机制优先级，不是该 loss 必然有效的证明；penetration 必须保持
独立，不得捆绑。D2-W 至此停止，未训练、未启动 CM、未选择或蒸馏任何 checkpoint。

#### 2026-07-23 Phase 1B D2-X FK-foot temporal gradient-routing screen 预注册

D2-V 已证明当前 232-D diffusion HOIPrior 在 61,440,000 processed windows 后能够从随机初始化学到接近
released baseline 的 MPJPE、object-goal、object-translation 与 contact；其唯一 absolute capability gate
失败项是 foot sliding，且 penetration 仍较弱。D2-W 进一步排除了中途 checkpoint 或后半程 constant-LR
过冲作为主要解释，并确认现有 velocity objective 只监督 direct normalized joint/object-translation
channels，而 official foot sliding 完全由 predicted root/rotations 经 FK 后得到。D2-X 因此只检验一个
loss-gradient routing 假设：把既有 velocity residual 中四个足部关节的水平分量对齐到 evaluator 使用的
rotation-to-FK trajectory，是否能降低脚滑且不破坏 D2-V 已形成的能力。

1. **唯一 manipulated factor。** Training run id 固定为
   p1-hoi-d2x-fk-foot-temporal-routing-s42-20260723、subphase 1B-D2-X0、seed 42。相对 D2-V，
   velocity loss 的 tensor shape、元素数、mean-square reduction 与全局 weight 0.1 均不变；只替换
   joints [7,8,10,11] 的 x/z 共 8 个 predicted temporal residual slots。Prediction 侧先用 predicted
   root、22 个 6D rotations、既有 24-joint parents/rest offsets 得到 FK positions，再用既有
   position min/max 映射到 direct position channels 相同的 normalized scale。Target 侧继续使用
   normalized clean direct foot positions。第一个 future frame 的 previous prediction 必须是 immutable
   GT history 最后一帧；其后 previous prediction 使用前一帧 predicted normalized FK position。四个
   足部 y、其余 joint xyz、object translation 以及所有 target residual 保持 D2-V 原实现。不得增加
   新 loss term、可调 weight、height/contact gate 或 sampler correction。
2. **固定训练 contract。** 必须重新随机初始化，保留 D2-V 的 232-D state、16 frames、2 history
   frames、512-wide/16-head/8-layer Transformer、500-step clean-x0 diffusion、OMOMO split、全部
   conditions、FK/object-surface/velocity/goal weights
   0.3569973401779424/0.4772322188400037/0.1/1.0。只在 infbagel-4gpu/node01 运行：
   4×RTX 3090、batch/GPU 512、accumulation 1、effective batch 2048、Adam (0.9,0.999)、
   constant LR 1e-4、FP32、无 warmup/scheduler/weight decay/clipping/AMP/EMA，固定
   61,440,000 processed windows、30,000 updates、32,768 validation windows 以及每
   3,072,000 windows validation/checkpoint cadence。禁止加载 released、author、source、current、
   balanced、D2-T/U/V、prior、resume 或任何 EMA checkpoint；不得设置 d2m_candidate。
3. **实现和 lifecycle。** 在任何 production 代码改动前完成本 plan 与 registry 预注册。新增独立
   D2-X Hydra config、fail-closed mode/host/provenance contract 和 tests；D2-T/U/V 的 exact config、
   loss 与 checkpoint-resume contract 必须保持不变。测试至少证明：routing 关闭时与原 velocity
   formula bitwise/数值等价；开启时足部 x/z formula 正确；第一 future frame 只读取 immutable
   history；非足部、足部 y 与 object residual 不变；rotation/root 获得 velocity gradient 而 direct
   foot x/z 不再获得该分量梯度；随机初始化与 forbidden-checkpoint guard 生效。代码、config、
   tests、plan、registry 和 target-only evaluator 必须组成一个 logical commit。authority 全量验证后，
   worker 只能主动 fast-forward 到相同 commit。训练前归档 fully-resolved config、同一 escalated
   context 的四卡 preflight，并用 tools/experiment.py start 创建不可覆盖的 reportable manifest。
4. **固定 official evaluation。** Evaluation run id 固定为
   p1-hoi-d2x-native-eval-s42-20260723、subphase 1B-D2-X0-eval。只评估 fixed final online
   checkpoint 一次，使用 author-native HOI official-438、每 sequence 三窗口、500-step unguided
   diffusion；无 CFG、dynamic perception、guidance、CHOIS selection、FID 或 R-precision。D2-V
   final official per-sequence records 与 aggregate 作为 immutable paired control，只读复用且不重新
   生成。统计单位为 sequence，10,000 次 paired bootstrap，seed 42。
5. **mechanism/effective gates。** Mechanism gate 要求同时满足：D2-V minus D2-X foot-sliding
   difference 的 paired 95% CI 下界 > 0；D2-X/D2-V 的 MPJPE、end-object、xy 与
   object-translation ratio CI 上界均 <= 1.10；D2-X minus D2-V contact-F1 difference CI 下界
   >= -0.02；D2-X/D2-V 的 hand-penetration-loss 与 human-penetration-loss ratio CI 上界均
   <= 1.10；以及 contract、finite loss/gradient、normalization、history 与 checkpoint provenance
   全通过。Absolute effective-diffusion gate 继续使用 Phase-0 released ratios：MPJPE <= 1.30、
   end-object <= 2.00、xy <= 1.50、object-translation <= 1.50、foot sliding <= 1.10，且
   contact F1 >= 0.60。Mechanism 与 absolute gate 均通过时分类
   fk-foot-temporal-routing-positive-candidate-stop；仅 mechanism 通过时分类
   fk-foot-temporal-routing-positive-but-not-effective-stop；任一 mechanism 条件失败分类
   fk-foot-temporal-routing-negative-stop；contract/hash/lifecycle 失败分类
   fk-foot-temporal-routing-contract-failure-stop。
6. **停止规则和 artifacts。** 无论结果如何，完成 fixed-budget training 与一次 official evaluation
   后停止；不得观察中间 checkpoint 后选择 favorable state、延长预算、同时修 penetration/contact、
   修改 architecture/condition、启动 sampler heuristic 或 consistency distillation。即使两个 gate 均
   通过，本 subphase 也只产生 foot-protected diffusion candidate，checkpoint 选择及后续 penetration
   intervention 必须另行确认；CM 仍不授权。必须保留 resolved config、machine preflight、manifest、
   完整 logs/metrics、全部 cadence checkpoint hashes、resume evidence、loss/gradient routing audit、
   official aggregate/per-sequence/bootstrap/gate JSON、artifact-tree hash 和任何 negative result。

#### 2026-07-23 Phase 1B D2-X penetration finite-mask measurement amendment

在 D2-X implementation 尚未提交、未创建 run directory、未启动任何 GPU workload 时，对 immutable
D2-V official-438 control 的 per-sequence artifact 做 fail-closed schema check，发现
hand_pen_loss_omomo 与 human_pen_loss_infbagel 仅在相同的固定 181 条 sequence 上为 finite，其余
257 条均为 null；D2-V aggregate 的两个 penetration mean 也正是按这 181 条计算。该缺失由既有 author
evaluator/object asset coverage 决定，不是 D2-X 结果或 favorable selection。固定 181-sequence ID
列表按字典序、每行一个 ID 且末尾换行的 SHA-256 为
2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec。

D2-X measurement contract 因而在训练前明确为：MPJPE、end-object、xy、object-translation、
foot-sliding 与 contact-F1 继续使用全部 438 个 paired sequences；两个 penetration ratio 只使用上述
预先由 control 封存的 181 条 finite mask，并且 target 两个字段的 finite/null mask 必须分别与 control
完全相同、两个 target penetration masks 也必须相同。任何 mask identity/count/hash 不一致均分类为
contract failure，不允许按 target 数值重新筛选或用 aggregate-only ratio 绕过。bootstrap replicates、
seed、ratio 上界 1.10、其余 gate、训练唯一变量、run id、预算和停止规则全部不变。

#### 2026-07-23 Phase 1B D2-X pre-optimizer dispatch failure and r1 operational amendment

首次 reportable lifecycle p1-hoi-d2x-fk-foot-temporal-routing-s42-20260723 在 clean worker commit
48e2e6c31d281af8809d35b5c8ce2ac8123205d1、resolved config 与四卡 preflight 均通过后启动，但在
任何 optimizer update、processed window 或 checkpoint 之前 fail-closed。确定原因是
code/train_hoi_prior.py 的 pre-optimizer locked loss-weight dispatch 仍只将 D2-U/D2-V 识别为
balanced weights mode，遗漏已由独立 exact contract 验证的 D2-X；因此合法的
FK/object-surface weights 0.3569973401779424/0.4772322188400037 被旧 generic guard 误判。
该失败不是 loss 数值、显存、数据、梯度或 checkpoint lineage 结果，不能用于判断 D2-X 科学假设。

失败 lifecycle 已永久保留并标记 failed：exit code 1、optimizer updates 0、processed windows 0、
checkpoint 0，未加载 released/author/prior checkpoint、未启动 consistency。manifest/metrics/
resolved/preflight/log/run-local-registry SHA-256 分别为
936984115a66c7142b5254c4cc5d33874e855be46972ceec5e75f69483be8c7e、
e52a6eebb4d3fc40e9c204e5381520ba5b65c42ec37b3d3aae945fa4ef501ff7、
5746293fc4f5455a3619b4aec46e5b899aa51f2751416d29577bd54f83b4f226、
c8e1fee0ca60fcd92d482a838601396ec7f2ee4e0ac9d3c6bd62ea2390cd9ae9、
71fa61f14a230fc05bc1dec3f234c1e258149b6a88602762ad6584baf9a01dd4 和
5e7486c4991990654a448ba10314a11db2ab157d68d0c834c48280ec456279e5；完整 tree 为
11 files / 33,752 bytes，SHA-256
4088e04b1b92d412d25aa1842cb6cd2d6d4191a48c79388fdc3c8229bf16ab95。

用户已授权继续 D2-X，因此只允许一次 operational r1：

1. 唯一实现修复是让 pre-optimizer balanced loss-weight dispatch 明确包含 D2-X，并增加直接回归测试，
   证明 D2-U/D2-V/D2-X 返回相同 locked balanced weights，而普通/D2-T mode 仍返回 50/50。不得改动
   loss formula、routing、weights、data、architecture、condition、optimization、budget 或 evaluator gate。
2. 新 training run id 固定为
   p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723、subphase 1B-D2-X0-r1；新 evaluation
   run id 固定为 p1-hoi-d2x-native-eval-r1-s42-20260723、subphase 1B-D2-X0-eval-r1。旧 training
   id 永不复用，旧未启动 evaluation id 不得绑定到 r1 checkpoint。
3. r1 完整继承原 D2-X 和 penetration-mask amendment 的唯一变量、随机初始化、61,440,000-window/
   30,000-update contract、4×3090、official-438/181 finite-mask statistics、所有 gates、classifications、
   artifact contract 与停止规则。必须形成新完整 commit、worker 主动 fast-forward、重新生成
   resolved config/preflight/manifest；禁止从失败 lifecycle resume 或加载任何 checkpoint。
4. 若同一 dispatch blocker 再现或出现新的 pre-optimizer contract failure，r1 必须保留并停止，不得继续
   自由重试。若 r1 进入稳定训练，则按原 D2-X 协议完成；仍不授权 checkpoint selection、penetration
   intervention 或 CM。

#### 2026-07-23 D2-X r1 completion and negative gate

D2-X r1 在 `3af3facaf73d3bbffcca6d6181bac1ff89909a24`、`infbagel-4gpu/node01` 上从随机初始化完成了
固定的 61,440,000 processed windows / 30,000 optimizer updates。effective batch 为 2,048（4×3090、
每卡 512、accumulation 1），loss/gradient 全部 finite，20 个 cadence checkpoint 和 80 个 RNG sidecar
均保留；未加载 released、author、source、current、balanced、prior 或任何 resume/EMA checkpoint。
最终 online checkpoint SHA-256 为
`b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51`，training manifest/metrics/
resolved/preflight/run-local-registry SHA-256 分别为
`2011ded7310f851d3a1278bd65fe2d19fdca8f7b859d289d7e240b3d8d347d85`、
`0c99ac8b5880b1e7419cc6fe9c4be6388e7f806f066e9863da0ac320336693f3`、
`5e913957d0dde5bd3d589e98248436a14353f831ccfd422a5a96d152f7273130`、
`1ef9206e8521ca73ef78c0dec7bddc1ef1b192e138f3765d1f06ff4ef7187cac` 和
`9dde11270e443f39e86ab77b26f0a80fd097cba2f42b7591d314f87373a1e856`；完整 training tree 为
112 files / 7,127,226,145 bytes，SHA-256 `3f95773270e4701310daac9128c19d822d0a4e887ad7c5ddd40008f1b98a47c6`。

首次 evaluation preflight 只因错误地把 `--chois-root` 指向 checkout 内的资产目录而失败
（未创建 evaluation manifest、未加载 checkpoint、未启动 GPU workload）；该失败文件保留。随后使用
固定的 `/home/yujinlun/data/evaluators/chois_release` 重新捕获 preflight，所有 pinned checks、四卡
idle/无 CUDA compute process 和 Xorg 显示层约束通过。唯一一次 official-438 evaluation 使用 final
online checkpoint、每序列 3 windows、500-step unguided diffusion，无 CFG、dynamic perception、guidance、
CHOIS、FID 或 R-precision。D2-V final records 作为只读 paired control，181-sequence penetration mask
保持完全一致。

target aggregate 为 MPJPE `12.0508`、end-object `3.7402`、xy `4.0505`、object translation `15.9940`、
foot sliding `0.3630`、contact F1 `0.6374`、hand/human penetration loss `0.2454/3.8691`。相对 D2-V，
foot-sliding mean 改善 `0.0152733`，但 10,000-replicate paired bootstrap 95% CI 为
`[-0.0037904, 0.0341575]`，没有满足预注册的 CI 下界 `>0`；其余 kinematic/object/contact/penetration
protection checks 均通过，且相对 released baseline 的所有 absolute checks 也通过。因此分类严格为
`fk-foot-temporal-routing-negative-stop`。这是一个保留 D2-V 能力但未证实 foot-sliding 因果改善的负结果，
不是 evaluator 或 artifact contract 失败；不得从该 checkpoint 选择、延长预算、修改 sampler/contact/
penetration，或启动 consistency distillation。

evaluation manifest/metrics/aggregate/per-sequence/resolved/preflight/run-local-registry SHA-256 分别为
`fa19565eb96155f735a7b8c1569a95e1069267b8641ffe7968e878554fee4550`、
`f2cb76d0c248c4d4b8ce4571758c1937e2e19170199c6e9631df21858dc1c807`、
`3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b`、
`69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`、
`206eb63d15231d5986adf001295810b1a83cf88511f7435379d43f35a8c03617`、
`3262548e948038f3725978e645c4f2f14893253bddfa5349ec80e018b18cf76e` 和
`6739dd94d7181c76020f33d3cdbf24aff5f3fe24495c411363808722fc3ba1a`；evaluation tree 为 16 files /
366,190 bytes，SHA-256 `c4a853d99659ac92ac830621a0e8caf68aea3db9f3d954b3486d3aa4d3d3eb74`，worker/authority
完全一致。compact aggregate 为
`experiments/results/p1_hoi_phase1b_d2x_fk_foot_temporal_routing_r1_s42_20260723.json`，SHA-256
`fdf21f8b0042d1d26ac2a3b4cf8a073a43cd20283f0292d55572aa66de6e42f6`。D2-X 至此停止；
任何新的 HOIPrior 机制必须另行 dated plan/registry amendment，且 consistency stage 仍未授权。

#### 2026-07-23 Phase 1B D2-Y routed-foot residual amplification screen 预注册

D2-X 已证明 evaluator-aligned FK-foot temporal routing 没有显著破坏 D2-V：official-438 上
MPJPE、object、contact 与 penetration protection 均通过，foot sliding 点估计从 `0.3783` 降至
`0.3630`；但 paired mean improvement `0.0152733` 的 10,000-replicate sequence bootstrap 95% CI
`[-0.0037904,0.0341575]` 包含 0，因此不能选择 checkpoint。封存代码与只读 audit 同时确认，
velocity tensor 每 future frame 有 87 个 residual slots，routed foot x/z 仅 8 个，占
`8/87 = 9.1954%`；D2-X final validation 的 unweighted velocity MSE 为 `0.0001015743360`，
乘全局 weight `0.1` 后只占 total validation loss 的 `0.0208%`。固定 seed-42 probe 上 late routed
scalar 只占 weighted velocity scalar 的 `5.42%--5.79%`，routed gradient/FK-gradient norm ratio
仅 `0.00640%--0.01049%`，且 routed 与 FK gradient cosine 为 `-0.7435` 至 `-0.8086`。这支持
“routing 正确但 global mean/weight 使信号过弱”为首选解释，并把“rotation/root 上与 FK 的竞争抵消”
保留为次要竞争解释；official metric 的 30 Hz interpolation、predicted floor 与 near-ground nonlinear
gate 仍构成尚未消除的语义差异。上述 audit 没有发现新的确定 implementation defect；D2-X r0 的
0-update dispatch defect 继续作为独立 lifecycle failure 永久保留。

D2-Y 只检验一个可证伪的 loss-geometry 机制：保持 D2-X 的 8 个 FK-routed residual slots 和所有其他
训练条件不变，仅对这 8 个 squared residual errors 使用固定乘数 `1024`，仍在原 87×14 tensor 上做
一次 global mean，且全局 velocity weight 仍为 `0.1`。`1024` 是预先由封存 late probe 选择的最小
2 的幂，使三个 preregistered timestep strata 的 routed/FK gradient-norm ratio 全部至少为 5%
（线性外推为 `8.00%/6.55%/10.74%`）；不得根据 smoke、中间 validation 或训练结果调整或 sweep。

1. **唯一 manipulated factor。** Training run id 固定为
   `p1-hoi-d2y-routed-foot-amplification-s42-20260723`、subphase `1B-D2-Y0`、seed 42。对
   D2-X routed slots（joints `[7,8,10,11]` 的 x/z）令
   `L_velocity = mean(w_j * (predicted_residual_j-target_residual_j)^2)`，其中 routed slots
   `w_j=1024`，其余 79 slots `w_j=1`。Residual construction、target、first-future immutable GT
   history、later predicted-FK history、tensor shape/element count、全局 velocity weight、FK/
   reconstruction/object-surface/goal formula 全部保持 D2-X 不变；不得增加 height/contact gate、
   penetration term、sampler correction或第二个 loss knob。
2. **固定训练 contract。** 必须重新随机初始化并保留 232-D representation、16 frames、2 history
   frames、512-wide/16-head/8-layer Transformer、500-step clean-x0 diffusion、OMOMO seed-42 split、
   全部 conditions，以及 FK/object-surface/velocity/goal weights
   `0.3569973401779424/0.4772322188400037/0.1/1.0`。只允许在
   `infbagel-4gpu/node01` 用 4×RTX 3090、batch/GPU 512、accumulation 1、effective batch 2048、
   Adam `(0.9,0.999)`、constant LR `1e-4`、FP32、无 warmup/scheduler/weight decay/clipping/AMP/EMA，
   固定 61,440,000 processed windows / 30,000 updates、32,768 validation windows 和每
   3,072,000 windows validation/checkpoint cadence。禁止加载 released、author、D2-V、D2-X、
   prior、resume、EMA 或任何其他 checkpoint；不得设置 `d2m_candidate`。若 reportable lifecycle
   在日期变化后才启动，必须先按启动机器真实 `date` 追加只改 run/subphase 标识的 operational
   amendment，旧 id 不得创建或复用。
3. **固定 internal mechanism diagnostic。** 复用 D2-W 封存的 32-sequence/96-window selection
   （selection SHA-256
   `30524c88481f6cb81e8063073d510ad01543be92d91eb4ef9b2b8a376cc4fbae`），用相同 clean batch、
   timestep、noise、condition dropout 与 D2-X early/mid/final online checkpoints作 paired control。
   D2-Y 必须报告 early/mid/final 和 timestep `0/249/499` 的 routed normalized residual RMS，以及
   root、rotation、input projection、output projection、Transformer parameter-gradient norm；
   同时报告 routed gradient 与 reconstruction、FK、object-surface、goal gradient cosine。Internal
   mechanism gate 固定为 final D2-X minus D2-Y routed-foot normalized residual MSE 在 timestep
   `249` 和 `499` 的 sequence-paired 10,000-replicate bootstrap 95% CI 下界均 `>0`。Internal gate
   只诊断放大是否被优化器吸收，不用于 checkpoint selection。
4. **固定 official evaluation。** Evaluation run id 固定为
   `p1-hoi-d2y-native-eval-s42-20260723`、subphase `1B-D2-Y0-eval`。只评估 fixed final online
   checkpoint 一次，使用 D2-X 完全相同的 author-native official-438、每 sequence 三窗口、
   500-step unguided diffusion；无 CFG、dynamic perception、guidance、CHOIS selection、FID 或
   R-precision。D2-X r1 final aggregate/per-sequence records 和 checkpoint SHA-256
   `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
   `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a` /
   `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51`
   作为 immutable paired control，只读复用。统计单位为 sequence，bootstrap 10,000 次、seed 42；
   penetration 继续绑定 D2-X 已封存的相同 181-sequence finite mask 和
   `2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec` ID hash。
5. **success、negative 与 effective gates。** Official mechanism gate 要求：D2-X minus D2-Y
   foot-sliding difference CI 下界 `>0`；D2-Y/D2-X 的 MPJPE、end-object、xy、
   object-translation、hand-penetration-loss 与 human-penetration-loss ratio CI 上界均
   `<=1.10`；D2-Y minus D2-X contact-F1 difference CI 下界 `>=-0.02`；所有 contract、finite、
   normalization、history、mask 与 provenance checks 通过。Absolute gate 继续相对 Phase-0
   released baseline 使用 MPJPE/end-object/xy/object-translation/foot-sliding ratio
   `<=1.30/2.00/1.50/1.50/1.10` 且 contact F1 `>=0.60`。
   internal 与 official mechanism gates、absolute gate 全通过时分类
   `routed-foot-amplification-positive-candidate-stop`；internal 通过而 official foot gate 失败为
   `routed-foot-amplification-transfer-negative-stop`，区分“信号确实变强但训练 surrogate 未转移到
   near-ground official metric”；official foot 改善但任一 protection gate 失败为
   `routed-foot-amplification-conflict-negative-stop`，支持 gradient conflict/capability tradeoff；
   internal gate 失败为 `routed-foot-amplification-optimization-negative-stop`；mechanism 通过但
   absolute gate 失败为 `routed-foot-amplification-positive-but-not-effective-stop`；artifact/
   lifecycle contract 失败为 `routed-foot-amplification-contract-failure-stop`。
6. **实现、停止规则与 artifact contract。** Plan/registry 后才允许新增 fail-closed D2-Y config、
   weighted reduction、contract、diagnostic、target-only evaluator 与 tests；D2-T/U/V/X 的 exact
   config、loss 和 checkpoint-resume contract 必须保持不变。authority clean validation 后形成一个
   logical commit，worker 只能主动 fast-forward 到相同 Git object；训练前必须归档 fully-resolved
   config、同一 execution context 的 machine preflight，并用 `tools/experiment.py start` 创建
   reportable manifest。必须保留 manifest/resolved/preflight、完整 logs/metrics、20 个 cadence
   checkpoints 与 80 个 RNG sidecars及 hashes、resume evidence、internal per-sequence/gradient/
   bootstrap audit、official aggregate/per-sequence/bootstrap/gate JSON、normalization/mask/provenance
   audits、artifact-tree hash和任何 failure。无论分类如何，完成 fixed-budget training、internal
   diagnostic 和一次 official evaluation 后停止；不得选择中间或最终 checkpoint、延长预算、从
   D2-V/D2-X resume、post-hoc 改 multiplier/gate、启动 penetration intervention、HSIPrior、Mixer 或
   consistency distillation。即使分类 positive，本 subphase 也只产生待用户另行确认的 diffusion
   candidate，绝不自动授权 checkpoint selection 或 CM。

#### 2026-07-24 Phase 1B D2-Y post-training lifecycle date amendment

D2-Y training lifecycle 已于真实日期 2026-07-23 启动，使用已预注册的
`p1-hoi-d2y-routed-foot-amplification-s42-20260723`，并跨午夜完成固定的 61,440,000-window /
30,000-update contract。训练 exit code 为 0；final online checkpoint SHA-256 为
`8734431f89cf8739283828d5fb683212ca43143ae3482ad0473f6ed5717eb7a7`。在 2026-07-24 检查完成时，
原计划中的 internal/evaluation lifecycle 均尚未创建 manifest、加载 checkpoint 或启动 GPU，而其
预注册 run id 仍带 `20260723`。为满足所有新 lifecycle 使用真实启动日期且不得复用旧 id 的规则，
只允许以下 identity-only amendment：

1. internal diagnostic run id 改为
   `p1-hoi-d2y-routed-foot-amplification-internal-s42-20260724`；subphase 仍为
   `1B-D2-Y0-internal`；
2. official evaluation run id 改为 `p1-hoi-d2y-native-eval-s42-20260724`；subphase 仍为
   `1B-D2-Y0-eval`；
3. 旧的 `p1-hoi-d2y-native-eval-s42-20260723` 预注册标识在任何 workload 前被 supersede，
   永不创建、复用或绑定 checkpoint；这不是 evaluator/科学 failure，也不产生 retry entitlement。

除上述两个后训练 lifecycle 标识外，D2-X/D2-Y early/mid/final checkpoint、32-sequence/96-window
selection、timestep/noise/dropout pairing、1024 multiplier、bootstrap、official-438/181 mask、所有
success/negative/protection/absolute gates、artifact contract 与停止规则完全不变。不得由此修改训练、
重新生成 D2-X control、选择 checkpoint、启动 CM、HSIPrior 或 Mixer。

#### 2026-07-24 Phase 1B D2-Y completion：surrogate 吸收但 official transfer 失败

D2-Y 已完成唯一预注册的 from-random training、internal mechanism diagnostic 和 official-438
evaluation。训练严格使用 4×RTX 3090、effective batch 2,048、61,440,000 processed windows /
30,000 updates、seed 42 和 final online weights；20 个 cadence checkpoints、80 个 rank RNG
sidecars、finite loss、required gradients、零 overflow、random-init/no-restored-state contract 均通过。
Final checkpoint SHA-256 为
`8734431f89cf8739283828d5fb683212ca43143ae3482ad0473f6ed5717eb7a7`。该 checkpoint 仅是
negative-result artifact，不得选择、resume 或初始化后续 prior/CM。

固定 32-sequence/96-window internal diagnostic 上，final D2-X minus D2-Y routed residual MSE 在
timestep 249/499 的 paired mean 为 `2.85917e-05 / 3.13497e-05`，10,000-replicate bootstrap
95% CI 分别为 `[2.12425e-05,3.62503e-05]` 和
`[2.39972e-05,3.95998e-05]`，两个下界均大于 0，因此 internal gate 通过：1024× amplification
确实被优化器吸收并降低了 teacher-forced surrogate。但 final D2-Y routed/FK gradient cosine 在
timestep 249/499 为 `-0.5001/-0.4522`，说明 late noisy strata 的目标竞争仍然存在。

唯一 official evaluation 使用真实日期 run
`p1-hoi-d2y-native-eval-s42-20260724`、438 sequences、每序列 3 windows、500-step unguided
diffusion，并只读复用 sealed D2-X records。D2-Y 的 MPJPE/end-object/xy/object translation/
foot sliding/contact F1/hand penetration/human penetration 为
`12.1246/4.8506/3.9676/16.3290/0.3572/0.6351/0.2184/3.4353`。D2-X minus D2-Y
foot-sliding paired mean 为 `0.00585353`，95% CI
`[-0.0162119,0.0279164]` 包含 0，official foot gate 失败；D2-Y minus D2-X contact-F1 CI
`[-0.0231844,0.0184971]` 的下界低于 `-0.02`，且 end-object ratio CI
`[1.22629,1.36180]` 的上界超过 1.10，因此 protection gate 也失败。MPJPE、xy、
object-translation、两项 penetration protection、固定 181-sequence mask，以及相对 released
baseline 的全部 absolute diffusion checks 均通过。

严格分类为 `routed-foot-amplification-transfer-negative-stop`：internal surrogate 显著改善，
但未 transfer 为显著 official foot-sliding 改善；end-object/contact 的保护失败作为额外负证据完整
保留，不能 post-hoc 改成 positive 或选择 checkpoint。已验证的是 weight/dilution 机制被成功改变且
official gate 失败；基于证据的推断是 global-mean dilution 并非 D2-X 的充分解释，剩余主要候选为
temporal surrogate 与 nonlinear near-ground metric 的语义差异，以及 routed/FK/object capability
冲突；尚未证明任何新的 official-semantic loss 或 conflict-resolution geometry 有效。没有发现确定
implementation defect，point-estimate foot 改善仍与注册的 sequence-sampling uncertainty 相容。

training/internal/evaluation artifact trees 已由 worker 主动回收到 authority staging，worker 与
authority SHA-256 全部一致，分别为
`177eb44fa53ee46518a714f04ee2fe864aa2a1d755f4377f3fd47fa8e40bf0f8`（112 files /
7,128,453,002 bytes）、
`d57e48d5d4285b4f42ab090e2aef0b1ee641217ea960382ec0b60c6bf0e8d05f`（6 files /
1,493,189 bytes）和
`3e81a59fb1a97e7043e856fda5c502bae3683f24a179b0423143fe910837bdc0`（15 files /
1,631,051 bytes）。compact aggregate 为
`experiments/results/p1_hoi_phase1b_d2y_routed_foot_amplification_s42_20260724.json`，SHA-256
`0bc2fffd4304bb3411176cf355dacddfe731e9f1d46eb01cc9bfefe3c215f875`；完整 handoff 为
`docs/phase_summaries/PHASE_1B_D2Y.md`。D2-Y 至此停止；未经新的 dated plan/registry amendment
和用户确认，不得继续 HOIPrior 机制、选择 checkpoint、启动 consistency、进入 HSIPrior 或 Mixer。

#### 2026-07-24 Phase 1B D2-Z immutable-GT near-ground routed amplification 预注册（plan-only）

D2-Y 已证实固定 1024× routed-foot amplification 能被优化器吸收：timestep 249/499 的
D2-X minus D2-Y routed residual MSE paired bootstrap 95% CI 下界均大于 0；但 official
foot-sliding improvement CI `[-0.0162119,0.0279164]` 包含 0，且 end-object/contact protection
失败。因此“8 slots 在 global mean 中过弱”不再是充分解释。D2-X/D2-Y 的训练 residual 对所有
future frames 一视同仁，而 official foot sliding 只在 previous-frame ankle/toe 相对 inferred floor
低于 `0.08/0.04 m` 时计入 horizontal displacement；uniform amplification 可能把共享模型容量花在
official metric 不计分的非近地 foot motion 上，并加剧 late noisy strata 已观察到的 routed/FK 冲突。

只读 post-hoc sequence audit 仅用于形成新假设，不修改 D2-Y gate 或结论：D2-Y 相对 D2-X 的
438-sequence foot difference 为 239 improve / 199 degrade，mean `0.00585353`、median
`0.0136075`、SD `0.237995`；按 D2-X foot sliding 四分位分层，最低/最高四分位的 mean improvement
分别为 `-0.149215/+0.162256`，13 个 object family 中 6 个为正、7 个为负。foot improvement 与
end-object/contact change 的 Pearson correlation 仅 `-0.0556/-0.0533`。这支持优先检验
near-ground support mismatch，而不是继续增大 multiplier；这些 exploratory 数字不是 selection gate。

D2-Z 是当前 registry 中下一个未占用的 Phase 1B 标识。本 amendment 只预注册科学机制、gates、
stop rule 与 artifact contract；不授权实现、GPU smoke、training、evaluation、checkpoint loading/
selection 或 consistency。具体 workload 日期和 run ids 延迟到用户另行授权后：必须先在执行日运行
`date`，再 append identity-only lifecycle binding；不得预占或复用本 amendment 日期作为未来
workload 日期。预留 run stem 为 `p1-hoi-d2z-immutable-gt-near-ground-gating-s42-<YYYYMMDD>`，
subphase 为 `1B-D2-Z0`。

1. **唯一 manipulated factor。** D2-Z 保留 D2-Y 的 D2-X FK temporal routing、相同 8 个 foot x/z
   residual slots、相同 target 和固定 multiplier 数值，只把 per-element coefficient 从“所有 routed
   entries 均为 1024”改为 immutable-GT binary near-ground gating。对 current sampled future frame
   `t` 的 residual `position[t]-position[t-1]`、joint `j` 和 x/z component `c`：
   `w(t,j,c)=1024` iff
   `y_GT[t-1,j]-floor_GT(sequence)<H_j`，否则 `w(t,j,c)=1`；`H_7=H_8=0.08 m`，
   `H_10=H_11=0.04 m`。First future residual 的 previous frame 仍是 immutable GT history 最后一帧。
   Gate 必须 stop-gradient，不能读取 prediction、diffusion timestep、contact prediction、object、
   text 或 loss；同一 joint 的 x/z 共用 gate。不得加入 official exponential height factor、soft gate、
   zero-velocity target、learned/adaptive threshold、weight normalization 或 multiplier sweep。
2. **GT floor 与 gate provenance。** `floor_GT(sequence)` 必须只由当前 immutable OMOMO snapshot 中
   该 sequence 的完整 30 Hz `human_joints_aligned.npy[start_idx:end_idx]` 计算，精确复用
   `code/eval_metrics.py::determine_floor_height_and_contacts`；该文件当前 SHA-256 为
   `445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547`。Gate height 使用对应
   10 Hz training-window sampled previous frame的 raw aligned GT y；window-local x/z origin/yaw
   不得进入 height/floor。不得使用 generated/predicted floor、per-window minimum、official test
   motion 或任何新增外部资产。232-D tensor、conditions 和 model input/output 不增加 field；
   derived floor/gate 只能作为 loss metadata。
3. **固定 gate preflight contract。** 实现后、任何 GPU 或 reportable manifest 前，必须在 authority
   CPU 上对 train/internal-validation split 全量生成 deterministic gate audit，记录 algorithm/schema、
   ordered sequence/window IDs、source/split/function hashes、per-joint active counts、floor finite/range、
   overall/per-window occupancy、nonfinite count 和 aggregate SHA-256。封存的
   32-sequence/96-window selection（SHA-256
   `30524c88481f6cb81e8063073d510ad01543be92d91eb4ef9b2b8a376cc4fbae`）必须精确复现：
   joints 7/8/10/11 active counts `1096/1081/1211/1232`，各自 denominator `1344`，overall
   active fraction `4620/5376 = 0.859375`，32 个 GT floor 全 finite 且 min/median/max
   `0.0260237856/0.0450961320/0.0530758165 m`。Mismatch、零 active entries、source hash change
   或 nonfinite floor/gate 均为 contract failure；不得通过改 threshold/floor algorithm 修复后沿用同一
   lifecycle。
4. **固定训练 contract。** 若后续另行授权，D2-Z 必须重新随机初始化，initial model-state SHA-256
   仍应为 `ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e`；保留
   232-D representation、16 frames、2 history frames、512-wide/16-head/8-layer Transformer、
   500-step clean-x0 diffusion、OMOMO seed-42 split（SHA-256
   `019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e`）、conditions 和
   FK/object-surface/velocity/goal weights
   `0.3569973401779424/0.4772322188400037/0.1/1.0`。只允许
   `infbagel-4gpu/node01`、4×RTX 3090、batch/GPU 512、accumulation 1、effective batch 2048、
   Adam `(0.9,0.999)`、constant LR `1e-4`、FP32、无 warmup/scheduler/weight decay/clipping/
   AMP/EMA，固定 61,440,000 processed windows / 30,000 updates、32,768 validation windows 和
   3,072,000-window validation/checkpoint cadence。禁止加载 released、author、D2-V、D2-X、D2-Y、
   prior、resume、EMA 或任何 checkpoint。
5. **固定 internal diagnostic。** 完整训练后复用上述 sealed 32-sequence/96-window selection，
   对 D2-X/D2-Y/D2-Z early/mid/final online checkpoints 和 timestep `0/249/499` 使用相同 clean
   windows、noise 与 condition dropout。必须分别报告 near-ground active 与 inactive routed
   residual RMS/per-sequence MSE、gate occupancy，以及 gated routed contribution 对 root、rotation、
   input/output projection、Transformer 的 gradient norm；报告其与 reconstruction、FK、
   object-surface、goal 的 gradient cosine，并报告相对同一 D2-Z prediction 上 uniform-D2-Y routed
   contribution 的 gradient-norm retained fraction/cosine。Internal checks 只验证 gate provenance、
   finite/nonzero gradient 和竞争机制，不作为 checkpoint selection 或调 multiplier 的依据。
6. **固定 official evaluation 与 controls。** 只允许一次 final-online author-native official-438、
   每 sequence 3 windows、500-step unguided diffusion；无 CFG、dynamic perception、guidance、
   CHOIS selection、FID 或 R-precision。Primary candidate control 为 immutable D2-X r1 records：
   checkpoint/aggregate/per-sequence SHA-256 分别为
   `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51` /
   `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
   `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`。D2-Y
   aggregate/per-sequence SHA-256
   `776e6c35acdaa190ffcbab047b170ed4ab559c23f454714c31ad980db4dd8c70` /
   `ea2cde99372392c5f16446708e3acf3789a68be9f1b7cc95134fd45390b12c02`
   仅作为 uniform-amplification mechanism comparator，不重新生成且不取代 D2-X candidate gate。
   统计单位仍为 sequence、bootstrap 10,000 次、seed 42；penetration 绑定相同 181-sequence mask 和
   `2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec` ID hash。
7. **Success、negative 与 absolute gates。** Candidate mechanism gate 相对 D2-X 要求：
   D2-X minus D2-Z foot-sliding difference CI 下界 `>0`；D2-Z/D2-X 的 MPJPE、end-object、xy、
   object-translation、hand/human penetration ratio CI 上界均 `<=1.10`；D2-Z minus D2-X
   contact-F1 difference CI 下界 `>=-0.02`；所有 gate/lifecycle/provenance/finite/normalization/
   history/mask contracts 通过。Absolute gate 保持相对 Phase-0 released baseline 的
   MPJPE/end-object/xy/object-translation/foot-sliding ratio
   `<=1.30/2.00/1.50/1.50/1.10` 且 contact F1 `>=0.60`。D2-Y comparator 的 paired foot、
   contact、end-object 及其余 protection CI 必须完整报告，但不新增 best-of-two selection。
8. **分类、停止规则与 artifact contract。** Mechanism/protection/absolute gates 全通过时分类
   `immutable-gt-near-ground-positive-candidate-stop`；foot gate 失败但 D2-X protection 通过为
   `immutable-gt-near-ground-transfer-negative-stop`，支持 binary support restriction 减少冲突但
   未解决 rollout/predicted-floor 语义差异；foot gate 通过但任一 protection 失败为
   `immutable-gt-near-ground-conflict-negative-stop`；foot 与 protection 均失败为
   `immutable-gt-near-ground-joint-negative-stop`；absolute gate 失败为
   `immutable-gt-near-ground-positive-but-not-effective-stop`；任何 contract failure 为
   `immutable-gt-near-ground-contract-failure-stop`。Artifacts 必须包括 manifest/resolved/
   same-context preflight、GT-floor/gate audit及 hashes、完整 logs/metrics、20 checkpoints 与 80 RNG
   sidecars、resume evidence、internal per-sequence/gradient records、official aggregate/
   per-sequence/bootstrap/gates、normalization/mask/provenance audits、tree hashes 和所有 failure。
   完成一个 fixed-budget run、internal diagnostic 和一次 official evaluation 后必须停止；不得改为
   soft/continuous gate、zero-velocity objective、PCGrad、sampler/contact/penetration intervention，
   不得 sweep threshold/multiplier、选择 midpoint、resume D2-Y/D2-Z、post-hoc 改 gate，或启动
   consistency、HSIPrior、Mixer。即使 positive，也只产生待用户另行确认的 diffusion candidate。

#### 2026-07-24 Phase 1B D2-Z implementation/lifecycle identity amendment

用户已于 2026-07-24 在 plan-only commit
`2b818726093fc15ff819e76fd12119eb329343bf` 后单独授权开始 D2-Z CPU/code implementation。
Authority 在任何 source change 前执行 `date`，得到 `2026-07-24 10:42:57 CST (+0800)`；因此只绑定
以下当前日期 identities：

- CPU gate audit：`p1-hoi-d2z-gate-audit-s42-20260724`，
  subphase `1B-D2-Z0-gate-audit`；
- training：`p1-hoi-d2z-immutable-gt-near-ground-gating-s42-20260724`，
  subphase `1B-D2-Z0`；
- internal diagnostic：
  `p1-hoi-d2z-immutable-gt-near-ground-gating-internal-s42-20260724`，
  subphase `1B-D2-Z0-internal`；
- official evaluation：`p1-hoi-d2z-native-eval-s42-20260724`，
  subphase `1B-D2-Z0-eval`。

本 amendment 只授权实现 gate audit、dataset-derived loss metadata、binary gated velocity reduction、
config/fail-closed contracts、internal diagnostic、target-only evaluator 和 CPU tests，并允许在 clean
implementation commit 后用 authority CPU 执行已绑定的 deterministic gate audit。仍不授权 GPU
smoke、training、official evaluation、任何 checkpoint loading/selection 或 consistency；CPU gate
audit 不得加载 checkpoint。若任一 GPU lifecycle 未在 2026-07-24 启动，必须在未来用户另行授权的
执行日先运行 `date`，append identity-only replacement，并永久禁用未使用的 20260724 workload id。
不得借 identity replacement 修改 D2-Z scientific mechanism、controls、gates、budget 或 stop rule。

#### 2026-07-24 Phase 1B D2-Z CPU/code implementation contract

已按上述授权完成实现，但尚未执行 CPU gate audit 或任何 GPU/checkpoint workload：

- 保持既有 `code/priors/data.py` 与 `code/priors/losses.py` 的封存 SHA-256 分别为
  `62132421b973b1d77c273f80ce48b81507966c0fe75563acd8c1e2158cb54cc5` 和
  `e14cee19e59e9ac698d4d412ccd388f9d0bf903f22e6774b13cc736087d9d1be`；D2-Z 的
  dataset metadata 与 gated reduction 独立实现在 `code/priors/d2z.py`，防止改变 D2-X/Y
  及更早训练路径。`D2ZPriorWindowDataset` 只向 loss batch 增加 `[14,4]` bool gate，不改变
  232-D `x`、condition 或 model API；`d2z_hoi_training_losses` 复用未改变的 D2-X residual/
  FK/reconstruction 逻辑，只替换 velocity scalar reduction。all-false gate 必须与 D2-X
  velocity/total bit-exact，all-true gate 必须与 D2-Y bit-exact。
- `tools/audit_hoi_d2z_gate.py` 是 authority/CPU-only、0 checkpoint load、0 optimizer update 的
  reportable audit；它对 train/internal-validation 的每个完整 aligned GT sequence 调用哈希锁定的
  official floor 函数，封存 source hashes、ordered sequence/window IDs、per-joint counts、
  per-window occupancy、floors 和 sealed 32-sequence/96-window counts。训练 config 中 audit
  path/SHA 当前故意为 `null`；在 clean implementation commit 产生 reportable audit 并另行
  append hash-binding amendment 前，D2-Z validator 必须 fail closed。
- `tools/diagnose_hoi_d2z.py` 固定 D2-X/Y/Z early/mid/final 与 timestep `0/249/499`，输出
  active/inactive per-sequence residual、gate occupancy、root/rotation/input/output/Transformer
  gradient norms、与 reconstruction/FK/object-surface/goal cosine，以及同一 prediction 上
  uniform-D2-Y vs gated-D2-Z gradient retained fraction/cosine；该 artifact 仅验证合同，不参与
  checkpoint selection。
- `tools/run_hoi_d2z_evaluation.py` 以封存 D2-X records 为唯一 primary control，完整报告但不以
  D2-Y comparator 或 internal diagnostic 作选择；六种 classification、Phase-0 absolute gate、
  penetration finite-mask contract 和所有 control hashes 均保持预注册定义。共享 D2-X evaluator
  只新增默认空的 extra-artifact hash hook，D2-X/D2-Y 行为不变。
- Authority 使用规定 Python 完成 282 项 CPU 测试，全部通过；registry validation 仍须在包含本
  amendment 的 logical commit 前再次通过。测试覆盖 gate 的 immutable-previous-frame/strict-
  threshold 语义、active slot 梯度倍率、all-false/all-true loss parity、audit hash/schema/split/
  coverage fail-closed、from-random config/provenance、internal record contract 和全部六种 evaluator
  classification。测试过程中未调用 CUDA，未加载 checkpoint，未创建 optimizer/training result，
  未选择 checkpoint，也未授权 consistency。

下一步只允许先形成 clean implementation commit，然后由 `tools/experiment.py` 启动已绑定的
authority CPU gate audit。若 audit 与 sealed preflight 任一 count/hash 不一致，立即登记
`contract-failure` 并停止；若通过，再以单独 plan/registry hash-binding commit 固定 audit
artifact 和 worker-local path。该步骤本身仍不授权 worker fast-forward、GPU smoke、training、
internal/official evaluation、checkpoint load/selection 或 consistency。

#### 2026-07-24 Phase 1B D2-Z CPU gate-audit r0 failure / r1 amendment

Implementation commit `3034510418fdc76e334cb8643c74414ded045726` clean 后，authority 通过
`tools/experiment.py` 启动 `p1-hoi-d2z-gate-audit-s42-20260724`。该 run 完成 manifest、
resolved-config 和 CPU preflight 后，在首次 full-split window gate shape validation 发现
`human_joints_aligned.npy` 的锁定 raw shape 是 `[T,28,3]`，而新 helper 错误限制为
`[16,24,3]`，因此在 0 checkpoint load、0 optimizer update、0 CUDA call 时失败；未写出
`gate_audit.json`，无科学 counts/result。失败 manifest SHA-256 为
`ea5dd7473989c50de9fedf8268aa7ba2133dac514755b5fed50c5be8da33083d`，failure metrics 为
`a5e8f25dad08a481a772b2b67a5e9ec5d5129375916e7b1ff74b319c0839b89e`，preflight 为
`fdee59171a6c2c868ac4abecb2302595ee79436ca41d69ad43b40e1122d36b02`，resolved config 为
`e8d25543975f0fd2182ada97e1bd10091549d4792bf915428c9f0dfd2fe756ed`，4-file failure tree
SHA-256 为 `11553516ef660118e2e66b8ca9eb0f277c34aa21af10c8fca782a15b4b6cdc9f`。原 run id 永久禁用，
不得把该 operational failure 混入 D2-Z 科学结论。

Authority 在任何修复 source change 前再次执行 `date`，得到
`2026-07-24 11:10:24 CST (+0800)`。允许唯一的 fail-closed 修复：gate helper 的输入合同从错误的
exact `[16,24,3]` 改为锁定 raw aligned source 的 exact `[16,28,3]`；foot joints
`[7,8,10,11]`、previous sampled frame、thresholds、floor、binary comparison、counts、
selection 和所有科学机制均不变。修复与 regression tests 必须单独提交；随后只允许使用新
`p1-hoi-d2z-gate-audit-r1-s42-20260724`、subphase `1B-D2-Z0-gate-audit-r1` 从头重跑 CPU
audit。r1 仍为 0 checkpoint/GPU/training；若任何 sealed count/floor/hash 不匹配，登记失败并停止。
本 amendment 不授权 worker sync、GPU smoke、training、internal/official evaluation、
checkpoint load/selection 或 consistency。

#### 2026-07-24 Phase 1B D2-Z CPU gate-audit r1 completion / hash binding

Shape-fix commit `1bb5ff5cd71eb67c0697094f7c89b0aed9c0643f` clean 后，reportable authority
CPU run `p1-hoi-d2z-gate-audit-r1-s42-20260724` 完成。Sealed 32-sequence/96-window
preflight 精确复现：

- joint `[7,8,10,11]` active counts 为 `[1096,1081,1211,1232]`，每 joint denominator
  `1344`，总计 `4620/5376=0.859375`；
- floor min/median/max 为
  `[0.026023785583674908,0.04509613197296858,0.05307581648230553]`；
- selection SHA-256 为
  `30524c88481f6cb81e8063073d510ad01543be92d91eb4ef9b2b8a376cc4fbae`；
- independent loader replay 再次得到相同 96-window counts，canonical payload replay 与 artifact
  内的 `30add2511afbc651546a6e5d038532455a22b82fc02b2763e01bd8ebc9b911f4`
  相同。

Full train split 为 4,088 sequences / 568,486 windows，active counts
`[6501264,6638139,7366360,7473398]`、total `27979161/31835216=0.8788745457`；
internal-validation 为 216 sequences / 29,382 windows，active counts
`[333433,337070,378355,382271]`、total `1431129/1645392=0.8697799673`。两边
nonfinite floor/gate 均为 0，zero-active windows 均为 0；train/internal fully-active windows
分别为 139,629/7,000。该高覆盖率是预注册 binary threshold 在真实 split 上的已验证性质，不授权
改 threshold、soft gate 或 weight normalization。

Authority immutable artifacts：

- gate audit：
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2z-gate-audit-r1-s42-20260724/gate_audit.json`，
  SHA-256 `d56f1cbc5297b82d768cd396ab1a49c6e33d4101d156c0375501bf32ae055faa`；
- manifest SHA-256 `2e522c3b15b6f255d76a0d9dd8458bf7fa103b2066525efb5b10818e89032637`；
- preflight SHA-256 `cc0e88ac024a6f4ba5a99e2c81d3eef04e710f95376d9f1fc9424a717a6fa643`；
- resolved-config SHA-256
  `d5e73b58725d94bb17ec8fab5fcf3029681831a97138d2538d109b3bb88e5934`；
- 4-file completed artifact tree SHA-256
  `2c3db75fd2227974ecaf939a1e22dd6cccd6ebaf739ee424d1c8ce6aa55cdbca`
  （69,313,536 bytes；manifest 内嵌 sealed metrics）。

D2-Z training config 现在只绑定 portable checkout-local artifact path
`${repo_root}/results/experiments/p1-hoi-d2z-gate-audit-r1-s42-20260724/gate_audit.json`
与上述 gate-audit SHA；worker 若以后另行获准，必须主动复制该 immutable file、校验 SHA，并在
同一 committed Git object 上通过 dataset/config preflight。当前仍不授权 worker sync、artifact
transfer、GPU smoke、training、checkpoint load、internal/official evaluation、checkpoint
selection 或 consistency；本 session 在 hash-binding commit 后停止等待用户确认。

#### 2026-07-24 Phase 1B D2-Z worker/GPU execution authorization amendment

用户已于本 amendment 前明确授权继续执行 D2-Z worker sync、artifact transfer、GPU smoke 与固定
正式训练，并要求正式训练稳定后停止轮询、待用户通知训练结束。Authority 与 worker 在任何本次
文件修改或 GPU workload 前分别执行 `date`，均得到
`2026-07-24 11:23:24 CST (+0800)`；authority 为 clean
`phase/01b-hoi@4d087689d38d338d4a62abffa70bfb77ab46b0fe`，worker 为 clean
`phase/01b-hoi@2cd4991fa3a39f0b6b0d98912a710a7d4c70a964`。worker 的旧 HEAD 是已封存结果
提交，不是 corruption；必须在本 amendment 形成新的 clean authority logical commit 后由 worker
主动 fast-forward 到完全相同的 Git object，worker 不得编辑 source。

本次只新增 execution authorization，不改变 D2-Z 的 scientific mechanism、threshold、multiplier、
loss reduction、representation、architecture、conditions、split、seed、effective batch、budget、
evaluator、gates、classifications 或 stop rule。授权范围固定如下：

1. worker 主动拉取 exact authority commit，并主动复制 authority 的 immutable
   `p1-hoi-d2z-gate-audit-r1-s42-20260724/gate_audit.json` 到 config 已绑定的
   checkout-local ignored path；必须校验 SHA-256
   `d56f1cbc5297b82d768cd396ab1a49c6e33d4101d156c0375501bf32ae055faa`。
2. worker 使用 `/home/yujinlun/data/envs/infbagel/bin/python`、`ROOT_DIR` 等于 checkout root、
   `INFBAGEL_WORKER_EXPERT=hoi`，完成 role-applicable CPU tests、registry validation、fully
   resolved Hydra config 与同一 GPU execution context 的四卡 idle/preflight archive。任一
   commit、audit、data、asset、config、idle 或 unresolved-interpolation contract 失败即停止，
   不得自由重试 formal run id。
3. 允许一个独立 operational GPU smoke：
   `p1-hoi-d2z-gpu-smoke-s42-20260724`、subphase `1B-D2-Z0-gpu-smoke`。它只在 worker GPU 0
   以 seed 42、随机初始化、真实 D2-Z train windows 和 production model/diffusion/loss path执行
   一次 forward/backward；必须验证 audit hash、binary gate 同时含 active/inactive entries、全部
   loss finite、motion input/output/Transformer 关键 parameter gradient finite 且 nonzero，以及
   peak allocated/reserved memory。它不得创建 optimizer、执行 update、写 checkpoint、加载任何
   checkpoint 或形成科学 selection；无论成功失败都必须保留 manifest、resolved config、
   preflight、metrics 和 log，原 smoke id 不得复用。
4. smoke 通过后允许启动已绑定的唯一 formal training
   `p1-hoi-d2z-immutable-gt-near-ground-gating-s42-20260724`、subphase `1B-D2-Z0`。必须使用
   4×RTX 3090、batch/GPU 512、effective batch 2048、seed 42、随机初始化、61,440,000 windows /
   30,000 updates 和全部既有 exact config；禁止设置或加载 init/resume/weight-init/EMA/released/
   author/D2-V/D2-X/D2-Y/prior checkpoint。通过 `tools/experiment.py start` 创建 reportable
   manifest，并在 worker-owned persistent `tmux` 中运行，不依赖反向 SSH 存活。
5. 只监测 initial stability interval：resolved/preflight/manifest 通过，loss 与所需 gradients finite/
   nonzero，无 overflow，四卡显存满足既有 headroom contract，并产生首个 3,072,000-window
   checkpoint 及四个 RNG sidecar。对该完整原子 checkpoint 做只读 schema/hash/contract
   inspection 作为 resumable evidence，不实际 resume、不创建第二训练 lineage。满足后记录吞吐和
   ETA，停止连续轮询；训练结束、artifact recovery、internal diagnostic 与 official evaluation 等待
   用户再次通知。若此前失败，保留失败并停止，不得复用 id。

本授权不允许 checkpoint selection、D2-Z/D2-Y resume、修改 gate 或旧 gate、internal/official
evaluation、consistency distillation、HSIPrior 或 Mixer。Formal training 即使完成也只产生尚未
评估且不可选择的 D2-Z artifact；后续 lifecycle 仍受原预注册 stop rule 约束。

上述 operational smoke 已实现为 `tools/smoke_hoi_d2z.py`（SHA-256
`504fa048b142bb95ed13426ae7693f0801a9ce3050dd0ccb011dc8a5b16e68e0`）。工具只构造随机
HOIPrior、读取真实 D2-Z batch、执行 production diffusion/loss forward/backward 并原子写 metrics；
source 中不存在 optimizer、`step`、`torch.load` 或 `torch.save`，因此不能更新参数或读写
checkpoint。Authority Python 完成 py_compile 与 15 项 D2-Z tests，全部通过；截至该验证时
GPU smoke/formal training 均未启动。必须先提交该工具、tests 与本记录，再由 worker 主动
fast-forward 到 exact commit 后执行。

#### 2026-07-24 Phase 1B D2-Z GPU smoke r0 failure / r1 amendment

worker 在 exact commit `875a544679c0de7b493d9aa3c4ec55b2c5b547f0` 上通过 resolved config、
四卡 idle preflight 与 `tools/experiment.py start` 启动
`p1-hoi-d2z-gpu-smoke-s42-20260724`。r0 在任何 model forward/backward、optimizer、update 或
checkpoint 操作前 fail-closed：工具先创建 DataLoader iterator/读取 batch，再构造随机 model；
PyTorch DataLoader iterator 会消耗 global Torch RNG，因此 observed initial model SHA-256 为
`c1e9ce860311307f0844d4d31daa5b431583b6781cbddf2bf58f9a653b85554c`，不等于正式训练 rank-0
锁定的 `ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e`。这是独立 smoke
工具的执行顺序 defect，不是 D2-Z dataset/loss/trainer/scientific mechanism defect；formal training
尚未启动，原 smoke id 永久禁用。

r0 manifest/metrics/resolved/preflight/run-local-registry SHA-256 分别为
`e5771a7a02fc511f3f3f2380f289883989848ca15eff866428f982975f63af92` /
`b8ee77f88d74e993c7233c1c4b0b90daea1341537716582f59c892c09fdb25eb` /
`b81490a2941193679b9d9c1b9e85713884ef27ec2ed8076bb06c39cb6d202c26` /
`cfa94d0e598b4ce58b006b771c79197fa0b2df7ca52ab9fe8376ac98d3bcd1c7` /
`2aff76bce456c5ef1da963fe4648b529212ab79a0ecec3c0a1e259b132b79374`。7-file tree
SHA-256 为 `cdd6934b5cb68fc42a71feee1f165695256376b08044ff3e2ced76806193edd3`，worker 主动
回收到 authority staging 后 tree hash 完全一致。

Authority/worker 在任何 fix 前再次执行 `date`，分别得到
`2026-07-24 11:36:41/11:36:42 CST (+0800)`。只允许一个 execution-order fix：在 smoke 工具中
先按 seed 42 构造并校验随机 model，再创建 DataLoader iterator/读取 batch；production trainer、
D2-Z dataset/loss、config、gate、timesteps、batch、gradient checks 和所有科学合同不变。Fresh r1
run id 固定为 `p1-hoi-d2z-gpu-smoke-r1-s42-20260724`、subphase
`1B-D2-Z0-gpu-smoke-r1`。必须先提交本 failure/amendment，再实施单一 fix、更新 tests/tool identity
并形成新 clean commit；worker 主动 fast-forward 后从头生成 r1 resolved/preflight/manifest。若 r1
失败，保留并停止 formal training；若通过，才按已授权合同启动唯一 formal run。

r1 tool fix 已严格只把 model construction/hash validation 移至
`raw_batch = next(iter(loader))` 之前，并将 tool identity 改为 fresh r1；修复后工具 SHA-256 为
`9279665f727abff959d0f069d6509273ec031ac7d196fb772e5d6cd023b57ed2`。Authority py_compile
与 15 项 D2-Z tests 全部通过；新增 regression 明确断言 model hash 语句位于 iterator 之前，并继续
静态禁止 optimizer step、checkpoint load/write。截至该验证时 r1 GPU smoke 与 formal training
均未启动。

r1 GPU smoke 已在 exact commit `f67a6437f1ca261ad78f5a8eceab6daabaeb40b5` 上完成：
initial model SHA 精确为 `ad6980ce...`，真实 8-window batch 的 gate 为 441 active / 7 inactive，
全部 loss finite；motion input/output、Transformer、predicted root translation 与 joint rotations
gradient norm 分别为 `55.2236/107.2537/44.4260/1.08801/0.167582`，均 finite/nonzero。GPU 0
peak allocated/reserved 为 `249,652,224/299,892,736 bytes`，headroom
`24,996,151,296 bytes`；optimizer created/updates、checkpoint loads/writes 均为 0。

r1 manifest/metrics/resolved/preflight/run-local-registry SHA-256 分别为
`d0c224fb9af233ceb3a88722ef0c8704cb6568247061279d5be6350e4dba8fbc` /
`ff66f45ab0fcb72c06c1e8664404a11e3d7950bc57d1e7b766ef0bd98a20d2a4` /
`b81490a2941193679b9d9c1b9e85713884ef27ec2ed8076bb06c39cb6d202c26` /
`fe0e3aab513cbec61d2e6b53872ba4b251ce47fc98e3bc7ea70bd6aaaab60230` /
`ebcd17935cdd28a6165e333d858db94e5e7e24ac33401d9cf3d3cb6ca74a4e89`；7-file tree
SHA-256 `3a36211aaa3143a965828f37c2cbbc53872188162905dd93fe3017f55a12f148`
已由 worker 主动回收且双端一致。Operational GPU smoke gate 通过；只有在本 completion metadata
形成新 clean commit、worker 主动 fast-forward，并重新生成 formal-run resolved config 与四卡
idle preflight 后，才允许启动唯一 D2-Z formal training。

#### 2026-07-24 Phase 1B D2-Z formal training completion

worker 在 exact commit `2634cea35e86cd054c9283fdfddb89fe507dc066`、clean
`phase/01b-hoi` 上完成唯一
`p1-hoi-d2z-immutable-gt-near-ground-gating-s42-20260724`。训练从随机初始化
`ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e`
开始，未加载 released/author/D2-V/D2-X/D2-Y/prior/resume/EMA 或任何 checkpoint state；
weight-initialization record 的 source 为 null、restored components 为空，所有旧 optimizer/EMA/
scheduler/scaler/RNG state counts 均为 0。

固定 61,440,000 processed windows / 983,040,000 frames / 30,000 optimizer updates 完整结束，
exit code 0。Loss 与 required gradients 全程 finite/nonzero，AMP overflow 0；4×RTX 3090、
micro-batch/GPU 512、effective batch 2048、Adam `1e-4`、FP32 和所有 exact contracts 未变。
训练 wall time `19,319.6752 s`，throughput `3,180.1777 windows/s`，每 rank minimum memory
headroom `21,208,694,784 bytes`。Mean training total/velocity 为
`0.0478505/0.0273295`；final 32,768-window validation total/velocity 为
`0.0502446/0.00381642`，finite。20 个 3,072,000-window cadence checkpoints 与 80 个 RNG
sidecars 完整，逐 checkpoint 实际 SHA/bytes 与 metrics 全部一致；terminal online checkpoint
SHA-256 为 `44c1ff8c8cf4abc2c7312923f64183e1a4a307166d187c9fcaff03abdcc162b6`。
该 checkpoint 只是未评估 artifact，尚未选择。

Manifest/metrics/resolved/preflight/run-local-registry/resume-evidence SHA-256 分别为
`39a400060c01056f03c03eff28a6ba83f0e0b88b520394a43a72eaf0903b28df` /
`84b682c4ec78ce80402538c6304a419f8bfcf879b7bf3163ced68d8032d47d09` /
`b81490a2941193679b9d9c1b9e85713884ef27ec2ed8076bb06c39cb6d202c26` /
`a94293326f41243b89a9911d3d1c94a34755bd41cc6d70baf8a0f2c2dd83c38b` /
`f5e2848696009542ad50353de8d572c9498c46945c8d4a69bf438410c8659188` /
`fed416229e281f2097a4a7a5ddb232f920f4a1fd8dd616bbba1a1cec2e43330b`。
完整 113-file / 7,127,278,269-byte training tree 已由 worker 主动回收到 authority staging，
双端 SHA-256 均为
`41de8a4a2b94b225d82d628ca3d074408b33619550f8809e1d6576ef2b1f4726`。

Training lifecycle gate 通过，只授权使用 fixed early/mid/final online checkpoints执行已预注册的
internal diagnostic，然后对 fixed final online checkpoint 运行唯一 official evaluation。不得根据
training/validation loss 或中间 checkpoint 选择模型，不得 resume、增加预算或启动 consistency。

#### 2026-07-24 Phase 1B D2-Z internal diagnostic r0 failure / r1 amendment

首次 fixed internal run
`p1-hoi-d2z-immutable-gt-near-ground-gating-internal-s42-20260724`
在 exact commit `0b94844c58bb2a39c6606acae3aea83c43e4ee9b` 上通过 clean-worktree
preflight 并开始读取预注册的 9 个 fixed D2-X/D2-Y/D2-Z early/mid/final online checkpoints，
但在生成任何 scientific diagnostic metrics 前失败。确定原因是 reporting contract 错误地要求
每条 sealed sequence 的 active 与 inactive residual strata 都非空。固定 32-sequence/96-window
selection 的 active/inactive entries 为 `4620/756`；所有序列均有 active support，但 sequence
indices `[2,5,16]` 是合法的 fully-active sequences，其 inactive count 为 0，因此对应
per-sequence inactive MSE 数学上未定义。该情况未违反既有 gate audit（它只要求全 selection
active 非零，并明确允许 fully-active windows），不构成 D2-Z training、checkpoint、gate 或
scientific mechanism defect。

r0 必须永久保留为 failed，禁止复用 run id。其 manifest/failure-metrics/preflight/
run-local-registry/log/exit-code SHA-256 分别为
`ca44be99e85c90f2f437bc4236566141e3c01f092df9e80b442c5f0571d6263a` /
`39f77a06082696572920003c8e7362e0b9e6d34dc0110dabb000fff9e06152c4` /
`5e73c4f031a190b4d75e6aff8df66df895ce8a78805751118d80aec7d8dd8847` /
`cc9173f31684ed7ce26a978753b6a1ae29289013bd75b2892ac3084e055790a0` /
`8784e793ddedf37cef059bcfb38addd2bb70e11e8633267c266ae47df66b893f` /
`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`。
完整 6-file / 69,454-byte tree 已由 worker 主动回收到 authority staging，双端 SHA-256 均为
`239fb2bf4a4ce7ff456bea8a0ebd265bcde8b12cef14a8ae0e81ba081b472544`。
r0 未创建 optimizer、未训练/写 checkpoint、未使用 official-test sequence、未选择 checkpoint，
也未启动 consistency。

只允许一个 fresh replacement：
`p1-hoi-d2z-immutable-gt-near-ground-gating-internal-r1-s42-20260724`
（subphase `1B-D2-Z0-internal-r1`）。唯一 implementation delta 是：保留每条 sequence 的 exact
support count；count 大于 0 时按原公式报告 per-sequence MSE，count 等于 0 时在 JSON 中报告
`null` 和 count `0`，不得填 0、插值、删除 sequence 或改变 aggregate denominator。Finite
validation 只适用于 count 大于 0 的 strata；全 selection 的 active/inactive aggregate 仍必须
各自非空且 finite。必须增加 regression 覆盖 zero-support 的显式未定义语义。

r1 的 selection SHA、32 sequences/96 windows、全部 9 个 checkpoint paths/SHA、timesteps
`[0,249,499]`、paired noise/dropout、所有 residual/gradient/cosine computations、D2-X/D2-Y/D2-Z
比较、bootstrap、seed 42、GPU device 和 non-selection 用途保持不变。不得借此修改 training、
checkpoint、gate、official evaluator、candidate gate 或 classification。必须先提交本
failure/amendment，再修改工具并形成新的 clean logical commit；worker 主动 fast-forward 后通过
fresh preflight/manifest 执行 r1。只有 r1 完整封存后，才可运行已预注册的唯一 official-438
evaluation。

r1 tool fix 已严格实现上述契约：undefined entries 在内部 tensor 中保持 `NaN` 语义，仅在
serialization 时根据 exact zero count 转为 JSON `null`；count-positive entries 沿用原除法，
aggregate active/inactive RMS 与 denominator 未变。`diagnostic_summary` 现在逐项验证
`null iff count==0`、所有 defined MSE finite、两类 aggregate support 非空，并继续禁止 selection。
工具 identity 已更新为 fresh r1，tool/test SHA-256 分别为
`02799a5037fb8f0884eebabf0fea24e70a1f17dd6d3c572c2b10862a2c2a17cf` /
`acf57c66be14b3e5e0f816cbada0e62adcf0c8e3d1b195d4d2154cf9bd600bfd`。
Authority py_compile、16 项 D2-Z tests 与 registry validation 均通过；截至本验证时 internal r1
尚未启动。

internal r1 已在 exact commit `ff5148a7040cc6c9679393557a395e3f147a43b8`、clean worker
上完成，exit code 0、runtime `12.1607 s`，固定 9 checkpoints × 3 timesteps 的全部 required
records/scalars 通过合同。3 个 fully-active sequences 在每个记录中原位保持
inactive `null/count=0`，其余 defined MSE 及 aggregate active/inactive RMS、gradient norms/
cosines 均 finite；selection use、official-test sequences、optimizer/update/checkpoint write/
selection、consistency 均为 0/false。Manifest/metrics/preflight/run-local-registry/log/exit-code
SHA-256 分别为
`d5f3d3bd04bdd290ec5dbd599d706a99daac78e1ec51259b88f9280fc9a0043d` /
`0540afa33b485f3a893973d827fe0c48bfca08df3e0b3fdd54fa1f14ce9256e3` /
`abca0349a533bf288f5a6ef80bfee447423c13563a7c9875ff2d098959e5626b` /
`b09e80dcc370f31a2aa43f9a39d571ee01477bc5999236a99ea002dd0f1f4f50` /
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` /
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。
完整 6-file / 515,725-byte tree 已由 worker 主动回收，双端 SHA-256 为
`039894526a9044865ea0fcfbdee1ba9c51f148a74860048ba16fdd6b1f31e960`。

该 non-selection diagnostic 给出三项描述性证据。第一，final D2-X/D2-Y/D2-Z 的 active routed
RMS 在 timestep `0/249/499` 分别为
`0.007179/0.005561/0.005675`、`0.008951/0.007029/0.007054`、
`0.010113/0.007667/0.007912`；D2-Z 保持 D2-Y 的大部分 surrogate gain，但没有进一步优于
D2-Y。第二，D2-Z final gated-vs-uniform gradient cosine 为
`0.9949/0.9592/0.9483`，norm retained fraction 为 `0.7767/0.7833/0.8587`；由于 sealed
selection 的 active occupancy 达 `85.94%`，binary gating 主要缩小 uniform signal 而没有形成
明显不同方向。第三，D2-Z gated routed gradient 与 FK 的 cosine 为
`0.6908/-0.5110/-0.1521`，在两个 noisy strata 仍冲突；timestep 249 的 gated all-parameter norm
`0.01431` 也远小于 reconstruction/FK/object-surface norms `0.79246/0.22699/0.37023`。
这些只支持“near-ground restriction 是相对温和、且 noisy-step FK conflict 未消失”的推断；
不能预测或替代 official foot-sliding paired CI，也不能选择 checkpoint。

official evaluator 在实现时绑定的是已失败的 r0 internal identity，故唯一允许的 pre-evaluation
contract amendment 是把 expected internal run id 改为上述 r1，并使 validator 与 sealed r1
schema 一致：逐 sequence 要求 `count==0 iff MSE is null`，count-positive MSE finite，active/
inactive aggregate 均非空；其余 checkpoint/control hashes、target generation、bootstrap、
candidate/protection/absolute gates、classifications 与 stop rule全部不变。必须先提交本 completion/
binding amendment，再修改 evaluator/tests 并形成 clean commit；worker 主动 fast-forward 后才可
创建原已绑定且尚未使用的 `p1-hoi-d2z-native-eval-s42-20260724` manifest。只允许该一次
official-438 run。

evaluator binding fix 已只更新 expected internal run id/SHA，并逐 sequence 验证上述
`null/count` schema 和 aggregate nonempty；target generation、controls、statistics、gates 与
classification code 未变。Evaluator/test SHA-256 分别为
`b02954b3f72fe27162fdcdc0fb56d5bc6e119bdd8fdca58408151d2256dcc8fa` /
`23433022b5b990a8f85f864691d8e5d2a690969fc1a1646053e599ef5831ab40`。
Authority py_compile、17 项 D2-Z tests 和 registry validation 均通过；截至本验证时 official
evaluation 尚未启动。

#### 2026-07-24 Phase 1B D2-Z official completion / joint-negative stop

唯一 `p1-hoi-d2z-native-eval-s42-20260724` 已在 exact commit
`38d7e409a1b6208049d1b4ce358eacfef0dc9f3a`、clean worker 上完成，exit code 0。协议严格为
official 438 sequences、每 sequence 3 windows、500-step unguided diffusion、fixed final online
D2-Z checkpoint；D2-X/D2-Y records 均原样复用，D2-Y 只作 non-selection comparator。

D2-Z 的 MPJPE/end-object/xy/object-translation 为
`12.2655/4.4567/3.8216/16.3945`，foot sliding/contact F1 为
`0.363433/0.630798`，hand/human penetration loss 为 `0.219640/3.447872`。Primary D2-X minus
D2-Z foot-sliding paired mean 为 `-0.000423015`，10,000-replicate sequence bootstrap 95% CI
`[-0.0221524,0.0209344]`，故 registered foot gate 失败。D2-Z minus D2-X contact-F1 mean/CI 为
`-0.00662759/[-0.0274279,0.0137967]`，下界低于 `-0.02`；D2-Z/D2-X end-object ratio CI 为
`[1.12841,1.25042]`，上界高于 `1.10`。其余 MPJPE、xy、object-translation 与两项 penetration
protection 通过，固定 181-sequence finite mask 精确匹配。所有 Phase-0 released-baseline
absolute diffusion checks、provenance/lifecycle、normalization 与 finite contracts 通过。

因此严格 classification 为 `immutable-gt-near-ground-joint-negative-stop`：official foot gate
与 protection gate 均失败。不得因 absolute checks 或任一 point estimate 改写分类；final
checkpoint 仍未选择，consistency 未授权/未启动。

Evaluation manifest/metrics/aggregate/per-sequence/resolved-target/resolved-lifecycle/preflight/
run-local-registry SHA-256 分别为
`c134947136eee8a867222c55584ad744f6a8cffa72742b39a77980f541e50c6e` /
`c20738824f0475294e42551121fd7796c2041fd48949e4630e012ad2d4959ae3` /
`fb58a5ab3bd5ad0336ce02ff9a15cd7d97af8446599b147c9e2c806208a56162` /
`9f0f0e65bd0eaa4fe3ec1f495f6e4a4489c88d842256dccc3a6b9b57a1e9113f` /
`f8f1350678f3a706d6763fff2094826c618e10956f8714002dd0cb0a974efa3e` /
`86610e7c646d08474072b67826ad2a7268b7c8a20cefe93e217f446da3f244ed` /
`981c21f0d565e99ee1b9d9ae5e52667de650ddf0986e6f815448edf1a151b4af` /
`80c908260c73d1c5e27e2f4da4c76e8927575ae329be71d0d0da9171b080c9b9`。
完整 15-file / 389,634-byte tree 已由 worker 主动回收，双端 SHA-256
`9613bb8762dc1ba67e29c068dee966461cfa4ac4284a2d84c4dc671625e13bfe`。

Compact aggregate
`experiments/results/p1_hoi_phase1b_d2z_immutable_gt_near_ground_gating_s42_20260724.json`
SHA-256 为
`c2a0ff494784fac6d42485d189646b8b9618205f8fc8ed6608270ff871c16af4`；
handoff 为 `docs/phase_summaries/PHASE_1B_D2Z.md`。D2-Z 到此停止。任何下一 HOIPrior 机制必须
在新 session 先只读审计、核验 registry 下一 ID，再作 dated plan/registry preregistration；
不得选择或 resume D2-Z、改变旧 gate、重跑 official evaluation、启动 consistency、HSIPrior 或
Mixer。

#### 2026-07-24 Phase 1B D2-AA Table-5 completeness evaluation 预注册

论文 Table 5 的完整 HOI 列为 `Te/Txy/FS/Rprec/FID/Cprec/Crec/Cf1/C%/Pbody/MPJPE/`
`Troot/Tobj/Oobj/FPS`。D2-V/X/Y/Z 的封存 official-438 native evaluations 已覆盖其中除
`Rprec/FID` 外的质量列，但其 preregistered target overrides 明确设置
`save_chois_eval_npz=false`、`fid_rprecision_used=false`；既有 generation FPS 又是
batch-438 throughput，不能冒充论文未充分说明的 batch-1 timing。另一方面，仓库锁定的
CHOIS release evaluator 输出的是 `R-Precision@1/2/3`，论文只给一个未进一步定义的
`Rprec` 标量；D2-AA 不得 post-hoc 把任一个 rank 指标冒充论文标量。用户要求补齐这些
缺失指标后，registry 中下一个未占用 Phase 1B 标识经全量检索确认为 D2-AA。

1. **唯一目的与固定候选。** 唯一 reportable umbrella run id 为
   `p1-hoi-d2aa-table5-completion-s42-20260724`、subphase `1B-D2-AA0`、seed 42，只允许在
   `infbagel-4gpu/node01` 单卡顺序运行。固定、对称地包括 D2-V/X/Y/Z 四个 final-online
   checkpoints，SHA-256 分别为
   `e0705681bbaeed40d353494852494d8b7bdaf4d32da92368c0d2ceedea4c01a4` /
   `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51` /
   `8734431f89cf8739283828d5fb683212ca43143ae3482ad0473f6ed5717eb7a7` /
   `44c1ff8c8cf4abc2c7312923f64183e1a4a307166d187c9fcaff03abdcc162b6`。
   四者全部运行，不得根据已有 native 点估计只补有利候选；不加载 optimizer、EMA、RNG、
   released/author checkpoint，也不训练、resume 或写 checkpoint。
2. **Table-5 quality/export protocol。** 每个 candidate 从固定 checkpoint 重新执行一次
   official 438 sequences × 3 windows、500-step unguided production diffusion，online weights、
   seed 42、无 CFG/dynamic perception/guidance/scene/CHOIS selection，并只新增
   `save_chois_eval_npz=true` 的 matched 438-pair、126-frame、Z-up `global_jpos[T,24,3]`
   export。重新计算的全部 native aggregate/per-sequence 值必须与各自封存 official result
   在浮点容差内一致；它们只作为 deterministic regeneration audit，绝不替代原记录、重开旧 gate
   或成为第二次 selection evaluation。四份 GT export 的 ordered IDs、shape/content tree hash
   必须彼此相同并匹配 Phase-0 GT contract；否则 contract failure。
3. **锁定 CHOIS 指标。** 对四份 prediction/GT export 分别使用 pinned CHOIS commit
   `8ec585aa0200fd2a890ffb12897bcf69ae719463`、text-to-motion commit
   `72df96ec453edea2fbe9603b1d58a955eaf71636`、feature checkpoint SHA-256
   `a125bc15ffd9772686737111c7501ecee0a2d8571d9aca348ec1195ddef78775` 和 Phase-0
   normalization/annotation/GloVe assets。完整报告 FID、Matching Score、
   R-Precision@1/2/3 与 Diversity；必须同时报告 upstream `drop_last=true` 后实际进入 embedding
   指标的 sequence 数与被丢弃 ID，不能只写 export count 438。R-Precision@1/2/3 与
   Matching Score 使用 10,000-replicate、seed-42 sequence bootstrap；非加性 FID 使用固定
   200-replicate paired sequence bootstrap，仅作不确定性描述，不设 capability gate。不得改变
   upstream batching、normalization、feature network 或 retrieval definition来追求论文数值。
4. **本地 efficiency 补充。** 四个 checkpoint 还要各执行一次相同确定性首序列的 warmup 后
   batch-1、3-window timing-only workload，CUDA 在测量边界同步；报告 generation seconds、
   generated frames、FPS 与 end-to-end seconds。该本地 batch-1 数值与 batch-438 throughput
   分列，并与论文 Table-5 FPS 分列引用；论文未公开完整 timing protocol 时不得声称严格同协议。
   Timing subset 的质量指标不得并入 official-438 aggregate。
5. **成功/失败与停止。** 只有 checkpoint/source/data/evaluator/assets hashes 全部匹配、
   clean worker manifest/preflight/resolved configs 完整、四个 candidate native regeneration
   audit 通过、每组 438 matched exports finite/同 ID、四组 CHOIS 点估计和注册 uncertainty
   finite、四组 batch-1 timing finite，才分类
   `table5-completion-pass-nonselection-stop`；任一失败分类
   `table5-completion-contract-failure-stop` 并保留全部 partial artifacts，不得复用 run id。
   D2-AA 没有模型质量 success gate，不排序或选择 checkpoint，不改变
   D2-V/X/Y/Z classification，不授权 consistency、HSIPrior、Mixer 或任何训练。
6. **Artifact contract。** 保留 umbrella manifest、fully resolved lifecycle/candidate/timing/
   CHOIS configs、same-context GPU preflight、commands/logs/exit codes、checkpoint和封存
   aggregate/per-sequence hashes、四组 native regeneration aggregates/per-sequence、438-pair
   NPZ trees/IDs/hashes、CHOIS point/bootstrap records、batch-1 timing、environment/dependency/
   hardware hashes、run-local registry、完整 tree hash及所有失败。大 NPZ 留在 worker/authority
   staging，不进入 Git；Git 只记录 compact aggregate、summary 与 hashes。

#### 2026-07-24 Phase 1B D2-AA CPU/code implementation contract

D2-AA 实现只新增 non-selection orchestration 与 opt-in CHOIS reporting，不修改
`code/test_infbagel_hoi.py`、`code/eval_metrics.py`、production sampler、model、loss、checkpoint
或旧 evaluation artifacts。`tools/run_hoi_d2aa_table5.py` 固定四个 final-online checkpoint、
封存 aggregate/per-sequence hashes、438×3/500-step export、相同首序列 batch-1 timing、source/
resolved/preflight/dirty-worker contracts，任一候选失败即保留 partial artifacts 并 fail closed。
Tool SHA-256 为
`5727c2a8c3e262e7de133258cb46427938025f4178cbefb2f0e293df160f1fb8`。

`tools/run_chois_evaluator.py` 的既有 point-estimate 路径保持原公式和 RNG 顺序；新增参数默认均为
disabled。D2-AA opt-in 后记录 embedded/dropped sequence IDs、对 Matching Score 与
R-Precision@1/2/3 做 10,000-replicate bootstrap，并对 paired FID 做 200-replicate bootstrap。
同时显式暴露 upstream loader 的 `batch_size=32, drop_last=true`：438 exports 中 416 条进入
embedding metrics、22 条被丢弃。这里没有改变 upstream behavior，只修复此前只报告 input count、
未报告 effective count 的 reporting omission。更新后 adapter SHA-256 为
`1038e7e1e7dc2882a2a199396b972ef1eb05335da0877c8f05310ff5ad738b4b`；
新增 7 项 D2-AA tests SHA-256
`ef70c1375d949a70652f3133514435bcf6dfd71f679e6707a051397ae6992827`。

Authority 使用指定 `infbagel` Python 完成 py_compile、7 项 D2-AA tests、22 项 governance tests、
全量 293 tests、registry validation（151 records）与 `git diff --check`。此时尚未 worker
fast-forward、创建 reportable manifest、加载 checkpoint 或启动任何 CPU/GPU evaluator workload。
下一步只能提交本 logical implementation，worker 主动 fast-forward exact object，生成
same-context preflight/resolved configs并以 `tools/experiment.py start` 创建唯一 umbrella
lifecycle；不得在 dirty worker 或未绑定 artifacts 上运行。

#### 2026-07-24 Phase 1B D2-AA completion / Table-5 reporting pass

唯一 umbrella run 已在 exact commit
`82ef8f212e77042abd4d6cedfc03fff16d9756eb`、clean `infbagel-4gpu` worker 上完成，exit code 0，
runtime `1879.9277 s`。四个 fixed final-online checkpoints 均重新运行 official 438 sequences ×
3 windows、500-step unguided diffusion；D2-V/X/Y/Z 各自重新生成的 native aggregate 18 个
scalars 与 per-sequence 5,104 个 scalars 对封存记录的最大绝对差均为 `0.0`、mismatch count
均为 0。四份 GT tree SHA-256 均为
`d439a98ea32f5d67964bc98431fe25bdffc24b63e00b42601c5355445d01742c`。

Pinned CHOIS evaluator 的 D2-V/X/Y/Z FID 分别为
`1.578132/1.775477/1.941435/1.935600`，paired embedded-sequence bootstrap 95% CI 分别为
`[1.188574,2.253517]`、`[1.303928,2.494238]`、`[1.509327,2.574472]`、
`[1.507692,2.567234]`。R-Precision@1/2/3 分别为
`0.149038/0.283654/0.413462`、`0.151442/0.283654/0.420673`、
`0.151442/0.286058/0.415865`、`0.153846/0.293269/0.406250`；Matching Score 分别为
`3.858646/3.879792/3.951129/3.918002`，Diversity 分别为
`8.740726/8.781011/8.397693/8.448088`。所有 additive metrics 使用 10,000-replicate、
seed-42 sequence bootstrap，FID 使用固定 200-replicate paired bootstrap。Upstream
`batch_size=32, drop_last=true` 未改变：每组 438 exports 中实际 416 条进入 embedding metrics，
22 条被丢弃；ordered dropped-ID SHA-256 为
`b7ddcb96dae95814e44d1df8f4fe1791c2c7930ed3ddfca55c3ea3fcde31bd15`。
这修复的是 effective-count reporting omission，不是 metric formula 或 scientific evaluator
defect；论文单一 `Rprec` 仍不得映射为任一 rank 指标。

同一 RTX 3090、相同首序列、batch-1、3-window、排除一次 warmup、CUDA synchronized 的
D2-V/X/Y/Z generation FPS 分别为
`15.2809/18.1699/18.0095/18.0700`；这些只用于本地对称 timing，不与论文未充分披露协议的
FPS 宣称严格同协议。完整 Table-5-aligned point estimates、embedding uncertainty、额外
hand/ratio/contact diagnostics 与论文原表逐列数据已写入
`experiments/results/p1_hoi_phase1b_d2aa_table5_completion_s42_20260724.json`，SHA-256
`d791c04bf1a896f4230a55e77518368cf4c5cb5c691c6ce98de65c18a87914d8`。

Manifest/metrics/resolved/preflight-r1/run-local-registry SHA-256 分别为
`a98f6aeab22859c82fba06646888922deae066be9593a97ba44a3de7f229650a` /
`9512467b4eebf5cce5b0ae40a60de63bda107e135cc45e54d9c8a4e3ffc59889` /
`c70286c63ea9fb16d19188d4b3651b0492d9cdd027d3c6c10198cb025c295495` /
`70bee2758200a7879b5bb6ec0be3d179b527deff408c1dbf63273d347e8e6097` /
`082f77f7199f3229f566622e5117f826b0631d21797d50a617e8efc482a306c1`。
完整 3,601-file / 132,167,442-byte artifact tree 已由 worker 主动回收到
`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2aa-table5-completion-s42-20260724`，双端
SHA-256 均为
`1fa7f570f935d58adaf1baddd8db2367ae49aa05804310b0a8c2d1cc7febeb77`。

首次 preflight flatten 因 worker 未安装 `jq` 在任何 scientific work/manifest/checkpoint/GPU
workload 前失败，空 `preflight.json`（SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`）永久保留；
随后只使用 worker `infbagel` Python 生成并绑定 `preflight_r1.json`。原异步 start 已创建 manifest
后的一次重复 start 被 fail-closed overwrite check 拒绝，未覆盖 manifest、未重启 workload、
未复用 run id。两者均为保留的 operational events，不影响点估计或 evaluator 路径。

严格分类为 `table5-completion-pass-nonselection-stop`。D2-AA 不产生新的模型质量 gate，
不改变 D2-V/X/Y/Z 的 negative classifications，不选择任何 checkpoint，不授权训练、consistency、
HSIPrior 或 Mixer。Handoff 为 `docs/phase_summaries/PHASE_1B_D2AA.md`。

#### 2026-07-24 Phase 1B D2-AA integrated-table reporting amendment

用户澄清最终对照必须显式区分并同时列出三类来源：

1. 论文 Table 5 的 InfBaGel 1/8/16 published rows；
2. 作者提供 released consistency-distilled checkpoint 在本地 Phase-0 evaluator 上的结果；
3. 先前从 integration baseline 独立 fork 的 `phase/01b-author-repro` 上，按作者
   diffusion→consistency 训练代码在本地 8×RTX 3090 完整复现所得 CM checkpoint 的历史
   official-438 HOI evaluation。

该 amendment 只整合已存在的不可变记录，不启动 evaluator/GPU、训练、checkpoint selection 或
consistency。禁止修改或覆盖已封存的
`p1_hoi_phase1b_d2aa_table5_completion_s42_20260724.json`；新增 integrated aggregate 必须使用
独立文件
`experiments/results/p1_hoi_phase1b_d2aa_integrated_table_s42_20260724.json`，并逐行记录
`source/protocol/reporting_status`，不能把 paper-reported 与 locally-evaluated 数值冒充同协议。

历史本地作者复现的训练 commit 为
`1e982bc3a35301287e6e6a9d56325e103655f0e1`，使用 seed 42、FP32、8×RTX 3090、
per-GPU batch 256/effective batch 2048；先完整训练 diffusion epoch500 checkpoint
`44a723d20a4bbf13de8c2db78c3c375472dba20be0530993c9e00ab780747aac`，再按作者 schedule
蒸馏至 CM epoch200 checkpoint
`a9f5a72e617c2789f84e902829e0aae9118cddfa264047ce9ec320b1cc0df8ab`。Pipeline exit code 0，
但原 branch registry 将其定义为 exploratory reproduction、`reportable=false`、
`final_table_authorized=false`，且不具备 `tools/experiment.py` reportable manifest；本次用户
授权只允许将其作为明确标注的 historical local reproduction row 引用，不能提升为正式
HOIPrior candidate 或选择证据。

该 checkpoint 的既有 evaluation 使用 seed 42、official 438 sequences × 3 windows、
`sample_type=consistency`、`cm_timesteps=16`，aggregate SHA-256
`97cf420d1f2b2596efc957daee4a812a6fd04ba288eacf45abd9b020a41ad573`。Native point estimates
为 Te/Txy/FS `3.555258/3.764048/0.331956`，Cprec/Crec/Cf1/C%
`0.790091/0.749855/0.745333/0.623777`，Pbody/MPJPE/Troot/Tobj/Oobj
`2.777503/11.819468/7.788341/15.882751/1.018622`。该历史 run 没有 per-sequence uncertainty，
且 `save_chois_eval_npz=false`，所以 FID、Matching、R-Precision@1/2/3、Diversity 都必须报告
为“未评估”，禁止用 released checkpoint 或论文行代填。其 `321.6566 FPS` 是 full-438
descriptive throughput，不是独立 batch-1 或论文 FPS，必须移到额外 timing 列，Table-5 FPS
保持 null。

论文 InfBaGel 1/8/16 的完整 15 列直接引用
`https://arxiv.org/pdf/2604.04843` Table 5；它们的 `Rprec` 是论文单一标量，仍不得与本地
R-Precision@1/2/3 映射。Integrated table 完成后只允许追加 registry completion、更新 handoff
中的 reporting appendix 并提交；不得因此重开 D2-V/X/Y/Z gate、选择 checkpoint 或开始下一
科学 subphase。

#### 2026-07-24 Phase 1B D2-AA integrated-table reporting completion

只读逐项交叉检查通过：论文 InfBaGel 1/8/16 的 45 个 scalars 与已封存论文转录完全一致；
released、D2-V/X/Y/Z 的 75 个 Table-5-aligned scalars 与 D2-AA compact 完全一致；历史
author-code CM e200 的 12 个可用 Table-5 native scalars 与原 official-438 aggregate 完全一致。
历史行的 Rprec/FID/FPS 均保持 null，并记录未导出 CHOIS、没有独立 batch-1 timing 的原因；
full-438 `321.6566 FPS` 只保留为 descriptive throughput。

独立结果为
`experiments/results/p1_hoi_phase1b_d2aa_integrated_table_s42_20260724.json`，SHA-256
由 completion registry 绑定。原 compact SHA-256 复核仍为
`d791c04bf1a896f4230a55e77518368cf4c5cb5c691c6ce98de65c18a87914d8`，未修改。分类为
`integrated-table-reporting-pass-nonselection-stop`：没有运行 GPU/evaluator、训练、distillation
或 checkpoint selection；历史复现没有被提升为 reportable/selectable HOIPrior，D2-V/X/Y/Z
结论与正式 HOIPrior 缺失状态不变。

