# Phase 1B P11 — 手-物几何项根平移梯度 detach

**状态:** 完成（否定，方向关闭） · **日期:** 2026-08-15 · **seed:** 42 · **预算:** 固定 299.52M
**分类:** `root-coupling-negative-stop`
**紧凑结果:** `experiments/results/p1_hoi_p11_root_detach_s42_20260815.json`
**完整 bootstrap:** `results/experiments/p1-hoi-p11-geom-rootdetach-r1-s42-20260814/chain/bootstrap_p1-hoi-p11-geom-rootdetach-r1-eval-guided-s42-20260815.json`
**预注册:** `docs/plan/PHASE_1B_HOI/06_GEOMETRY_TERM.md`「2026-08-14 Phase 1B P11」；科学预注册行
`p1-hoi-p11-root-coupling-repair-preregister-s42-20260814`，严格重试行
`p1-hoi-p11-geom-rootdetach-r1-preregister-s42-20260814`。

---

## 结论先行

**PRIMARY 以相反方向失败。分类为 `root-coupling-negative-stop`，不选取任何 checkpoint。**

P11 只操纵 `hand_object_contact_detach_root=true`：几何项读取由
`predicted_positions[..., 0, :].detach()` 构建的 FK；forward 值不变，只有几何项通往
`prediction[..., 0:3]` 的梯度归零。其余设置逐项固定为封存 W3。

结果不是边界阴性。`trans_dist` 从 **8.43721016280387** 升到 **17.52702829738458**，配对差
**+9.089818134580709**，95% CI **[+8.02119313337368, +10.220037403018914]**；
`pelvis_goal_error_cm` 从 **4.650959228569446** 升到 **12.904819034754414**，差
**+8.253859806184968**，95% CI **[+7.431480166806359, +9.113061047471703]**。
两项都不是显著改善，而是大幅显著变差。

接触侧也没有换来预期交易：`contact_percent` **0.642295426542002 → 0.6169638327172573**，
`contact_f1` **−0.019094690227612555**
**[−0.033875306030107216, −0.004885658194303703]**，显著变差。14 项中 10 项显著，
全部朝坏方向，0 项显著改善。

---

## 前置 probe 与被证伪的读法

正式训练前，`code/priors/hoi/diagnostics.py` 的 `root_gradient_share_probe` 在封存 W3 checkpoint
上测得 `root_gradient_share = 1.011`；几何项与非几何项在根平移通道上的梯度余弦为 **−0.319**。
几何项主导根平移梯度，且非几何项在那里与它反向。该结果通过了事前 5% 启动门。

假设把这种主导读成病理性捷径：模型可能靠平移全身满足接触，而不是调整手的姿态。正式结果说明，
**在本次受控检验中，这条根平移梯度是 global placement 的承重信号。** 移除它没有把几何项重新导向
旋转，而是移除了模型依赖的信号。`trans_dist` 超过翻倍，`pelvis_goal_error_cm` 接近三倍；接触参与也略降。
这是测量支持的读法；本记录不对未测机制继续外推。

---

## 原生判据逐条裁决

全部判据按官方 438 序列、10,000 次 sequence-level paired bootstrap、seed 42、共享重采样索引，
相对封存 W3 裁决。

| 判据 | 事前规则 | 结果 | 裁决 |
|---|---|---|---|
| (i) 参与保持 | `contact_percent ≥ 0.60` 且 `contact_f1` 不得显著差于 W3 | 0.6169638327172573 通过点估计下限，但 `contact_f1` 显著更差 | **失败** |
| (ii) 耦合打破（PRIMARY） | `trans_dist` 与 `pelvis_goal_error_cm` 双双显著优于 W3 | 两项均大幅显著更差 | **失败，且方向相反** |
| (iii) 保护 | `end_obj_trans_err` 与 `mpjpe` 均不得显著差于 W3 | `end_obj` +3.8659282808763686，`mpjpe` +2.790050367616355，均显著更差 | **失败** |

按写死的停止规则，(ii) 失败即 `root-coupling-negative-stop`。

- `root-coupling-repair-positive` 要求三项全过，本轮三项全败；
- `root-coupling-repair-partial` 要求 (ii) 成立，本轮不成立；
- `contact-engagement-was-root-translation-bought` 也要求 (ii) 成立，且要求参与回落到 H0 的
  `0.53519`；本轮 (ii) 失败，`contact_percent=0.6169638327172573` 也未回落到该水平，两个子句都不满足。

---

## 配对结果（B=P11，A=封存 W3）

| 指标 | A 均值 | B 均值 | B−A [95% CI] | 显著性 | n |
|---|---:|---:|---:|---|---:|
| `contact_f1` | 0.7788589083944136 | 0.7597642181668011 | −0.019094690227612555 [−0.033875306030107216, −0.004885658194303703] | B 更低 | 438 |
| `contact_precision` | 0.8105943170284625 | 0.8074512964314415 | −0.0031430205970210206 [−0.01505269587715201, +0.008127435599593983] | 跨零 | 438 |
| `contact_recall` | 0.7894479590204786 | 0.7598492653008142 | −0.02959869371966442 [−0.04680228155163989, −0.012768373599568818] | B 更低 | 438 |
| `end_obj_trans_err` | 4.69178380888905 | 8.557712089765419 | +3.8659282808763686 [+3.3629333145136506, +4.3608128869779526] | B 更大 | 438 |
| `foot_sliding` | 0.32083540148759215 | 0.346336941981994 | +0.025501540494401755 [−0.004324656247849458, +0.054953689806693316] | 跨零 | 438 |
| `hand_pen_loss_omomo` | 0.25874164798574417 | 0.25649464418195883 | −0.0022470038037853135 [−0.034427549739748375, +0.02746974221545946] | 跨零 | 181 |
| `hand_pen_ratio` | 0.15000439230455548 | 0.16991143579497028 | +0.01990704349041479 [+0.007263389389236619, +0.034090959733675076] | B 更大 | 181 |
| `human_pen_loss_infbagel` | 4.07787677508961 | 4.055461823665692 | −0.022414951423917556 [−0.5284404323228983, +0.4469194330911492] | 跨零 | 181 |
| `human_pen_ratio` | 0.15245988654250597 | 0.1755678425311466 | +0.023107955988640613 [+0.008838361435046668, +0.039147111828524324] | B 更大 | 181 |
| `mpjpe` | 12.313383755584558 | 15.103434123200913 | +2.790050367616355 [+2.3676287655646333, +3.22629809485893] | B 更大 | 438 |
| `obj_rot_dist` | 0.9654721531033464 | 1.000268710010062 | +0.03479655690671563 [+0.004136575723872358, +0.06580889358764871] | B 更大 | 438 |
| `obj_trans_dist` | 16.590142601108123 | 19.496280411656056 | +2.906137810547933 [+2.367099603690776, +3.442842623003349] | B 更大 | 438 |
| `pelvis_goal_error_cm` | 4.650959228569446 | 12.904819034754414 | +8.253859806184968 [+7.431480166806359, +9.113061047471703] | B 更大 | 438 |
| `trans_dist` | 8.43721016280387 | 17.52702829738458 | +9.089818134580709 [+8.02119313337368, +10.220037403018914] | B 更大 | 438 |

所有指标均为 0 个 `nan_replicates`。十个常规指标使用 438/438 配对；四个穿透指标
`hand_pen_loss_omomo`、`hand_pen_ratio`、`human_pen_loss_infbagel`、`human_pen_ratio`
各丢弃 257 个非有限配对，只使用 **181/438**。其中 `hand_pen_ratio` 与 `human_pen_ratio` 的显著变差
因此建立在 n=181 上，不能写成全体 438 序列的显著性。

---

## W3 复用与估计量口径

P11 的每一个 `mean_a` 都与 P10 紧凑结果中对应的 A00 cell mean 完全相等，确认复用的是同一个封存 W3
逐序列文件，而不是重新生成。输入文件为 275,581 bytes，sha256
`bbcd9e1b550d42bf4ac19f9a55db4b9eebb896a8ddb2d562b5226a11b297f6b2`。

P8/P9 封存聚合文件中的 W3 `end_obj_trans_err` 是 **4.6820018440485**；P10 与 P11 的 bootstrap
使用逐序列值的平均 **4.69178380888905**。这是 aggregate-file estimator 与 per-sequence estimator
的口径差异，不是错误或更正。

两侧 `gt_contact_percent` 都严格为 **0.6618830180474017**，也确认相同 438 序列与相同 GT pipeline。

---

## 训练与评测记录

训练 run `p1-hoi-p11-geom-rootdetach-r1-s42-20260814` 从随机初始化完成 299,520,000 windows
（4,792,320,000 frames），146,250 updates，526.8731331994104 epochs equivalent。有效批次 2048：
4 卡 × 每卡 512，gradient accumulation 1。优化器 Adam，lr 0.0001，无 scheduler、warmup、AMP、
gradient clipping；`amp_overflow_skips=0`。`status=stable`、loss finite、关键梯度存在，训练器 clean-worktree
preflight 已执行并通过。

运行于 worker `node01` 的 4× NVIDIA GeForce RTX 3090：
`2026-08-14T13:18:54Z` 至 `2026-08-15T14:35:52Z`，`wall_seconds=91000.61439041095`
= **25.28 h**，吞吐 3291.4063493571475 windows/s、52662.50158971436 frames/s。终态 checkpoint：
`p1-hoi-p11-geom-rootdetach-r1-s42-20260814_windows299520000.pth`，356,284,715 bytes，sha256
`5e59fb65e83a7580d6b3c0d5f81c5f850203e6edb7c97614ffdde9a86fa2eb98`。released checkpoint 未使用。

成功评测 run `p1-hoi-p11-geom-rootdetach-r1-eval-guided-s42-20260815` 于
`2026-08-15T15:31:33Z` 至 `2026-08-15T15:37:23Z` 在 cwd
`/home/yujinlun/data/work/InfBaGel-release/code` 完成，returncode 0。24 个评测 override 固定 official
438 序列、三窗口、500-step diffusion、`load_scene=false`、online checkpoint 与 P7 封存 Arm B 引导：
scale 1000、last 10 steps、update clamp 1、predicted mask threshold 0.95、contact weight 3、
consistency weight 1 / author normalization、object-goal weight 1。

---

## 失败记录

### 首次 training id 的操作合同失败

`p1-hoi-p11-geom-rootdetach-s42-20260814` 因 worker 环境漏出
`INFBAGEL_WORKER_EXPERT=hoi` 在任何训练前被 guard 拒绝，已按 `contract-failure-stop` 保留；严格 r1
预注册使用新 id，本轮没有复用失败 id。完整细节保留在原 registry 行与计划 amendment。

### r1 的首次 evaluation contract failure

第一次评测记录保留在
`results/experiments/p1-hoi-p11-geom-rootdetach-r1-s42-20260814/chain_failed_eval_20260814/`。
`code/constants.py:4` 把 `ROOT_DIR='..'` 写成供 `code/` cwd 使用的字面量，
`code/test_infbagel_hoi.py` 也按 cwd=`code/` 编写；当时 `tools/hoi_chain.py` 却从仓库根执行每个 stage，
数据路径因此解析到 checkout 上方。评测在 21 s 后于 `test_infbagel_hoi.py:502` 以
`FileNotFoundError: '../data/test/seq_id.pkl'` 失败，发生在任何 GPU 工作之前，没有 metrics 或 partial outputs。

失败评测 id `p1-hoi-p11-geom-rootdetach-r1-eval-guided-s42-20260814` 没有复用；成功评测改用
`-20260815`。修复提交 `fdfd671f78683e08cb5745ea030f7de0bd304a23` 只改变编排：按 stage 设工作目录、
在 stage status JSON 记录 cwd、增加回归测试；该提交全量套件 292 passed。

---

## Provenance：训练与评测位于不同 HEAD

训练从开始到完成都记录
`c924e7a5a217df7eb5cf911df9fcc50bfc0146cc`；成功 evaluation chain preflight 记录
`fdfd671f78683e08cb5745ea030f7de0bd304a23`。两者明确不同。差异只包含上述 orchestration 修复，
不触及 model、loss、sampler、evaluator 或任何 config value；本记录不把两个 HEAD 写成相同。

---

## 关键产物与哈希

- bootstrap：`99e92a39be50685bc279a578d03f5831c635dc25e93dd17afa2f5f87c6668ef8`
- W3 per-sequence 输入（275,581 bytes）：`bbcd9e1b550d42bf4ac19f9a55db4b9eebb896a8ddb2d562b5226a11b297f6b2`
- P11 per-sequence 输入（276,169 bytes）：`cfc388a3af0df3e1e9bcb8888e8fca78a98cc6df7cfa06c078506d061e05fa6d`
- P11 aggregate metrics：`58800593da45e0292cf7006fe163a721a1f1c89db3783f8235e844637b276984`
- final checkpoint：`5e59fb65e83a7580d6b3c0d5f81c5f850203e6edb7c97614ffdde9a86fa2eb98`
- shared resample index：`1a22b313a8441d58b3f6db4947a76b426c09c3d611402bccbe31b75d4acd5724`
- split：`019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e`
- data contract：`a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf`
- BPS：`fdff7204b4697e105457cb7e39267b9555bc0d8d854dbc92cd67e2d8c3e77042`

---

## 验证命令

- `git diff --check`
- 逐行解析 `experiments/registry.jsonl`，确认 275 → 276 行，并比较 HEAD 与工作树前 275 行 sha256。
- 解析新紧凑 JSON；逐项对比 14 个指标的 `mean_a/mean_b/mean_delta/ci_low/ci_high/significant/direction/n_pairs_used` 与 bootstrap 源文件。
- `/data/yujinlun/anaconda3/envs/infbagel/bin/python -m pytest tests -q`
- `git status --porcelain`

收尾提交前的精确输出写入本次交付报告；这里记录可复现命令，不提前伪造运行结果。

---

## 局限

- 四个 penetration 指标只在 181/438 序列上有限；各丢弃 257 个非有限配对。两个显著变差的 ratio
  指标同样只有 n=181。
- `contact_percent` 是 aggregate point estimate，没有 sequence-level CI。
- 单 seed、单 lineage；不主张跨 seed 或跨谱系外推。
- **开放且未解决的 sampler `_target_` caveat：** W3 的 Hydra config line 107 是
  `priors.diffusion.HOIPriorSampler`，P11 是 `priors.hoi.diffusion.HOIPriorSampler`。refactor
  `9259d3a` 把 GaussianDiffusion 等迁入 frozen `priors.core.ddpm` 并 re-export；迁移已验证为结构性，
  但跨 refactor 的数值等价性**尚未证明**。这是任何 post-refactor arm 对 sealed pre-refactor baseline
  的既有 branch-wide 属性，不是 P11 引入，也不能在本记录中写成已解决。
- 训练与评测 HEAD 不同；差异已限定为 orchestration-only，但仍作为显式 provenance caveat 保留。

---

## 模型侧计数与几何谱系

P11 **不增加 model-side negative 的运行计数**。结论 13 的 D2-AJ 仍是第十次 model-side failure；
连同 D2-AH 的前置诊断阴性，到 P6 为止仍是十一项干预。那些干预都向网络增加内容，P11 则从既有 loss
移除一条梯度路径，按自己的预注册与 P8 同属 objective-side。

几何谱系的算术是：P8/P9/P9b/P9c 的八点剂量响应产生唯一一次正向结果并选中 W3；P10 公式 2×2 阴性；
P11 root detach 阴性。因此 P11 是该几何谱系中继 P10 之后**连续第二个 objective-side negative**，
不是第十一次 model-side failure，也不是 model-side 计数加一。

---

## 下一入口

按预注册回到 P10 的 **“objective has attractors but no repulsor”** 指针。P11 排除的是“切断几何项的根平移
梯度会改善耦合”这一具体方向；它没有授权新机制，也没有测量排斥项本身。下一科学方向仍由用户决定。
