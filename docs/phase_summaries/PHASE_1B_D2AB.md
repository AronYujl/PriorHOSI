# Phase 1B D2-AB：predicted-support no-slip objective

## Scope and outcome

D2-AB 是有限剩余 HOIPrior campaign 的第一次、也是本轮唯一实际执行的完整 from-random
训练。唯一 manipulated factor 是：在保持 D2-X 其余训练协议不变时，用 differentiable
predicted-state support-aware physical no-slip residual 替换四个足部 joints
`[7,8,10,11]` 的 8 个 FK-routed x/z velocity residual slots。该 objective 不调用
official evaluator 的 floor helper、near-ground threshold、DBSCAN 或 foot-sliding
reduction，因此不是 evaluator-threshold trick。

全部预注册 lifecycle 已完成，最终 classification 为
`predicted-support-no-slip-optimization-negative-stop`。训练本身稳定并保留了 D2-X 的
object/contact/penetration capability，但 registered internal mechanism quantity 向错误方向
移动，official foot sliding 也未改善。因此 D2-AB checkpoint 不可选择；第二次完整训练预算、
fallback、checkpoint selection、consistency、HSIPrior 和 Mixer 均未授权或启动。

## Commits and fixed protocol

- Starting authority HEAD：`effc909da9693e25a8bb417749028b53675f4790`。
- D2-AB implementation/start commit：
  `3fce4767111f7b4c01b5c2af252f6c3ef362cf43`。
- Continuous detached lifecycle amendment：
  `3d10eb48bbc8c480a9f474e1e01dd133f96a2bed`。
- Hash-bound resume target commit：
  `0db60d82e454dd722320832e9f7b3f228a90ef72`。
- Completion-manifest transition/evaluation commit：
  `de38d27368dd87f6b30e5aa2a8287421910a93d5`。
- Completion documentation/registry/result commit：the final Git commit that adds this summary,
  compact result and tracked completion records (identify it with
  `git log --format=%H -- docs/phase_summaries/PHASE_1B_D2AB.md`)。
- Branch：`phase/01b-hoi`。
- Worker：`infbagel-4gpu/node01`，4×RTX 3090。

固定训练 contract：

- seed 42，固定 OMOMO train/internal split；
- 232-D representation、16-frame windows、2-frame history；
- clean-x0 500-step diffusion，512-wide/16-head/8-layer scene-free HOIPrior；
- per-GPU batch 512、effective batch 2,048、accumulation 1；
- 61,440,000 processed windows / 983,040,000 processed frames /
  30,000 optimizer updates；
- FP32 Adam、LR `1e-4`、betas `(0.9,0.999)`，无 warmup/scheduler/weight decay/
  gradient clipping/AMP/EMA；
- loss weights：FK `0.3569973401779424`、object surface
  `0.4772322188400037`、velocity `0.1`、terminal object goal `1.0`；
- production sampler、conditions、data、native evaluator 与 D2-X control 不变；
- released、author、D2-V/X/Y/Z、prior、consistency 或 EMA checkpoint 禁止作为初始化。

公式中每个 left/right foot pair 使用 predicted clean-x0 经 denormalization、rotation decode
和同一 FK 得到的 foot height 定义 soft support；第一个 future residual 的 previous state
来自 immutable GT history，后续 previous state 来自 predicted FK trajectory。固定
support scale `0.03925712490454316 m`、sample interval `0.1 s`、velocity scale
`0.029363068377844033 s/m`。train-only support metadata SHA-256：
`807978580221910ad00260c2dff4f33ddacbb1bf72bad7443bf21ac48f31f079`。

## GPU smoke

Run：`p1-hoi-d2ab-gpu-smoke-s42-20260725`。

- Exact Git object：`3fce4767111f7b4c01b5c2af252f6c3ef362cf43`。
- Device：`cuda:0`，四卡可见且无 compute contention。
- Real-data batch 8，timesteps `0/249/499`。
- Initial model-state SHA-256：
  `ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e`。
- Loss 与关键 root/rotation/model gradients finite/nonzero。
- Support mean/q05/median/q95：
  `0.0241769 / 1.47077e-16 / 1.19719e-09 / 0.238128`。
- Support occupancy：`7.14286% >0.05`、`100% <0.95`，non-collapse contract 通过。
- Memory headroom：`24,996,151,296 bytes`。
- Optimizer created/updates、checkpoint loads/writes、selection、consistency：
  `false/0/0/0/false/false`。

Smoke manifest/metrics/resolved/preflight/run-local-registry SHA-256：
`f36f5317...588` / `da54b224...b9e` / `3ee53802...e4f` /
`5cb07043...b25` / `3b3582a4...493`。完整 tree 为 15 files / 91,202 bytes，
SHA-256 `654733afafcde8ffed20f41d0812e46cd4a62d91f3f8ccedb4d9d1c837823bd6`。

## Formal training and continuous resume

Run：`p1-hoi-d2ab-predicted-support-no-slip-s42-20260725`。

- Random initialization；initial model-state SHA-256 与 smoke 完全相同。
- 完成 `61,440,000` windows / `30,000` updates，exit code 0。
- Loss finite、required gradients present、AMP overflow skips 0。
- Final validation total/reconstruction/FK/object-surface/velocity：
  `0.0488587866 / 0.0411070134 / 0.0103592101 / 0.0082613681 /
  0.000100191943`。
- Wall time：`18,382.995 s`。
- Throughput：`3,342.219 windows/s` / `53,475.508 frames/s`。
- Minimum memory headroom：`21,261,123,584 bytes/rank`。
- Cadence artifacts：20 checkpoints、80 rank RNG sidecars。
- Final checkpoint SHA-256：
  `3eb68cc55cae15fd4bd3ff5279131ffd9a35ba0399e8e90557e89cb301631d8e`。
- Manifest/metrics/run-local-registry SHA-256：
  `f62837fe...86a` / `4b0d0426...07d` / `32006088...d86`。

首个 `3,072,000`-window checkpoint
`ceb73ebc3a72d6290fc63e2546533c1565912b905a980381d406ea71b39a2ecc`
只用于证明 resumability。按照用户要求，Codex 在 resumed initial stability 通过后停止轮询，
但 worker-owned persistent training 持续运行到完整预算；没有在 Codex yield 时停止、暂停、
重启或新建 run id。

Continuation provenance 明确绑定：

- checkpoint/source commit：
  `3fce4767111f7b4c01b5c2af252f6c3ef362cf43`；
- workload target commit：
  `0db60d82e454dd722320832e9f7b3f228a90ef72`；
- `git diff --binary` SHA-256：
  `9c777e1058ddc78ffdf2455141870e3d08eee37b51621bae4ffa45b32448ec86`；
- changed-file allowlist：
  `code/config/config_train_hoi_prior.yaml`、`code/train_hoi_prior.py`、
  `docs/EXPERIMENT_PLAN.md`、`experiments/registry.jsonl`、
  `tests/test_hoi_d2ab.py`。

训练没有加载 released、author、D2-V/X/Y/Z、其他 prior 或 EMA state。恢复的是同一 D2-AB
lineage 的 optimizer/model/RNG state，不是第二次训练或 checkpoint selection。

## Fixed internal diagnostic

Run：`p1-hoi-d2ab-predicted-support-no-slip-internal-s42-20260725`。

- Sealed internal-validation cohort：32 sequences / 96 windows。
- Selection SHA-256：
  `30524c88481f6cb81e8063073d510ad01543be92d91eb4ef9b2b8a376cc4fbae`。
- Control/target：D2-X/D2-AB final-online checkpoints。
- Timesteps：`249`、`499`。
- Pairing：same clean windows、timestep、noise、condition dropout。
- Uncertainty：seed 42、10,000 bootstrap replicates、sequence unit。
- Runtime：`4.9357 s`。
- Official-test sequences、optimizer、updates、checkpoint write/selection、
  consistency：`0/false/0/false/false/false`。

Primary registered quantity `D2-X minus D2-AB supported velocity`：

| timestep | mean | sequence-bootstrap 95% CI | gate |
|---:|---:|---:|---|
| 249 | -0.0013696933 | [-0.0021679224, -0.0006211924] | fail |
| 499 | -0.0017119791 | [-0.0029663018, -0.0008443758] | fail |

两个 CI 都完全低于 0：D2-AB 在 predicted-support 区域的水平足速显著高于 D2-X，方向与
假设相反。

Target/control support-mass ratio：

| timestep | mean ratio | sequence-bootstrap 95% CI | `[0.80,1.20]` sanity |
|---:|---:|---:|---|
| 249 | 1.0086136 | [1.0050990, 1.0122306] | pass |
| 499 | 1.0088666 | [1.0045668, 1.0130727] | pass |

因此负结果不是通过抬脚、关闭 support 或减少 mass 得到的。作为附加诊断，
`D2-X minus D2-AB` no-slip residual difference 在 `t=249` 为
`-0.0009748401`、CI `[-0.0021468720,0.0001270317]`；在 `t=499` 为
`-0.0010461475`、CI `[-0.0019493901,-0.0002256633]`。

Internal manifest/metrics/preflight/run-local-registry SHA-256：
`8f15b8b3...c72` / `80568dd4...8cf3` / `feaeeb50...5f0` /
`12732b32...146`。完整 tree 为 18 files / 258,193 bytes，SHA-256
`605cfd89381ceb9eb5e35adc4d274feccd102b06f26fbde6570499d6db9cdd55`。

## Native evaluation

Run：`p1-hoi-d2ab-native-eval-s42-20260725`。

- Official 438 sequences × 3 windows。
- 500-step unguided diffusion，fixed final online weights。
- CFG、dynamic perception、guidance、scene、consistency 均关闭。
- D2-X final-online aggregate/per-sequence records 原样复用，不重新生成 control。
- Penetration finite mask：181/438，sequence IDs SHA-256
  `2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec`，
  contract passed。
- Registered uncertainty：seed 42、10,000 bootstrap replicates、sequence unit。

D2-AB native point estimates：

| Te | Txy | FS | Cprec | Crec | Cf1 | C% | Pbody | MPJPE | Troot | Tobj | Oobj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.6840 | 4.0892 | 0.3661 | 0.7896 | 0.5953 | 0.6383 | 0.4775 | 3.7714 | 12.0639 | 8.1101 | 15.9248 | 1.0244 |

对 D2-X 的 registered gates：

- `D2-X minus D2-AB` foot sliding mean `-0.0030536116`，95% CI
  `[-0.0175441732,0.0110958477]`：official improvement fail。
- `D2-AB minus D2-X` contact F1 mean `+0.0008802922`，95% CI
  `[-0.0046342095,0.0066400986]`：protection pass。
- D2-AB/D2-X MPJPE ratio CI `[0.9966440,1.0057098]`：pass。
- End-object ratio CI `[0.9712100,0.9986953]`：pass。
- Txy ratio CI `[1.0019801,1.0173954]`：pass。
- Object-translation ratio CI `[0.9914255,0.9999431]`：pass。
- Hand-penetration ratio CI `[0.9287033,1.0102184]`：pass。
- Human-penetration ratio CI `[0.9319976,1.0134658]`：pass。

全部 released-baseline absolute diffusion checks 通过；其中 FS ratio
`1.0980924 <= 1.10`，contact F1 `0.638306 >= 0.60`。保护能力没有成为负结果来源。

Runtime `392.837 s`；generation `62.556 s`；end-to-end generation
`386.303 s`；55,188 generated frames；CUDA-synchronized descriptive throughput
`882.212 FPS`。当前 evaluator 没有生成 FID、Matching、R-Precision 或 Diversity，
`fid_rprecision_used=false`；这些字段未被删除，也没有从 D2-AA、released 或其他 checkpoint
代填。

Native manifest/metrics/resolved-config/resolved-target/preflight/aggregate/per-sequence/
run-local-registry SHA-256：
`7e5c51a0...7794` / `34127732...d58` / `57909f26...09e5` /
`631867b8...6b3` / `e10f7b4f...eb4` / `f16c1718...6bd5` /
`ebf2e8f6...108ca` / `452a7b5f...0cd`。完整 tree 为 22 files / 379,944 bytes，
SHA-256 `d56bff74b8982a6f63efd62e784c2759233d9f5591222dbe9deb9922e17b5d42`。

## Verified facts, inference, and unresolved questions

### Verified facts

- D2-AB 完成 exact from-random fixed-budget contract，训练稳定、无 numerical failure。
- Support sanity 在两个 timestep 均通过。
- Internal supported-velocity gate 在两个 timestep 均显著向相反方向移动。
- Official foot-sliding improvement 未通过。
- D2-X protection 与 released-baseline absolute gates 全部通过。
- Provenance、normalization、finite-value、penetration-mask 和 recovered-tree contracts
  全部通过。
- 没有发现确定的 training loss、model、mathematical、data 或 official evaluator
  scientific implementation defect。

### Evidence-based inference

Predicted-state support 与 zero-slip target 本身不足以在固定 objective mixture 下优化出更低
supported foot velocity。由于 internal gate 已在 official rollout 之前显示方向性失败，而
object/contact/penetration protection 保持良好，本结果更符合“该 auxiliary 没有被优化成目标
机制”，而不是“机制已经学会但只在 official evaluator 上没有 transfer”，也不是 protected
capability tradeoff。

预注册 gradient-projection fallback 要求 D2-AB internal mechanism gate、support sanity 和
late-timestep conflict trigger 同时通过。当前第一个必要条件失败，所以 fallback trigger 不成立；
不得把 D2-Y/Z 已知的 gradient conflict 单独抽出来，绕过触发条件启动第二次训练。

### Unresolved questions

- 独立 preregistered timestep/SNR-aware geometry weighting 是否能改变 optimization direction；
- local gradient-conflict resolution 在不同、合法触发条件下是否有效；
- finite predicted-history exposure 是否能缩小 train/rollout distribution gap。

这些问题均未得到本 subphase 授权，也不能据此选择 D2-AB checkpoint、扩预算或启动 consistency。

## Preserved operational failures

下列 lifecycle/contract 问题均已保留，没有被删除或伪装为 scientific evidence：

- pre-workload smoke support threshold 会误拒绝合法 random initialization，已改为
  direction-neutral non-degeneracy contract；
- internal diagnostic 的 import、per-window→sequence reduction 与 schema defect；
- native resolved-config wrapper monkey-patch recursion；
- governance-only commit 使原 exact-commit resume guard 无法表达合法同-lineage continuation，
  由 hash-bound source/target/diff guard 修复；
- completed run 的 manifest start commit 与 workload target commit 不同，普通 finish/register
  无法封存已绑定 transition，由最小 fail-closed completion guard 修复；
- malformed first resume failure JSONL、stale inspection timestamp、两次 internal manifest
  pre-workload command transcription repair，以及一次 read-only post-launch SSH quoting
  failure。

这些事件均未改变 D2-AB formula、data、optimizer、budget、checkpoint、native evaluator 或
scientific gates。没有因此重启训练、复用 run id、覆盖 artifact 或遗漏失败记录。

## Verification

Implementation lifecycle 已使用 authority Python
`/data/yujinlun/anaconda3/envs/infbagel/bin/python` 完成 metadata/hash checks、targeted D2-AB
tests、governance tests、full CPU suite、registry validation、worker HOI-applicable tests、
fully resolved configs 和 same-context GPU preflights。Smoke、training、internal diagnostic
与 native evaluator 的 manifests 均通过 `tools/experiment.py start/finish/register`，正式
workload 均来自 clean committed worker Git object。

Completion documentation session 重新执行并通过：

- `INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python`；
- `"$INFBAGEL_PYTHON" -m unittest -v tests.test_hoi_d2ab`：16/16 passed；
- `"$INFBAGEL_PYTHON" -m unittest -v tests.test_research_governance`：23/23 passed；
- `"$INFBAGEL_PYTHON" -m unittest discover -s tests -v`：310 tests，308 passed、2 个
  按 `INFBAGEL_WORKER_EXPERT=hoi` 预期 skip；
- `"$INFBAGEL_PYTHON" tools/experiment.py validate`：167 registry records、2 splits、
  2 evaluators、1 training protocol valid；
- compact result JSON parse、registry JSONL parse 与 `git diff --check`。

authority 环境没有 `pytest` 模块；两次 pytest 命令均在 import 前退出，未替代或污染
unittest 结果。整个 completion documentation session 没有 GPU 或 scientific evaluation。

## Tracked and external artifacts

- Compact result：
  `experiments/results/p1_hoi_phase1b_d2ab_predicted_support_no_slip_s42_20260726.json`。
- Training authority staging：
  `/data/yujinlun/InfBaGel-worker-staging/p1-hoi-d2ab-final-r2-20260726`。
- Internal authority staging：
  `/data/yujinlun/InfBaGel-worker-staging/p1-hoi-d2ab-internal-r1-20260726`。
- Native-eval authority staging：
  `/data/yujinlun/InfBaGel-worker-staging/p1-hoi-d2ab-native-eval-r1-20260726`。
- Smoke authority staging：
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ab-gpu-smoke-s42-20260725`。
- Compact training/internal/native tree hashes：
  `e357b0c6...63ab` / `605cfd89...dd55` / `d56bff74...5d42`。

No merge commit or immutable tag was created because the D2-AB scientific gate failed.

## Exact next entry point

D2-AB 在此关闭为 controlled optimization-negative result。其 checkpoint 不可选择、不可作为
consistency 或后续 prior 初始化，也不得 resume。当前 campaign 的第二次完整 from-random
training budget 仍未消耗，但 D2-AB preregistered fallback trigger 未成立，因此本 session
不能使用该预算。

任何未来 HOIPrior 动作必须先执行真实 `date`，完整扫描 plan/registry 确认下一 unused Phase 1B
identifier，新增 dated plan 与 append-only registry hypothesis，并取得用户新的明确授权。
不得重命名重复 D2-Q/R/S/W/X/Y/Z/AB 已削弱的路线，不得启动 parameter sweep、checkpoint
selection、consistency、HSIPrior 或 Mixer。
