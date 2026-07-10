# 独立 HOI / HSI Prior 实现改动说明

## 1. 改动目标

这次改动把原来由同一个 InfBaGel 模型交替学习 OMOMO 和 LINGO 样本的流程，拆成两个具有明确数据边界和状态空间的扩散 prior：

- **HOI prior**：只从 OMOMO 学习人体与物体之间的交互，不接收任何 scene 条件。
- **HSI prior**：只从 LINGO 学习人体与真实场景之间的交互，不预测物体和接触状态。

这样设计的直接目的，是为后续通过 score composition、Mixer 或物理 guidance 组合出 HOSI 模型提供两个干净、可审计的专家，避免继续使用“由 OMOMO motion 反推 synthetic scene”的条件分布。

本次实现不是简单增加两份 YAML。模型输入维度、条件入口、loss、dataset 加载、验证切分、checkpoint 和评估指标都按 prior 类型进行了实质拆分。

## 2. 最终状态空间

两个 prior 共享相同的 16 帧、stride 3 人体表示和 500-step linear diffusion schedule，但状态维度不同。

| 状态切片 | 含义 | HOI prior | HSI prior |
|---|---|---:|---:|
| `0:84` | 28 个全局关节点位置 | 使用 | 使用 |
| `84:216` | 22 个关节的 6D 全局旋转 | 使用 | 使用 |
| `216:219` | 物体平移 | 使用 | 不存在 |
| `219:228` | 物体相对旋转矩阵 | 使用 | 不存在 |
| `228:232` | 4 个接触标签 | 使用 | 不存在 |
| 总维度 |  | **232** | **216** |

这一区别由代码中的 `PriorSpec` 强制执行，而不是依赖调用者自觉遵守。任何维度错误都会在 forward 时直接抛出异常。

实现位置：[code/models/priors.py](code/models/priors.py)

## 3. 两个独立 Prior 的实现

### 3.1 HOIPrior

`HOIPrior` 继承原 InfBaGel `Unet`，但在构造阶段强制设置：

```python
load_scene = False
load_scene_goal = False
load_object_goal = True
scene_type = None
dim_input = dim_output = 232
```

这意味着：

- 网络中不会实例化 scene encoder；
- OMOMO 的 synthetic occupancy 不会被读取；
- 模型保留 text、progress、pelvis goal、object goal 和 object BPS 条件；
- 输出完整的人体、物体和接触状态。

这里选择在架构层关闭 scene，而不是训练时随机把 scene 置空。这样从 checkpoint 的参数结构就可以验证该 prior 是否真正 scene-independent。

### 3.2 HSIPrior

`HSIPrior` 强制设置：

```python
load_scene = True
load_scene_goal = True
load_object_goal = False
dim_input = dim_output = 216
```

HSI prior：

- 使用 LINGO 中的真实 scene occupancy；
- locomotion 使用 `pelvis_goal`；
- sit/lie 等 scene interaction 使用 `scene_goal`；
- 不包含 object translation、object rotation 和 contact 输出；
- 不实例化 object BPS encoder；
- 如果错误地传入 `is_object=True` 样本，会立即失败，而不是静默学习 dummy object。

## 4. Dataset 层改动

修改文件：[code/datasets/infbagel.py](code/datasets/infbagel.py)

### 4.1 支持真正的 scene-free OMOMO

原 `InfBaGelDataset` 只有在 `load_scene=True` 时才读取 `scene_name.pkl`，但 object BPS、contact label 和 sequence name 又依赖该字段，因此过去无法同时使用：

```text
load_scene=False + load_object_goal=True
```

现在 sequence/scene name 与 occupancy 解耦加载，使 HOI prior 可以完全不加载 scene，同时仍正确读取 object 和 contact 数据。

### 4.2 避免无意义的大数组加载

scene-free HOI 训练不需要 per-frame `object_points.npy`。现在只有 scene occupancy 确实启用时才加载该数组，避免额外占用约数 GB 到 10 GB 级内存。

新增 `lazy_object_bps`：

- 不再在 dataset 初始化时把所有 sequence 的逐帧 BPS 拼成一个巨大 tensor；
- 每个 worker 只在取样时 mmap 对应 sequence 文件并复制所需的一帧；
- HOI 训练和测试配置默认启用该模式。

### 4.3 泛化 occupancy 查询

原 `get_occ_for_points()` 假设每次查询点数恒为 `32^3`，因此可以查询 voxel grid，却不能直接查询 28 个关节点。现在 batch index 根据实际 query point 数构造，可同时服务：

- scene voxel encoding；
- temporal occupancy；
- evaluator 中的 joint-scene penetration 计算。

### 4.4 HSI temporal occupancy 不再注入 dummy object

原 `_compute_occ()` 和 `_compute_occ_sample()` 在 `occ_temp` 模式下硬编码读取 `216:228` 的 object state。对 216 维 HSI 状态，这不仅会越界，也意味着逻辑上仍依赖 dummy object。

现在只有满足以下条件时才构造 object occupancy：

```text
state_dim >= 232 and dataset.load_object_goal
```

HSI 的当前、目标和 temporal occupancy 都只包含真实 scene。

## 5. Loss 和采样器改动

修改文件：[code/models/infbagel.py](code/models/infbagel.py)

`Sampler.p_losses()` 不再假设所有输入都是 232 维。

公共人体损失：

```text
L_human = MSE(joint positions) + L1(joint rotations)
```

HOI 额外损失：

```text
MSE(object translation)
+ L1(object rotation)
+ L1(contact labels)
+ transformed object keypoint loss
+ hand/foot forward-kinematics loss
```

HSI 返回的 object loss 字段为 `None`，不是对全零 dummy target 计算出的零损失。这样日志和训练代码可以明确区分“该 loss 不属于这个 prior”和“该 loss 数值恰好为零”。

同时修复了原 `p_losses()` 清理阶段引用未定义 `occ_goal` / `occ_temp` 的问题，并把 `pi`、`end_pi` 和 `seq_length` 在模型入口统一转换为整数索引，避免部分 LINGO 样本产生 embedding index 类型错误。

## 6. 训练系统

新增入口：[code/train_prior.py](code/train_prior.py)

训练器支持：

- 单卡、CPU 和多卡 DDP；
- AMP mixed precision；
- AdamW、gradient clipping 和 gradient accumulation；
- TensorBoard；
- train/validation component loss；
- best、last 和定期 epoch checkpoint；
- 从结构化 checkpoint 恢复模型、optimizer 和 AMP scaler；
- 固定 seed 和可选 deterministic CuDNN；
- `max_steps_per_epoch` 小规模烟雾测试。

### 6.1 数据切分

相邻 InfBaGel window 共享最多 47/48 个源帧，因此不能随机进行 window-level 切分。本次实现增加了稳定 hash 分组：

- **OMOMO/HOI**：按完整 source sequence 切分；
- **LINGO/HSI**：按完整 scene 切分。

HSI validation scene 中的所有 sequence 都不会出现在训练集。

当 validation 配置了最大 batch 数时，不再取文件开头连续的 windows，而是跨 held-out sequence/scene 做 deterministic round-robin 抽样，降低验证集集中在单个 sequence 或 scene 上的偏差。

实现位置：[code/prior_utils.py](code/prior_utils.py)

### 6.2 训练配置

HOI：

- [code/config/config_train_hoi_prior.yaml](code/config/config_train_hoi_prior.yaml)
- [code/config/dataset/hoi_prior.yaml](code/config/dataset/hoi_prior.yaml)
- [code/config/model/hoi_prior.yaml](code/config/model/hoi_prior.yaml)
- [code/config/sampler/hoi_prior.yaml](code/config/sampler/hoi_prior.yaml)

HSI：

- [code/config/config_train_hsi_prior.yaml](code/config/config_train_hsi_prior.yaml)
- [code/config/dataset/hsi_prior.yaml](code/config/dataset/hsi_prior.yaml)
- [code/config/model/hsi_prior.yaml](code/config/model/hsi_prior.yaml)
- [code/config/sampler/hsi_prior.yaml](code/config/sampler/hsi_prior.yaml)

默认模型仍采用原 InfBaGel 规模：`dim_model=512`、8 层 Transformer、16 heads。`per_device_batch_size` 表示每张 GPU 的 batch size。

## 7. Checkpoint 设计

旧训练只保存 raw `state_dict`，无法确定 checkpoint 来自哪个 prior、使用了什么数据切分和归一化。

新 checkpoint 包含：

```text
schema_version
prior_type / prior_spec
epoch / global_step
model_state_dict
optimizer_state_dict
scaler_state_dict
完整 resolved Hydra config
dataset contract
validation metrics
Python / NumPy / Torch RNG state
```

其中 dataset contract 记录：

- 数据目录绝对路径；
- window size 和 stride；
- source sequence / scene 数量；
- train/validation window 数量；
- split unit、fraction 和 seed；
- human normalization 的 SHA-256。

未来组合 HOI 和 HSI prior 时，应先比较 `human_norm_sha256`、window size、state spec 和 diffusion schedule，再允许组合。

新训练使用 strict checkpoint loading，并拒绝把 HOI checkpoint 加载到 HSI 模型。原 InfBaGel 的 raw checkpoint 仍可通过兼容后的 [code/utils.py](code/utils.py) 加载。

## 8. 专用评估系统

新增入口：[code/evaluate_prior.py](code/evaluate_prior.py)

### 8.1 固定噪声去噪评估

默认在以下 timestep 使用固定 Gaussian noise：

```text
[0, 50, 100, 250, 499]
```

这组指标主要用于：

- checkpoint selection；
- 检查高噪声阶段是否退化；
- 检查 HSI 是否忽略 scene；
- 比较训练 seed。

它们是诊断指标，不能替代最终的 HOSI sampling 指标和 human study。

### 8.2 完整生成评估

设置：

```yaml
sampling.enabled: true
```

评估器会执行完整 500-step ancestral sampling，并在 `sample/*` 命名空间报告生成结果指标。默认关闭是因为该过程远慢于固定噪声诊断。

### 8.3 指标

公共指标：

- human MPJPE，单位米；
- human rotation L1；
- pelvis/scene goal error；
- denoised 与 sampled 指标分开记录。

HOI 指标：

- object translation error；
- 投影回 SO(3) 后的 object rotation geodesic error；
- object goal error；
- contact MAE / F1；
- contact-conditioned hand-object relative error。

HSI 指标：

- scene joint penetration rate；
- scene-condition effect：对同一个 noisy input 分别启用和关闭 scene 后，两次预测的距离。

评估 JSON 会保存 checkpoint epoch、完整配置、可用/选中 window 数和所有指标。

评估配置：

- [code/config/config_eval_hoi_prior.yaml](code/config/config_eval_hoi_prior.yaml)
- [code/config/config_eval_hsi_prior.yaml](code/config/config_eval_hsi_prior.yaml)
- [code/config/dataset/hoi_prior_test.yaml](code/config/dataset/hoi_prior_test.yaml)
- [code/config/dataset/hsi_prior_eval.yaml](code/config/dataset/hsi_prior_eval.yaml)

## 9. 使用命令

从 `code/` 目录运行。

### 9.1 训练

```bash
cd code

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_prior.py --config-name config_train_hoi_prior

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_prior.py --config-name config_train_hsi_prior
```

恢复训练：

```bash
python train_prior.py --config-name config_train_hoi_prior \
  resume_from=../results/priors/hoi_prior/checkpoints/last.pth
```

### 9.2 评估

```bash
python evaluate_prior.py --config-name config_eval_hoi_prior \
  checkpoint=../results/priors/hoi_prior/checkpoints/best.pth

python evaluate_prior.py --config-name config_eval_hsi_prior \
  checkpoint=../results/priors/hsi_prior/checkpoints/best.pth
```

完整 sampling：

```bash
python evaluate_prior.py --config-name config_eval_hsi_prior \
  checkpoint=../results/priors/hsi_prior/checkpoints/best.pth \
  sampling.enabled=true sampling.max_batches=32
```

更完整的训练协议和论文消融建议见 [docs/independent_priors.md](docs/independent_priors.md)。

## 10. 测试与验证状态

新增测试：[tests/test_independent_priors.py](tests/test_independent_priors.py)

自动测试覆盖：

1. HOI=232、HSI=216 的 state contract；
2. sequence split 不泄漏；
3. group-balanced validation subset；
4. HSI loss 不产生 dummy object loss；
5. HOI 架构不含 scene；
6. 结构化 checkpoint 严格 round-trip。

已经完成的验证：

- `python -m compileall` 通过；
- 6 个 `unittest` 全部通过；
- 使用真实 OMOMO 数据完成 scene-free HOI 的单步训练和 validation；
- 使用真实 LINGO 数据完成 216 维 HSI 的单步训练和 scene-held-out validation；
- 两个 prior 均完成固定噪声 evaluator smoke test；
- 两个 prior 均完成缩短为 5 步的 ancestral sampling smoke test；
- structured checkpoint 已实际保存并回载。

烟雾测试使用缩小模型、batch size 1 和单步训练，只用于证明数据流、反向传播、checkpoint 和评估代码可运行。其 loss 和指标**不具有论文实验意义**。

受当前执行沙箱限制，PyTorch 无法实际使用 GPU，因而多卡 NCCL/DDP 路径尚未在本次会话中运行。正式默认规模模型也尚未完成 501 epochs 训练，目前没有可用于论文汇报的正式 prior 权重或最终数值。

## 11. Git 版本管理

所有改动位于分支：

```text
feature/independent-hoi-hsi-priors
```

提交记录：

```text
edb6318  add independent HOI and HSI prior training
0f20915  add prior evaluation and reproducibility protocol
84cc8d7  harden independent prior runtime contracts
```

三个提交分别对应：

1. 模型、dataset、loss、训练器、配置和 checkpoint；
2. 评估器、单元测试和实验协议文档；
3. DDP unused-parameter 与 validation contract 加固。

## 12. 对后续论文工作的影响

当前代码完成的是“两个独立且条件边界清晰的 prior”，还没有实现最终 HOI/HSI composition。后续组合阶段应遵守：

1. HOI 作为完整 human-object-contact 生成 prior；
2. HSI 只对 human projection 提供 scene-aware score correction；
3. object-scene collision 必须由额外 SDF/occupancy guidance 约束，因为 HSI prior 没有物体；
4. 不应直接对两个 raw `x0` 做无结构线性平均；
5. 必须加入 scene intervention 实验：固定文本、物体和初始化，只改变障碍布局，验证全局路径改变而局部 HOI 关系保持。

正式投稿实验至少应使用 3 个随机种子，保存每个 seed 的 resolved config、best checkpoint、evaluation JSON 和 Git commit hash。模型选择只能使用规定的 validation partition，不能根据 OMOMO test 或 held-out LINGO scene 的结果反向选 checkpoint。
