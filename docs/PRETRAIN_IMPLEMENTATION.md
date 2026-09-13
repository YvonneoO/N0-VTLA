# N0-VTLA Human Pretrain Implementation

整理日期：2026-09-12。代码核对基准：本地 `main`，`7c9aea8`。
本文汇总当前 human-data pretrain 实现，代码路径均相对本仓库根目录。
实验数字来自已有运行记录；本次整理没有重新训练或查询服务器实时状态。

## Outline

1. [范围与记录来源](#1-范围与记录来源)
2. [模型架构与代码实现](#2-模型架构与代码实现)
3. [训练目标与训练循环](#3-训练目标与训练循环)
4. [Data Preprocess](#4-data-preprocess)
5. [数据处理结果与例子](#5-数据处理结果与例子)
6. [Smoke Run](#6-smoke-run)
7. [后续短训练与评估协议](#7-后续短训练与评估协议)
8. [复跑入口与已知限制](#8-复跑入口与已知限制)

## 1. 范围与记录来源

### 1.1 这里的 pretrain 指什么

当前实现是 action-free **Stage-1 tactile predictor grounding**，沿用项目记录中
N0-VTLA paper Section 4.2 的阶段划分。它冻结已有 VLA base，使用 human RGB +
tactile 学习未来触觉变化，不需要机器人 action labels。

它不是 Section 4.1 的完整 base pretraining，也不是 robot post-training。
实际运行从 `n0-vtla-base` 加载，包括 checkpoint 中已有的 tactile projection 和
predictor；只有新增 reconstruction head 随机初始化。因此应描述为已有模型上的
human-domain Stage-1 adaptation，不能声称 tactile pathway 从零初始化复现。

### 1.2 原始记录各自负责什么

| 文档 | 内容 | 本文采用方式 |
|---|---|---|
| `docs/STAGE1_PREDICTOR_PRETRAINING.md` | Stage-1 实现、loss、配置、偏差与限制 | 架构和训练实现来源 |
| `docs/ITW_PRESSURE_STAGE1_SMOKE.md` | 两个 episode、3-update 端到端 smoke 的确切结果 | 保留历史实验数字与路径 |
| `docs/HUMAN_PRETRAIN_AND_ROBOT_POSTTRAIN.md` | Human study 与独立 robot study、实验对照和审计 | 只提取 human pretrain 部分，不混入 robot adapter/action 设置 |
| `docs/STAGE1_REPORT_RUN.md` | 后续 50/10 episode、2,000-update 协议及初始化评估 | 区分已记录结果和待核实的最终结果 |

原文档保留作为详细记录。特别是其中“仍在运行”“到达 update 950”等表述，
只表示写入时的快照，不代表 2026-09-12 的实时状态。

## 2. 模型架构与代码实现

### 2.1 前向数据流

令 `T_0` 为 episode 第 0 帧触觉图，`T_t` 为当前帧，`T_f=T_(t+50)` 为未来帧。
下面的差分发生在模型图像张量空间，不直接对原始物理压力做预测。

```text
当前三路 RGB + prompt + constant-zero state tokens
    -> 已有 RGB encoder / PaliGemma prefix（冻结，无梯度）
    -> VL context
                                      |
T_t - T_0（每只手分别计算）             | masked pooling -> query conditioner
    -> frozen DINOv2                   |
    -> trainable tactile_proj          v
    -> 拼接有效视图 tokens g -> tactile_kv predictor -> z [B,5,D]
                                  2 layers / 8 heads       |       |
                                                         |       +-> mean pool
                                                         |           -> MLP
                                                         |           -> [B,8,8]
                                                         |           -> L1 loss
                                                         v
                                                   symmetric InfoNCE
                                                         ^
T_f - T_t -> 同一个 encoder + projection（目标分支 no_grad） |
    -> 按有效视图平均 -> z_star [B,10,D] --------------------+
    -> 图像差分按有效视图、RGB channel 平均 -> pool 8x8 -> L1 target
```

`D` 为已有 PaliGemma 的 hidden size，构造时从模型配置读取。目标的 10 tokens
与预测的 5 tokens 无需一一对应：InfoNCE 两边都先 mean-pool 成一个向量。

### 2.2 已有模块与本次补充

这里的“已有”指继承的 upstream 代码；“补充/修改”指本 fork 的 human Stage-1
工作，不把同一个文件里的其他 robot/inference 修改算作 pretrain 新增内容。

| 模块 | 来源与当前作用 | Repo 文件与入口 |
|---|---|---|
| VLA base、RGB/VL prefix、action expert | 已有。Stage 1 仅运行 prefix；action expert 保留在模型和 checkpoint 中，但不进入此 loss | `n0vtla/models_pytorch/pi0_pytorch.py`：`PI0Pytorch`；`n0vtla/models_pytorch/n0vtla_policy.py`：`_prefix_forward` |
| Tactile backbone | 已有 DINOv2-base，768-D，冻结；每图 1 CLS + 3x3 pooled spatial tokens，共 10 tokens | `n0vtla/models_pytorch/tactile_encoder.py`：`FrozenDINOv2TactileEncoder` |
| Tactile projection | 已有 `Linear(768,D)`，Stage 1 允许训练 | 同上：`tactile_proj`、`project` |
| Predictor | 已有 `TactileActionPredictor`；当前选 `tactile_kv`、5 queries、2 layers、8 heads | `n0vtla/models_pytorch/tactile_predictor.py`：`_forward_tactile_kv` |
| 当前触觉差分与 VL 条件连接 | 复用已有 `_build_g`、`_compute_z`；有效视图 token 拼接并保留 mask | `n0vtla/models_pytorch/n0vtla_policy.py` |
| Future target 与 Stage-1 loss | 本 fork 补充目标分支、valid-mask/global InfoNCE、reconstruction loss 和独立 forward | `n0vtla/models_pytorch/n0vtla_policy.py`：`_build_future_target`、`_stage1_infonce_loss`、`forward_stage1` |
| Reconstruction head | 本 fork 新增 `D -> 4D -> 64`、GELU，两层 MLP；reshape 成 8x8 | `n0vtla/models_pytorch/tactile_recon_head.py`：`TactileReconHead` |
| Stage-1 配置 | 在已有配置系统中新增 `vtla_stage1_predictor_pretrain`，隔离 action 数据变换 | `n0vtla/training/config.py` |
| Human observation transform | 修改已有 canonical transform，加入 `Stage1ObservationOnly` 与未来帧 clamp-mask 处理 | `n0vtla/policies/canonical_tactile_policy.py` |
| Loader、图像/文本变换 | 复用已有 MP4/parquet loader、图像 resize、tokenizer；不另写一套训练 loader | `n0vtla/training/data_loader.py`、`n0vtla/transforms.py`、`n0vtla/models/tokenizer.py` |
| 独立训练入口 | 本 fork 新增，不调用 action-supervised train forward | `scripts/train_stage1_predictor.py` |
| ITW 数据与报告 | 本 fork 新增，见第 4、7 节 | `scripts/itw_pressure.py`、`scripts/itw_tactile_smoke_adapter.py`、`scripts/eval_stage1_report.py` |

`tactile_kv` 中只有触觉 `g` 作为 key/value。VL context 的 masked mean 经线性层
加到 attention query；不直接加进 query residual，也不作为 value。这不意味着模型
不依赖 RGB：RGB 仍会通过 query conditioner 改变触觉 attention。

### 2.3 冻结、训练与 action-free 边界

训练入口仅解冻三个参数前缀：

```text
tactile_encoder.tactile_proj.
tactile_predictor.
tactile_recon_head.
```

其余 base/DINOv2 参数冻结；base 保持 eval，prefix forward 额外置于 `no_grad`。
已验证 smoke 的参数量为 **123,818,048 trainable / 3,705,436,177 frozen**。

`forward_stage1(observation)` 不接收 actions，不构造 action suffix，也没有
flow-matching action loss。`Stage1ObservationOnly` 提供接口所需的零 state/action，
真实 action 列不会用于监督；Pi0.5 tokenizer 仍看见 constant-zero state tokens。
当前配置 `prompt_from_task=False`，使用 `Perform the task.`，而非逐 episode 的任务文本。

## 3. 训练目标与训练循环

### 3.1 两个监督目标

1. **Latent target**：对每个有效视图的 `T_f - T_t` 编码，再按视图取平均得到
   `z_star`。目标分支 stop-gradient，但与当前分支共享会更新的 projection，所以
   target 会随训练变化；没有独立的 EMA target encoder。
2. **Reconstruction target**：同一未来差分按有效视图和 RGB channel 平均，
   adaptive average pooling 得到 8x8 场。模型预测的是双手平均的粗粒度变化，
   不是两张可独立恢复的 full-hand 高分辨率图，也不是物理力单位输出。

InfoNCE 对预测和目标分别 token mean-pool、L2 normalize，以 cosine similarity
除以 `temperature=0.07` 构造 logits；两个方向交叉熵取平均。

```text
L_total = L_symmetric_InfoNCE + 0.5 * L1(predicted_8x8, target_8x8)
```

### 3.2 无效样本、DDP 与更新计数

- 当前与未来必须有有效触觉视图；缺失视图和 episode 尾部被 clamp 的未来帧不作为 target/negative。
- baseline 通过负时间偏移钳到 episode 第 0 帧，这是预期行为，不按 future-clamp 规则丢弃。
- DDP 在筛 valid rows 前 gather 等长 batch；预测使用支持 autograd 的 gather，形成 global-batch negatives。
- Reconstruction 按全局 valid count 归一化，允许某个 rank 无有效样本。
- 全局少于两个有效样本时 InfoNCE 为零；若仍有一个有效样本，reconstruction 可以训练。
- 全局无有效样本时不执行 AdamW/weight decay；连续一个 epoch 都无有效 batch 则报错。
- `global_step` 表示已完成 optimizer updates，不是已读取 batch 数。

实现位置：`scripts/train_stage1_predictor.py` 和
`n0vtla/models_pytorch/n0vtla_policy.py`。CPU 双进程 gloo 回归已验证；lab GPU-DDP
没有完成等价验证，不能将单 GPU smoke 当作多 GPU 通过。

### 3.3 当前训练设置

| 项目 | 当前值 |
|---|---|
| Config | `vtla_stage1_predictor_pretrain` |
| Global batch / workers | 64 / 8 |
| Future horizon | 50 frames，30 Hz 下约 1.67 秒 |
| Latent tokens / predictor | 5 / `tactile_kv` |
| Reconstruction / temperature | 8x8，weight 0.5 / 0.07 |
| Optimizer / grad clip | AdamW / 1.0 |
| LR schedule | warmup 500；peak 1e-4；20,000 步衰减到 1e-5 |
| 默认总步数 | 20,000；历史 smoke 覆盖为 3，short run 覆盖为 2,000 |
| Precision 配置 | `bfloat16`；不表示所有权重、累积和 loss 都是 BF16 |
| Log / save interval | 50 / 2,000；最后一个 update 也保存 |
| W&B / EMA | W&B 默认关闭；配置保留 `ema_decay=0.999`，本 trainer 未实现 EMA |

Checkpoint 包含全模型、optimizer 和 metadata，因冻结 base 也被保存而较大。
Resume 恢复权重、optimizer 与 update count，不恢复精确 RNG/data cursor。
未实现 checkpoint retention pruning，不应假定 `keep_period` 会自动清理磁盘。

## 4. Data Preprocess

### 4.1 原始输入

每个 recording/episode 目录使用以下文件；不依赖 depth，也不要求从 H5 合成 action。

```text
<episode_uuid>/
  rgb_head.mp4       rgb_head.csv
  wrist_left.mp4     wrist_left.csv
  wrist_right.mp4    wrist_right.csv
  left_hand_data.npz
  right_hand_data.npz
  task_info.json
```

Camera CSV 使用 `frame_index` 和 `timestamp_s`，允许逗号后的空格。
每个 glove NPZ 使用 `timestamps` 和 `tactile_<pad_id>` 压力阵列，shape 为
`[time, pad_height, pad_width]`。只读压力，不使用两路 shear。
每手 15 pads / 880 taxels；pad 顺序固定为：

```text
0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 18
```

### 4.2 Train-only 固定归一化

代码：`scripts/itw_pressure.py` 的 `fit_normalization`、`load_normalization`、
`normalize_pressure`。参考另一个仓库 **tacWAM** 的
`tacwam/cosmos_tactile/pad_data.py::fit_pad_normalization`，参考 commit
`1e4ad399718e8adc80aaf8b0185d17cd0450766f`。

1. 按 official recording split 只选 train IDs，并按 episode 排序。
2. seed=42；每个 recording、每只手抽 `min(4,n_frames)` 帧，有放回；取这些帧所有 taxels 的有限压力值。
3. 对每手每 pad 分别汇总，计算 baseline=P5，raw_scale=max(P99.9-P5,1e-6)。
4. 收集 raw_scale>1e-5 的值；global floor 为其 median 的 5%，至少 1e-5；若无此类值，floor=1e-3。
5. final_scale=max(raw_scale,global_floor)，`normal=clip((raw-baseline)/final_scale,-1,8)`。

共 **30 组参数**：左手 15 + 右手 15；同一物理 pad 内所有 taxels 共用参数，
跨所有 episode 固定。不是 per-episode、per-frame 或左右手共享 max。
Validation/test 复用 train 参数。重新 fit 必须生成新版本并重新渲染，不能混用。

为保持参考统计格式，还保存 contact threshold：raw threshold 为
`max(median + 4*MAD, baseline + 0.1*raw_scale)`，归一化后至少 0.1。
当前 MP4 渲染/Stage-1 loss **不使用此 threshold 截断压力或判定 contact**。

Fit 阶段仅收集有限值；当前渲染对非有限压力报错，不自动补零。参考代码会跳过
某些读取异常，本 fitter 对损坏输入采取报错，不能将错误输入静默视为有效统计。

### 4.3 RGB / tactile 时间对齐

代码：`scripts/itw_pressure.py` 的 `aligned_timeline`、`nearest_indices`、
`write_aligned_rgb`。

- 对三路 RGB 和两路 glove timestamps 取公共时间交集，转换成整数纳秒。
- 在交集上建 30 Hz 网格：使用 `round(i * 1e9/30)` 偏移，时间小于交集终点。
- 各流 nearest-neighbor 选样，相同距离取较早的一项；camera 取 CSV 的实际 `frame_index`。
- Camera 误差上限 17.5 ms，tactile 上限 20 ms；公共序列必须超过 50 帧。
- RGB 按映射重新编码，保持长宽比 letterbox 到 224x224；tactile 使用同一网格索引。
- 不删除坏 tick 后压缩时间。当前采用整 episode 严格 gate，比 tacWAM window-level 筛选更保守。

Adapter 的 `--split-manifest` 与 `--split` 必须配对；显式 split 模式先过滤 IDs、
再检查对齐，记录拒绝原因，直到达到指定数量。不足时写 `selection_failure.json`
并报错。支持 `--extra-raw-root` 依次补充日期；未匹配 split 的 IDs 不选用。
未传 split 参数的旧 smoke 模式只取前 N 个目录，不能直接当作正式 train 选集流程。

### 4.4 Pressure rasterization 与 MP4

代码：`scripts/itw_pressure.py::pressure_rgb/write_pressure_video/video_writer`；
布局复用 `scripts/itw_tactile_smoke_adapter.py::TACTILE_SLOT_LAYOUT`。

| 项目 | 实现 |
|---|---|
| 布局 | 固定简化手型 pad arrangement，左手镜像、palm 旋转；不是连续整片皮肤，也非 QC 图的精确外轮廓 |
| 画布 | 每手独立 224x224；无传感器区域黑色 |
| 通道 | `R=G=B=round((clip(normal,-1,8)+1)*255/9)` |
| 值映射例子 | normal=-1 -> 0；normal=0 -> 28；normal=1 -> 57；normal=8 -> 255 |
| 编码 | 30 fps、H.264 baseline、yuv420p、CRF18、faststart |
| 内存 | writer 逐帧输出，不堆积全部渲染帧；每只手的 pressure arrays 仍加载到内存 |

零归一化压力是深灰，不再是旧 pressure/shear 编码的青蓝色。旧 helper 和
`scripts/preview_itw_full_hand.py` 保留历史预览用途，不是当前训练编码入口。
不做逐 episode 自动拉伸；黑背景的静态部分在时间差分中抵消。

MP4 是额外的有损 transport：每个灰度级约 0.0353 normalized units，随后还有
H.264 误差。它并不等同于 tacWAM 直接使用 float pressure tensors。
训练直接从 MP4 按需解码，不需要预先保存逐帧 JPEG/PNG；当前实现是离线渲染，
不是在训练 loader 中 online rasterize。

### 4.5 Canonical 输出与 loader

代码：`scripts/itw_tactile_smoke_adapter.py`、`n0vtla/policies/canonical_schema.py`。

```text
<dataset>/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/<canonical_image_key>/episode_000000.mp4
  alignment/episode_000000.npz
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  meta/source_episodes.json
  meta/selection.json
  meta/tactile_normalization.json
  meta/tactile_encoding.json
```

五个真实视频 key 是 `observation.image.` 加以下后缀：

```text
third_view
left_wrist_view
right_wrist_view
left_wrist_left_tactile
right_wrist_right_tactile
```

模型的四个 canonical tactile slots 中，只有两个对应真实手套；另外两个由
transform 生成 placeholder + false mask，不伪装成真实观测。图像转换成模型短 key，
避免 tactile 被当作 RGB 输进 prefix。Loader 读取 baseline/current/future 三帧栈。

Parquet 每帧一行：`timestamp`、`frame_index`、`episode_index`、全局 `index`、
`task_index`，以及 32-D float32 零 `observation.state` / `action`、全 false 的
32-D `action_mask`。这些是兼容接口的占位，不是 human action labels。
Action-free loader 不请求 action sequence、不做 action delta/normalization。

`selection.json` 保留真实 split 和 source IDs；`info.json` 中的 LeRobot
`splits.train` 是本地索引范围，不能据此把 validation 数据误判成研究协议的 train。

## 5. 数据处理结果与例子

### 5.1 固定 normalization asset

历史使用的 lab 文件：

```text
/DATA2/qianqian/n0vtla_data_manifests/normalization_0803_train_v1.json
```

拟合来源：08/03 的 **453 个 official train recordings**；scale 范围
**0.00858294–0.971982**。这是开发子集的统计，方法沿用 tacWAM，但不是照搬
tacWAM 全语料的旧数值。文件 SHA256：

```text
ed9c91636d8f22c8d976bffa5516477b904798dab0df08a2fca1f5d021d78e6f
```

### 5.2 两个 episode 的处理结果

| Episode UUID | Split | 输出对齐帧数 | 时长 |
|---|---|---:|---:|
| `02452e46-f019-4bda-84e3-86f54ca47f44` | train | 448 | 14.93 s |
| `02947400-4456-4f0a-a528-c57e0e0378b3` | train | 650 | 21.67 s |

共 1,098 个时间步、10 个 MP4；五路流合计 **5,490 帧**完整解码通过，均为
224x224 / 30 fps。视频总大小 **4,563,298 bytes**；不含全部 parquet/metadata。

第一段各流最大对齐误差的实测例子：head 8.574 ms、left wrist 13.580 ms、
right wrist 1.044 ms、left tactile 19.085 ms、right tactile 18.473 ms，均通过 gate。
这些是 timestamp-nearest 误差，不额外证明传感器物理时钟已校准。

Lab 数据与双手预览：

```text
/DATA2/qianqian/n0vtla_itw_canonical_smoke/itw08-03_pressure_tacwam_v1/
/DATA2/qianqian/n0vtla_itw_canonical_smoke/itw08-03_pressure_tacwam_v1/pressure_hands_preview.mp4
```

已下载到本机的预览在 workspace 的 `n0vtla_lab_tools/previews/pressure_tacwam_v1/`
目录中，不是本 repo 的 tracked artifact。预览与训练输入保持同一固定尺度；
弱接触偏暗，不能通过逐段增强亮度掩盖这个数值特性。

## 6. Smoke Run

### 6.1 实际环境与命令

2026-09-07 在 lab Docker `n0vtla`、GPU 4 完成。训练代码 commit 为
`a88d4e9bac2cc726009b16e684c940960514eae8`。
Blackwell 环境当时已使用 `torch==2.7.1+cu128`；此前 cu126 wheel 的 GPU kernel
不兼容问题已处理。这是运行时记录，不代表旧 Docker image tag 自动更新。

在 Docker 的 `/DATA2/qianqian/N0-VTLA` 中执行：

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu \
VTLA_DATASET_PATH=/DATA2/qianqian/n0vtla_itw_canonical_smoke/itw08-03_pressure_tacwam_v1 \
VTLA_ASSET_ID=itw08-03_pressure_tacwam_v1 \
VTLA_PRETRAINED_CHECKPOINT=/DATA2/qianqian/N0-VTLA/checkpoints/n0-vtla-base \
python scripts/train_stage1_predictor.py vtla_stage1_predictor_pretrain \
  --exp-name=itw0803_pressure_tacwam_v1_smoke_20260907 --num-train-steps=3
```

这是历史确切命令；重跑应使用新 exp-name 并先确认 GPU 空闲，不覆盖已有实验。
仅 duration 为 smoke 覆盖，其余配置见 3.3。模型初始化约 55 秒，更新及保存约 72 秒。

### 6.2 实测结果

| 记录区间 | InfoNCE | Reconstruction L1 | Total | Valid / batch 64 | Grad norm |
|---|---:|---:|---:|---:|---:|
| Update 1 | 4.0521 | 0.1447 | 4.1245 | 55 | 36.4582 |
| Updates 2–3 平均 | 4.0253 | 0.1446 | 4.0976 | 58.5 | 25.7655 |

第二行不是单独 step 3 的指标。55/64=85.94%，58.5/64=91.41%，三步合计
172/192=89.58%，不能把 valid count 当作百分比。日志 LR 显示 `0.0000` 是
四位小数舍入，不是学习率真的为零。

已验证：进程正常结束；metadata 记录 3 completed updates；38 个 trainable tensors
全部有限值；抽查 projection/predictor 权重相对初始化最大变化约 1.20e-6；抽查
frozen action projection 未变。未对全部 frozen 参数逐个比较，也未做此次 checkpoint
的完整重启续训测试。

Checkpoint 含模型、optimizer、metadata，总计 **9,313,676,066 bytes / 8.67 GiB**。

```text
Log:
/DATA2/qianqian/n0vtla_logs/stage1_itw0803_pressure_tacwam_v1_smoke_20260907.log

Checkpoint:
/DATA2/qianqian/N0-VTLA/checkpoints/vtla_stage1_predictor_pretrain/itw0803_pressure_tacwam_v1_smoke_20260907/3/
```

### 6.3 测试覆盖与解释

历史 smoke 配套验证包括 `tests/test_itw_pressure.py` 的 4 项测试，以及
`tests/test_stage1.py` 的 8 项测试：固定统计/映射、对齐 gap gate、loss/gradient、
empty/singleton batch、CPU 两进程 DDP、action-free transform、checkpoint update 计数等。
后续 `tests/test_stage1_report.py` 覆盖报告逻辑；不将它冒充初次 smoke 的 12 项测试。

“跑通”证明数据读取、前向、反向、optimizer 和保存链路可执行。
三步 loss 变化不证明收敛、有效 contact forecasting 或 robot task success。
Loader 原有 read-only NumPy-to-tensor warning 仍存在；缺失 robot norm stats 的日志
不表示 tactile normalization 没应用，因为 tactile 参数已经写入视频。

## 7. 后续短训练与评估协议

### 7.1 不要与两段 smoke 混淆

后续入口为 `scripts/run_itw_stage1_report.sh`；评估和可视化在
`scripts/eval_stage1_report.py`，独立记录为 `docs/STAGE1_REPORT_RUN.md`。

| 项目 | 已记录协议/结果 |
|---|---|
| Train | 08/03 的 50 个 official train episodes，21,338 aligned frames |
| Validation | 10 个 official validation episodes，5,451 aligned frames，与 train 无 UUID 重叠 |
| Validation 候选日期 | 按 08/03、08/06、08/07、08/10、08/11 补足；不是纯同日期验证 |
| 对齐筛选 | 达到目标数量前拒绝 29 个 train、25 个 validation candidates |
| Norm | 继续用 453 个 08/03 train recordings 拟合的固定文件；没有 validation 参与 |
| Duration | 2,000 updates；训练 batch 与其他设置不变 |
| Eval | batch 16，无 augmentation；每 5 帧选一个 `t+50` 不越界样本，共 994 个 |

Norm 拟合集大于实际训练的 50 个 episode，属于协议的一部分，不是 validation
leakage。按 UUID 分离不保证 unseen-task/unseen-person 隔离。

### 7.2 已有评估数字与状态边界

| 初始化评估 | 模型 MAE | Zero-change MAE |
|---|---:|---:|
| 全部 994 samples | 0.1550473 | 0.00148208 |
| Active subset，102 samples | 0.1585377 | 0.00714330 |

Active 定义为真实变化场 mean absolute value > 1/255，单位为模型图像差分，
不是物理 contact detector。初始化 reconstruction head 是随机的，所以训练后
优于初始化并不够，还需比较 zero-change baseline、macro-episode MAE 与 active MAE。

`docs/HUMAN_PRETRAIN_AND_ROBOT_POSTTRAIN.md` 曾记录 update 950 的区间训练均值：
InfoNCE 0.9754、reconstruction 0.0423。该数字不是 held-out improvement。
目前用于本次汇总的文档没有确认最终 after/comparison 数字，因此本文不声明
2,000 步训练和后评估已经完成，也不把旧文档的“in progress”当作当前状态。

### 7.3 报告产物

Lab report root：`/DATA2/qianqian/n0vtla_reports/itw0803_pressure_report_v1_2000/`。
协议预期产生 `before/metrics.json`、`before/predictions.npz`、`training.log`、
`after/metrics.json`、`after/predictions.npz`、`comparison/comparison.json` 和
`comparison/contact.mp4` / `ordinary.mp4`。列出路径不等于确认文件已经生成。

可视化是实际 head 的 8x8 future-change field：红色增加、蓝色减少、白色零；
before/after/target 共用 +/-0.1 显示范围。指标使用未显示截断的预测。
每 5 个原始帧一张预测，6 fps 播放保持物理时间；contact clip 根据真实 target
energy 选取，而非模型表现。它不是 action rollout 或未来 RGB generation。

## 8. 复跑入口与已知限制

### 8.1 数据准备命令

以下路径用占位符表示。全部计算在 Docker 内执行，代码修改遵循本地 commit/push
到 fork，再由 Docker checkout `git pull --ff-only`；不在 server 直接改代码。

```bash
# 首次 fit：已有固定 asset 时直接复用，不重算、不覆盖。
python scripts/itw_pressure.py \
  --raw-root /path/to/itw08-03 \
  --split-manifest /path/to/tacwam_v10_split.json \
  --output /path/to/new_normalization_version.json

# 输出目录必须是新目录；显式锁定 train split。
python scripts/itw_tactile_smoke_adapter.py \
  /path/to/itw08-03 /path/to/new_train_dataset \
  --normalization /path/to/fixed_normalization.json \
  --split-manifest /path/to/tacwam_v10_split.json --split train --max-episodes 2
```

第 6 节为实际 smoke 命令；第 7 节的 report shell 会准备数据、评估、训练再评估，
不是只读查看结果命令。它不自动恢复训练，遇到部分产物应先检查而不是删目录重来。

### 8.2 不能省略的限制

| 边界 | 当前实现/风险 |
|---|---|
| 从零复现 | 加载已有 tactile weights；不等同于新初始化 pathway |
| Paper 对应 | 项目原记录指出 paper latent count=10、未缩放 cosine；当前保留 5 和 temperature=0.07 |
| 自选实现 | Target stop-gradient、8x8 view/channel-mean reconstruction、MLP、weight=0.5 均需标明，不能当作完整官方 recipe |
| 传感器分布 | 人手压力阵列 rasterization 不等同于 vision-based tactile；frozen DINOv2 能否有效表征需实验 |
| DINO 输入 | 当前是 float 图像差分，encoder 不额外套 uint8 分支的 ImageNet normalization；见 `_dinov2_preprocess` |
| Baseline | Episode frame 0 不一定无接触；其误差会进入当前触觉条件 |
| 数值精度 | 固定 per-pad scaling 不保留跨 pad 绝对压力相等关系；8-bit + H.264 会损失弱信号 |
| 评估解释 | 大量无变化样本可使 zero-change 很强；需同时看 active subset 和 episode-level 指标 |
| 工程边界 | GPU-DDP、精确恢复、长期磁盘 retention 未完成验证/实现；短 smoke 不替代这些检查 |
| Robot transfer | Human Stage-1 不自动解决机器人 action contract、物理同步、robot calibration 与部署安全；另见 robot 文档 |

本次整理只新增实现文档，没有调整模型、训练参数、normalization 或已有数据。
