# SatFormer 张量补全

本目录用于在卫星星间路径流量张量上运行 SatFormer。

```text
sat_path_bytes_mb_tensor.npy: 120 x 120 x 60
sat_connectivity_tensor_dynamic_60s_1000ms.npz: 120 x 120 x 60
```

拓扑张量中 `1` 表示对应时刻两颗卫星存在 ISL 邻接关系，`0` 表示不存在 ISL。`.npz` 默认读取键名为 `sat_connectivity` 的数组。

## 模型结构

实现保留 SatFormer 的主体结构：

```text
masked traffic tensor
-> Encoder Spatio-Temporal Modules
-> Transfer Module
-> Decoder Spatio-Temporal Modules
-> recovered traffic
```

每个 Spatio-Temporal Module 包含：

- 两层归一化 OD-GCN；
- SatFormer block: `LayerNorm -> ASSIT -> LayerNorm -> MLP`；
- 残差连接。

当前 OD-GCN 使用 `[source, destination, channel]` 隐表示，同时沿源卫星维度和目的卫星维度传播拓扑信息：

```text
source axis:      A_norm @ H
destination axis: H @ A_norm^T
```

ASSIT 使用局部 OD region、多头注意力、中心窗口 mask 和自适应稀疏门控。Transfer Module 对每个 OD pair 的时间序列做 temporal attention。

距离/时延权重矩阵 `W` 不使用，因为 `sat_path_bytes_mb_tensor.npy` 已经是真实星间路径流量矩阵，不需要再融合额外权重矩阵。

## 训练方式

论文的整体任务可以理解为：

```text
输入: masked N x N x T 流量张量
输出: recovered N x N x T 流量张量
```

但在当前 120 星、60 时间步、OD-aware 隐表示、ASSIT attention 和 10 层 encoder/decoder 设置下，整张 `120 x 120 x 60` full-batch 反向传播显存和时间开销过大。

因此默认训练采用更合理的工程化复现方式：

```text
输入: masked history window [N, N, history_window]
输出: target_time 的恢复矩阵 [N, N]
loss: 只在该 target_time 的训练观测项上计算
```

每个 epoch 会遍历训练集中出现的时间步；每个 optimizer step 预测一个目标时间步的完整 `N x N` 矩阵，然后在该时间步的训练 entries 上计算 MSE loss。默认 `history_window=8`；设置 `--history-window 0` 表示使用从时间 0 到目标时间的完整历史窗口。

## 数据划分和指标

划分方式参考 CoSTCo 项目：

- 只使用有限且非零的 entries 作为候选样本；
- 原始零值不参与 train/validation/test；
- observed subset 作为训练项；
- validation/test 从未观测项中划分；
- split 保存到 `splits/`；
- 结果保存到 `results/`。

最终评价指标使用原始流量尺度上的 NMAE 和 NRMSE：

```text
NMAE  = sum(abs(y_true - y_pred)) / sum(abs(y_true))
NRMSE = sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))
```

训练目标默认使用 `max(train_values)` 归一化，所以训练日志里的 loss 是归一化尺度上的 MSE，不是 MB 原始尺度误差。

## 环境

建议在单独环境中安装：

```bash
conda activate TZ-Satformer
pip install -r requirements.txt
```

如果使用 CUDA 12.8 的 PyTorch，请优先按 PyTorch 官方 cu128 index 安装对应版本。

## 运行

在 `Satfomer/` 目录下直接运行默认实验：

```bash
python run_sat_tensor_experiment.py
```

常用参数示例：

```bash
python run_sat_tensor_experiment.py \
  --missing-rate 0.90 \
  --epochs 200 \
  --feature-dim 128 \
  --gcn-hidden-dim 128 \
  --num-modules 10 \
  --heads 8 \
  --history-window 8 \
  --batch-size 512 \
  --lr 0.001 \
  --target-normalization max \
  --seed 3
```

快速调试一轮：

```bash
python run_sat_tensor_experiment.py \
  --missing-rate 0.90 \
  --epochs 1 \
  --max-train-steps-per-epoch 1 \
  --log-every 1
```

默认最终只评估 `test`，避免训练结束后长时间扫完整 train/val/test。需要全部评估时使用：

```bash
python run_sat_tensor_experiment.py --eval-splits all
```

可选值：

```text
--eval-splits none
--eval-splits test
--eval-splits val-test
--eval-splits all
```

后台运行示例：

```bash
mkdir -p logs
nohup python -u run_sat_tensor_experiment.py \
  --missing-rate 0.90 \
  --epochs 200 \
  --log-every 1 \
  --eval-splits test \
  > logs/satformer_mr90_seed3.log 2>&1 &
```

查看日志：

```bash
tail -f logs/satformer_mr90_seed3.log
```

## 输出

默认输出示例：

```text
splits/random_observed10_val10_seed_3.npz
results/random_observed10_val10_seed3_dim128_layers10_batch512_hist8_norm_max.json
```

最终从 JSON 的 `test` 字段读取 NMAE 和 NRMSE。
