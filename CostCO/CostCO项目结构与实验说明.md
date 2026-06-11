# CostCO 项目结构与实验说明

本文档用于说明 `CostCO` 目录下当前代码的职责划分、数据流、已有模型功能、文本多模态实验逻辑，以及哪些文件属于可忽略的运行产物。当前项目核心任务是对卫星路径流量张量进行补全，并在 CoSTCo 基线基础上逐步加入动态拓扑 GCN 与文本模态。

## 1. 总体任务

项目输入主要包含两个张量：

- `sat_path_bytes_mb_tensor.npy`：路径流量张量，形状为 `[source, destination, time] = [120, 120, 60]`，单位为 MB。
- `sat_connectivity_tensor_dynamic_60s_1000ms.npz`：动态卫星拓扑邻接矩阵，形状规范化为 `[time, node, node] = [60, 120, 120]`。

训练目标是随机观测一部分非零路径流量，补全剩余非零流量。当前 split 设计是随机传导式补全：

- 仅使用有限且非零的流量条目。
- 零值流量不参与训练、验证和测试。
- split 文件固定后复用，避免不同实验使用不同 mask。
- 指标在原始 MB 尺度上计算。

核心指标为：

- `NMAE = sum(abs(y_true - y_pred)) / sum(abs(y_true))`
- `NRMSE = sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))`

## 2. 顶层文件职责

`CostCO` 顶层包含三类内容：基础模型、数据文件、论文/设计材料。

### 2.1 基础模型代码

- `costco_model.py`
  - 定义原始 CoSTCo 模型。
  - 输入为 source、destination、time 三个 index。
  - 使用三个 mode embedding，经卷积和 dense 层输出路径流量。
  - 提供 `rmse`、`mae`、`nmae`、`nrmse`、`transform_indices` 等公共函数。

- `run_sat_tensor_experiment.py`
  - 原始 CoSTCo 实验入口。
  - 负责加载流量张量、创建或读取随机 split、归一化 target、训练、评估、保存 metrics。
  - 可选使用手工拓扑特征融合，但当前主线更推荐使用 GCN 版本。

- `gcn_costco_model.py`
  - 定义 GCN-CoSTCo 模型。
  - 在 CoSTCo 流量分支之外加入 `TemporalGCNPairLayer`。
  - GCN 分支根据 time index 选择当前 `A_t`，对可训练卫星节点 embedding 做两层图传播。
  - 对 source/destination 节点表示构造 pair 表示：`src, dst, abs(src-dst), src*dst`。

- `run_gcn_sat_tensor_experiment.py`
  - GCN-CoSTCo 实验入口。
  - 不使用文本，仅使用流量张量和动态拓扑邻接矩阵。
  - 是当前多模态实验的重要强基线。

### 2.2 数据文件

- `sat_path_bytes_mb_tensor.npy`
  - 路径流量张量。

- `sat_connectivity_tensor_dynamic_60s_1000ms.npz`
  - 动态拓扑邻接矩阵。
  - 支持的 key 包括 `sat_connectivity`、`arr_0`、`connectivity`、`adjacency`。

### 2.3 论文和设计材料

- `TOWARDS MULTIMODAL TIME SERIES ANOMALY DETECTION W.pdf`
  - MindTS / 多模态时间序列异常检测相关论文。
  - 当前项目借鉴了其内生文本、外生文本、多模态融合、对比对齐和内容压缩思想，但没有照搬任务形式。

- `MULTIMODAL.txt`
  - 对多模态论文或方法的文字描述材料。

- `REQUEST.md`、`已有的模型.txt`
  - 历史说明材料，不参与训练。

## 3. 多模态文本目录结构

多模态扩展集中在：

```text
CostCO/multimodal_text/
```

主要分为四层：

```text
multimodal_text/
  build_satellite_texts.py
  build_texts_deepseek.py
  encode_satellite_texts.py
  run_text_preparation_pipeline.py
  experiment_description.md
  deepseek.env.example
  diagnose_text_embeddings.py
  shared/
  models/
  text_data/
```

## 4. 文本生成流水线

当前文本分为内生文本和外生文本。

### 4.1 内生文本

内生文本描述每个时间片的动态拓扑状态。当前不再使用 DeepSeek 生成内生文本，而是使用确定性模板生成：

```text
A_t 动态拓扑邻接矩阵
  -> 结构化拓扑统计
  -> 统一 deterministic template
  -> text_data/endo_texts.json
```

入口文件：

- `build_satellite_texts.py`

它生成：

- `text_data/time_stats_topo_only.json`
- `text_data/endo_texts.json`

内生文本只使用拓扑信息，不读取流量张量，不读取随机 split，不使用 train/val/test mask，因此不包含流量泄露风险。

当前内生文本涉及的拓扑统计包括：

- 时间片编号和归一化时间相位。
- 卫星数量和无向 ISL 链路数。
- 是否连通。
- 平均最短路、最短路方差、网络直径、长路径比例。
- 代数连通度 `lambda2` 及其相邻时间变化。
- 最近 5 步链路变化均值。
- Top-3 edge betweenness 瓶颈链路及其 shortest-path usage 占比。

### 4.2 外生文本

外生文本描述整个实验都成立的全局配置背景，例如仿真设置、星座参数、链路容量、路由机制和张量语义。

输入文件：

- `experiment_description.md`

生成入口：

- `build_texts_deepseek.py`

输出文件：

- `text_data/exo_text_segments.json`

当前 `build_texts_deepseek.py` 只负责外生文本生成。

### 4.3 文本编码

文本编码入口：

- `encode_satellite_texts.py`

默认本地模型路径：

```text
CostCO/multimodal_text/models/all-MiniLM-L6-v2
```

生成：

- `text_data/endo_text_embeddings.npy`
- `text_data/exo_text_embeddings.npy`
- `text_data/text_embedding_metadata.json`

编码逻辑：

- 优先使用 `sentence-transformers` 加载本地 `all-MiniLM-L6-v2`。
- 如果本地 SentenceTransformer pooling 配置不兼容，则回退到 `transformers AutoTokenizer + AutoModel + mean pooling`。
- 默认 L2 normalize。

### 4.4 一键流水线

入口：

- `run_text_preparation_pipeline.py`

执行顺序：

```text
stats:    build_satellite_texts.py
deepseek: build_texts_deepseek.py
encode:   encode_satellite_texts.py
```

常用命令：

```bash
cd CostCO/multimodal_text

python run_text_preparation_pipeline.py \
  --stage all \
  --topology-path ../sat_connectivity_tensor_dynamic_60s_1000ms.npz \
  --env-path deepseek.env \
  --config-path experiment_description.md \
  --embedding-batch-size 32
```

## 5. 多模态模型结构

多模态模型分为 CoSTCo+Text 和 GCN-CoSTCo+Text 两组。

### 5.1 CoSTCo + Text

目录：

```text
CostCO/multimodal_text/models/costco_text/
```

文件：

- `text_costco_model.py`
- `run_text_sat_tensor_experiment.py`

结构：

```text
CoSTCo flow branch
Text branch
  -> fusion
  -> projection
[flow_x || text_x]
  -> prediction
```

### 5.2 GCN-CoSTCo + Text

目录：

```text
CostCO/multimodal_text/models/gcn_costco_text/
```

文件：

- `gcn_text_costco_model.py`
- `run_gcn_text_sat_tensor_experiment.py`

结构：

```text
CoSTCo flow branch
GCN topology branch
Text branch
  -> fusion
  -> projection
[flow_x || graph_x || text_x]
  -> prediction
```

这是当前多模态实验主线。

## 6. 共享模块

目录：

```text
CostCO/multimodal_text/shared/
```

### 6.1 `mindtext_layers.py`

负责文本融合层和对比学习层。

主要类和函数：

- `MindTextFusionLayer`
  - 统一实现不同 `text_stage`。

- `TemporalSemanticAlignmentLayer`
  - 实现 flow-text / graph-text 的 temporal InfoNCE 对齐损失。

- `create_text_projection`
  - 将文本 embedding 投影到训练所需维度。
  - 当前使用 `Dense -> LayerNorm -> GELU -> Dense -> LayerNorm`，防止高维文本压制 flow/graph。

### 6.2 `experiment_utils.py`

负责实验辅助功能：

- 加载 text embeddings。
- 文本消融：`real`、`endo_only`、`exo_only`、`shuffle_endo`、`zero`、`random`。
- 读取固定 split。
- 解析路径。
- 生成结果中的 stage flags。

## 7. 当前支持的文本融合阶段

训练入口通过 `--text-stage` 选择文本融合方式。

### 7.1 `global_context_concat`

```text
z_text = concat(endo_t, mean(exo_segments))
```

含义：

- 内生文本是时间片动态状态。
- 外生文本是全局实验配置，先平均池化为全局上下文。

这是最朴素也最符合当前外生文本性质的主线 baseline。

### 7.2 `concat`

当前与 `global_context_concat` 等价，也执行：

```text
concat(endo_t, mean(exo_segments))
```

保留该名字主要用于历史实验兼容。

### 7.3 当前 Content Condenser 的实现

当前 content condenser 已从早期的 softmax segment attention 改为 Soft IB Condenser，更接近 MindTS 中 Information Bottleneck 风格的内容压缩器。

设当前待压缩文本片段为：

```text
segments = [s_1, s_2, ..., s_N]
```

模型先对每个 segment 独立估计保留概率：

```text
psi_i = sigmoid(MLP([context, s_i]))
```

其中 `psi_i` 表示第 `i` 个文本片段被保留的概率。与 softmax 不同，sigmoid 不要求所有 segment 互相竞争，也不强制权重和为 1。

聚合时使用归一化 masked pooling：

```text
z_con = sum(psi_i * s_i) / max(sum(psi_i), epsilon)
```

同时引入 Bernoulli 先验：

```text
G(Z_con) ~ Bernoulli(mu)
```

其中 `mu` 是期望保留率：

- `mu=0.2`：强压缩。
- `mu=0.5`：中等压缩。
- `mu=0.8`：弱压缩，更接近全保留。

KL 约束为：

```text
KL(Bernoulli(psi_i) || Bernoulli(mu))
= psi_i * log(psi_i / mu)
  + (1 - psi_i) * log((1 - psi_i) / (1 - mu))
```

代码中：

- `--condenser-mu` 控制 `mu`。
- `--condenser-loss-weight` 控制 KL loss 权重。
- `--condenser-epsilon` 用于 clamp `psi_i`，避免 `log(0)`，也用于归一化稳定。
- `--condenser-alpha` 控制原始均值表示与压缩表示的残差混合比例。严格 Soft IB 版本建议使用 `1.0`，表示直接使用压缩后的 `z_con`；若训练不稳定，可降到 `0.5` 引入残差平滑。

当前暂未实现跨模态重构损失 `L_rec`。主任务预测损失和可选 contrastive loss 会间接约束压缩文本是否有用。若后续发现文本分支被压掉，可以再加入轻量重构约束。

### 7.4 `global_context_condenser`

```text
exo_segments -> condenser -> condensed_exo_t
z_text = concat(endo_t, condensed_exo_t)
```

只压缩外生文本片段，不压缩内生文本。该阶段不调用 cross-attention。

### 7.5 `global_joint_condenser`

```text
segments_t = [endo_t, exo_1, ..., exo_m]
condensed_joint_t = condenser(segments_t)
z_text = concat(endo_t, condensed_joint_t)
```

压缩内生与外生文本合体，同时保留 `endo_t` 直连，属于 residual joint condenser。该阶段不调用 cross-attention。

### 7.6 `global_joint_condenser_only`

```text
segments_t = [endo_t, exo_1, ..., exo_m]
z_text = condenser(segments_t)
```

只使用压缩后的 joint 文本，不再额外拼接 `endo_t`。这是最纯粹的 joint content condenser。该阶段不调用 cross-attention。

### 7.7 `cross_attention`

```text
Query = endo_t
Key/Value = exo_segments
```

用于测试“当前时间片从外生文本片段中选择相关信息”的假设。

由于当前外生文本是全局实验配置，而不是动态事件库，因此它不一定适合作为主线。

### 7.8 `semantic_gating`

在 cross-attention 后增加语义门控。

### 7.9 `segment_condenser`

历史 MindTS 风格组合阶段：

```text
exo_segments -> condenser
-> cross_attention
-> semantic gate
```

注意：该阶段会调用 cross-attention，不等价于 global condenser。

## 8. 对比学习

对比学习由 `TemporalSemanticAlignmentLayer` 实现，训练入口参数包括：

- `--flow-text-loss-weight`
- `--graph-text-loss-weight`
- `--alignment-projection-dim`
- `--alignment-temperature`
- `--temporal-delta`

三种常见设置：

```text
只做 flow-text 对齐：
  --flow-text-loss-weight > 0
  --graph-text-loss-weight 0

只做 graph-text 对齐：
  --flow-text-loss-weight 0
  --graph-text-loss-weight > 0

两者都对齐：
  --flow-text-loss-weight > 0
  --graph-text-loss-weight > 0
```

当前对齐对象是按 batch 内相同 time index 聚合后的时间级表示。`temporal_delta` 用来屏蔽相邻时间片作为负样本，降低近邻时间片过于相似造成的误导。

## 9. 输出层稳定性设计

文本模型曾出现全 0 预测崩塌。当前已做两处稳定性处理：

- 输出层使用 `Dense(1, activation=None)`，训练阶段不使用 ReLU。
- 输出层 bias 初始化为训练集归一化 target 均值。
- 评估阶段继续执行 `pred = np.maximum(pred, 0.0)`，保证最终指标基于非负预测。

这样避免 ReLU 输出层在初期进入全 0 死区后失去梯度。

## 10. 典型实验命令

### 10.1 GCN-CoSTCo 基线

```bash
cd CostCO

python run_gcn_sat_tensor_experiment.py \
  --tensor-path sat_path_bytes_mb_tensor.npy \
  --topology-path sat_connectivity_tensor_dynamic_60s_1000ms.npz \
  --rank 50 \
  --nc 64 \
  --node-dim 64 \
  --gcn-dim 128 \
  --lr 1e-3 \
  --epochs 200 \
  --batch-size 256 \
  --target-normalization max \
  --seed 3
```

### 10.2 GCN-CoSTCo + 全局文本

```bash
cd CostCO/multimodal_text/models/gcn_costco_text

python run_gcn_text_sat_tensor_experiment.py \
  --rank 50 \
  --nc 64 \
  --node-dim 64 \
  --gcn-dim 128 \
  --text-stage global_context_concat \
  --text-ablation real \
  --text-projection-dim 128 \
  --lr 1e-3 \
  --epochs 200 \
  --batch-size 256 \
  --target-normalization max \
  --seed 3
```

### 10.3 Joint condenser only + flow-text contrastive

```bash
cd CostCO/multimodal_text/models/gcn_costco_text

python run_gcn_text_sat_tensor_experiment.py \
  --rank 50 \
  --nc 64 \
  --node-dim 64 \
  --gcn-dim 128 \
  --text-stage global_joint_condenser_only \
  --text-ablation real \
  --text-projection-dim 128 \
  --condenser-alpha 1.0 \
  --condenser-epsilon 0.05 \
  --condenser-mu 0.5 \
  --condenser-loss-weight 1e-4 \
  --flow-text-loss-weight 0.01 \
  --graph-text-loss-weight 0.0 \
  --alignment-projection-dim 128 \
  --alignment-temperature 0.3 \
  --temporal-delta 2 \
  --lr 1e-3 \
  --epochs 200 \
  --batch-size 256 \
  --target-normalization max \
  --seed 3
```

## 11. 文本消融实验

通过 `--text-ablation` 控制：

- `real`：使用真实内生和外生文本。
- `endo_only`：只使用内生文本，外生文本置零。
- `exo_only`：只使用外生文本，内生文本置零。
- `zero`：内生和外生文本全部置零，用于检查提升是否只来自额外参数。
- `shuffle_endo`：打乱内生文本时间顺序，用于检查时间语义是否有效。
- `random`：使用随机归一化文本向量。

推荐至少报告：

```text
GCN-CoSTCo
GCN-CoSTCo + real text
GCN-CoSTCo + zero text
GCN-CoSTCo + endo_only
GCN-CoSTCo + exo_only
GCN-CoSTCo + shuffle_endo
```

## 12. 运行产物与可清理文件

以下属于运行产物，不是核心源码：

- `__pycache__/`
- `*.pyc`
- `CostCO/results/`
- `CostCO/splits/`
- `CostCO/multimodal_text/text_data/*.npy`
- `CostCO/multimodal_text/text_data/*.json`

其中 `text_data` 下的文件虽然是运行产物，但训练文本模型需要它们。如果要复现实验，应保留或重新生成：

- `endo_texts.json`
- `exo_text_segments.json`
- `endo_text_embeddings.npy`
- `exo_text_embeddings.npy`
- `text_embedding_metadata.json`
- `time_stats_topo_only.json`

本地 API key 文件：

- `CostCO/multimodal_text/deepseek.env`

不应提交仓库。仓库中只保留：

- `deepseek.env.example`

## 13. 当前清理结果

本次整理做了以下清理：

- 将 `build_texts_deepseek.py` 的职责收敛为只生成外生文本。
- 移除 DeepSeek 生成内生文本的旧逻辑，避免和当前确定性内生模板冲突。
- 明确当前内生文本由 `build_satellite_texts.py` 生成，不使用流量张量和随机 mask。
- 保留所有训练入口和实验 stage，避免破坏既有实验对比。

未删除的内容：

- `TOWARDS MULTIMODAL TIME SERIES ANOMALY DETECTION W.pdf` 和 `MULTIMODAL.txt` 属于方法参考材料，保留。
- `multimodal_plan/` 下文档属于设计和命令记录，保留。
- `text_data/` 是生成产物目录，服务器训练需要其中 embedding 文件，保留目录。
- `__pycache__` 属于可清理缓存，但不影响项目逻辑。

## 14. 推荐后续实验顺序

建议按以下顺序推进，避免被复杂模块干扰：

1. 固定 split，先确认 GCN-CoSTCo 强基线。
2. 跑 `global_context_concat + real/zero/endo_only/exo_only/shuffle_endo`。
3. 如果文本有效，再跑 `global_joint_condenser_only`。
4. 最后加入 flow-text 或 graph-text contrastive。
5. `cross_attention` 和 `segment_condenser` 作为对照，不建议作为当前主线。

当前最重要的判断不是“模型是否复杂”，而是文本是否提供了 GCN 邻接矩阵之外的新信息。若 `zero` 优于 `real`，说明当前文本语义或融合方式仍在干扰主任务；若 `endo_only` 或 `exo_only` 优于 `real`，说明两类文本存在互相干扰，需要重新压缩或简化文本内容。
