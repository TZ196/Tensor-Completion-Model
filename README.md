# 张量补全模型

本仓库用于存放卫星网络流量张量补全实验代码。当前包含两个并列的模型项目：

- `CostCO`：CoSTCo 神经网络张量补全基线模型。
- `Satfomer`：结合动态星间链路拓扑的 SatFormer 模型。

两个项目都使用星间路径流量张量：

```text
sat_path_bytes_mb_tensor.npy: 120 x 120 x 60
```

其中 `Satfomer` 还使用动态 ISL 拓扑张量：

```text
sat_connectivity_tensor_dynamic_60s_1000ms.npz: 120 x 120 x 60
```

拓扑张量中，`1` 表示对应时间步两颗卫星相邻，`0` 表示没有星间链路。

## 目录结构

```text
CostCO/
Satfomer/
README.md
.gitignore
```

如果后续新增模型，请直接在仓库根目录下新建一个与 `CostCO`、`Satfomer` 并列的文件夹。

## SatFormer 运行方式

进入 `Satfomer` 目录：

```bash
pip install -r requirements.txt
python run_sat_tensor_experiment.py
conda activate TZ-Satformer
conda activate TZ-costco
```

不同观测率实验：

```bash
python run_sat_tensor_experiment.py --observed-ratio 0.02
python run_sat_tensor_experiment.py --observed-ratio 0.04
python run_sat_tensor_experiment.py --observed-ratio 0.06
python run_sat_tensor_experiment.py --observed-ratio 0.08
python run_sat_tensor_experiment.py --observed-ratio 0.10
```

结果会写入：

```text
Satfomer/results/
```

数据划分会写入：

```text
Satfomer/splits/
```

## CoSTCo 运行方式

进入 `CostCO` 目录：

```bash
pip install -r requirements.txt
python run_sat_tensor_experiment.py
```

结果会写入：

```text
CostCO/results/
```

数据划分会写入：

```text
CostCO/splits/
```

## 新增模型的推荐方式

假设要新增一个模型 `NewModel`，推荐结构如下：

```text
NewModel/
  README.md
  requirements.txt
  run_sat_tensor_experiment.py
  new_model.py
  sat_path_bytes_mb_tensor.npy
```

如果模型需要动态拓扑，也把拓扑文件放在该模型目录下：

```text
NewModel/
  sat_connectivity_tensor_dynamic_60s_1000ms.npz
```

推荐保持和现有项目一致的实验入口：

```bash
python run_sat_tensor_experiment.py
```

并尽量支持以下参数，方便不同模型横向比较：

```text
--observed-ratio
--missing-rate
--val-ratio
--epochs
--seed
--metrics-path
```

指标建议统一使用：

```text
NMAE  = sum(abs(y_true - y_pred)) / sum(abs(y_true))
NRMSE = sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))
```

新增模型后，在仓库根目录执行：

```bash
git add NewModel README.md .gitignore
git commit -m "Add NewModel tensor completion experiment"
git push
```
