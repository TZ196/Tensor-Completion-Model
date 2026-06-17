# 张量补全模型

本仓库用于存放卫星网络流量张量补全实验代码。当前包含两个并列 baseline：

- `CostCO`: CoSTCo 神经网络张量补全 baseline。
- `TimesNet`: 基于 TimesNet 的时序 masked imputation 张量补全 baseline。
- `ModernTCN-imputation`: 基于 ModernTCN 的时序 masked imputation 张量补全 baseline。

两个项目都使用卫星路径流量张量：

```text
sat_path_bytes_mb_tensor.npy: 120 x 120 x 60
```

`TimesNet` 的数据默认放在：

```text
TimesNet/data/sat_path_bytes_mb_tensor.npy
```

`ModernTCN-imputation` 的数据默认放在：

```text
ModernTCN-imputation/data/sat_path_bytes_mb_tensor.npy
```

## 目录结构

```text
CostCO/
TimesNet/
ModernTCN-imputation/
README.md
.gitignore
```

## TimesNet 运行方式

进入 `TimesNet` 目录：

```bash
pip install -r requirements.txt
python run_sat_tensor_experiment.py
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
TimesNet/results/
```

数据划分会写入：

```text
TimesNet/splits/
```

## ModernTCN 运行方式

进入 `ModernTCN-imputation` 目录：

```bash
pip install -r requirements.txt
python run_sat_tensor_experiment.py
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
ModernTCN-imputation/results/
```

数据划分会写入：

```text
ModernTCN-imputation/splits/
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

## 指标

指标统一使用：

```text
NMAE  = sum(abs(y_true - y_pred)) / sum(abs(y_true))
NRMSE = sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))
```
