# 张量补全模型 (Tensor Completion Model)

本工作区包含用于**卫星流量恢复**的张量补全实验代码。

## 项目列表

* **`CostCO/CoSTCo`**：CoSTCo 神经网络张量补全基线模型（Baseline）。
* **`Satfomer`**：利用动态星间链路（ISL）拓扑张量实现的 SatFormer 模型。

两个项目均需要使用以下卫星路径流量张量数据：

```text
sat_path_bytes_tensor.npy: 120 x 120 x 60

```

此外，SatFormer 还需要额外使用以下拓扑连接张量数据：

```text
sat_connectivity_tensor_dynamic_60s_1000ms.npz: 120 x 120 x 60

```

---

## SatFormer 运行指南

在 `Satfomer` 目录中执行以下命令：

```bash
pip install -r requirements.txt
python run_sat_tensor_experiment.py

```

可以通过调整 `--observed-ratio` 参数来测试不同的**观测比例**：

```bash
python run_sat_tensor_experiment.py --observed-ratio 0.02
python run_sat_tensor_experiment.py --observed-ratio 0.04
python run_sat_tensor_experiment.py --observed-ratio 0.06
python run_sat_tensor_experiment.py --observed-ratio 0.08
python run_sat_tensor_experiment.py --observed-ratio 0.10

```

---

## CoSTCo 运行指南

在 `CostCO/CoSTCo` 目录中执行以下命令：

```bash
pip install -r requirements.txt
python run_sat_tensor_experiment.py

```