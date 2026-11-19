# 水热碳化液相生物油性质预测软件

> 基于堆叠集成学习（Stacking Ensemble）的水热碳化（Hydrothermal Carbonization, HTC）液相产物——生物油——产率与理化性质的快速预测工具。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange)

---

## 项目简介

水热碳化可将湿生物质直接转化为水热炭与液相产物（生物油），液相产物的产率、元素组成与热值决定了后续提质与利用路径。实验测定这些指标成本高、周期长，本软件用机器学习模型替代常规实验筛选：

- 只需输入生物质的**生化组成、元素分析、灰分、操作条件、溶剂类型**；
- 即可同时输出 **7 项**液相产物指标：生物油产率、C/H/O/N/S 含量与高位热值（HHV）；
- 附带完整的**训练脚本**、**建模模块**与**训练好的模型文件**，结果可复现、模型可再训练。

软件提供 Tkinter 图形界面，无需编程即可使用；也支持在 Python 中直接调用模型做批量预测。

---

## 主要特性

| 特性 | 说明 |
|---|---|
| 多目标预测 | 一次输入同时预测 7 个目标变量，7 个目标各训练一套独立模型 |
| 堆叠集成 | 5 个基模型（RF、XGBoost、GBR、SVR、LightGBM）+ 线性回归元模型，基模型使用 5 折交叉验证生成元特征 |
| 超参数优化 | RandomizedSearchCV 对每个基模型独立寻优，最优结果与全部参数组合性能均可导出 |
| 特征重要性 | 基于 SHAP 值给出特征重要性排序，兼顾置换式（`feature_importances_`）回退方案 |
| 数据预处理 | 均值缺失填补 + 标准化，分类变量（反应溶剂 RS、分离溶剂 SS）按训练时的类别水平做独热编码 |
| 图形界面 | Tkinter 界面，输入 15 个特征即可完成预测 |
| 结果导出 | 训练脚本输出预测值、测试集明细、SHAP 排名、超参数搜索结果与模型性能汇总 |

---

## 模型性能（测试集）

数据集共 3430 条样本，按 8.5 : 1.5 划分训练集/测试集（随机种子 42），测试集 515 条：

| 目标变量 | 含义 | 单位 | 测试集 R² | RMSE | MAE |
|---|---|---|---|---|---|
| Yield_oil | 生物油产率 | wt% | **0.978** | 1.926 | 1.103 |
| Oil_C | 生物油碳含量 | % | 0.950 | 0.955 | 0.533 |
| Oil_H | 生物油氢含量 | % | 0.971 | 0.262 | 0.153 |
| Oil_O | 生物油氧含量 | % | 0.955 | 1.079 | 0.587 |
| Oil_N | 生物油氮含量 | % | 0.947 | 0.397 | 0.189 |
| Oil_S | 生物油硫含量 | % | **0.984** | 0.133 | 0.056 |
| Oil_HHV | 生物油高位热值 | MJ/kg | 0.950 | 0.725 | 0.376 |

> 指标由 `生物油性质预测分析结果（5折交叉验证）高斯数据增强10/模型性能汇总.csv` 导出，各基模型与元模型的细分结果见同一目录下的 `模型性能分析/`。

---

## 输入特征

共 15 个输入特征，其中 13 个数值型、2 个分类型：

| 类别 | 特征名 | 说明 | 单位 |
|---|---|---|---|
| 生化组成 | `Pro (%)` | 蛋白质 | % |
| 生化组成 | `Lip (%)` | 脂肪 | % |
| 生化组成 | `CH (%)` | 碳水化合物 | % |
| 元素分析 | `C (%)` | 碳 | % |
| 元素分析 | `H (%)` | 氢 | % |
| 元素分析 | `O (%)` | 氧 | % |
| 元素分析 | `N (%)` | 氮 | % |
| 元素分析 | `S (%)` | 硫 | % |
| 灰分 | `Ash (%)` | 灰分 | % |
| 操作条件 | `Time (min)` | 停留时间 | min |
| 操作条件 | `T (℃)` | 反应温度 | ℃ |
| 操作条件 | `Size (L)` | 反应器体积 | L |
| 操作条件 | `Ratio` | 生物质/溶剂比 | — |
| 溶剂 | `RS` | 反应溶剂 | 类别 |
| 溶剂 | `SS` | 分离溶剂 | 类别 |

溶剂类别取值（训练数据中出现过的水平）：

- `RS`：`H2O`、`CH3OH`、`C2H6O`、`(CH2OH)2`、`C4H8O2`
- `SS`：`CH2Cl2`、`CHCl3`、`CH3OH`、`C3H6O`、`C4H10O`

---

## 环境要求

- Python **3.13**（Windows / macOS / Linux 均可，图形界面需要系统自带 tkinter）
- 依赖见 [`requirements.txt`](requirements.txt)

> 注意：`trained_models/*.pkl` 是 joblib 序列化的完整模型对象，对 `scikit-learn`、`xgboost`、`lightgbm` 等库的版本较敏感。请尽量使用 `requirements.txt` 中给出的版本，否则可能出现加载失败或预测结果偏差。

---

## 安装与使用

### 1. 克隆仓库（模型文件用 Git LFS 托管，需要先安装 Git LFS）

```bash
git lfs install
git clone https://github.com/<your-name>/predict-app.git
cd predict-app
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. 启动预测软件

```bash
python 预测软件.py
```

在界面中填写 15 个特征后点击「预测」，下方文本框会输出 7 个指标的预测值：

```
Yield_oil: 28.650
Oil_C: 71.820
Oil_H: 6.540
Oil_O: 20.940
Oil_N: 0.820
Oil_S: 0.260
Oil_HHV: 34.120
```

### 4. 在代码中调用（批量预测）

```python
import joblib
import pandas as pd
# 必须导入建模模块，否则 pkl 中的类引用无法解析
from stacking_model import RobustDataPreprocessor, HyperparameterOptimizer, StackingEnsembleModel

meta = joblib.load("trained_models/Yield_oil.pkl")
model = meta["model"]

# 构造与训练一致的特征表（含 RS/SS 原始类别列，独热编码在内部完成）
df = pd.read_csv("液相产物预测数据高斯增强10.csv", encoding="gbk")
print(model.predict(df[meta["feature_order"]].head()))
```

---

## 重新训练

```bash
python 训练脚本.py
```

运行后会弹出文件选择框，选择 `液相产物预测数据高斯增强10.csv`（或任意列名一致的 CSV/XLSX），脚本会依次：

1. 划分训练/测试集（`test_size=0.15`）；
2. 对每个目标变量分别做超参数优化并训练堆叠模型；
3. 把模型写入 `trained_models/<目标名>.pkl`，供 `预测软件.py` 直接加载；
4. 把预测值、SHAP 排名、超参数搜索明细、性能指标写入 `生物油性质预测分析结果（5折交叉验证）高斯数据增强10/` 目录。

> 训练耗时较长（5 个基模型 × 7 个目标 × 5 折交叉验证 × 20 组随机搜索），建议在多核机器上运行。

---

## 项目结构

```
predict-app/
├── 预测软件.py                                  # Tkinter 图形界面预测程序（入口）
├── 训练脚本.py                                  # 训练与结果导出流水线
├── stacking_model.py                            # 预处理 / 超参优化 / 堆叠集成模型类定义
├── 液相产物预测数据高斯增强10.csv                # 训练数据（3430 条，GBK 编码）
├── trained_models/                              # 训练好的 7 个模型（Git LFS 托管）
│   ├── Yield_oil.pkl
│   ├── Oil_C.pkl  Oil_H.pkl  Oil_O.pkl
│   ├── Oil_N.pkl  Oil_S.pkl  Oil_HHV.pkl
├── 生物油性质预测分析结果（5折交叉验证）高斯数据增强10/
│   ├── 模型性能汇总.csv                          # 7 个目标在测试集上的 R²/RMSE/MAE
│   ├── 模型性能分析/                             # 基模型、元模型、各折 CV 分数
│   ├── 超参数优化结果/                           # 每个模型的全部随机搜索结果
│   ├── <目标>_全部预测值.csv
│   ├── <目标>_测试集完整数据.csv
│   ├── <目标>_特征SHAP重要性排行.csv
│   └── <目标>_特征目标完整关系数据.csv
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 数据说明

- 文件：`液相产物预测数据高斯增强10.csv`
- 规模：**3430 行 × 22 列**（原始实验数据经高斯噪声数据增强 10 倍扩充）
- 编码：**GBK**。训练脚本内置 `utf-8 → gbk → latin-1 → utf-8-sig` 多编码回退读取；若自行读取请显式指定 `encoding="gbk"`
- 列：15 个输入特征 + 7 个目标变量

---

## 常见问题

| 问题 | 处理 |
|---|---|
| `joblib.load` 报 `No module named 'stacking_model'` | 加载前先 `from stacking_model import ...`，并保证脚本与 `stacking_model.py` 在同一目录 |
| 报 `ModuleNotFoundError: sklearn` / 版本告警 | 按 `requirements.txt` 创建虚拟环境重装依赖 |
| 模型文件缺失或异常 | 确认已执行 `git lfs install && git lfs pull` |
| 图形界面报 `No module named 'tkinter'` | Linux 下安装系统包 `python3-tk` |
| 提示「未在 trained_models 目录找到模型文件」 | 在项目根目录运行 `预测软件.py`，`MODEL_DIR` 是相对当前工作目录的路径 |

---

## 免责声明

本项目面向科研与教学用途，模型预测结果基于有限的历史实验数据，不能完全替代实测；工程设计与商业化应用前请结合实验验证。

---

## 引用与致谢

如果本项目对你的研究有帮助，欢迎引用或 star。模型训练思路参考了堆叠集成学习与 SHAP 可解释性分析的通用做法，数据来源于公开文献与自有实验整理。

---

## English Abstract

**predict-app** is a GUI + library toolkit for predicting the yield and properties (C/H/O/N/S content, HHV) of bio-oil from hydrothermal carbonization of biomass. Given biomass composition (protein, lipid, carbohydrate), ultimate analysis, ash, reaction conditions (time, temperature, reactor volume, biomass/solvent ratio) and solvent types (reaction solvent RS, separation solvent SS), seven stacking-ensemble models predict the liquid-phase product properties simultaneously. Each target uses five base learners (Random Forest, XGBoost, Gradient Boosting, SVR, LightGBM) with randomized hyper-parameter search and a linear meta-learner trained on out-of-fold predictions. Test-set R² ranges from 0.947 to 0.984. See the table above for details.
