# 互信息估计方法比较实验：KSG vs MINE vs MIST

本项目复现论文中五组数值实验，比较三种互信息估计方法（KSG、MINE、MIST）在不同维度和样本量下的表现。
论文仿照 MIST 论文 (arXiv:2511.18945) Section 4 的实验设计。


## 一、目录结构

├── experiments.py          # 实验主脚本（5组实验）
├── environment.yml         # conda 环境配置文件
├── readme.txt              # 本文件


## 二、环境依赖

### 2.1 系统要求

- 操作系统：Linux（推荐 Ubuntu 20.04+）
- Python：3.12
- GPU：建议有 NVIDIA GPU + CUDA 12.6（纯 CPU 也可运行，但 MINE 估计会更慢）
- 磁盘空间：约 5 GB（含 MIST 源码和预训练权重）

### 2.2 核心依赖

| 依赖库 | 版本 | 用途 |
|--------|------|------|
| torch | >=2.11 | MINE 和 MIST 的深度学习框架 |
| numpy | >=2.4 | 数值计算 |
| jax / jaxlib | 0.4.26 | BMI 采样器的随机数生成 |
| scipy | >=1.17 | KSG 中的 digamma 函数 |
| scikit-learn | >=1.8 | KSG 中的近邻搜索 (NearestNeighbors) |
| matplotlib | >=3.10 | 绘图 |
| bmi | >=0.1.3 | 基准互信息采样库 |
| mist-statinf | 0.1.1 | MIST 估计器接口 |
| benchmark-mi | 0.1.3 | BMI 依赖 |

### 2.3 关键外部依赖

实验代码除了本目录的文件外，还需要以下两个外部仓库：

#### (1) MIST 源码与预训练权重

MEST 估计器需要从源码中导入 `AdditiveUniformSamplerMulti` 采样器，并加载预训练的 Set Transformer 权重。

```bash
# 克隆 MIST 仓库（例如放到 ~/mist）
cd ~
git clone https://github.com/grgera/mist.git

# 确认预训练权重存在
ls ~/mist/checkpoints/mist/weights.ckpt
```

如果预训练权重不存在，可从以下来源获取：
- HuggingFace: https://huggingface.co/grgera/MIST
- 或按 MIST 仓库 README 中 "Quickstart" 部分的说明下载

#### (2) BMI 基准互信息库

```bash
pip install benchmark-mi
```

BMI 提供了 SplitMultinormal、SplitStudentT 等采样器，用于生成已知真实互信息值的测试数据。


## 三、环境搭建

### 3.1 使用 conda 创建环境

```bash
# 从 environment.yml 创建环境
conda env create -f environment.yml

# 激活环境
conda activate fuck

# 如果希望重命名环境（可选）：
#   先修改 environment.yml 中第一行 name: 的值，再执行上述命令
```

### 3.2 手动安装（如果不使用 environment.yml）

```bash
conda create -n mi_experiments python=3.12
conda activate mi_experiments

# 核心依赖
pip install torch numpy scipy scikit-learn matplotlib jax jaxlib

# BMI 基准库
pip install benchmark-mi

# MIST
pip install mist-statinf

# 其他可能需要的
pip install omegaconf tqdm pyyaml pandas seaborn pytorch-lightning
```

### 3.3 验证环境

```bash
python -c "
import torch, numpy, scipy, sklearn, jax, matplotlib, bmi
from mist_statinf.data.multiadditive_noise import AdditiveUniformSamplerMulti
print('All imports OK')
print('CUDA available:', torch.cuda.is_available())
"
```


## 四、运行前的关键配置

### 4.1 修改硬编码路径

experiments.py 中有两处硬编码的绝对路径，需要根据你的实际目录进行修改：

**路径 1（第 42 行）：MIST 源码路径**

```python
# 找到这行：
sys.path.insert(0, "/home/rxhgg/mist/src")

# 修改为你的 MIST 源码 src 目录，例如：
sys.path.insert(0, "/home/your_username/mist/src")
```

**路径 2（第 53 行）：MIST 根目录（用于定位预训练权重）**

```python
# 找到这行：
MIST_ROOT = "/home/rxhgg/mist"

# 修改为你的 MIST 仓库根目录，例如：
MIST_ROOT = "/home/your_username/mist"
```

修改后确保以下路径指向正确：
- `MIST_ROOT/src/mist_statinf/data/multiadditive_noise.py`（采样器）
- `MIST_ROOT/checkpoints/mist/weights.ckpt`（预训练权重）

### 4.2 路径验证

修改完路径后，运行以下命令验证：

```bash
python -c "
import sys
sys.path.insert(0, '/你的MIST路径/mist/src')
from mist_statinf.data.multiadditive_noise import AdditiveUniformSamplerMulti
print('AdditiveSampler OK')

import os
ckpt = '/你的MIST路径/mist/checkpoints/mist/weights.ckpt'
print('Checkpoint exists:', os.path.exists(ckpt))
"
```


## 五、运行实验

### 5.1 运行全部 5 组实验

```bash
python experiments.py
```

这会依次运行以下 5 组实验并生成对应图片到 `figures/` 目录：
- Exp1: 平均 MSE 比较（按样本量分组的柱状图）
- Exp2: 预测 MI vs 真实 MI 散点图
- Exp3: MSE / Bias / Variance 热力图（维度-样本量网格）
- Exp4: 达到目标 MSE 所需的样本量
- Exp5: 推理时间比较

### 5.2 运行单组实验

```bash
python experiments.py --exp 1    # 仅运行实验 1
python experiments.py --exp 2    # 仅运行实验 2
python experiments.py --exp 3    # 仅运行实验 3
python experiments.py --exp 4    # 仅运行实验 4
python experiments.py --exp 5    # 仅运行实验 5
```

### 5.3 运行时间预估

| 实验 | 预计耗时（CPU） | 预计耗时（GPU） |
|------|----------------|----------------|
| Exp1 | 约 30-60 分钟 | 约 15-30 分钟 |
| Exp2 | 约 20-40 分钟 | 约 10-20 分钟 |
| Exp3 | 约 30-60 分钟 | 约 15-30 分钟 |
| Exp4 | 约 30-60 分钟 | 约 15-30 分钟 |
| Exp5 | 约 5-10 分钟 | 约 3-5 分钟 |
| 全部 | 约 2-3 小时 | 约 1-1.5 小时 |

> 注：耗时主要由 MINE 决定，因每次估计需从头训练神经网络 1000 次迭代。
> KSG 和 MIST 的单次推理通常在秒级完成。

### 5.4 输出结果

实验完成后，`figures/` 目录下会生成 5 张 PNG 图片：

| 文件名 | 对应实验 | 内容 |
|--------|---------|------|
| exp1_mse_table.png | Exp1 | 按样本量分组的平均 MSE 柱状图（IMD/OoMD） |
| exp2_pred_vs_true.png | Exp2 | 三种方法的预测 MI vs 真实 MI 散点图 |
| exp3_heatmaps.png | Exp3 | 3×3 热力图矩阵（MSE/Bias/Variance × 3种方法） |
| exp4_sample_requirement.png | Exp4 | 达到目标 MSE 所需样本量随维度变化曲线 |
| exp5_inference_time.png | Exp5 | 推理时间随样本量和维度的变化曲线 |


## 六、实验原理简述

实验代码在三种已知真实互信息的分布族上测试估计方法：

1. **多元高斯分布**：X, Y 联合服从高斯分布，协方差矩阵可密集或稀疏
2. **多元 Student-t 分布**：重尾分布，测试方法的鲁棒性
3. **加性均匀噪声分布**：Y = X + N，N 为均匀噪声，各维度独立可算解析 MI

对采样数据施加保互信息的非线性变换（halfcube、asinh、wigglify）以增加多样性。
分布族分为元分布内（IMD，MIST 训练集中见过）和元分布外（OoMD，未见过）两组。

三种估计方法：
- **KSG**：基于 k 近邻的非参数估计，k=5
- **MINE**：基于 Donsker-Varadhan 对偶的神经网络估计，每次从头训练
- **MIST**：基于元学习的监督估计，加载预训练 Set Transformer 权重后直接推理
