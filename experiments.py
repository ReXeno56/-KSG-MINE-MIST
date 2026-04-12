"""
互信息估计方法比较实验：KSG vs MINE vs MIST
==============================================
仿照 MIST 论文 (arXiv:2511.18945) Section 4 的实验设计：
  Exp1: 总体 MSE 比较 (类似 Table 1) — 按样本量分组
  Exp2: Predicted vs True MI 散点图 (类似 Figure 2)
  Exp3: Bias/Variance/MSE 热力图 (类似 Figure 3) — 按 (dim, n) 分格
  Exp4: 维度扩展的样本需求 (类似 Figure 4) — 达到目标 MSE 所需样本量
  Exp5: 推理时间比较 (类似 Section 4.6)

数据生成: 使用 BMI 原生采样器 + MIST 的 AdditiveUniformSamplerMulti。

运行方式:
  python experiments.py            # 运行全部实验
  python experiments.py --exp 1    # 仅运行实验1
"""

import sys
import os
import time
import math
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn

# BMI 采样器
import jax
import jax.random
from bmi.samplers import SplitMultinormal, SplitStudentT
from bmi.samplers._matrix_utils import GaussianLVMParametrization

# MIST 的多维 additive noise 采样器
sys.path.insert(0, "/home/rxhgg/mist/src")
from mist_statinf.data.multiadditive_noise import AdditiveUniformSamplerMulti

# ---- 设置字体 ----
rcParams["font.family"] = "serif"
rcParams["font.size"] = 11
rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")

# ---- 路径设置 ----
MIST_ROOT = "/home/rxhgg/mist"
MIST_CKPT = os.path.join(MIST_ROOT, "checkpoints", "mist", "weights.ckpt")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
#  BMI 原生数据生成
# ============================================================

def _make_dense_cov(d, off_diag):
    """密集协方差/散布矩阵: (1-off_diag)*I + off_diag*11^T"""
    dim = 2 * d
    return np.eye(dim) * (1 - off_diag) + off_diag


def _make_sparse_cov(d, n_interact, strength):
    """稀疏协方差/散布矩阵: 使用 GaussianLVMParametrization"""
    param = GaussianLVMParametrization(
        dim_x=d, dim_y=d,
        n_interacting=min(n_interact, d),
        alpha=0.0, lambd=strength,
        beta_x=0.0, eta_x=strength,
    )
    return param.correlation


# ---- 非线性保 MI 变换 (匹配 BMI/MIST 实现) ----
def _transform_halfcube(data):
    """x * sqrt(|x|), 即 signed x^(3/2)"""
    return data * np.sqrt(np.abs(data))


def _transform_asinh(data):
    return np.arcsinh(data)


def _transform_wigglify_x(data):
    """wigglify for X (BMI wiggly_x)"""
    return data + 0.4 * np.sin(1.0 * data) + 0.2 * np.sin(1.7 * data + 1) + 0.03 * np.sin(3.3 * data - 2.5)


def _transform_wigglify_y(data):
    """wigglify for Y (BMI wiggly_y) — 与 X 不同的非线性"""
    return data - 0.4 * np.sin(0.4 * data) + 0.17 * np.sin(1.3 * data + 3.5) + 0.02 * np.sin(4.3 * data - 2.5)


# 对称变换 (X 和 Y 用相同函数)
SYMMETRIC_TRANSFORMS = {
    "base": lambda x: x,
    "halfcube": _transform_halfcube,
    "asinh": _transform_asinh,
}


# 分布族定义 (仿 MIST 论文 Table 4)
IMD_FAMILIES = [
    ("multi_normal", "dense", "base"),
    ("multi_normal", "dense", "halfcube"),
    ("multi_normal", "sparse", "base"),
    ("multi_student", "dense", "base"),
    ("multi_student", "sparse", "base"),
    ("multi_student", "sparse", "asinh"),
]

OOMD_FAMILIES = [
    ("multi_normal", "dense", "wigglify"),
    ("multi_normal", "sparse", "halfcube"),
    ("multi_student", "dense", "halfcube"),
    ("multi_student", "sparse", "wigglify"),
    ("multi_additive_noise", "base", "base"),
    ("multi_additive_noise", "base", "halfcube"),
]


def generate_single_sample(family, n, d, seed):
    """
    给定分布族，使用 BMI 原生采样器生成 (X, Y, true_mi)。
    参数范围匹配 MIST 论文的 distribution_generator.py。
    """
    rng = np.random.RandomState(seed)
    base_dist, structure, transform_name = family

    if base_dist == "multi_normal":
        if structure == "dense":
            off_diag = rng.uniform(0.0, 0.5)
            cov = _make_dense_cov(d, off_diag)
            sampler = SplitMultinormal(dim_x=d, dim_y=d, covariance=cov)
        elif structure == "sparse":
            n_interact = rng.randint(1, d + 1)
            strength = rng.uniform(0.1, 5.0)
            cov = _make_sparse_cov(d, n_interact, strength)
            sampler = SplitMultinormal(dim_x=d, dim_y=d, covariance=cov)
        else:
            raise ValueError(f"Unknown structure: {structure}")

        true_mi = float(sampler.mutual_information())
        key = jax.random.PRNGKey(seed)
        xy = sampler.sample(n, key)
        X, Y = np.array(xy[0]), np.array(xy[1])

    elif base_dist == "multi_student":
        df = rng.uniform(1.0, 10.0)
        if structure == "dense":
            off_diag = rng.uniform(0.0, 0.5)
            disp = _make_dense_cov(d, off_diag)
        elif structure == "sparse":
            n_interact = rng.randint(1, d + 1)
            strength = rng.uniform(0.1, 5.0)
            disp = _make_sparse_cov(d, n_interact, strength)
        else:
            raise ValueError(f"Unknown structure: {structure}")

        sampler = SplitStudentT(dim_x=d, dim_y=d, dispersion=disp, df=df)
        true_mi = float(sampler.mutual_information())
        # SplitStudentT.sample 接受 int seed
        xy = sampler.sample(n, seed)
        X, Y = np.array(xy[0]), np.array(xy[1])

    elif base_dist == "multi_additive_noise":
        epsilons = [rng.uniform(0.1, 2.0) for _ in range(d)]
        sampler = AdditiveUniformSamplerMulti(epsilon=epsilons, dim=d)
        true_mi = float(sampler.mutual_information())
        key = jax.random.PRNGKey(seed)
        xy = sampler.sample(n, key)
        X, Y = np.array(xy[0]), np.array(xy[1])

    else:
        raise ValueError(f"Unknown base_dist: {base_dist}")

    # 应用变换
    if transform_name == "wigglify":
        X = _transform_wigglify_x(X)
        Y = _transform_wigglify_y(Y)
    elif transform_name in SYMMETRIC_TRANSFORMS:
        tfm = SYMMETRIC_TRANSFORMS[transform_name]
        X = tfm(X)
        Y = tfm(Y)
    else:
        raise ValueError(f"Unknown transform: {transform_name}")

    return X.astype(np.float32), Y.astype(np.float32), float(true_mi)


# ============================================================
#  KSG 估计器
# ============================================================
class KSGEstimator:
    """KSG-1 互信息估计器 (Kraskov et al., 2004)."""

    def __init__(self, k=5):
        self.k = k

    def estimate(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        if Y.ndim == 1: Y = Y.reshape(-1, 1)

        n = X.shape[0]
        k = min(self.k, n - 1)
        if k < 1:
            return 0.0

        XY = np.hstack([X, Y])
        nn_joint = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev")
        nn_joint.fit(XY)
        distances, _ = nn_joint.kneighbors(XY)
        eps = distances[:, k]

        nn_x = NearestNeighbors(metric="chebyshev")
        nn_x.fit(X)
        nn_y = NearestNeighbors(metric="chebyshev")
        nn_y.fit(Y)

        n_x = np.zeros(n, dtype=int)
        n_y = np.zeros(n, dtype=int)

        for i in range(n):
            eps_i = max(eps[i], 1e-10)
            idx_x = nn_x.radius_neighbors([X[i]], radius=eps_i, return_distance=False)[0]
            idx_y = nn_y.radius_neighbors([Y[i]], radius=eps_i, return_distance=False)[0]
            n_x[i] = len(idx_x) - 1
            n_y[i] = len(idx_y) - 1

        mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
        return max(mi, 0.0)


# ============================================================
#  MINE 估计器 (基于 mine-pytorch 核心逻辑)
# ============================================================
EPS_MINE = 1e-6


class _EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        return input.exp().mean().log()

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / (running_mean + EPS_MINE) / input.shape[0]
        return grad, None


def _ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = alpha * t_exp + (1.0 - alpha) * running_mean.item()
    t_log = _EMALoss.apply(x, running_mean)
    return t_log, running_mean


class _MineNet(nn.Module):
    def __init__(self, T_net, alpha=0.01):
        super().__init__()
        self.T = T_net
        self.running_mean = 0
        self.alpha = alpha

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0], device=x.device)]
        t = self.T(torch.cat([x, z], dim=1)).mean()
        t_marg = self.T(torch.cat([x, z_marg], dim=1))
        second_term, self.running_mean = _ema_loss(t_marg, self.running_mean, self.alpha)
        return -t + second_term

    @torch.no_grad()
    def mi(self, x, z):
        return -self.forward(x, z)


class MINEEstimator:
    """MINE 互信息估计器 (Belghazi et al., 2018)."""

    def __init__(self, hidden_dim=128, n_layers=2, lr=1e-4, iters=1000, batch_size=256):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr = lr
        self.iters = iters
        self.batch_size = batch_size

    def _build_network(self, input_dim):
        layers = [nn.Linear(input_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(self.n_layers - 1):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(self.hidden_dim, 1))
        return nn.Sequential(*layers)

    def estimate(self, X, Y):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        if X.ndim == 1: X = X.reshape(-1, 1)
        if Y.ndim == 1: Y = Y.reshape(-1, 1)

        T_net = self._build_network(X.shape[1] + Y.shape[1])
        model = _MineNet(T_net)
        model.to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        X_t = torch.from_numpy(X).float().to(DEVICE)
        Y_t = torch.from_numpy(Y).float().to(DEVICE)
        n = X_t.shape[0]

        for it in range(self.iters):
            idx = torch.randint(0, n, (min(self.batch_size, n),), device=DEVICE)
            opt.zero_grad()
            loss = model(X_t[idx], Y_t[idx])
            if torch.isnan(loss) or torch.isinf(loss):
                # 梯度爆炸，重置优化器动量并跳过
                opt = torch.optim.Adam(model.parameters(), lr=self.lr)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        mi_est = model.mi(X_t, Y_t)
        val = float(mi_est.item())
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return max(val, 0.0)


# ============================================================
#  MIST 估计器
# ============================================================
MIST_MAX_N = 500
MIST_N_SUBSAMPLES = 5


class MISTEstimator:
    """MIST 互信息估计器 (Gritsai et al., 2025). 使用预训练 Set Transformer."""

    def __init__(self, checkpoint_path=MIST_CKPT):
        self.checkpoint_path = checkpoint_path
        self._model = None

    def _load_model(self):
        if self._model is None:
            from mist_statinf.quickstart import MISTQuickEstimator
            self._model = MISTQuickEstimator(
                loss="mse", checkpoint=self.checkpoint_path, device=DEVICE,
            )
        return self._model

    def _standardize(self, X):
        std = X.std(axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)
        return (X - X.mean(axis=0, keepdims=True)) / std

    def estimate(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        if Y.ndim == 1: Y = Y.reshape(-1, 1)

        X = self._standardize(X)
        Y = self._standardize(Y)
        model = self._load_model()
        n = X.shape[0]

        if n <= MIST_MAX_N:
            mi = model.estimate_point(X, Y)
            return max(float(mi), 0.0)
        else:
            rng = np.random.RandomState(SEED)
            estimates = []
            for _ in range(MIST_N_SUBSAMPLES):
                idx = rng.choice(n, size=MIST_MAX_N, replace=False)
                mi = model.estimate_point(X[idx], Y[idx])
                estimates.append(float(mi))
            return max(float(np.mean(estimates)), 0.0)


# ============================================================
#  工具函数
# ============================================================
def run_estimator(estimator, X, Y):
    start = time.time()
    mi = estimator.estimate(X, Y)
    return mi, time.time() - start


def _make_mine(d, n):
    """根据维度和样本量构建合适的 MINE 估计器。"""
    return MINEEstimator(
        hidden_dim=128,
        n_layers=2,
        lr=1e-4,
        iters=1000,
        batch_size=min(256, n),
    )


# ---- Exp 1: 总体 MSE 比较 (仿 Table 1) ----
def experiment1_mse_table():
    print("\n" + "=" * 70)
    print("Exp 1: Average MSE comparison (grouped by sample size)")
    print("=" * 70)

    dims = [3, 5, 8, 10, 15]
    n_bins = [(10, 100), (100, 300), (300, 500)]
    n_per_config = 2

    ksg = KSGEstimator(k=5)
    mist = MISTEstimator()

    all_results = {"IMD": {b: {"KSG": [], "MINE": [], "MIST": []} for b in n_bins},
                   "OoMD": {b: {"KSG": [], "MINE": [], "MIST": []} for b in n_bins}}

    # 计算总任务数用于进度显示
    total_tasks = sum(len(fams) for _, fams in [("IMD", IMD_FAMILIES), ("OoMD", OOMD_FAMILIES)]) \
                  * len(dims) * n_per_config * len(n_bins)
    done = 0

    seed_c = SEED
    for label, families in [("IMD", IMD_FAMILIES), ("OoMD", OOMD_FAMILIES)]:
        for family in families:
            fname = "-".join(family)
            for d in dims:
                for rep in range(n_per_config):
                    for n_lo, n_hi in n_bins:
                        n = np.random.randint(n_lo, n_hi + 1)
                        X, Y, true_mi = generate_single_sample(family, n, d, seed_c)
                        seed_c += 1

                        mine = _make_mine(d, n)

                        mi_ksg = ksg.estimate(X, Y)
                        mi_mine = mine.estimate(X, Y)
                        mi_mist = mist.estimate(X, Y)

                        b = (n_lo, n_hi)
                        all_results[label][b]["KSG"].append((mi_ksg - true_mi) ** 2)
                        all_results[label][b]["MINE"].append((mi_mine - true_mi) ** 2)
                        all_results[label][b]["MIST"].append((mi_mist - true_mi) ** 2)

                        done += 1
                        if done % 10 == 0 or done == total_tasks:
                            print(f"  [{done}/{total_tasks}] {label} {fname} d={d} n={n} "
                                  f"| KSG={mi_ksg:.3f} MINE={mi_mine:.3f} MIST={mi_mist:.3f} "
                                  f"(true={true_mi:.3f})")

    # 打印表格
    for label in ["IMD", "OoMD"]:
        print(f"\n  {label}:")
        print(f"  {'n range':>12s} | {'KSG MSE':>12s} | {'MINE MSE':>12s} | {'MIST MSE':>12s}")
        print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
        for b in n_bins:
            vals = {m: np.nanmean(all_results[label][b][m]) for m in ["KSG", "MINE", "MIST"]}
            print(f"  [{b[0]:3d},{b[1]:3d}]   | {vals['KSG']:12.4f} | {vals['MINE']:12.4f} | {vals['MIST']:12.4f}")

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"KSG": "#2196F3", "MINE": "#FF5722", "MIST": "#4CAF50"}
    bin_labels = [f"[{lo},{hi}]" for lo, hi in n_bins]

    for ax_idx, label in enumerate(["IMD", "OoMD"]):
        ax = axes[ax_idx]
        x = np.arange(len(n_bins))
        w = 0.25
        for i, method in enumerate(["KSG", "MINE", "MIST"]):
            vals = [np.nanmean(all_results[label][b][method]) for b in n_bins]
            ax.bar(x + i * w, vals, w, label=method, color=colors[method], alpha=0.85)
        ax.set_xticks(x + w)
        ax.set_xticklabels(bin_labels)
        ax.set_xlabel("Sample size range (n)")
        ax.set_ylabel("Mean Squared Error")
        ax.set_title(f"{label} distributions")
        ax.legend()
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Exp 1: Average MSE by Sample Size Range", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "exp1_mse_table.png"), dpi=150)
    plt.close(fig)
    print(f"\n  [Saved] {OUTPUT_DIR}/exp1_mse_table.png")


# ---- Exp 2: Predicted vs True MI (仿 Figure 2) ----
def experiment2_pred_vs_true():
    print("\n" + "=" * 70)
    print("Exp 2: Predicted MI vs True MI scatter plot")
    print("=" * 70)

    dims = [3, 5, 8, 10, 15, 20]
    families = IMD_FAMILIES + OOMD_FAMILIES
    n_per_config = 2

    ksg = KSGEstimator(k=5)
    mist = MISTEstimator()

    records = []
    seed_c = SEED + 10000

    total_tasks = len(families) * len(dims) * n_per_config
    done = 0

    for family in families:
        fname = "-".join(family)
        for d in dims:
            for _ in range(n_per_config):
                n = np.random.randint(50, 500)
                X, Y, true_mi = generate_single_sample(family, n, d, seed_c)
                seed_c += 1

                mine = _make_mine(d, n)

                mi_ksg = ksg.estimate(X, Y)
                mi_mine = mine.estimate(X, Y)
                mi_mist = mist.estimate(X, Y)
                records.append((true_mi, mi_ksg, mi_mine, mi_mist))

                done += 1
                if done % 10 == 0 or done == total_tasks:
                    print(f"  [{done}/{total_tasks}] {fname} d={d} n={n} "
                          f"| true={true_mi:.3f} KSG={mi_ksg:.3f} MINE={mi_mine:.3f} MIST={mi_mist:.3f}")

    records = np.array(records)
    true_mi = records[:, 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    methods = ["KSG", "MINE", "MIST"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    mi_max = max(true_mi.max(), records[:, 1:].max()) * 1.1

    for i, (ax, method, color) in enumerate(zip(axes, methods, colors)):
        pred = records[:, i + 1]
        ax.scatter(true_mi, pred, alpha=0.4, s=15, color=color, edgecolors="none")
        ax.plot([0, mi_max], [0, mi_max], "k--", linewidth=1, alpha=0.5, label="y = x")

        # 使用 quantile-based binning，保证每个 bin 都有数据
        n_bins = 10
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(true_mi, percentiles)
        bin_centers, bin_means = [], []
        for j in range(len(bin_edges) - 1):
            if j == len(bin_edges) - 2:  # 最后一个 bin 包含右边界
                mask = (true_mi >= bin_edges[j]) & (true_mi <= bin_edges[j + 1])
            else:
                mask = (true_mi >= bin_edges[j]) & (true_mi < bin_edges[j + 1])
            if mask.sum() > 0:
                bin_centers.append(true_mi[mask].mean())  # 用实际均值作为中心点
                bin_means.append(pred[mask].mean())
        ax.plot(bin_centers, bin_means, "o-", color="black", markersize=5, linewidth=2, label="Binned mean")

        mse = np.mean((pred - true_mi) ** 2)
        bias = np.mean(pred - true_mi)
        ax.set_xlabel("True MI (nats)", fontsize=12)
        ax.set_ylabel("Predicted MI (nats)", fontsize=12)
        ax.set_title(f"{method}  (MSE={mse:.2f}, Bias={bias:+.2f})", fontsize=13)
        ax.set_xlim(0, mi_max)
        ax.set_ylim(0, mi_max)
        ax.set_aspect("equal")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Exp 2: Predicted MI vs True MI", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "exp2_pred_vs_true.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {OUTPUT_DIR}/exp2_pred_vs_true.png")


# ---- Exp 3: Bias/Variance/MSE 热力图 (仿 Figure 3) ----
def experiment3_heatmaps():
    print("\n" + "=" * 70)
    print("Exp 3: MSE / Bias / Variance heatmaps by (dim, n)")
    print("=" * 70)

    dims = [3, 5, 8, 10, 15, 20]
    n_sizes = [30, 50, 100, 200, 350, 500]
    families = IMD_FAMILIES[:4]
    n_trials = 3

    ksg = KSGEstimator(k=5)
    mist = MISTEstimator()
    methods = ["KSG", "MINE", "MIST"]

    shape = (len(dims), len(n_sizes))
    results = {m: {"mse": np.zeros(shape), "bias": np.zeros(shape), "var": np.zeros(shape)}
               for m in methods}

    total_cells = len(dims) * len(n_sizes)
    done = 0

    seed_c = SEED + 20000
    for di, d in enumerate(dims):
        for ni, n in enumerate(n_sizes):
            method_preds = {m: [] for m in methods}
            method_errors = {m: [] for m in methods}

            for family in families:
                for _ in range(n_trials):
                    X, Y, true_mi = generate_single_sample(family, n, d, seed_c)
                    seed_c += 1

                    mine = _make_mine(d, n)

                    for est_name, est in [("KSG", ksg), ("MINE", mine), ("MIST", mist)]:
                        pred = est.estimate(X, Y)
                        method_preds[est_name].append(pred)
                        method_errors[est_name].append(pred - true_mi)

            for m in methods:
                errors = np.array(method_errors[m])
                preds = np.array(method_preds[m])
                results[m]["mse"][di, ni] = np.mean(errors ** 2)
                results[m]["bias"][di, ni] = np.mean(errors)
                results[m]["var"][di, ni] = np.var(preds)

            done += 1
            print(f"  [{done}/{total_cells}] d={d:2d}, n={n:3d} | "
                  + " | ".join(f"{m}: MSE={results[m]['mse'][di, ni]:.3f}" for m in methods))

    metrics = ["mse", "bias", "var"]
    metric_labels = ["MSE", "Bias", "Variance"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))

    for mi_idx, metric in enumerate(metrics):
        if metric != "bias":
            vmin = min(results[m][metric].min() for m in methods)
            vmax = max(results[m][metric].max() for m in methods)
        else:
            vmax_abs = max(abs(results[m][metric]).max() for m in methods)
            vmin, vmax = -vmax_abs, vmax_abs

        for mj, method in enumerate(methods):
            ax = axes[mi_idx, mj]
            data = results[method][metric]

            if metric == "bias":
                im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)

            ax.set_xticks(range(len(n_sizes)))
            ax.set_xticklabels([str(s) for s in n_sizes], fontsize=9)
            ax.set_yticks(range(len(dims)))
            ax.set_yticklabels([str(d) for d in dims], fontsize=9)

            for y in range(len(dims)):
                for x in range(len(n_sizes)):
                    val = data[y, x]
                    ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(val) > (vmax - vmin) * 0.6 + vmin else "black")

            if mi_idx == 0:
                ax.set_title(method, fontsize=13, fontweight="bold")
            if mj == 0:
                ax.set_ylabel(f"{metric_labels[mi_idx]}\n\nDimension (d)", fontsize=11)
            if mi_idx == len(metrics) - 1:
                ax.set_xlabel("Sample size (n)", fontsize=11)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Exp 3: MSE / Bias / Variance by (Dimension, Sample Size)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "exp3_heatmaps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [Saved] {OUTPUT_DIR}/exp3_heatmaps.png")


# ---- Exp 4: 达到目标 MSE 所需的样本量 (仿 Figure 4) ----
def experiment4_sample_requirement():
    print("\n" + "=" * 70)
    print("Exp 4: Samples required to achieve target MSE by dimension")
    print("=" * 70)

    dims = [3, 5, 8, 10, 15, 20]
    n_candidates = [30, 50, 100, 150, 200, 300, 400, 500]
    mse_thresholds = [1.0, 3.0, 5.0]
    families = IMD_FAMILIES[:4]
    n_trials = 3

    ksg = KSGEstimator(k=5)
    mist = MISTEstimator()
    methods = ["KSG", "MINE", "MIST"]

    mse_grid = {m: np.full((len(dims), len(n_candidates)), np.nan) for m in methods}

    total_cells = len(dims) * len(n_candidates)
    done = 0

    seed_c = SEED + 30000
    for di, d in enumerate(dims):
        for ni, n in enumerate(n_candidates):
            errors = {m: [] for m in methods}
            for family in families:
                for _ in range(n_trials):
                    X, Y, true_mi = generate_single_sample(family, n, d, seed_c)
                    seed_c += 1

                    mine = _make_mine(d, n)

                    for est_name, est in [("KSG", ksg), ("MINE", mine), ("MIST", mist)]:
                        pred = est.estimate(X, Y)
                        errors[est_name].append((pred - true_mi) ** 2)

            for m in methods:
                mse_grid[m][di, ni] = np.mean(errors[m])

            done += 1
            print(f"  [{done}/{total_cells}] d={d:2d}, n={n:3d} | "
                  + " | ".join(f"{m}: {mse_grid[m][di, ni]:.3f}" for m in methods))

    fig, axes = plt.subplots(1, len(mse_thresholds), figsize=(6 * len(mse_thresholds), 5))
    colors = {"KSG": "#2196F3", "MINE": "#FF5722", "MIST": "#4CAF50"}
    markers = {"KSG": "o", "MINE": "s", "MIST": "^"}

    for ti, threshold in enumerate(mse_thresholds):
        ax = axes[ti]
        for method in methods:
            required_n = []
            for di in range(len(dims)):
                mses = mse_grid[method][di, :]
                achieved = np.where(mses <= threshold)[0]
                if len(achieved) > 0:
                    required_n.append(n_candidates[achieved[0]])
                else:
                    required_n.append(600)

            ax.plot(dims, required_n, label=method, color=colors[method],
                    marker=markers[method], markersize=7, linewidth=2)

        ax.axhline(y=500, color="gray", linestyle=":", alpha=0.5, label="n=500 limit")
        ax.set_xlabel("Dimension (d)", fontsize=12)
        ax.set_ylabel("Samples required", fontsize=12)
        ax.set_title(f"Target MSE = {threshold}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 650)

    fig.suptitle("Exp 4: Sample Requirement to Achieve Target MSE", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "exp4_sample_requirement.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [Saved] {OUTPUT_DIR}/exp4_sample_requirement.png")


# ---- Exp 5: 推理时间 (仿 Section 4.6) ----
def experiment5_inference_time():
    print("\n" + "=" * 70)
    print("Exp 5: Inference time comparison")
    print("=" * 70)

    dims = [5, 10, 20]
    n_sizes = [50, 100, 200, 500]

    ksg = KSGEstimator(k=5)
    mine = MINEEstimator(hidden_dim=128, n_layers=2, lr=1e-4, iters=1000, batch_size=256)
    mist = MISTEstimator()

    times = {m: np.zeros((len(dims), len(n_sizes))) for m in ["KSG", "MINE", "MIST"]}

    for di, d in enumerate(dims):
        for ni, n in enumerate(n_sizes):
            X, Y, _ = generate_single_sample(IMD_FAMILIES[0], n, d, SEED + di * 100 + ni)

            # warmup
            _ = ksg.estimate(X[:min(10, n)], Y[:min(10, n)])

            _, t = run_estimator(ksg, X, Y)
            times["KSG"][di, ni] = t
            _, t = run_estimator(mine, X, Y)
            times["MINE"][di, ni] = t
            _, t = run_estimator(mist, X, Y)
            times["MIST"][di, ni] = t

            print(f"  d={d:2d}, n={n:3d} | "
                  f"KSG: {times['KSG'][di, ni]:.4f}s | "
                  f"MINE: {times['MINE'][di, ni]:.4f}s | "
                  f"MIST: {times['MIST'][di, ni]:.4f}s")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"KSG": "#2196F3", "MINE": "#FF5722", "MIST": "#4CAF50"}

    for i, method in enumerate(["KSG", "MINE", "MIST"]):
        ax = axes[i]
        for di, d in enumerate(dims):
            ax.plot(n_sizes, times[method][di, :], marker="o", label=f"d={d}", linewidth=2)
        ax.set_xlabel("Sample size (n)", fontsize=12)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_title(method, fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    fig.suptitle("Exp 5: Inference Time Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "exp5_inference_time.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n  Average time per sample:")
    for method in ["KSG", "MINE", "MIST"]:
        total_time = times[method].sum()
        total_samples = sum(n_sizes) * len(dims)
        print(f"    {method}: {total_time / total_samples:.6f} s/sample")

    print(f"\n  [Saved] {OUTPUT_DIR}/exp5_inference_time.png")


# ============================================================
#  主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MI Estimation Experiments (MIST paper style)")
    parser.add_argument("--exp", type=int, default=0, help="Run specific experiment (1-5), 0=all")
    args = parser.parse_args()

    experiments = {
        1: ("MSE comparison table", experiment1_mse_table),
        2: ("Predicted vs True MI", experiment2_pred_vs_true),
        3: ("Bias/Variance/MSE heatmaps", experiment3_heatmaps),
        4: ("Sample requirement profiles", experiment4_sample_requirement),
        5: ("Inference time", experiment5_inference_time),
    }

    print("=" * 70)
    print("Mutual Information Estimation: KSG vs MINE vs MIST")
    print("Experimental design following Gritsai et al. (2025), arXiv:2511.18945")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    if args.exp == 0:
        for idx in sorted(experiments.keys()):
            name, func = experiments[idx]
            print(f"\n>>> Experiment {idx}: {name}")
            func()
    elif args.exp in experiments:
        name, func = experiments[args.exp]
        print(f"\n>>> Experiment {args.exp}: {name}")
        func()
    else:
        print(f"Error: experiment {args.exp} not found. Choose 1-5.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("All experiments completed. Figures saved to:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
