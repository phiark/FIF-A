# benchmark_improved.py
# Attention vs FIF 玩具基准：多 seed、误差条、CSV、能量曲线
# 特色：
# - 冲突率强制 clamp 到 [0,1]
# - μ 使用对角向量（Softplus 保正）
# - 多 seed (默认5) 重复实验，均值±标准差
# - 可选“压力 q”项，用 query 嵌入模拟任务压力
# - 导出 CSV：results.csv

import os, csv, math, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# -------------------------
# Utils
# -------------------------
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def param_count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# -------------------------
# Dataset
# -------------------------
class ConflictDataset(Dataset):
    """
    生成两模态序列 A,B；标签由 A 决定；B 以概率 conflict_rate 发生强冲突
    """

    def __init__(
        self, n_samples=1024, seq_len=8, d=16, conflict_rate=0.3, use_pressure=False
    ):
        self.n = n_samples
        self.seq_len = seq_len
        self.d = d
        self.use_pressure = use_pressure
        self.w = np.random.randn(d)
        # clamp 概率到 [0,1]
        self.conflict = float(np.clip(conflict_rate, 0.0, 1.0))
        self.data = []
        for _ in range(n_samples):
            A = np.random.randn(seq_len, d)
            score = A.sum(axis=0) @ self.w
            y = 1 if score > 0 else 0
            B = A.copy()
            mask = np.random.rand(seq_len) < self.conflict
            for t in range(seq_len):
                if mask[t]:
                    B[t] = -A[t] + 0.5 * np.random.randn(d)  # 强冲突
                else:
                    B[t] = A[t] + 0.1 * np.random.randn(d)  # 轻噪声
            # 可选“压力 q”：强度随冲突率缩放（仅用于可视化可控性）
            q = None
            if self.use_pressure:
                q = np.ones((seq_len, d), dtype=np.float32) * self.conflict
            self.data.append((A.astype(np.float32), B.astype(np.float32), int(y), q))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.data[i]


def collate(batch):
    A = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
    B = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    y = torch.tensor([b[2] for b in batch], dtype=torch.long)
    # q 可能为 None
    if batch[0][3] is None:
        q = None
    else:
        q = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
    return A, B, y, q


# -------------------------
# Baseline: tiny attention-like
# -------------------------
class BaselineAttention(nn.Module):
    def __init__(self, d=16, hidden=64, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(2 * d, hidden)
        self.trans = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, batch_first=True
        )
        self.cls = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 2)
        )

    def forward(self, A, B, q=None):
        x = torch.cat([A, B], dim=-1)
        h = self.input_proj(x)
        h = self.trans(h)
        return self.cls(h.mean(dim=1))


# -------------------------
# FIF Layer (改进版：μ 为对角向量 + 轻平滑 + 可选压力)
# -------------------------
class FIFLayer(nn.Module):
    def __init__(self, d=16, hidden=32, K=6, eta=0.15, lam=0.05):
        super().__init__()
        self.K, self.eta, self.lam = K, eta, lam
        self.mu_net = nn.Sequential(
            nn.Linear(2 * d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
            nn.Softplus(),  # μ >= 0，逐通道
        )
        # 压力度量映射，把 q 投到每通道增益上（可选）
        self.q_gate = nn.Sequential(nn.Linear(d, d), nn.Tanh())

    def smooth1d(self, x):
        # x: [b, seq, d] 简单 1D 平滑
        x_left = torch.roll(x, 1, dims=1)
        x_right = torch.roll(x, -1, dims=1)
        return x - self.lam * (2 * x - x_left - x_right)

    def forward(self, A, B, q=None):
        mu = self.mu_net(torch.cat([A, B], dim=-1))  # [b, seq, d]
        hA, hB = A.clone(), B.clone()
        if q is not None:
            # q: [b, seq, d]，开关强度映射后调制 μ
            gate = self.q_gate(q)  # [-1,1]
            mu = mu * (1.0 + 0.5 * gate)  # 少量调制即可
        for _ in range(self.K):
            delta = hA - hB
            hA = hA - self.eta * (mu * delta)
            hB = hB + self.eta * (mu * delta)
            hA = self.smooth1d(hA)
            hB = self.smooth1d(hB)
        energy = (mu * (hA - hB).pow(2)).sum(dim=[1, 2]) * 0.5
        return hA, hB, energy


class FIFModel(nn.Module):
    def __init__(self, d=16, hidden=64, K=6, eta=0.15, lam=0.05):
        super().__init__()
        self.fif = FIFLayer(d=d, hidden=32, K=K, eta=eta, lam=lam)
        self.cls = nn.Sequential(
            nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, 2)
        )

    def forward(self, A, B, q=None):
        hA, hB, energy = self.fif(A, B, q)
        vA = hA.mean(dim=1)
        vB = hB.mean(dim=1)
        v = torch.cat([vA, vB], dim=-1)
        logits = self.cls(v)
        return logits, energy


# -------------------------
# Train / Eval
# -------------------------
def train_epoch(model, opt, loader, device):
    model.train()
    total = 0
    loss_sum = 0
    for A, B, y, q in loader:
        A, B, y = A.to(device), B.to(device), y.to(device)
        q = q.to(device) if q is not None else None
        opt.zero_grad()
        if isinstance(model, FIFModel):
            logits, _ = model(A, B, q)
        else:
            logits = model(A, B, q)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * A.size(0)
        total += A.size(0)
    return loss_sum / total


def eval_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    energies = []
    with torch.no_grad():
        for A, B, y, q in loader:
            A, B, y = A.to(device), B.to(device), y.to(device)
            q = q.to(device) if q is not None else None
            if isinstance(model, FIFModel):
                logits, energy = model(A, B, q)
                energies.append(energy.cpu().numpy())
            else:
                logits = model(A, B, q)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    mean_energy = float(np.mean(np.concatenate(energies))) if energies else 0.0
    return acc, mean_energy


# -------------------------
# Runner
# -------------------------
def run(
    conflicts,
    seeds=5,
    train_mixture=(0.0, 0.2, 0.4, 0.6, 0.8),
    epochs=8,
    d=16,
    seq_len=8,
    use_pressure=True,
    device="cpu",
):
    results = []  # list of dict rows
    for c in conflicts:
        accB_list = []
        accF_list = []
        en_list = []
        for s in range(seeds):
            set_seed(1000 + s)
            # 训练集：混合冲突
            train_ds = ConflictDataset(
                n_samples=1024,
                seq_len=seq_len,
                d=d,
                conflict_rate=random.choice(train_mixture),
                use_pressure=use_pressure,
            )
            train_loader = DataLoader(
                train_ds, batch_size=64, shuffle=True, collate_fn=collate
            )

            # 测试集：固定冲突 c（会被 clamp 到 [0,1]）
            test_ds = ConflictDataset(
                n_samples=512,
                seq_len=seq_len,
                d=d,
                conflict_rate=c,
                use_pressure=use_pressure,
            )
            test_loader = DataLoader(
                test_ds, batch_size=64, shuffle=False, collate_fn=collate
            )

            base = BaselineAttention(d=d, hidden=64, n_heads=4).to(device)
            fif = FIFModel(d=d, hidden=64, K=6, eta=0.15, lam=0.05).to(device)
            opt_b = torch.optim.Adam(base.parameters(), lr=2e-3)
            opt_f = torch.optim.Adam(fif.parameters(), lr=2e-3)

            for _ in range(epochs):
                train_epoch(base, opt_b, train_loader, device)
                train_epoch(fif, opt_f, train_loader, device)

            accB, _ = eval_model(base, test_loader, device)
            accF, en = eval_model(fif, test_loader, device)
            accB_list.append(accB)
            accF_list.append(accF)
            en_list.append(en)

        row = {
            "conflict": float(np.clip(c, 0, 1)),
            "acc_baseline_mean": float(np.mean(accB_list)),
            "acc_baseline_std": float(np.std(accB_list)),
            "acc_fif_mean": float(np.mean(accF_list)),
            "acc_fif_std": float(np.std(accF_list)),
            "fif_energy_mean": float(np.mean(en_list)),
            "fif_energy_std": float(np.std(en_list)),
        }
        results.append(row)
        print(
            f"conflict={c:.2f} | "
            f"Base {row['acc_baseline_mean']:.3f}±{row['acc_baseline_std']:.3f} | "
            f"FIF {row['acc_fif_mean']:.3f}±{row['acc_fif_std']:.3f} | "
            f"Energy {row['fif_energy_mean']:.3f}"
        )
    return results


def save_csv(rows, path="results.csv"):
    keys = rows[0].keys()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {path}")


def plot_curves(rows, out_prefix="plot"):
    xs = [r["conflict"] for r in rows]
    b_mean = [r["acc_baseline_mean"] for r in rows]
    b_std = [r["acc_baseline_std"] for r in rows]
    f_mean = [r["acc_fif_mean"] for r in rows]
    f_std = [r["acc_fif_std"] for r in rows]
    e_mean = [r["fif_energy_mean"] for r in rows]
    e_std = [r["fif_energy_std"] for r in rows]

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        xs, b_mean, yerr=b_std, fmt="o--", capsize=3, label="Baseline Attention"
    )
    plt.errorbar(xs, f_mean, yerr=f_std, fmt="s-", capsize=3, label="FIF (friction)")
    plt.xlabel("Conflict rate (difficulty)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Difficulty (mean±std over seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_acc.png", dpi=160)
    print(f"[saved] {out_prefix}_acc.png")

    plt.figure(figsize=(6, 3.6))
    plt.errorbar(xs, e_mean, yerr=e_std, fmt="^-", capsize=3, label="FIF energy")
    plt.xlabel("Conflict rate (difficulty)")
    plt.ylabel("Energy")
    plt.title("FIF Energy vs Difficulty (mean±std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_energy.png", dpi=160)
    print(f"[saved] {out_prefix}_energy.png")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    print("Device:", args.device)
    # 打印参数量对齐情况（大致同级）
    base_tmp = BaselineAttention()
    fif_tmp = FIFModel()
    print(
        f"Param baseline: {param_count(base_tmp):,d} | FIF: {param_count(fif_tmp):,d}"
    )

    conflicts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = run(
        conflicts=conflicts, seeds=args.seeds, epochs=args.epochs, device=args.device
    )

    save_csv(rows, "results.csv")
    plot_curves(rows, "results")
    print("Done.")
