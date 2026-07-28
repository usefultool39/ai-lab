"""T002 · 感知机训练入口：造数 → 训练 → 评估 → 画图。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from perceptron import Perceptron

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
RESULT_PATH = ROOT / "results.json"

SEED = 42
N_SAMPLES = 200
TEST_SIZE = 0.25
LR = 1.0
MAX_EPOCHS = 100


def make_toy_data(n_samples: int = N_SAMPLES, seed: int = SEED):
    """生成二维线性可分数据（保证 PLA 可收敛）。"""
    rng = np.random.default_rng(seed)
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos
    # 两类高斯簇，均值拉开，保证线性可分
    X_pos = rng.normal(loc=[2.0, 2.0], scale=0.55, size=(n_pos, 2))
    X_neg = rng.normal(loc=[-2.0, -2.0], scale=0.55, size=(n_neg, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_pos + [-1] * n_neg, dtype=float)
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def plot_decision_boundary(model: Perceptron, X: np.ndarray, y: np.ndarray, path: Path):
    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.decision_function(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, zz, levels=0, colors=["#fbb6ce", "#9ae6b4"], alpha=0.35)
    ax.contour(xx, yy, zz, levels=[0], colors=["#2d3748"], linewidths=2)

    ax.scatter(
        X[y == 1, 0], X[y == 1, 1],
        c="#38a169", edgecolors="white", s=55, label="+1", zorder=3,
    )
    ax.scatter(
        X[y == -1, 0], X[y == -1, 1],
        c="#e53e3e", edgecolors="white", s=55, label="-1", zorder=3,
    )
    ax.set_title("Perceptron Decision Boundary")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(errors_history: list[int], path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = np.arange(1, len(errors_history) + 1)
    ax.plot(epochs, errors_history, marker="o", color="#667eea", linewidth=2)
    ax.fill_between(epochs, errors_history, alpha=0.15, color="#667eea")
    ax.set_title("Training Convergence (misclassifications per epoch)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Errors")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)

    X, y = make_toy_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    model = Perceptron(lr=LR, max_epochs=MAX_EPOCHS)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    boundary_path = FIG_DIR / "decision_boundary.png"
    conv_path = FIG_DIR / "convergence.png"
    plot_decision_boundary(model, X, y, boundary_path)
    plot_convergence(model.errors_history, conv_path)

    results = {
        "n_samples": int(N_SAMPLES),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "lr": LR,
        "max_epochs": MAX_EPOCHS,
        "epochs_ran": model.n_epochs_ran,
        "n_updates": model.n_updates,
        "final_errors": model.errors_history[-1] if model.errors_history else None,
        "errors_history": model.errors_history,
        "w": model.w.tolist() if model.w is not None else None,
        "b": float(model.b),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "figures": [
            str(boundary_path.relative_to(ROOT)).replace("\\", "/"),
            str(conv_path.relative_to(ROOT)).replace("\\", "/"),
        ],
    }
    RESULT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== T002 Perceptron ===")
    print(f"epochs_ran : {model.n_epochs_ran}")
    print(f"n_updates  : {model.n_updates}")
    print(f"w, b       : {model.w}, {model.b:.4f}")
    print(f"train_acc  : {train_acc:.2%}")
    print(f"test_acc   : {test_acc:.2%}")
    print(f"figures    : {boundary_path.name}, {conv_path.name}")
    print(f"results    : {RESULT_PATH.name}")


if __name__ == "__main__":
    main()
