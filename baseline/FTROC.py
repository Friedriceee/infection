import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

OUTPUT_DIR = "figures"
FT_DIR = "ft_tune"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. 手动录入调参结果
# =========================
results = pd.DataFrame([
    ["ft_01", 16, 2, 1, 0.1, 0.001, 0.0001, 128, 50, 8, 43, 0.8948, 0.7869, 0.6207, 0.6792, 0.6486, 0.8488],
    ["ft_02", 16, 2, 2, 0.2, 0.0005, 0.0001, 128, 50, 8, 9, 0.6943, 0.3552, 0.3054, 0.9623, 0.4636, 0.6668],
    ["ft_03", 32, 4, 1, 0.2, 0.001, 0.0001, 128, 50, 8, 47, 0.8980, 0.8251, 0.7143, 0.6604, 0.6863, 0.8530],
    ["ft_04", 32, 4, 2, 0.2, 0.0005, 0.0001, 128, 50, 8, 50, 0.8658, 0.7705, 0.5797, 0.7547, 0.6557, 0.8518],
    ["ft_05", 32, 4, 2, 0.3, 0.0005, 0.001, 128, 50, 8, 50, 0.8642, 0.7486, 0.5422, 0.8491, 0.6618, 0.8589],
    ["ft_06", 64, 4, 2, 0.3, 0.0002, 0.0001, 64, 50, 8, 50, 0.8281, 0.6448, 0.4302, 0.6981, 0.5324, 0.7418],
], columns=[
    "name", "d_token", "n_heads", "n_layers", "dropout", "lr", "weight_decay",
    "batch_size", "max_epochs", "patience", "best_epoch", "best_val_auc",
    "test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc"
])

results["config_id"] = np.arange(1, len(results) + 1)
results["config_label"] = results["name"]

# =========================
# 2. 图一：参数趋势 + PR散点合成图
# =========================
def plot_ft_parameter_and_pr(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # A. Parameter trend
    ax = axes[0]
    ax.plot(
        results["config_id"],
        results["test_auc"],
        marker="o",
        linewidth=2,
        label="AUC"
    )
    ax.plot(
        results["config_id"],
        results["test_f1"],
        marker="s",
        linewidth=2,
        label="F1-score"
    )

    # 标注 ft_03
    ft3 = results[results["name"] == "ft_03"].iloc[0]
    ax.scatter(ft3["config_id"], ft3["test_auc"], s=80, zorder=5)
    ax.text(
        ft3["config_id"],
        ft3["test_auc"] + 0.015,
        "Selected: ft_03",
        ha="center",
        fontsize=9
    )

    ax.set_xticks(results["config_id"])
    ax.set_xticklabels(results["config_label"], rotation=30)
    ax.set_xlabel("FT-Transformer Configurations")
    ax.set_ylabel("Score")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("(A) Parameter Trend of FT-Transformer")
    ax.legend()
    ax.grid(alpha=0.3)

    # B. Precision-Recall scatter
    ax = axes[1]
    ax.scatter(
        results["test_precision"],
        results["test_recall"],
        s=90
    )

    for _, row in results.iterrows():
        ax.text(
            row["test_precision"] + 0.008,
            row["test_recall"] + 0.008,
            row["name"],
            fontsize=9
        )

    ax.scatter(
        ft3["test_precision"],
        ft3["test_recall"],
        s=160,
        marker="*",
        zorder=6,
        label="Selected ft_03"
    )

    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_xlim(0.25, 0.80)
    ax.set_ylim(0.60, 1.00)
    ax.set_title("(B) Precision–Recall Trade-off")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_ft_parameter_pr_analysis.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_ft_parameter_pr_analysis.pdf"), bbox_inches="tight")
    plt.show()


# =========================
# 3. 图二：ft_03 训练曲线 + ROC
# =========================
def plot_ft03_training_and_roc():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # A. Training curve
    history_path = os.path.join(FT_DIR, "ft_03_history.csv")
    hist = pd.read_csv(history_path)

    ax1 = axes[0]
    ax1.plot(hist["epoch"], hist["train_loss"], linewidth=2, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("(A) Training Curve of ft_03")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(hist["epoch"], hist["val_auc"], linewidth=2, linestyle="--", label="Validation AUC")
    ax2.set_ylabel("Validation AUC")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right", fontsize=9)

    # B. ROC
    pred_path = os.path.join(FT_DIR, "ft_03_test_predictions.csv")
    pred = pd.read_csv(pred_path)

    y_true = pred["y_true"].values
    y_prob = pred["y_prob"].values

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    ax = axes[1]
    ax.plot(fpr, tpr, linewidth=2, label=f"ft_03 (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("(B) ROC Curve of ft_03")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_ft03_training_roc.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_ft03_training_roc.pdf"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_ft_parameter_and_pr(results)
    plot_ft03_training_and_roc()