import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

RESULT_DIR = "light_amformer_baseline_results"
BEST_RUN_DIR = os.path.join(RESULT_DIR, "best_run_detailed")
SAVE_DIR = os.path.join(RESULT_DIR, "figures")
os.makedirs(SAVE_DIR, exist_ok=True)

learning_files = sorted(glob.glob(os.path.join(BEST_RUN_DIR, "fold_*_learning_curve.csv")))
prediction_files = sorted(glob.glob(os.path.join(BEST_RUN_DIR, "fold_*_predictions.csv")))

# =========================
# 左图：五折平均训练曲线
# =========================

curve_dfs = [pd.read_csv(f) for f in learning_files]
min_len = min(len(df) for df in curve_dfs)

train_loss_all = np.array([df["train_loss"].values[:min_len] for df in curve_dfs])
val_auc_all = np.array([df["val_auc"].values[:min_len] for df in curve_dfs])

epochs = np.arange(min_len)

train_loss_mean = train_loss_all.mean(axis=0)
train_loss_std = train_loss_all.std(axis=0)

val_auc_mean = val_auc_all.mean(axis=0)
val_auc_std = val_auc_all.std(axis=0)


# =========================
# 右图：平均 ROC 曲线
# =========================

mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []

for file in prediction_files:
    df = pd.read_csv(file)

    y_true = df["y_true"].values
    y_prob = df["y_prob"].values

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fold_auc = auc(fpr, tpr)
    aucs.append(fold_auc)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

tprs = np.array(tprs)

mean_tpr = tprs.mean(axis=0)
std_tpr = tprs.std(axis=0)

mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)


# =========================
# 画图
# =========================

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：训练曲线
ax1 = axes[0]
ax2 = ax1.twinx()

l1, = ax1.plot(
    epochs,
    train_loss_mean,
    linewidth=2,
    label="Train Loss"
)

ax1.fill_between(
    epochs,
    train_loss_mean - train_loss_std,
    train_loss_mean + train_loss_std,
    alpha=0.15
)

l2, = ax2.plot(
    epochs,
    val_auc_mean,
    linestyle="--",
    linewidth=2,
    color=l1.get_color(),
    label="Validation AUC"
)

ax2.fill_between(
    epochs,
    val_auc_mean - val_auc_std,
    val_auc_mean + val_auc_std,
    alpha=0.10,
    color=l1.get_color()
)

ax1.set_title("(A) Five-fold Mean Training Curve of AMFormer")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train Loss")
ax2.set_ylabel("Validation AUC")
ax2.set_ylim(0.45, 1.00)
ax1.grid(alpha=0.3)
ax1.legend([l1, l2], ["Train Loss", "Validation AUC"], loc="center right")


# 右图：平均 ROC 曲线
ax3 = axes[1]

ax3.plot(
    mean_fpr,
    mean_tpr,
    linewidth=2,
    label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})"
)

ax3.fill_between(
    mean_fpr,
    np.maximum(mean_tpr - std_tpr, 0),
    np.minimum(mean_tpr + std_tpr, 1),
    alpha=0.15,
    label="± 1 SD"
)

ax3.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random"
)

ax3.set_title("(B) Mean ROC Curve of AMFormer")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1.05)
ax3.grid(alpha=0.3)
ax3.legend(loc="lower right", fontsize=9)

plt.tight_layout()

save_path = os.path.join(SAVE_DIR, "amformer_mean_training_mean_roc.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)