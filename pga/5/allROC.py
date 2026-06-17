import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# ============================================================
# 路径设置
# ============================================================

BASELINE_DIR = "/Users/wangqinyang.5/Desktop/Infection/baseline/baselineresults"
AMFORMER_DIR = "/Users/wangqinyang.5/Desktop/Infection/amformerv2/light_amformer_baseline_results/best_run_detailed"

# 这里改成你 simulated 六个文件所在的文件夹
PGA_AMFORMER_DIR = "/Users/wangqinyang.5/Desktop/Infection/pga/5/pga"

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 模型名称、颜色、线宽
# ============================================================

MODEL_NAMES = {
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "amformer": "AMFormer",
    "pga_amformer": "PGFormer",
}

MODEL_COLORS = {
    "logistic": "#7F7F7F",
    "random_forest": "#4C78A8",
    "xgboost": "#F58518",
    "amformer": "#54A24B",
    "pga_amformer": "#D62728",
}

MODEL_LINEWIDTHS = {
    "logistic": 2.0,
    "random_forest": 2.0,
    "xgboost": 2.0,
    "amformer": 2.0,
    "pga_amformer": 2.8,
}


# ============================================================
# 绘图风格
# ============================================================

def set_plot_style():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti SC",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 9

    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"


# ============================================================
# 自动识别列名
# ============================================================

def find_col(df, candidates, file_path):
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"\nCannot find required column in file:\n{file_path}\n"
        f"Candidate columns: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def normalize_prediction_dataframe(df, file_path):
    y_col = find_col(
        df,
        candidates=[
            "y_true", "label", "target", "outcome",
            "true_label", "true", "y"
        ],
        file_path=file_path
    )

    prob_col = find_col(
        df,
        candidates=[
            "y_prob", "prob", "pred_prob", "predict_prob",
            "prediction", "probability", "pga_prob",
            "positive_prob", "predicted_probability",
            "pred_proba", "score"
        ],
        file_path=file_path
    )

    out = pd.DataFrame({
        "y_true": df[y_col].astype(int),
        "y_prob": df[prob_col].astype(float)
    })

    for c in ["fold", "fold_id", "cv_fold", "Fold"]:
        if c in df.columns:
            out["fold"] = df[c].astype(int)
            break

    return out


# ============================================================
# 读取传统模型 pooled OOF
# ============================================================

def load_standard_pooled_predictions(model_name, result_dir, n_folds=5):
    dfs = []

    for fold in range(1, n_folds + 1):
        path = os.path.join(result_dir, f"{model_name}_fold{fold}_predictions.csv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)
        df = normalize_prediction_dataframe(df, path)
        df["fold"] = fold
        dfs.append(df)

    return pd.concat(dfs, axis=0, ignore_index=True)


# ============================================================
# 读取 AMFormer pooled OOF
# ============================================================

def load_amformer_pooled_predictions(result_dir, n_folds=5):
    dfs = []

    for fold in range(1, n_folds + 1):
        path = os.path.join(result_dir, f"fold_{fold}_predictions.csv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)
        df = normalize_prediction_dataframe(df, path)
        df["fold"] = fold
        dfs.append(df)

    return pd.concat(dfs, axis=0, ignore_index=True)


# ============================================================
# 读取 PGFormer pooled OOF
# 重点：优先读取五个 fold 文件，而不是只读 merged
# ============================================================

def load_pga_amformer_pooled_predictions(result_dir, n_folds=5):
    """
    优先读取：
        simulated_fold1_predictions.csv
        simulated_fold2_predictions.csv
        ...
        simulated_fold5_predictions.csv

    然后自动 concat 成 pooled OOF。
    同时检查 simulated_merged_fold_predictions_target.csv 是否与五折合并结果一致。
    """

    print("\n" + "=" * 90)
    print("Loading PGFormer from five simulated fold files")
    print("=" * 90)
    print("PGA_AMFORMER_DIR =", result_dir)

    if not os.path.exists(result_dir):
        raise FileNotFoundError(f"PGA_AMFORMER_DIR does not exist: {result_dir}")

    print("\nFiles in PGA_AMFORMER_DIR:")
    for f in sorted(os.listdir(result_dir)):
        print(" -", f)

    dfs = []

    for fold in range(1, n_folds + 1):
        fold_file = os.path.join(result_dir, f"simulated_fold{fold}_predictions.csv")

        if not os.path.exists(fold_file):
            raise FileNotFoundError(
                f"\nMissing PGA fold file:\n{fold_file}\n\n"
                f"请确认文件名必须是：simulated_fold{fold}_predictions.csv"
            )

        raw = pd.read_csv(fold_file)
        df = normalize_prediction_dataframe(raw, fold_file)
        df["fold"] = fold

        auc_fold = roc_auc_score(df["y_true"], df["y_prob"])
        ap_fold = average_precision_score(df["y_true"], df["y_prob"])

        print(f"\nFold {fold}:")
        print(f"  file = {fold_file}")
        print(f"  n = {len(df)}")
        print(f"  positive = {int(df['y_true'].sum())}")
        print(f"  negative = {int(len(df) - df['y_true'].sum())}")
        print(f"  AUC = {auc_fold:.6f}")
        print(f"  AP  = {ap_fold:.6f}")
        print(f"  prob min/max = {df['y_prob'].min():.6f} / {df['y_prob'].max():.6f}")

        dfs.append(df)

    pooled = pd.concat(dfs, axis=0, ignore_index=True)

    pooled_auc = roc_auc_score(pooled["y_true"], pooled["y_prob"])
    pooled_ap = average_precision_score(pooled["y_true"], pooled["y_prob"])

    print("\n" + "-" * 90)
    print("PGAFormer pooled result from five fold files")
    print("-" * 90)
    print(f"n = {len(pooled)}")
    print(f"positive = {int(pooled['y_true'].sum())}")
    print(f"negative = {int(len(pooled) - pooled['y_true'].sum())}")
    print(f"pooled AUC = {pooled_auc:.6f}")
    print(f"pooled AP  = {pooled_ap:.6f}")
    print("-" * 90)

    # ------------------------------------------------------------
    # 检查 merged 文件是否一致
    # ------------------------------------------------------------

    merged_file = os.path.join(result_dir, "simulated_merged_fold_predictions_target.csv")

    if os.path.exists(merged_file):
        raw_merged = pd.read_csv(merged_file)
        merged = normalize_prediction_dataframe(raw_merged, merged_file)

        merged_auc = roc_auc_score(merged["y_true"], merged["y_prob"])
        merged_ap = average_precision_score(merged["y_true"], merged["y_prob"])

        print("\nChecking merged file:")
        print(f"  file = {merged_file}")
        print(f"  n = {len(merged)}")
        print(f"  merged AUC = {merged_auc:.6f}")
        print(f"  merged AP  = {merged_ap:.6f}")

        same_len = len(merged) == len(pooled)
        same_y = same_len and np.array_equal(
            merged["y_true"].values,
            pooled["y_true"].values
        )
        same_prob = same_len and np.allclose(
            merged["y_prob"].values,
            pooled["y_prob"].values,
            atol=1e-12
        )

        if same_len and same_y and same_prob:
            print("  Result: merged file is consistent with five fold files.")
        else:
            print("\n  WARNING:")
            print("  merged file is NOT exactly consistent with five fold files.")
            print("  当前画图将使用 five fold files 合并结果，而不是 merged 文件。")

    else:
        print("\nNo merged file found.")
        print("Current plotting uses five fold files only.")

    print("=" * 90 + "\n")

    return pooled


# ============================================================
# 模型来源
# ============================================================

def get_model_sources():
    return [
        ("logistic", BASELINE_DIR, "standard"),
        ("random_forest", BASELINE_DIR, "standard"),
        ("xgboost", BASELINE_DIR, "standard"),
        ("amformer", AMFORMER_DIR, "amformer"),
        ("pga_amformer", PGA_AMFORMER_DIR, "pga_amformer"),
    ]


def load_model_pooled_data(model, result_dir, source_type):
    if source_type == "standard":
        return load_standard_pooled_predictions(model, result_dir, n_folds=5)

    if source_type == "amformer":
        return load_amformer_pooled_predictions(result_dir, n_folds=5)

    if source_type == "pga_amformer":
        return load_pga_amformer_pooled_predictions(result_dir, n_folds=5)

    raise ValueError(f"Unknown source_type: {source_type}")


def load_model_fold_data(model, result_dir, source_type, n_folds=5):
    dfs = []

    if source_type == "standard":
        for fold in range(1, n_folds + 1):
            path = os.path.join(result_dir, f"{model}_fold{fold}_predictions.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

            df = pd.read_csv(path)
            df = normalize_prediction_dataframe(df, path)
            df["fold"] = fold
            dfs.append(df)
        return dfs

    if source_type == "amformer":
        for fold in range(1, n_folds + 1):
            path = os.path.join(result_dir, f"fold_{fold}_predictions.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

            df = pd.read_csv(path)
            df = normalize_prediction_dataframe(df, path)
            df["fold"] = fold
            dfs.append(df)
        return dfs

    if source_type == "pga_amformer":
        for fold in range(1, n_folds + 1):
            path = os.path.join(result_dir, f"simulated_fold{fold}_predictions.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

            df = pd.read_csv(path)
            df = normalize_prediction_dataframe(df, path)
            df["fold"] = fold
            dfs.append(df)
        return dfs

    raise ValueError(f"Unknown source_type: {source_type}")


MODEL_POOLED_CACHE = {}
MODEL_FOLD_CACHE = {}


def get_model_pooled_data_cached(model, result_dir, source_type):
    key = (model, result_dir, source_type)
    if key not in MODEL_POOLED_CACHE:
        MODEL_POOLED_CACHE[key] = load_model_pooled_data(model, result_dir, source_type)
    return MODEL_POOLED_CACHE[key]


def get_model_fold_data_cached(model, result_dir, source_type):
    key = (model, result_dir, source_type)
    if key not in MODEL_FOLD_CACHE:
        MODEL_FOLD_CACHE[key] = load_model_fold_data(model, result_dir, source_type, n_folds=5)
    return MODEL_FOLD_CACHE[key]


def ensure_probability_range(y_prob, model_name):
    y_prob = np.asarray(y_prob, dtype=float)

    min_val = float(np.min(y_prob))
    max_val = float(np.max(y_prob))

    if min_val < 0.0 or max_val > 1.0:
        if max_val - min_val < 1e-12:
            print(f"[Warning] {model_name} has near-constant scores; map to 0.5 for calibration/DCA.")
            return np.full_like(y_prob, 0.5, dtype=float)

        print(f"[Warning] {model_name} scores outside [0,1], min-max scaling is applied for calibration/DCA.")
        y_prob = (y_prob - min_val) / (max_val - min_val + 1e-12)

    return np.clip(y_prob, 1e-6, 1 - 1e-6)


def collect_fold_metrics():
    rows = []

    for model, result_dir, source_type in get_model_sources():
        fold_dfs = get_model_fold_data_cached(model, result_dir, source_type)

        for fold_df in fold_dfs:
            fold_id = int(fold_df["fold"].iloc[0])

            y_true = fold_df["y_true"].values.astype(int)
            y_prob = fold_df["y_prob"].values.astype(float)
            y_prob_for_cal = ensure_probability_range(y_prob, MODEL_NAMES[model])

            rows.append({
                "model_key": model,
                "model": MODEL_NAMES[model],
                "fold": fold_id,
                "n": len(fold_df),
                "positive": int(np.sum(y_true)),
                "negative": int(len(y_true) - np.sum(y_true)),
                "prevalence": float(np.mean(y_true)),
                "auc": float(roc_auc_score(y_true, y_prob)),
                "ap": float(average_precision_score(y_true, y_prob)),
                "brier": float(brier_score_loss(y_true, y_prob_for_cal)),
            })

    out = pd.DataFrame(rows).sort_values(["model_key", "fold"]).reset_index(drop=True)
    return out


# ============================================================
# 检查 pooled 数据
# ============================================================

def inspect_pooled_data(model_name, df):
    y_true = df["y_true"].values
    y_prob = df["y_prob"].values

    auc_value = roc_auc_score(y_true, y_prob)
    ap_value = average_precision_score(y_true, y_prob)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)
    print(f"n = {len(df)}")
    print(f"positive = {int(np.sum(y_true))}")
    print(f"negative = {int(len(y_true) - np.sum(y_true))}")
    print(f"prevalence = {np.mean(y_true):.4f}")
    print(f"prob min/max = {np.min(y_prob):.6f} / {np.max(y_prob):.6f}")
    print(f"unique prob count = {len(np.unique(np.round(y_prob, 6)))}")
    print(f"pooled AUC = {auc_value:.6f}")
    print(f"pooled AP  = {ap_value:.6f}")

    return auc_value, ap_value


# ============================================================
# 绘制 pooled ROC
# ============================================================

def plot_pooled_roc():
    set_plot_style()

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=300)

    for model, result_dir, source_type in get_model_sources():
        pooled_df = get_model_pooled_data_cached(model, result_dir, source_type)
        auc_value, _ = inspect_pooled_data(MODEL_NAMES[model], pooled_df)

        y_true = pooled_df["y_true"].values
        y_prob = pooled_df["y_prob"].values

        fpr, tpr, _ = roc_curve(y_true, y_prob)

        ax.plot(
            fpr,
            tpr,
            color=MODEL_COLORS[model],
            linewidth=MODEL_LINEWIDTHS[model],
            label=f"{MODEL_NAMES[model]} (AUC = {auc_value:.3f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.1,
        color="#9E9E9E",
        label="Random"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        fontsize=8.5
    )

    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "fig_pooled_oof_roc_5models.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved ROC figure: {out_path}")


# ============================================================
# 绘制 pooled PR
# ============================================================

def plot_pooled_pr():
    set_plot_style()

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=300)

    prevalence_values = []

    for model, result_dir, source_type in get_model_sources():
        pooled_df = get_model_pooled_data_cached(model, result_dir, source_type)
        _, ap_value = inspect_pooled_data(MODEL_NAMES[model], pooled_df)

        y_true = pooled_df["y_true"].values
        y_prob = pooled_df["y_prob"].values

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        prevalence_values.append(np.mean(y_true))

        ax.plot(
            recall,
            precision,
            color=MODEL_COLORS[model],
            linewidth=MODEL_LINEWIDTHS[model],
            label=f"{MODEL_NAMES[model]} (AP = {ap_value:.3f})"
        )

    prevalence = float(np.mean(prevalence_values))

    ax.hlines(
        prevalence,
        xmin=0,
        xmax=1,
        linestyle="--",
        linewidth=1.1,
        color="#9E9E9E",
        label=f"Prevalence = {prevalence:.3f}"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        fontsize=8.5
    )

    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "fig_pooled_oof_pr_5models.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved PR figure: {out_path}")


# ============================================================
# 绘制 pooled Calibration
# ============================================================

def plot_pooled_calibration(n_bins=10, strategy="quantile"):
    set_plot_style()

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=300)

    for model, result_dir, source_type in get_model_sources():
        pooled_df = get_model_pooled_data_cached(model, result_dir, source_type)

        y_true = pooled_df["y_true"].values.astype(int)
        y_prob = pooled_df["y_prob"].values.astype(float)
        y_prob_for_cal = ensure_probability_range(y_prob, MODEL_NAMES[model])

        frac_pos, mean_pred = calibration_curve(
            y_true,
            y_prob_for_cal,
            n_bins=n_bins,
            strategy=strategy
        )
        brier = brier_score_loss(y_true, y_prob_for_cal)

        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            markersize=3.8,
            color=MODEL_COLORS[model],
            linewidth=MODEL_LINEWIDTHS[model],
            label=f"{MODEL_NAMES[model]} (Brier = {brier:.3f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.1,
        color="#9E9E9E",
        label="Perfect Calibration"
    )

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        fontsize=8.5
    )

    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "fig_pooled_oof_calibration_5models.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved Calibration figure: {out_path}")


# ============================================================
# 绘制 pooled Decision Curve (DCA)
# ============================================================

def compute_net_benefit(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    net_benefits = []

    for t in thresholds:
        pred_pos = y_prob >= t
        tp = np.sum((pred_pos == 1) & (y_true == 1))
        fp = np.sum((pred_pos == 1) & (y_true == 0))

        nb = (tp / n) - (fp / n) * (t / (1 - t))
        net_benefits.append(nb)

    return np.asarray(net_benefits, dtype=float)


def plot_pooled_decision_curve():
    set_plot_style()

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=300)
    thresholds = np.linspace(0.01, 0.99, 99)

    prevalence_values = []

    for model, result_dir, source_type in get_model_sources():
        pooled_df = get_model_pooled_data_cached(model, result_dir, source_type)

        y_true = pooled_df["y_true"].values.astype(int)
        y_prob = pooled_df["y_prob"].values.astype(float)
        y_prob_for_dca = ensure_probability_range(y_prob, MODEL_NAMES[model])

        prevalence_values.append(float(np.mean(y_true)))
        nb = compute_net_benefit(y_true, y_prob_for_dca, thresholds)

        ax.plot(
            thresholds,
            nb,
            color=MODEL_COLORS[model],
            linewidth=MODEL_LINEWIDTHS[model],
            label=MODEL_NAMES[model]
        )

    prevalence = float(np.mean(prevalence_values))
    treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    treat_none = np.zeros_like(thresholds)

    ax.plot(
        thresholds,
        treat_all,
        linestyle="--",
        linewidth=1.1,
        color="#7A7A7A",
        label="Treat All"
    )
    ax.plot(
        thresholds,
        treat_none,
        linestyle=":",
        linewidth=1.2,
        color="#9E9E9E",
        label="Treat None"
    )

    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_xlim(0.01, 0.99)
    ax.grid(alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        fontsize=8.5
    )

    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "fig_pooled_oof_decision_curve_5models.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved Decision Curve figure: {out_path}")


# ============================================================
# 绘制 fold-level 训练曲线（5折代理曲线）
# ============================================================

def plot_fold_training_curves():
    set_plot_style()

    fold_metrics = collect_fold_metrics()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), dpi=300, sharex=True)

    ax_auc, ax_ap = axes

    for model, _, _ in get_model_sources():
        sub = (
            fold_metrics[fold_metrics["model_key"] == model]
            .sort_values("fold")
            .reset_index(drop=True)
        )

        ax_auc.plot(
            sub["fold"],
            sub["auc"],
            marker="o",
            markersize=4.0,
            color=MODEL_COLORS[model],
            linewidth=MODEL_LINEWIDTHS[model],
            label=MODEL_NAMES[model]
        )

        ax_ap.plot(
            sub["fold"],
            sub["ap"],
            marker="o",
            markersize=4.0,
            color=MODEL_COLORS[model],
            linewidth=MODEL_LINEWIDTHS[model],
            label=MODEL_NAMES[model]
        )

    x_ticks = sorted(fold_metrics["fold"].unique().tolist())

    ax_auc.set_title("Fold-wise AUC")
    ax_auc.set_xlabel("Fold")
    ax_auc.set_ylabel("AUC")
    ax_auc.set_xticks(x_ticks)
    ax_auc.grid(alpha=0.25)
    ax_auc.spines["top"].set_visible(False)
    ax_auc.spines["right"].set_visible(False)

    ax_ap.set_title("Fold-wise AP")
    ax_ap.set_xlabel("Fold")
    ax_ap.set_ylabel("Average Precision")
    ax_ap.set_xticks(x_ticks)
    ax_ap.grid(alpha=0.25)
    ax_ap.spines["top"].set_visible(False)
    ax_ap.spines["right"].set_visible(False)

    handles, labels = ax_auc.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        fontsize=8.5
    )

    fig.suptitle("Model Training Curves (5-fold Proxy)", y=1.03, fontsize=14)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "fig_training_curves_by_fold_5models.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved Training Curves figure: {out_path}")


# ============================================================
# 额外保存 pooled 指标表
# ============================================================

def save_metrics_summary():
    rows = []

    for model, result_dir, source_type in get_model_sources():
        pooled_df = get_model_pooled_data_cached(model, result_dir, source_type)

        y_true = pooled_df["y_true"].values.astype(int)
        y_prob = pooled_df["y_prob"].values.astype(float)
        y_prob_for_cal = ensure_probability_range(y_prob, MODEL_NAMES[model])

        auc_value = roc_auc_score(y_true, y_prob)
        ap_value = average_precision_score(y_true, y_prob)
        brier_value = brier_score_loss(y_true, y_prob_for_cal)

        rows.append({
            "model": MODEL_NAMES[model],
            "n": len(pooled_df),
            "positive": int(np.sum(y_true)),
            "negative": int(len(y_true) - np.sum(y_true)),
            "prevalence": float(np.mean(y_true)),
            "auc": auc_value,
            "ap": ap_value,
            "brier": brier_value,
        })

    summary = pd.DataFrame(rows)

    out_csv = os.path.join(OUTPUT_DIR, "pooled_oof_metrics_summary.csv")
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\nSaved metrics summary:")
    print(out_csv)
    print(summary)


# ============================================================
# 额外保存 fold 指标表
# ============================================================

def save_fold_metrics_summary():
    fold_metrics = collect_fold_metrics()

    out_csv = os.path.join(OUTPUT_DIR, "fold_metrics_summary.csv")
    fold_metrics.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\nSaved fold metrics summary:")
    print(out_csv)
    print(fold_metrics)


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("Start plotting pooled OOF ROC / PR curves")
    print("=" * 90)

    print("BASELINE_DIR =", BASELINE_DIR)
    print("AMFORMER_DIR =", AMFORMER_DIR)
    print("PGA_AMFORMER_DIR =", PGA_AMFORMER_DIR)
    print("OUTPUT_DIR =", OUTPUT_DIR)

    plot_pooled_roc()
    plot_pooled_pr()
    plot_pooled_calibration()
    plot_pooled_decision_curve()
    plot_fold_training_curves()
    save_metrics_summary()
    save_fold_metrics_summary()

    print("\nDone.")