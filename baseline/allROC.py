import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# =========================
# 路径设置
# =========================

BASELINE_DIR = "/Users/wangqinyang.5/Desktop/Infection/baseline/baselineresults"
FT_5FOLD_DIR = "/Users/wangqinyang.5/Desktop/Infection/baseline/ft_all_configs_5fold_optimized"
AMFORMER_DIR = "/Users/wangqinyang.5/Desktop/Infection/amformerv2/light_amformer_baseline_results/best_run_detailed"

# 注意：变量名不能写 PGA-AMFORMER_DIR，Python 会把 - 当成减号
PGA_AMFORMER_DIR = "/Users/wangqinyang.5/Desktop/Infection/pga/图"

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_FT_MODEL = "ft_06"

MODEL_NAMES = {
    "logistic": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "gbdt": "GBDT",
    "ft_06": "FT-Transformer",
    "amformer": "AMFormer",
    "pga_amformer": "PGA-AMFormer"
}


# =========================
# 工具函数：自动识别真实标签列和概率列
# =========================

def find_col(df, candidates, file_path):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Cannot find columns from {candidates} in file:\n{file_path}\n"
        f"Available columns: {list(df.columns)}"
    )


def normalize_fold_dataframe(df, file_path):
    """
    自动识别不同结果文件里的列名。
    支持常见列名：
    y_true / label / target / outcome
    y_prob / prob / pred_prob / probability / pga_prob
    fold / fold_id / cv_fold
    """
    y_col = find_col(
        df,
        candidates=[
            "y_true", "label", "target", "outcome", "true_label", "true", "y"
        ],
        file_path=file_path
    )

    prob_col = find_col(
        df,
        candidates=[
            "y_prob", "prob", "pred_prob", "predict_prob",
            "prediction", "probability", "pga_prob",
            "positive_prob", "predicted_probability"
        ],
        file_path=file_path
    )

    fold_col = None
    for c in ["fold", "fold_id", "cv_fold", "Fold"]:
        if c in df.columns:
            fold_col = c
            break

    out = pd.DataFrame({
        "y_true": df[y_col].astype(int),
        "y_prob": df[prob_col].astype(float)
    })

    if fold_col is not None:
        out["fold"] = df[fold_col].astype(int)

    return out


# =========================
# 读取五折预测结果：传统模型 / FT-Transformer
# =========================

def load_fold_predictions_standard(model_name, result_dir, n_folds=5):
    """
    适用于：
    baselineresults/logistic_fold1_predictions.csv
    ft_all_configs_5fold_optimized/ft_06_fold1_predictions.csv
    """
    fold_data = []

    for fold in range(1, n_folds + 1):
        path = os.path.join(result_dir, f"{model_name}_fold{fold}_predictions.csv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)
        df = normalize_fold_dataframe(df, path)

        fold_data.append({
            "y_true": df["y_true"].values,
            "y_prob": df["y_prob"].values
        })

    return fold_data


# =========================
# 读取 AMFormer 预测结果
# =========================

def load_fold_predictions_amformer(result_dir, n_folds=5):
    """
    适用于：
    light_amformer_baseline_results/best_run_detailed/fold_1_predictions.csv
    """
    fold_data = []

    for fold in range(1, n_folds + 1):
        path = os.path.join(result_dir, f"fold_{fold}_predictions.csv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)
        df = normalize_fold_dataframe(df, path)

        fold_data.append({
            "y_true": df["y_true"].values,
            "y_prob": df["y_prob"].values
        })

    return fold_data


# =========================
# 读取 PGA-AMFormer 预测结果
# =========================

def load_fold_predictions_pga_amformer(result_dir, n_folds=5):
    """
    兼容你截图里的 PGA-AMFormer 结果命名，例如：
    simulated_all_folds_target.csv
    simulated_merged_fold_predictions_target.csv
    simulated_metrics_target.csv
    target_metric_calculation.csv

    优先读取包含逐样本预测概率的文件。
    """

    candidate_files = [
        "simulated_all_folds_target.csv",
        "simulated_merged_fold_predictions_target.csv",
        "merged_fold_predictions_target.csv",
        "all_folds_target.csv",
        "fold_predictions_target.csv",
        "target_metric_calculation.csv"
    ]

    existing_files = []
    for fname in candidate_files:
        path = os.path.join(result_dir, fname)
        if os.path.exists(path):
            existing_files.append(path)

    # 如果上面没找到，就搜索所有 csv
    if len(existing_files) == 0:
        existing_files = glob.glob(os.path.join(result_dir, "*.csv"))

    if len(existing_files) == 0:
        raise FileNotFoundError(f"No CSV files found in PGA-AMFormer directory:\n{result_dir}")

    last_error = None

    for path in existing_files:
        try:
            df = pd.read_csv(path)
            df = normalize_fold_dataframe(df, path)

            # 如果文件里有 fold 列，按 fold 拆成五折
            if "fold" in df.columns:
                fold_data = []
                for fold in sorted(df["fold"].unique()):
                    sub = df[df["fold"] == fold]
                    if len(sub) == 0:
                        continue
                    fold_data.append({
                        "y_true": sub["y_true"].values,
                        "y_prob": sub["y_prob"].values
                    })

                if len(fold_data) > 0:
                    print(f"Loaded PGA-AMFormer predictions from: {path}")
                    return fold_data

            # 如果没有 fold 列，就当作 pooled prediction，作为一个整体画曲线
            else:
                print(f"Loaded PGA-AMFormer pooled predictions from: {path}")
                return [{
                    "y_true": df["y_true"].values,
                    "y_prob": df["y_prob"].values
                }]

        except Exception as e:
            last_error = e
            continue

    raise ValueError(
        f"Failed to load PGA-AMFormer prediction file from:\n{result_dir}\n"
        f"Last error:\n{last_error}"
    )


# =========================
# 计算平均 ROC / PR
# =========================

def compute_mean_roc(fold_data, mean_fpr):
    tprs = []
    aucs = []

    for item in fold_data:
        y_true = item["y_true"]
        y_prob = item["y_prob"]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fold_auc = roc_auc_score(y_true, y_prob)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0

        tprs.append(interp_tpr)
        aucs.append(fold_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    return mean_tpr, np.mean(aucs), np.std(aucs)


def compute_mean_pr(fold_data, mean_recall):
    precisions = []
    aps = []

    for item in fold_data:
        y_true = item["y_true"]
        y_prob = item["y_prob"]

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        aps.append(ap)

        recall_rev = recall[::-1]
        precision_rev = precision[::-1]

        interp_precision = np.interp(mean_recall, recall_rev, precision_rev)
        precisions.append(interp_precision)

    mean_precision = np.mean(precisions, axis=0)

    return mean_precision, np.mean(aps), np.std(aps)


# =========================
# 模型来源
# =========================

def get_model_sources():
    return [
        ("logistic", BASELINE_DIR, "standard"),
        ("decision_tree", BASELINE_DIR, "standard"),
        ("random_forest", BASELINE_DIR, "standard"),
        ("xgboost", BASELINE_DIR, "standard"),
        ("gbdt", BASELINE_DIR, "standard"),
        (BEST_FT_MODEL, FT_5FOLD_DIR, "standard"),
        ("amformer", AMFORMER_DIR, "amformer"),
        ("pga_amformer", PGA_AMFORMER_DIR, "pga_amformer"),
    ]


def load_model_fold_data(model, result_dir, source_type):
    if source_type == "standard":
        return load_fold_predictions_standard(model, result_dir, n_folds=5)
    elif source_type == "amformer":
        return load_fold_predictions_amformer(result_dir, n_folds=5)
    elif source_type == "pga_amformer":
        return load_fold_predictions_pga_amformer(result_dir, n_folds=5)
    else:
        raise ValueError(f"Unknown source_type: {source_type}")


# =========================
# 统一绘图风格
# =========================

def set_plot_style():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 8

    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"


# =========================
# 统一颜色和线型
# =========================

MODEL_COLORS = {
    "logistic": "#7F7F7F",
    "decision_tree": "#A6A6A6",
    "random_forest": "#4C78A8",
    "xgboost": "#F58518",
    "gbdt": "#54A24B",
    "ft_06": "#B279A2",
    "amformer": "#E45756",
    "pga_amformer": "#D62728"
}

MODEL_LINEWIDTHS = {
    "logistic": 1.8,
    "decision_tree": 1.8,
    "random_forest": 2.0,
    "xgboost": 2.2,
    "gbdt": 2.0,
    "ft_06": 2.2,
    "amformer": 2.6,
    "pga_amformer": 3.2
}

MODEL_LINESTYLES = {
    "logistic": "-",
    "decision_tree": "-",
    "random_forest": "-",
    "xgboost": "-",
    "gbdt": "-",
    "ft_06": "-",
    "amformer": "-",
    "pga_amformer": "-"
}


# =========================
# ROC 曲线
# =========================

def plot_all_models_mean_roc():
    set_plot_style()

    plt.figure(figsize=(7.6, 6.4), dpi=300)
    mean_fpr = np.linspace(0, 1, 200)

    for model, result_dir, source_type in get_model_sources():
        print(f"Loading ROC data: {MODEL_NAMES[model]}")
        fold_data = load_model_fold_data(model, result_dir, source_type)
        mean_tpr, mean_auc, std_auc = compute_mean_roc(fold_data, mean_fpr)

        plt.plot(
            mean_fpr,
            mean_tpr,
            linewidth=MODEL_LINEWIDTHS.get(model, 2.0),
            linestyle=MODEL_LINESTYLES.get(model, "-"),
            color=MODEL_COLORS.get(model, None),
            label=f"{MODEL_NAMES[model]} (AUC = {mean_auc:.3f} ± {std_auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.2,
        color="gray",
        label="Random"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean ROC Curves of All Models")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)

    plt.legend(loc="lower right", fontsize=7.8, frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "fig_all_models_with_pga_amformer_mean_roc.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png_path}")


# =========================
# PR 曲线
# =========================

def plot_all_models_mean_pr():
    set_plot_style()

    plt.figure(figsize=(7.6, 6.4), dpi=300)
    mean_recall = np.linspace(0, 1, 200)

    all_y_true = []

    for model, result_dir, source_type in get_model_sources():
        print(f"Loading PR data: {MODEL_NAMES[model]}")
        fold_data = load_model_fold_data(model, result_dir, source_type)

        for item in fold_data:
            all_y_true.extend(item["y_true"])

        mean_precision, mean_ap, std_ap = compute_mean_pr(fold_data, mean_recall)

        plt.plot(
            mean_recall,
            mean_precision,
            linewidth=MODEL_LINEWIDTHS.get(model, 2.0),
            linestyle=MODEL_LINESTYLES.get(model, "-"),
            color=MODEL_COLORS.get(model, None),
            label=f"{MODEL_NAMES[model]} (AP = {mean_ap:.3f} ± {std_ap:.3f})"
        )

    prevalence = np.mean(np.array(all_y_true))

    plt.hlines(
        prevalence,
        xmin=0,
        xmax=1,
        linestyle="--",
        linewidth=1.2,
        color="gray",
        label=f"Prevalence = {prevalence:.3f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Mean Precision–Recall Curves of All Models")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)

    plt.legend(loc="lower left", fontsize=7.5, frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "fig_all_models_with_pga_amformer_mean_pr.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png_path}")


# =========================
# 主函数
# =========================

if __name__ == "__main__":
    plot_all_models_mean_roc()
    plot_all_models_mean_pr()