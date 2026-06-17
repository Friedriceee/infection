# plot_simulated_figures_target_style.py
# -*- coding: utf-8 -*-
"""
说明：
1. 本脚本读取 simulated_*_target 文件并保存图片，不会修改任何数据文件。
2. 如果找不到 *_target 文件，会自动回退到原 simulated_* 文件。
3. 图中统一使用 Arial 字体、英文模型名、英文小写指标名。
4. 所有图片保存到 figures_simulated/，不会 plt.show() 直接弹图。
"""

import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# =========================================================
# 0. 路径配置
# =========================================================

DATA_DIR = "."
SAVE_DIR = "figures_simulated"
os.makedirs(SAVE_DIR, exist_ok=True)

# 优先读取现在这批 target 文件；如果不存在，自动回退到原文件。
def choose_file(target_name, fallback_name):
    target_path = os.path.join(DATA_DIR, target_name)
    fallback_path = os.path.join(DATA_DIR, fallback_name)
    if os.path.exists(target_path):
        return target_path
    return fallback_path

METRICS_CSV = choose_file("simulated_metrics_target.csv", "simulated_metrics.csv")
ALL_FOLDS_CSV = choose_file("simulated_all_folds_target.csv", "simulated_all_folds.csv")
SUMMARY_JSON = choose_file("simulated_summary_target.json", "simulated_summary.json")
PRED_CSV = choose_file("simulated_merged_fold_predictions_target.csv", "simulated_merged_fold_predictions.csv")

# =========================================================
# 1. 论文图全局样式
# =========================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.linewidth"] = 1.1
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9

# 温和论文配色，不指定过刺眼的颜色。
COLORS = {
    "auc": "#4C72B0",
    "accuracy": "#55A868",
    "precision": "#C44E52",
    "recall": "#8172B2",
    "f1": "#CCB974",
    "ap": "#64B5CD",
    "mcc": "#8C8C8C",
    "random": "#999999",
}

MODEL_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

# =========================================================
# 2. 通用函数：读文件、列名兼容、模型名清洗、保存图片
# =========================================================

COLUMN_ALIASES = {
    "模型版本": "model_version",
    "model": "model_version",
    "model_version": "model_version",
    "先验注意力": "prior_attention",
    "prior_attention": "prior_attention",
    "attention": "prior_attention",
    "先验算术": "prior_arithmetic",
    "prior_arithmetic": "prior_arithmetic",
    "arithmetic": "prior_arithmetic",
    "fold": "fold",
    "sample_id": "sample_id",
    "y_true": "y_true",
    "label": "y_true",
    "target": "y_true",
    "y_prob": "y_prob",
    "prob": "y_prob",
    "score": "y_prob",
    "y_pred": "y_pred",
    "pred": "y_pred",
    "threshold": "threshold",
    "accuracy": "accuracy",
    "acc": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "sensitivity": "recall",
    "f1": "f1",
    "f1_score": "f1",
    "auc": "auc",
    "roc_auc": "auc",
    "ap": "ap",
    "average_precision": "ap",
    "mcc": "mcc",
    "data_type": "data_type",
    "note": "note",
}

MODEL_LABEL_MAP = {
    "原始 AMFormer": "AMFormer",
    "Original AMFormer": "AMFormer",
    "AMFormer": "AMFormer",
    "+PGA-Attention": "PGA-Attention",
    "+PGA‑Attention": "PGA-Attention",
    "PGA-Attention": "PGA-Attention",
    "+PGA-Arithmetic": "PGA-Arithmetic",
    "+PGA‑Arithmetic": "PGA-Arithmetic",
    "PGA-Arithmetic": "PGA-Arithmetic",
    "完整 PGA-AMFormer": "PGA-AMFormer",
    "完整 PGA‑AMFormer": "PGA-AMFormer",
    "Full PGA-AMFormer": "PGA-AMFormer",
    "PGA-AMFormer": "PGA-AMFormer",
}

# 用户要求横坐标直接写英文，不加中文。
# 如果只想显示两个名字，可以把下面两个变量改成 True / False。
KEEP_ONLY_AMFORMER_AND_FULL = False
SHOW_SHORT_TWO_MODEL_LABELS = False


def _clean_one_col_name(col):
    col = str(col).replace("\ufeff", "").strip()
    col_lower = col.lower().strip()
    return COLUMN_ALIASES.get(col, COLUMN_ALIASES.get(col_lower, col_lower))


def read_csv_clean(path):
    """读取 CSV，并把列名统一成小写英文标准名。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [_clean_one_col_name(c) for c in df.columns]
    if "model_version" in df.columns:
        df["model_label"] = df["model_version"].apply(format_model_label)
    return df


def read_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_cols(df, cols, source_name="DataFrame"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{source_name} missing required columns: {missing}\n"
            f"Current columns: {df.columns.tolist()}"
        )


def format_model_label(name):
    name = str(name).strip()
    return MODEL_LABEL_MAP.get(name, name.replace("+", "").replace("完整 ", "").replace("原始 ", ""))


def filter_models_for_plot(df):
    """
    默认保留四个模型：AMFormer / PGA-Attention / PGA-Arithmetic / PGA-AMFormer。
    如果 KEEP_ONLY_AMFORMER_AND_FULL=True，则只保留 AMFormer 和 PGA-AMFormer。
    """
    if not KEEP_ONLY_AMFORMER_AND_FULL:
        return df.copy()
    keep = ["AMFormer", "PGA-AMFormer"]
    return df[df["model_label"].isin(keep)].copy()


def get_xlabels(df):
    labels = df["model_label"].tolist()
    if SHOW_SHORT_TWO_MODEL_LABELS:
        labels = ["AMFormer" if x == "AMFormer" else "PGA-AMFormer" for x in labels]
    return labels


def save_fig(filename):
    path = os.path.join(SAVE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def clean_filename(name):
    name = str(format_model_label(name))
    name = name.replace("+", "plus")
    name = re.sub(r"[\\/:*?\"<>|\s]+", "_", name)
    return name.strip("_")


def get_full_model_name(df):
    require_cols(df, ["model_version"], "model dataframe")
    if "model_label" not in df.columns:
        df["model_label"] = df["model_version"].apply(format_model_label)
    full = df[df["model_label"] == "PGA-AMFormer"]
    if len(full) > 0:
        return full["model_version"].iloc[0]
    return df["model_version"].iloc[-1]


def flag_to_int(x):
    return 1 if str(x).strip() in ["√", "1", "True", "true", "yes", "是", "Y", "y"] else 0


def metric_label(metric):
    """用户要求 accuracy / recall / precision 等显示为小写。"""
    return metric.lower()


def get_metric_values_from_summary(summary, metrics):
    means, stds, labels = [], [], []
    for m in metrics:
        if m in summary:
            item = summary[m]
            if isinstance(item, dict):
                means.append(item.get("mean", np.nan))
                stds.append(item.get("std", 0))
            else:
                means.append(item)
                stds.append(0)
            labels.append(metric_label(m))
    return labels, means, stds


def annotate_bars(ax, bars, dy=0.015):
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

# =========================================================
# 3. 消融实验图
# =========================================================


def plot_ablation_grouped_bar():
    df = read_csv_clean(METRICS_CSV)
    require_cols(df, ["model_version", "auc", "precision", "recall", "f1", "ap"], "metrics csv")
    df = filter_models_for_plot(df)

    metrics = ["auc", "precision", "recall", "f1", "ap"]
    x = np.arange(len(df))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, metric in enumerate(metrics):
        bars = ax.bar(
            x + (i - 2) * width,
            df[metric],
            width,
            label=metric_label(metric),
            color=COLORS[metric],
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(get_xlabels(df), rotation=0, ha="center")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Ablation study of PGA-AMFormer")
    ax.legend(ncol=5, frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig("fig_01_ablation_grouped_bar.png")


def plot_ablation_auc_recall_ap():
    df = read_csv_clean(METRICS_CSV)
    require_cols(df, ["model_version", "auc", "recall", "ap"], "metrics csv")
    df = filter_models_for_plot(df)

    metrics = ["auc", "recall", "ap"]
    x = np.arange(len(df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.5, 5))

    for i, metric in enumerate(metrics):
        bars = ax.bar(
            x + (i - 1) * width,
            df[metric],
            width,
            label=metric_label(metric),
            color=COLORS[metric],
            edgecolor="white",
            linewidth=0.6,
        )
        annotate_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(get_xlabels(df), rotation=0, ha="center")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Key metrics in ablation study")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig("fig_02_ablation_auc_recall_ap.png")


def plot_ablation_line_trend():
    df = read_csv_clean(METRICS_CSV)
    require_cols(df, ["model_version", "auc", "precision", "recall", "f1", "ap"], "metrics csv")
    df = filter_models_for_plot(df)

    metrics = ["auc", "precision", "recall", "f1", "ap"]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    for metric in metrics:
        ax.plot(
            get_xlabels(df),
            df[metric],
            marker="o",
            linewidth=2,
            markersize=5,
            label=metric_label(metric),
            color=COLORS[metric],
        )

    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Performance trend across model variants")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig("fig_03_ablation_line_trend.png")

# =========================================================
# 4. 五折结果图
# =========================================================


def plot_fold_metrics_bar():
    df = read_csv_clean(ALL_FOLDS_CSV)
    require_cols(df, ["model_version", "fold", "auc", "recall", "f1", "ap"], "all folds csv")

    target_model = get_full_model_name(df)
    full_df = df[df["model_version"] == target_model].copy()

    metrics = ["auc", "recall", "f1", "ap"]
    x = np.arange(len(full_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8.5, 5))

    for i, metric in enumerate(metrics):
        ax.bar(
            x + (i - 1.5) * width,
            full_df[metric],
            width,
            label=metric_label(metric),
            color=COLORS[metric],
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {int(f)}" for f in full_df["fold"]])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Five-fold performance of PGA-AMFormer")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig("fig_04_five_fold_metrics_bar.png")


def plot_fold_metric_lines():
    df = read_csv_clean(ALL_FOLDS_CSV)
    require_cols(df, ["model_version", "fold", "auc", "precision", "recall", "f1", "ap"], "all folds csv")

    target_model = get_full_model_name(df)
    full_df = df[df["model_version"] == target_model].copy()

    metrics = ["auc", "precision", "recall", "f1", "ap"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for metric in metrics:
        ax.plot(
            full_df["fold"],
            full_df[metric],
            marker="o",
            linewidth=2,
            markersize=5,
            label=metric_label(metric),
            color=COLORS[metric],
        )

    ax.set_xticks(full_df["fold"])
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("fold")
    ax.set_ylabel("score")
    ax.set_title("Metric variation across five folds")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig("fig_05_five_fold_metric_lines.png")


def plot_fold_boxplot():
    df = read_csv_clean(ALL_FOLDS_CSV)
    require_cols(df, ["model_version", "auc", "precision", "recall", "f1", "ap"], "all folds csv")

    target_model = get_full_model_name(df)
    full_df = df[df["model_version"] == target_model].copy()

    metrics = ["auc", "precision", "recall", "f1", "ap"]
    data = [full_df[m].values for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 5))
    box = ax.boxplot(
        data,
        labels=[metric_label(m) for m in metrics],
        showmeans=True,
        patch_artist=True,
    )

    for patch, metric in zip(box["boxes"], metrics):
        patch.set_facecolor(COLORS[metric])
        patch.set_alpha(0.55)

    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Five-fold metric distribution of PGA-AMFormer")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig("fig_06_five_fold_boxplot.png")

# =========================================================
# 5. Summary 总体指标图
# =========================================================


def plot_summary_bar():
    if not os.path.exists(SUMMARY_JSON):
        print(f"[Skip] {SUMMARY_JSON} not found")
        return

    summary = read_json(SUMMARY_JSON)
    metrics = ["accuracy", "precision", "recall", "f1", "auc", "mcc", "ap"]
    labels, means, stds = get_metric_values_from_summary(summary, metrics)

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = [COLORS.get(m, "#8C8C8C") for m in metrics if m in summary]
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=bar_colors, edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Overall performance summary")
    annotate_bars(ax, bars)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig("fig_07_summary_bar.png")


def plot_summary_radar():
    if not os.path.exists(SUMMARY_JSON):
        print(f"[Skip] {SUMMARY_JSON} not found")
        return

    summary = read_json(SUMMARY_JSON)
    metrics = ["accuracy", "precision", "recall", "f1", "auc", "mcc", "ap"]
    labels, values, _ = get_metric_values_from_summary(summary, metrics)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values = values + values[:1]
    angles = angles + angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, color=COLORS["auc"])
    ax.fill(angles, values, alpha=0.18, color=COLORS["auc"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Overall performance radar chart")

    save_fig("fig_08_summary_radar.png")

# =========================================================
# 6. ROC 曲线
# =========================================================


def plot_roc_by_model():
    df = read_csv_clean(PRED_CSV)
    require_cols(df, ["model_version", "y_true", "y_prob"], "predictions csv")
    df = filter_models_for_plot(df)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for idx, (model_name, sub_df) in enumerate(df.groupby("model_label", sort=False)):
        y_true = sub_df["y_true"].values
        y_prob = sub_df["y_prob"].values
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} auc={roc_auc:.3f}",
            color=MODEL_COLORS[idx % len(MODEL_COLORS)],
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color=COLORS["random"], label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves of model variants")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig("fig_09_roc_by_model.png")


def plot_roc_by_fold_for_full_model():
    df = read_csv_clean(PRED_CSV)
    require_cols(df, ["model_version", "fold", "y_true", "y_prob"], "predictions csv")

    target_model = get_full_model_name(df)
    full_df = df[df["model_version"] == target_model].copy()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    aucs = []

    for idx, (fold, sub_df) in enumerate(full_df.groupby("fold")):
        y_true = sub_df["y_true"].values
        y_prob = sub_df["y_prob"].values
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)
        ax.plot(fpr, tpr, linewidth=1.8, label=f"fold {int(fold)} auc={fold_auc:.3f}")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color=COLORS["random"], label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"Five-fold ROC curves of PGA-AMFormer\nmean auc={np.mean(aucs):.3f}±{np.std(aucs):.3f}")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig("fig_10_roc_by_fold_full_model.png")

# =========================================================
# 7. PR / AP 曲线
# =========================================================


def plot_pr_by_model():
    df = read_csv_clean(PRED_CSV)
    require_cols(df, ["model_version", "y_true", "y_prob"], "predictions csv")
    df = filter_models_for_plot(df)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for idx, (model_name, sub_df) in enumerate(df.groupby("model_label", sort=False)):
        y_true = sub_df["y_true"].values
        y_prob = sub_df["y_prob"].values
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{model_name} ap={ap:.3f}",
            color=MODEL_COLORS[idx % len(MODEL_COLORS)],
        )

    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall curves of model variants")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig("fig_11_pr_by_model.png")


def plot_pr_by_fold_for_full_model():
    df = read_csv_clean(PRED_CSV)
    require_cols(df, ["model_version", "fold", "y_true", "y_prob"], "predictions csv")

    target_model = get_full_model_name(df)
    full_df = df[df["model_version"] == target_model].copy()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    aps = []

    for fold, sub_df in full_df.groupby("fold"):
        y_true = sub_df["y_true"].values
        y_prob = sub_df["y_prob"].values
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        aps.append(ap)
        ax.plot(recall, precision, linewidth=1.8, label=f"fold {int(fold)} ap={ap:.3f}")

    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"Five-fold PR curves of PGA-AMFormer\nmean ap={np.mean(aps):.3f}±{np.std(aps):.3f}")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig("fig_12_pr_by_fold_full_model.png")

# =========================================================
# 8. 混淆矩阵
# =========================================================


def _draw_confusion_matrix(cm, title, save_name):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["negative", "positive"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("true label")
    ax.set_xlabel("predicted label")
    save_fig(save_name)


def plot_confusion_matrix_full_model():
    df = read_csv_clean(PRED_CSV)
    require_cols(df, ["model_version", "y_true", "y_prob"], "predictions csv")

    target_model = get_full_model_name(df)
    full_df = df[df["model_version"] == target_model].copy()

    y_true = full_df["y_true"].values.astype(int)
    if "y_pred" in full_df.columns:
        y_pred = full_df["y_pred"].values.astype(int)
    else:
        y_pred = (full_df["y_prob"].values >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    _draw_confusion_matrix(cm, "Confusion matrix of PGA-AMFormer", "fig_13_confusion_matrix_full_model.png")


def plot_confusion_matrix_by_model():
    df = read_csv_clean(PRED_CSV)
    require_cols(df, ["model_version", "y_true", "y_prob"], "predictions csv")
    df = filter_models_for_plot(df)

    for model_name, sub_df in df.groupby("model_label", sort=False):
        y_true = sub_df["y_true"].values.astype(int)
        if "y_pred" in sub_df.columns:
            y_pred = sub_df["y_pred"].values.astype(int)
        else:
            y_pred = (sub_df["y_prob"].values >= 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        safe_name = clean_filename(model_name)
        _draw_confusion_matrix(cm, f"Confusion matrix of {model_name}", f"fig_14_confusion_matrix_{safe_name}.png")

# =========================================================
# 9. 阈值-指标曲线
# =========================================================


def plot_threshold_metric_curve_full_model():
    df = read_csv_clean(PRED_CSV)
    require_cols(df, ["model_version", "y_true", "y_prob"], "predictions csv")

    target_model = get_full_model_name(df)
    full_df = df[df["model_version"] == target_model].copy()

    y_true = full_df["y_true"].values.astype(int)
    y_prob = full_df["y_prob"].values
    thresholds = np.linspace(0.01, 0.99, 99)

    precision_list, recall_list, f1_list, mcc_list = [], [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precision_list.append(precision_score(y_true, y_pred, zero_division=0))
        recall_list.append(recall_score(y_true, y_pred, zero_division=0))
        f1_list.append(f1_score(y_true, y_pred, zero_division=0))
        mcc_list.append(matthews_corrcoef(y_true, y_pred))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, precision_list, linewidth=2, label="precision", color=COLORS["precision"])
    ax.plot(thresholds, recall_list, linewidth=2, label="recall", color=COLORS["recall"])
    ax.plot(thresholds, f1_list, linewidth=2, label="f1", color=COLORS["f1"])
    ax.plot(thresholds, mcc_list, linewidth=2, label="mcc", color=COLORS["mcc"])

    ax.set_xlabel("decision threshold")
    ax.set_ylabel("score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Threshold-metric curve of PGA-AMFormer")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig("fig_15_threshold_metric_curve_full_model.png")

# =========================================================
# 10. 先验模块贡献热力图
# =========================================================


def plot_prior_module_heatmap():
    df = read_csv_clean(METRICS_CSV)
    require_cols(df, ["prior_attention", "prior_arithmetic", "auc"], "metrics csv")

    df["att_flag"] = df["prior_attention"].apply(flag_to_int)
    df["arith_flag"] = df["prior_arithmetic"].apply(flag_to_int)

    heat = np.full((2, 2), np.nan)
    for _, row in df.iterrows():
        heat[int(row["att_flag"]), int(row["arith_flag"])] = row["auc"]

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(heat, interpolation="nearest", cmap="Blues", vmin=np.nanmin(heat), vmax=np.nanmax(heat))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="auc")
    ax.set_title("AUC under prior module combinations")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["arithmetic off", "arithmetic on"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["attention off", "attention on"])

    for i in range(2):
        for j in range(2):
            text = "NA" if np.isnan(heat[i, j]) else f"{heat[i, j]:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=12)

    ax.set_xlabel("prior arithmetic")
    ax.set_ylabel("prior attention")
    save_fig("fig_16_prior_module_auc_heatmap.png")

# =========================================================
# 11. 自动生成所有图
# =========================================================


def main():
    print("Using files:")
    print("  METRICS_CSV   =", METRICS_CSV)
    print("  ALL_FOLDS_CSV =", ALL_FOLDS_CSV)
    print("  SUMMARY_JSON  =", SUMMARY_JSON)
    print("  PRED_CSV      =", PRED_CSV)

    plot_ablation_grouped_bar()
    plot_ablation_auc_recall_ap()
    plot_ablation_line_trend()

    plot_fold_metrics_bar()
    plot_fold_metric_lines()
    plot_fold_boxplot()

    plot_summary_bar()
    plot_summary_radar()

    plot_roc_by_model()
    plot_roc_by_fold_for_full_model()

    plot_pr_by_model()
    plot_pr_by_fold_for_full_model()

    plot_confusion_matrix_full_model()
    plot_confusion_matrix_by_model()

    plot_threshold_metric_curve_full_model()
    plot_prior_module_heatmap()


if __name__ == "__main__":
    main()
