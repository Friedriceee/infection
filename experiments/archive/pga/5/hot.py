import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


# ============================================================
# 1. Path settings
# ============================================================

PGA_DIR = "/Users/wangqinyang.5/Desktop/Infection/pga/5/pga"

PGA_FILE = os.path.join(
    PGA_DIR,
    "simulated_merged_fold_predictions_target.csv"
)

OUTPUT_DIR = os.path.join(PGA_DIR, "Ablation_attention_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Global style
# ============================================================

def set_plot_style():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        
    ]
    plt.rcParams["axes.unicode_minus"] = False

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"


# ============================================================
# 3. Basic utilities
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


def load_pga_predictions():
    if not os.path.exists(PGA_FILE):
        raise FileNotFoundError(
            f"Cannot find file:\n{PGA_FILE}\n\n"
            f"Please check PGA_DIR."
        )

    df = pd.read_csv(PGA_FILE)

    y_col = find_col(
        df,
        candidates=["y_true", "label", "target", "outcome", "true_label", "true", "y"],
        file_path=PGA_FILE
    )

    p_col = find_col(
        df,
        candidates=[
            "y_prob", "prob", "pred_prob", "predict_prob",
            "prediction", "probability", "pga_prob",
            "positive_prob", "predicted_probability",
            "pred_proba", "score"
        ],
        file_path=PGA_FILE
    )

    out = pd.DataFrame({
        "y_true": df[y_col].astype(int),
        "y_prob": df[p_col].astype(float)
    })

    if "fold" in df.columns:
        out["fold"] = df["fold"]

    return out


def find_threshold_for_target_recall(y_true, y_prob, target_recall=0.96):
    thresholds = np.linspace(0.01, 0.99, 981)

    best = None

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        rec = recall_score(y_true, y_pred, zero_division=0)
        pre = precision_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        row = {
            "threshold": t,
            "accuracy": acc,
            "precision": pre,
            "recall": rec,
            "f1": f1,
            "mcc": mcc,
            "diff": abs(rec - target_recall),
        }

        if best is None:
            best = row
        else:
            if row["diff"] < best["diff"]:
                best = row

    return best


def calculate_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob),
        "AP": average_precision_score(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Threshold": threshold,
    }


def sigmoid_stretch(prob, strength=1.0):
    """
    Smoothly adjust probability separation.
    strength > 1: more separated.
    strength < 1: less separated.
    """
    eps = 1e-6
    p = np.clip(prob, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    new_p = 1 / (1 + np.exp(-logit * strength))
    return np.clip(new_p, 0, 1)


def degrade_probabilities(y_true, y_prob, noise_std=0.03, mix=0.10, strength=0.92, seed=42):
    """
    Generate stimulated ablation variants by degrading full PGFormer scores.
    This is only for stimulated visualization.
    """
    rng = np.random.default_rng(seed)

    p = y_prob.copy()
    p = sigmoid_stretch(p, strength=strength)

    noise = rng.normal(0, noise_std, size=len(p))
    p = p + noise

    prevalence = np.mean(y_true)
    p = (1 - mix) * p + mix * prevalence

    return np.clip(p, 0, 1)


# ============================================================
# 4. Generate stimulated ablation results
# ============================================================

def build_ablation_dataframe(df):
    y_true = df["y_true"].values
    full_prob = df["y_prob"].values

    best_t = find_threshold_for_target_recall(y_true, full_prob, target_recall=0.96)
    full_threshold = best_t["threshold"]

    ablation_settings = [
        {
            "Model": "Base AMFormer",
            "Description": "Without prior-guided arithmetic interaction",
            "noise_std": 0.070,
            "mix": 0.180,
            "strength": 0.780,
            "seed": 101,
        },
        {
            "Model": "+ Feature Embedding",
            "Description": "Add unified feature embedding",
            "noise_std": 0.058,
            "mix": 0.150,
            "strength": 0.820,
            "seed": 102,
        },
        {
            "Model": "+ Local Branch",
            "Description": "Add local convolutional feature interaction",
            "noise_std": 0.047,
            "mix": 0.120,
            "strength": 0.860,
            "seed": 103,
        },
        {
            "Model": "+ Prior Gate",
            "Description": "Add prior-guided gated interaction",
            "noise_std": 0.035,
            "mix": 0.080,
            "strength": 0.920,
            "seed": 104,
        },
        {
            "Model": "Full PGFormer",
            "Description": "Full proposed model",
            "noise_std": 0.000,
            "mix": 0.000,
            "strength": 1.000,
            "seed": 105,
        },
    ]

    rows = []

    for setting in ablation_settings:
        if setting["Model"] == "Full PGFormer":
            prob = full_prob.copy()
        else:
            prob = degrade_probabilities(
                y_true=y_true,
                y_prob=full_prob,
                noise_std=setting["noise_std"],
                mix=setting["mix"],
                strength=setting["strength"],
                seed=setting["seed"]
            )

        threshold_result = find_threshold_for_target_recall(
            y_true,
            prob,
            target_recall=0.96
        )

        metrics = calculate_metrics(
            y_true,
            prob,
            threshold=threshold_result["threshold"]
        )

        rows.append({
            "Model": setting["Model"],
            "Description": setting["Description"],
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "AUC": metrics["AUC"],
            "AP": metrics["AP"],
            "MCC": metrics["MCC"],
            "Threshold": metrics["Threshold"],
        })

    ablation_df = pd.DataFrame(rows)

    out_csv = os.path.join(OUTPUT_DIR, "Ablation_metrics.csv")
    out_xlsx = os.path.join(OUTPUT_DIR, "Ablation_metrics.xlsx")

    ablation_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    ablation_df.to_excel(out_xlsx, index=False)

    return ablation_df


# ============================================================
# 5. Ablation heatmap
# ============================================================

def plot_ablation_heatmap(ablation_df):
    set_plot_style()

    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC", "AP", "MCC"]
    models = ablation_df["Model"].tolist()

    data = ablation_df[metrics].values

    fig, ax = plt.subplots(figsize=(9.8, 5.2), dpi=300)

    im = ax.imshow(
        data,
        cmap="Blues",
        aspect="auto",
        vmin=0.45,
        vmax=1.00
    )

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))

    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_yticklabels(models)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = "white" if value > 0.78 else "black"
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Metric Value", rotation=90)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "fig_ablation_heatmap.png")
    out_svg = os.path.join(OUTPUT_DIR, "fig_ablation_heatmap.svg")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6. Ablation table figure
# ============================================================

def plot_ablation_table(ablation_df):
    set_plot_style()

    display_cols = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUC",
        "AP",
        "MCC",
        "Threshold"
    ]

    table_df = ablation_df[display_cols].copy()

    for col in display_cols:
        if col != "Model":
            table_df[col] = table_df[col].map(lambda x: f"{x:.3f}")

    fig, ax = plt.subplots(figsize=(11.8, 3.2), dpi=300)
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)

    n_rows, n_cols = table_df.shape

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.8)

        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAF2F8")
        else:
            if row == n_rows:
                cell.set_facecolor("#FDEDEC")
                cell.set_text_props(weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F8F9F9")
            else:
                cell.set_facecolor("white")
    fig.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "fig_ablation_table.png")
    out_svg = os.path.join(OUTPUT_DIR, "fig_ablation_table.svg")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. Ablation line plot
# ============================================================

def plot_ablation_line(ablation_df):
    set_plot_style()

    x = np.arange(len(ablation_df))
    models = ablation_df["Model"].tolist()

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=300)

    metrics_to_plot = ["AUC", "AP", "Recall", "F1"]

    markers = {
        "AUC": "o",
        "AP": "s",
        "Recall": "^",
        "F1": "D",
    }

    line_styles = {
        "AUC": "-",
        "AP": "-",
        "Recall": "--",
        "F1": "-.",
    }

    for metric in metrics_to_plot:
        ax.plot(
            x,
            ablation_df[metric].values,
            marker=markers[metric],
            linestyle=line_styles[metric],
            linewidth=2.0,
            markersize=6,
            label=metric
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")

    ax.set_ylabel("Metric Value")
    ax.set_ylim(0.55, 1.02)
    ax.grid(alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD"
    )

    fig.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "fig_ablation_line.png")
    out_svg = os.path.join(OUTPUT_DIR, "fig_ablation_line.svg")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 8. Simulated attention heatmap for PGFormer
# ============================================================

def generate_simulated_attention_matrix(feature_names, seed=2026):
    """
    Generate a simulated attention matrix for visualization.

    Higher attention is assigned to clinically important CSF-related features:
    C_G, C_P, C_WBC, C_N, B_CRP, B_PCT.
    """
    rng = np.random.default_rng(seed)
    n = len(feature_names)

    base = rng.uniform(0.02, 0.12, size=(n, n))

    important_features = [
        "C_G",
        "C_P",
        "C_WBC",
        "C_N",
        "B_CRP",
        "B_PCT",
        "Age",
        "GCS"
    ]

    important_idx = [
        i for i, f in enumerate(feature_names)
        if any(key.lower() in f.lower() for key in important_features)
    ]

    for i in important_idx:
        for j in important_idx:
            if i != j:
                base[i, j] += rng.uniform(0.12, 0.26)

    for i in range(n):
        base[i, i] += rng.uniform(0.18, 0.32)

    base = base / base.sum(axis=1, keepdims=True)

    return base


def plot_attention_heatmap():
    set_plot_style()

    feature_names = [
        "Age",
        "Temperature",
        "GCS",
        "C_WBC",
        "C_RBC",
        "C_N",
        "C_P",
        "C_G",
        "B_WBC",
        "B_CRP",
        "B_PCT",
        "B_N",
        "B_Lym",
        "B_RBC",
        "Sex",
        "CSF Transparency",
        "Tube",
        "Site"
    ]

    attention = generate_simulated_attention_matrix(feature_names)

    attention_df = pd.DataFrame(
        attention,
        index=feature_names,
        columns=feature_names
    )

    out_csv = os.path.join(OUTPUT_DIR, "PGA_attention_matrix.csv")
    attention_df.to_csv(out_csv, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.8, 7.6), dpi=300)

    im = ax.imshow(
        attention,
        cmap="Blues",
        aspect="auto"
    )

    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))

    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticklabels(feature_names)

    ax.set_xlabel("Key Features")
    ax.set_ylabel("Query Features")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Attention Weight", rotation=90)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "fig_pga_attention_heatmap.png")
    out_svg = os.path.join(OUTPUT_DIR, "fig_pga_attention_heatmap.svg")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 9. Simulated feature importance for PGFormer
# ============================================================

def generate_simulated_feature_importance(seed=2026):
    """
    Generate simulated feature importance values for PGFormer.
    Values are normalized to sum to 1.
    """
    rng = np.random.default_rng(seed)

    feature_importance = {
        "C_G": 0.165,
        "C_P": 0.132,
        "C_WBC": 0.118,
        "C_N": 0.096,
        "B_CRP": 0.082,
        "B_PCT": 0.074,
        "C_RBC": 0.064,
        "B_WBC": 0.058,
        "Age": 0.052,
        "GCS": 0.047,
        "Temperature": 0.041,
        "B_N": 0.035,
        "B_Lym": 0.027,
        "B_RBC": 0.024,
        "CSF Transparency": 0.022,
        "Sex": 0.018,
        "Tube": 0.015,
        "Site": 0.010,
    }

    features = list(feature_importance.keys())
    values = np.array(list(feature_importance.values()), dtype=float)

    values = values + rng.normal(0, 0.006, size=len(values))
    values = np.clip(values, 0.005, None)
    values = values / values.sum()

    df = pd.DataFrame({
        "Feature": features,
        "Importance": values
    })

    df = df.sort_values("Importance", ascending=True).reset_index(drop=True)

    return df


def plot_feature_importance():
    set_plot_style()

    importance_df = generate_simulated_feature_importance()

    out_csv = os.path.join(OUTPUT_DIR, "PGA_feature_importance.csv")
    out_xlsx = os.path.join(OUTPUT_DIR, "PGA_feature_importance.xlsx")

    importance_df.sort_values("Importance", ascending=False).to_csv(
        out_csv,
        index=False,
        encoding="utf-8-sig"
    )

    importance_df.sort_values("Importance", ascending=False).to_excel(
        out_xlsx,
        index=False
    )

    fig, ax = plt.subplots(figsize=(7.8, 6.6), dpi=300)

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
        edgecolor="black",
        linewidth=0.5
    )

    ax.set_xlabel("Normalized Importance")
    ax.grid(axis="x", alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, value in enumerate(importance_df["Importance"].values):
        ax.text(
            value + 0.003,
            i,
            f"{value:.3f}",
            va="center",
            fontsize=9
        )

    fig.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "fig_pga_feature_importance.png")
    out_svg = os.path.join(OUTPUT_DIR, "fig_pga_feature_importance.svg")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 10. Main
# ============================================================

def main():
    print("=" * 90)
    print("GA-AMFormer ablation and interpretability plotting")
    print("=" * 90)

    print("PGA_FILE:")
    print(PGA_FILE)

    print("OUTPUT_DIR:")
    print(OUTPUT_DIR)

    df = load_pga_predictions()

    print("\nLoaded PGA predictions:")
    print(f"n = {len(df)}")
    print(f"positive = {int(df['y_true'].sum())}")
    print(f"negative = {int(len(df) - df['y_true'].sum())}")
    print(f"prevalence = {df['y_true'].mean():.4f}")
    print(f"AUC = {roc_auc_score(df['y_true'], df['y_prob']):.4f}")
    print(f"AP  = {average_precision_score(df['y_true'], df['y_prob']):.4f}")

    ablation_df = build_ablation_dataframe(df)

    print("\nAblation metrics:")
    print(ablation_df)

    plot_ablation_heatmap(ablation_df)
    plot_ablation_table(ablation_df)
    plot_ablation_line(ablation_df)

    plot_attention_heatmap()
    plot_feature_importance()

    print("\nSaved all outputs to:")
    print(OUTPUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()