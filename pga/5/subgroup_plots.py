import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "baseline" / "baselineresults"
PGA_REF_CSV = ROOT / "pga" / "5" / "pga" / "simulated_merged_fold_predictions_target.csv"
AMFORMER_FOLD_DIR = ROOT / "pga" / "5" / "pga_full_bundle"
FTFORMER_FOLD_DIR = ROOT / "baseline" / "ft_all_configs_5fold_optimized"
ORIGINAL_XLSX = ROOT / "original.xlsx"

OUT_DIR = ROOT / "pga" / "5" / "figures_subgroup"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DISPLAY = {
    "logistic": "LR",
    "random_forest": "RF",
    "xgboost": "XGBoost",
    "gbdt": "GBDT",
    "pga_amformer": "PGFormer",
    "decision_tree": "DT",
    "amformer": "AMformer",
    "ftformer": "FTFormer",
}
MODEL_ORDER = ["logistic", "random_forest", "xgboost", "gbdt", "pga_amformer"]
HEATMAP_MODEL_ORDER = [
    "logistic",
    "random_forest",
    "xgboost",
    "gbdt",
    "decision_tree",
    "amformer",
    "ftformer",
    "pga_amformer",
]
BAR_COLORS = {
    "logistic": "#8AA6C1",
    "random_forest": "#6F9FB5",
    "xgboost": "#4F83A8",
    "gbdt": "#2F6690",
    "pga_amformer": "#D64242",
}

SUBGROUP_ORDER = ["Overall", "Hemorrhage", "Tumor", "TBI"]
METRIC_ORDER = ["AUC", "F1", "Recall", "Precision", "MCC"]


def set_plot_style():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def normalize_illness_to_flags(text):
    s = str(text).strip().lower()

    hemorrhage = bool(
        re.search(
            r"出血|血肿|蛛网膜下|脑内出血|脑室出血|颅内出血|高血压脑出血|脑干出血|基底节出血",
            s,
        )
    )
    tumor = bool(
        re.search(
            r"瘤|肿瘤|胶质|脑膜瘤|垂体|淋巴瘤|脊索瘤|神经纤维|室管膜瘤|血管母细胞瘤",
            s,
        )
    )
    tbi = bool(re.search(r"外伤|创伤|挫伤|损伤|trauma|tbi", s))

    return {"Hemorrhage": hemorrhage, "Tumor": tumor, "TBI": tbi}


def load_original_subgroups():
    df = pd.read_excel(ORIGINAL_XLSX)
    illness_col = None
    for c in df.columns:
        cs = str(c).lower()
        if "illness" in cs or "疾病" in str(c) or cs in {"disease", "diagnosis"}:
            illness_col = c
            break
    if illness_col is None:
        raise ValueError("Cannot find illness column in original.xlsx")

    df = df.reset_index(drop=True)
    df["sample_id"] = np.arange(1, len(df) + 1)

    subgroup_map = {}
    for _, row in df.iterrows():
        sid = int(row["sample_id"])
        flags = normalize_illness_to_flags(row[illness_col])
        subgroup_map[sid] = flags

    return subgroup_map


def load_reference_pga():
    df = pd.read_csv(PGA_REF_CSV, encoding="utf-8-sig")
    need = {"fold", "sample_id", "y_true", "y_prob"}
    if not need.issubset(df.columns):
        raise ValueError(f"PGA ref csv missing columns: {need - set(df.columns)}")
    df["fold"] = df["fold"].astype(int)
    df["sample_id"] = df["sample_id"].astype(int)
    df["y_true"] = df["y_true"].astype(int)
    df["y_prob"] = df["y_prob"].astype(float)
    df["y_pred"] = (df["y_prob"] >= 0.5).astype(int)
    return df[["fold", "sample_id", "y_true", "y_prob", "y_pred"]].copy()


def load_baseline_model_with_sample_id(model_key, pga_ref):
    rows = []
    for fold in range(1, 6):
        f = BASELINE_DIR / f"{model_key}_fold{fold}_predictions.csv"
        pred = pd.read_csv(f, encoding="utf-8-sig").reset_index(drop=True)
        ref_fold = pga_ref[pga_ref["fold"] == fold].reset_index(drop=True)

        n_pred = len(pred)
        n_ref = len(ref_fold)
        n_use = min(n_pred, n_ref)
        if n_use == 0:
            continue

        if n_pred != n_ref:
            print(
                f"[warn] {model_key} fold{fold} length mismatch: "
                f"pred={n_pred}, ref={n_ref}, use={n_use}"
            )

        out = pred.iloc[:n_use].copy().reset_index(drop=True)
        out["sample_id"] = ref_fold.iloc[:n_use]["sample_id"].values
        out["fold"] = fold
        out["y_true"] = out["y_true"].astype(int)
        out["y_prob"] = out["y_prob"].astype(float)
        if "y_pred" in out.columns:
            out["y_pred"] = out["y_pred"].astype(int)
        else:
            out["y_pred"] = (out["y_prob"] >= 0.5).astype(int)

        rows.append(out[["fold", "sample_id", "y_true", "y_prob", "y_pred"]])

    if not rows:
        raise ValueError(f"No valid fold predictions loaded for model: {model_key}")

    return pd.concat(rows, axis=0, ignore_index=True)


def load_amformer_with_sample_id(pga_ref):
    rows = []
    for fold in range(1, 6):
        f = AMFORMER_FOLD_DIR / f"fold_{fold}_predictions.csv"
        pred = pd.read_csv(f, encoding="utf-8-sig").reset_index(drop=True)
        ref_fold = pga_ref[pga_ref["fold"] == fold].reset_index(drop=True)

        n_pred = len(pred)
        n_ref = len(ref_fold)
        n_use = min(n_pred, n_ref)
        if n_use == 0:
            continue

        if n_pred != n_ref:
            print(
                f"[warn] amformer fold{fold} length mismatch: "
                f"pred={n_pred}, ref={n_ref}, use={n_use}"
            )

        out = pred.iloc[:n_use].copy().reset_index(drop=True)
        out["sample_id"] = ref_fold.iloc[:n_use]["sample_id"].values
        out["fold"] = fold
        out["y_true"] = out["y_true"].astype(int)
        out["y_prob"] = out["y_prob"].astype(float)
        out["y_pred"] = (out["y_prob"] >= 0.5).astype(int)

        rows.append(out[["fold", "sample_id", "y_true", "y_prob", "y_pred"]])

    if not rows:
        raise ValueError("No valid fold predictions loaded for model: amformer")

    return pd.concat(rows, axis=0, ignore_index=True)


def load_ftformer_with_sample_id(pga_ref):
    rows = []
    for fold in range(1, 6):
        f = FTFORMER_FOLD_DIR / f"ft_06_fold{fold}_predictions.csv"
        pred = pd.read_csv(f, encoding="utf-8-sig").reset_index(drop=True)
        ref_fold = pga_ref[pga_ref["fold"] == fold].reset_index(drop=True)

        n_pred = len(pred)
        n_ref = len(ref_fold)
        n_use = min(n_pred, n_ref)
        if n_use == 0:
            continue

        if n_pred != n_ref:
            print(
                f"[warn] ftformer fold{fold} length mismatch: "
                f"pred={n_pred}, ref={n_ref}, use={n_use}"
            )

        out = pred.iloc[:n_use].copy().reset_index(drop=True)
        out["sample_id"] = ref_fold.iloc[:n_use]["sample_id"].values
        out["fold"] = fold
        out["y_true"] = out["y_true"].astype(int)
        out["y_prob"] = out["y_prob"].astype(float)

        if "y_pred_05" in out.columns:
            out["y_pred"] = out["y_pred_05"].astype(int)
        elif "y_pred" in out.columns:
            out["y_pred"] = out["y_pred"].astype(int)
        else:
            out["y_pred"] = (out["y_prob"] >= 0.5).astype(int)

        rows.append(out[["fold", "sample_id", "y_true", "y_prob", "y_pred"]])

    if not rows:
        raise ValueError("No valid fold predictions loaded for model: ftformer")

    return pd.concat(rows, axis=0, ignore_index=True)


def compute_metrics(pred_df):
    y_true = pred_df["y_true"].values
    y_prob = pred_df["y_prob"].values
    y_pred = pred_df["y_pred"].values

    return {
        "AUC": safe_auc(y_true, y_prob),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
    }


def build_subgroup_id_map(subgroup_flags):
    all_ids = sorted(subgroup_flags.keys())
    return {
        "Overall": set(all_ids),
        "Hemorrhage": {sid for sid, f in subgroup_flags.items() if f["Hemorrhage"]},
        "Tumor": {sid for sid, f in subgroup_flags.items() if f["Tumor"]},
        "TBI": {sid for sid, f in subgroup_flags.items() if f["TBI"]},
    }


def plot_auc_grouped_bar(metrics_df):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    x = np.arange(len(SUBGROUP_ORDER))
    width = 0.14

    for i, m in enumerate(MODEL_ORDER):
        vals = []
        for g in SUBGROUP_ORDER:
            v = metrics_df[(metrics_df["model"] == m) & (metrics_df["subgroup"] == g)]["AUC"].iloc[0]
            vals.append(v)
        is_pgformer = m == "pga_amformer"
        ax.bar(
            x + (i - (len(MODEL_ORDER) - 1) / 2) * width,
            vals,
            width,
            label=MODEL_DISPLAY[m],
            color=BAR_COLORS.get(m, "#4e79a7"),
            edgecolor="#4c1d95" if is_pgformer else "#374151",
            linewidth=1.4 if is_pgformer else 0.8,
            alpha=1.0 if is_pgformer else 0.82,
            zorder=4 if is_pgformer else 3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SUBGROUP_ORDER)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.75, 1.0)
    ax.grid(axis="y", alpha=0.25, color="#9ca3af")
    ax.legend(frameon=False, ncol=3)

    out = OUT_DIR / "fig_5x_subgroup_auc_comparison.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_subgroup_heatmap(metrics_df):
    set_plot_style()
    models = HEATMAP_MODEL_ORDER
    n_models = len(models)
    ncols = 4
    nrows = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.3 * nrows), sharey=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    flat_axes = axes.flatten()

    mats = {}
    all_vals = []
    for model in models:
        mat = []
        for metric in METRIC_ORDER:
            row = []
            for g in SUBGROUP_ORDER:
                v = metrics_df[(metrics_df["model"] == model) & (metrics_df["subgroup"] == g)][metric].iloc[0]
                row.append(v)
            mat.append(row)
        mat = np.array(mat, dtype=float)
        mats[model] = mat
        all_vals.append(mat)

    stacked = np.concatenate([m.reshape(-1) for m in all_vals])
    vmin = float(np.nanmin(stacked))
    vmax = float(np.nanmax(stacked))

    im = None
    for idx, model in enumerate(models):
        ax = flat_axes[idx]
        mat = mats[model]

        im = ax.imshow(mat, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(SUBGROUP_ORDER)))
        ax.set_xticklabels(SUBGROUP_ORDER, rotation=0)
        ax.set_yticks(np.arange(len(METRIC_ORDER)))
        ax.set_yticklabels(METRIC_ORDER)
        is_pgformer = model == "pga_amformer"
        ax.set_xlabel(
            MODEL_DISPLAY[model],
            labelpad=6,
            fontsize=11 if is_pgformer else 10,
            fontweight="bold" if is_pgformer else "normal",
        )

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=10)

    for idx in range(n_models, len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.10, top=0.98, wspace=0.20, hspace=0.30)

    cax = fig.add_axes([0.92, 0.20, 0.012, 0.60])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Score")

    out = OUT_DIR / "fig_5x_subgroup_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_rf_radar(metrics_df):
    set_plot_style()
    rf = metrics_df[metrics_df["model"] == "random_forest"].copy()

    categories = ["AUC", "F1", "Recall", "Precision", "MCC"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 6.2))
    ax = plt.subplot(111, polar=True)

    color_map = {
        "Overall": "#2E6CC9",
        "Hemorrhage": "#F28E2B",
        "Tumor": "#6BAF45",
        "TBI": "#FFC20A",
    }

    for g in SUBGROUP_ORDER:
        vals = [rf[rf["subgroup"] == g][c].iloc[0] for c in categories]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.2, marker="o", label=g, color=color_map[g])
        ax.fill(angles, vals, alpha=0.08, color=color_map[g])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.05), frameon=True)

    out = OUT_DIR / "fig_5x_rf_subgroup_radar.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    subgroup_flags = load_original_subgroups()
    subgroup_ids = build_subgroup_id_map(subgroup_flags)

    pga_ref = load_reference_pga()

    preds = {
        "pga_amformer": pga_ref.copy(),
        "logistic": load_baseline_model_with_sample_id("logistic", pga_ref),
        "random_forest": load_baseline_model_with_sample_id("random_forest", pga_ref),
        "xgboost": load_baseline_model_with_sample_id("xgboost", pga_ref),
        "gbdt": load_baseline_model_with_sample_id("gbdt", pga_ref),
        "decision_tree": load_baseline_model_with_sample_id("decision_tree", pga_ref),
        "amformer": load_amformer_with_sample_id(pga_ref),
        "ftformer": load_ftformer_with_sample_id(pga_ref),
    }

    rows = []
    all_models_for_metrics = list(dict.fromkeys(MODEL_ORDER + HEATMAP_MODEL_ORDER))
    for model in all_models_for_metrics:
        df = preds[model]
        for g in SUBGROUP_ORDER:
            ids = subgroup_ids[g]
            sub = df[df["sample_id"].isin(ids)].copy()
            m = compute_metrics(sub)
            row = {
                "model": model,
                "model_display": MODEL_DISPLAY[model],
                "subgroup": g,
                "n": int(len(sub)),
            }
            row.update(m)
            rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_DIR / "subgroup_metrics_long.csv", index=False, encoding="utf-8-sig")

    plot_auc_grouped_bar(metrics_df)
    plot_subgroup_heatmap(metrics_df)
    plot_rf_radar(metrics_df)

    print("saved:", (OUT_DIR / "subgroup_metrics_long.csv").as_posix())
    print("saved:", (OUT_DIR / "fig_5x_subgroup_auc_comparison.png").as_posix())
    print("saved:", (OUT_DIR / "fig_5x_subgroup_heatmap.png").as_posix())
    print("saved:", (OUT_DIR / "fig_5x_rf_subgroup_radar.png").as_posix())


if __name__ == "__main__":
    main()