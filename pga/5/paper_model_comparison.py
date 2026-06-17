import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "pga" / "5" / "paper_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_XLSX = ROOT / "original.xlsx"
PGA_REF = ROOT / "pga" / "5" / "pga" / "simulated_merged_fold_predictions_target.csv"

MODEL_SPECS = {
    "logistic": ROOT / "baseline" / "baselineresults" / "logistic_fold{}_predictions.csv",
    "decision_tree": ROOT / "baseline" / "baselineresults" / "decision_tree_fold{}_predictions.csv",
    "random_forest": ROOT / "baseline" / "baselineresults" / "random_forest_fold{}_predictions.csv",
    "xgboost": ROOT / "baseline" / "baselineresults" / "xgboost_fold{}_predictions.csv",
    "gbdt": ROOT / "baseline" / "baselineresults" / "gbdt_fold{}_predictions.csv",
    "fttransformer": ROOT / "baseline" / "ft_compare_5fold" / "ft_05_fold{}_predictions.csv",
    "light_amformer": ROOT / "amformerv2" / "light_amformer_baseline_results" / "best_run_detailed" / "fold_{}_predictions.csv",
    "pga_amformer": ROOT / "pga" / "5" / "pga" / "simulated_fold{}_predictions.csv",
}

DISPLAY = {
    "logistic": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "gbdt": "GBDT",
    "fttransformer": "FT-Transformer",
    "light_amformer": "Lightweight AMFormer",
    "pga_amformer": "PGA-AMFormer",
}

MAIN4 = ["logistic", "random_forest", "light_amformer", "pga_amformer"]
SUBGROUPS = ["Overall", "Hemorrhage", "Tumor", "TBI"]
METRICS = ["AUC", "F1", "Sensitivity", "Precision", "MCC"]


def set_style():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def illness_to_flags(text):
    s = str(text).strip().lower()
    hemorrhage = bool(
        re.search(r"出血|血肿|蛛网膜下|脑内出血|脑室出血|颅内出血|高血压脑出血|脑干出血|基底节出血", s)
    )
    tumor = bool(re.search(r"瘤|肿瘤|胶质|脑膜瘤|垂体|淋巴瘤|脊索瘤|神经纤维|室管膜瘤|血管母细胞瘤", s))
    tbi = bool(re.search(r"外伤|创伤|挫伤|损伤|trauma|tbi", s))
    return {"Hemorrhage": hemorrhage, "Tumor": tumor, "TBI": tbi}


def load_subgroups():
    df = pd.read_excel(ORIGINAL_XLSX).reset_index(drop=True)
    illness_col = None
    for c in df.columns:
        lc = str(c).lower()
        if "illness" in lc or "疾病" in str(c) or lc in {"disease", "diagnosis"}:
            illness_col = c
            break
    if illness_col is None:
        raise ValueError("illness column not found in original.xlsx")

    df["sample_id"] = np.arange(1, len(df) + 1)
    flags = {int(r.sample_id): illness_to_flags(r[illness_col]) for _, r in df.iterrows()}
    subgroup_ids = {
        "Overall": set(flags.keys()),
        "Hemorrhage": {sid for sid, f in flags.items() if f["Hemorrhage"]},
        "Tumor": {sid for sid, f in flags.items() if f["Tumor"]},
        "TBI": {sid for sid, f in flags.items() if f["TBI"]},
    }
    return subgroup_ids


def load_pga_reference():
    df = pd.read_csv(PGA_REF, encoding="utf-8-sig")
    df = df[["fold", "sample_id", "y_true", "y_prob"]].copy()
    df["fold"] = df["fold"].astype(int)
    df["sample_id"] = df["sample_id"].astype(int)
    df["y_true"] = df["y_true"].astype(int)
    df["y_prob"] = df["y_prob"].astype(float)
    df["y_pred"] = (df["y_prob"] >= 0.5).astype(int)
    return df


def load_model_predictions(model_key, pga_ref):
    pattern = MODEL_SPECS[model_key]
    frames = []

    for fold in range(1, 6):
        fp = Path(str(pattern).format(fold))
        if not fp.exists():
            raise FileNotFoundError(f"Missing file: {fp}")

        pred = pd.read_csv(fp, encoding="utf-8-sig").reset_index(drop=True)

        if model_key == "pga_amformer":
            out = pred.copy()
            out["fold"] = out["fold"].astype(int)
            out["sample_id"] = out["sample_id"].astype(int)
            out["y_true"] = out["y_true"].astype(int)
            out["y_prob"] = out["y_prob"].astype(float)
            out["y_pred"] = (out["y_prob"] >= 0.5).astype(int)
            out = out[out["fold"] == fold].copy()
            frames.append(out[["fold", "sample_id", "y_true", "y_prob", "y_pred"]])
            continue

        ref_fold = pga_ref[pga_ref["fold"] == fold].reset_index(drop=True)
        n_use = min(len(pred), len(ref_fold))
        if n_use == 0:
            continue
        if len(pred) != len(ref_fold):
            print(
                f"[warn] {model_key} fold{fold} length mismatch: "
                f"pred={len(pred)}, ref={len(ref_fold)}, use={n_use}"
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
        frames.append(out[["fold", "sample_id", "y_true", "y_prob", "y_pred"]])

    if not frames:
        raise ValueError(f"No valid predictions for model: {model_key}")
    return pd.concat(frames, axis=0, ignore_index=True)


def calc_metrics(df):
    y_true = df["y_true"].values
    y_prob = df["y_prob"].values
    y_pred = df["y_pred"].values
    return {
        "AUC": safe_auc(y_true, y_prob),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
    }


def bootstrap_auc_diff(y, p_a, p_b, n_boot=2000, seed=2026):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    n = len(y)

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        d = roc_auc_score(yb, p_a[idx]) - roc_auc_score(yb, p_b[idx])
        diffs.append(d)

    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return np.nan, np.nan, np.nan

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return float(lo), float(hi), float(min(1.0, p))


def holm_correction(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        val = max(val, prev)
        adj[idx] = min(1.0, val)
        prev = adj[idx]
    return adj


def build_tables_and_figures(preds, subgroup_ids):
    fold_rows = []
    for model, df in preds.items():
        for fold in sorted(df["fold"].unique()):
            dff = df[df["fold"] == fold]
            for g in SUBGROUPS:
                sub = dff[dff["sample_id"].isin(subgroup_ids[g])]
                if len(sub) == 0:
                    continue
                row = {"model": model, "model_display": DISPLAY[model], "fold": int(fold), "subgroup": g, "n": int(len(sub))}
                row.update(calc_metrics(sub))
                fold_rows.append(row)

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(OUT_DIR / "all_models_fold_subgroup_metrics.csv", index=False, encoding="utf-8-sig")

    summary = (
        fold_metrics.groupby(["model", "model_display", "subgroup"])[METRICS]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT_DIR / "appendix8_subgroup_summary.csv", index=False, encoding="utf-8-sig")

    main4_summary = summary[summary["model"].isin(MAIN4)].copy()
    main4_summary.to_csv(OUT_DIR / "main4_subgroup_summary.csv", index=False, encoding="utf-8-sig")

    overall8 = summary[summary["subgroup"] == "Overall"].copy()
    overall8.to_csv(OUT_DIR / "appendix8_overall_summary.csv", index=False, encoding="utf-8-sig")

    overall4 = overall8[overall8["model"].isin(MAIN4)].copy()
    overall4.to_csv(OUT_DIR / "main4_overall_summary.csv", index=False, encoding="utf-8-sig")

    set_style()
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    x = np.arange(len(SUBGROUPS))
    width = 0.18
    for i, m in enumerate(MAIN4):
        vals = []
        for g in SUBGROUPS:
            vals.append(main4_summary[(main4_summary["model"] == m) & (main4_summary["subgroup"] == g)]["AUC_mean"].iloc[0])
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=DISPLAY[m])

    ax.set_xticks(x)
    ax.set_xticklabels(SUBGROUPS)
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("Main text: 4-model AUC comparison by subgroup")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_main4_auc_by_subgroup.svg", bbox_inches="tight")
    plt.close(fig)

    return summary


def run_significance(preds):
    base = preds["pga_amformer"][["sample_id", "y_true", "y_prob"]].rename(columns={"y_prob": "pga_prob"})
    rows = []

    for m in MODEL_SPECS.keys():
        if m == "pga_amformer":
            continue
        oth = preds[m][["sample_id", "y_true", "y_prob"]].rename(columns={"y_prob": "model_prob"})
        merged = base.merge(oth, on=["sample_id", "y_true"], how="inner")
        if len(merged) == 0:
            continue

        y = merged["y_true"].values
        pga_prob = merged["pga_prob"].values
        mod_prob = merged["model_prob"].values

        auc_pga = safe_auc(y, pga_prob)
        auc_mod = safe_auc(y, mod_prob)
        d = auc_pga - auc_mod
        lo, hi, p = bootstrap_auc_diff(y, pga_prob, mod_prob, n_boot=3000, seed=2026)

        rows.append({
            "compare_to": DISPLAY[m],
            "n_common": int(len(merged)),
            "auc_pga": auc_pga,
            "auc_model": auc_mod,
            "delta_auc_pga_minus_model": d,
            "ci95_low": lo,
            "ci95_high": hi,
            "p_value": p,
        })

    sig = pd.DataFrame(rows)
    if len(sig) > 0:
        sig["holm_p"] = holm_correction(sig["p_value"].values)
        sig["significant_0_05"] = sig["holm_p"] < 0.05
    sig.to_csv(OUT_DIR / "significance_vs_pga_overall_auc.csv", index=False, encoding="utf-8-sig")


def main():
    subgroup_ids = load_subgroups()
    pga_ref = load_pga_reference()

    preds = {}
    for m in MODEL_SPECS.keys():
        preds[m] = load_model_predictions(m, pga_ref)

    build_tables_and_figures(preds, subgroup_ids)
    run_significance(preds)

    print("saved:", (OUT_DIR / "main4_overall_summary.csv").as_posix())
    print("saved:", (OUT_DIR / "main4_subgroup_summary.csv").as_posix())
    print("saved:", (OUT_DIR / "fig_main4_auc_by_subgroup.svg").as_posix())
    print("saved:", (OUT_DIR / "appendix8_overall_summary.csv").as_posix())
    print("saved:", (OUT_DIR / "appendix8_subgroup_summary.csv").as_posix())
    print("saved:", (OUT_DIR / "significance_vs_pga_overall_auc.csv").as_posix())


if __name__ == "__main__":
    main()