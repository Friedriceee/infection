import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

RESULT_DIR = "baselineresults"
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAMES = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "gbdt": "GBDT"
}

def plot_mean_roc_for_models(model_list, output_name, n_folds=5):
    plt.figure(figsize=(7, 6), dpi=300)

    mean_fpr = np.linspace(0, 1, 100)

    for model in model_list:
        tprs = []
        aucs = []

        for fold in range(1, n_folds + 1):
            path = os.path.join(RESULT_DIR, f"{model}_fold{fold}_predictions.csv")
            df = pd.read_csv(path)

            y_true = df["y_true"].values
            y_prob = df["y_prob"].values

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fold_auc = roc_auc_score(y_true, y_prob)
            aucs.append(fold_auc)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        plt.plot(
            mean_fpr,
            mean_tpr,
            linewidth=2,
            label=f"{MODEL_NAMES[model]} (AUC = {mean_auc:.3f} ± {std_auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="gray", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean ROC Curves of Traditional Machine Learning Models")
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_name}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_name}.pdf"), bbox_inches="tight")
    plt.show()


def plot_mean_pr_for_models(model_list, output_name, n_folds=5):
    plt.figure(figsize=(7, 6), dpi=300)

    mean_recall = np.linspace(0, 1, 100)

    for model in model_list:
        precisions = []
        aps = []

        for fold in range(1, n_folds + 1):
            path = os.path.join(RESULT_DIR, f"{model}_fold{fold}_predictions.csv")
            df = pd.read_csv(path)

            y_true = df["y_true"].values
            y_prob = df["y_prob"].values

            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            fold_ap = average_precision_score(y_true, y_prob)
            aps.append(fold_ap)

            interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
            precisions.append(interp_precision)

        mean_precision = np.mean(precisions, axis=0)
        mean_ap = np.mean(aps)
        std_ap = np.std(aps)

        plt.plot(
            mean_recall,
            mean_precision,
            linewidth=2,
            label=f"{MODEL_NAMES[model]} (AP = {mean_ap:.3f} ± {std_ap:.3f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Mean PR Curves of Traditional Machine Learning Models")
    plt.legend(loc="lower left", fontsize=9, frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_name}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_name}.pdf"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    models = ["decision_tree", "random_forest", "xgboost", "gbdt"]
    plot_mean_roc_for_models(models, "fig_tree_models_mean_roc")
    plot_mean_pr_for_models(models, "fig_tree_models_mean_pr")