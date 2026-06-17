"""
plot.py

功能：
统一绘制论文中所有结果图，包括：
1. ROC曲线（baseline + AMFormer）
2. PR曲线
3. 模型性能对比柱状图
4. 混淆矩阵（AMFormer）
5. Loss曲线（如果有）
6. 特征重要性（XGBoost）
7. SHAP图
8. t-SNE可视化

使用前提：
需要存在以下文件：
- baseline_results.pkl
- amformer_results.pkl
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.manifold import TSNE

# =========================
# 1. 加载数据
# =========================
with open("baseline_results.pkl", "rb") as f:
    baseline_results = pickle.load(f)

with open("amformer_results.pkl", "rb") as f:
    amformer_results = pickle.load(f)


# =========================
# 2. ROC曲线
# =========================
def plot_roc():
    plt.figure(figsize=(8,6))

    # baseline模型
    for name, res in baseline_results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")

    # AMFormer
    fpr, tpr, _ = roc_curve(amformer_results["labels"], amformer_results["predictions"])
    plt.plot(fpr, tpr, linewidth=2, label=f"AMFormer (AUC={amformer_results['auc']:.3f})")

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=300)
    plt.show()


# =========================
# 3. PR曲线
# =========================
def plot_pr():
    plt.figure(figsize=(8,6))

    for name, res in baseline_results.items():
        precision, recall, _ = precision_recall_curve(res["y_true"], res["y_prob"])
        plt.plot(recall, precision, label=name)

    precision, recall, _ = precision_recall_curve(
        amformer_results["labels"], amformer_results["predictions"]
    )
    plt.plot(recall, precision, linewidth=2, label="AMFormer")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr_curve.png", dpi=300)
    plt.show()


# =========================
# 4. 模型对比柱状图
# =========================
def plot_metrics():
    models = list(baseline_results.keys()) + ["AMFormer"]
    metrics = ["auc", "accuracy", "recall", "f1"]

    values = {m: [] for m in metrics}

    for m in metrics:
        for res in baseline_results.values():
            values[m].append(res[m])
        values[m].append(amformer_results[m])

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(10,6))

    for i, m in enumerate(metrics):
        plt.bar(x + i*width, values[m], width, label=m)

    plt.xticks(x + width, models, rotation=30)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("metrics_bar.png", dpi=300)
    plt.show()


# =========================
# 5. 混淆矩阵（AMFormer）
# =========================
def plot_confusion():
    y_true = amformer_results["labels"]
    y_prob = amformer_results["predictions"]

    # 使用0.5阈值（也可以改成最优threshold）
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("AMFormer Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()


# =========================
# 6. 特征重要性（XGBoost）
# =========================
def plot_feature_importance():
    try:
        import pandas as pd
        from xgboost import XGBClassifier

        # 重新加载数据训练一个XGBoost（用于重要性）
        df = pd.read_csv("转化后_编码数据_最终版本.csv")

        y = df["outcome"]
        X = df.drop(columns=["outcome"])
        X = pd.get_dummies(X)

        model = XGBClassifier(n_estimators=200)
        model.fit(X, y)

        importances = model.feature_importances_
        idx = np.argsort(importances)[-10:]

        plt.figure(figsize=(8,5))
        plt.barh(range(10), importances[idx])
        plt.yticks(range(10), X.columns[idx])
        plt.title("Top 10 Feature Importance (XGBoost)")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300)
        plt.show()

    except:
        print("⚠️ 特征重要性绘制失败（可能未安装xgboost）")


# =========================
# 7. SHAP图
# =========================
def plot_shap():
    try:
        import shap
        import pandas as pd
        from xgboost import XGBClassifier

        df = pd.read_csv("转化后_编码数据_最终版本.csv")

        y = df["outcome"]
        X = df.drop(columns=["outcome"])
        X = pd.get_dummies(X)
        model = XGBClassifier(n_estimators=200)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        shap.summary_plot(shap_values, X)

    except:
        print("⚠️ SHAP绘制失败（可能未安装shap）")


# =========================
# 8. t-SNE
# =========================
def plot_tsne():
    try:
        import pandas as pd

        df = pd.read_csv("转化后_编码数据_最终版本.csv")

        y = df["outcome"]
        X = df.drop(columns=["outcome"])
        X = pd.get_dummies(X)
        X = X.fillna(0)

        tsne = TSNE(n_components=2, random_state=42)
        X_emb = tsne.fit_transform(X)

        plt.figure(figsize=(6,5))
        plt.scatter(X_emb[:,0], X_emb[:,1], c=y, cmap="coolwarm", s=10)
        plt.title("t-SNE Visualization")
        plt.tight_layout()
        plt.savefig("tsne.png", dpi=300)
        plt.show()

    except:
        print("⚠️ t-SNE绘制失败")


# =========================
# 主函数
# =========================
if __name__ == "__main__":
    plot_roc()
    plot_pr()
    plot_metrics()
    plot_confusion()
    plot_feature_importance()
    plot_shap()
    plot_tsne()