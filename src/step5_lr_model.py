import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
INT = PROJ / "outputs" / "intermediate"


def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return max(tpr - fpr)


def main():
    print("=" * 60)
    print("Step 5: 逻辑回归建模（无泄漏版）")
    print("=" * 60)
    X_tr = pd.read_csv(INT / "X_tr_final.csv")
    X_te = pd.read_csv(INT / "X_te_final.csv")
    y_tr = pd.read_csv(INT / "y_tr.csv")["target"]
    y_te = pd.read_csv(INT / "y_te.csv")["target"]

    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    lr.fit(X_tr, y_tr)
    p_tr = lr.predict_proba(X_tr)[:, 1]
    p_te = lr.predict_proba(X_te)[:, 1]

    ks_tr, ks_te = ks_stat(y_tr, p_tr), ks_stat(y_te, p_te)
    auc_tr, auc_te = roc_auc_score(y_tr, p_tr), roc_auc_score(y_te, p_te)
    print(f"训练集：KS={ks_tr:.3f}, AUC={auc_tr:.3f}")
    print(f"测试集：KS={ks_te:.3f}, AUC={auc_te:.3f}")

    coef = pd.DataFrame({"feature": X_tr.columns, "coef": lr.coef_[0]})
    coef["abs_coef"] = coef["coef"].abs()
    coef = coef.sort_values("abs_coef", ascending=False)
    top3 = coef.head(3)["feature"].tolist()
    neg = coef[coef["coef"] < 0]["feature"].tolist()
    print(f"\n系数三看：最强三个 {'、'.join(top3)}；负向 {('、'.join(neg) if neg else '无')}（收入/活跃类，方向由分箱结构决定）")

    coef.to_csv(INT / "lr_coef.csv", index=False)
    pd.DataFrame({"p_bad": p_te}).to_csv(INT / "p_te.csv", index=False)
    metrics = {"ks_tr": ks_tr, "ks_te": ks_te, "auc_tr": auc_tr, "auc_te": auc_te,
               "n_features": int(X_tr.shape[1])}
    (INT / "model_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    print(f"\n✅ Step 5 完成！逻辑回归建模（入模 {metrics['n_features']} 特征）")


if __name__ == "__main__":
    main()
