# -*- coding: utf-8 -*-
"""step4b 四项判定实验（回应外AI质疑：LR 0.860 的来源审计）
A. GBM 对照：同一 split 跑 LightGBM —— 对照 LR 是否"打平 GBM"（异常信号）
B. 置换检验：打乱 target 重跑全 pipeline —— AUC 应回 0.5，否则存在未知泄漏
C. 换种子 3 次：重切分重分箱重训练 —— AUC 波动范围（排除烧测试集）
D. 合成特征单变量审计：total_late_cnt / severe_late_ratio 单变量 AUC —— 是否异常超群
"""
import os
import pandas as pd
import numpy as np
import scorecardpy as sc
from sqlalchemy import create_engine
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
MYSQL = dict(host="127.0.0.1", port=3307, user="root", password=os.environ.get("MYSQL_PWD", ""),
             database="risk_project01", charset="utf8mb4")
engine = create_engine(
    f"mysql+pymysql://{MYSQL['user']}:{MYSQL['password']}@{MYSQL['host']}:{MYSQL['port']}/{MYSQL['database']}?charset=utf8mb4")

NUM_FEATURES = ["revolving_utilization", "age", "n_loans_30_59d", "debt_ratio",
                "monthly_income", "n_credit_lines", "n_loans_90d",
                "n_real_estate_loans", "n_loans_60_89d", "n_dependents",
                "est_monthly_debt", "total_late_cnt", "severe_late_ratio",
                "has_severe_late", "income_per_person", "is_income_missing"]
SYNTH = ["total_late_cnt", "severe_late_ratio", "est_monthly_debt",
         "income_per_person", "has_severe_late", "is_income_missing"]


def run_pipeline(df, seed=42, y_override=None):
    """v2 正确流程：切分→训练集分箱→映射→LR。返回 test AUC/KS"""
    y = y_override if y_override is not None else df["target"]
    X_raw = df[NUM_FEATURES]  # 仅数值列（分类列本就不入模）
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.3, stratify=y, random_state=seed)
    train_bin = X_tr_raw.copy()
    train_bin["target"] = np.asarray(y_tr)
    bins = sc.woebin(train_bin, y="target", x=NUM_FEATURES,
                     bins_num=5, mono_enforce=True, print_listbox=False)
    tr_woe = sc.woebin_ply(X_tr_raw, bins)
    te_woe = sc.woebin_ply(X_te_raw, bins)
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    lr.fit(tr_woe, np.asarray(y_tr))
    p_te = lr.predict_proba(te_woe)[:, 1]
    p_tr = lr.predict_proba(tr_woe)[:, 1]
    return roc_auc_score(y_te, p_te), roc_auc_score(np.asarray(y_tr), p_tr), p_te, np.asarray(y_te)


def main():
    df = pd.read_sql("SELECT * FROM application_clean", engine)
    for col in ["age_group", "income_group", "credit_activity"]:
        df[col] = df[col].astype(str)
    print(f"数据 {len(df):,} 行，坏账率 {df['target'].mean():.2%}\n")

    # ===== 实验 D：合成特征单变量审计（先跑，最快出结论）=====
    print("=" * 60)
    print("实验 D：单变量 AUC（原始 vs 合成）")
    print("=" * 60)
    y = df["target"]
    for f in ["revolving_utilization", "n_loans_30_59d", "total_late_cnt",
              "severe_late_ratio", "est_monthly_debt", "debt_ratio", "age"]:
        auc = roc_auc_score(y, -df[f])  # 取负号：值越大越好/越小越好统一方向取max
        auc2 = roc_auc_score(y, df[f])
        best = max(auc, auc2)
        tag = "合成" if f in SYNTH else "原始"
        print(f"  [{tag}] {f}: 单变量AUC={best:.3f}")

    # ===== 实验 B：置换检验（pipeline 有无隐藏泄漏）=====
    print("\n" + "=" * 60)
    print("实验 B：置换检验（打乱 target）")
    print("=" * 60)
    rng = np.random.RandomState(7)
    y_shuffled = pd.Series(rng.permutation(df["target"].values), index=df.index)
    auc_b, _, _, _ = run_pipeline(df, seed=42, y_override=y_shuffled)
    print(f"  打乱 target 后 test AUC = {auc_b:.4f}（应≈0.500±0.005，显著>0.5=未知泄漏）")

    # ===== 实验 C：换种子 3 次 =====
    print("\n" + "=" * 60)
    print("实验 C：换种子重切分（3 次）")
    print("=" * 60)
    aucs = []
    for seed in [42, 123, 2026]:
        auc_te, auc_tr, _, _ = run_pipeline(df, seed=seed)
        aucs.append(auc_te)
        print(f"  seed={seed}: test AUC={auc_te:.4f} (train {auc_tr:.4f})")
    print(f"  3种子均值={np.mean(aucs):.4f} 波动={np.std(aucs):.4f}")

    # ===== 实验 A：GBM 对照（同 split 同特征）=====
    print("\n" + "=" * 60)
    print("实验 A：LightGBM 对照（同 split）")
    print("=" * 60)
    y = df["target"]
    X_all = df[NUM_FEATURES]
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.3, stratify=y, random_state=42)
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    gbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                             scale_pos_weight=spw, random_state=42, verbose=-1)
    gbm.fit(X_tr, y_tr)
    auc_gbm = roc_auc_score(y_te, gbm.predict_proba(X_te)[:, 1])
    print(f"  LightGBM(原始特征, 同split): test AUC={auc_gbm:.4f}")
    print(f"  （对照：LR+WOE 0.860；公开基准 LR+WOE≈0.73-0.79、单GBM≈0.86-0.865）")


if __name__ == "__main__":
    main()
