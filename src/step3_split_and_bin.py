import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

import pandas as pd
import numpy as np
import scorecardpy as sc
from sqlalchemy import create_engine
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
INT = OUT / "intermediate"
MYSQL = dict(host="127.0.0.1", port=3307, user="root", password=os.environ.get("MYSQL_PWD", ""),
             database="risk_project01", charset="utf8mb4")
engine = create_engine(
    f"mysql+pymysql://{MYSQL['user']}:{MYSQL['password']}@{MYSQL['host']}:{MYSQL['port']}/{MYSQL['database']}?charset=utf8mb4"
)

NUM_FEATURES = ["revolving_utilization", "age", "n_loans_30_59d", "debt_ratio",
                "monthly_income", "n_credit_lines", "n_loans_90d",
                "n_real_estate_loans", "n_loans_60_89d", "n_dependents",
                "est_monthly_debt", "total_late_cnt", "severe_late_ratio",
                "has_severe_late", "income_per_person", "is_income_missing"]


BASE_SCORE = 600
PDO = 50
BASE_ODDS = 1 / 50


def main():
    print("=" * 60)
    print("Step 3: 读取原始清洗数据 + 先切分（防泄漏的关键）")
    print("=" * 60)
    df = pd.read_sql("SELECT * FROM application_clean", engine)
    print(f"全量 {df.shape[0]:,} 行，坏账率 {df['target'].mean():.2%}")

    y = df["target"]
    X_raw = df[NUM_FEATURES + ["user_id"]]
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.3, stratify=y, random_state=42)
    print(f"先切分：训练 {len(X_tr_raw):,} / 测试 {len(X_te_raw):,}（后续所有统计仅基于训练集）")

    print("\n" + "=" * 60)
    print("Step 3: 仅用训练集做 WOE 分箱 + IV 筛选")
    print("=" * 60)
    train_for_bin = X_tr_raw.copy()
    train_for_bin["target"] = y_tr.values
    bins = sc.woebin(train_for_bin, y="target", x=NUM_FEATURES,
                     bins_num=5, mono_enforce=True, print_listbox=False)
    print(f"训练集分箱完成 {len(bins)} 个特征")

    train_woe = sc.woebin_ply(X_tr_raw[NUM_FEATURES], bins)
    test_woe = sc.woebin_ply(X_te_raw[NUM_FEATURES], bins)
    print(f"WOE 映射完成：训练 {train_woe.shape} / 测试 {test_woe.shape}")

    iv_df = sc.iv(train_for_bin, y="target", x=NUM_FEATURES)
    iv_df.to_csv(OUT / "_iv_by_feature.csv", index=False)
    weak = iv_df[iv_df["info_value"] < 0.02]["variable"].tolist()
    print(f"IV<0.02 弱特征：{weak if weak else '无'}")

    iv_summary = (pd.concat({k: v[["variable", "bin", "count", "badprob", "woe", "bin_iv"]]
                             for k, v in bins.items()})
                  .reset_index(level=0, names="feature"))
    iv_summary["bin_iv"] = pd.to_numeric(iv_summary["bin_iv"], errors="coerce").fillna(0)
    iv_by_feature = (iv_summary.groupby("variable")["bin_iv"].sum()
                     .sort_values(ascending=False).rename("total_iv").reset_index())
    iv_by_feature["预测力"] = pd.cut(iv_by_feature["total_iv"],
                                    bins=[0, 0.02, 0.1, 0.3, float("inf")],
                                    labels=["无预测力(弃)", "弱", "中", "强"])
    print("\nIV 值排序（预测力体检，箱级求和口径）：")
    print(iv_by_feature.to_string(index=False))

    with pd.ExcelWriter(OUT / "03_woe_details.xlsx", engine="openpyxl") as w:
        iv_by_feature.to_excel(w, sheet_name="IV汇总", index=False)
        for feat in [f for f in NUM_FEATURES if f not in weak]:
            sheet = iv_summary[iv_summary["variable"] == feat].drop(columns=["variable"])
            sheet.to_excel(w, sheet_name=feat[:28], index=False)
    print(f"WOE 明细 Excel：{OUT / '03_woe_details.xlsx'}")

    iv_by_feature.to_csv(OUT / "03_iv_summary.csv", index=False, encoding="utf-8-sig")

    top_feats = iv_by_feature.head(8)["variable"].tolist()
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("各特征 WOE 趋势（单调性=可解释性）", fontsize=15)
    for ax, feat in zip(axes.flat, top_feats):
        sub = iv_summary[iv_summary["variable"] == feat]
        if len(sub) == 0:
            continue
        ax2 = ax.twinx()
        xs = range(len(sub))
        ax.bar(xs, sub["count"], alpha=0.3, color="#4A90D9", label="样本数")
        ax2.plot(xs, sub["woe"], "ro-", lw=1.5, label="WOE")
        ax.set_xticks(xs)
        ax.set_xticklabels(sub["bin"].astype(str).str.slice(0, 18), rotation=30, fontsize=6.5)
        ax2.set_ylabel("WOE", fontsize=8)
        ax.set_title(feat, fontsize=10)
        if feat == top_feats[0]:
            ax.legend(fontsize=7, loc="upper left")
            ax2.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    chart = OUT / "charts" / "03_woe_trends.png"
    plt.savefig(chart, dpi=110)
    plt.close()
    print(f"WOE 趋势图：{chart}")

    iv_by_feature.to_csv(OUT / "_iv_by_feature.csv", index=False, encoding="utf-8-sig")
    df_woe_all = sc.woebin_ply(df[NUM_FEATURES + ["user_id", "target"]], bins)
    df_woe_all.to_sql("application_woe", engine, if_exists="replace", index=False)
    bins_meta = pd.concat({k: v.assign(_feature=k) for k, v in bins.items()}).reset_index(level=0, names="feature")
    bins_meta.to_sql("woe_bins_meta", engine, if_exists="replace", index=False)

    drop_cols = [c for c in train_woe.columns
                 if any(w in c for w in weak) or c == "user_id"]
    X_tr = train_woe.drop(columns=[c for c in drop_cols if c in train_woe.columns])
    X_te = test_woe.drop(columns=[c for c in drop_cols if c in test_woe.columns])
    print(f"IV 筛选后剩余特征 {X_tr.shape[1]} 个")

    INT.mkdir(parents=True, exist_ok=True)
    X_tr.to_csv(INT / "X_tr_after_iv.csv", index=False)
    X_te.to_csv(INT / "X_te_after_iv.csv", index=False)
    X_tr_raw.to_csv(INT / "X_tr_raw.csv", index=False)
    X_te_raw.to_csv(INT / "X_te_raw.csv", index=False)
    y_tr.to_frame().to_csv(INT / "y_tr.csv", index=False)
    y_te.to_frame().to_csv(INT / "y_te.csv", index=False)
    iv_df.to_csv(INT / "iv_raw.csv", index=False)
    print(f"中间产物落盘：{INT}")
    print("\n✅ Step 3 完成！切分 + 训练集分箱 + IV 筛选（交付物齐）")


if __name__ == "__main__":
    main()
