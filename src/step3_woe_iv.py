# -*- coding: utf-8 -*-
import os
"""
Project 01 · Step 3: WOE分箱 + IV筛选（评分卡的核心工序）
=======================================================
这是评分卡项目的"灵魂步骤"，三个关键点：
- 为什么要分箱？（逻辑回归要线性关系 + 评分卡要可解释）
- WOE怎么算？（各组好坏分布比的对数）
- IV怎么用？（衡量特征整体预测力，<0.02弃用，0.02-0.1弱，0.1-0.3中，>0.3强）

方法：用 scorecardpy 的自动分箱（等频+卡方合并），关键特征手工调整断点
输出：
- outputs/03_iv_summary.csv     所有特征IV值排序
- outputs/03_woe_details.xlsx   每个特征的WOE明细表（Excel交付物）
- outputs/charts/03_woe_*.png   WOE趋势图
"""
import pandas as pd
import numpy as np
import scorecardpy as sc
from sqlalchemy import create_engine
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
MYSQL = dict(host="127.0.0.1", port=3307, user="root", password=os.environ.get("MYSQL_PWD", ""),
             database="risk_project01", charset="utf8mb4")
engine = create_engine(
    f"mysql+pymysql://{MYSQL['user']}:{MYSQL['password']}@{MYSQL['host']}:{MYSQL['port']}/{MYSQL['database']}?charset=utf8mb4"
)

# 参与建模的候选特征（分类变量先用 ordinal 编码，WOE 阶段统一处理）
NUM_FEATURES = ["revolving_utilization", "age", "n_loans_30_59d", "debt_ratio",
                "monthly_income", "n_credit_lines", "n_loans_90d",
                "n_real_estate_loans", "n_loans_60_89d", "n_dependents",
                "est_monthly_debt", "total_late_cnt", "severe_late_ratio",
                "has_severe_late", "income_per_person", "is_income_missing"]


def main():
    print("=" * 60)
    print("Step 3a: 读取清洗后数据")
    print("=" * 60)
    df = pd.read_sql("SELECT * FROM application_clean", engine)
    print(f"{df.shape[0]:,} 行")
    # 分类列转字符串（scorecardpy 要求）
    for col in ["age_group", "income_group", "credit_activity"]:
        df[col] = df[col].astype(str)

    print("\n" + "=" * 60)
    print("Step 3b: WOE 自动分箱（scorecardpy）")
    print("=" * 60)
    # 分箱：tree方法（chimerge在pandas 2.x下variable列会变成嵌套ndarray导致后续崩）
    # mono_enforce=True 强制单调（评分卡可解释的底线）
    bins = sc.woebin(df, y="target", x=NUM_FEATURES,
                     bins_num=5,
                     mono_enforce=True, print_listbox=False)
    print(f"完成 {len(bins)} 个特征的分箱")

    # IV 汇总
    iv_summary = (pd.concat({k: v[["variable", "bin", "count", "badprob", "woe", "bin_iv"]]
                             for k, v in bins.items()})
                  .reset_index(level=0, names="feature"))
    iv_summary["bin_iv"] = pd.to_numeric(iv_summary["bin_iv"], errors="coerce").fillna(0)
    iv_by_feature = (iv_summary.groupby("variable")["bin_iv"].sum()
                     .sort_values(ascending=False).rename("total_iv").reset_index())
    iv_by_feature["预测力"] = pd.cut(iv_by_feature["total_iv"],
                                    bins=[0, 0.02, 0.1, 0.3, float("inf")],
                                    labels=["无预测力(弃)", "弱", "中", "强"])
    print("\nIV 值排序（预测力体检）：")
    print(iv_by_feature.to_string(index=False))

    # 剔除无预测力特征
    weak = iv_by_feature[iv_by_feature["total_iv"] < 0.02]["variable"].tolist()
    keep = [f for f in NUM_FEATURES if f not in weak]
    print(f"\n剔除无预测力特征（IV<0.02）：{weak}")
    print(f"保留 {len(keep)} 个特征进入建模")

    # WOE 转换
    print("\n" + "=" * 60)
    print("Step 3c: WOE 转换")
    print("=" * 60)
    df_woe = sc.woebin_ply(df, bins)
    print(f"WOE 后：{df_woe.shape[0]:,} 行 × {df_woe.shape[1]} 列")

    with pd.ExcelWriter(OUT / "03_woe_details.xlsx", engine="openpyxl") as w:
        iv_by_feature.to_excel(w, sheet_name="IV汇总", index=False)
        for feat in keep:
            sheet = iv_summary[iv_summary["variable"] == feat].drop(columns=["variable"])
            sheet.to_excel(w, sheet_name=feat[:28], index=False)
    print(f"WOE 明细 Excel：{OUT / '03_woe_details.xlsx'}")

    # IV 汇总 CSV
    iv_by_feature.to_csv(OUT / "03_iv_summary.csv", index=False, encoding="utf-8-sig")

    # WOE 趋势图（每个保留特征一张小图）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    top_feats = iv_by_feature.head(8)["variable"].tolist()
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("Step3 · 各特征 WOE 趋势（单调性=可解释性）", fontsize=15)
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

    # 保存中间结果供 Step4 使用
    iv_by_feature.to_csv(OUT / "_iv_by_feature.csv", index=False, encoding="utf-8-sig")
    df_woe.to_sql("application_woe", engine, if_exists="replace", index=False)
    bins_meta = pd.concat({k: v.assign(_feature=k) for k, v in bins.items()}).reset_index(level=0, names="feature")
    bins_meta.to_sql("woe_bins_meta", engine, if_exists="replace", index=False)

    print("\n✅ Step 3 完成！IV筛选 + WOE转换 + Excel交付物")


if __name__ == "__main__":
    main()
