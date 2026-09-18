# -*- coding: utf-8 -*-
import os
"""
Project 01 · Step 2: 数据清洗 + 特征工程（策略视角）
==================================================
输入：MySQL risk_project01.application_train（Step1 已入库）
输出：
- 清洗后的表写回 MySQL（application_clean）
- 特征工程说明写入 outputs/02_feature_engineering.md

清洗决策（每一条都影响模型口径）：
1. monthly_income 缺失 19.8% → 中位数填充 + 加 is_income_missing 标记列（缺失本身就是信号）
2. n_dependents 缺失 2.6% → 填 0
3. revolving_utilization >1 → 截断到 1（业务上额度使用率封顶100%）
4. n_loans_90d >= 90 → 视为数据异常，截断到 96 分位处理（业务上逾期次数连记错误）
5. age < 18 → 剔除（未成年人不可能申请信贷，14条直接删）

特征工程（把原始字段变成"风控语言"）：
- debt_ratio 的口径问题：debt_ratio 月收入为0时无意义 → 构造 est_debt = debt_ratio * income
- 逾期总次数 = 30-59 + 60-89 + 90+ （三档合并看总逾期倾向）
- 严重逾期占比 = (60_89 + 90+) / 总逾期（区分"偶尔忘还"和"习惯性违约"）
- 收入分箱（业务断点）：低/中/高收入
- 年龄分箱（业务断点）：22-30 初入职场 / 31-45 稳定期 / 46+ 成熟期
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
MYSQL = dict(host="127.0.0.1", port=3307, user="root", password=os.environ.get("MYSQL_PWD", ""),
             database="risk_project01", charset="utf8mb4")
engine = create_engine(
    f"mysql+pymysql://{MYSQL['user']}:{MYSQL['password']}@{MYSQL['host']}:{MYSQL['port']}/{MYSQL['database']}?charset=utf8mb4"
)


def main():
    print("=" * 60)
    print("Step 2a: 从 MySQL 读取原始数据")
    print("=" * 60)
    df = pd.read_sql("SELECT * FROM application_train", engine)
    print(f"读回：{df.shape[0]:,} 行 × {df.shape[1]} 列")

    print("\n" + "=" * 60)
    print("Step 2b: 数据清洗（每条决策见文件头注释）")
    print("=" * 60)
    n0 = len(df)

    # 1. 年龄异常直接剔除
    bad_age = (df["age"] < 18) | (df["age"] > 100)
    df = df[~bad_age].copy()
    print(f"1. 剔除年龄异常 {bad_age.sum()} 条 → 剩 {len(df):,}")

    # 2. 额度使用率封顶
    over100 = (df["revolving_utilization"] > 1).sum()
    df["revolving_utilization"] = df["revolving_utilization"].clip(upper=1)
    print(f"2. 额度使用率>1 截断至 1：{over100:,} 条")

    # 3. 90天逾期极端值截断（99.8分位）
    cap90 = df["n_loans_90d"].quantile(0.998)
    extreme = (df["n_loans_90d"] > cap90).sum()
    df["n_loans_90d"] = df["n_loans_90d"].clip(upper=cap90)
    print(f"3. n_loans_90d 截断到 {cap90:.0f}：{extreme:,} 条")

    # 4. 缺失值
    df["is_income_missing"] = df["monthly_income"].isnull().astype(int)
    med_income = df["monthly_income"].median()
    df["monthly_income"] = df["monthly_income"].fillna(med_income)
    print(f"4. 月收入缺失填充（中位数 {med_income:,.0f}）+ 加缺失标记列")

    df["n_dependents"] = df["n_dependents"].fillna(0).astype(int)
    print("5. 家属数缺失填 0")

    assert df.isnull().sum().sum() == 0, "还有缺失没处理！"
    print(f"\n清洗完成：{n0:,} → {len(df):,} 行，零缺失")

    print("\n" + "=" * 60)
    print("Step 2c: 特征工程（风控语言）")
    print("=" * 60)

    # 特征1：估算月负债（负债率反推绝对负债）
    df["est_monthly_debt"] = (df["debt_ratio"] * df["monthly_income"]).round(2)

    # 特征2：总逾期次数
    df["total_late_cnt"] = (df["n_loans_30_59d"]
                            + df["n_loans_60_89d"]
                            + df["n_loans_90d"])

    # 特征3：严重逾期占比（60d+ / 总逾期）——区分偶尔忘还 vs 习惯性违约
    df["severe_late_ratio"] = np.where(
        df["total_late_cnt"] > 0,
        (df["n_loans_60_89d"] + df["n_loans_90d"]) / df["total_late_cnt"],
        0
    ).round(4)

    # 特征4：是否有过严重逾期（二值）
    df["has_severe_late"] = ((df["n_loans_60_89d"] + df["n_loans_90d"]) > 0).astype(int)

    # 特征5：年龄分组（业务断点）
    df["age_group"] = pd.cut(df["age"], bins=[0, 25, 35, 50, 100],
                             labels=["青年(<25)", "轻熟(26-35)", "中年(36-50)", "资深(50+)"])

    # 特征6：收入分组（按分位切三档，业务上对应普惠/标准/优质客群）
    q1, q2 = df["monthly_income"].quantile([0.33, 0.66])
    df["income_group"] = pd.cut(df["monthly_income"],
                                bins=[-1, q1, q2, float("inf")],
                                labels=["低收入", "中收入", "高收入"])

    # 特征7：信贷活跃度（开放信贷笔数分箱）
    df["credit_activity"] = pd.cut(df["n_credit_lines"], bins=[-1, 3, 8, 100],
                                   labels=["低活跃", "中活跃", "高活跃"])

    # 特征8：人均负担（家属越多负担越重）
    df["income_per_person"] = (df["monthly_income"] / (1 + df["n_dependents"])).round(2)

    new_feats = ["est_monthly_debt", "total_late_cnt", "severe_late_ratio",
                 "has_severe_late", "age_group", "income_group", "credit_activity",
                 "income_per_person"]
    print(f"新增 {len(new_feats)} 个特征：{new_feats}")

    print("\n" + "=" * 60)
    print("Step 2d: 特征与目标的初步关系验证（策略直觉检查）")
    print("=" * 60)
    # 每个新特征都要"讲得通业务"，这一步是验证
    lines = ["# Step2 特征工程报告\n"]

    # 按年龄组看坏账率
    bad_by_age = df.groupby("age_group", observed=True)["target"].agg(["mean", "count"])
    lines.append("\n## 年龄组 × 坏账率\n\n| 组 | 客户数 | 坏账率 |\n|---|---|---|")
    for g, row in bad_by_age.iterrows():
        lines.append(f"| {g} | {row['count']:,} | {row['mean']:.2%} |")

    # 按收入组看坏账率
    bad_by_inc = df.groupby("income_group", observed=True)["target"].agg(["mean", "count"])
    lines.append("\n## 收入组 × 坏账率\n\n| 组 | 客户数 | 坏账率 |\n|---|---|---|")
    for g, row in bad_by_inc.iterrows():
        lines.append(f"| {g} | {row['count']:,} | {row['mean']:.2%} |")

    # 逾期史 × 坏账率（这个必须强相关，否则数据有问题）
    bad_by_late = df.groupby("has_severe_late")["target"].agg(["mean", "count"])
    lines.append("\n## 是否有严重逾期史 × 坏账率\n\n| 有严重逾期 | 客户数 | 坏账率 |\n|---|---|---|")
    for g, row in bad_by_late.iterrows():
        label = "有" if g == 1 else "无"
        lines.append(f"| {label} | {row['count']:,} | {row['mean']:.2%} |")

    lines.append("\n> 💡 业务解读：以上三张表就是策略分析的第一课——**任何特征要进模型，先看它对坏账率的区分度**。\n")

    report_path = OUT / "02_feature_engineering.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"特征工程报告：{report_path}")

    print("\n" + "=" * 60)
    print("Step 2e: 清洗+特征后的数据写回 MySQL")
    print("=" * 60)
    df.to_sql("application_clean", engine, if_exists="replace", index=False)
    print("已写回：risk_project01.application_clean")

    print("\n✅ Step 2 完成！")
    return df


if __name__ == "__main__":
    main()
