# -*- coding: utf-8 -*-
"""
Project 01 · Step 1: 数据导入与业务体检
=====================================
流程：CSV → MySQL（LOAD DATA）→ 从库读回 pandas → 数据质量体检报告

跑本项目前确保：
1. MySQL 已启动，root 密码正确
2. 先在 MySQL 里执行 sql/01_create_database.sql
3. 修改下方 MYSQL_CONF 或设置环境变量

输出：
- outputs/01_data_quality_report.md   数据质量体检报告
- outputs/charts/*.png                分布图
"""
import os
import sys
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from pathlib import Path

# ---------- 路径 ----------
PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ.parent / "project02" / "data" / "cs-training.csv"
OUT = PROJ / "outputs"
(OUT / "charts").mkdir(parents=True, exist_ok=True)

# ---------- MySQL 连接 ----------
MYSQL_CONF = dict(host="127.0.0.1", port=3307, user="root", password=os.environ.get("MYSQL_PWD", ""),
                  database="risk_project01", charset="utf8mb4")
engine = create_engine(
    f"mysql+pymysql://{MYSQL_CONF['user']}:{MYSQL_CONF['password']}"
    f"@{MYSQL_CONF['host']}:{MYSQL_CONF['port']}/{MYSQL_CONF['database']}?charset=utf8mb4"
)

# ---------- 字段字典（Kaggle名 → 业务名） ----------
RENAME_MAP = {
    "Unnamed: 0": "user_id",
    "SeriousDlqin2yrs": "target",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "n_loans_30_59d",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "n_credit_lines",
    "NumberOfTimes90DaysLate": "n_loans_90d",
    "NumberRealEstateLoansOrLines": "n_real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "n_loans_60_89d",
    "NumberOfDependents": "n_dependents",
}

BUSINESS_MEANING = {
    "target": "目标变量：1=未来2年内逾期90天+（坏客户）",
    "revolving_utilization": "信用卡额度使用率：总额度用掉了多少（风控第一敏感特征）",
    "age": "借款人年龄",
    "n_loans_30_59d": "30-59天逾期笔数：短期逾期的信号",
    "debt_ratio": "负债率：每月还债支出/月收入（含赡养）",
    "monthly_income": "月收入",
    "n_credit_lines": "开放信贷/贷款笔数：信用活跃度",
    "n_loans_90d": "90天+逾期笔数：严重逾期信号",
    "n_real_estate_loans": "不动产/担保贷款笔数：资产背书",
    "n_loans_60_89d": "60-89天逾期笔数",
    "n_dependents": "家属人数：负担系数",
}


def main():
    print("=" * 60)
    print("Step 1a: 读取 CSV 并按业务语义重命名")
    print("=" * 60)
    df = pd.read_csv(DATA)
    df = df.rename(columns=RENAME_MAP)
    print(f"原始数据：{df.shape[0]:,} 行 × {df.shape[1]} 列")

    print("\n" + "=" * 60)
    print("Step 1b: 导入 MySQL（risk_project01.application_train）")
    print("=" * 60)
    # 先清空再写入（幂等）
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS application_train")
    df.to_sql("application_train", engine, if_exists="replace", index=False)
    print("导入完成")

    # 验证：从 MySQL 读回（确认数据真的在库里）
    df_db = pd.read_sql("SELECT * FROM application_train LIMIT 5", engine)
    assert len(df_db) == 5, "MySQL 读回失败"
    n_db = pd.read_sql("SELECT COUNT(*) AS n FROM application_train", engine)["n"][0]
    print(f"MySQL 验证：库里 {n_db:,} 行（预期 {df.shape[0]:,}）")
    assert n_db == len(df), "行数不一致！导入有问题"

    print("\n" + "=" * 60)
    print("Step 1c: 数据质量体检")
    print("=" * 60)
    lines = ["# 数据质量体检报告（Step 1）\n",
             f"数据源：GiveMeSomeCredit 训练集，{df.shape[0]:,} 行 × {df.shape[1]} 列\n"]

    # 1. 目标变量分布
    tgt = df["target"].value_counts()
    bad_rate = df["target"].mean()
    lines.append("\n## 1. 目标变量分布\n")
    lines.append(f"- 坏客户（target=1）：{tgt.get(1,0):,} 人，坏账率 **{bad_rate:.2%}**")
    lines.append(f"- 好客户（target=0）：{tgt.get(0,0):,} 人")
    lines.append(f"- 不平衡比：约 {tgt.get(0,0)/max(tgt.get(1,1),1):.0f} : 1（建模时需要处理，策略岗常用分层抽样或权重）\n")

    # 2. 缺失值
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    lines.append("\n## 2. 缺失值\n")
    if len(miss):
        for col, n in miss.items():
            lines.append(f"- **{col}**（{BUSINESS_MEANING.get(col, col)}）：缺失 {n:,}（{n/len(df):.1%}）")
        lines.append("\n> 处理决策：月收入缺失→用中位数填充并加缺失标记列；家属缺失→填0（无家属是合理值）\n")
    else:
        lines.append("- 无缺失\n")

    # 3. 异常值体检（业务规则）
    lines.append("\n## 3. 异常值（业务规则体检）\n")
    checks = [
        ("age", (df["age"] < 18) | (df["age"] > 100), "年龄超出[18,100]"),
        ("revolving_utilization", df["revolving_utilization"] > 1, "额度使用率>100%（理论上不该有，业务上常截断为1）"),
        ("monthly_income", df["monthly_income"] == 0, "月收入=0（失业？还是数据问题？）"),
        ("n_loans_90d", df["n_loans_90d"] >= 90, "90天逾期笔数≥90（极端值，可能是逾期次数连记）"),
    ]
    for col, mask, desc in checks:
        n = mask.sum()
        if n > 0:
            lines.append(f"- **{col}**：{desc}，{n:,} 条（{n/len(df):.2%}）")

    # 4. 字段字典表
    lines.append("\n## 4. 字段字典（业务语义）\n")
    lines.append("| 字段 | 业务含义 |")
    lines.append("|---|---|")
    for col in df.columns:
        lines.append(f"| {col} | {BUSINESS_MEANING.get(col, '主键') } |")

    report = "\n".join(lines)
    report_path = OUT / "01_data_quality_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n体检报告已写：{report_path}")

    # 5. 基础分布图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Step1 · 关键特征分布体检", fontsize=15)
    plot_cols = ["revolving_utilization", "age", "debt_ratio",
                 "monthly_income", "n_loans_30_59d", "n_credit_lines"]
    for ax, col in zip(axes.flat, plot_cols):
        vals = df[col].dropna()
        # 长尾特征用分位数截断显示
        q99 = vals.quantile(0.99)
        vals[vals <= q99].hist(bins=40, ax=ax, color="#4A90D9", edgecolor="white")
        ax.set_title(f"{col}（99分位截断）", fontsize=11)
        ax.axvline(vals.median(), color="red", ls="--", lw=1, label=f"中位数={vals.median():.1f}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    chart = OUT / "charts" / "01_distributions.png"
    plt.savefig(chart, dpi=110)
    plt.close()
    print(f"分布图已存：{chart}")

    print("\n✅ Step 1 完成！数据已入库 + 体检报告已产出")


if __name__ == "__main__":
    main()
