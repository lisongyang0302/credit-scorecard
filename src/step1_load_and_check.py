import os
import sys
import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data" / "cs-training.csv"
OUT = PROJ / "outputs"
(OUT / "charts").mkdir(parents=True, exist_ok=True)

MYSQL_CONF = dict(host="127.0.0.1", port=3307, user="root", password=os.environ.get("MYSQL_PWD", ""),
                  database="risk_project01", charset="utf8mb4")
engine = create_engine(
    f"mysql+pymysql://{MYSQL_CONF['user']}:{MYSQL_CONF['password']}"
    f"@{MYSQL_CONF['host']}:{MYSQL_CONF['port']}/{MYSQL_CONF['database']}?charset=utf8mb4"
)

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
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS application_train")
    df.to_sql("application_train", engine, if_exists="replace", index=False)
    print("导入完成")

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

    tgt = df["target"].value_counts()
    bad_rate = df["target"].mean()
    lines.append("\n## 1. 目标变量分布\n")
    lines.append(f"- 坏客户（target=1）：{tgt.get(1,0):,} 人，坏账率 **{bad_rate:.2%}**")
    lines.append(f"- 好客户（target=0）：{tgt.get(0,0):,} 人")
    lines.append(f"- 不平衡比：约 {tgt.get(0,0)/max(tgt.get(1,1),1):.0f} : 1（建模时需要处理，策略岗常用分层抽样或权重）")
    lines.append(f"> 📌 分析结论：坏账率 {bad_rate:.2%} 是全流程的基线数字——step2 的每张验证表、step3 的切档表"
                 "都围绕它读偏离；约 14:1 的不平衡决定建模口径（分层切分 + class_weight 加权），"
                 "评估看 KS/AUC 而不是准确率（全猜'好'也有 93% 准确率，无区分意义）。\n")

    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    lines.append("\n## 2. 缺失值\n")
    if len(miss):
        for col, n in miss.items():
            lines.append(f"- **{col}**（{BUSINESS_MEANING.get(col, col)}）：缺失 {n:,}（{n/len(df):.1%}）")
        lines.append("\n> 📌 处理决策与理由：月收入缺失 19.8% 占比大且收入右偏——均值被高收入拉偏，**中位数稳健**；"
                     "同时加缺失标记列，因为'没填'本身可能是信号（自雇/不配合/不愿暴露财务状况），"
                     "直接填充会抹掉这层信息，标记列保留它、有没有预测力交给后续 IV 裁决。"
                     "家属缺失仅 2.6% 且'无家属'（单身）是真实取值，缺失大概率就是'没有'的表达——填 0 语义自然。"
                     "**同一个'缺失'，处理力度跟着业务含义走，不搞一刀切。**\n")
    else:
        lines.append("- 无缺失\n")

    lines.append("\n## 3. 异常值（业务规则体检）\n")
    checks = [
        ("age", (df["age"] < 18) | (df["age"] > 100), "年龄超出[18,100]"),
        ("revolving_utilization", df["revolving_utilization"] > 1, "额度使用率>100%（理论上不该有，业务上常截断为1）"),
        ("monthly_income", df["monthly_income"] == 0, "月收入=0（失业？还是数据问题？）"),
        ("n_loans_90d", df["n_loans_90d"] >= 90, "90天逾期笔数≥90（极端值，可能是逾期次数连记）"),
    ]
    lines.append("\n> 📌 分析结论（逐条）：年龄超界是**数据错误**（未成年人不可能合法申请信贷）→ step2 剔除；"
                 "额度使用率>100% 业务上不可能（透支超过额度），多为统计时点错位的**口径瑕疵** → 截到 1（100%）"
                 "保住样本；月收入=0 的 1,634 条保留不动（'失业报 0 收入'在业务上真实存在，不是错误值）。"
                 "清洗方式的取舍逻辑：**数据错误的样本删掉，口径瑕疵的样本修正保留**。\n")
    lines.append("> 🔍 96/98 判定的完整证据链：官方字典没有写清口径（字段主语是'借款人'，单账户还是"
                 "跨账户合计未说明，且 30-59/60-89 两档定义带 'in the last 2 years' 而 90+ 档没有——字典自身不一致），"
                 "所以判定不依赖口径推理，拿数据形态说话，三重证据：**①断层**——正常取值 0~17 连续衰减，"
                 "18~95 零样本，96/98 与正常分布完全断开；**②孤立**——96 只出现 5 次、98 出现 264 次，"
                 "真实的计数分布不会恰好孤立在两个值上；**③三字段同步**——这 269 条在 30-59 档和 60-89 档"
                 "的取值也是同款 96/98，三档同时取这种值的概率为零，是同一错误源的系统性采集错误（连记或溢出）。"
                 "**处理**：step2 统一修正到正常上界 17（观测到的正常最大值，零误伤、修正位置有数据支撑）。\n")
    for col, mask, desc in checks:
        n = mask.sum()
        if n > 0:
            lines.append(f"- **{col}**：{desc}，{n:,} 条（{n/len(df):.2%}）")

    lines.append("\n## 4. 字段字典（业务语义）\n")
    lines.append("| 字段 | 业务含义 |")
    lines.append("|---|---|")
    for col in df.columns:
        lines.append(f"| {col} | {BUSINESS_MEANING.get(col, '主键') } |")

    report = "\n".join(lines)
    report_path = OUT / "01_data_quality_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n体检报告已写：{report_path}")

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
        q99 = vals.quantile(0.99)
        if col == "debt_ratio":
            shown = np.log1p(vals[vals <= q99].clip(lower=0))
            shown.hist(bins=40, ax=ax, color="#4A90D9", edgecolor="white")
            ax.set_title(f"{col}（log1p 变换显示）", fontsize=11)
            med_x = np.log1p(vals.median())
        elif col == "n_loans_30_59d":
            vals[vals <= 10].hist(bins=11, ax=ax, color="#4A90D9", edgecolor="white")
            ax.set_yscale("log")
            ax.set_title(f"{col}（0-10，y 对数）", fontsize=11)
            med_x = vals.median()
        else:
            vals[vals <= q99].hist(bins=40, ax=ax, color="#4A90D9", edgecolor="white")
            ax.set_title(f"{col}（99分位截断）", fontsize=11)
            med_x = vals.median()
        ax.axvline(med_x, color="red", ls="--", lw=1, label=f"中位数={vals.median():.1f}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    chart = OUT / "charts" / "01_distributions.png"
    plt.savefig(chart, dpi=110)
    plt.close()
    print(f"分布图已存：{chart}")

    print("\n✅ Step 1 完成！数据已入库 + 体检报告已产出")


if __name__ == "__main__":
    main()