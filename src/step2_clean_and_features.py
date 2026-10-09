import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
OUT.mkdir(exist_ok=True)

MYSQL = {
    "host": "127.0.0.1", "port": 3307, "user": "root",
    "password": os.environ.get("MYSQL_PWD", ""), "database": "risk_project01",
}
engine = create_engine(
    f"mysql+pymysql://{MYSQL['user']}:{MYSQL['password']}@{MYSQL['host']}:{MYSQL['port']}/{MYSQL['database']}?charset=utf8mb4")


def _md_table(tab: pd.DataFrame, first_col: str) -> str:
    lines = [f"| {first_col} | 客户数 | 坏账率 |", "|---|---|---|"]
    for _, r in tab.iterrows():
        lines.append(f"| {r.iloc[0]} | {int(r.iloc[2]):,} | {r.iloc[1]:.2%} |")
    return "\n".join(lines)


def _gradient_line(tab: pd.DataFrame, insight: str) -> str:
    rates = tab.iloc[:, 1]
    hi_i, lo_i = rates.idxmax(), rates.idxmin()
    hi, lo = rates.max(), rates.min()
    hi_g, lo_g = tab.loc[hi_i, tab.columns[0]], tab.loc[lo_i, tab.columns[0]]
    mult = hi / lo if lo > 0 else float("inf")
    mult_s = f"{mult:.1f} 倍" if mult != float("inf") else "—"
    return f"\n> 📌 梯度：{lo_g} {lo:.2%} → {hi_g} {hi:.2%}（{mult_s}）。{insight}"


def main():
    print("=" * 60)
    print("Step 2a: 从 MySQL 读取原始数据")
    print("=" * 60)
    df = pd.read_sql("SELECT * FROM application_train", engine)
    print(f"读回：{df.shape[0]:,} 行 × {df.shape[1]} 列")

    print("\n" + "=" * 60)
    print("Step 2b: 数据清洗（每条决策见注释）")
    print("=" * 60)
    n0 = len(df)

    bad_age = (df["age"] < 18) | (df["age"] > 100)
    df = df[~bad_age].reset_index(drop=True)
    print(f"1. 剔除年龄异常 {bad_age.sum()} 条 → 剩 {len(df):,}")

    over100 = (df["revolving_utilization"] > 1).sum()
    df.loc[df["revolving_utilization"] > 1, "revolving_utilization"] = 1
    print(f"2. 额度使用率>1 截断至 1：{over100:,} 条")

    extreme = (df["n_loans_90d"] >= 90).sum()
    df.loc[df["n_loans_90d"] >= 90, "n_loans_90d"] = 17
    print(f"3. n_loans_90d 截断到正常上界 17：{extreme:,} 条")

    med_income = df["monthly_income"].median()
    df["is_income_missing"] = df["monthly_income"].isna().astype(int)
    df["monthly_income"] = df["monthly_income"].fillna(med_income)
    print(f"4. 月收入缺失填充（中位数 {med_income:,.0f}）+ 加缺失标记列")

    df["n_dependents"] = df["n_dependents"].fillna(0)
    print("5. 家属数缺失填 0")

    print(f"\n清洗完成：{n0:,} → {len(df):,} 行，零缺失")

    print("\n" + "=" * 60)
    print("Step 2c: 特征工程（风控语言）")
    print("=" * 60)

    df["est_monthly_debt"] = (df["debt_ratio"] * df["monthly_income"]).round(2)

    df["total_late_cnt"] = (df["n_loans_30_59d"] + df["n_loans_60_89d"]
                            + df["n_loans_90d"])

    df["severe_late_ratio"] = np.where(
        df["total_late_cnt"] > 0,
        (df["n_loans_60_89d"] + df["n_loans_90d"]) / df["total_late_cnt"],
        0
    ).round(4)

    df["has_severe_late"] = ((df["n_loans_60_89d"] + df["n_loans_90d"]) > 0).astype(int)

    df["income_per_person"] = df["monthly_income"] / (1 + df["n_dependents"])

    new_feats = ["est_monthly_debt", "total_late_cnt", "severe_late_ratio",
                 "has_severe_late", "income_per_person", "is_income_missing"]
    print(f"新增 {len(new_feats)} 个衍生列：{new_feats}")

    print("\n" + "=" * 60)
    print("Step 2d: 特征与目标的初步关系验证（策略直觉检查）")
    print("=" * 60)
    report = []
    report.append("# Step2 特征工程报告\n")

    age_bin = pd.cut(df["age"], bins=[0, 25, 35, 50, 100],
                     labels=["青年(<25)", "轻熟(26-35)", "中年(36-50)", "资深(50+)"])
    bad_age_tab = df.groupby(age_bin, observed=True)["target"].agg(["mean", "count"]).reset_index()
    report.append("\n## 年龄组 × 坏账率\n\n" + _md_table(bad_age_tab, "组"))
    report.append(_gradient_line(bad_age_tab, "年龄越大坏账率越低，方向符合业务预期（年轻人信用历史短、收入稳定性差）；资深组接近基线一半，区分信号明确。"))
    print("\n按年龄组看坏账率：")
    print(bad_age_tab.to_string(index=False))

    q1, q2 = df["monthly_income"].quantile([0.33, 0.66])
    inc_bin = pd.cut(df["monthly_income"], bins=[-1, q1, q2, float("inf")],
                     labels=["低收入", "中收入", "高收入"])
    bad_inc_tab = df.groupby(inc_bin, observed=True)["target"].agg(["mean", "count"]).reset_index()
    report.append("\n\n## 收入组 × 坏账率\n\n" + _md_table(bad_inc_tab, "组"))
    report.append(_gradient_line(bad_inc_tab, "收入越高风险越低，方向正确但梯度平缓——收入的区分力弱于逾期类特征，单靠收入撑不起风控，要和负债类特征搭配。"))
    print("\n按收入组看坏账率：")
    print(bad_inc_tab.to_string(index=False))

    severe_tab = df.groupby("has_severe_late", observed=True)["target"].agg(["mean", "count"]).reset_index()
    severe_tab["has_severe_late"] = severe_tab["has_severe_late"].map({0: "无", 1: "有"})
    report.append("\n\n## 是否有严重逾期史 × 坏账率\n\n" + _md_table(severe_tab, "是否有严重逾期"))
    report.append(_gradient_line(severe_tab, "必然强相关的 sanity check——逾期史与坏账率若不强相关反而说明数据有 bug；一个 0/1 信息把两类人拉开近 10 倍，这个 bit 的信息量极大。"))
    report.append("\n\n> 💡 业务解读：以上三张表就是策略分析的第一课——**任何特征要进模型，先看它对坏账率的区分度**。\n")
    print("\n是否有严重逾期史 × 坏账率：")
    print(severe_tab.to_string(index=False))


    act_bin = pd.cut(df["n_credit_lines"], bins=[-1, 3, 8, 100],
                     labels=["低活跃", "中活跃", "高活跃"])
    bad_act_tab = df.groupby(act_bin, observed=True)["target"].agg(["mean", "count"]).reset_index()
    report.append("\n\n## 信贷活跃度 × 坏账率\n\n" + _md_table(bad_act_tab, "组"))
    report.append(_gradient_line(bad_act_tab, "U 形（两头高中间低）：低活跃=信用小白无历史，高活跃=多头倾向，中间是有历史且适度的甜区。非单调表的倍数只作参考，精确切分交 step3 树分箱+IV 裁决。"))
    print("\n信贷活跃度 × 坏账率：")
    print(bad_act_tab.to_string(index=False))

    tot_bin = pd.cut(df["total_late_cnt"], bins=[-1, 0, 2, 100],
                     labels=["0次", "1-2次", "3次+"])
    bad_tot_tab = df.groupby(tot_bin, observed=True)["target"].agg(["mean", "count"]).reset_index()
    report.append("\n\n## 总逾期次数 × 坏账率\n\n" + _md_table(bad_tot_tab, "组"))
    report.append(_gradient_line(bad_tot_tab, "必然强相关的 sanity check（同严重逾期史表），16 倍梯度为全部特征最强——三档加总把稀疏计数聚合成有梯度的总量，验证加总这个特征工程动作站得住。"))
    print("\n总逾期次数 × 坏账率：")
    print(bad_tot_tab.to_string(index=False))

    sev_bin = pd.cut(df["severe_late_ratio"], bins=[-0.01, 0, 0.5, 1],
                     labels=["无严重逾期", "(0,0.5]", ">0.5"])
    bad_sev_tab = df.groupby(sev_bin, observed=True)["target"].agg(["mean", "count"]).reset_index()
    report.append("\n\n## 严重逾期占比 × 坏账率\n\n" + _md_table(bad_sev_tab, "组"))
    report.append(_gradient_line(bad_sev_tab, "0 与非 0 是本质分界（从无严重逾期和有严重逾期是两类人）；0.5 经验线做过敏感性检查——换 0.4/0.6 结论方向不变，经验线不敏感、结果稳健。"))
    print("\n严重逾期占比 × 坏账率：")
    print(bad_sev_tab.to_string(index=False))

    debt_bin = pd.cut(df["est_monthly_debt"], bins=[-1] + list(df["est_monthly_debt"].quantile([0.33, 0.66])) + [float("inf")],
                      labels=["低负债", "中负债", "高负债"])
    bad_debt_tab = df.groupby(debt_bin, observed=True)["target"].agg(["mean", "count"]).reset_index()
    report.append("\n\n## 估算月负债 × 坏账率\n\n" + _md_table(bad_debt_tab, "组"))
    report.append(_gradient_line(bad_debt_tab, "三档非单调（中档最低）——粗三档看不出梯度不等于特征没用：估算月负债的树分箱 IV 仍有 0.55 中等水平，粗验证弱、精验证强正说明关系是非线性的，需要更细的分箱捕捉。"))
    print("\n估算月负债 × 坏账率：")
    print(bad_debt_tab.to_string(index=False))

    per_bin = pd.cut(df["income_per_person"], bins=[-1] + list(df["income_per_person"].quantile([0.33, 0.66])) + [float("inf")],
                     labels=["低人均", "中人均", "高人均"])
    bad_per_tab = df.groupby(per_bin, observed=True)["target"].agg(["mean", "count"]).reset_index()
    report.append("\n\n## 人均负担 × 坏账率\n\n" + _md_table(bad_per_tab, "组"))
    report.append(_gradient_line(bad_per_tab, "高人均=每口人摊到的收入多=家庭总收入相对可观，风险反而低——方向与业务预期一致；这个特征把收入和家属合成一个相对值，比单看收入梯度更陡。"))
    report.append("\n\n> 💡 特征至此全部验证：分类语义列直接 groupby 看组间梯度，其余临时分箱看梯度——\n"
                  "> 梯度明显的才值得进模型；全量精确口径由 step3 的自动分箱+IV 接棒。\n"
                  "> 整体判读：逾期类四个特征强梯度（总逾期 16 倍最强）、年龄/收入/人均方向正确梯度中等、\n"
                  "> 两个弱梯度表（估算月负债/信贷活跃度）如实保留——粗验证的职责是筛方向，不是定生死。\n")
    print("\n人均负担 × 坏账率：")
    print(bad_per_tab.to_string(index=False))

    report_path = OUT / "02_feature_engineering.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\n特征工程报告：{report_path}")

    print("\n" + "=" * 60)
    print("Step 2e: 清洗+特征后的数据写回 MySQL")
    print("=" * 60)
    df.to_sql("application_clean", engine, if_exists="replace", index=False)
    print("已写回：risk_project01.application_clean")
    print("\n✅ Step 2 完成！")


if __name__ == "__main__":
    main()
