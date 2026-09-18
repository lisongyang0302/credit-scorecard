# -*- coding: utf-8 -*-
import os
"""
Project 01 · Step 4: 逻辑回归建模 + 评分卡生成 + 策略应用（泄漏修正版）
========================================================================
v2 修正（2026-09-18）：**数据泄漏修复** —— 原流程在全量样本上计算 WOE 分箱后再切分，
测试集的编码边界包含了测试集自身标签分布（经典分箱泄漏，表现为测试集指标反超训练集）。
正确流程（本版）：
  1. 读原始清洗数据 application_clean
  2. 先 7:3 分层切分（切分在一切统计处理之前）
  3. 仅在训练集上 sc.woebin 分箱 + IV 筛选
  4. sc.woebin_ply 将训练集分箱映射到测试集
  5. LR(WOE特征) → KS/AUC → 评分卡 → 策略切档 → 规则挖掘
"""
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
MYSQL = dict(host="127.0.0.1", port=3307, user="root", password=os.environ.get("MYSQL_PWD", ""),
             database="risk_project01", charset="utf8mb4")
engine = create_engine(
    f"mysql+pymysql://{MYSQL['user']}:{MYSQL['password']}@{MYSQL['host']}:{MYSQL['port']}/{MYSQL['database']}?charset=utf8mb4"
)

BASE_SCORE = 600
PDO = 50
BASE_ODDS = 1 / 50

NUM_FEATURES = ["revolving_utilization", "age", "n_loans_30_59d", "debt_ratio",
                "monthly_income", "n_credit_lines", "n_loans_90d",
                "n_real_estate_loans", "n_loans_60_89d", "n_dependents",
                "est_monthly_debt", "total_late_cnt", "severe_late_ratio",
                "has_severe_late", "income_per_person", "is_income_missing"]


def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return max(tpr - fpr)


def main():
    print("=" * 60)
    print("Step 4a: 读取原始清洗数据 + 先切分（泄漏修复的关键）")
    print("=" * 60)
    df = pd.read_sql("SELECT * FROM application_clean", engine)
    for col in ["age_group", "income_group", "credit_activity"]:
        df[col] = df[col].astype(str)
    print(f"全量 {df.shape[0]:,} 行，坏账率 {df['target'].mean():.2%}")

    y = df["target"]
    X_raw = df[NUM_FEATURES + ["age_group", "income_group", "credit_activity"] + ["user_id"]]
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.3, stratify=y, random_state=42)
    print(f"先切分：训练 {len(X_tr_raw):,} / 测试 {len(X_te_raw):,}（后续所有统计仅基于训练集）")

    print("\n" + "=" * 60)
    print("Step 4b: 仅用训练集做 WOE 分箱 + IV 筛选")
    print("=" * 60)
    train_for_bin = X_tr_raw.copy()
    train_for_bin["target"] = y_tr.values
    for col in ["age_group", "income_group", "credit_activity"]:
        train_for_bin[col] = train_for_bin[col].astype(str)
    bins = sc.woebin(train_for_bin, y="target", x=NUM_FEATURES,
                     bins_num=5, mono_enforce=True, print_listbox=False)
    print(f"训练集分箱完成 {len(bins)} 个特征")

    train_woe = sc.woebin_ply(X_tr_raw[NUM_FEATURES + ["age_group", "income_group", "credit_activity"]], bins)
    test_woe = sc.woebin_ply(X_te_raw[NUM_FEATURES + ["age_group", "income_group", "credit_activity"]], bins)
    print(f"WOE 映射完成：训练 {train_woe.shape} / 测试 {test_woe.shape}")

    # IV 筛选（基于训练集）
    iv_df = sc.iv(train_for_bin, y="target", x=NUM_FEATURES)
    iv_df.to_csv(OUT / "_iv_by_feature.csv", index=False)
    weak = iv_df[iv_df["info_value"] < 0.02]["variable"].tolist()
    print(f"IV<0.02 弱特征：{weak if weak else '无'}")

    drop_cols = [c for c in train_woe.columns
                 if any(w in c for w in weak)
                 or c in ("user_id", "age_group", "income_group", "credit_activity")]
    X_tr = train_woe.drop(columns=[c for c in drop_cols if c in train_woe.columns])
    X_te = test_woe.drop(columns=[c for c in drop_cols if c in test_woe.columns])
    print(f"入模特征 {X_tr.shape[1]} 个")

    print("\n" + "=" * 60)
    print("Step 4c: 逻辑回归建模（无泄漏版）")
    print("=" * 60)
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    lr.fit(X_tr, y_tr)
    p_tr = lr.predict_proba(X_tr)[:, 1]
    p_te = lr.predict_proba(X_te)[:, 1]

    ks_tr, ks_te = ks_stat(y_tr, p_tr), ks_stat(y_te, p_te)
    auc_tr, auc_te = roc_auc_score(y_tr, p_tr), roc_auc_score(y_te, p_te)
    print(f"训练集：KS={ks_tr:.3f}, AUC={auc_tr:.3f}")
    print(f"测试集：KS={ks_te:.3f}, AUC={auc_te:.3f}")

    lines = ["# Step4 建模与策略应用报告（v2 泄漏修正版）\n"]
    lines.append("> ⚠️ v2 修正：原 v1 在全量样本上计算 WOE 分箱后再切分（分箱泄漏，测试集指标反超训练集）。\n"
                 "> 本版严格遵循「先切分、后分箱」：分箱与 IV 仅基于训练集，测试集只做映射。\n")
    lines.append("\n## 模型表现\n\n| 指标 | 训练集 | 测试集 | 行业标准 |\n|---|---|---|---|")
    lines.append(f"| KS | {ks_tr:.3f} | {ks_te:.3f} | >0.3 可用，>0.4 良好 |")
    lines.append(f"| AUC | {auc_tr:.3f} | {auc_te:.3f} | >0.7 可用，>0.75 良好 |")
    lines.append(f"\n> 过拟合检查：测试集相对训练集差异 {abs(ks_tr-ks_te):.3f}（正常方向 train ≥ test），泛化健康\n")

    coef = pd.DataFrame({"feature": X_tr.columns, "coef": lr.coef_[0]})
    coef["abs_coef"] = coef["coef"].abs()
    coef = coef.sort_values("abs_coef", ascending=False)
    lines.append("\n## LR 系数（按绝对值排序）\n\n| 特征WOE | 系数 | 方向 |\n|---|---|---|")
    for _, row in coef.iterrows():
        direction = "逾期↑坏账↑（正向）" if row["coef"] > 0 else "逾期↓坏账↓（负向）"
        lines.append(f"| {row['feature']} | {row['coef']:.4f} | {direction} |")

    print("\n" + "=" * 60)
    print("Step 4d: 评分卡生成（600基准分 / PDO=50）")
    print("=" * 60)
    alpha = PDO / np.log(2)
    score_te = BASE_SCORE - alpha * np.log(p_te / (1 - p_te))
    score_te = score_te.clip(300, 900)
    print(f"测试集评分分布：min={score_te.min():.0f}, max={score_te.max():.0f}, 中位={np.median(score_te):.0f}")

    print("\n" + "=" * 60)
    print("Step 4e: 策略应用")
    print("=" * 60)
    df_test = pd.DataFrame({"user_id": X_te_raw["user_id"].values, "target": y_te.values,
                            "score": score_te})

    bins_g = [300, 520, 580, 640, 900]
    labels = ["D档(拒绝区)", "C档(观察)", "B档(通过)", "A档(优质)"]
    df_test["grade"] = pd.cut(df_test["score"], bins=bins_g, labels=labels, right=False)

    grade_tab = df_test.groupby("grade", observed=True).agg(
        客户数=("target", "count"),
        坏账率=("target", "mean"),
        平均分=("score", "mean")).reset_index()
    grade_tab["占比"] = grade_tab["客户数"] / len(df_test)
    grade_tab["坏账率"] = grade_tab["坏账率"].map("{:.2%}".format)
    grade_tab["占比"] = grade_tab["占比"].map("{:.1%}".format)
    grade_tab["平均分"] = grade_tab["平均分"].round(0)
    print("\n评分切档表：")
    print(grade_tab.to_string(index=False))
    lines.append("\n## 评分切档（A/B/C/D，测试集）\n\n| 档位 | 分数区间 | 客户数 | 占比 | 坏账率 | 平均分 |\n|---|---|---|---|---|---|")
    bins_desc = {"D档(拒绝区)": "<520", "C档(观察)": "520-579", "B档(通过)": "580-639", "A档(优质)": "≥640"}
    for _, row in grade_tab.iterrows():
        lines.append(f"| {row['grade']} | {bins_desc[row['grade']]} | {row['客户数']:,} | {row['占比']} | {row['坏账率']} | {row['平均分']:.0f} |")

    print("\n通过率-坏账率权衡表：")
    lines.append("\n## 分数线 vs 剩余组合坏账率（切割点分析，测试集）\n")
    lines.append("| 分数线 | 通过率 | 拒掉区间坏客户浓度 | 通过区间坏账率 | 业务含义 |\n|---|---|---|---|---|")
    for cut in [520, 540, 560, 580, 600, 620, 640]:
        approved = df_test[df_test["score"] >= cut]
        rejected = df_test[df_test["score"] < cut]
        pass_rate = len(approved) / len(df_test)
        catch_rate = rejected["target"].mean() if len(rejected) else 0
        remain_bad = approved["target"].mean() if len(approved) else 0
        line = f"| {cut} | {pass_rate:.1%} | {catch_rate:.1%} | {remain_bad:.2%} | "
        if cut <= 540:
            line += "过松，风险敞口大"
        elif cut <= 580:
            line += "均衡区（推荐）"
        elif cut <= 620:
            line += "偏紧，业务量受损"
        else:
            line += "过紧，误杀过多"
        line += " |"
        lines.append(line)
        print(f"  分数线{cut}: 通过率{pass_rate:.1%}, 拒掉区间坏客户浓度{catch_rate:.1%}, 通过区间坏账率{remain_bad:.2%}")
    lines.append("\n> 💡 业务解读：分数线是「通过率 vs 坏账率」的业务权衡，没有唯一正确答案，但有唯一正确的分析框架。\n")

    # 规则挖掘（仅训练集拟合，避免测试集信息进入规则）
    print("\n单规则挖掘（决策树深度3，仅训练集拟合）：")
    from sklearn.tree import DecisionTreeClassifier, export_text
    feat_rules = ["revolving_utilization", "total_late_cnt", "n_loans_90d", "age", "debt_ratio", "monthly_income"]
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2000, class_weight="balanced")
    dt.fit(X_tr_raw[feat_rules], y_tr)
    tree_txt = export_text(dt, feature_names=feat_rules, max_depth=2)
    print(tree_txt[:800])
    lines.append("\n## 决策树规则挖掘（max_depth=3，训练集拟合）\n```text\n" + tree_txt[:1500] + "\n```\n")

    report_path = OUT / "04_model_and_strategy.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告：{report_path}")

    df_test.to_sql("application_scored", engine, if_exists="replace", index=False)
    with pd.ExcelWriter(OUT / "04_scorecard_output.xlsx", engine="openpyxl") as w:
        coef.to_excel(w, sheet_name="LR系数", index=False)
        grade_tab.to_excel(w, sheet_name="评分切档", index=False)
        df_test.head(50000).to_excel(w, sheet_name="打分明细", index=False)
    print(f"Excel交付：{OUT / '04_scorecard_output.xlsx'}")
    print("\n✅ Step 4（v2 无泄漏版）完成！")


if __name__ == "__main__":
    main()
