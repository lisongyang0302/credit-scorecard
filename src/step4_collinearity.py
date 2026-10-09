import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
INT = OUT / "intermediate"


def main():
    print("=" * 60)
    print("Step 4: 共线检查（相关矩阵 + VIF，入模前的最后防线）")
    print("=" * 60)
    X_tr = pd.read_csv(INT / "X_tr_after_iv.csv")
    X_te = pd.read_csv(INT / "X_te_after_iv.csv")
    iv_df = pd.read_csv(INT / "iv_raw.csv")
    iv_map = dict(zip(iv_df["variable"], iv_df["info_value"]))

    corr_abs = X_tr.corr().abs()
    corr_m = X_tr.corr()
    disp_names = [c.replace("_woe", "") for c in corr_m.columns]
    corr_m.index = disp_names
    corr_m.columns = disp_names
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_m.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_m)))
    ax.set_xticklabels(corr_m.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr_m)))
    ax.set_yticklabels(corr_m.index, fontsize=8)
    for ii in range(len(corr_m)):
        for jj in range(len(corr_m)):
            v = corr_m.iloc[ii, jj]
            if ii != jj and abs(v) > 0.8:
                ax.text(jj, ii, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("入模特征相关矩阵热力图（训练集口径，|corr|>0.8 加粗标出=共线红线）")
    fig.tight_layout()
    (OUT / "charts").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "charts" / "09_corr_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"共线检查热力图已保存：charts/09_corr_heatmap.png")
    drop_by_corr = set()
    for i, j in combinations(range(len(corr_abs)), 2):
        c1, c2 = corr_abs.index[i], corr_abs.columns[j]
        if c1 in drop_by_corr or c2 in drop_by_corr or corr_abs.iloc[i, j] <= 0.8:
            continue
        f1, f2 = c1.replace("_woe", ""), c2.replace("_woe", "")
        looser = f1 if iv_map.get(f1, 0) < iv_map.get(f2, 0) else f2
        drop_by_corr.add(looser + "_woe")
        print(f"  共线对 |corr|={corr_abs.iloc[i, j]:.3f}: {c1} vs {c2} -> 剔 IV 低者 {looser}")
    if drop_by_corr:
        X_tr = X_tr.drop(columns=[c for c in drop_by_corr if c in X_tr.columns])
        X_te = X_te.drop(columns=[c for c in drop_by_corr if c in X_te.columns])
    def vif_prune(X, iv_map, max_vif=10.0):
        cols = list(X.columns); dropped = []
        while len(cols) >= 3:
            vifs = {}
            for c in cols:
                others = [x for x in cols if x != c]
                r2 = LinearRegression().fit(X[others], X[c]).score(X[others], X[c])
                vifs[c] = 1 / max(1 - r2, 1e-9)
            over = [c for c in cols if vifs[c] > max_vif]
            if not over:
                break
            victim = min(over, key=lambda c: iv_map.get(c.replace("_woe", ""), 0))
            dropped.append(f"{victim}(VIF={vifs[victim]:.1f})")
            cols.remove(victim)
            X = X.drop(columns=[victim])
        return X, dropped
    X_tr, vif_dropped = vif_prune(X_tr, iv_map)
    X_te = X_te[X_tr.columns]
    if vif_dropped:
        print(f"  VIF 迭代剔除（>10，组内保 IV 高者）：{vif_dropped}")
    corr_check_summary = ("|corr|>0.8 剔除 " + str(len(drop_by_corr)) + " 个：" +
                          (str(sorted(drop_by_corr)) if drop_by_corr else "无") + "；" +
                          "VIF>10 剔除 " + str(len(vif_dropped)) + " 个：" +
                          (str(vif_dropped) if vif_dropped else "无"))
    print(f"共线检查完成 -> 入模特征 {X_tr.shape[1]} 个")

    X_tr.to_csv(INT / "X_tr_final.csv", index=False)
    X_te.to_csv(INT / "X_te_final.csv", index=False)
    (INT / "collinearity_summary.txt").write_text(corr_check_summary, encoding="utf-8")
    print(f"\n✅ Step 4 完成！共线检查（入模特征 {X_tr.shape[1]} 个）")


if __name__ == "__main__":
    main()
