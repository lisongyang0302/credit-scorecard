# credit-scorecard · 信贷评分卡与审批策略设计

基于 Kaggle GiveMeSomeCredit 数据集（15 万条真实信贷申请记录）的评分卡全流程建模项目：数据入库与质量体检 → 清洗与特征工程 → WOE/IV 分箱 → 共线检验 → 逻辑回归评分卡 → 评分刻度与审批策略 → 防泄露审计实验。

## 数据

- 来源：Kaggle [GiveMeSomeCredit](https://www.kaggle.com/c/GiveMeSomeCredit)（cs-training.csv，150,000 行 × 12 列）
- 目标变量：SeriousDlqin2yrs（未来 2 年内是否逾期 90 天以上），坏账率 6.68%，不平衡比约 14:1
- 存储：MySQL 8.0（库 `risk_project01`），数据文件放 `data/`（不入库，见 .gitignore）

## 建模链路（src/）

| 脚本 | 步骤 |
|---|---|
| step1_load_and_check.py | CSV → MySQL 落库（读回验证），输出数据质量体检报告 |
| step2_clean_and_features.py | 缺失/异常清洗 + 业务特征构造 |
| step3_split_and_bin.py | 分层切分 + train 口径 WOE/IV 分箱 |
| step4_collinearity.py | 相关性 + VIF 双防线共线检验 |
| step5_lr_model.py | 逻辑回归（WOE 值入模）+ KS/AUC 评估 |
| step6_score_and_strategy.py | 评分刻度（基准分 600）+ A/B/C/D 切档 + 审批策略 |
| step7_audit_experiments.py | 置换检验 / 多种子重切 / GBM 对照 / 单变量排查 |

## 关键结果

| 指标 | 数值 |
|---|---|
| 测试集 KS / AUC | 0.569 / 0.860（训练集 0.564 / 0.859） |
| 特征预测力（IV，train 口径） | total_late_cnt 1.34、revolving_utilization 1.07 |
| 入模特征 | 13 个，系数方向全部符合业务直觉 |
| 评分切档坏账率 | A 档 1.52% 逐级抬升至 D 档 37.37% |
| 审批策略 | 拒批 D 档，组合坏账率 6.68% → 3.43% |

评分刻度采用行业标准的基准分 600 设计，分数越高风险越低；A/B/C/D 四档梯度清晰，D 档为明确的风险集中区。

数据修正与口径：269 条采集错误与缺失/异常在清洗环节处理；月收入 19.8% 缺失取中位数填充并加缺失标记列；评估全程以 KS/AUC 为准（不平衡场景下准确率失真）。

## 复现

```bash
# 1. MySQL 8.0 启动后执行建库脚本
mysql -u root -p < sql/01_create_database.sql

# 2. 数据文件放 data/（cs-training.csv），设置环境变量
export MYSQL_PWD="你的密码"

# 3. 依次执行
python src/step1_load_and_check.py
python src/step2_clean_and_features.py
python src/step3_split_and_bin.py
python src/step4_collinearity.py
python src/step5_lr_model.py
python src/step6_score_and_strategy.py
python src/step7_audit_experiments.py
```

依赖：Python 3.10+；pandas、numpy、sqlalchemy、pymysql、scorecardpy、scikit-learn、lightgbm、matplotlib、openpyxl。

## 产出

- `outputs/01_data_quality_report.md` 数据质量体检报告
- `outputs/02_feature_engineering.md` 特征工程报告
- `outputs/03_iv_summary.csv` / `03_woe_details.xlsx` 分箱与 IV 明细
- `outputs/04_model_and_strategy.md` 建模与策略应用报告
- `outputs/04_scorecard_output.xlsx` 打分明细与切档策略
- `outputs/charts/` 特征分布、WOE 趋势、相关性热力图、ROC/KS 曲线
