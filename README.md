# Project 01 · 个人信贷评分卡建模与策略应用

> 环境：Python 3.11 + MySQL 8.0 | 数据：[GiveMeSomeCredit](https://www.kaggle.com/datasets/uciml/give-me-some-credit-future-pred)（15万样本，Kaggle经典信贷风控赛题）

## 📌 一句话介绍

基于 15 万客户申请数据，完成从**数据入库（MySQL）→ 数据体检 → 特征工程 → WOE分箱/IV筛选 → 逻辑回归评分卡（测试集 KS 0.569 / AUC 0.860）→ 评分切档 → 通过率-坏账率权衡分析 → 决策树规则挖掘**的完整风控链路，并延伸到**策略层应用**（切档策略、拒绝线决策依据）——覆盖风控策略岗"建模+策略"双核心能力。

## 🧹 数据清洗亮点（GiveMeSomeCredit 的经典坑）

- **逾期计数哨兵值**：`n_loans_90d` 等字段存在 96/98 采集异常哨兵值，将 ≥96 的取值截断（原始计数不可能达到 98 次）；
- **额度使用率封顶**：`revolving_utilization > 1` 全量截断至 1（业务上使用率上限 100%，超 1 均为口径错误而非真实风险）；
- **收入缺失双层处理**：19.8% 缺失取中位数填充，同时生成 `is_income_missing` 标记列——缺失本身就是风险信号；
- **派生特征**：`total_late_cnt`（三档逾期之和）、`severe_late_ratio`（严重逾期占比）、`est_monthly_debt`（估算月负债）。

## 📊 模型表现（测试集）

| 指标 | 训练集 | 测试集 | 行业标准 |
|---|---|---|---|
| **KS** | 0.564 | **0.569** | >0.3 可用，>0.4 良好 |
| **AUC** | 0.859 | **0.860** | >0.7 可用，>0.75 良好 |



## 🎯 策略层产出（差异化亮点）

**评分切档（600基准分/PDO=50）：**

| 档位 | 分数区间 | 占比 | 坏账率 |
|---|---|---|---|
| A档(优质) | ≥640 | 66.4% | 1.53% |
| B档(通过) | 580-639 | 16.0% | 6.91% |
| C档(观察) | 520-579 | 8.0% | 12.20% |
| D档(拒绝) | <520 | 9.6% | 37.53% |

**通过率-坏账率权衡表**：给出 520-640 共 7 个分数线的完整权衡分析，每档附业务含义（激进增长/均衡/偏紧/过紧）。

**决策树规则挖掘**：最强的单条规则——`额度使用率>67% 且 有过逾期` 的人群坏账率显著高于均值，可直接翻译为上线规则。

## 🗂 项目结构

```
credit-scorecard/
├── sql/
│   └── 01_create_database.sql      # 建库建表（业务语义命名字段）
├── src/
│   ├── step1_load_and_check.py     # CSV→MySQL + 数据质量体检
│   ├── step2_clean_and_features.py # 清洗决策 + 8个业务特征工程
│   ├── step3_woe_iv.py             # WOE分箱 + IV筛选 + Excel交付
│   └── step4_model_scorecard.py    # LR建模 + 评分卡 + 策略应用
├── outputs/
│   ├── 01_data_quality_report.md   # 数据体检报告（坏账率6.68%、缺失值、异常值）
│   ├── 02_feature_engineering.md   # 特征×坏账率交叉验证（年龄/收入/逾期史）
│   ├── 03_woe_details.xlsx         # WOE明细（16特征）
│   ├── 03_iv_summary.csv           # IV排序（top: 逾期史1.37/额度使用率1.08）
│   ├── 04_model_and_strategy.md    # 建模+策略完整报告
│   ├── 04_scorecard_output.xlsx    # 系数+切档+打分明细
│   └── charts/                     # 分布图+WOE趋势图
└── README.md
```

## 🔧 技术要点

1. **数据入库而非裸跑CSV**：全流程在 MySQL 中管理（application_train → application_clean → application_woe → application_scored 四张表，一条数据血缘链）——策略岗日常就是数据库工作
2. **每个清洗决策有业务理由**：月收入缺失19.8%→中位数填充+缺失标记列（缺失本身是信号）；额度使用率>100%→业务截断到1
3. **特征工程用"风控语言"**：严重逾期占比（区分偶尔忘还vs习惯性违约）、人均收入（负担系数）——不是无脑套模板
4. **WOE单调性强制**：mono_enforce保证每个特征的WOE随风险单调，这是评分卡可解释性的底线
5. **评分卡标准换算**：600基准分/PDO=50（行业标准参数）
6. **策略层延伸**：不止步于模型KS——把它转成切档策略和拒绝线建议，这才是策略岗要的闭环

## 📚 参考与借鉴

- 开源参考：[scorecardpipeline](https://github.com/itlubber/scorecardpipeline)（评分卡pipeline结构）、[skorecard](https://github.com/ing-bank/skorecard)（ING银行工程规范）
- 业务方法：西提泡泡《风控策略学前班》（vintage/迁徙率/监控体系）
- 数据：[GiveMeSomeCredit](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset)
- 开发方式：AI辅助编程工作流（Claude 协作完成代码框架与调试，业务决策与验证人工把关）
