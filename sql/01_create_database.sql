# Project 01：个人信贷评分卡建模与策略应用
# 信贷评分卡项目 | 环境说明见 README.md
# ============================================
# 建库建表脚本：把 GiveMeSomeCredit 数据导入 MySQL
# 数据：cs-training.csv（15万样本，12特征+1目标变量）
# ============================================

-- 建库（数据库名与数据集对应，避免和 sql_learning01 混淆）
CREATE DATABASE IF NOT EXISTS risk_project01
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
USE risk_project01;

-- 建表：按业务语义命名（不是 Kaggle 的 c 开头缩写名，策略岗风格）
-- 字段名映射见项目 README 的「字段字典」章节
CREATE TABLE IF NOT EXISTS application_train (
    user_id             INT PRIMARY KEY COMMENT '客户主键（数据集自带索引列）',
    target              TINYINT NOT NULL COMMENT '目标变量：1=逾期90天以上，0=正常',
    revolving_utilization DECIMAL(10,5) COMMENT '信用卡额度使用率（RevolvingUtilizationOfUnsecuredLines）',
    age                 INT COMMENT '借款人年龄',
    n_loans_30_59d      INT COMMENT '30-59天逾期笔数（NumberOfTime30-59DaysPastDueNotWorse）',
    debt_ratio          DECIMAL(10,5) COMMENT '负债率（DebtRatio，月负债支出/月收入）',
    monthly_income      DECIMAL(12,2) COMMENT '月收入',
    n_credit_lines      INT COMMENT '开放式信贷与贷款数量（NumberOfOpenCreditLinesAndLoans）',
    n_loans_90d         INT COMMENT '90天逾期笔数（NumberOfTimes90DaysLate）',
    n_real_estate_loans INT COMMENT '不动产与担保贷款笔数（NumberRealEstateLoansOrLines）',
    n_loans_60_89d      INT COMMENT '60-89天逾期笔数（NumberOfTime60-89DaysPastDueNotWorse）',
    n_dependents        INT COMMENT '家属人数（NumberOfDependents）'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户申请与逾期表现表（GiveMeSomeCredit训练集）';

-- 数据加载：在命令行执行（路径按实际调整）
-- LOAD DATA LOCAL INFILE 'D:/Code/Agent_Projects/project02/data/cs-training.csv'
-- INTO TABLE application_train
-- FIELDS TERMINATED BY ','
-- IGNORE 1 LINES
-- (@row, target, revolving_utilization, age, n_loans_30_59d, debt_ratio,
--  monthly_income, n_credit_lines, n_loans_90d, n_real_estate_loans, n_loans_60_89d, n_dependents)
-- SET user_id = @row;
