# Project 01：个人信贷评分卡建模与策略应用
# 信贷评分卡项目 | 环境说明见 README.md
# ============================================
# 建库建表脚本：把 GiveMeSomeCredit 数据导入 MySQL
# 数据：cs-training.csv（15万样本，12特征+1目标变量）
# ============================================

--   字符集决定"怎么存"，collation 决定"怎么比"：WHERE name='abc' 是否区分大小写、
CREATE DATABASE IF NOT EXISTS risk_project01
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
USE risk_project01;

-- 【列的格式】列名  类型  [约束] COMMENT '说明'
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
-- COMMENT='...' → 表级注释：说明这张表整体是干什么的，和列级 COMMENT 同一道理

-- LOAD DATA LOCAL INFILE 'D:/Code/Agent_Projects/秋招/project01_信贷评分卡与审批策略设计/data/cs-training.csv'
-- FIELDS TERMINATED BY ','
--   → 字段分隔符：逗号（标准 CSV）。遇到 \t 分隔的文件换成 '\t'。
--   → 还可加 ENCLOSED BY '"'（字段被引号包裹时用，本数据不需要）。
--     SET 还能做清洗转换（如 SET monthly_income = NULLIF(monthly_income, '')）。
--   不变的骨架：LOAD DATA [LOCAL] INFILE '路径' INTO TABLE 表名
--     FIELDS TERMINATED BY '分隔符'  IGNORE n LINES  (列清单)  [SET 加工];
