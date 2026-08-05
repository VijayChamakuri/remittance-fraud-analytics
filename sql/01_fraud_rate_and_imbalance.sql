-- 01_fraud_rate_and_imbalance.sql  (BigQuery standard SQL)
-- Overall fraud rate + class imbalance, the way you'd compute it in the warehouse.
-- Table assumed: `project.fraud.transactions` with columns matching the pipeline
-- (Class INT64 where 1=fraud, Amount FLOAT64, Time INT64 seconds, plus entity cols).

SELECT
  COUNT(*)                                   AS n_txns,
  SUM(Class)                                 AS n_fraud,
  COUNTIF(Class = 0)                         AS n_legit,
  SAFE_DIVIDE(SUM(Class), COUNT(*))          AS fraud_rate,
  SAFE_DIVIDE(COUNTIF(Class = 0), SUM(Class)) AS imbalance_ratio,   -- legit per 1 fraud
  ROUND(SUM(IF(Class = 1, Amount, 0)), 2)    AS fraud_dollars,
  ROUND(SAFE_DIVIDE(SUM(IF(Class=1,Amount,0)), SUM(Amount)) * 100, 3) AS pct_dollars_fraud
FROM `project.fraud.transactions`;
