-- 02_fraud_by_segment.sql  (BigQuery standard SQL)
-- Fraud rate by amount band and by corridor/device -- the EDA segment view,
-- expressed as warehouse SQL (mirrors src/eda.py).

WITH banded AS (
  SELECT
    corridor,
    device_type,
    Class,
    Amount,
    CASE
      WHEN Amount < 10    THEN '0-10'
      WHEN Amount < 50    THEN '10-50'
      WHEN Amount < 100   THEN '50-100'
      WHEN Amount < 250   THEN '100-250'
      WHEN Amount < 500   THEN '250-500'
      WHEN Amount < 1000  THEN '500-1k'
      WHEN Amount < 5000  THEN '1k-5k'
      ELSE '5k+'
    END AS amount_band
  FROM `project.fraud.transactions`
)
SELECT
  amount_band,
  corridor,
  COUNT(*)                          AS n,
  SUM(Class)                        AS n_fraud,
  ROUND(AVG(Class) * 100, 3)        AS fraud_rate_pct,
  ROUND(SUM(IF(Class=1, Amount, 0)), 2) AS fraud_dollars
FROM banded
GROUP BY amount_band, corridor
HAVING n >= 50
ORDER BY fraud_rate_pct DESC
LIMIT 50;
