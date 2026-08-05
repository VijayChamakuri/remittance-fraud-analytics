-- 03_velocity_and_entity_features.sql  (BigQuery standard SQL)
-- Leakage-safe velocity + entity-history features computed with window functions.
-- This is the SQL analog of the engineered features in src/features.py; a data
-- pipeline would materialize this as a feature table feeding the model.

WITH ordered AS (
  SELECT
    *,
    -- prior transactions for this card, ordered by time (excludes current row)
    COUNT(*)  OVER card_prior                               AS card_hist_txn_count,
    AVG(Amount) OVER card_prior                             AS card_hist_amount_mean,
    STDDEV_SAMP(Amount) OVER card_prior                     AS card_hist_amount_std,
    -- inter-transaction gap
    Time - LAG(Time) OVER (PARTITION BY card_id ORDER BY Time) AS txn_gap_s,
    -- trailing-1h and trailing-24h velocity via RANGE windows on seconds
    COUNT(*) OVER (
      PARTITION BY card_id ORDER BY Time
      RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW)         AS txn_count_1h,
    COUNT(*) OVER (
      PARTITION BY card_id ORDER BY Time
      RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW)        AS txn_count_24h
  FROM `project.fraud.transactions`
  WINDOW card_prior AS (
    PARTITION BY card_id ORDER BY Time
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
  )
)
SELECT
  card_id, Time, Amount, corridor, device_type, account_age_days, Class,
  IFNULL(txn_gap_s, 999999)                                 AS txn_gap_s,
  txn_count_1h,
  txn_count_24h,
  card_hist_txn_count,
  -- amount anomaly vs the card's own history (z-score)
  SAFE_DIVIDE(Amount - card_hist_amount_mean,
              NULLIF(card_hist_amount_std, 0))              AS amount_zscore_vs_card,
  -- structuring flag: just under the reporting/limit threshold
  IF(Amount >= 900 AND Amount < 1000, 1, 0)                AS near_threshold,
  -- takeover combo
  IF(device_change = 1 AND corridor_new = 1, 1, 0)         AS device_change_x_new_corridor
FROM ordered;
