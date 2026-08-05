-- 04_model_monitoring.sql  (BigQuery standard SQL)
-- Daily production monitoring: score distribution drift + precision/recall over
-- time from a scored-predictions table joined to eventual fraud labels.
-- Assumed table: `project.fraud.scored` (txn_id, score_date, model_score,
-- decision INT64 [1=flagged], label INT64 [1=confirmed fraud, may arrive later]).

WITH daily AS (
  SELECT
    score_date,
    COUNT(*)                                           AS n_scored,
    AVG(model_score)                                   AS avg_score,
    APPROX_QUANTILES(model_score, 100)[OFFSET(95)]     AS p95_score,
    COUNTIF(decision = 1)                              AS n_flagged,
    COUNTIF(decision = 1 AND label = 1)                AS tp,
    COUNTIF(decision = 1 AND label = 0)                AS fp,
    COUNTIF(decision = 0 AND label = 1)                AS fn
  FROM `project.fraud.scored`
  WHERE label IS NOT NULL                              -- matured labels only
  GROUP BY score_date
)
SELECT
  score_date,
  n_scored,
  ROUND(avg_score, 4)                                  AS avg_score,
  ROUND(p95_score, 4)                                  AS p95_score,
  n_flagged,
  SAFE_DIVIDE(tp, tp + fp)                             AS precision,
  SAFE_DIVIDE(tp, tp + fn)                             AS recall,
  -- population stability index vs a fixed baseline avg would be added here;
  -- simple drift proxy: day-over-day shift in average score
  ROUND(avg_score - LAG(avg_score) OVER (ORDER BY score_date), 4) AS avg_score_dod_shift
FROM daily
ORDER BY score_date DESC
LIMIT 90;
