{{ config(materialized='table') }}

SELECT
    run_id,
    dataset,
    composite_score,
    tier,
    created_at
FROM monitoring.semantic_scores
WHERE tier = 'clean'
