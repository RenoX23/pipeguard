{{ config(materialized='table') }}

SELECT
    run_id,
    dataset,
    if_score,
    ks_psi_score,
    adwin_score,
    composite_score,
    tier,
    created_at
FROM monitoring.semantic_scores
WHERE tier = 'review'
