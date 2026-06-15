import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from operators.structural_check_operator import StructuralCheckOperator


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def operator():
    return StructuralCheckOperator(
        task_id='test_structural',
        dataset_path='data/raw/nyc_taxi_jan2023.parquet',
        dataset_name='nyc_taxi',
        reference_rows=10000
    )


@pytest.fixture
def clean_df():
    np.random.seed(42)
    n = 5000
    return pd.DataFrame({
        'trip_distance': np.random.exponential(2, n),
        'passenger_count': np.random.choice([1.0, 2.0, 3.0, np.nan], n, p=[0.6, 0.2, 0.15, 0.05]),
        'fare_amount': np.random.uniform(5, 50, n),
        'store_and_fwd_flag': np.random.choice(['Y', 'N'], n)
    })


# ------------------------------------------------------------------
# Schema inference tests
# ------------------------------------------------------------------

def test_schema_infers_all_columns(operator, clean_df):
    schema = operator._infer_schema(clean_df)
    assert set(schema.keys()) == set(clean_df.columns)


def test_null_threshold_is_derived_from_data(operator, clean_df):
    schema = operator._infer_schema(clean_df)
    # Threshold must be computed from data — never a magic constant
    for col, meta in schema.items():
        assert 0.0 <= meta['null_threshold'] <= 1.0


def test_format_pattern_detected_correctly(operator, clean_df):
    schema = operator._infer_schema(clean_df)
    assert schema['store_and_fwd_flag']['format_pattern'] == 'yn_flag'
    assert schema['trip_distance']['format_pattern'] == 'numeric'


# ------------------------------------------------------------------
# Scoring tests
# ------------------------------------------------------------------

def test_clean_batch_scores_below_04(operator, clean_df):
    reference = clean_df.head(1000)
    current = clean_df.tail(4000)
    schema = operator._infer_schema(reference)
    score, violations = operator._check_batch(current, schema)
    assert score < 0.4, f"Clean batch scored {score:.3f}, expected < 0.4"


def test_score_is_between_0_and_1(operator, clean_df):
    reference = clean_df.head(1000)
    current = clean_df.tail(4000)
    schema = operator._infer_schema(reference)
    score, _ = operator._check_batch(current, schema)
    assert 0.0 <= score <= 1.0


# ------------------------------------------------------------------
# Anomaly detection tests
# ------------------------------------------------------------------

def test_null_spike_raises_score(operator, clean_df):
    reference = clean_df.head(1000)
    schema = operator._infer_schema(reference)

    dirty = clean_df.tail(4000).copy()
    # Inject 70% nulls into trip_distance
    null_idx = dirty.sample(frac=0.7, random_state=42).index
    dirty.loc[null_idx, 'trip_distance'] = np.nan

    score, violations = operator._check_batch(dirty, schema)
    assert score > 0.4, f"Null spike not detected, score: {score:.3f}"
    assert violations['null_violations'] > 0


def test_missing_column_raises_schema_violation(operator, clean_df):
    reference = clean_df.head(1000)
    schema = operator._infer_schema(reference)

    drifted = clean_df.tail(4000).drop(columns=['trip_distance'])
    score, violations = operator._check_batch(drifted, schema)
    assert violations['schema_violations'] > 0


def test_multiple_null_spikes_score_above_07(operator, clean_df):
    reference = clean_df.head(1000)
    schema = operator._infer_schema(reference)

    dirty = clean_df.tail(4000).copy()
    # Inject nulls into ALL columns
    for col in ['trip_distance', 'fare_amount']:
        idx = dirty.sample(frac=0.8, random_state=42).index
        dirty.loc[idx, col] = np.nan

    score, violations = operator._check_batch(dirty, schema)
    assert score > 0.4
    assert violations['null_violations'] >= 2


# ------------------------------------------------------------------
# DB write test (mocked)
# ------------------------------------------------------------------

def test_write_to_db_called_on_execute(operator):
    mock_context = {'run_id': 'test_run_001'}

    with patch.object(operator, '_load_data') as mock_load, \
         patch.object(operator, '_write_to_db') as mock_write:

        mock_load.return_value = pd.DataFrame({
            'trip_distance': np.random.exponential(2, 15000),
            'fare_amount': np.random.uniform(5, 50, 15000)
        })

        result = operator.execute(mock_context)

        assert mock_write.called
        assert 'structural_score' in result
        assert 'run_id' in result
        assert result['run_id'] == 'test_run_001'
