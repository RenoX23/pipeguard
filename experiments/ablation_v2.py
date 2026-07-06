"""
PipeGuard Ablation v2 — Mixed Anomaly Injection
Each anomalous batch contains BOTH a null spike (Stage 1 / IF target)
AND a distribution shift (KS-PSI target), injected into different columns.
This tests whether each detector catches ITS OWN target anomaly type,
and whether only the full composite catches both simultaneously.
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock

class MockBaseOperator:
    def __init__(self, task_id=None, **kwargs):
        self.task_id = task_id
        self.log = MagicMock()

def mock_apply_defaults(func):
    return func

am = MagicMock()
am.models.BaseOperator = MockBaseOperator
am.utils.decorators.apply_defaults = mock_apply_defaults
sys.modules['airflow'] = am
sys.modules['airflow.models'] = am.models
sys.modules['airflow.utils'] = am.utils
sys.modules['airflow.utils.decorators'] = am.utils.decorators

import numpy as np
import pandas as pd
from tqdm import tqdm

from operators.structural_check_operator import StructuralCheckOperator
from detectors.isolation_forest import IsolationForestDetector
from detectors.ks_psi import KSPSIDetector
from experiments.anomaly_injector import AnomalyInjector

REFERENCE_ROWS = 10000
BATCH_SIZE     = 5000
SEEDS          = [42, 123, 456]
THRESHOLD      = 0.28

# null_col: target for Stage 1 / IF | shift_col: target for KS-PSI
DATASET_CONFIG = {
    'nyc_taxi': {
        'path': 'data/raw/nyc_taxi_jan2023.parquet',
        'null_col': 'trip_distance', 'shift_col': 'fare_amount'
    },
    'credit_card': {
        'path': 'data/raw/creditcard.csv',
        'null_col': 'Amount', 'shift_col': 'V1'
    },
    'uci_census': {
        'path': 'data/raw/uci_census.csv',
        'null_col': 0, 'shift_col': 12  # age, hours-per-week
    },
    'smd': {
        'path': 'data/raw/smd/test/machine-1-1.txt',
        'null_col': 0, 'shift_col': 1
    }
}

CONFIGS = {
    'stage1_only':    {'w_struct': 1.0, 'w_if': 0.0, 'w_ks': 0.0},
    'if_only':        {'w_struct': 0.0, 'w_if': 1.0, 'w_ks': 0.0},
    'ks_psi_only':    {'w_struct': 0.0, 'w_if': 0.0, 'w_ks': 1.0},
    'no_structural':  {'w_struct': 0.0, 'w_if': 0.6, 'w_ks': 0.4},
    'full_pipeguard': {'w_struct': 0.35,'w_if': 0.35,'w_ks': 0.20},
}


def load_df(path):
    if path.endswith('.parquet'): return pd.read_parquet(path)
    if path.endswith('.txt'):     return pd.read_csv(path, header=None)
    if 'census' in path:          return pd.read_csv(path, header=None, low_memory=False)
    return pd.read_csv(path, low_memory=False)


def get_mixed_batches(config, seed):
    df = load_df(config['path'])
    reference   = df.iloc[:REFERENCE_ROWS]
    clean_batch = df.iloc[REFERENCE_ROWS:REFERENCE_ROWS+BATCH_SIZE].reset_index(drop=True)

    injector = AnomalyInjector()
    # Step 1: null spike on null_col
    mixed, _ = injector.inject_null_spike(
        clean_batch.copy(), config['null_col'],
        null_pct=0.85, window_pct=0.80, seed=seed)
    # Step 2: distribution shift on shift_col (applied on top)
    mixed, _ = injector.inject_distribution_shift(
        mixed, config['shift_col'], shift_factor=4.0, tail_pct=0.50, seed=seed)

    return reference, clean_batch, mixed


def score_batch(reference, batch, weights):
    op = StructuralCheckOperator.__new__(StructuralCheckOperator)
    op.log = MagicMock(); op.dataset_name = 'ablation_v2'
    schema = op._infer_schema(reference)
    s_score, _ = op._check_batch(batch, schema)

    num_cols = batch.select_dtypes(include=[np.number]).columns.tolist()
    X_ref = reference[num_cols].fillna(0).values
    X_cur = batch[num_cols].fillna(0).values

    ifd = IsolationForestDetector(contamination=0.10, random_state=42)
    ifd.fit(X_ref)
    i_score = ifd.score(X_cur)

    ksd = KSPSIDetector()
    ksd.fit(reference[num_cols])
    k_score = ksd.score(batch[num_cols])

    composite = (weights['w_struct']*s_score +
                 weights['w_if']*i_score +
                 weights['w_ks']*k_score)
    return float(np.clip(composite, 0, 1))


if __name__ == '__main__':
    results = []
    combos = [(cfg, ds, s) for cfg in CONFIGS
              for ds in DATASET_CONFIG for s in SEEDS]

    for config_name, dataset_name, seed in tqdm(combos, desc='Ablation v2', unit='run'):
        try:
            ref, clean, mixed = get_mixed_batches(DATASET_CONFIG[dataset_name], seed)
            weights = CONFIGS[config_name]
            clean_score = score_batch(ref, clean, weights)
            anom_score  = score_batch(ref, mixed, weights)
            results.append({
                'config': config_name, 'dataset': dataset_name, 'seed': seed,
                'clean_score': round(clean_score, 4),
                'anom_score':  round(anom_score, 4),
                'detected':    int(anom_score  >= THRESHOLD),
                'false_alarm': int(clean_score >= THRESHOLD)
            })
        except Exception as e:
            print(f"  ERROR {config_name}/{dataset_name}/s={seed}: {e}")

    df = pd.DataFrame(results)
    print("\n=== MIXED-INJECTION ABLATION SUMMARY ===")
    summary = df.groupby(['config','dataset'])[['clean_score','anom_score']].agg(['mean','std'])
    print(summary)

    os.makedirs('experiments/results', exist_ok=True)
    df.to_csv('experiments/results/ablation_v2.csv', index=False)
    print("\nSaved: experiments/results/ablation_v2.csv")
