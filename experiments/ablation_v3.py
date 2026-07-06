import sys, os
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

DATASET_CONFIG = {
    'nyc_taxi':    {'path': 'data/raw/nyc_taxi_jan2023.parquet', 'null_col': 'trip_distance', 'shift_col': 'fare_amount'},
    'credit_card': {'path': 'data/raw/creditcard.csv',           'null_col': 'Amount',        'shift_col': 'V1'},
    'uci_census':  {'path': 'data/raw/uci_census.csv',           'null_col': 0,                'shift_col': 12},
    'smd':         {'path': 'data/raw/smd/test/machine-1-1.txt', 'null_col': 0,                'shift_col': 1}
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


def get_batches(config, seed):
    df = load_df(config['path'])
    reference   = df.iloc[:REFERENCE_ROWS]
    clean_batch = df.iloc[REFERENCE_ROWS:REFERENCE_ROWS+BATCH_SIZE].reset_index(drop=True)
    injector = AnomalyInjector()

    null_only, _  = injector.inject_null_spike(
        clean_batch.copy(), config['null_col'], null_pct=0.85, window_pct=0.80, seed=seed)

    shift_only, _ = injector.inject_distribution_shift(
        clean_batch.copy(), config['shift_col'], shift_factor=4.0, tail_pct=0.50, seed=seed)

    mixed, _ = injector.inject_null_spike(
        clean_batch.copy(), config['null_col'], null_pct=0.85, window_pct=0.80, seed=seed)
    mixed, _ = injector.inject_distribution_shift(
        mixed, config['shift_col'], shift_factor=4.0, tail_pct=0.50, seed=seed)

    return reference, clean_batch, null_only, shift_only, mixed


def score_batch(reference, batch, weights):
    op = StructuralCheckOperator.__new__(StructuralCheckOperator)
    op.log = MagicMock(); op.dataset_name = 'ablation_v3'
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

    composite = (weights['w_struct']*s_score + weights['w_if']*i_score + weights['w_ks']*k_score)
    return float(np.clip(composite, 0, 1))


if __name__ == '__main__':
    results = []
    combos = [(cfg, ds, s) for cfg in CONFIGS for ds in DATASET_CONFIG for s in SEEDS]

    for config_name, dataset_name, seed in tqdm(combos, desc='Ablation v3', unit='run'):
        try:
            ref, clean, null_only, shift_only, mixed = get_batches(DATASET_CONFIG[dataset_name], seed)
            weights = CONFIGS[config_name]

            clean_s  = score_batch(ref, clean, weights)
            null_s   = score_batch(ref, null_only, weights)
            shift_s  = score_batch(ref, shift_only, weights)
            mixed_s  = score_batch(ref, mixed, weights)

            results.append({
                'config': config_name, 'dataset': dataset_name, 'seed': seed,
                'clean_score':  round(clean_s, 4),
                'null_score':   round(null_s, 4),  'null_detected':  int(null_s  >= THRESHOLD),
                'shift_score':  round(shift_s, 4),  'shift_detected': int(shift_s >= THRESHOLD),
                'mixed_score':  round(mixed_s, 4),  'mixed_detected': int(mixed_s >= THRESHOLD),
                'false_alarm':  int(clean_s >= THRESHOLD)
            })
        except Exception as e:
            print(f"  ERROR {config_name}/{dataset_name}/s={seed}: {e}")

    df = pd.DataFrame(results)
    os.makedirs('experiments/results', exist_ok=True)
    df.to_csv('experiments/results/ablation_v3.csv', index=False)

    print("\n=== DETECTION RATE BY ANOMALY TYPE (mean across datasets+seeds) ===")
    summary = df.groupby('config')[['null_detected','shift_detected','mixed_detected','false_alarm']].mean()
    print(summary.round(3))
    print("\nSaved: experiments/results/ablation_v3.csv")
