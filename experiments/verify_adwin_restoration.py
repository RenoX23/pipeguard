import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock
class MockBaseOperator:
    def __init__(self, task_id=None, **kwargs):
        self.task_id = task_id; self.log = MagicMock()
def mock_apply_defaults(func): return func
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
from detectors.adwin_detector import ADWINDetector
from experiments.anomaly_injector import AnomalyInjector

REFERENCE_ROWS = 10000
BATCH_SIZE     = 5000
SEEDS          = [42, 123, 456]
THRESHOLD      = 0.28

DATASET_CONFIG = {
    'nyc_taxi':    {'path': 'data/raw/nyc_taxi_jan2023.parquet', 'shift_col': 'fare_amount'},
    'credit_card': {'path': 'data/raw/creditcard.csv',           'shift_col': 'V1'},
    'uci_census':  {'path': 'data/raw/uci_census.csv',           'shift_col': 12},
    'smd':         {'path': 'data/raw/smd/test/machine-1-1.txt', 'shift_col': 1}
}

def load_df(path):
    if path.endswith('.parquet'): return pd.read_parquet(path)
    if path.endswith('.txt'):     return pd.read_csv(path, header=None)
    if 'census' in path:          return pd.read_csv(path, header=None, low_memory=False)
    return pd.read_csv(path, low_memory=False)

def score_with_adwin(reference, batch, shift_col):
    op = StructuralCheckOperator.__new__(StructuralCheckOperator)
    op.log = MagicMock(); op.dataset_name = 'adwin_check'
    schema = op._infer_schema(reference)
    s_score, _ = op._check_batch(batch, schema)

    num_cols = batch.select_dtypes(include=[np.number]).columns.tolist()
    ifd = IsolationForestDetector(contamination=0.10, random_state=42)
    ifd.fit(reference[num_cols].fillna(0).values)
    i_score = ifd.score(batch[num_cols].fillna(0).values)

    ksd = KSPSIDetector(); ksd.fit(reference[num_cols])
    k_score = ksd.score(batch[num_cols])

    adwin = ADWINDetector(delta=0.002)
    a_score = adwin.detect(batch[shift_col].dropna()) if shift_col in batch.columns else 0.0

    composite = 0.35*s_score + 0.35*i_score + 0.20*k_score + 0.10*a_score
    return float(np.clip(composite, 0, 1)), a_score

results = []
for dataset_name, cfg in tqdm(DATASET_CONFIG.items(), desc='ADWIN restoration check'):
    df = load_df(cfg['path'])
    reference = df.iloc[:REFERENCE_ROWS]
    for seed in SEEDS:
        clean_batch = df.iloc[REFERENCE_ROWS:REFERENCE_ROWS+BATCH_SIZE].reset_index(drop=True)
        injector = AnomalyInjector()
        shift_batch, _ = injector.inject_distribution_shift(
            clean_batch.copy(), cfg['shift_col'], shift_factor=4.0, tail_pct=0.50, seed=seed)

        composite, adwin_flag = score_with_adwin(reference, shift_batch, cfg['shift_col'])
        results.append({'dataset': dataset_name, 'seed': seed,
                        'shift_score_with_adwin': round(composite, 4),
                        'adwin_fired': adwin_flag})

df_res = pd.DataFrame(results)
summary = df_res.groupby('dataset')['shift_score_with_adwin'].agg(['mean','std'])
print(summary.round(4))
print(f"\nThreshold: {THRESHOLD}")
print(f"ADWIN fire rate: {df_res['adwin_fired'].mean():.2f}")
