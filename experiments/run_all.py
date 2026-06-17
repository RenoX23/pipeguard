"""
PipeGuard Experiments — Batch-Level Evaluation
Unit of measurement: the batch, not the row.
PipeGuard and GE both evaluated on whether they correctly
classify a batch as CLEAN or ANOMALOUS.
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock
class _MockBase:
    def __init__(self, task_id=None, **kwargs): self.task_id=task_id; self.log=MagicMock()
sys.modules["airflow"]=MagicMock()
sys.modules["airflow.models"]=MagicMock()
sys.modules["airflow.models"].BaseOperator=_MockBase
sys.modules["airflow.utils"]=MagicMock()
sys.modules["airflow.utils.decorators"]=MagicMock()
sys.modules["airflow.utils.decorators"].apply_defaults=lambda f:f

import numpy as np
import pandas as pd
import great_expectations as ge
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

DATASETS = {
    'nyc_taxi': {
        'path': 'data/raw/nyc_taxi_jan2023.parquet',
        'inject': ('null_spike', 'trip_distance'),
        'streaming_col': 'trip_distance'
    },
    'credit_card': {
        'path': 'data/raw/creditcard.csv',
        'inject': ('null_spike', 'Amount'),
        'streaming_col': 'Amount'
    },
    'uci_census': {
        'path': 'data/raw/uci_census.csv',
        'inject': ('null_spike', 0),
        'streaming_col': None
    },
    'smd': {
        'path': 'data/raw/smd/test/machine-1-1.txt',
        'inject': ('null_spike', 0),
        'streaming_col': 0
    }
}


def load_df(path):
    if path.endswith('.parquet'): return pd.read_parquet(path)
    if path.endswith('.txt'):     return pd.read_csv(path, header=None)
    if 'census' in path:          return pd.read_csv(path, header=None, low_memory=False)
    return pd.read_csv(path, low_memory=False)


def get_batches(config, seed):
    """Returns reference, clean_batch, anomalous_batch."""
    df = load_df(config['path'])
    reference    = df.iloc[:REFERENCE_ROWS]
    clean_batch  = df.iloc[REFERENCE_ROWS:REFERENCE_ROWS + BATCH_SIZE].reset_index(drop=True)

    injector = AnomalyInjector()
    inj_type, col = config['inject']
    if inj_type == 'null_spike':
        anom_batch, _ = injector.inject_null_spike(
            clean_batch.copy(), col, null_pct=0.85, window_pct=0.80, seed=seed)
    else:
        anom_batch, _ = injector.inject_distribution_shift(
            clean_batch.copy(), col, shift_factor=4.0, tail_pct=0.50, seed=seed)

    return reference, clean_batch, anom_batch


def pipeguard_score(reference, batch):
    """Compute PipeGuard composite score for a batch."""
    # Stage 1 — structural
    op = StructuralCheckOperator.__new__(StructuralCheckOperator)
    op.log = MagicMock()
    op.dataset_name = 'exp'
    schema = op._infer_schema(reference)
    structural_score, _ = op._check_batch(batch, schema)

    # Stage 2 — IF
    num_cols = batch.select_dtypes(include=[np.number]).columns.tolist()
    X_ref = reference[num_cols].fillna(0).values
    X_cur = batch[num_cols].fillna(0).values

    if_det = IsolationForestDetector(contamination=0.05, random_state=42)
    if_det.fit(X_ref)
    if_score = if_det.score(X_cur)

    # Stage 2 — KS-PSI
    ks_det = KSPSIDetector()
    ks_det.fit(reference[num_cols])
    ks_score = ks_det.score(batch[num_cols])

    # Composite
    composite = (0.35 * structural_score +
                 0.35 * if_score +
                 0.20 * ks_score +
                 0.10 * 0.0)  # ADWIN skipped for speed
    return float(np.clip(composite, 0, 1))


def ge_detected(reference, batch):
    """
    Run Great Expectations with adaptive expectations from reference.
    Returns True if batch is flagged as anomalous.
    """
    ge_ref = ge.from_pandas(reference)
    num_cols = reference.select_dtypes(include=[np.number]).columns[:8].tolist()

    for col in num_cols:
        ref_col = reference[col].dropna()
        if len(ref_col) == 0:
            continue
        null_rate = reference[col].isnull().mean()
        # Adaptive null expectation
        ge_ref.expect_column_values_to_not_be_null(
            col, mostly=max(0.0, 1.0 - null_rate - 0.10))
        # Adaptive mean expectation
        mean, std = ref_col.mean(), ref_col.std() + 1e-8
        ge_ref.expect_column_mean_to_be_between(
            col,
            min_value=float(mean - 4 * std),
            max_value=float(mean + 4 * std)
        )

    ge_batch = ge.from_pandas(batch)
    suite = ge_ref.get_expectation_suite(discard_failed_expectations=False)
    results = ge_batch.validate(expectation_suite=suite)
    failed = sum(1 for r in results['results'] if not r['success'])
    return failed > 0


def run_experiment(dataset_name, config, seed):
    reference, clean_batch, anom_batch = get_batches(config, seed)

    # PipeGuard scores
    pg_clean = pipeguard_score(reference, clean_batch)
    pg_anom  = pipeguard_score(reference, anom_batch)
    pg_tp    = int(pg_anom  >= THRESHOLD)  # detects anomalous batch
    pg_fp    = int(pg_clean >= THRESHOLD)  # falsely flags clean batch

    # GE detections
    ge_clean = ge_detected(reference, clean_batch)
    ge_anom  = ge_detected(reference, anom_batch)
    ge_tp    = int(ge_anom)
    ge_fp    = int(ge_clean)

    return {
        'dataset':       dataset_name,
        'seed':          seed,
        'pg_score_clean': round(pg_clean, 4),
        'pg_score_anom':  round(pg_anom,  4),
        'pg_detected':   pg_tp,
        'pg_false_alarm': pg_fp,
        'ge_detected':   ge_tp,
        'ge_false_alarm': ge_fp,
    }


if __name__ == '__main__':
    os.makedirs('experiments/results', exist_ok=True)
    results = []
    combos  = [(d, s) for d in DATASETS for s in SEEDS]

    for dataset_name, seed in tqdm(combos, desc='Experiments', unit='run'):
        config = DATASETS[dataset_name]
        try:
            r = run_experiment(dataset_name, config, seed)
            print(f"  {dataset_name} s={seed} | PG:{r['pg_score_clean']:.2f}->{r['pg_score_anom']:.2f} "
                  f"(det={r['pg_detected']},fp={r['pg_false_alarm']}) | "
                  f"GE(det={r['ge_detected']},fp={r['ge_false_alarm']})")
            results.append(r)
        except Exception as e:
            print(f"  ERROR {dataset_name} s={seed}: {e}")
            import traceback; traceback.print_exc()

    # Summary table
    print("\n=== BATCH-LEVEL SUMMARY ===")
    print(f"{'Dataset':<15} {'PG_DR':>6} {'PG_FPR':>7} {'GE_DR':>6} {'GE_FPR':>7}")
    for ds in DATASETS:
        ds_r = [r for r in results if r['dataset'] == ds]
        if not ds_r:
            print(f"{ds:<15}  NO DATA")
            continue
        pg_dr  = round(sum(r['pg_detected']    for r in ds_r) / len(ds_r), 2)
        pg_fpr = round(sum(r['pg_false_alarm'] for r in ds_r) / len(ds_r), 2)
        ge_dr  = round(sum(r['ge_detected']    for r in ds_r) / len(ds_r), 2)
        ge_fpr = round(sum(r['ge_false_alarm'] for r in ds_r) / len(ds_r), 2)
        print(f"{ds:<15} {pg_dr:>6} {pg_fpr:>7} {ge_dr:>6} {ge_fpr:>7}")

    out = 'experiments/results/metrics.csv'
    if results:
        import csv
        with open(out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {out}")
