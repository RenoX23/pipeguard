import sys
from unittest.mock import MagicMock

class MockBaseOperator:
    def __init__(self, task_id=None, **kwargs):
        self.task_id = task_id
        self.log = MagicMock()
    def execute(self, context):
        raise NotImplementedError

sys.modules["airflow"] = MagicMock()
sys.modules["airflow.models"] = MagicMock()
sys.modules["airflow.models"].BaseOperator = MockBaseOperator
sys.modules["airflow.utils"] = MagicMock()
sys.modules["airflow.utils.decorators"] = MagicMock()
sys.modules["airflow.utils.decorators"].apply_defaults = lambda f: f


"""
PipeGuard Experiments Runner
Evaluates PipeGuard vs Great Expectations across 4 datasets.
Outputs: experiments/results/metrics.csv
"""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from experiments.anomaly_injector import AnomalyInjector

from operators.structural_check_operator import StructuralCheckOperator
from operators.semantic_check_operator import SemanticCheckOperator
from detectors.isolation_forest import IsolationForestDetector


SEEDS = [42, 123, 456]
REFERENCE_ROWS = 10000

DATASETS = {
    'nyc_taxi': {
        'path': 'data/raw/nyc_taxi_jan2023.parquet',
        'inject': ('null_spike', 'trip_distance'),
        'has_labels': False,
        'streaming_metric': 'trip_distance'
    },
    'credit_card': {
        'path': 'data/raw/creditcard.csv',
        'inject': ('distribution_shift', 'Amount'),
        'has_labels': True,
        'label_col': 'Class',
        'streaming_metric': 'Amount'
    },
    'uci_census': {
        'path': 'data/raw/uci_census.csv',
        'inject': ('null_spike', 0),
        'has_labels': False,
        'streaming_metric': None
    },
    'smd': {
        'path': 'data/raw/smd/test/machine-1-1.txt',
        'train_path': 'data/raw/smd/train/machine-1-1.txt',
        'label_path': 'data/raw/smd/test_label/machine-1-1.txt',
        'inject': None,
        'has_labels': True,
        'streaming_metric': '0'
    }
}


def load_dataset(config):
    path = config['path']
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    elif path.endswith('.txt'):
        return pd.read_csv(path, header=None)
    else:
        return pd.read_csv(path, low_memory=False, header=None
                          if 'census' in path else 0)


def inject_anomalies(df, config, seed):
    injector = AnomalyInjector()
    inject_type, col = config['inject']
    if inject_type == 'null_spike':
        return injector.inject_null_spike(df, col, null_pct=0.40, seed=seed)
    elif inject_type == 'distribution_shift':
        return injector.inject_distribution_shift(df, col, shift_factor=3.0, seed=seed)
    return df, []


def run_pipeguard(df_injected, config, seed):
    """Run PipeGuard Stage 1 + Stage 2, return binary predictions per row."""
    ref = df_injected.head(REFERENCE_ROWS)
    current = df_injected.iloc[REFERENCE_ROWS:]

    # Stage 1 — structural score
    struct_op = StructuralCheckOperator.__new__(StructuralCheckOperator)
    struct_op.log = type('L', (), {'info': print, 'warning': print})()
    struct_op.dataset_name = 'experiment'
    schema = struct_op._infer_schema(ref)
    structural_score, _ = struct_op._check_batch(current, schema)

    # Stage 2 — semantic score
    numeric_cols = current.select_dtypes(include=[np.number]).columns.tolist()
    X_ref = ref[numeric_cols].fillna(0).values
    X_cur = current[numeric_cols].fillna(0).values

    if_detector = IsolationForestDetector(contamination=0.05, random_state=seed)
    if_detector.fit(X_ref)
    row_preds = if_detector.predict(X_cur)  # -1 = anomaly

    return (row_preds == -1).astype(int), structural_score


def run_great_expectations(df_injected, config):
    """Run Great Expectations on reference window, validate current batch."""
    import great_expectations as ge
    ref = df_injected.head(REFERENCE_ROWS)
    current = df_injected.iloc[REFERENCE_ROWS:]

    ge_df = ge.from_pandas(ref)
    suite = ge_df.get_expectation_suite(discard_failed_expectations=False)

    # Auto-generate expectations
    for col in ref.select_dtypes(include=[np.number]).columns[:10]:
        ge_df.expect_column_values_to_not_be_null(col)
        ge_df.expect_column_mean_to_be_between(
            col,
            min_value=float(ref[col].mean() - 3 * ref[col].std()),
            max_value=float(ref[col].mean() + 3 * ref[col].std())
        )

    ge_current = ge.from_pandas(current)
    results = ge_current.validate(expectation_suite=ge_df.get_expectation_suite())
    failed = sum(1 for r in results['results'] if not r['success'])
    total = len(results['results'])

    # GE gives batch-level result — broadcast to row level
    ge_detected = 1 if failed > 0 else 0
    n_current = len(current)
    return np.full(n_current, ge_detected, dtype=int)


def compute_metrics(preds, labels, n_rows):
    """Compute Detection Rate, FPR, F1."""
    preds = np.array(preds[:len(labels)])
    labels = np.array(labels[:len(preds)])

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    dr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * dr / (prec + dr) if (prec + dr) > 0 else 0.0

    return {'detection_rate': round(dr, 4), 'fpr': round(fpr, 4),
            'f1': round(f1, 4), 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def run_experiment(dataset_name, config, seed):
    print(f"\n--- {dataset_name} | seed={seed} ---")
    df = load_dataset(config)
    print(f"  Loaded: {df.shape}")

    # Get or inject ground truth
    if config['inject'] is not None:
        df_injected, anomaly_idx = inject_anomalies(df, config, seed)
        labels = np.zeros(len(df_injected) - REFERENCE_ROWS, dtype=int)
        for idx in anomaly_idx:
            adj = idx - REFERENCE_ROWS
            if 0 <= adj < len(labels):
                labels[adj] = 1
    elif config.get('label_path'):
        df_injected = df
        labels_full = np.loadtxt(config['label_path'], delimiter=',').astype(int)
        labels = labels_full[REFERENCE_ROWS:]
    elif config.get('label_col'):
        df_injected = df
        labels = df[config['label_col']].values[REFERENCE_ROWS:].astype(int)
    else:
        df_injected = df
        labels = np.zeros(len(df) - REFERENCE_ROWS, dtype=int)

    # PipeGuard
    pg_preds, struct_score = run_pipeguard(df_injected, config, seed)
    pg_metrics = compute_metrics(pg_preds, labels, len(labels))
    print(f"  PipeGuard: DR={pg_metrics['detection_rate']} FPR={pg_metrics['fpr']} F1={pg_metrics['f1']}")

    # Great Expectations
    ge_preds = run_great_expectations(df_injected, config)
    ge_metrics = compute_metrics(ge_preds, labels, len(labels))
    print(f"  GE:        DR={ge_metrics['detection_rate']} FPR={ge_metrics['fpr']} F1={ge_metrics['f1']}")

    return {
        'dataset': dataset_name,
        'seed': seed,
        'pipeguard_dr': pg_metrics['detection_rate'],
        'pipeguard_fpr': pg_metrics['fpr'],
        'pipeguard_f1': pg_metrics['f1'],
        'ge_dr': ge_metrics['detection_rate'],
        'ge_fpr': ge_metrics['fpr'],
        'ge_f1': ge_metrics['f1']
    }


from tqdm import tqdm

if __name__ == '__main__':
    os.makedirs('experiments/results', exist_ok=True)
    results = []
    combos = [(d, s) for d in DATASETS for s in SEEDS]

    for dataset_name, seed in tqdm(combos, desc='Experiments', unit='run'):
        config = DATASETS[dataset_name]
        try:
            result = run_experiment(dataset_name, config, seed)
            results.append(result)
        except Exception as e:
            print(f'  ERROR {dataset_name} seed={seed}: {e}')

    out_path = 'experiments/results/metrics.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f'Results saved to {out_path}')
    print(f'Total: {len(results)} experiments')
