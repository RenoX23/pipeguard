"""
PipeGuard Streaming Simulation
NYC Taxi processed as sequential batches ordered by timestamp.
Anomalies injected into batches 25-35.
PipeGuard composite score vs GE pass/fail — both plotted over time.
Produces: experiments/results/streaming_results.csv
          experiments/results/streaming_plot.png
"""
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import great_expectations as ge
from tqdm import tqdm

from operators.structural_check_operator import StructuralCheckOperator
from detectors.isolation_forest import IsolationForestDetector
from detectors.ks_psi import KSPSIDetector
from detectors.adwin_detector import ADWINDetector

# ── CONFIG ──────────────────────────────────────────────────────────────────
BATCH_SIZE       = 1000
REFERENCE_BATCHES = 10       # first 10 batches = reference window (10K rows)
TOTAL_BATCHES    = 50        # total batches to simulate
ANOMALY_START    = 25        # inject anomaly from batch 25
ANOMALY_END      = 35        # to batch 35
ANOMALY_COL      = 'trip_distance'
NULL_PCT         = 0.70      # 70% nulls injected
THRESHOLD        = 0.28


def load_ordered():
    """Load NYC Taxi ordered by pickup time — simulates real stream."""
    df = pd.read_parquet('data/raw/nyc_taxi_jan2023.parquet')
    df = df.sort_values('tpep_pickup_datetime').reset_index(drop=True)
    return df


def inject_batch(batch, col, null_pct, seed):
    """Inject nulls into a batch."""
    rng = np.random.default_rng(seed)
    injected = batch.copy()
    n = int(len(injected) * null_pct)
    idx = rng.choice(len(injected), size=n, replace=False)
    injected.loc[idx, col] = np.nan
    return injected


def pipeguard_score(reference, batch):
    """Full composite score."""
    op = StructuralCheckOperator.__new__(StructuralCheckOperator)
    op.log = MagicMock()
    op.dataset_name = 'stream'
    schema = op._infer_schema(reference)
    s, _  = op._check_batch(batch, schema)

    num   = batch.select_dtypes(include=[np.number]).columns.tolist()
    X_ref = reference[num].fillna(0).values
    X_cur = batch[num].fillna(0).values

    ifd = IsolationForestDetector(contamination=0.10, random_state=42)
    ifd.fit(X_ref)
    i = ifd.score(X_cur)

    ksd = KSPSIDetector()
    ksd.fit(reference[num])
    k = ksd.score(batch[num])

    adwin = ADWINDetector(delta=0.002)
    a = adwin.detect(batch[ANOMALY_COL].dropna())

    composite = 0.35*s + 0.35*i + 0.20*k + 0.10*a
    return float(np.clip(composite, 0, 1)), adwin.mttd()


def ge_check(reference, batch):
    """GE adaptive check on all numeric columns."""
    ge_ref = ge.from_pandas(reference)
    num    = reference.select_dtypes(include=[np.number]).columns.tolist()
    for col in num:
        rc = reference[col].dropna()
        if len(rc) == 0: continue
        nr = reference[col].isnull().mean()
        ge_ref.expect_column_values_to_not_be_null(
            col, mostly=max(0.0, 1.0 - nr - 0.10))
        m, s = rc.mean(), rc.std() + 1e-8
        ge_ref.expect_column_mean_to_be_between(
            col, float(m - 4*s), float(m + 4*s))
    suite = ge_ref.get_expectation_suite(discard_failed_expectations=False)
    res   = ge.from_pandas(batch).validate(expectation_suite=suite)
    failed = sum(1 for r in res['results'] if not r['success'])
    return 1 if failed > 0 else 0


def plot_results(results_df, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('PipeGuard vs Great Expectations — Streaming Simulation\n'
                 'NYC Taxi Dataset (50 batches × 1,000 rows)',
                 fontsize=14, fontweight='bold')

    batches = results_df['batch'].values
    pg      = results_df['pg_score'].values
    ge_flag = results_df['ge_flag'].values

    # Shade anomaly zone
    for ax in [ax1, ax2]:
        ax.axvspan(ANOMALY_START, ANOMALY_END,
                   alpha=0.15, color='red', label='Injection zone')
        ax.axvline(ANOMALY_START, color='red', linestyle='--', alpha=0.5)

    # Panel 1: PipeGuard composite score
    ax1.plot(batches, pg, color='#2196F3', linewidth=2, label='PipeGuard score')
    ax1.axhline(THRESHOLD, color='orange', linestyle='--',
                linewidth=1.5, label=f'Detection threshold ({THRESHOLD})')
    ax1.fill_between(batches, pg, THRESHOLD,
                     where=(np.array(pg) >= THRESHOLD),
                     alpha=0.3, color='red', label='Anomaly detected')
    ax1.set_ylabel('Composite Score', fontsize=11)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_title('PipeGuard — Composite Anomaly Score per Batch', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Panel 2: GE pass/fail
    colors = ['#F44336' if g else '#4CAF50' for g in ge_flag]
    ax2.bar(batches, ge_flag, color=colors, alpha=0.8, width=0.8)
    ax2.set_ylabel('GE Flag (1=Fail)', fontsize=11)
    ax2.set_ylim(0, 1.5)
    ax2.set_xlabel('Batch Number', fontsize=11)
    ax2.set_title('Great Expectations — Batch Pass/Fail', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    green = mpatches.Patch(color='#4CAF50', label='Pass (clean)')
    red   = mpatches.Patch(color='#F44336', label='Fail (flagged)')
    ax2.legend(handles=[green, red], loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {save_path}")


if __name__ == '__main__':
    os.makedirs('experiments/results', exist_ok=True)

    print("Loading NYC Taxi ordered by timestamp...")
    df = load_ordered()
    print(f"Loaded {len(df)} rows")

    # Reference window = first 10 batches
    reference = df.iloc[:REFERENCE_BATCHES * BATCH_SIZE]
    print(f"Reference: {len(reference)} rows ({REFERENCE_BATCHES} batches)")

    results = []
    mttd_batch = None

    for b in tqdm(range(REFERENCE_BATCHES, TOTAL_BATCHES), desc='Batches'):
        start = b * BATCH_SIZE
        end   = start + BATCH_SIZE
        batch = df.iloc[start:end].reset_index(drop=True)

        # Inject anomaly in batches 25-35
        is_anomalous = ANOMALY_START <= b < ANOMALY_END
        if is_anomalous:
            batch = inject_batch(batch, ANOMALY_COL, NULL_PCT, seed=b)

        pg, adwin_mttd = pipeguard_score(reference, batch)
        ge_f            = ge_check(reference, batch)

        # MTTD: first batch where PipeGuard detects after injection starts
        if is_anomalous and pg >= THRESHOLD and mttd_batch is None:
            mttd_batch = b

        results.append({
            'batch':        b,
            'is_anomalous': int(is_anomalous),
            'pg_score':     round(pg, 4),
            'pg_detected':  int(pg >= THRESHOLD),
            'ge_flag':      ge_f,
        })

    df_res = pd.DataFrame(results)

    # MTTD calculation
    if mttd_batch is not None:
        mttd_rows = (mttd_batch - ANOMALY_START) * BATCH_SIZE
        print(f"\nMTTD: detected at batch {mttd_batch} "
              f"({mttd_batch - ANOMALY_START} batches after injection = "
              f"{mttd_rows} rows)")
    else:
        print("\nMTTD: not detected in injection window")

    # GE false alarm rate on clean batches
    clean_batches  = df_res[df_res['is_anomalous']==0]
    anom_batches   = df_res[df_res['is_anomalous']==1]
    pg_dr  = anom_batches['pg_detected'].mean()
    pg_fpr = clean_batches['pg_detected'].mean()
    ge_dr  = anom_batches['ge_flag'].mean()
    ge_fpr = clean_batches['ge_flag'].mean()

    print(f"\n{'Metric':<25} {'PipeGuard':>12} {'GE':>12}")
    print(f"{'Detection Rate':<25} {pg_dr:>12.3f} {ge_dr:>12.3f}")
    print(f"{'False Positive Rate':<25} {pg_fpr:>12.3f} {ge_fpr:>12.3f}")
    print(f"{'MTTD (batches)':<25} {mttd_batch - ANOMALY_START if mttd_batch else 'N/A':>12}")
    print(f"{'MTTD (rows)':<25} {mttd_rows if mttd_batch else 'N/A':>12}")

    df_res.to_csv('experiments/results/streaming_results.csv', index=False)
    print("\nSaved: experiments/results/streaming_results.csv")

    plot_results(df_res, 'experiments/results/streaming_plot.png')
