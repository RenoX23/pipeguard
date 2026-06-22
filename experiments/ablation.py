"""
PipeGuard Ablation Study
5 configurations x 4 datasets x 3 seeds
Shows each component's individual contribution.
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
from tqdm import tqdm
from experiments.run_all import get_batches, DATASETS, SEEDS, THRESHOLD
from operators.structural_check_operator import StructuralCheckOperator
from detectors.isolation_forest import IsolationForestDetector
from detectors.ks_psi import KSPSIDetector

CONFIGS = {
    'stage1_only':     {'w_struct': 1.0, 'w_if': 0.0, 'w_ks': 0.0},
    'if_only':         {'w_struct': 0.0, 'w_if': 1.0, 'w_ks': 0.0},
    'ks_psi_only':     {'w_struct': 0.0, 'w_if': 0.0, 'w_ks': 1.0},
    'no_structural':   {'w_struct': 0.0, 'w_if': 0.6, 'w_ks': 0.4},
    'full_pipeguard':  {'w_struct': 0.35,'w_if': 0.35,'w_ks': 0.20},
}


def score_batch(reference, batch, weights):
    mock_log = MagicMock()
    op = StructuralCheckOperator.__new__(StructuralCheckOperator)
    op.log = mock_log
    op.dataset_name = 'ablation'
    schema = op._infer_schema(reference)
    s_score, _ = op._check_batch(batch, schema)

    num_cols = batch.select_dtypes(include=[np.number]).columns.tolist()
    X_ref = reference[num_cols].fillna(0).values
    X_cur = batch[num_cols].fillna(0).values

    if_det = IsolationForestDetector(contamination=0.05, random_state=42)
    if_det.fit(X_ref)
    i_score = if_det.score(X_cur)

    ks_det = KSPSIDetector()
    ks_det.fit(reference[num_cols])
    k_score = ks_det.score(batch[num_cols])

    composite = (weights['w_struct'] * s_score +
                 weights['w_if']    * i_score +
                 weights['w_ks']    * k_score)
    return float(np.clip(composite, 0, 1))


if __name__ == '__main__':
    results = []
    combos = [(cfg, ds, s) for cfg in CONFIGS
              for ds in DATASETS for s in SEEDS]

    for config_name, dataset_name, seed in tqdm(combos, desc='Ablation', unit='run'):
        try:
            ref, clean, anom = get_batches(DATASETS[dataset_name], seed)
            weights = CONFIGS[config_name]
            clean_score = score_batch(ref, clean, weights)
            anom_score  = score_batch(ref, anom,  weights)
            detected    = int(anom_score  >= THRESHOLD)
            false_alarm = int(clean_score >= THRESHOLD)
            results.append({
                'config': config_name, 'dataset': dataset_name, 'seed': seed,
                'clean_score': round(clean_score, 3),
                'anom_score':  round(anom_score,  3),
                'detected': detected, 'false_alarm': false_alarm
            })
        except Exception as e:
            print(f"  ERROR {config_name}/{dataset_name}/s={seed}: {e}")

    print("\n=== ABLATION SUMMARY (DR / FPR) ===")
    print(f"{'Config':<18} {'NYC':>8} {'CC':>8} {'Census':>8} {'SMD':>8}")
    for cfg in CONFIGS:
        row = f"{cfg:<18}"
        for ds in DATASETS:
            rs = [r for r in results if r['config']==cfg and r['dataset']==ds]
            if rs:
                dr  = sum(r['detected']    for r in rs) / len(rs)
                fpr = sum(r['false_alarm'] for r in rs) / len(rs)
                row += f"  {dr:.1f}/{fpr:.1f}"
            else:
                row += "   -/-  "
        print(row)

    out = 'experiments/results/ablation.csv'
    os.makedirs('experiments/results', exist_ok=True)
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {out}")
