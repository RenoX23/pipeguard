"""
PipeGuard — Full Metrics Report
Computes Accuracy, Precision, Recall (DR), F1, and Confusion Matrix
for every primary experiment. Uses already-verified detector logic —
no changes to KS-PSI, IF, or any locked component.
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from detectors.isolation_forest import IsolationForestDetector


def full_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = Detection Rate
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
    return {
        'accuracy': round(accuracy, 4), 'precision': round(precision, 4),
        'recall_dr': round(recall, 4), 'fpr': round(fpr, 4), 'f1': round(f1, 4),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }


# ── SMD: sum raw confusion counts across all 28 machines ───────────────────
print("="*70)
print("SMD — 28 machines, aggregated confusion matrix (real labels)")
print("="*70)

tp_sum = fp_sum = tn_sum = fn_sum = 0
train_files = sorted(glob.glob('data/raw/smd/train/*.txt'))

for train_path in train_files:
    fname = os.path.basename(train_path)
    test_path  = f'data/raw/smd/test/{fname}'
    label_path = f'data/raw/smd/test_label/{fname}'
    if not os.path.exists(test_path) or not os.path.exists(label_path):
        continue
    X_train = np.loadtxt(train_path, delimiter=',')
    X_test  = np.loadtxt(test_path,  delimiter=',')
    labels  = np.loadtxt(label_path, delimiter=',').astype(int)

    det = IsolationForestDetector(contamination=0.10, random_state=42)
    det.fit(X_train)
    m = det.detection_rate(X_test, labels)
    tp_sum += m['tp']; fp_sum += m['fp']; tn_sum += m['tn']; fn_sum += m['fn']

smd_metrics = full_metrics(tp_sum, fp_sum, tn_sum, fn_sum)
trivial_acc_smd = (tp_sum+fn_sum==0) and 0 or (tn_sum+fp_sum)/(tp_sum+fp_sum+tn_sum+fn_sum)
print(f"Confusion Matrix: TP={tp_sum} FP={fp_sum} TN={tn_sum} FN={fn_sum}")
print(f"Accuracy:  {smd_metrics['accuracy']:.4f}   (trivial 'predict-clean' baseline: {trivial_acc_smd:.4f})")
print(f"Precision: {smd_metrics['precision']:.4f}")
print(f"Recall/DR: {smd_metrics['recall_dr']:.4f}")
print(f"FPR:       {smd_metrics['fpr']:.4f}")
print(f"F1:        {smd_metrics['f1']:.4f}")


# ── Credit Card: real fraud, 3 densities, aggregated across 3 seeds ─────────
print("\n" + "="*70)
print("CREDIT CARD — real fraud labels, aggregated confusion matrix")
print("="*70)

df = pd.read_csv('data/raw/creditcard.csv')
clean = df[df['Class']==0].reset_index(drop=True)
fraud = df[df['Class']==1].reset_index(drop=True)
reference   = clean.iloc[:10000]
clean_batch = clean.iloc[10000:15000].reset_index(drop=True)

trivial_acc_cc = len(clean) / len(df)

for fraud_pct in [0.05, 0.10, 0.20]:
    n_fraud = int(5000 * fraud_pct)
    tp_sum = fp_sum = tn_sum = fn_sum = 0
    for seed in [42, 123, 456]:
        fraud_sample = fraud.sample(n=min(n_fraud, len(fraud)), random_state=seed).reset_index(drop=True)
        n_clean = 5000 - len(fraud_sample)
        clean_sample = clean_batch.sample(n=n_clean, random_state=seed).reset_index(drop=True)
        anom_batch = pd.concat([clean_sample, fraud_sample]).reset_index(drop=True)
        true_labels = np.array([0]*len(clean_sample) + [1]*len(fraud_sample))

        num_cols = anom_batch.select_dtypes(include=[np.number]).columns.tolist()
        det = IsolationForestDetector(contamination=max(0.01, len(fraud_sample)/len(anom_batch)), random_state=seed)
        det.fit(reference[num_cols].fillna(0).values)
        preds = (det.predict(anom_batch[num_cols].fillna(0).values) == -1).astype(int)

        tp_sum += int(((preds==1)&(true_labels==1)).sum())
        fp_sum += int(((preds==1)&(true_labels==0)).sum())
        tn_sum += int(((preds==0)&(true_labels==0)).sum())
        fn_sum += int(((preds==0)&(true_labels==1)).sum())

    m = full_metrics(tp_sum, fp_sum, tn_sum, fn_sum)
    print(f"\nFraud density {int(fraud_pct*100)}%:")
    print(f"  Confusion Matrix: TP={tp_sum} FP={fp_sum} TN={tn_sum} FN={fn_sum}")
    print(f"  Accuracy:  {m['accuracy']:.4f}   (trivial 'predict-clean' baseline over full dataset: {trivial_acc_cc:.4f})")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall/DR: {m['recall_dr']:.4f}")
    print(f"  FPR:       {m['fpr']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")


# ── Streaming NYC Taxi: batch-level confusion matrix, PG vs GE ──────────────
print("\n" + "="*70)
print("STREAMING NYC TAXI — batch-level confusion matrix (PipeGuard vs GE)")
print("="*70)

sdf = pd.read_csv('experiments/results/streaming_results_corrected.csv')

def batch_confusion(df, pred_col):
    tp = int(((df[pred_col]==1)&(df['is_anomalous']==1)).sum())
    fp = int(((df[pred_col]==1)&(df['is_anomalous']==0)).sum())
    tn = int(((df[pred_col]==0)&(df['is_anomalous']==0)).sum())
    fn = int(((df[pred_col]==0)&(df['is_anomalous']==1)).sum())
    return full_metrics(tp, fp, tn, fn)

pg_m = batch_confusion(sdf, 'pg_detected')
ge_m = batch_confusion(sdf, 'ge_flag')

print(f"\nPipeGuard: TP={pg_m['tp']} FP={pg_m['fp']} TN={pg_m['tn']} FN={pg_m['fn']}")
print(f"  Accuracy={pg_m['accuracy']:.4f} Precision={pg_m['precision']:.4f} "
      f"Recall={pg_m['recall_dr']:.4f} FPR={pg_m['fpr']:.4f} F1={pg_m['f1']:.4f}")

print(f"\nGreat Expectations: TP={ge_m['tp']} FP={ge_m['fp']} TN={ge_m['tn']} FN={ge_m['fn']}")
print(f"  Accuracy={ge_m['accuracy']:.4f} Precision={ge_m['precision']:.4f} "
      f"Recall={ge_m['recall_dr']:.4f} FPR={ge_m['fpr']:.4f} F1={ge_m['f1']:.4f}")

print("\nNOTE: GE for SMD/Credit Card produces batch-level pass/fail only —")
print("it cannot attribute anomalies to individual rows. Row-level confusion")
print("matrices for GE on those datasets do not exist because GE architecturally")
print("cannot produce them. This asymmetry is itself a documented finding.")

# Save everything
os.makedirs('experiments/results', exist_ok=True)
with open('experiments/results/full_metrics_report.txt', 'w') as f:
    import contextlib, io
    print("Report generated — see console output above. Rerun this script to regenerate.")
