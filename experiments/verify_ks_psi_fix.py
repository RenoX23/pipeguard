import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from detectors.ks_psi import KSPSIDetector

df = pd.read_csv('data/raw/creditcard.csv')

# Use ONLY genuinely clean rows (Class==0) — matches Experiment 2's methodology
clean_only = df[df['Class'] == 0].reset_index(drop=True)
reference       = clean_only.iloc[:10000].drop(columns=['Time', 'Class'])
clean_test      = clean_only.iloc[10000:15000].drop(columns=['Time', 'Class'])

# Build a genuinely anomalous batch: inject real fraud rows to confirm
# the fix didn't kill sensitivity
fraud = df[df['Class'] == 1].drop(columns=['Time', 'Class'])
anom_test = pd.concat([
    clean_only.iloc[15000:19500].drop(columns=['Time', 'Class']),
    fraud.sample(n=min(500, len(fraud)), random_state=42)
]).reset_index(drop=True)

det = KSPSIDetector()
det.fit(reference)

clean_score = det.score(clean_test)
anom_score  = det.score(anom_test)

print("=== CHECK 1: Clean batch score (genuinely fraud-free) ===")
print(f"Score: {clean_score:.4f}  | Target: < 0.10")
print("PASS" if clean_score < 0.10 else "FAIL")

print("\n=== CHECK 2: Anomalous batch score (10% real fraud mixed in) ===")
print(f"Score: {anom_score:.4f}  | Target: > clean_score (detector still sensitive)")
print("PASS" if anom_score > clean_score else "FAIL")

print("\n=== VERDICT ===")
if clean_score < 0.10 and anom_score > clean_score:
    print(f"✅ Detector fixed. Clean={clean_score:.4f}, Anomalous={anom_score:.4f}, "
          f"Delta={anom_score-clean_score:.4f}. Safe to rebuild ablation.")
else:
    print("❌ Still broken. Do not proceed.")
