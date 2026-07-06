import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from detectors.ks_psi import KSPSIDetector

df = pd.read_csv('data/raw/creditcard.csv')
clean_only = df[df['Class'] == 0].reset_index(drop=True)
reference  = clean_only.iloc[:10000].drop(columns=['Time', 'Class'])
clean_test = clean_only.iloc[10000:15000].drop(columns=['Time', 'Class'])

fraud = df[df['Class'] == 1].drop(columns=['Time', 'Class'])
fraud_test = pd.concat([
    clean_only.iloc[15000:19500].drop(columns=['Time', 'Class']),
    fraud.sample(n=min(500, len(fraud)), random_state=42)
]).reset_index(drop=True)

det = KSPSIDetector()
det.fit(reference)

clean_score = det.score(clean_test)
fraud_score = det.score(fraud_test)

print(f"{'Scenario':<30} {'Score':>8}   Target")
print(f"{'Clean (no anomaly)':<30} {clean_score:>8.4f}   < 0.10")
print(f"{'Real fraud (multi-column)':<30} {fraud_score:>8.4f}   > clean_score")

ok = clean_score < 0.10 and fraud_score > clean_score
print("\n✅ LOCKED — reverted version confirmed, proceeding to ablation_v3"
      if ok else "❌ Reversion itself broken — stop everything, escalate")
