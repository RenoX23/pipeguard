import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/creditcard.csv')
clean_only = df[df['Class'] == 0].reset_index(drop=True)
reference  = clean_only.iloc[:10000].drop(columns=['Time', 'Class'])
clean_test = clean_only.iloc[10000:15000].drop(columns=['Time', 'Class'])

fraud = df[df['Class'] == 1].drop(columns=['Time', 'Class'])
anom_test = pd.concat([
    clean_only.iloc[15000:19500].drop(columns=['Time', 'Class']),
    fraud.sample(n=min(500, len(fraud)), random_state=42)
]).reset_index(drop=True)

def psi_only_score(ref_df, cur_df, bins, psi_threshold=0.2):
    flags = []
    for col in ref_df.columns:
        ref, cur = ref_df[col].values, cur_df[col].values
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(ref, quantiles))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        rc, _ = np.histogram(ref, bins=edges)
        cc, _ = np.histogram(cur, bins=edges)
        rp = np.where(rc/(len(ref)+1e-8)==0, 1e-4, rc/(len(ref)+1e-8))
        cp = np.where(cc/(len(cur)+1e-8)==0, 1e-4, cc/(len(cur)+1e-8))
        psi = abs(np.sum((cp-rp)*np.log(cp/rp)))
        flags.append(int(psi > psi_threshold))
    return np.mean(flags)

print(f"{'Bins':>6} {'Clean Score':>14} {'Anomalous Score':>18} {'Separation':>12}")
for bins in [5, 10, 15, 20]:
    c = psi_only_score(reference, clean_test, bins)
    a = psi_only_score(reference, anom_test, bins)
    print(f"{bins:>6} {c:>14.4f} {a:>18.4f} {a-c:>12.4f}")
