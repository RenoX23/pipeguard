import numpy as np
import pandas as pd


class KSPSIDetector:
    """
    PipeGuard — Distribution drift detection using Population Stability Index.

    KS-test was evaluated and rejected: p-values are not sample-size
    independent, producing false positives at n>5000 regardless of
    threshold tuning. PSI with quintile binning (bins=5) is the sole
    drift signal — standard practice in credit-risk population
    stability monitoring, robust to batch-size noise.

    Known limitation: mean-aggregation across many columns dilutes
    single-column distributional shifts. Compensated architecturally
    by Isolation Forest (Stage 2) and the structural layer (Stage 1)
    in the full composite score — see Limitations, Section VI.

    PSI < 0.10  : no significant drift
    PSI 0.10-0.20: moderate drift
    PSI > 0.20  : significant drift (industry threshold)
    """

    def __init__(self, psi_threshold: float = 0.2, bins: int = 5):
        self.psi_threshold = psi_threshold
        self.bins = bins
        self.reference_data = None

    def fit(self, reference: pd.DataFrame) -> 'KSPSIDetector':
        self.reference_data = reference.copy()
        return self

    def score(self, current: pd.DataFrame) -> float:
        if self.reference_data is None:
            raise RuntimeError("Call fit() before score()")

        numeric_cols = current.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c in self.reference_data.columns]
        if not numeric_cols:
            return 0.0

        drift_flags = []
        for col in numeric_cols:
            ref_col = self.reference_data[col].dropna().values
            cur_col = current[col].dropna().values
            if len(ref_col) < 10 or len(cur_col) < 10:
                continue
            psi_val = self._psi_value(ref_col, cur_col)
            drift_flags.append(int(psi_val > self.psi_threshold))

        return float(np.mean(drift_flags)) if drift_flags else 0.0

    def column_report(self, current: pd.DataFrame) -> dict:
        if self.reference_data is None:
            raise RuntimeError("Call fit() before column_report()")
        report = {}
        numeric_cols = current.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c in self.reference_data.columns]
        for col in numeric_cols:
            ref_col = self.reference_data[col].dropna().values
            cur_col = current[col].dropna().values
            if len(ref_col) < 10 or len(cur_col) < 10:
                continue
            psi_val = self._psi_value(ref_col, cur_col)
            report[col] = {
                'psi': round(psi_val, 4),
                'psi_drift': bool(psi_val > self.psi_threshold)
            }
        return report

    def _psi_value(self, ref: np.ndarray, cur: np.ndarray) -> float:
        quantiles = np.linspace(0, 1, self.bins + 1)
        bin_edges = np.unique(np.quantile(ref, quantiles))
        if len(bin_edges) < 3:
            return 0.0
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf
        ref_counts, _ = np.histogram(ref, bins=bin_edges)
        cur_counts, _ = np.histogram(cur, bins=bin_edges)
        ref_pct = ref_counts / (len(ref) + 1e-8)
        cur_pct = cur_counts / (len(cur) + 1e-8)
        ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
        cur_pct = np.where(cur_pct == 0, 1e-4, cur_pct)
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return abs(psi)
