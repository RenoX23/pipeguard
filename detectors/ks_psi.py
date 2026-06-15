import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class KSPSIDetector:
    """
    PipeGuard — Distribution drift detection using KS-Test and PSI.

    KS-Test: detects if two samples come from different distributions.
             p-value < alpha means statistically significant drift.

    PSI (Population Stability Index): measures magnitude of distribution shift.
             PSI < 0.1  = no significant change
             PSI 0.1-0.2 = moderate change
             PSI > 0.2  = significant shift (industry standard threshold)

    Returns a score in [0.0, 1.0].
    Higher = more drift detected.
    """

    def __init__(self, psi_threshold: float = 0.2, ks_alpha: float = 0.05, bins: int = 10):
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.bins = bins
        self.reference_data = None

    def fit(self, reference: pd.DataFrame) -> 'KSPSIDetector':
        """Store reference distribution."""
        self.reference_data = reference.copy()
        return self

    def score(self, current: pd.DataFrame) -> float:
        """
        Returns drift score [0.0, 1.0].
        Computed as proportion of numeric columns showing drift
        by either KS-Test or PSI threshold.
        """
        if self.reference_data is None:
            raise RuntimeError("Call fit() before score()")

        numeric_cols = current.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c in self.reference_data.columns]

        if not numeric_cols:
            return 0.0

        drift_flags = []
        for col in numeric_cols:
            ref_col = self.reference_data[col].dropna()
            cur_col = current[col].dropna()
            if len(ref_col) < 10 or len(cur_col) < 10:
                continue

            ks_drift = self._ks_test(ref_col.values, cur_col.values)
            psi_drift = self._psi(ref_col.values, cur_col.values)
            drift_flags.append(int(ks_drift or psi_drift))

        return float(np.mean(drift_flags)) if drift_flags else 0.0

    def column_report(self, current: pd.DataFrame) -> dict:
        """
        Per-column drift report.
        Returns dict: {col: {'ks_drift': bool, 'psi': float, 'psi_drift': bool}}
        """
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
            _, p_value = ks_2samp(ref_col, cur_col)
            report[col] = {
                'ks_p_value': round(float(p_value), 4),
                'ks_drift': bool(p_value < self.ks_alpha),
                'psi': round(psi_val, 4),
                'psi_drift': bool(psi_val > self.psi_threshold)
            }
        return report

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _ks_test(self, ref: np.ndarray, cur: np.ndarray) -> bool:
        _, p_value = ks_2samp(ref, cur)
        return bool(p_value < self.ks_alpha)

    def _psi(self, ref: np.ndarray, cur: np.ndarray) -> bool:
        return bool(self._psi_value(ref, cur) > self.psi_threshold)

    def _psi_value(self, ref: np.ndarray, cur: np.ndarray) -> float:
        min_val = min(ref.min(), cur.min())
        max_val = max(ref.max(), cur.max())
        if max_val - min_val < 1e-8:
            return 0.0

        bin_edges = np.linspace(min_val, max_val, self.bins + 1)
        ref_counts, _ = np.histogram(ref, bins=bin_edges)
        cur_counts, _ = np.histogram(cur, bins=bin_edges)

        ref_pct = ref_counts / (len(ref) + 1e-8)
        cur_pct = cur_counts / (len(cur) + 1e-8)

        # Avoid log(0)
        ref_pct = np.where(ref_pct == 0, 1e-8, ref_pct)
        cur_pct = np.where(cur_pct == 0, 1e-8, cur_pct)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return abs(psi)
