import numpy as np
import pandas as pd


class AnomalyInjector:
    """
    Injects controlled anomalies into datasets that lack ground truth labels.
    Records exact injection positions as ground truth.
    """

    def inject_null_spike(
        self, df: pd.DataFrame, column: str,
        null_pct: float = 0.40, window_pct: float = 0.10, seed: int = 42
    ):
        rng = np.random.default_rng(seed)
        n_window = int(len(df) * window_pct)
        window_idx = rng.choice(len(df), size=n_window, replace=False)
        n_null = int(n_window * null_pct)
        null_idx = rng.choice(window_idx, size=n_null, replace=False)
        injected = df.copy()
        injected.loc[null_idx, column] = np.nan
        return injected, list(null_idx)

    def inject_distribution_shift(
        self, df: pd.DataFrame, column: str,
        shift_factor: float = 3.0, tail_pct: float = 0.20, seed: int = 42
    ):
        cutoff = int(len(df) * (1 - tail_pct))
        affected = list(range(cutoff, len(df)))
        injected = df.copy()
        injected.loc[cutoff:, column] = injected.loc[cutoff:, column] * shift_factor
        return injected, affected

    def inject_schema_drift(self, df: pd.DataFrame, drop_col: str):
        injected = df.drop(columns=[drop_col])
        return injected, list(range(len(df)))
