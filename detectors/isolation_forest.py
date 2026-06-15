import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolationForestDetector:
    """
    PipeGuard — Multivariate anomaly detection using Isolation Forest.

    Trained on a clean reference window.
    Scores incoming batches for anomalous rows.
    Returns a normalized score in [0.0, 1.0].

    Higher score = more anomalous.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Train on clean reference data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> float:
        """
        Returns batch-level anomaly score [0.0, 1.0].
        Aggregates row-level IF scores via mean.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before score()")

        X_scaled = self.scaler.transform(X)
        # decision_function: negative = anomalous, positive = normal
        raw = self.model.decision_function(X_scaled)
        # Flip so higher = more anomalous, then normalize to [0, 1]
        flipped = -raw
        min_val, max_val = flipped.min(), flipped.max()
        if max_val - min_val < 1e-8:
            return 0.0
        normalized = (flipped - min_val) / (max_val - min_val)
        return float(normalized.mean())

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns row-level binary predictions.
        -1 = anomaly, 1 = normal (sklearn convention).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def detection_rate(self, X: np.ndarray, labels: np.ndarray) -> dict:
        """
        Compute Detection Rate, FPR, F1 against ground truth labels.
        labels: 1 = anomaly, 0 = normal
        """
        preds = self.predict(X)
        # sklearn IF: -1 = anomaly → convert to 1 for anomaly
        pred_binary = (preds == -1).astype(int)
        true_binary = labels.astype(int)

        tp = int(((pred_binary == 1) & (true_binary == 1)).sum())
        fp = int(((pred_binary == 1) & (true_binary == 0)).sum())
        tn = int(((pred_binary == 0) & (true_binary == 0)).sum())
        fn = int(((pred_binary == 0) & (true_binary == 1)).sum())

        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * detection_rate / (precision + detection_rate)
              if (precision + detection_rate) > 0 else 0.0)

        return {
            'detection_rate': round(detection_rate, 4),
            'fpr': round(fpr, 4),
            'precision': round(precision, 4),
            'f1': round(f1, 4),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
