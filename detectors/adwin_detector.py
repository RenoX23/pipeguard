import numpy as np
import pandas as pd
from river.drift import ADWIN


class ADWINDetector:
    """
    PipeGuard — Concept drift detection using ADWIN.

    ADWIN (ADaptive WINdowing) maintains a window of recent values
    and detects when the mean shifts significantly.
    Runs in O(log n) time — suitable for streaming.

    Monitors a single numeric metric stream per batch.
    Returns 1.0 if drift detected, 0.0 otherwise.
    Also records the index where drift was first detected (MTTD proxy).
    """

    def __init__(self, delta: float = 0.002):
        """
        delta: confidence parameter.
               Lower = more sensitive (more detections, more false positives).
               Higher = less sensitive (fewer detections, fewer false positives).
               Default 0.002 is the standard value from the original paper.
        """
        self.delta = delta
        self.drift_index = None

    def detect(self, stream: pd.Series) -> float:
        """
        Feed a numeric series as a stream.
        Returns 1.0 if concept drift detected, 0.0 otherwise.
        Sets self.drift_index to the position of first detected drift.
        """
        detector = ADWIN(delta=self.delta)
        self.drift_index = None

        for i, value in enumerate(stream.dropna()):
            detector.update(float(value))
            if detector.drift_detected:
                self.drift_index = i
                return 1.0

        return 0.0

    def mttd(self) -> int:
        """
        Mean Time To Detect — rows processed before drift detected.
        Returns None if no drift was detected.
        """
        return self.drift_index

    def detect_multi(self, df: pd.DataFrame, columns: list = None) -> dict:
        """
        Run ADWIN on multiple columns independently.
        Returns per-column results and an overall score.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        results = {}
        any_drift = False

        for col in columns:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 10:
                continue
            drift = self.detect(series)
            results[col] = {
                'drift_detected': bool(drift),
                'drift_index': self.drift_index
            }
            if drift:
                any_drift = True

        results['overall_score'] = 1.0 if any_drift else 0.0
        return results
