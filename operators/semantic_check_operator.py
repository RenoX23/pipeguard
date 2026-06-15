import os
import numpy as np
import pandas as pd
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from sqlalchemy import create_engine, text

from detectors.isolation_forest import IsolationForestDetector
from detectors.ks_psi import KSPSIDetector
from detectors.adwin_detector import ADWINDetector


class SemanticCheckOperator(BaseOperator):
    """
    PipeGuard Stage 2 — Semantic Quality Check.

    Combines three unsupervised ML detectors:
      - Isolation Forest  : multivariate anomaly detection
      - KS-Test + PSI     : distribution drift detection
      - ADWIN             : concept drift detection

    Composite score formula (locked):
      score = 0.35 * structural + 0.35 * if_score + 0.20 * ks_psi + 0.10 * adwin

    DLQ routing:
      score >= 0.7  -> high_risk  (DLQ + alert)
      score >= 0.4  -> review     (DLQ + review)
      score <  0.4  -> clean      (silver table)
    """

    WEIGHTS = {
        'structural': 0.35,
        'if':         0.35,
        'ks_psi':     0.20,
        'adwin':      0.10
    }

    @apply_defaults
    def __init__(
        self,
        dataset_path: str,
        dataset_name: str,
        reference_rows: int = 10000,
        if_contamination: float = 0.05,
        streaming_metric: str = None,
        structural_task_id: str = 'structural_check',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.reference_rows = reference_rows
        self.if_contamination = if_contamination
        self.streaming_metric = streaming_metric
        self.structural_task_id = structural_task_id

    def execute(self, context):
        run_id = str(context['run_id'])

        # Pull structural score from Stage 1 via XCom
        ti = context.get('ti')
        structural_score = 0.0
        if ti:
            result = ti.xcom_pull(task_ids=self.structural_task_id)
            if result:
                structural_score = float(result.get('structural_score', 0.0))

        df = self._load_data()
        reference = df.head(self.reference_rows)
        current = df.iloc[self.reference_rows:]

        if len(current) == 0:
            self.log.warning("Empty current batch — returning zero scores")
            return self._build_result(run_id, 0.0, 0.0, 0.0, 0.0, structural_score)

        numeric_cols = current.select_dtypes(include=[np.number]).columns.tolist()
        X_ref = reference[numeric_cols].fillna(0).values
        X_cur = current[numeric_cols].fillna(0).values

        # --- Isolation Forest ---
        if_detector = IsolationForestDetector(
            contamination=self.if_contamination,
            random_state=42
        )
        if_detector.fit(X_ref)
        if_score = if_detector.score(X_cur)
        self.log.info(f"[{self.dataset_name}] IF score: {if_score:.4f}")

        # --- KS-Test + PSI ---
        ks_detector = KSPSIDetector(psi_threshold=0.2, ks_alpha=0.05)
        ks_detector.fit(reference[numeric_cols])
        ks_psi_score = ks_detector.score(current[numeric_cols])
        self.log.info(f"[{self.dataset_name}] KS-PSI score: {ks_psi_score:.4f}")

        # --- ADWIN ---
        adwin_detector = ADWINDetector(delta=0.002)
        metric_col = self.streaming_metric or numeric_cols[0]
        if metric_col in current.columns:
            adwin_score = adwin_detector.detect(current[metric_col])
        else:
            adwin_score = 0.0
        self.log.info(f"[{self.dataset_name}] ADWIN score: {adwin_score:.4f}")

        # --- Composite score ---
        composite = float(np.clip(
            self.WEIGHTS['structural'] * structural_score +
            self.WEIGHTS['if']         * if_score +
            self.WEIGHTS['ks_psi']     * ks_psi_score +
            self.WEIGHTS['adwin']      * adwin_score,
            0.0, 1.0
        ))

        self.log.info(
            f"[{self.dataset_name}] Composite: {composite:.4f} | "
            f"structural={structural_score:.3f} if={if_score:.3f} "
            f"ks_psi={ks_psi_score:.3f} adwin={adwin_score:.3f}"
        )

        return self._build_result(
            run_id, if_score, ks_psi_score, adwin_score, composite, structural_score
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        if self.dataset_path.endswith('.parquet'):
            return pd.read_parquet(self.dataset_path)
        return pd.read_csv(self.dataset_path, low_memory=False)

    def _assign_tier(self, score: float) -> str:
        if score >= 0.7:
            return 'high_risk'
        elif score >= 0.4:
            return 'review'
        return 'clean'

    def _build_result(self, run_id, if_s, ks_s, adwin_s, composite, structural_s):
        tier = self._assign_tier(composite)
        self._write_to_db(run_id, if_s, ks_s, adwin_s, composite, tier)
        result = {
            'run_id':          run_id,
            'dataset':         self.dataset_name,
            'if_score':        round(if_s, 4),
            'ks_psi_score':    round(ks_s, 4),
            'adwin_score':     round(adwin_s, 4),
            'composite_score': round(composite, 4),
            'tier':            tier
        }
        self.log.info(f"[{self.dataset_name}] Tier: {tier.upper()}")
        return result

    def _write_to_db(self, run_id, if_s, ks_s, adwin_s, composite, tier):
        db_conn = os.environ.get('PIPEGUARD_DB_CONN')
        if not db_conn:
            self.log.warning("PIPEGUARD_DB_CONN not set — skipping DB write")
            return
        engine = create_engine(db_conn)
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO monitoring.semantic_scores
                    (run_id, dataset, if_score, ks_psi_score,
                     adwin_score, composite_score, tier)
                    VALUES (:run_id, :dataset, :if_s, :ks_s,
                            :adwin_s, :composite, :tier)
                """),
                {
                    'run_id':    run_id,
                    'dataset':   self.dataset_name,
                    'if_s':      if_s,
                    'ks_s':      ks_s,
                    'adwin_s':   adwin_s,
                    'composite': composite,
                    'tier':      tier
                }
            )
        self.log.info(f"Scores written to monitoring.semantic_scores")
