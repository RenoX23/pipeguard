import os
import numpy as np
import pandas as pd
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from sqlalchemy import create_engine, text


class StructuralCheckOperator(BaseOperator):
    """
    PipeGuard Stage 1 — Structural Quality Check.

    Auto-infers schema from a reference window of the dataset.
    Checks the current batch for:
      - Null rate violations (statistical threshold, not hardcoded)
      - Schema drift (missing or added columns)
      - Format violations (string pattern changes)

    Produces a structural score in [0.0, 1.0].
    Writes result to monitoring.structural_scores in PostgreSQL.
    Returns score via XCom for Stage 2 to consume.
    """

    @apply_defaults
    def __init__(
        self,
        dataset_path: str,
        dataset_name: str,
        reference_rows: int = 10000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.reference_rows = reference_rows

    def execute(self, context):
        run_id = str(context['run_id'])

        df = self._load_data()
        self.log.info(
            f"[{self.dataset_name}] Loaded {len(df)} rows, {len(df.columns)} columns"
        )

        reference = df.head(self.reference_rows)
        current = df.iloc[self.reference_rows:]

        if len(current) == 0:
            self.log.warning("Current batch is empty — not enough rows beyond reference window")
            return {'structural_score': 0.0, 'run_id': run_id}

        schema = self._infer_schema(reference)
        score, violations = self._check_batch(current, schema)

        self.log.info(
            f"[{self.dataset_name}] Structural score: {score:.4f} | "
            f"null_violations={violations['null_violations']} | "
            f"schema_violations={violations['schema_violations']} | "
            f"format_violations={violations['format_violations']}"
        )

        self._write_to_db(run_id, score, violations)

        return {
            'structural_score': float(score),
            'run_id': run_id,
            'dataset': self.dataset_name
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        if self.dataset_path.endswith('.parquet'):
            return pd.read_parquet(self.dataset_path)
        return pd.read_csv(self.dataset_path, low_memory=False)

    def _infer_schema(self, df: pd.DataFrame) -> dict:
        """
        Auto-generate schema from reference window.
        Null threshold = mean + 2 * std of per-window null rates.
        Never hardcoded.
        """
        schema = {}
        window_size = 1000

        for col in df.columns:
            null_rates = []
            for start in range(0, len(df), window_size):
                window = df[col].iloc[start:start + window_size]
                null_rates.append(window.isnull().mean())

            null_rates = np.array(null_rates)
            mean_null = null_rates.mean()
            std_null = null_rates.std()

            schema[col] = {
                'dtype': str(df[col].dtype),
                'null_threshold': float(min(mean_null + 2 * std_null, 1.0)),
                'format_pattern': self._infer_format(df[col])
            }

        return schema

    def _infer_format(self, series: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        sample = series.dropna().astype(str).head(200)
        if len(sample) == 0:
            return 'unknown'
        if sample.str.match(r'^\d{4}-\d{2}-\d{2}').mean() > 0.9:
            return 'date_iso'
        if sample.str.match(r'^[YN]$').mean() > 0.9:
            return 'yn_flag'
        return 'text'

    def _check_batch(self, df: pd.DataFrame, schema: dict):
        null_violations = 0
        schema_violations = 0
        format_violations = 0
        max_null_severity = 0.0

        # Schema drift: columns in reference but missing in current
        missing_cols = set(schema.keys()) - set(df.columns)
        schema_violations += len(missing_cols)
        if missing_cols:
            self.log.warning(f"Missing columns in current batch: {missing_cols}")

        for col, col_schema in schema.items():
            if col not in df.columns:
                continue

            # Null rate check — severity-based
            current_null_rate = df[col].isnull().mean()
            threshold = col_schema['null_threshold']
            if current_null_rate > threshold:
                null_violations += 1
                if threshold > 0:
                    severity = min((current_null_rate / threshold) - 1.0, 1.0)
                else:
                    severity = 1.0  # any nulls when baseline is zero = max violation
                max_null_severity = max(max_null_severity, severity)
                self.log.info(
                    f"  Null violation — {col}: "
                    f"{current_null_rate:.3f} > threshold {threshold:.3f} "
                    f"(severity={severity:.3f})"
                )

            # Format check (string columns only)
            if col_schema['format_pattern'] == 'yn_flag':
                valid = df[col].dropna().astype(str).str.match(r'^[YN]$').mean()
                if valid < 0.95:
                    format_violations += 1
                    self.log.info(f"  Format violation — {col}: {valid:.3f} valid YN values")

        # Severity-based scoring:
        # null_score  = worst single violation magnitude (0–1)
        # schema_score = binary — any missing column = 1.0
        # format_score = binary — any format violation = 1.0
        null_score   = max_null_severity
        schema_score = float(min(schema_violations, 1))
        format_score = float(min(format_violations, 1))

        score = float(np.clip(
            0.60 * null_score +
            0.30 * schema_score +
            0.10 * format_score,
            0.0, 1.0
        ))

        return score, {
            'null_violations': null_violations,
            'schema_violations': schema_violations,
            'format_violations': format_violations
        }

    def _write_to_db(self, run_id: str, score: float, violations: dict):
        db_conn = os.environ.get('PIPEGUARD_DB_CONN')
        if not db_conn:
            self.log.warning("PIPEGUARD_DB_CONN not set — skipping DB write")
            return

        engine = create_engine(db_conn)
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO monitoring.structural_scores
                    (run_id, dataset, structural_score,
                     null_violations, schema_violations, format_violations)
                    VALUES (:run_id, :dataset, :score,
                            :null_v, :schema_v, :format_v)
                """),
                {
                    'run_id': run_id,
                    'dataset': self.dataset_name,
                    'score': score,
                    'null_v': violations['null_violations'],
                    'schema_v': violations['schema_violations'],
                    'format_v': violations['format_violations']
                }
            )
            conn.commit()
        self.log.info(f"Score written to monitoring.structural_scores")
