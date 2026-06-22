import sys, numpy as np
sys.path.insert(0, '.')

from unittest.mock import MagicMock

class MockBaseOperator:
    def __init__(self, task_id=None, **kwargs):
        self.task_id = task_id
        self.log = MagicMock()

def mock_apply_defaults(func):
    return func

am = MagicMock()
am.models.BaseOperator = MockBaseOperator
am.utils.decorators.apply_defaults = mock_apply_defaults
sys.modules['airflow'] = am
sys.modules['airflow.models'] = am.models
sys.modules['airflow.utils'] = am.utils
sys.modules['airflow.utils.decorators'] = am.utils.decorators

from operators.structural_check_operator import StructuralCheckOperator
from detectors.isolation_forest import IsolationForestDetector
from detectors.ks_psi import KSPSIDetector
import pandas as pd

df  = pd.read_parquet('data/raw/nyc_taxi_jan2023.parquet')
ref = df.head(10000)
bat = df.iloc[10000:15000].reset_index(drop=True)

op = StructuralCheckOperator.__new__(StructuralCheckOperator)
op.log = MagicMock()
op.dataset_name = 'sens'
schema = op._infer_schema(ref)
s, _ = op._check_batch(bat, schema)

num = bat.select_dtypes(include=[np.number]).columns.tolist()
ifd = IsolationForestDetector(contamination=0.10, random_state=42)
ifd.fit(ref[num].fillna(0).values)
i = ifd.score(bat[num].fillna(0).values)
ksd = KSPSIDetector()
ksd.fit(ref[num])
k = ksd.score(bat[num])

print(f'Raw: struct={s:.3f} IF={i:.3f} KS={k:.3f}')
configs = [
    ('Equal 0.33/0.33/0.33',      0.33, 0.33, 0.33),
    ('Struct-heavy 0.50/0.30/0.20', 0.50, 0.30, 0.20),
    ('IF-heavy 0.20/0.60/0.20',   0.20, 0.60, 0.20),
    ('PipeGuard 0.35/0.35/0.20',  0.35, 0.35, 0.20),
    ('KS-heavy 0.20/0.30/0.50',   0.20, 0.30, 0.50),
]
print(f"\n{'Config':<35} {'Score':>8}")
for name, w1, w2, w3 in configs:
    print(f'{name:<35} {w1*s+w2*i+w3*k:>8.4f}')
