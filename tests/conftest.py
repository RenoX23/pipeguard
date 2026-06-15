import sys
from unittest.mock import MagicMock


class MockBaseOperator:
    """Minimal mock of Airflow BaseOperator for local testing."""

    def __init__(self, task_id=None, **kwargs):
        self.task_id = task_id
        self.log = MagicMock()

    def execute(self, context):
        raise NotImplementedError


def mock_apply_defaults(func):
    return func


# Inject mocks before any operator module is imported
airflow_mock = MagicMock()
airflow_mock.models.BaseOperator = MockBaseOperator
airflow_mock.utils.decorators.apply_defaults = mock_apply_defaults

sys.modules['airflow'] = airflow_mock
sys.modules['airflow.models'] = airflow_mock.models
sys.modules['airflow.utils'] = airflow_mock.utils
sys.modules['airflow.utils.decorators'] = airflow_mock.utils.decorators
