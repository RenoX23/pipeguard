from datetime import datetime
from airflow import DAG
from airflow.utils.dates import days_ago
from operators.structural_check_operator import StructuralCheckOperator

with DAG(
    dag_id='pipeguard_stage1_test',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=['pipeguard', 'stage1', 'test']
) as dag:

    structural_nyc = StructuralCheckOperator(
        task_id='structural_check_nyc_taxi',
        dataset_path='/opt/airflow/data/raw/nyc_taxi_jan2023.parquet',
        dataset_name='nyc_taxi',
        reference_rows=10000
    )
