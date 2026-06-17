from datetime import datetime, timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from operators.structural_check_operator import StructuralCheckOperator
from operators.semantic_check_operator import SemanticCheckOperator

default_args = {
    'owner': 'pipeguard',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='pipeguard_main',
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=['pipeguard', 'data-quality']
) as dag:

    # NYC Taxi — freshness + volume + schema drift
    structural_nyc = StructuralCheckOperator(
        task_id='structural_check_nyc_taxi',
        dataset_path='/opt/airflow/data/raw/nyc_taxi_jan2023.parquet',
        dataset_name='nyc_taxi',
        reference_rows=10000
    )

    semantic_nyc = SemanticCheckOperator(
        task_id='semantic_check_nyc_taxi',
        dataset_path='/opt/airflow/data/raw/nyc_taxi_jan2023.parquet',
        dataset_name='nyc_taxi',
        reference_rows=10000,
        if_contamination=0.05,
        streaming_metric='trip_distance',
        structural_task_id='structural_check_nyc_taxi'
    )

    structural_nyc >> semantic_nyc
