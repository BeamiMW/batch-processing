from datetime import timedelta
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "bmw",
    "retry_delay": timedelta(minutes=5),
}

spark_dag = DAG(
    dag_id="spark_airflow",
    default_args=default_args,
    schedule_interval=None,
    dagrun_timeout=timedelta(minutes=60),
    start_date=days_ago(1),
)

Extract = SparkSubmitOperator(
    task_id="spark_submit",
    application="/spark-scripts/online_retail.py",
    conn_id="spark_main",
    packages="org.postgresql:postgresql:42.2.18",
    dag=spark_dag
)

Extract
