import os
import shutil
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

def delete_old_logs():
    logs_dir = "/home/melicharovykrecek/diploma/logs"
    now = datetime.now()
    delta = timedelta(days=1)
    cutoff_time = now - delta
    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getmtime(file_path) < cutoff_time.timestamp():
                os.remove(file_path)

dag = DAG(
    dag_id="delete_old_logs",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 3, 12),
    catchup=False,
)

delete_logs_task = PythonOperator(
    task_id="delete_logs",
    python_callable=delete_old_logs,
    dag=dag,
)

delete_logs_task
