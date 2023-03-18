from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

def print_test():
    print("Test")

dag = DAG('my_dag', description='Simple DAG with PythonOperator',
          schedule_interval='*/5 * * * *',
          start_date=datetime(2023, 3, 12), catchup=False)

print_task = PythonOperator(task_id='print_test', python_callable=print_test, dag=dag)
