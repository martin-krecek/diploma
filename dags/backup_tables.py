from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.mysql_operator import MySqlOperator
from datetime import datetime


def_entity = 'backup_tables'
def_conn_id = "mysql-db"

# Define the DAG
dag = DAG(
    dag_id=def_entity,
    start_date=datetime(2023, 3, 12),
    schedule_interval=None,
    catchup=False,
    template_searchpath=["/home/melicharovykrecek/diploma/sql"]
)


parking_measurements = MySqlOperator(
    sql='/parking/measurements/100_backup.sql',
    task_id="backup_table_parking_measurements",
    mysql_conn_id=def_conn_id,
)

parking_spaces = MySqlOperator(
    sql='/parking/100_backup.sql',
    task_id="backup_table_parking",
    mysql_conn_id=def_conn_id,
)

# Define a task to trigger all tasks at once
trigger_tasks = PythonOperator(
    task_id='trigger_tasks',
    python_callable=lambda: None,
    dag=dag
)

# Set task dependencies

trigger_tasks >> [parking_measurements, parking_spaces]


