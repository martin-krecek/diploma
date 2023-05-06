from airflow import DAG
from airflow.operators.mysql_operator import MySqlOperator
from datetime import datetime

def_conn_id = "mysql-db"

# Define the DAG
dag = DAG(
    dag_id='output_stage',
    start_date=datetime(2023, 3, 12),
    schedule_interval=None,
    catchup=False,
    template_searchpath=["/home/melicharovykrecek/diploma/sql"]
)

create_output_stage = MySqlOperator(
    sql=f'/output_stage.sql',
    task_id=f"output_stage",
    mysql_conn_id=def_conn_id,
    dag=dag
)

# Set task dependencies
create_output_stage