import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import subprocess

def run_python_script():
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)
    
    # Get the parent directory of the current file
    parent_directory = os.path.dirname(current_file_path)
    
    # Construct the full path to the Python file
    python_file_path = os.path.join(parent_directory, '/python/model_train.py.py')
    
    # Call the Python file
    subprocess.call(['python', python_file_path])


# Define the DAG
dag = DAG(
    dag_id='model_train',
    start_date=datetime(2023, 3, 12),
    schedule_interval=None,
    catchup=False,
    template_searchpath=["/home/melicharovykrecek/diploma/sql"]
)

model_train_task = PythonOperator(
    task_id='model_train_task',
    python_callable=run_python_script,
    dag=dag
)

# Set task dependencies
model_train_task