import requests
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.mysql_operator import MySqlOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import json
import os
import logging

headers = {"Content-Type": "application/json; charset=utf-8", "x-access-token": 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1hcnRpbmtyZWNlazlAZ21haWwuY29tIiwiaWQiOjE0NTIsIm5hbWUiOm51bGwsInN1cm5hbWUiOm51bGwsImlhdCI6MTY2NDM1Nzc3NCwiZXhwIjoxMTY2NDM1Nzc3NCwiaXNzIjoiZ29sZW1pbyIsImp0aSI6IjU1MWMxM2I2LTZiYzktNDg4My05NTNmLTA0MWRkNmYwZjVjOCJ9.jss-5Fw6bCRxVWZuzm4Og2D353afsmcAyDxkxMWCdik'}

fromm = '{{ dag_run.conf["from"]}}'
to = '{{ dag_run.conf["to"]}}'
source_id = '{{ dag_run.conf["source_id"]}}'
#source = f'&sourceId={source_id}'
source = f'&sourceId=543015'

def_entity = 'parking_measurements_history'
def_endpoint = 'parking/measurements'
def_query = f'?source=TSK{source}&limit=10000&from={fromm}&to={to}'
def_conn_id = "mysql-db"

jsondata = f'/tmp/{def_entity}.csv'
file = f'/tmp/{def_entity}.csv'
if os.path.exists(file):
    with open (file,'r') as f:
        jsondata = json.loads(f.read())
    jsondata = str((jsondata)).replace('\'','\"').replace('None','null').replace('True', 'true').replace('False', 'false')

# Define the DAG
dag = DAG(
    dag_id=def_entity,
    start_date=datetime(2023, 3, 12),
    schedule_interval=None,
    catchup=False,
    template_searchpath=["/home/melicharovykrecek/diploma/sql"]
)

# Define a Python function to check if the endpoint is available
def check_endpoint(headers, endpoint, query):
    url = f'https://api.golemio.cz/v2/{endpoint}{query}'
    logging.info(url)
    response = requests.get(url,headers=headers)
    if response.status_code != 200:
        raise ValueError('Endpoint not available')

# Define a Python function to download the file
def download_file(headers, endpoint, query, filename):
    url = f'https://api.golemio.cz/v2/{endpoint}{query}'
    response = requests.get(url, headers=headers, stream=True)
    with open(f'/tmp/{filename}.csv', 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

# Define the function to decide which task to execute next
def timedelta_fn(source_id, fromm, to, **kwargs):
    fromm = datetime.strptime(fromm, "%Y-%m-%dT%H:%M:%S.%fZ") + timedelta(hours=12)
    frommm = fromm.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    to = datetime.strptime(to, "%Y-%m-%dT%H:%M:%S.%fZ") + timedelta(hours=12)
    too = to.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    logging.info(frommm)
    logging.info(too)
    logging.info(source_id)
    TriggerDagRunOperator(
        task_id='trigger_next_dag_run',
        trigger_dag_id=def_entity,
        conf={
            'from': frommm,
            'to': too,
            'source_id': source_id
        }
    ).execute(context=kwargs)

# Define a task to check if the endpoint is available
check_endpoint_task = PythonOperator(
    task_id='check_endpoint',
    python_callable=check_endpoint,
    op_kwargs={
        'headers': headers,
        'endpoint': def_endpoint,
        'query': def_query,
        'filename': def_entity
    },
    dag=dag,
)

# Define a task to download the file
download_file_task = PythonOperator(
    task_id='download_file',
    python_callable=download_file,
    op_kwargs={
        'headers': headers,
        'endpoint': def_endpoint,
        'query': def_query,
        'filename': def_entity
    },
    dag=dag,
)

create_table = MySqlOperator(
    sql=f'/{def_entity}/create_table.sql',
    task_id=f"create_table_{def_entity}",
    mysql_conn_id=def_conn_id,
)

insert_values = MySqlOperator(
    sql=f"INSERT INTO pre_{def_entity} VALUES (\'0\',\'{jsondata}\',\' 0 \',\'"+str(datetime.now())+"\',\'0\');",
    task_id=f"insert_values_{def_entity}",
    mysql_conn_id=def_conn_id,
)

src_table = MySqlOperator(
    sql=f'/{def_entity}/010_src.sql',
    task_id=f"src_table_{def_entity}",
    mysql_conn_id=def_conn_id,
)

stg_table = MySqlOperator(
    sql=f'/{def_entity}/020_stg.sql',
    task_id=f"stg_table_{def_entity}",
    mysql_conn_id=def_conn_id,
)

timedelta_add = PythonOperator(
    task_id='timedelta_add',
    python_callable=timedelta_fn,
    op_kwargs={
        'source_id': source_id,
        'fromm': fromm,
        'to': to
    },
    dag=dag,
)

# Set task dependencies
check_endpoint_task >> download_file_task >> create_table >> insert_values >> src_table >> stg_table >> timedelta_add