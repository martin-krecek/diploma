import requests
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.mysql_operator import MySqlOperator
from datetime import datetime, timedelta
import json
import os

headers = {"Content-Type": "application/json; charset=utf-8", "x-access-token": 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1hcnRpbmtyZWNlazlAZ21haWwuY29tIiwiaWQiOjE0NTIsIm5hbWUiOm51bGwsInN1cm5hbWUiOm51bGwsImlhdCI6MTY2NDM1Nzc3NCwiZXhwIjoxMTY2NDM1Nzc3NCwiaXNzIjoiZ29sZW1pbyIsImp0aSI6IjU1MWMxM2I2LTZiYzktNDg4My05NTNmLTA0MWRkNmYwZjVjOCJ9.jss-5Fw6bCRxVWZuzm4Og2D353afsmcAyDxkxMWCdik'}

def_entity = 'meteosensors'
def_endpoint = 'meteosensors'
def_query = ''
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
    schedule_interval='0 * * * *',
    catchup=False,
    template_searchpath=["/home/melicharovykrecek/diploma/sql"]
)

# Define a Python function to check if the endpoint is available
def check_endpoint(headers, endpoint, query):
    url = f'https://api.golemio.cz/v2/{endpoint}{query}'
    response = requests.get(url,headers=headers)
    if response.status_code != 200:
        raise ValueError('Endpoint not available')

# Define a Python function to download the file
def download_file(headers, endpoint, query, filename):
    url = f'https://api.golemio.cz/v2/{endpoint}{query}'
    response = requests.get(url, headers=headers, stream=True)
    response = response['features']
    with open(f'/tmp/{filename}.csv', 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

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
    sql=f'/{def_endpoint}/create_table.sql',
    task_id=f"create_table_{def_entity}",
    mysql_conn_id=def_conn_id,
)

insert_values = MySqlOperator(
    sql=f"INSERT INTO pre_{def_entity} VALUES (\'0\',\'{jsondata}\',\' 0 \',\'"+str(datetime.now())+"\',\'0\');",
    task_id=f"insert_values_{def_entity}",
    mysql_conn_id=def_conn_id,
)

src_table = MySqlOperator(
    sql=f'/{def_endpoint}/010_src.sql',
    task_id=f"src_table_{def_entity}",
    mysql_conn_id=def_conn_id,
)

stg_table = MySqlOperator(
    sql=f'/{def_endpoint}/020_stg.sql',
    task_id=f"stg_table_{def_entity}",
    mysql_conn_id=def_conn_id,
)

# Set task dependencies
check_endpoint_task >> download_file_task >> create_table >> insert_values >> src_table >> stg_table
