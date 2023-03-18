import requests
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.mysql_operator import MySqlOperator
from datetime import datetime, timedelta
import json
import os

headers = {"Content-Type": "application/json; charset=utf-8", "x-access-token": 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1hcnRpbmtyZWNlazlAZ21haWwuY29tIiwiaWQiOjE0NTIsIm5hbWUiOm51bGwsInN1cm5hbWUiOm51bGwsImlhdCI6MTY2NDM1Nzc3NCwiZXhwIjoxMTY2NDM1Nzc3NCwiaXNzIjoiZ29sZW1pbyIsImp0aSI6IjU1MWMxM2I2LTZiYzktNDg4My05NTNmLTA0MWRkNmYwZjVjOCJ9.jss-5Fw6bCRxVWZuzm4Og2D353afsmcAyDxkxMWCdik'}

jsondata_parking_measurements = '/tmp/parking_measurements.csv'
file = '/tmp/parking_measurements.csv'
if os.path.exists(file):
    with open (file,'r') as f:
        jsondata_parking_measurements = json.loads(f.read())
    jsondata_parking_measurements = str((jsondata_parking_measurements)).replace('\'','\"').replace('None','null').replace('True', 'true').replace('False', 'false')

# Define the DAG
dag = DAG(
    dag_id='get_file_from_endpoint',
    start_date=datetime(2023, 3, 12),
    schedule_interval='0 * * * *',
    catchup=False,
    template_searchpath=["/home/melicharovykrecek/diploma/sql"]
)

# Define a Python function to check if the endpoint is available
def check_endpoint():
    url = 'https://api.golemio.cz/v2/parking/measurements?source=TSK&limit=10000'
    response = requests.get(url,headers=headers)
    if response.status_code != 200:
        raise ValueError('Endpoint not available')

# Define a Python function to download the file
def download_file(headers, endpoint, query):
    url = f'https://api.golemio.cz/v2/{endpoint}{query}'
    response = requests.get(url, headers=headers, stream=True)
    with open('/tmp/parking_measurements.csv', 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

# Define a task to check if the endpoint is available
check_endpoint_task = PythonOperator(
    task_id='check_endpoint',
    python_callable=check_endpoint,
    dag=dag,
)

# Define a task to download the file
download_file_task = PythonOperator(
    task_id='download_file',
    python_callable=download_file,
    op_kwargs={
        'headers': headers,
        'endpoint': 'parking/measurements',
        'query': '?source=TSK&limit=10000'
    },
    dag=dag,
)

create_table_parking_measurements = MySqlOperator(
    sql='/parking/measurements/create_table.sql',
    task_id="create_table_parking_measurements",
    mysql_conn_id="mysql-db",
)

insert_values_parking_measurements = MySqlOperator(
    sql=f"INSERT INTO pre_parking_measurements VALUES (\'0\',\'{jsondata_parking_measurements}\',\' 0 \',\'"+str(datetime.now())+"\',\'1\');",
    task_id="insert_values_parking_measurements",
    mysql_conn_id="mysql-db",
)

src_table_parking_measurements = MySqlOperator(
    sql='/parking/measurements/010_src.sql',
    task_id="src_table_parking_measurements",
    mysql_conn_id="mysql-db",
)

stg_table_parking_measurements = MySqlOperator(
    sql='/parking/measurements/020_stg.sql',
    task_id="stg_table_parking_measurements",
    mysql_conn_id="mysql-db",
)

# Set task dependencies
check_endpoint_task >> download_file_task >> create_table_parking_measurements >> insert_values_parking_measurements >> src_table_parking_measurements >> stg_table_parking_measurements
