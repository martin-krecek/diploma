import requests
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.mysql_operator import MySqlOperator
from datetime import datetime
import datetime
import json
import os

def_entity = 'weather_archive'
def_endpoint = 'archive'

current_date = datetime.date.today()
start_date = current_date -  datetime.timedelta(days=4)
end_date = current_date -  datetime.timedelta(days=10)

def_query = f'?latitude=50.09&longitude=14.42&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,precipitation,rain,snowfall&daily=sunrise&timezone=Europe%2FBerlin'
def_conn_id = "mysql-db"

jsondata = f'/tmp/{def_entity}.csv'
file = f'/tmp/{def_entity}.csv'
if os.path.exists(file):
    with open (file,'r') as f:
        jsondata = json.loads(f.read())

    hourly_time = jsondata['hourly']['time']
    # Extract remaining data from hourly data
    precipitation = jsondata['hourly']['precipitation']
    rain = jsondata['hourly']['rain']
    snow = jsondata['hourly']['snowfall']
    temperature = jsondata['hourly']['temperature_2m']
    data = []
    for i in range(len(hourly_time)):
            data.append([hourly_time[i], precipitation[i], rain[i], snow[i], temperature[i]])

    jsondata = str((data)).replace('\'','\"').replace('None','null').replace('True', 'true').replace('False', 'false')

# Define the DAG
dag = DAG(
    dag_id=def_entity,
    start_date=datetime(2023, 3, 12),
    schedule_interval=None,
    catchup=False,
    template_searchpath=["/home/melicharovykrecek/diploma/sql"]
)

# Define a Python function to check if the endpoint is available
def check_endpoint(endpoint, query):
    url = f'https://archive-api.open-meteo.com/v1/{endpoint}{query}'
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError('Endpoint not available')

# Define a Python function to download the file
def download_file(endpoint, query, filename):
    url = f'https://archive-api.open-meteo.com/v1/{endpoint}{query}'
    response = requests.get(url, stream=True)
    with open(f'/tmp/{filename}.csv', 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

# Define a task to check if the endpoint is available
check_endpoint_task = PythonOperator(
    task_id='check_endpoint',
    python_callable=check_endpoint,
    op_kwargs={
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
        'endpoint': def_endpoint,
        'query': def_query,
        'filename': def_entity
    },
    dag=dag,
)

create_table = MySqlOperator(
    sql=f'/weather/{def_endpoint}/create_table.sql',
    task_id=f"create_table_{def_entity}",
    mysql_conn_id=def_conn_id,
)

insert_values = MySqlOperator(
    sql=f"INSERT INTO pre_{def_entity} VALUES (\'0\',\'{jsondata}\',\' 0 \',\'"+str(datetime.now())+"\',\'0\');",
    task_id=f"insert_values_{def_entity}",
    mysql_conn_id=def_conn_id,
)

src_table = MySqlOperator(
    sql=f'/weather/{def_endpoint}/010_src.sql',
    task_id=f"src_table_{def_entity}",
    mysql_conn_id=def_conn_id,
)

stg_table = MySqlOperator(
    sql=f'/weather/{def_endpoint}/020_stg.sql',
    task_id=f"stg_table_{def_entity}",
    mysql_conn_id=def_conn_id,
)

# Set task dependencies
check_endpoint_task >> download_file_task >> create_table >> insert_values >> src_table >> stg_table
