
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

import requests
import json

headers = {"Content-Type": "application/json; charset=utf-8", "x-access-token": 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1hcnRpbmtyZWNlazlAZ21haWwuY29tIiwiaWQiOjE0NTIsIm5hbWUiOm51bGwsInN1cm5hbWUiOm51bGwsImlhdCI6MTY2NDM1Nzc3NCwiZXhwIjoxMTY2NDM1Nzc3NCwiaXNzIjoiZ29sZW1pbyIsImp0aSI6IjU1MWMxM2I2LTZiYzktNDg4My05NTNmLTA0MWRkNmYwZjVjOCJ9.jss-5Fw6bCRxVWZuzm4Og2D353afsmcAyDxkxMWCdik'}

def get_json_data(headers, endpoint, query):
    url = f'https://api.golemio.cz/v2/{endpoint}{query}'
    print(url)
    response=requests.get(url, headers=headers)
    data = response.json()
    with open(f'/opt/airflow/data/{endpoint}.json', 'w') as f:
        f.write(json.dumps(data, sort_keys=True, indent=4))
    print(data)
    return data



default_args = {
    'headers': {'Content-Type': 'application/json; charset=utf-8', 'x-access-token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1hcnRpbmtyZWNlazlAZ21haWwuY29tIiwiaWQiOjE0NTIsIm5hbWUiOm51bGwsInN1cm5hbWUiOm51bGwsImlhdCI6MTY2NDM1Nzc3NCwiZXhwIjoxMTY2NDM1Nzc3NCwiaXNzIjoiZ29sZW1pbyIsImp0aSI6IjU1MWMxM2I2LTZiYzktNDg4My05NTNmLTA0MWRkNmYwZjVjOCJ9.jss-5Fw6bCRxVWZuzm4Og2D353afsmcAyDxkxMWCdik'}
}

with DAG(
    dag_id="parking_measurements",
    start_date=datetime(2023,12,3),
    schedule_interval='@daily',
) as dag:

    api_parking_measurements = PythonOperator(
        task_id="api_parking_measurements",
        python_callable=get_json_data,
        op_kwargs={
            'headers': headers,
            'endpoint': 'parking/measurements',
            'query': '?source=TSK&limit=10000'
        }
    )
    

    api_parking_measurements
