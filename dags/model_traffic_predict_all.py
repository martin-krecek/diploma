import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import subprocess

def run_python_script(table, gtfs_route_id):
    #
    # PREDICT - best results so far
    # WEEKLY PREDICTION (CANNOT BE DIFFERENT - THERE ARE SOME TIMESTAMP CALCULATED IN WEATHER TABLE FOR EXACTLY 1 WEEK PREDICTION)
    # Weather 7 days in the future
    # Parking 7 days back
    # Prediction should take place on Monday (8th) AM hours
    #   - weather data - 8th to 15th (7 days into future)
    #   - parking data - 1st to 7th (7 days back)
    # https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

    import mysql.connector
    import pandas as pd
    from math import sqrt
    import numpy as np
    from keras.models import Sequential
    # univariate multi-step lstm for the power usage dataset
    from numpy import split, array
    from sklearn.metrics import mean_squared_error
    from matplotlib import pyplot
    from keras.layers import Dense, LSTM
    import matplotlib.pyplot as plt
    from tensorflow.keras.optimizers import Adam
    from keras.models import load_model
    import datetime
    from sklearn.preprocessing import MinMaxScaler

    # db parameters
    user = 'martin'
    pw = 'admin'
    db = 'mysql'    
    host = '34.77.219.108'

    # establishing the connection
    conn = mysql.connector.connect(user=user, password=pw, host=host, database=db)

    current_date = datetime.date.today()
    one_day_ago = current_date - datetime.timedelta(days=1)
    two_days_ago = current_date - datetime.timedelta(days=2)
    eight_days_ago = current_date - datetime.timedelta(days=8)
    seven_days_ago = current_date - datetime.timedelta(days=7)
    seven_days_forward = current_date + datetime.timedelta(days=7)

    # Get the current date
    current_date = datetime.date.today()
    # Calculate the offset to Monday (0 represents Monday, 1 represents Tuesday, and so on)
    offset_to_monday = (current_date.weekday() - 0) % 7
    # Calculate the date of Monday
    monday_date = current_date - datetime.timedelta(days=offset_to_monday)

    monday_date_minus_one = monday_date - datetime.timedelta(days=1)

    # Reading from db to dataframe
    query_predict = f"SELECT origin_timestamp, actual FROM stg_{table} where gtfs_route_id = '{gtfs_route_id}' AND origin_timestamp > '{eight_days_ago}' AND origin_timestamp < '{one_day_ago}' ORDER BY origin_timestamp;"
    query_predict_weather = f"SELECT time_ts_shifted as time_ts, temperature, precipitation FROM out_weather WHERE time_ts >= '{current_date}' AND time_ts < '{seven_days_forward}';"

    df_predict = pd.read_sql(query_predict, conn)

    # -------------------------------------------------------------------#
    # ----------- PREDICTION DATA
    # -------------------------------------------------------------------#
    df_predict['origin_timestamp'] = pd.to_datetime(df_predict['origin_timestamp'])
    df_predict['actual'] = df_predict['actual'].astype(str).astype(int)

    df_predict.set_index('origin_timestamp', inplace=True)

    # Resampling df to hourly frequency
    df_predict = df_predict.resample('H').mean().interpolate(method='linear')

    idx = pd.date_range(start=f'{eight_days_ago} 00:00:00', end=f'{two_days_ago} 23:00:00', freq='H')
    df_hourly = df_predict.reindex(idx)
    # Interpolate the missing values
    df_predict = df_hourly.interpolate()
    # Reset the index back to a column
    df_predict.reset_index(inplace=True)
    df_predict.set_index('index', inplace=True)

    # set values from 10pm to 5am to 0
    df_predict.loc[(df_predict.index.hour >= 22) | (df_predict.index.hour < 5), 'actual'] = 0

    df_predict['Monday'] = df_predict.index.dayofweek == 0
    df_predict['Tuesday'] = df_predict.index.dayofweek == 1
    df_predict['Wednesday'] = df_predict.index.dayofweek == 2
    df_predict['Thursday'] = df_predict.index.dayofweek == 3
    df_predict['Friday'] = df_predict.index.dayofweek == 4
    df_predict['Saturday'] = df_predict.index.dayofweek == 5
    df_predict['Sunday'] = df_predict.index.dayofweek == 6

    df_predict[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']] = df_predict[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']].astype(int)

    df_weather_predict = pd.read_sql(query_predict_weather, conn)

    # -------------------------------------------------------------------#
    # ----------- PREDICTION DATA
    # -------------------------------------------------------------------#
    # Preprocessing df_weather_predict
    df_weather_predict['time_ts'] = pd.to_datetime(df_weather_predict['time_ts'])

    df_weather_predict['temperature'] = df_weather_predict['temperature'].astype(str).astype(float)
    df_weather_predict['precipitation'] = df_weather_predict['precipitation'].astype(str).astype(float)

    df_weather_predict.set_index('time_ts', inplace=True)
    df_weather_predict = df_weather_predict.resample('H').mean().interpolate(method='linear')
    df_predict = df_predict.join(df_weather_predict, how='inner')

    # make a forecast
    def forecast(model, history, n_input_day):
        # flatten data
        data = array(history)
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        # retrieve last observations for input data
        input_x = data[:n_input_day, :]
        # reshape into [1, n_input, n]
        input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
        # forecast the next week
        yhat = model.predict(input_x, verbose=0)
        # we only want the vector forecast
        yhat = yhat[0]
        return yhat

    # evaluate a single model
    def evaluate_model(test_predict, n_input):
        # fit model
        model = load_model(f'diploma/models/model_{table}_{gtfs_route_id}.h5')
        # history is a list of weekly data
        history = [x for x in test_predict]
        # walk-forward validation over each week
        predictions = list()
        predictions_normalized = list()
        n_input_day = n_input
        for i in range(len(test_predict)):
            # predict the week
            yhat_sequence = forecast(model, history, n_input_day)
            predictions_normalized.append(yhat_sequence)
            # Step 1: Reshape the normalized values to match the scaler's expectations
            normalized_values = np.array(yhat_sequence).reshape(-1, 1)
            # Step 2: Perform inverse normalization
            inverse_normalized_values = scaler.inverse_transform(normalized_values)
            # Step 3: Convert the inverse normalized values back to a list
            yhat_sequence = inverse_normalized_values.flatten().tolist()
            predictions.append(yhat_sequence)
            history.pop(0)
            arr = np.array(predictions)
            arr = arr.reshape(24+i*24,)
        # evaluate predictions days for each week
        predictions = array(predictions)
        np.savetxt(f'diploma/streamlit/predictions/input/predictions_{table}_{gtfs_route_id}_{current_date}.csv', predictions, delimiter=',', fmt='%.1f')
        predictions_normalized = array(predictions_normalized)
        np.savetxt(f'diploma/streamlit/predictions/input/predictions_normalized_{table}_{gtfs_route_id}_{current_date}.csv', predictions_normalized, delimiter=',', fmt='%.1f')

        return predictions

    # Create an instance of MinMaxScaler
    scaler_main = MinMaxScaler()
    # Fit the scaler to your data
    # Select the columns to be normalized
    columns_to_normalize = ['temperature', 'precipitation']
    # Fit the scaler to your data
    scaler_main.fit(df_predict[columns_to_normalize])
    # Transform the selected columns
    df_predict[columns_to_normalize] = scaler_main.transform(df_predict[columns_to_normalize])

    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()
    # Fit the scaler to your data
    # Select the columns to be normalized
    columns_to_normalize = ['actual']
    # Fit the scaler to your data
    scaler.fit(df_predict[columns_to_normalize])
    # Transform the selected columns
    df_predict[columns_to_normalize] = scaler.transform(df_predict[columns_to_normalize])



    # split a univariate dataset into train/test sets
    def split_dataset_preidct(data):
        test = array(split(data, len(data)/24))
        return test
    # load the new file
    dataset_predict = df_predict
    # split into train and test
    test_predict = split_dataset_preidct(dataset_predict.values)

    n_input = 24

    predictions = evaluate_model(test_predict, n_input)



def run_file_merge(table, gtfs_route_id):
    from datetime import date, datetime, timedelta
    import csv

    current_date = date.today()

    # Open the CSV file for reading
    with open(f'diploma/streamlit/predictions/input/predictions_{table}_{gtfs_route_id}_{current_date}.csv', 'r') as file:
        reader = csv.reader(file)
        
        # Read all rows into a list
        rows = list(reader)

    # Merge values into one row
    merged_row = []
    for row in rows:
        merged_row.extend(row)

    # Starting datetime (tomorrow at 00:00)
    start_datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    # Increment in hours
    increment = timedelta(hours=1)

    # Create a list to store the modified rows
    modified_rows = []

    # Add timestamp to every row and set value to 0 between 22:00 and 5:00
    for value in merged_row:
        timestamp = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
        if start_datetime.hour >= 22 or start_datetime.hour < 5:
            value = 0
        value = int(float(value))
        modified_rows.append([timestamp, value])
        start_datetime += increment

    # Open a new CSV file for writing
    with open(f'diploma/streamlit/predictions/output/predictions_{table}_{gtfs_route_id}_{current_date}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['timestamp', 'value'])
        
        # Write each modified row to the CSV file
        writer.writerows(modified_rows)

    # Create 1 consolidated file with all data
    with open(f'diploma/streamlit/predictions/output/predictions_{table}_{current_date}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # Write the header - for 1st parking lot
        if gtfs_route_id == 'L1':
            writer.writerow(['timestamp', 'value'])
        # Write each modified row to the CSV file
        writer.writerows(modified_rows)



    # Open the CSV file for reading
    with open(f'diploma/streamlit/predictions/input/predictions_normalized_{table}_{gtfs_route_id}_{current_date}.csv', 'r') as file:
        reader = csv.reader(file)
        
        # Read all rows into a list
        rows = list(reader)

    # Merge values into one row
    merged_row = []
    for row in rows:
        merged_row.extend(row)

    # Starting datetime (tomorrow at 00:00)
    start_datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    # Increment in hours
    increment = timedelta(hours=1)

    # Create a list to store the modified rows
    modified_rows = []

    # Add timestamp to every row and set value to 0 between 22:00 and 5:00
    for value in merged_row:
        timestamp = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
        if start_datetime.hour >= 22 or start_datetime.hour < 5:
            value = 0
        value = float(value)
        modified_rows.append([timestamp, value])
        start_datetime += increment

    # Open a new CSV file for writing
    with open(f'diploma/streamlit/predictions/output/predictions_normalized_{table}_{gtfs_route_id}_{current_date}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['timestamp', 'value'])
        
        # Write each modified row to the CSV file
        writer.writerows(modified_rows)

    # Create 1 consolidated file with all data
    with open(f'diploma/streamlit/predictions/output/predictions_normalized_{table}_{current_date}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # Write the header - for 1st parking lot
        if gtfs_route_id == 'L1':
            writer.writerow(['timestamp', 'value'])
        # Write each modified row to the CSV file
        writer.writerows(modified_rows)




def run_remove_files(table):
    from datetime import date

    current_date = date.today()

    file_path = f'diploma/streamlit/predictions/output/predictions_{table}_{current_date}.csv'
    if os.path.exists(file_path):
        os.remove(file_path) 

# Define the DAG
dag = DAG(
    dag_id='model_traffic_predict_all',
    start_date=datetime(2023, 3, 12),
    schedule_interval='0 3 * * *',
    catchup=False,
    template_searchpath=["/home/melicharovykrecek/diploma/sql"]
)

remove_files = PythonOperator(
    task_id='remove_files',
    python_callable=run_remove_files,
    op_kwargs={
        'table': 'vehiclepositions_model'
    },
    dag=dag   
)

model_train_1 = PythonOperator(
    task_id='model_train_1',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L1'
    },
    dag=dag,
)

file_merge_task_1 = PythonOperator(
    task_id='file_merge_task_1',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L1'
    },
    dag=dag
)

model_train_2 = PythonOperator(
    task_id='model_train_2',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L2'
    },
    dag=dag,
)

file_merge_task_2 = PythonOperator(
    task_id='file_merge_task_2',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L2'
    },
    dag=dag
)

model_train_3 = PythonOperator(
    task_id='model_train_3',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L3'
    },
    dag=dag,
)

file_merge_task_3 = PythonOperator(
    task_id='file_merge_task_3',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L3'
    },
    dag=dag
)

model_train_5 = PythonOperator(
    task_id='model_train_5',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L5'
    },
    dag=dag,
)

file_merge_task_5 = PythonOperator(
    task_id='file_merge_task_5',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L5'
    },
    dag=dag
)

model_train_6 = PythonOperator(
    task_id='model_train_6',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L6'
    },
    dag=dag,
)

file_merge_task_6 = PythonOperator(
    task_id='file_merge_task_6',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L6'
    },
    dag=dag
)

model_train_7 = PythonOperator(
    task_id='model_train_7',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L7'
    },
    dag=dag,
)

file_merge_task_7 = PythonOperator(
    task_id='file_merge_task_7',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L7'
    },
    dag=dag
)

model_train_8 = PythonOperator(
    task_id='model_train_8',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L8'
    },
    dag=dag,
)

file_merge_task_8 = PythonOperator(
    task_id='file_merge_task_8',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L8'
    },
    dag=dag
)

model_train_9 = PythonOperator(
    task_id='model_train_9',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L9'
    },
    dag=dag,
)

file_merge_task_9 = PythonOperator(
    task_id='file_merge_task_9',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L9'
    },
    dag=dag
)

model_train_10 = PythonOperator(
    task_id='model_train_10',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L10'
    },
    dag=dag,
)

file_merge_task_10 = PythonOperator(
    task_id='file_merge_task_10',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L10'
    },
    dag=dag
)

model_train_11 = PythonOperator(
    task_id='model_train_11',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L11'
    },
    dag=dag,
)

file_merge_task_11 = PythonOperator(
    task_id='file_merge_task_11',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L11'
    },
    dag=dag
)

model_train_12 = PythonOperator(
    task_id='model_train_12',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L12'
    },
    dag=dag,
)

file_merge_task_12 = PythonOperator(
    task_id='file_merge_task_12',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'vehiclepositions_model',
        'gtfs_route_id': 'L12'
    },
    dag=dag
)

# Set task dependencies
model_train_1 >> file_merge_task_1 >> model_train_2 >> file_merge_task_2 >> model_train_3 >> file_merge_task_3 >> model_train_5 >> file_merge_task_5 >> model_train_6 >> file_merge_task_6 >> model_train_7 >> file_merge_task_7 >> model_train_8 >> file_merge_task_8 >> model_train_9 >> file_merge_task_9 >> model_train_10 >> file_merge_task_10 >> model_train_11 >> file_merge_task_11  >> model_train_12 >> file_merge_task_12