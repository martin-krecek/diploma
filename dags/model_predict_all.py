import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import subprocess

def run_model_predict_all(table, parking_id):
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
    seven_days_forward = current_date + datetime.timedelta(days=7)

    query_predict = f"SELECT date_modified, occupied_spot_number FROM out_{table} WHERE parking_id = '{parking_id}' AND date_modified >= '{eight_days_ago}' AND date_modified < '{one_day_ago}';"
    query_predict_weather = f"SELECT time_ts_shifted as time_ts, temperature, precipitation FROM out_weather WHERE time_ts >= '{current_date}' AND time_ts < '{seven_days_forward}';"

    df_predict = pd.read_sql(query_predict, conn)

    # -------------------------------------------------------------------#
    # ----------- PREDICTION DATA
    # -------------------------------------------------------------------#
    # Preprocessing df_predict
    df_predict['date_modified'] = pd.to_datetime(df_predict['date_modified'])
    df_predict['occupied_spot_number'] = df_predict['occupied_spot_number'].astype(str).astype(int)

    df_predict.set_index('date_modified', inplace=True)

    # Define the rolling window size
    window_size_mean = 15
    window_size_std = 15
    # Calculate the rolling mean and standard deviation
    rolling_mean = df_predict.rolling(window=window_size_mean).mean()
    rolling_std = df_predict.rolling(window=window_size_std).std()

    rolling_std = rolling_std.replace(0, 1)
    rolling_mean = rolling_mean.rename(columns={'occupied_spot_number': 'rolling_mean'})
    rolling_std = rolling_std.rename(columns={'occupied_spot_number': 'rolling_std'})
    result = df_predict.join([rolling_mean, rolling_std], how='outer')
    z_scores = (result['occupied_spot_number'] - result['rolling_mean']) / result['rolling_std']
    # Threshold for outlier detection
    threshold = 3

    # Find the outliers using the threshold
    outliers = np.abs(z_scores) > threshold

    # Remove the outliers from the data
    df_predict = df_predict.mask(outliers)

    # Resampling df_predict to hourly frequency

    df_predict = df_predict.resample('H').mean().interpolate(method='linear')

    idx = pd.date_range(start=f'{eight_days_ago} 00:00:00', end=f'{two_days_ago} 23:00:00', freq='H')
    df_hourly = df_predict.reindex(idx)
    # Interpolate the missing values
    df_predict = df_hourly.interpolate()
    # Reset the index back to a column
    df_predict.reset_index(inplace=True)
    df_predict.set_index('index', inplace=True)

    # set values from 10pm to 5am to 0
    df_predict.loc[(df_predict.index.hour >= 22) | (df_predict.index.hour < 5), 'occupied_spot_number'] = 0

    df_predict['Monday'] = df_predict.index.dayofweek == 0
    df_predict['Tuesday'] = df_predict.index.dayofweek == 1
    df_predict['Wednesday'] = df_predict.index.dayofweek == 2
    df_predict['Thursday'] = df_predict.index.dayofweek == 3
    df_predict['Friday'] = df_predict.index.dayofweek == 4
    df_predict['Saturday'] = df_predict.index.dayofweek == 5
    df_predict['Sunday'] = df_predict.index.dayofweek == 6

    df_predict[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']] = df_predict[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']].astype(int)

    # ------------------------------------------------------------------------#


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
        model = load_model(f'diploma/models/model_{table}_{parking_id}.h5')
        # history is a list of weekly data
        history = [x for x in test_predict]
        # walk-forward validation over each week
        predictions = list()
        predictions_normalized = list()
        n_input_day = n_input
        for i in range(len(test_predict)):
        #for i in range(7):
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
        np.savetxt(f'diploma/streamlit/predictions/input/predictions_{table}_{parking_id}_{current_date}.csv', predictions, delimiter=',', fmt='%.1f')
        predictions_normalized = array(predictions_normalized)
        np.savetxt(f'diploma/streamlit/predictions/input/predictions/normalized_{table}_{parking_id}_{current_date}.csv', predictions_normalized, delimiter=',', fmt='%.2f')

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
    columns_to_normalize = ['occupied_spot_number']
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












def run_file_merge(table, parking_id, latitude, longitude, name):
    from datetime import date, datetime, timedelta
    import csv

    current_date = date.today()

    # Open the CSV file for reading
    with open(f'diploma/streamlit/predictions/input/predictions_{table}_{parking_id}_{current_date}.csv', 'r') as file:
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
        lat = latitude
        lon = longitude
        value = int(float(value))
        modified_rows.append([timestamp, value, lat, lon, name])
        start_datetime += increment

    # Open a new CSV file for writing
    with open(f'diploma/streamlit/predictions/output/predictions_{table}_{parking_id}_{current_date}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['timestamp', 'value', 'lat', 'lon', 'name'])
        
        # Write each modified row to the CSV file
        writer.writerows(modified_rows)

    # Create 1 consolidated file with all data
    with open(f'diploma/streamlit/predictions/output/predictions_{table}_{current_date}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # Write the header - for 1st parking lot
        if parking_id == 'tsk-534017':
            writer.writerow(['timestamp', 'value', 'lat', 'lon', 'name'])
        # Write each modified row to the CSV file
        writer.writerows(modified_rows)




    # Open the CSV file for reading
    with open(f'diploma/streamlit/predictions/input/predictions/normalized_{table}_{parking_id}_{current_date}.csv', 'r') as file:
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
        lat = latitude
        lon = longitude
        value = float(value)
        modified_rows.append([timestamp, value])
        start_datetime += increment

    # Open a new CSV file for writing
    with open(f'diploma/streamlit/predictions/output/predictions/normalized_{table}_{parking_id}_{current_date}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['timestamp', 'value'])
        
        # Write each modified row to the CSV file
        writer.writerows(modified_rows)

    # Create 1 consolidated file with all data
    with open(f'diploma/streamlit/predictions/output/predictions/normalized_{table}_{current_date}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # Write the header - for 1st parking lot
        if parking_id == 'tsk-534017':
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
    dag_id='model_predict_all',
    start_date=datetime(2023, 3, 12),
    schedule_interval='0 4 * * *',
    catchup=False,
    template_searchpath=["/home/melicharovykrecek/diploma/sql"]
)

remove_files = PythonOperator(
    task_id='remove_files',
    python_callable=run_remove_files,
    op_kwargs={
        'table': 'parking_measurements'
    },
    dag=dag   
)

model_predict_all_task_1 = PythonOperator(
    task_id='model_predict_all_task_1',
    python_callable=run_model_predict_all,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534017'
    },
    dag=dag
)

file_merge_task_1 = PythonOperator(
    task_id='file_merge_task_1',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534017',
        'latitude': '50.032074',
        'longitude': '14.492015',
        'name': 'Chodov'
    },
    dag=dag
)

model_predict_all_task_2 = PythonOperator(
    task_id='model_predict_all_task_2',
    python_callable=run_model_predict_all,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534016'
    },
    dag=dag
)

file_merge_task_2 = PythonOperator(
    task_id='file_merge_task_2',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534016',
        'latitude': '50.125168',
        'longitude': '14.514741',
        'name': 'Letnany'
    },
    dag=dag
)

model_predict_all_task_3 = PythonOperator(
    task_id='model_predict_all_task_3',
    python_callable=run_model_predict_all,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534015'
    },
    dag=dag,
)

file_merge_task_3 = PythonOperator(
    task_id='file_merge_task_3',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534015',
        'latitude': '50.07622',
        'longitude': '14.517897',
        'name': 'Depo Hostivar'
    },
    dag=dag
)

model_predict_all_task_5 = PythonOperator(
    task_id='model_predict_all_task_5',
    python_callable=run_model_predict_all,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534012'
    },
    dag=dag,
)

file_merge_task_5 = PythonOperator(
    task_id='file_merge_task_5',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534012',
        'latitude': '50.069622',
        'longitude': '14.507447',
        'name': 'Skalka'
    },
    dag=dag
)

model_predict_all_task_6 = PythonOperator(
    task_id='model_predict_all_task_6',
    python_callable=run_model_predict_all,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534011'
    },
    dag=dag,
)

file_merge_task_6 = PythonOperator(
    task_id='file_merge_task_6',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534011',
        'latitude': '50.110535',
        'longitude': '14.582672',
        'name': 'Cerny Most'
    },
    dag=dag
)

model_predict_all_task_8 = PythonOperator(
    task_id='model_predict_all_task_8',
    python_callable=run_model_predict_all,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534008'
    },
    dag=dag,
)

file_merge_task_8 = PythonOperator(
    task_id='file_merge_task_8',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534008',
        'latitude': '50.054527',
        'longitude': '14.28977',
        'name': 'Zlicin 1'
    },
    dag=dag
)

model_predict_all_task_9 = PythonOperator(
    task_id='model_predict_all_task_9',
    python_callable=run_model_predict_all,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534005'
    },
    dag=dag,
)

file_merge_task_9 = PythonOperator(
    task_id='file_merge_task_9',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534005',
        'latitude': '50.02662',
        'longitude': '14.509208',
        'name': 'Opatov'
    },
    dag=dag
)

model_predict_all_task_11 = PythonOperator(
    task_id='model_predict_all_task_11',
    python_callable=run_model_predict_all,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534002'
    },
    dag=dag,
)

file_merge_task_11 = PythonOperator(
    task_id='file_merge_task_11',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534002',
        'latitude': '50.109318',
        'longitude': '14.441252',
        'name': 'Holesovice'
    },
    dag=dag
)

# Set task dependencies
model_predict_all_task_1 >> file_merge_task_1 >> model_predict_all_task_2 >> file_merge_task_2 >> model_predict_all_task_3 >> file_merge_task_3 >> model_predict_all_task_5 >> file_merge_task_5 >> model_predict_all_task_6 >> file_merge_task_6 >> model_predict_all_task_8 >> file_merge_task_8 >> model_predict_all_task_9 >> file_merge_task_9 >> model_predict_all_task_11 >> file_merge_task_11