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
    eight_days_ago = current_date - datetime.timedelta(days=8)
    seven_days_ago = current_date - datetime.timedelta(days=7)
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
        n_input_day = n_input
        for i in range(len(test_predict)):
        #for i in range(7):
            # predict the week
            yhat_sequence = forecast(model, history, n_input_day)

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
            hours = [num for num in range(1, 25+i*24)]

            pyplot.plot(hours, arr, marker='o', label='lstm')
            pyplot.show()

        # evaluate predictions days for each week
        predictions = array(predictions)
        np.savetxt(f'diploma/predictions/input/predictions_{table}_{parking_id}_{current_date}.csv', predictions, delimiter=',', fmt='%.1f')

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












def run_file_merge(table, parking_id):
    from datetime import date, datetime, timedelta
    import csv

    current_date = date.today()

    # Open the CSV file for reading
    with open(f'diploma/predictions/input/predictions_{table}_{parking_id}_{current_date}.csv', 'r') as file:
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
        modified_rows.append([timestamp, value])
        start_datetime += increment

    # Open a new CSV file for writing
    with open(f'diploma/predictions/output/predictions_{table}_{parking_id}_{current_date}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['Timestamp', 'Value'])
        
        # Write each modified row to the CSV file
        writer.writerows(modified_rows)




# Define the DAG
dag = DAG(
    dag_id='model_predict_all',
    start_date=datetime(2023, 3, 12),
    schedule_interval=None,
    catchup=False,
    template_searchpath=["/home/melicharovykrecek/diploma/sql"]
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

file_merge_tas_1 = PythonOperator(
    task_id='file_merge_tas_1',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534017'
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

file_merge_tas_2 = PythonOperator(
    task_id='file_merge_tas_2',
    python_callable=run_file_merge,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534016'
    },
    dag=dag
)

# Set task dependencies
model_predict_all_task_1 >> file_merge_tas_1 >> model_predict_all_task_2 >>file_merge_tas_2