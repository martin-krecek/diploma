import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import subprocess

def run_python_script(table, parking_id):

    import mysql.connector
    import pandas as pd
    from math import sqrt
    import numpy as np
    from keras.models import Sequential
    from numpy import split, array
    from sklearn.metrics import mean_squared_error
    from matplotlib import pyplot
    from keras.layers import Dense, LSTM
    import matplotlib.pyplot as plt
    from tensorflow.keras.optimizers import Adam
    import datetime
    from sklearn.preprocessing import MinMaxScaler

    # db parameters
    user = 'martin'
    pw = 'admin'
    db = 'mysql'    
    host = '34.77.219.108'

    # establishing the connection
    conn = mysql.connector.connect(user=user, password=pw, host=host, database=db)

    # Get the current date
    current_date = datetime.date.today()
    # Calculate the offset to Monday (0 represents Monday, 1 represents Tuesday, and so on)
    offset_to_monday = (current_date.weekday() - 0) % 7
    # Calculate the date of Monday
    monday_date = current_date - datetime.timedelta(days=offset_to_monday)


    # Reading from db to dataframe
    query = f"SELECT date_modified, occupied_spot_number FROM out_{table} WHERE parking_id = '{parking_id}' AND date_modified >= '2021-05-03' AND date_modified < '{monday_date}';"
    query_weather = f"SELECT time_ts, temperature, precipitation FROM out_weather WHERE time_ts >= '2021-05-03' AND time_ts < '{monday_date}';"

    df = pd.read_sql(query, conn)


    ###########################################################################################
    #                                    PARKING DATA                                         #
    ###########################################################################################
    df['date_modified'] = pd.to_datetime(df['date_modified'])
    df['occupied_spot_number'] = df['occupied_spot_number'].astype(str).astype(int)

    df.set_index('date_modified', inplace=True)

    # Define the rolling window size
    window_size_mean = 15
    window_size_std = 15
    # Calculate the rolling mean and standard deviation
    rolling_mean = df.rolling(window=window_size_mean).mean()
    rolling_std = df.rolling(window=window_size_std).std()

    rolling_std = rolling_std.replace(0, 1)
    rolling_mean = rolling_mean.rename(columns={'occupied_spot_number': 'rolling_mean'})
    rolling_std = rolling_std.rename(columns={'occupied_spot_number': 'rolling_std'})
    result = df.join([rolling_mean, rolling_std], how='outer')
    z_scores = (result['occupied_spot_number'] - result['rolling_mean']) / result['rolling_std']
    # Threshold for outlier detection
    threshold = 3

    # Find the outliers using the threshold
    outliers = np.abs(z_scores) > threshold

    # Remove the outliers from the data
    df = df.mask(outliers)

    # Resampling df to hourly frequency
    df = df.resample('H').mean().interpolate(method='linear')

    # set values from 10pm to 5am to 0
    df.loc[(df.index.hour >= 22) | (df.index.hour < 5), 'occupied_spot_number'] = 0

    df['Monday'] = df.index.dayofweek == 0
    df['Tuesday'] = df.index.dayofweek == 1
    df['Wednesday'] = df.index.dayofweek == 2
    df['Thursday'] = df.index.dayofweek == 3
    df['Friday'] = df.index.dayofweek == 4
    df['Saturday'] = df.index.dayofweek == 5
    df['Sunday'] = df.index.dayofweek == 6

    df[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']] = df[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']].astype(int)

    ###########################################################################################
    #                                    WEATHER DATA                                         #
    ###########################################################################################

    df_weather = pd.read_sql(query_weather, conn)
    #df_weather_predict = pd.read_sql(query_predict_weather, conn)

    # Preprocessing df_weather
    df_weather['time_ts'] = pd.to_datetime(df_weather['time_ts'])
    df_weather['temperature'] = df_weather['temperature'].astype(str).astype(float)
    df_weather['precipitation'] = df_weather['precipitation'].astype(str).astype(float)

    df_weather.set_index('time_ts', inplace=True)

    # Resampling df_weather to hourly frequency
    df_weather = df_weather.resample('H').mean().interpolate(method='linear') 

    ###########################################################################################
    #                             JOIN WEATHER AND PARKING DATA                               #
    ###########################################################################################

    df = df.join(df_weather, how='inner')

    ###########################################################################################
    #                                     FUNCTIONS                                           #
    ###########################################################################################

    # split a univariate dataset into train/test sets
    def split_dataset(data):
        # split into standard weeks
        days = int(len(data)/24)
        split_index = int((days * 0.9))
        day_split_index = int(split_index * 24)
        train, test = data[:day_split_index], data[day_split_index:]
        # restructure into windows of weekly data
        train = array(split(train, len(train)/24))
        test = array(split(test, len(test)/24))
        return train, test

    # evaluate one or more weekly forecasts against expected values
    def evaluate_forecasts(actual, predicted):
        scores = list()
        # calculate an RMSE score for each day
        for i in range(actual.shape[1]):
            # calculate mse
            mse = mean_squared_error(actual[:, i], predicted[:, i])
            # calculate rmse
            rmse = sqrt(mse)
            # store
            scores.append(rmse)
        # calculate overall RMSE
        s = 0
        for row in range(actual.shape[0]):
            for col in range(actual.shape[1]):
                s += (actual[row, col] - predicted[row, col])**2
        score = sqrt(s / (actual.shape[0] * actual.shape[1]))
        return score, scores

    # summarize scores
    def summarize_scores(name, score, scores):
        s_scores = ', '.join(['%.1f' % s for s in scores])
        print('%s: [%.3f] %s' % (name, score, s_scores))

    # convert history into inputs and outputs
    def to_supervised(train, n_input, n_out=24):
        # flatten data
        data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
        X, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance
            if out_end < len(data):
                x_input = data[in_start:in_end, 0]
                x_input = x_input.reshape((len(x_input), 1))
                X.append(data[in_start:in_end, :])
                y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1
        return array(X), array(y)

    # train the model
    def build_model(train, n_input):
        # prepare data
        train_size = int(len(train) * 0.8)
        train_data = train[:train_size]
        val_data = train[train_size:]
        train_x, train_y = to_supervised(train_data, n_input)
        val_x, val_y = to_supervised(val_data, n_input)
        # define parameters
        verbose, epochs, batch_size = 1, 100, 24
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

        optimizer = Adam(learning_rate=0.001)
        # define model
        model = Sequential()
        model.add(LSTM(200, activation='tanh', input_shape=(n_timesteps, n_features)))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs))
        model.compile(loss='mse', optimizer=optimizer)
        # fit network
        history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size, verbose=verbose)

        # Save the model
        model.save(f'diploma/models/model_{table}_{parking_id}.h5')

        return model



    # make a forecast
    def forecast(model, history, n_input_day):
        # flatten data
        data = array(history)
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        # retrieve last observations for input data
        input_x = data[:n_input_day, :]
        input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
        # forecast the next week
        yhat = model.predict(input_x, verbose=0)
        # we only want the vector forecast
        yhat = yhat[0]
        return yhat

    # evaluate a single model
    def evaluate_model(train, test, n_input):
        # fit model
        model = build_model(train, n_input)
        # history is a list of weekly data
        history = [x for x in test]
        # walk-forward validation over each week
        predictions = list()
        n_input_day = n_input
        for i in range(len(test)):
        #for i in range(7):
            # predict the week
            yhat_sequence = forecast(model, history, n_input_day)
            # store the predictions
            predictions.append(yhat_sequence)
            history.pop(0)
            arr = np.array(predictions)
            arr = arr.reshape(24+i*24,)
            hours = [num for num in range(1, 25+i*24)]
            pyplot.plot(hours, arr, marker='o', label='lstm')
            pyplot.show()
        # evaluate predictions days for each week
        predictions = array(predictions)
        score, scores = evaluate_forecasts(test[:, :, 0], predictions)
        return score, scores


    # load the new file
    dataset = df

    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()
    # Fit the scaler to your data
    # Select the columns to be normalized
    columns_to_normalize = ['occupied_spot_number', 'temperature', 'precipitation']
    # Fit the scaler to your data
    scaler.fit(df[columns_to_normalize])
    # Transform the selected columns
    df[columns_to_normalize] = scaler.transform(df[columns_to_normalize])

    # split into train and test
    train, test = split_dataset(dataset.values)
    # evaluate model and get scores
    n_input = 24

    score, scores = evaluate_model(train, test, n_input)

    # summarize scores
    summarize_scores('lstm', score, scores)


# Define the DAG
dag = DAG(
    dag_id='model_train_all',
    start_date=datetime(2023, 3, 12),
    schedule_interval='10 22 * * 3',
    catchup=False,
    template_searchpath=["/home/melicharovykrecek/diploma/sql"]
)

model_train_1 = PythonOperator(
    task_id='model_train_1',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534017'
    },
    dag=dag,
)

model_train_2 = PythonOperator(
    task_id='model_train_2',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534016'
    },
    dag=dag,
)

model_train_3 = PythonOperator(
    task_id='model_train_3',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534015'
    },
    dag=dag,
)

model_train_4 = PythonOperator(
    task_id='model_train_4',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534014'
    },
    dag=dag,
)

model_train_5 = PythonOperator(
    task_id='model_train_5',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534012'
    },
    dag=dag,
)

model_train_6 = PythonOperator(
    task_id='model_train_6',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534011'
    },
    dag=dag,
)

model_train_7 = PythonOperator(
    task_id='model_train_7',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534009'
    },
    dag=dag,
)

model_train_8 = PythonOperator(
    task_id='model_train_8',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534008'
    },
    dag=dag,
)

model_train_9 = PythonOperator(
    task_id='model_train_9',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534005'
    },
    dag=dag,
)

model_train_10 = PythonOperator(
    task_id='model_train_10',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534004'
    },
    dag=dag,
)

model_train_11 = PythonOperator(
    task_id='model_train_11',
    python_callable=run_python_script,
    op_kwargs={
        'table': 'parking_measurements',
        'parking_id': 'tsk-534002'
    },
    dag=dag,
)

# Set task dependencies
model_train_1 >> model_train_2 >> model_train_3 >> model_train_4 >> model_train_5 >> model_train_6 >> model_train_7 >> model_train_8 >> model_train_9 >> model_train_10 >> model_train_11  