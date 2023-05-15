'''import streamlit as st
import pandas as pd

# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

import mysql.connector


# db parameters
user = 'martin'
pw = 'admin'
db = 'mysql'
host = '34.77.219.108'

# establishing the connection
conn = mysql.connector.connect(user=user, password=pw, host=host, database=db)

# Reading from db to dataframe
#query = "SELECT date_modified, occupied_spot_number FROM out_parking_measurements WHERE parking_id = 'tsk-534017' AND date_modified > '2021-05-03' AND date_modified < '2023-04-24';"
query = "SELECT date_modified, occupied_spot_number FROM out_parking_measurements WHERE parking_id = 'tsk-534017' AND date_modified > '2023-04-23' AND date_modified < '2023-04-24';"

df = pd.read_sql(query, conn)



st.write("Here's our first attempt at using data to create a table:")
st.write(df)'''


import streamlit as st
import pandas as pd
from datetime import date

st.title('P+R Chodov')

current_date = date.today()
DATE_COLUMN = 'timestamp'

@st.cache_data
def load_data(entity, parking):
    data = pd.read_csv(f'predictions/output/predictions_{entity}_{parking}_{current_date}.csv')
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    print('\n\n\n',data)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    data.set_index('timestamp', inplace=True)
    return data

data_load_state = st.text('Loading data...')

################################################################## 17
table = 'parking_measurements'
parking_id = 'tsk-534017'
data = load_data(table, parking_id)

if st.checkbox('Show raw data', key='checkbox1'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of used parking spaces by hour')

# Rename the 'value' column to a different name
data = data.rename(columns={"value": "chodov"})
# Display the bar chart
st.bar_chart(data["chodov"])

################################################################## 16
table = 'parking_measurements'
parking_id = 'tsk-534016'
data = load_data(table, parking_id)

if st.checkbox('Show raw data', key='checkbox2'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of used parking spaces by hour')

# Rename the 'value' column to a different name
data = data.rename(columns={"value": "letnany"})
# Display the bar chart
st.bar_chart(data["letnany"])