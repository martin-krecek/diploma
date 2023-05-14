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
import numpy as np
from io import StringIO

st.title('Uber pickups in NYC')

DATE_COLUMN = 'timestamp'
DATA_URL = ('output.csv')

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    print('\n\n\n',data)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("Done! (using st.cache_data)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

'''st.subheader('Number of pickups by hour')
hist_values = np.histogram(data, bins=168)
st.bar_chart(hist_values)'''

#df = pd.read_csv(StringIO(data))

st.write("Here's our first attempt at using data to create a table:")
st.write(data)

st.subheader('Number of pickups by hour')
print('\ndata["value"] \n',data['value'])
#print('\nhist_values: \n',d)
st.bar_chart(data['value'])