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

# Title
st.title('P+R Forecast Detail')
st.caption('Forecast of the number of available parking spaces in P+R parking houses')

current_date = '2023-05-22'
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

################################################################## 17
st.header('P+R Chodov')

table = 'parking_measurements'
parking_id = 'tsk-534017'
data = load_data(table, parking_id)

df = data

if st.checkbox('Show raw data', key='checkbox1'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Forecast of occupied parking spaces for 7 days in advance')

# Rename the 'value' column to a different name
data = data.rename(columns={"value": "Chodov"})
# Display the bar chart
st.bar_chart(data["Chodov"])

################################################################## 16
st.header('P+R Letnany')

table = 'parking_measurements'
parking_id = 'tsk-534016'
data2 = load_data(table, parking_id)

if st.checkbox('Show raw data', key='checkbox2'):
    st.subheader('Raw data')
    st.write(data2)

st.subheader('Forecast of occupied parking spaces for 7 days in advance')

# Rename the 'value' column to a different name
data2 = data2.rename(columns={"value": "Letnany"})
# Display the bar chart
st.bar_chart(data2["Letnany"])

################################################################## 15
st.header('P+R Depo Hostivar')

table = 'parking_measurements'
parking_id = 'tsk-534015'
data3 = load_data(table, parking_id)

if st.checkbox('Show raw data', key='checkbox3'):
    st.subheader('Raw data')
    st.write(data3)

st.subheader('Forecast of occupied parking spaces for 7 days in advance')

# Rename the 'value' column to a different name
data3 = data3.rename(columns={"value": "Depo Hostivar"})
# Display the bar chart
st.bar_chart(data3["Depo Hostivar"])

################################################################## 12
st.header('P+R Skalka')

table = 'parking_measurements'
parking_id = 'tsk-534012'
data5 = load_data(table, parking_id)

if st.checkbox('Show raw data', key='checkbo54'):
    st.subheader('Raw data')
    st.write(data5)

st.subheader('Forecast of occupied parking spaces for 7 days in advance')

# Rename the 'value' column to a different name
data5 = data5.rename(columns={"value": "Skalka"})
# Display the bar chart
st.bar_chart(data5["Skalka"])

################################################################## 11
st.header('P+R Cerny Most')

table = 'parking_measurements'
parking_id = 'tsk-534011'
data6 = load_data(table, parking_id)

if st.checkbox('Show raw data', key='checkbox6'):
    st.subheader('Raw data')
    st.write(data6)

st.subheader('Forecast of occupied parking spaces for 7 days in advance')

# Rename the 'value' column to a different name
data6 = data6.rename(columns={"value": "Cerny Most"})
# Display the bar chart
st.bar_chart(data6["Cerny Most"])

################################################################## 08
st.header('P+R Zlicin 1')

table = 'parking_measurements'
parking_id = 'tsk-534008'
data8 = load_data(table, parking_id)

if st.checkbox('Show raw data', key='checkbox8'):
    st.subheader('Raw data')
    st.write(data8)

st.subheader('Forecast of occupied parking spaces for 7 days in advance')

# Rename the 'value' column to a different name
data8 = data8.rename(columns={"value": "Zlicin 1"})
# Display the bar chart
st.bar_chart(data8["Zlicin 1"])

################################################################## 05
st.header('P+R Opatov')

table = 'parking_measurements'
parking_id = 'tsk-534005'
data9 = load_data(table, parking_id)

if st.checkbox('Show raw data', key='checkbox9'):
    st.subheader('Raw data')
    st.write(data9)

st.subheader('Forecast of occupied parking spaces for 7 days in advance')

# Rename the 'value' column to a different name
data9 = data9.rename(columns={"value": "Opatov"})
# Display the bar chart
st.bar_chart(data9["Opatov"])

################################################################## 02
st.header('P+R Holesovice')

table = 'parking_measurements'
parking_id = 'tsk-534002'
data11 = load_data(table, parking_id)

if st.checkbox('Show raw data', key='checkbox11'):
    st.subheader('Raw data')
    st.write(data11)

st.subheader('Forecast of occupied parking spaces for 7 days in advance')

# Rename the 'value' column to a different name
data11 = data11.rename(columns={"value": "Holesovice"})
# Display the bar chart
st.bar_chart(data11["Holesovice"])