import streamlit as st
import pandas as pd
from datetime import date

# Title
st.title('Public Transport Forecast Detail')
st.caption('Forecast of the Delay in [seconds] for specific Tram number')

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

################################################################## 1
st.header('Tram no. 1')

table = 'vehiclepositions_model'
gtfs_route_id = 'L1'
data = load_data(table, gtfs_route_id)

df = data

if st.checkbox('Show raw data', key='checkbox1'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Forecast of tram delays in seconds for 7 days in advance')

# Rename the 'value' column to a different name
data = data.rename(columns={"value": "1"})
# Display the bar chart
st.bar_chart(data["1"])

################################################################## 2
st.header('Tram no. 2')

table = 'vehiclepositions_model'
gtfs_route_id = 'L2'
data2 = load_data(table, gtfs_route_id)

if st.checkbox('Show raw data', key='checkbox2'):
    st.subheader('Raw data')
    st.write(data2)

st.subheader('Forecast of tram delays in seconds for 7 days in advance')

# Rename the 'value' column to a different name
data2 = data2.rename(columns={"value": "2"})
# Display the bar chart
st.bar_chart(data2["2"])


################################################################## 3
st.header('Tram no. 3')

table = 'vehiclepositions_model'
gtfs_route_id = 'L3'
data3 = load_data(table, gtfs_route_id)

if st.checkbox('Show raw data', key='checkbox3'):
    st.subheader('Raw data')
    st.write(data3)

st.subheader('Forecast of tram delays in seconds for 7 days in advance')

# Rename the 'value' column to a different name
data3 = data3.rename(columns={"value": "3"})
# Display the bar chart
st.bar_chart(data3["3"])

################################################################## 5
st.header('Tram no. 5')

table = 'vehiclepositions_model'
gtfs_route_id = 'L5'
data5 = load_data(table, gtfs_route_id)

if st.checkbox('Show raw data', key='checkbox4'):
    st.subheader('Raw data')
    st.write(data5)

st.subheader('Forecast of tram delays in seconds for 7 days in advance')

# Rename the 'value' column to a different name
data5 = data5.rename(columns={"value": "5"})
# Display the bar chart
st.bar_chart(data5["5"])

################################################################## 6
st.header('Tram no. 6')

table = 'vehiclepositions_model'
gtfs_route_id = 'L6'
data6 = load_data(table, gtfs_route_id)

if st.checkbox('Show raw data', key='checkbox6'):
    st.subheader('Raw data')
    st.write(data6)

st.subheader('Forecast of tram delays in seconds for 7 days in advance')

# Rename the 'value' column to a different name
data6 = data6.rename(columns={"value": "6"})
# Display the bar chart
st.bar_chart(data6["6"])

################################################################## 7
st.header('Tram no. 7')

table = 'vehiclepositions_model'
gtfs_route_id = 'L7'
data8 = load_data(table, gtfs_route_id)

if st.checkbox('Show raw data', key='checkbox8'):
    st.subheader('Raw data')
    st.write(data8)

st.subheader('Forecast of tram delays in seconds for 7 days in advance')

# Rename the 'value' column to a different name
data8 = data8.rename(columns={"value": "7"})
# Display the bar chart
st.bar_chart(data8["7"])

################################################################## 8
st.header('Tram no. 8')

table = 'vehiclepositions_model'
gtfs_route_id = 'L8'
data9 = load_data(table, gtfs_route_id)

if st.checkbox('Show raw data', key='checkbox9'):
    st.subheader('Raw data')
    st.write(data9)

st.subheader('Forecast of tram delays in seconds for 7 days in advance')

# Rename the 'value' column to a different name
data9 = data9.rename(columns={"value": "8"})
# Display the bar chart
st.bar_chart(data9["8"])

################################################################## 9
st.header('Tram no. 9')

table = 'vehiclepositions_model'
gtfs_route_id = 'L9'
data11 = load_data(table, gtfs_route_id)

if st.checkbox('Show raw data', key='checkbox10'):
    st.subheader('Raw data')
    st.write(data11)

st.subheader('Forecast of tram delays in seconds for 7 days in advance')

# Rename the 'value' column to a different name
data11 = data11.rename(columns={"value": "9"})
# Display the bar chart
st.bar_chart(data11["9"])

################################################################## 10
st.header('Tram no. 10')

table = 'vehiclepositions_model'
gtfs_route_id = 'L10'
data11 = load_data(table, gtfs_route_id)

if st.checkbox('Show raw data', key='checkbox7'):
    st.subheader('Raw data')
    st.write(data11)

st.subheader('Forecast of tram delays in seconds for 7 days in advance')

# Rename the 'value' column to a different name
data11 = data11.rename(columns={"value": "10"})
# Display the bar chart
st.bar_chart(data11["10"])

################################################################## 11
st.header('Tram no. 11')

table = 'vehiclepositions_model'
gtfs_route_id = 'L11'
data11 = load_data(table, gtfs_route_id)

if st.checkbox('Show raw data', key='checkbox11'):
    st.subheader('Raw data')
    st.write(data11)

st.subheader('Forecast of tram delays in seconds for 7 days in advance')

# Rename the 'value' column to a different name
data11 = data11.rename(columns={"value": "11"})
# Display the bar chart
st.bar_chart(data11["11"])

################################################################## 12
st.header('Tram no. 12')

table = 'vehiclepositions_model'
gtfs_route_id = 'L12'
data11 = load_data(table, gtfs_route_id)

if st.checkbox('Show raw data', key='checkbox12'):
    st.subheader('Raw data')
    st.write(data11)

st.subheader('Forecast of tram delays in seconds for 7 days in advance')

# Rename the 'value' column to a different name
data11 = data11.rename(columns={"value": "12"})
# Display the bar chart
st.bar_chart(data11["12"])