import streamlit as st
import pandas as pd
from datetime import date
import pydeck as pdk
import numpy as np

st.set_page_config(page_title='Macro - Cross Chain Monitoring', page_icon='🚗', layout='wide')


# Title
st.title('P+R Forecast')
st.caption('Forecast of the number of available parking spaces in P+R parking houses')

current_date = date.today()
DATE_COLUMN = 'timestamp'

@st.cache_data
def load_data(entity):
    data = pd.read_csv(f'predictions/output/predictions_{entity}_{current_date}.csv')
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    #print('\n\n\n',data)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    data.set_index('timestamp', inplace=True)
    return data

def load_data_pydeck(entity):
    data = pd.read_csv(f'predictions/output/predictions_{entity}_{current_date}.csv')
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    #print('\n\n\n',data)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    data.set_index('timestamp', inplace=False)
    return data

################################################################## PYDECK
table = 'parking_measurements'
data_pydeck = load_data_pydeck(table)
data = load_data(table)

chart_data = pd.DataFrame(
   np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
   columns=['lat', 'lon'])




min_timestamp = data.index.min()
max_timestamp = data.index.max()

print('\n\n')
print(max_timestamp)
print(min_timestamp)

from datetime import timedelta

# Calculate the minimum and maximum timestamps
min_timestamp = data.index.min()
max_timestamp = data.index.max()

# Calculate the time range in hours
time_range = max_timestamp - min_timestamp
total_hours = time_range.total_seconds() / 3600

# Set the slider step size in hours
slider_step = 1

# Convert the step size to timedelta
step_timedelta = timedelta(hours=slider_step)

# Set the slider values to hourly intervals
selected_timestamp = st.slider(
    'Select Timestamp',
    min_value=min_timestamp.to_pydatetime(),
    max_value=max_timestamp.to_pydatetime(),
    step=step_timedelta,
    value=min_timestamp.to_pydatetime() + timedelta(hours=12),
    format="YYYY-MM-DD HH:mm:ss"
).strftime("%Y-%m-%d %H:%M:%S")

filtered_data = data[data.index == selected_timestamp]

print(filtered_data)


st.pydeck_chart(pdk.Deck(
    map_style='road',
    initial_view_state=pdk.ViewState(
        latitude=50.072074,
        longitude=14.502015,
        zoom=10.5,
        pitch=50,
    ),
    layers=
        pdk.Layer(
           'ColumnLayer',
           data=filtered_data,
           get_position='[lon, lat]',
           get_elevation='value',
           get_color='[200, 30, 0, 200]',
           radius=350,
           elevation_scale=10,
           elevation_range=[0, 1000],
           pickable=True,
           extruded=True,
        ),        tooltip = {
        "html": "P+R {name}: <b>{value}</b> occupied spots", "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"}}
    ,
))