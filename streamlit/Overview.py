import streamlit as st
import pandas as pd
from datetime import date
import pydeck as pdk
import numpy as np

st.set_page_config(page_title='City Traffic Forecasting', page_icon='🏙️', layout='wide')


# Title
st.title('City traffic time series forecasting')


st.caption('Forecast of the number of available parking spaces in P+R parking houses')

current_date = date.today()
DATE_COLUMN = 'timestamp'

@st.cache_data
def load_data():
    data = pd.read_csv(f'predictions/output/predictions/average_data.csv')
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    print('\n\n\n',data)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    data.set_index('timestamp', inplace=True)
    return data

################################################################## 17
st.header('Prediction')

data = load_data()

df = data

if st.checkbox('Show raw data', key='checkbox1'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Forecast of occupied parking spaces for 7 days in advance')

# Rename the 'value' column to a different name
data = data.rename(columns={"value": "Prediction"})
# Display the bar chart
st.bar_chart(data["Prediction"])

# Extract the date component from the timestamp
data['date'] = data.index.date

print(data)

# Group the data by date and find the maximum value for each date
max_values = data.groupby('date')['Prediction'].max()
# Print the maximum values for each date
print(max_values)

# Create 7 columns for each day
columns = st.columns(7)

# Display the maximum values for each day in the columns
for i, (col, day) in enumerate(zip(columns, max_values.index)):
    #col.metric(label=str(day), value=max_values[i])
    col.metric(label=str(day), value=f"{100*max_values[i]:.0f} %")
