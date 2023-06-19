import streamlit as st
import pandas as pd
from datetime import date
import pydeck as pdk
import numpy as np

st.set_page_config(page_title='City Traffic Forecasting', page_icon='🏙️', layout='wide')


# Title
st.title('City traffic time series forecasting')


st.caption('Percentile predicting the level of traffic density in the city for the next 7 days')

current_date = '2023-05-22'
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

################################################################## AVG
data = load_data()

df = data

st.divider()

# Extract the date component from the timestamp
data['date'] = data.index.date

# Group the data by date and find the maximum value for each date
max_values = data.groupby('date')['value'].max()

print(max_values)

# Create 7 columns for each day
columns = st.columns(7)

# Display the maximum values for each day in the columns
for i, (col, day) in enumerate(zip(columns, max_values.index)):
    #col.metric(label=str(day), value=max_values[i])

    if 100*max_values[i] <= 20:
        help = 'Very light traffic'
    if 100*max_values[i] > 20 and 100*max_values[i] <= 40:
        help = 'Light traffic'
    elif 100*max_values[i] > 40 and 100*max_values[i] <= 60:
        help = 'Regular traffic'
    elif 100*max_values[i] > 60 and 100*max_values[i] <= 80:
        help = 'Heavy traffic'
    elif 100*max_values[i] > 80:
        help = 'Very heavy traffic'

    col.metric(label=str(day), value=f"{100*max_values[i]:.0f} %", help=help)

st.divider()

# Rename the 'value' column to a different name
data = data.rename(columns={"value": "Prediction"})
# Display the bar chart
st.bar_chart(data["Prediction"], height=300)

# Help part
st.divider()

st.subheader('Help')
st.caption('Different levels of traffic intensity, ranging from very light to very heavy, based on percentage ranges.')

numbers = ['0% - 20%', '20% - 40%', '40% - 60%', '60% - 80%', '80% - 100%']
text = ['Very light', 'Light', 'Normal', 'Heavy', 'Very heavy']
columns = st.columns(5)
# Display the metrics for each category
for i, col in enumerate(columns):
    col.metric(label=f'{text[i]} traffic', value=numbers[i])