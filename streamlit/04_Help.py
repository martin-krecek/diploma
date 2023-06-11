import streamlit as st

st.set_page_config(page_title='City Traffic Forecasting', page_icon='📖', layout='wide') #, layout='wide'


import pandas as pd
import numpy as np

st.header('Help page')

st.subheader('Overview')
numbers = ['0%-20%', '20%-40%', '40%-60%', '60%-80%', '80%-100%']
text = ['Very light', 'Light', 'Normal', 'Heavy', 'Very heavy']
columns = st.columns(5)
# Display the metrics for each category
for i, col in enumerate(columns):
    col.metric(label=f'{text[i]} traffic', value=numbers[i])

for i in range(len(text)):
    st.markdown(f"* If the percentile is between **{numbers[i]}**, It should be **{text[i]}** traffic")

st.subheader('Parking Predictions')

st.subheader('Parking Predictions Detail')

st.subheader('Public Transport Predictions')