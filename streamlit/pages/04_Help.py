import streamlit as st

st.set_page_config(page_title='City Traffic Forecasting', page_icon='📖') #, layout='wide'


st.header('Help page')

st.subheader('Overview')
numbers = ['0% and 20%', '20% and 40%', '40% and 60%', '60% and 80%', '80% and 100%']
text = ['Very light', 'Light', 'Normal', 'Heavy', 'Very heavy']

for i in range(len(text)):
    st.markdown(f"* If the percentile is between {numbers[i]}, It should be {text[i]} traffic")

st.subheader('Parking Predictions')

st.subheader('Parking Predictions Detail')

st.subheader('Public Transport Predictions')