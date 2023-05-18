import streamlit as st
import pandas as pd
from datetime import date
import pydeck as pdk
import numpy as np

st.set_page_config(page_title='City Traffic Forecasting', page_icon='🏙️', layout='wide')


# Title
st.title('City traffic time series forecasting')