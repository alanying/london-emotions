#london central 51.50722,-0.1275

import streamlit as st
import pydeck as pdk
import numpy as np
import pandas as pd

import os
mapbox_api_key = os.getenv('MAPBOX_API_KEY')

st.title('Emotions on the map')
st.write('How are you feeling today?')

map_df = pd.read_csv('raw_data/prediction.csv')
map_df.rename(columns={'lng':'lon'}, inplace=True)
data = pd.DataFrame([map_df['lat'], map_df['lon']])
st.map(data=map_df)

option = st.sidebar.selectbox(
    'Which emotion would you search for?',
     ['Choose an emotion', '2']) # replace iwth df['Emotion']
'You selected:', option

#### 3d map
map_df = pd.read_csv('raw_data/prediction.csv')
map_df.rename(columns={'lng':'lon'}, inplace=True)
data = pd.DataFrame([map_df['lat'], map_df['lon']])
data = data.T

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=51.50722,
        longitude=-0.1275,
        zoom=11,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
           'HexagonLayer',
           data=data,
           get_position='[lon, lat]',
           radius=200,
           elevation_scale=4,
           elevation_range=[0, 1000],
           #pickable=True,
           extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=data,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=200,
        ),
    ],
))
