#london central 51.50722,-0.1275

#map styles
# mapbox://styles/mapbox/streets-v11
# mapbox://styles/mapbox/outdoors-v11
# mapbox://styles/mapbox/light-v10
# mapbox://styles/mapbox/dark-v10
# mapbox://styles/mapbox/satellite-v9
# mapbox://styles/mapbox/satellite-streets-v11

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

#### 3d map dataframe
map_df = pd.read_csv('raw_data/prediction.csv')
map_df.rename(columns={'lng':'lon'}, inplace=True)
data = pd.DataFrame([map_df['lat'], map_df['lon']])
data = data.T

### ScatterplotLayer
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/streets-v11',
    initial_view_state=pdk.ViewState(
        latitude=51.50722,
        longitude=-0.1275,
        zoom=9,
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
           get_fill_color='[0, 128, 255, 160]',
           #pickable=True,
           extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=data,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            # get_color='[0, 128, 255, 160]',
            get_radius=200,
        ),
    ],
))

### ColumnLayer
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/dark-v10',
    initial_view_state=pdk.ViewState(
        latitude=51.50722,
        longitude=-0.1275,
        zoom=9,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            'HeatmapLayer',
            data=data,
            get_position='[lon, lat]',
            get_color='[0, 128, 255, 160]',
            get_radius=200,
        ),
    ],
))


### IconLayer
DATA_URL = "https://raw.githubusercontent.com/ajduberstein/geo_datasets/master/biergartens.json"
HAPPY_URL="https://image.emojipng.com/807/12075807.jpg"
SAD_URL="https://upload.wikimedia.org/wikipedia/commons/c/c4/Projet_bi%C3%A8re_logo_v2.png"

happy_icon = {
    "url": HAPPY_URL,
    "width": 242,
    "height": 242,
    "anchorY": 242,
}

sad_icon = {
    "url": SAD_URL,
    "width": 242,
    "height": 242,
    "anchorY": 242,
}

data = pd.read_json(DATA_URL)
data["icon_data"] = None
for i in data.index:
    data["icon_data"][i] = icon_data

view_state = pdk.data_utils.compute_view(data[["lon", "lat"]], 0.1)

icon_layer = pdk.Layer(
    type="IconLayer",
    data=data,
    get_icon="icon_data",
    get_size=4,
    size_scale=15,
    get_position=["lon", "lat"],
    mapbox_api_key=MAPBOX_API_KEY
    #pickable=True,
)

r = pdk.Deck(layers=[icon_layer], initial_view_state=view_state, tooltip={"text": "{tags}"})
r.to_html("streamlit.html")


