#london central 51.50722,-0.1275

#map styles
# mapbox://styles/mapbox/streets-v11
# mapbox://styles/mapbox/outdoors-v11
# mapbox://styles/mapbox/light-v10
# mapbox://styles/mapbox/dark-v10
# mapbox://styles/mapbox/satellite-v9
# mapbox://styles/mapbox/satellite-streets-v11

##########################################################
#          Reminder for integration
##########################################################
# change key: #updatekey
# -import the packge
# -update csv, dataframe output from the model
# -update all dataframes
# -update how we trigger the model prediction
#
# can try to improve performance by process df per map?
##########################################################

import LondonEmotions  #updatekey
import streamlit as st
import pydeck as pdk
import numpy as np
import pandas as pd

import os
mapbox_api_key = os.getenv('MAPBOX_API_KEY')

def main():
    ### Dataframe must be loaded before any maps
    # data_for_static = pd.read_csv('raw_data/pred_testset.csv') #updatekey
    # data_for_static.rename(columns={'lng':'lon'}, inplace=True)
    # data = pd.DataFrame([data_for_static['lat'], data_for_static['lon']])
    # data = data.T
    data = pd.read_csv('raw_data/pred_testset.csv') #updatekey
    data.rename(columns={'lng':'lon'}, inplace=True)

    # define emoji
    JOY_URL="https://res.cloudinary.com/dq7pjfkgz/image/upload/v1606418659/joy_gabpby.png"
    SAD_URL="https://res.cloudinary.com/dq7pjfkgz/image/upload/v1606418659/sad_icpf1w.png"
    WORRY_URL="https://res.cloudinary.com/dq7pjfkgz/image/upload/v1606418659/worry_rwobfs.png"
    ANGRY_URL="https://res.cloudinary.com/dq7pjfkgz/image/upload/v1606418659/angry_shqypp.png"
    NEUTRAL_URL="https://res.cloudinary.com/dq7pjfkgz/image/upload/v1606418659/neutral_evi6qa.png"
    joy_icon = {"url": JOY_URL, "width": 242, "height": 242, "anchorY": 242,}
    sad_icon = {"url": SAD_URL, "width": 242, "height": 242, "anchorY": 242,}
    worry_icon = {"url": WORRY_URL, "width": 242, "height": 242, "anchorY": 242,}
    angry_icon = {"url": ANGRY_URL, "width": 242, "height": 242, "anchorY": 242,}
    neutral_icon = {"url": NEUTRAL_URL, "width": 242, "height": 242, "anchorY": 242,}

    # split dataframe to emotions
    #joy_df = data[:30]
    joy_df = data[data['emotion']=='joy'] #updatekey
    joy_df["emoji"] = None
    for i in joy_df.index:
        joy_df["emoji"][i] = joy_icon
    joy_df = joy_df.T

    #sad_df = data[400:430]
    sad_df = data[data['emotion']=='sad']  == 'sad' #updatekey
    sad_df["emoji"] = None
    for i in sad_df.index:
        sad_df["emoji"][i] = sad_icon
    sad_df = sad_df.T

    #worry_df = data[800:830]
    worry_df = data[data['emotion']=='worry']  #updatekey
    worry_df["emoji"] = None
    for i in worry_df.index:
        worry_df["emoji"][i] = worry_icon
    worry_df = worry_df.T

    #angry_df = data[1200:1230] #
    angry_df = data[data['emotion']=='angry']  #updatekey
    angry_df["emoji"] = None
    for i in angry_df.index:
        angry_df["emoji"][i] = angry_icon
    angry_df = angry_df.T

    #neutral_df = data[1600:1630]
    neutral_df = data[data['emotion']=='neutral']  #updatekey
    neutral_df["emoji"] = None
    for i in neutral_df.index:
        neutral_df["emoji"][i] = neutral_icon
    neutral_df = neutral_df.T

    # analysis = st.sidebar.selectbox("Choose your map", ["Data spread", "All-in-one", "Joy", "Sad", "Worry", "Neutral" "Unknown?!"])
    # if analysis == "Data spread":
    if st.checkbox('Data spread'):
        st.header("Dots on the map")
        st.markdown("this is a placeholder text")
        st.map(data=data_for_static)

    if st.checkbox('All-in-one'):
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
                    type="IconLayer",
                    data=joy_df,
                    get_icon="emoji",
                    get_size=3,
                    size_scale=15,
                    get_position=["lon", "lat"],
                 ),
                pdk.Layer(
                    type="IconLayer",
                    data=sad_df,
                    get_icon="emoji",
                    get_size=3,
                    size_scale=15,
                    get_position=["lon", "lat"],
                 ),
                pdk.Layer(
                    type="IconLayer",
                    data=worry_df,
                    get_icon="emoji",
                    get_size=3,
                    size_scale=15,
                    get_position=["lon", "lat"],
                 ),
                pdk.Layer(
                    type="IconLayer",
                    data=angry_df,
                    get_icon="emoji",
                    get_size=3,
                    size_scale=15,
                    get_position=["lon", "lat"],
                 ),
                pdk.Layer(
                    type="IconLayer",
                    data=neutral_df,
                    get_icon="emoji",
                    get_size=3,
                    size_scale=15,
                    get_position=["lon", "lat"],
                 ),
            ],
        ))

    # if analysis == "joy":
    if st.checkbox('joy'):
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
                   'HexagonLayer',
                   data=data,     #updatekey
                   get_position='[lon, lat]',
                   radius=200,
                   elevation_scale=8,
                   elevation_range=[0, 2000],
                   get_fill_color='[0, 128, 255, 160]',
                   #pickable=True,
                   extruded=True,
                ),
                pdk.Layer(
                    'ScatterplotLayer',
                    data=data,     #updatekey
                    get_position='[lon, lat]',
                    get_color='[200, 30, 0, 160]',
                    # get_color='[0, 128, 255, 160]',
                    get_radius=200,
                ),
            ],
        ))

    # if analysis == "Sad":
    if st.checkbox('Sad'):
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
                    data=data,     #updatekey
                    get_position='[lon, lat]',
                    get_color='[0, 128, 255, 160]',
                    get_radius=100,
                ),
            ],
        ))

    if st.checkbox('Neutral'):
        # JOY_URL="https://upload.wikimedia.org/wikipedia/commons/c/c4/Projet_bi%C3%A8re_logo_v2.png"
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/satellite-streets-v11',
            initial_view_state=pdk.ViewState(
                latitude=51.50722,
                longitude=-0.1275,
                zoom=9,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    type="IconLayer",
                    data=neutral_df,
                    get_icon="emoji",
                    get_size=3,
                    size_scale=15,
                    get_position=["lon", "lat"],
                 ),
            ],
        ))

    if st.checkbox('Try it yourself!'):
        ### input text
        default = "Type something"
        user_input = st.text_area("Try it yourself", default)
        to_predict = pd.DataFrame([user_input])
        #response = pipeline.predict(to_predict) # depends how we trigger the prediction #updatekey
        #st.write("I see you are feeling ", response[0])


if __name__ == "__main__":
    main()
