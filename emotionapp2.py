
import LondonEmotions  #updatekey
import streamlit as st
import pydeck as pdk
import numpy as np
import pandas as pd
# from LondonEmotions.params import BUCKET_NAME, PRETEST_PATH
from LondonEmotions.data import clean_data
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from tensorflow.python.lib.io import file_io


import os
mapbox_api_key = os.getenv('MAPBOX_API_KEY')

def main():

    ###############################################################################
    #   Dataframe and emoji setup
    # https://commons.wikimedia.org/wiki/Emoji
    # https://github.com/caiyongji/emoji-list
    ###############################################################################
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://wallpapercave.com/wp/wp2610923.jpg");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    ### Dataframe must be loaded before any maps
    data = pd.read_csv("streamlit_prep/predicted_reviews.csv")
    # data = data.drop(columns=['manually_added_emotion', 'joy, sad, worry, anger, neutral'])
    data.rename(columns={'lng':'lon'}, inplace=True)
    data.fillna('nan', inplace=True)

    # grab top 5 place_id's and convert to dataframe
    joy_df = data[data['emotion']=='joy'] #
    joyest = joy_df['place_id'].value_counts().index.tolist()[:3]
    top_joy_1 = joy_df.loc[(joy_df['place_id']==joyest[0])][:1]
    top_joy_2 = joy_df.loc[(joy_df['place_id']==joyest[1])][:1]
    top_joy_3 = joy_df.loc[(joy_df['place_id']==joyest[2])][:1]
    joy_df = pd.concat([top_joy_1, top_joy_2, top_joy_3])

    sad_df = data[data['emotion']=='sadness']
    saddest = sad_df['place_id'].value_counts().index.tolist()[:3]
    top_sad_1 = sad_df.loc[(sad_df['place_id']==saddest[0])][:1]
    top_sad_2 = sad_df.loc[(sad_df['place_id']==saddest[1])][:1]
    top_sad_3 = sad_df.loc[(sad_df['place_id']==saddest[2])][:1]
    sad_df = pd.concat([top_sad_1, top_sad_2, top_sad_3])

    worry_df = data[data['emotion']=='worry']
    full_worry_df = worry_df   # for a combine heatmap
    worriest = worry_df['place_id'].value_counts().index.tolist()[:3]
    top_worry_1 = worry_df.loc[(worry_df['place_id']==worriest[0])][:1]
    top_worry_2 = worry_df.loc[(worry_df['place_id']==worriest[1])][:1]
    top_worry_3 = worry_df.loc[(worry_df['place_id']==worriest[2])][:1]
    worry_df = pd.concat([top_worry_1, top_worry_2, top_worry_3])

    anger_df = data[data['emotion']=='anger']
    full_anger_df = anger_df   # for a combine heatmap
    angriest = anger_df['place_id'].value_counts().index.tolist()[:3]
    top_anger_1 = anger_df.loc[(anger_df['place_id']==angriest[0])][:1]
    top_anger_2 = anger_df.loc[(anger_df['place_id']==angriest[1])][:1]
    top_anger_3 = anger_df.loc[(anger_df['place_id']==angriest[2])][:1]
    anger_df = pd.concat([top_anger_1, top_anger_2, top_anger_3])

    full_anger_worry_df = pd.concat([full_worry_df, full_anger_df])   # for a combine heatmap

    neutral_df = data[data['emotion']=='neutral']
    neutralist = neutral_df['place_id'].value_counts().index.tolist()[:3]
    top_neutral_1 = neutral_df.loc[(neutral_df['place_id']==neutralist[0])][:1]
    top_neutral_2 = neutral_df.loc[(neutral_df['place_id']==neutralist[1])][:1]
    top_neutral_3 = neutral_df.loc[(neutral_df['place_id']==neutralist[2])][:1]
    neutral_df = pd.concat([top_neutral_1, top_neutral_2, top_neutral_3])

    # define emoji
    JOY_URL="https://res.cloudinary.com/dq7pjfkgz/image/upload/v1606418659/joy_gabpby.png"
    SAD_URL="https://res.cloudinary.com/dq7pjfkgz/image/upload/v1606418659/sad_icpf1w.png"
    WORRY_URL="https://res.cloudinary.com/dq7pjfkgz/image/upload/v1606418659/worry_rwobfs.png"
    ANGER_URL="https://res.cloudinary.com/dq7pjfkgz/image/upload/v1606418659/angry_shqypp.png"
    NEUTRAL_URL="https://res.cloudinary.com/dq7pjfkgz/image/upload/v1606418659/neutral_evi6qa.png"
    joy_icon = {"url": JOY_URL, "width": 242, "height": 242, "anchorY": 242,}
    sad_icon = {"url": SAD_URL, "width": 242, "height": 242, "anchorY": 242,}
    worry_icon = {"url": WORRY_URL, "width": 242, "height": 242, "anchorY": 242,}
    anger_icon = {"url": ANGER_URL, "width": 242, "height": 242, "anchorY": 242,}
    neutral_icon = {"url": NEUTRAL_URL, "width": 242, "height": 242, "anchorY": 242,}


    ###############################################################################
    #   Page presentation starts
    ###############################################################################

    # Define the sidebar
    st.sidebar.title("London Emotions")
    menu = st.sidebar.selectbox("Menu", ["Top spots", "Watchout!", "Guess my mood"])
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write("Legends:")
    st.sidebar.markdown(":blush:   Happy")
    st.sidebar.markdown(":worried:   Sad")
    st.sidebar.markdown(":fearful:   Fear")
    st.sidebar.markdown(":rage:   Angry")
    st.sidebar.markdown(":sunglasses:   Neutral")



    if menu == "Top spots":
        # define emoji
        joy_df["emoji"] = None
        for i in joy_df.index:
            joy_df["emoji"][i] = joy_icon

        sad_df["emoji"] = None
        for i in sad_df.index:
            sad_df["emoji"][i] = sad_icon

        worry_df["emoji"] = None
        for i in worry_df.index:
            worry_df["emoji"][i] = worry_icon

        anger_df["emoji"] = None
        for i in anger_df.index:
            anger_df["emoji"][i] = anger_icon

        neutral_df["emoji"] = None
        for i in neutral_df.index:
            neutral_df["emoji"][i] = neutral_icon

        st.header("Data points on the map")
        st.markdown("Data collection spread across London.")
        # st.map(data=data)
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/dark-v10',
            initial_view_state=pdk.ViewState(
                latitude=51.50722,
                longitude=-0.1275,
                zoom=9,
                pitch=50,
            ),
            layers = [
                pdk.Layer(
                    'ScatterplotLayer',
                    data=data,     #updatekey
                    get_position='[lon, lat]',
                    # get_color='[200, 30, 0, 160]',
                    get_color='[0, 128, 255, 160]',
                    get_radius=250,
                ),
            ],
        ))

        st.write(" ")

        expander = st.beta_expander("The full emotion map suppose to look like......")
        expander.image("streamlit_prep/capture.PNG", use_column_width=True)

        st.write(" ")
        st.write("_____________________________________________________________")

        st.header("Top spots for each emotions")
        if st.button('Very emotional spots'):
            st.write("Top 3 spots for each emotions")
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
                        data=anger_df,
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

            ### Button for google map outer link #https://discuss.streamlit.io/t/how-to-link-a-button-to-a-webpage/1661/4
            joyest = joy_df['place_id'].value_counts().index.tolist()[0]
            address = f"https://www.google.com/maps/place/?q=place_id:{joyest}"
            link = f'[Let\'s find out the most joyful place in London]({address})'
            st.markdown(link, unsafe_allow_html=True)

            sadest = sad_df['place_id'].value_counts().index.tolist()[0]
            address = f"https://www.google.com/maps/place/?q=place_id:{sadest}"
            link = f'[Let\'s find out the most depressive place in London]({address})'
            st.markdown(link, unsafe_allow_html=True)


    if menu == "Watchout!":
        st.header("Angry & Fear at the same time!")
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
                    data=full_worry_df,
                    opacity=0.5,
                    get_position='[lon, lat]',
                    get_color='[0, 128, 255, 160]',
                    get_radius=100,
                ),
                pdk.Layer(
                   'HexagonLayer',
                   data=full_anger_df,
                   get_position='[lon, lat]',
                   radius=200,
                   elevation_scale=8,
                   elevation_range=[0, 2000],
                   get_fill_color='[0, 180, 180, 180]',
                   extruded=True,
                ),
            ],
        ))
        angriest = full_anger_df['place_id'].value_counts().index.tolist()[0]
        address = f"https://www.google.com/maps/place/?q=place_id:{angriest}"
        link = f'[Place we should avoid in London!]({address})'
        st.markdown(link, unsafe_allow_html=True)



    if menu == "Guess my mood":
        # take user input
        st.header("Try it out, talk to me.")
        default = " "
        text = st.text_area(" ", default)
        text_df = {
            'Text': [text]
        }

        # convert user input to DF and clean
        text_df = pd.DataFrame(text_df)
        clean_text = clean_data(text_df)
        text_value = clean_text['tokenized_text']

        # load tokenizer locally
        with file_io.FileIO('streamlit_prep/model_tokenizer', 'rb') as handle:
            tokenizer = pickle.load(handle)

        # numericalise user input text
        max_seq_len = 300
        index_of_words = tokenizer.word_index
        vocab_size = len(index_of_words) + 1
        sequence_text = tokenizer.texts_to_sequences(text_value)
        text_pad = pad_sequences(sequence_text, maxlen=max_seq_len)

        # prediction
        model = load_model('streamlit_prep/saved_model.pb')
        preds = model.predict(text_pad)

        # prep results for presentation
        angry = str(round(preds[0][0]*100,2))
        joy = str(round(preds[0][1]*100,2))
        worry = str(round(preds[0][2]*100,2))
        neutral = str(round(preds[0][3]*100,2))
        sad = str(round(preds[0][4]*100,2))

        # presentation
        expander = st.beta_expander("Your mood prediction.")
        expander.subheader(f":blush:  Joy: {joy}%")
        expander.subheader(f":worried:  Sad: {sad}%")
        expander.subheader(f":fearful:  Fear: {worry}%")
        expander.subheader(f":sunglasses:  Neutral: {neutral}%")
        expander.subheader(f":rage:  Angry: {angry}%")

if __name__ == "__main__":
    main()
