#HERE_API_KEY = 'hYjanuWQ92oW2KK_Um_1mmNuR7jW14th3hst9jNO_sc'

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

st.title('Emotions on the map')
st.write('How are you feeling today?')
st.write("Perhaps this table will show the rank of emotions?:")
st.write(pd.DataFrame({
    'first column': ['emo1', 'emo1', 'emo1', 'emo1', 'emo1'],
    'second column': [10, 20, 30, 40, 50]
}))

#line chat
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])
st.line_chart(chart_data)

# plot the map
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])
st.map(map_data)
