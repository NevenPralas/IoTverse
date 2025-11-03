from database import get_latest_data
import streamlit as st
import time

st.title("IoT Dash")
placeholder = st.empty()

while True:
    df = get_latest_data(200)
    with placeholder.container():
        st.line_chart(df, x="time", y="temperature")
    time.sleep(5)
