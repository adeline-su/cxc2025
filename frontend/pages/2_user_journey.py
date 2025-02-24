import streamlit as st
import plotly.io as pio

st.set_page_config(layout="wide")

st.write("# User Journey Mapping Challenge")

# User Journey Flow
with open("../json/User_Journey_Flow.json", "r") as f:
    fig4 = pio.from_json(f.read())
st.plotly_chart(fig4, key="fig4")

