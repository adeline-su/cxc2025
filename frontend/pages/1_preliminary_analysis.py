import streamlit as st
import plotly.io as pio

st.set_page_config(layout="wide")

st.write("## Preliminary Exploration")

# Distribution_of_Number_of_Events
st.image('notebooks/Distribution_of_Number_of_Events.png')

# Scatter_Plot_of_Number_of_Events_vs_Elapsed_Time
st.image('notebooks/Scatter_Plot_of_Number_of_Events_vs_Elapsed_Time.png')

# Event Types Hierarchy
with open("notebooks/Event_Types_Sankey_Diagram.json", "r") as f:
    fig3 = pio.from_json(f.read())
st.plotly_chart(fig3, key="fig3")
