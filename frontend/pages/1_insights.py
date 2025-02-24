import streamlit as st
import plotly.io as pio

st.set_page_config(layout="wide")

st.write("## Insights")

# Distribution_of_Number_of_Events
st.image('notebooks/Distribution_of_Number_of_Events.png')

# Scatter_Plot_of_Number_of_Events_vs_Elapsed_Time
st.image('notebooks/Scatter_Plot_of_Number_of_Events_vs_Elapsed_Time.png')

st.image('notebooks/julia_eda.png')


# Event Types Hierarchy
with open("notebooks/Event_Types_Sankey_Diagram.json", "r") as f:
    fig3 = pio.from_json(f.read())
st.plotly_chart(fig3, key="fig3")


st.image('notebooks/Weekly_Retention_Cohorts.png')
st.image('notebooks/Feature_Correlation_with_Retention.png')
st.image('notebooks/Session_Duration_Distribution.png')
st.image('notebooks/Top_Events_in_Long_Sessions.png')

with open("notebooks/Filtered_Event_Flowchart_with_Likeliest_Transition_Percentages.json", "r") as f:
    fig4 = pio.from_json(f.read())
st.plotly_chart(fig4, key="fig4")



