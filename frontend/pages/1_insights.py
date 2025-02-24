import streamlit as st
import plotly.io as pio

st.set_page_config(layout="wide")

st.write("## Insights")


st.write("### Preliminary Insights")
st.image('notebooks/julia_eda_basic_visualizations.png')
st.image('notebooks/julia_eda_devices_and_os.png')


st.write("### Events")
with open("notebooks/Event_Types_Sankey_Diagram.json", "r") as f:
    fig3 = pio.from_json(f.read())
st.plotly_chart(fig3, key="fig3")

st.image('notebooks/Distribution_of_Number_of_Events.png')
st.image('notebooks/Scatter_Plot_of_Number_of_Events_vs_Elapsed_Time.png')

st.write("### Session Length")

st.image('notebooks/julia_eda_session_length.png')
st.image('notebooks/julia_eda_session_length_distribution.png')
st.image('notebooks/julia_eda_retention.png')


st.image('notebooks/Feature_Correlation_with_Retention.png')
st.image('notebooks/Top_Events_in_Long_Sessions.png')

with open("notebooks/Filtered_Event_Flowchart_with_Likeliest_Transition_Percentages.json", "r") as f:
    fig4 = pio.from_json(f.read())
st.plotly_chart(fig4, key="fig4")



