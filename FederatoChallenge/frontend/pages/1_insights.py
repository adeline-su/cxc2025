import streamlit as st
import plotly.io as pio

st.set_page_config(layout="wide")

st.write("## Data Insights")

# st.write("### Preliminary Insights")
# Image with full width and caption
st.write("#### Basic Visualizations")
st.image('../images/julia_eda_basic_visualizations.png', use_container_width=True)

st.write("#### Devices and OS")
st.image('../images/julia_eda_devices_and_os.png', use_container_width=True)

st.write("### Events")
with open("../json/Event_Types_Sankey_Diagram.json", "r") as f:
    fig3 = pio.from_json(f.read())
st.plotly_chart(fig3, key="fig3")

st.write("#### Distribution of Number of Events")
st.image('../images/Distribution_of_Number_of_Events.png', use_container_width=True)

st.write("#### Scatter Plot of Number of Events vs Elapsed Time")
st.image('../images/Scatter_Plot_of_Number_of_Events_vs_Elapsed_Time.png', use_container_width=True)

st.write("### Session Length")

st.write("#### Session Length Overview")
st.image('../images/julia_eda_session_length.png', use_container_width=True)

st.write("#### Session Length Distribution")
st.image('../images/julia_eda_session_length_distribution.png', use_container_width=True)

st.write("#### Retention Analysis")
st.image('../images/julia_eda_retention.png', use_container_width=True)

st.write("#### Feature Correlation with Retention")
st.image('../images/Feature_Correlation_with_Retention.png', use_container_width=True)

st.write("#### Top Events in Long Sessions")
st.image('../images/Top_Events_in_Long_Sessions.png', use_container_width=True)

with open("../json/Filtered_Event_Flowchart_with_Likeliest_Transition_Percentages.json", "r") as f:
    fig4 = pio.from_json(f.read())
st.plotly_chart(fig4, key="fig4")
