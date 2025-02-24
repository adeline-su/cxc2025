# import streamlit as st

# st.write("# hello world")
# x = st.text_area("fav movie")
# st.write(f"ur fav movie is {x}")

# # Initialize y in session state if it doesn't exist
# if "y" not in st.session_state:
#     st.session_state.y = 0

# if st.button("Press me"):
#     st.session_state.y += 1

# st.write(f"Button pressed? {st.session_state.y}")


import streamlit as st
import plotly.io as pio

# Event Types Hierarchy
with open("notebooks/Event_Types_Sankey_Diagram.json", "r") as f:
    fig = pio.from_json(f.read())
st.plotly_chart(fig)




