import streamlit as st
from components.node_palette import display_node_palette
from components.canvas import display_canvas
from components.node_properties_panel import display_node_properties_panel
from components.workflow_manager import manage_workflow

# UI Layout
st.set_page_config(layout="wide")
st.sidebar.title("Node Palette")
display_node_palette()

# Main canvas area
col1, col2 = st.columns([3, 1])
with col1:
    display_canvas()
with col2:
    display_node_properties_panel()

manage_workflow()
