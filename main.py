import streamlit as st
from streamlit_option_menu import option_menu
from components.node_palette import display_node_palette
from components.canvas import display_canvas
from components.node_properties_panel import display_node_properties_panel
from components.api_key_settings import display_api_key_settings
from components.workflow_manager import manage_workflow

# Set wide layout
st.set_page_config(layout="wide")

# Sidebar with tabs for Node Properties and API Key Settings
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Node Properties", "API Key Settings"],
        icons=["pencil", "key"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Node Properties":
    st.header("Configure Node")
    st.caption("Adjust the parameters of the selected node here.")
    display_node_properties_panel()
elif selected == "API Key Settings":
    st.header("API Key Management")
    st.caption("Enter and manage your Replicate API key.")
    display_api_key_settings()

# Clean Node Palette UI
st.sidebar.title("Node Palette")
st.sidebar.caption("Drag nodes to the canvas and build your AI pipeline.")
display_node_palette()

# Main canvas area for workflow building
col1, col2 = st.columns([4, 1])
with col1:
    st.subheader("AI Pipeline Workflow")
    display_canvas()

# Handle workflow management
manage_workflow()
