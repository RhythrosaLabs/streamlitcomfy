import streamlit as st
from streamlit_elements import elements, dashboard, mui

# Initialize Replicate API key
if 'replicate_key' not in st.session_state:
    st.session_state['replicate_key'] = ''

# Sidebar for API key input
st.sidebar.title("API Key")
st.sidebar.text_input("Replicate API Key", key="replicate_key", type="password")

# Main interactive node-based UI using Streamlit Elements
with elements("main"):
    # Correctly initialize the Grid dashboard layout with draggable and resizable properties
    with dashboard.Grid(cols=12, rowHeight=160) as grid:
        # Define Node for LLaMA Text Generation
        with grid.item("llama", 0, 0, 6, 2):
            with mui.Paper(elevation=3):
                st.write("### LLaMA 70B - Text Generation")
                llama_prompt = st.text_input("Enter your text prompt for LLaMA", key="llama_prompt")
                if st.button("Generate Text"):
                    st.write(f"Generated Text: {llama_prompt}")

        # Define Node for Flux Text-to-Image
        with grid.item("flux", 6, 0, 6, 2):
            with mui.Paper(elevation=3):
                st.write("### Flux - Text to Image Generation")
                flux_prompt = st.text_input("Enter your image prompt for Flux", key="flux_prompt")
                if st.button("Generate Image"):
                    st.write(f"Generated Image from prompt: {flux_prompt}")

        # Define Node for Real-ESRGAN Image Upscaling
        with grid.item("esrgan", 0, 2, 6, 2):
            with mui.Paper(elevation=3):
                st.write("### Real-ESRGAN - Image Upscaling")
                image_url = st.text_input("Enter image URL for upscaling", key="image_url")
                if st.button("Upscale Image"):
                    st.write(f"Upscaled Image from URL: {image_url}")

    grid.save()
