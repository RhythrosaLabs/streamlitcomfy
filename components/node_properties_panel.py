import streamlit as st

def display_node_properties_panel():
    st.write("Node Properties")
    st.text_input("Prompt", key="prompt", help="Enter the prompt to generate the image.")
    st.slider("Guidance Scale", 0.0, 20.0, 7.5, key="guidance_scale", help="Adjusts the influence of the prompt on image generation.")
    st.number_input("Steps", min_value=1, max_value=100, value=50, key="steps", help="Number of steps for generating the image.")
    st.text_input("Image Dimensions", value="512x512", key="image_dimensions", help="Specify the dimensions of the output image.")
    st.file_uploader("Input Image", type=["png", "jpg"], key="input_image", help="Upload an input image for transformation.")
