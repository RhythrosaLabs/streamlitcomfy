import streamlit as st
from streamlit_elements import elements, mui, dashboard
from streamlit_lottie import st_lottie
import replicate

# API Key for Replicate
st.sidebar.text_input("Replicate API Key", key='replicate_key', type='password')

with elements("main"):
    # Grid layout for nodes
    with dashboard.Grid(draggableHandle=".drag-handle") as layout:
        # Node 1: Text Generation (LLaMA)
        with layout.item("llama", 0, 0, 6, 2):
            st.title("LLaMA 70B - Text Generation")
            llama_prompt = st.text_input("Text prompt", key="llama_prompt")
            if st.button("Generate Text"):
                output = replicate.run("meta-llama/Llama-2-70b", input={"text": llama_prompt})
                st.write(output)

        # Node 2: Image Generation (Flux)
        with layout.item("flux", 0, 2, 6, 2):
            st.title("Flux - Text to Image")
            image_prompt = st.text_input("Image prompt", key="flux_prompt")
            if st.button("Generate Image"):
                image = replicate.run("black-forest-labs/flux-schnell", input={"prompt": image_prompt})
                st.image(image)

        # Node 3: Upscale (Real-ESRGAN)
        with layout.item("esrgan", 0, 4, 6, 2):
            st.title("Real-ESRGAN - Image Upscaling")
            image_url = st.text_input("Image URL", key="esrgan_url")
            if st.button("Upscale Image"):
                upscale_output = replicate.run("nightmareai/real-esrgan", input={"image": image_url, "scale": 4})
                st.image(upscale_output)

