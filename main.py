import streamlit as st
from streamlit_elements import elements, mui
import replicate
import requests

# Initialize session state for API keys
if 'llama_key' not in st.session_state:
    st.session_state['llama_key'] = ''
if 'replicate_key' not in st.session_state:
    st.session_state['replicate_key'] = ''

# Function to call LLaMA 70B
def llama_generate(prompt):
    headers = {"Authorization": f"Bearer {st.session_state['llama_key']}"}
    url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b"
    response = requests.post(url, json={"inputs": prompt}, headers=headers)
    return response.json()

# Function to call Flux for image generation
def flux_generate(prompt):
    output = replicate.run(
        "black-forest-labs/flux-schnell", 
        input={"prompt": prompt},
        api_token=st.session_state['replicate_key']
    )
    return output

# Function to call Real-ESRGAN for upscaling
def upscale_image(image_url):
    output = replicate.run(
        "nightmareai/real-esrgan", 
        input={"image": image_url, "scale": 4},
        api_token=st.session_state['replicate_key']
    )
    return output

# Sidebar for API keys
st.sidebar.title("API Keys")
st.sidebar.text_input("LLaMA API Key", key='llama_key', type='password')
st.sidebar.text_input("Replicate API Key", key='replicate_key', type='password')

# Streamlit Elements UI
with elements("main"):
    with mui.Container():
        # Text generation Node
        with mui.Paper(elevation=3):
            st.write("### LLaMA 70B - Text Generation")
            prompt = st.text_input("Enter your text prompt for LLaMA", "")
            if st.button("Generate Text"):
                if prompt:
                    text_output = llama_generate(prompt)
                    st.write(f"Generated Text: {text_output}")
                else:
                    st.write("Please provide a prompt.")

        # Image generation Node
        with mui.Paper(elevation=3):
            st.write("### Flux - Text to Image")
            image_prompt = st.text_input("Enter your text prompt for Flux", "")
            if st.button("Generate Image"):
                if image_prompt:
                    image_output = flux_generate(image_prompt)
                    st.image(image_output)
                else:
                    st.write("Please provide an image prompt.")

        # Image Upscaling Node
        with mui.Paper(elevation=3):
            st.write("### Real-ESRGAN - Image Upscaling")
            image_url = st.text_input("Enter image URL for upscaling", "")
            if st.button("Upscale Image"):
                if image_url:
                    upscale_output = upscale_image(image_url)
                    st.image(upscale_output)
                else:
                    st.write("Please provide an image URL.")
