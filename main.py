import streamlit as st
from streamlit_elements import elements, dashboard, mui
import replicate

# Initialize the Replicate API key in session state
if 'replicate_key' not in st.session_state:
    st.session_state['replicate_key'] = ''

# Sidebar for API key input
st.sidebar.title("API Keys")
st.sidebar.text_input("Replicate API Key", key='replicate_key', type='password')

# Function to call LLaMA 70B for text generation
def llama_generate(prompt):
    try:
        output = replicate.run(
            "meta-llama/Llama-2-70b",
            input={"text": prompt},
            api_token=st.session_state['replicate_key']
        )
        return output
    except Exception as e:
        st.error(f"Error generating text: {e}")
        return None

# Function to call Flux for text-to-image generation
def flux_generate(prompt):
    try:
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={"prompt": prompt},
            api_token=st.session_state['replicate_key']
        )
        return output
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

# Function to call Real-ESRGAN for image upscaling
def upscale_image(image_url):
    try:
        output = replicate.run(
            "nightmareai/real-esrgan",
            input={"image": image_url, "scale": 4},
            api_token=st.session_state['replicate_key']
        )
        return output
    except Exception as e:
        st.error(f"Error upscaling image: {e}")
        return None

# Main interactive node-based UI using Streamlit Elements
with elements("main"):
    # Correctly initialize the Grid dashboard layout
    layout = dashboard.Grid(cols=12, rowHeight=160, draggableHandle=".drag-handle")

    # Node for LLaMA Text Generation
    with layout.item("llama", 0, 0, 6, 2):
        with mui.Paper(elevation=3):
            st.write("### LLaMA 70B - Text Generation")
            llama_prompt = st.text_input("Enter your text prompt for LLaMA", key="llama_prompt")
            if st.button("Generate Text", key="generate_llama"):
                if llama_prompt:
                    text_output = llama_generate(llama_prompt)
                    if text_output:
                        st.write(f"Generated Text: {text_output}")
                else:
                    st.write("Please provide a prompt.")

    # Node for Flux Text-to-Image Generation
    with layout.item("flux", 6, 0, 6, 2):
        with mui.Paper(elevation=3):
            st.write("### Flux - Text to Image Generation")
            flux_prompt = st.text_input("Enter your text prompt for Flux", key="flux_prompt")
            if st.button("Generate Image", key="generate_flux"):
                if flux_prompt:
                    image_output = flux_generate(flux_prompt)
                    if image_output:
                        st.image(image_output)
                else:
                    st.write("Please provide a prompt.")

    # Node for Real-ESRGAN Image Upscaling
    with layout.item("esrgan", 0, 2, 6, 2):
        with mui.Paper(elevation=3):
            st.write("### Real-ESRGAN - Image Upscaling")
            image_url = st.text_input("Enter image URL for upscaling", key="image_url")
            if st.button("Upscale Image", key="upscale_esrgan"):
                if image_url:
                    upscale_output = upscale_image(image_url)
                    if upscale_output:
                        st.image(upscale_output)
                else:
                    st.write("Please provide an image URL.")

    # Save the layout
    layout.save()
