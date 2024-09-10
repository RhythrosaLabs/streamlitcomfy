import streamlit as st
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionUpscalePipeline
import torch
from PIL import Image
import time

# Title and Description
st.title("ComfyUI - Streamlit Edition")
st.write("A modular, node-based interface for Stable Diffusion workflows.")

# Sidebar: GPU Info
gpu_available = torch.cuda.is_available()
gpu_stats = torch.cuda.get_device_properties(0) if gpu_available else None

st.sidebar.title("System Info")
st.sidebar.metric("GPU Available", str(gpu_available))
if gpu_available:
    st.sidebar.metric("GPU Name", gpu_stats.name)
    st.sidebar.metric("GPU Memory", f"{gpu_stats.total_memory // 1e9} GB")

# Sidebar: Model Selection
task = st.sidebar.selectbox(
    "Select Task", ["Text to Image", "Inpainting", "Upscaling"]
)

model_selection = st.sidebar.selectbox(
    "Select Model", ["SD 1.x", "SD 2.x", "SDXL"]
)

# Common Parameters
cfg_scale = st.sidebar.slider("CFG Scale", 1.0, 20.0, 7.5)
steps = st.sidebar.slider("Steps", 10, 150, 50)
height = st.sidebar.slider("Image Height", 256, 1024, 512)
width = st.sidebar.slider("Image Width", 256, 1024, 512)

# Load the correct model based on task
@st.cache_resource(show_spinner=False)
def load_model(task, model_choice):
    if task == "Text to Image":
        if model_choice == "SD 1.x":
            return StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
        elif model_choice == "SD 2.x":
            return StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
        else:
            return StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl").to("cuda")
    elif task == "Inpainting":
        return StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-inpainting").to("cuda")
    elif task == "Upscaling":
        return StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler").to("cuda")

# Input fields based on task selection
if task == "Text to Image":
    st.header("Text to Image Task")
    prompt = st.text_area("Enter your text prompt:", "A futuristic city skyline at dusk")
    
    if st.button("Generate Image"):
        pipe = load_model(task, model_selection)
        with st.spinner("Generating image..."):
            image = pipe(prompt, num_inference_steps=steps, guidance_scale=cfg_scale, height=height, width=width).images[0]
            st.image(image, caption="Generated Image")
            st.success("Image Generated!")
        
        # Option to download
        with open("generated_image.png", "wb") as f:
            image.save(f, format="PNG")
        st.download_button(label="Download Image", data=open("generated_image.png", "rb"), file_name="image.png", mime="image/png")

elif task == "Inpainting":
    st.header("Inpainting Task")
    uploaded_image = st.file_uploader("Upload an image for inpainting", type=["png", "jpg"])
    uploaded_mask = st.file_uploader("Upload a mask image", type=["png", "jpg"])

    if uploaded_image and uploaded_mask:
        image = Image.open(uploaded_image)
        mask = Image.open(uploaded_mask)

        if st.button("Inpaint"):
            pipe = load_model(task, model_selection)
            with st.spinner("Inpainting..."):
                result_image = pipe(image=image, mask_image=mask, num_inference_steps=steps, guidance_scale=cfg_scale).images[0]
                st.image(result_image, caption="Inpainted Image")
                st.success("Inpainting Completed!")

            # Option to download
            with open("inpainted_image.png", "wb") as f:
                result_image.save(f, format="PNG")
            st.download_button(label="Download Inpainted Image", data=open("inpainted_image.png", "rb"), file_name="inpainted_image.png", mime="image/png")

elif task == "Upscaling":
    st.header("Upscaling Task")
    uploaded_image = st.file_uploader("Upload an image to upscale", type=["png", "jpg"])

    if uploaded_image:
        image = Image.open(uploaded_image)

        if st.button("Upscale Image"):
            pipe = load_model(task, model_selection)
            with st.spinner("Upscaling..."):
                upscaled_image = pipe(image=image).images[0]
                st.image(upscaled_image, caption="Upscaled Image")
                st.success("Upscaling Completed!")

            # Option to download
            with open("upscaled_image.png", "wb") as f:
                upscaled_image.save(f, format="PNG")
            st.download_button(label="Download Upscaled Image", data=open("upscaled_image.png", "rb"), file_name="upscaled_image.png", mime="image/png")

# Progress Bars for Long Operations
progress_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.05)  # Simulating a long operation
    progress_bar.progress(percent_complete + 1)

st.write("Processing complete!")

# Exception Handling (Placeholder Example)
try:
    # Example: Pretend this is some process that may fail
    result = 1 / 0  # This will trigger an exception
except ZeroDivisionError as e:
    st.error(f"An error occurred: {e}")

