import streamlit as st
from streamlit_elements import elements, mui, dashboard
from diffusers import StableDiffusionPipeline
import torch

# Initialize models dictionary to store nodes' models
models = {}

# Function to load models dynamically, running on CPU to avoid high memory usage
@st.cache_resource(show_spinner=False)
def load_model(model_choice):
    device = "cpu"  # Using CPU for stability in low-resource environments
    if model_choice == "SD 1.x":
        return StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    elif model_choice == "SD 2.x":
        return StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to(device)
    else:
        return StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl").to(device)

# Sidebar for task and model selection
task = st.sidebar.selectbox("Select Task", ["Text to Image"])
model_selection = st.sidebar.selectbox("Select Model", ["SD 1.x", "SD 2.x", "SDXL"])

# Image resolution sliders (reduced default resolution)
height = st.sidebar.slider("Image Height", 128, 512, 256)
width = st.sidebar.slider("Image Width", 128, 512, 256)
steps = st.sidebar.slider("Steps", 10, 50, 25)
guidance_scale = st.sidebar.slider("CFG Scale", 1.0, 20.0, 7.5)

# Dashboard layout settings for draggable nodes
layout = [
    dashboard.Item("node1", 0, 0, 3, 3),  # Node1 grid position and size
    dashboard.Item("node2", 3, 0, 3, 3),  # Node2 grid position and size
]

with elements("dashboard"):
    with dashboard.Grid(layout, draggable=True, resizable=True):
        # Node 1 for input prompt and model selection
        with mui.Paper(id="node1", elevation=3):
            st.write("Node 1: Input Prompt")
            prompt = st.text_input("Enter prompt for Node 1", "A futuristic city skyline")
            models["node1"] = load_model(model_selection)
        
        # Node 2 to connect and process
        with mui.Paper(id="node2", elevation=3):
            st.write("Node 2: Generate Image")
            if st.button("Generate from Node 1"):
                if "node1" in models:
                    pipe = models["node1"]
                    with st.spinner("Generating image..."):
                        image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale, height=height, width=width).images[0]
                        st.image(image, caption="Generated Image from Node 1")
                else:
                    st.warning("Node 1 is not connected or assigned a model.")

# Generate button for the entire pipeline
if st.sidebar.button("Generate Entire Pipeline"):
    if "node1" in models:
        pipe = models["node1"]
        st.sidebar.write("Generating image from Node 1...")
        with st.spinner("Generating image..."):
            image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale, height=height, width=width).images[0]
            st.sidebar.image(image, caption="Generated Image")
    else:
        st.sidebar.error("No model assigned to Node 1")
