import streamlit as st
import replicate

# Initialize session state for nodes and outputs
if "nodes" not in st.session_state:
    st.session_state["nodes"] = []

if "outputs" not in st.session_state:
    st.session_state["outputs"] = {}

# Input for the API Key
api_key = st.text_input("Enter your Replicate API Key", type="password")

if api_key:
    client = replicate.Client(api_token=api_key)

    # List of available models and their parameters
    available_models = {
        "LLaMA 70b (Text Generation)": {
            "id": "replicate/llama-70b",  # Double check this model name
            "params": ["prompt", "max_length", "temperature"]
        },
        "Flux Pro (Art Generation)": {
            "id": "black-forest-labs/flux-pro",  # Double check this model name
            "params": ["prompt", "style", "guidance_scale"]
        },
        "Image Upscaler": {
            "id": "stability-ai/stable-diffusion-x4-upscaler",  # Confirm this slug exists
            "params": ["image", "upscale_factor"]
        }
    }

    # Add new node
    if st.button("Add Node"):
        st.session_state["nodes"].append({
            "model": "LLaMA 70b (Text Generation)",  # Default model
            "params": {},
            "output": None
        })

    # Display and configure nodes
    for i, node in enumerate(st.session_state["nodes"]):
        st.write(f"### Node {i + 1}")
        
        # Select model
        model_name = st.selectbox(f"Choose Model for Node {i + 1}", list(available_models.keys()), key=f"model_{i}")
        st.session_state["nodes"][i]["model"] = model_name

        # Get model info
        model_info = available_models[model_name]

        # Input parameters based on model
        if model_name == "LLaMA 70b (Text Generation)":
            prompt = st.text_area(f"Prompt for Node {i + 1}", key=f"prompt_{i}")
            max_length = st.slider(f"Max Length for Node {i + 1}", 50, 500, 150, key=f"max_length_{i}")
            temperature = st.slider(f"Temperature for Node {i + 1}", 0.0, 1.0, 0.75, key=f"temperature_{i}")
            st.session_state["nodes"][i]["params"] = {
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature
            }

        elif model_name == "Flux Pro (Art Generation)":
            prompt = st.text_area(f"Art Prompt for Node {i + 1}", key=f"prompt_{i}")
            style = st.selectbox(f"Style for Node {i + 1}", ["abstract", "cyberpunk", "fantasy", "realistic"], key=f"style_{i}")
            guidance_scale = st.slider(f"Guidance Scale for Node {i + 1}", 0.5, 20.0, 7.5, key=f"guidance_scale_{i}")
            st.session_state["nodes"][i]["params"] = {
                "prompt": prompt,
                "style": style,
                "guidance_scale": guidance_scale
            }

        elif model_name == "Image Upscaler":
            uploaded_file = st.file_uploader(f"Upload an Image for Node {i + 1}", type=["png", "jpg", "jpeg"], key=f"image_{i}")
            upscale_factor = st.slider(f"Upscale Factor for Node {i + 1}", 2, 8, 4, key=f"upscale_factor_{i}")
            st.session_state["nodes"][i]["params"] = {
                "image": uploaded_file,
                "upscale_factor": upscale_factor
            }

        # Button to remove node
        if st.button(f"Remove Node {i + 1}"):
            st.session_state["nodes"].pop(i)
            st.experimental_rerun()

        # Run model and get output using replicate.run
        if st.button(f"Run Node {i + 1}"):
            try:
                model_id = model_info["id"]
                output = replicate.run(
                    f"{model_id}:latest",
                    input=st.session_state["nodes"][i]["params"]
                )
                st.session_state["nodes"][i]["output"] = output
                st.session_state["outputs"][f"Node_{i+1}_output"] = output
                st.write(f"Output for Node {i + 1}: {output}")
            except Exception as e:
                st.error(f"Error in Node {i + 1}: {e}")

    # Chain outputs
    st.write("### Chain Outputs Between Nodes")
    for i in range(len(st.session_state["nodes"]) - 1):
        if f"Node_{i+1}_output" in st.session_state["outputs"]:
            st.write(f"Passing output from Node {i + 1} to Node {i + 2}")
            next_node = st.session_state["nodes"][i + 1]
            if isinstance(st.session_state["outputs"][f"Node_{i+1}_output"], str):  # For text models
                next_node["params"]["prompt"] = st.session_state["outputs"][f"Node_{i+1}_output"]
            elif isinstance(st.session_state["outputs"][f"Node_{i+1}_output"], bytes):  # For image models
                next_node["params"]["image"] = st.session_state["outputs"][f"Node_{i+1}_output"]

