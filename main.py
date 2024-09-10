import streamlit as st
import replicate
import requests
from PIL import Image
import io
import networkx as nx
import logging
import zipfile
import os
import base64
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AINode:
    def __init__(self, id, name, model_id, input_type, output_type, parameters):
        self.id = id
        self.name = name
        self.model_id = model_id
        self.input_type = input_type
        self.output_type = output_type
        self.parameters = parameters

def verify_api_key(api_key):
    try:
        client = replicate.Client(api_token=api_key)
        client.models.list()
        return True
    except replicate.exceptions.ReplicateError as e:
        logger.error(f"API key verification failed: {str(e)}")
        return False

def process_replicate(node, input_data, api_key, parameters):
    try:
        if not api_key:
            logger.error("API key is missing")
            return None

        logger.info(f"Processing node: {node.name}")
        logger.info(f"Input type: {type(input_data)}")

        model_input = parameters.copy()
        if node.input_type == "text":
            model_input["prompt"] = input_data
        elif node.input_type == "image":
            if isinstance(input_data, Image.Image):
                img_byte_arr = io.BytesIO()
                input_data.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                model_input["image"] = img_byte_arr
            elif isinstance(input_data, str) and input_data.startswith("http"):
                model_input["image"] = input_data
            else:
                raise ValueError(f"Unsupported image input type: {type(input_data)}")
        
        logger.info(f"Model input: {model_input}")
        
        client = replicate.Client(api_token=api_key)
        output = client.run(node.model_id, input=model_input)
        
        logger.info(f"Raw output: {output}")
        
        if isinstance(output, list) and len(output) > 0:
            return output[0] if isinstance(output[0], str) and output[0].startswith("http") else output[0]
        elif isinstance(output, str) and output.startswith("http"):
            return output
        else:
            return output
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return None

def download_file(url):
    response = requests.get(url)
    return response.content

def create_download_zip(files):
    zip_filename = "generated_files.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for i, file in enumerate(files):
            if isinstance(file, str) and file.startswith("http"):
                content = download_file(file)
                ext = file.split(".")[-1] if "." in file else "bin"
                zipf.writestr(f"file_{i}.{ext}", content)
            elif isinstance(file, Image.Image):
                img_byte_arr = io.BytesIO()
                file.save(img_byte_arr, format='PNG')
                zipf.writestr(f"file_{i}.png", img_byte_arr.getvalue())
            else:
                zipf.writestr(f"file_{i}.txt", str(file))
    
    with open(zip_filename, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:application/zip;base64,{b64}" download="{zip_filename}">Download Generated Files</a>'
    
    os.remove(zip_filename)
    return href

def main():
    st.set_page_config(page_title="AI Pipeline Builder", layout="wide")
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'nodes' not in st.session_state:
        st.session_state.nodes = {}
    if 'connections' not in st.session_state:
        st.session_state.connections = []
    if 'generated_files' not in st.session_state:
        st.session_state.generated_files = []

    st.sidebar.title("Settings")
    api_key = st.sidebar.text_input("Replicate API Key", type="password", value=st.session_state.api_key)
    if api_key:
        st.session_state.api_key = api_key
        if verify_api_key(api_key):
            st.sidebar.success("API key verified successfully!")
        else:
            st.sidebar.error("Invalid API key. Please check and try again.")

    available_nodes = [
        AINode("flux", "Flux Schnell", "black-forest-labs/flux-schnell", "text", "image", 
               {"num_outputs": 1, "aspect_ratio": "1:1", "output_format": "webp", "output_quality": 80}),
        AINode("sdxl", "Stable Diffusion XL", "stability-ai/sdxl:a00d0b7dcbb9c3fbb34ba87d2d5b46c56969c84a628bf778a7fdaec30b1b99c5", "text", "image", 
               {"width": 1024, "height": 1024, "num_outputs": 1, "prompt_strength": 0.8}),
        AINode("video", "Video Generation", "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351", "text", "video", 
               {"num_frames": 50, "fps": 25}),
        AINode("upscale", "Image Upscaling", "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b", "image", "image", 
               {"scale": 2}),
        AINode("music-gen", "MusicGen", "meta/musicgen:7a76a8258b23fae65c5a22debb8841d1d7e816b75c2f24218cd2bd8573787906", "text", "audio", 
               {"duration": 10, "top_k": 250, "top_p": 0.0, "temperature": 1.0, "classifier_free_guidance": 3.0}),
    ]

    st.title("AI Pipeline Builder")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Pipeline Nodes")
        for node_id, node in st.session_state.nodes.items():
            st.write(f"{node.name} (ID: {node_id})")
        
        st.subheader("Node Connections")
        for source, target in st.session_state.connections:
            st.write(f"{st.session_state.nodes[source].name} -> {st.session_state.nodes[target].name}")

    with col2:
        st.subheader("Add Node")
        selected_node = st.selectbox("Select a node to add", available_nodes, format_func=lambda x: x.name)
        if st.button("Add Node"):
            new_id = str(uuid.uuid4())
            st.session_state.nodes[new_id] = selected_node
            st.success(f"Added {selected_node.name} to the pipeline")

        st.subheader("Connect Nodes")
        if len(st.session_state.nodes) > 1:
            source = st.selectbox("From", list(st.session_state.nodes.keys()), format_func=lambda x: st.session_state.nodes[x].name)
            target = st.selectbox("To", [n for n in st.session_state.nodes.keys() if n != source], format_func=lambda x: st.session_state.nodes[x].name)
            if st.button("Connect Nodes"):
                st.session_state.connections.append((source, target))
                st.success(f"Connected {st.session_state.nodes[source].name} to {st.session_state.nodes[target].name}")

    st.subheader("Node Parameters")
    for node_id, node in st.session_state.nodes.items():
        with st.expander(f"{node.name} Parameters"):
            for param, value in node.parameters.items():
                if isinstance(value, bool):
                    node.parameters[param] = st.checkbox(f"{param} ({node.name})", value)
                elif isinstance(value, (int, float)):
                    node.parameters[param] = st.slider(f"{param} ({node.name})", 0, 100, int(value))
                else:
                    node.parameters[param] = st.text_input(f"{param} ({node.name})", str(value))

    if st.button("Run Pipeline"):
        if st.session_state.api_key and st.session_state.nodes:
            with st.spinner("Processing pipeline..."):
                G = nx.DiGraph()
                G.add_edges_from(st.session_state.connections)
                
                start_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
                
                for start_node in start_nodes:
                    current_node = st.session_state.nodes[start_node]
                    if current_node.input_type == "text":
                        user_input = st.text_input(f"Enter text for {current_node.name}:")
                    elif current_node.input_type == "image":
                        user_input = st.file_uploader(f"Upload image for {current_node.name}:", type=["png", "jpg", "jpeg"])
                        if user_input:
                            user_input = Image.open(user_input)
                    
                    if user_input:
                        current_output = user_input
                        for node_id in nx.topological_sort(G):
                            node = st.session_state.nodes[node_id]
                            with st.expander(f"Processing: {node.name}", expanded=True):
                                st.info(f"Processing node: {node.name}")
                                current_output = process_replicate(node, current_output, st.session_state.api_key, node.parameters)
                                
                                if current_output is None:
                                    st.error(f"Processing failed at node: {node.name}")
                                    break
                                
                                st.success(f"Output from {node.name}:")
                                if isinstance(current_output, Image.Image):
                                    st.image(current_output, caption=f"Output from {node.name}", use_column_width=True)
                                elif isinstance(current_output, str) and current_output.startswith("http"):
                                    if node.output_type == "audio":
                                        st.audio(current_output)
                                    elif node.output_type == "video":
                                        st.video(current_output)
                                    else:
                                        st.image(current_output, caption=f"Output from {node.name}", use_column_width=True)
                                else:
                                    st.write(current_output)
                                
                                st.session_state.generated_files.append(current_output)
                        
                        st.success("Pipeline execution completed!")
                    else:
                        st.warning(f"Please provide input for {current_node.name}.")
        elif not st.session_state.api_key:
            st.warning("Please enter your Replicate API key in the sidebar.")
        else:
            st.warning("Please add at least one node to the pipeline.")

    if st.session_state.generated_files:
        st.subheader("Download Generated Files")
        download_link = create_download_zip(st.session_state.generated_files)
        st.markdown(download_link, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
