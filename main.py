import streamlit as st
import replicate
import requests
from PIL import Image
import io
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import streamlit.components.v1 as components
import logging
import zipfile
import os
import base64
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AINode:
    def __init__(self, id, name, model_id, input_type, output_type, params=None):
        self.id = id
        self.name = name
        self.model_id = model_id
        self.input_type = input_type
        self.output_type = output_type
        self.params = params or {}

def verify_api_key(api_key):
    try:
        client = replicate.Client(api_token=api_key)
        client.models.list()
        return True
    except replicate.exceptions.ReplicateError as e:
        logger.error(f"API key verification failed: {str(e)}")
        return False

def process_replicate(node, input_data, **kwargs):
    try:
        if not st.session_state.get('replicate_api_key'):
            logger.error("API key is missing")
            st.error("API key is missing. Please enter your Replicate API key in the sidebar.")
            return None

        logger.info(f"Processing node: {node.name}")
        logger.info(f"Input type: {type(input_data)}")
        logger.info(f"Additional kwargs: {kwargs}")

        if node.input_type == "text":
            model_input = {"prompt": input_data}
        elif node.input_type == "image":
            if isinstance(input_data, Image.Image):
                img_byte_arr = io.BytesIO()
                input_data.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                model_input = {"image": img_byte_arr}
            elif isinstance(input_data, str) and input_data.startswith("http"):
                model_input = {"image": input_data}
            else:
                raise ValueError(f"Unsupported image input type: {type(input_data)}")
        else:
            raise ValueError(f"Unsupported input type: {node.input_type}")
        
        model_input.update(kwargs)
        model_input.update(node.params)
        
        logger.info(f"Model input: {model_input}")
        
        client = replicate.Client(api_token=st.session_state['replicate_api_key'])
        output = client.run(node.model_id, input=model_input)
        
        logger.info(f"Raw output: {output}")
        
        if isinstance(output, list) and len(output) > 0:
            return output[0] if isinstance(output[0], str) and output[0].startswith("http") else output[0]
        elif isinstance(output, str) and output.startswith("http"):
            return output
        else:
            return output
    except replicate.exceptions.ReplicateError as e:
        logger.error(f"Replicate API Error: {str(e)}")
        st.error(f"Replicate API Error: {str(e)}")
        st.error("Please check your API key and model settings.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

def download_file(url, filename):
    response = requests.get(url)
    return response.content

def create_download_zip(files):
    zip_filename = "generated_files.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for i, file in enumerate(files):
            if isinstance(file, str) and file.startswith("http"):
                content = download_file(file, f"file_{i}")
                ext = "txt" if file.endswith(".txt") else "png"  # Default to png for images
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

def create_node(node_type, x, y):
    node_id = f"{node_type}_{len(st.session_state.workflow.nodes)}"
    node = next((AINode(n.id, n.name, n.model_id, n.input_type, n.output_type, n.params) 
                 for n in available_nodes if n.id == node_type), None)
    if node:
        st.session_state.workflow.add_node(node_id, node=node)
        st.session_state.node_positions[node_id] = (x, y)
        return node_id
    return None

def update_node_position(node_id, x, y):
    st.session_state.node_positions[node_id] = (x, y)

def create_edge(source_id, target_id, edge_type):
    if not st.session_state.workflow.has_edge(source_id, target_id):
        st.session_state.workflow.add_edge(source_id, target_id, type=edge_type)

def remove_node(node_id):
    st.session_state.workflow.remove_node(node_id)
    del st.session_state.node_positions[node_id]

def remove_edge(source_id, target_id):
    st.session_state.workflow.remove_edge(source_id, target_id)

def update_node_properties(node_id, properties):
    node = st.session_state.workflow.nodes[node_id]['node']
    for key, value in properties.items():
        if key in node.params:
            node.params[key] = value
        else:
            setattr(node, key, value)

def display_node_properties_in_sidebar(node):
    st.sidebar.subheader(f"Node: {node.name} Properties")

    # Dynamically update the sidebar based on node type
    for param, value in node.params.items():
        if isinstance(value, str):
            node.params[param] = st.sidebar.text_input(param, value)
        elif isinstance(value, (int, float)):
            node.params[param] = st.sidebar.number_input(param, value=value)
        elif isinstance(value, bool):
            node.params[param] = st.sidebar.checkbox(param, value)
        elif isinstance(value, Image.Image):
            uploaded_file = st.sidebar.file_uploader(f"Upload {param}", type=["png", "jpg", "jpeg"])
            if uploaded_file:
                node.params[param] = Image.open(uploaded_file)

def main():
    st.set_page_config(page_title="Advanced Interactive AI Pipeline Builder", layout="wide")
    
    st.title("Advanced Interactive AI Pipeline Builder")
    st.sidebar.header("Settings")

    # Initialize session state
    if 'replicate_api_key' not in st.session_state:
        st.session_state.replicate_api_key = ""
    if 'generated_files' not in st.session_state:
        st.session_state.generated_files = []
    if 'workflow' not in st.session_state:
        st.session_state.workflow = nx.DiGraph()
    if 'node_positions' not in st.session_state:
        st.session_state.node_positions = {}
    if 'selected_node' not in st.session_state:
        st.session_state.selected_node = None
    if 'edge_creation_mode' not in st.session_state:
        st.session_state.edge_creation_mode = False

    # API key input
    replicate_api_key = st.sidebar.text_input("Replicate API Key", type="password", value=st.session_state.replicate_api_key)

    # Verify and save Replicate API key
    if replicate_api_key and replicate_api_key != st.session_state.replicate_api_key:
        st.session_state.replicate_api_key = replicate_api_key
        if verify_api_key(replicate_api_key):
            st.sidebar.success("Replicate API key verified successfully!")
        else:
            st.sidebar.error("Invalid Replicate API key. Please check and try again.")

    global available_nodes
    available_nodes = [
        AINode("text_input", "Text Input", None, None, "text", {"text": ""}),
        AINode("image_input", "Image Input", None, None, "image", {"image": None}),
        AINode("text_to_image", "Text to Image", "stability-ai/sdxl:a00d0b7dcbb9c3fbb34ba87d2d5b46c56969c84a628bf778a7fdaec30b1b99c5", "text", "image", {"prompt": "", "negative_prompt": ""}),
        AINode("image_to_image", "Image to Image", "stability-ai/sdxl:a00d0b7dcbb9c3fbb34ba87d2d5b46c56969c84a628bf778a7fdaec30b1b99c5", "image", "image", {"prompt": "", "strength": 0.75}),
        AINode("upscale", "Upscale", "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b", "image", "image", {"scale": 2}),
        AINode("remove_bg", "Remove Background", "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003", "image", "image", {}),
        AINode("music_gen", "Music Generation", "meta/musicgen:7a76a8258b23fae65c5a22debb8841d1d7e816b75c2f24218cd2bd8573787906", "text", "audio", {"prompt": ""}),
    ]

    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        st.subheader("Available Nodes")
        for node in available_nodes:
            if st.button(node.name, key=f"add_{node.id}"):
                create_node(node.id, 0, 0)

    with col2:
        st.subheader("Interactive Workflow Visualizer")
        
        nodes = [Node(id=n, label=data['node'].name, 
                      x=st.session_state.node_positions[n][0], 
                      y=st.session_state.node_positions[n][1],
                      color="#00BFFF" if n == st.session_state.selected_node else "#FFFFFF") 
                 for n, data in st.session_state.workflow.nodes(data=True)]
        edges = [Edge(source=u, target=v, type=data['type']) 
                 for u, v, data in st.session_state.workflow.edges(data=True)]
        
        config = Config(width=800, 
                        height=600, 
                        directed=True, 
                        physics=False, 
                        hierarchical=False)
        
        return_value = agraph(nodes=nodes, 
                              edges=edges, 
                              config=config)

        # Ensure return_value exists and has the necessary 'type' key
        if return_value and isinstance(return_value, dict) and 'type' in return_value:
            if return_value['type'] == 'add_node':
                node_type = st.selectbox("Select node type", [n.id for n in available_nodes])
                create_node(node_type, return_value['x'], return_value['y'])
            elif return_value['type'] == 'move_node':
                update_node_position(return_value['id'], return_value['x'], return_value['y'])
            elif return_value['type'] == 'select_node':
                st.session_state.selected_node = return_value['id']
                selected_node = st.session_state.workflow.nodes[st.session_state.selected_node]['node']
                display_node_properties_in_sidebar(selected_node)
            elif return_value['type'] == 'add_edge':
                edge_type = st.selectbox("Select edge type", ["data", "control"])
                create_edge(return_value['source'], return_value['target'], edge_type)
            elif return_value['type'] == 'remove_edge':
                remove_edge(return_value['source'], return_value['target'])

    with col3:
        st.subheader("Node Properties")
        if st.session_state.selected_node:
            node = st.session_state.workflow.nodes[st.session_state.selected_node]['node']
            st.write(f"Editing: {node.name}")
            
            # Display and edit node properties in the main area
            new_properties = {}
            for attr in ['input_type', 'output_type', 'model_id']:
                new_value = st.text_input(attr, getattr(node, attr))
                new_properties[attr] = new_value
            
            # Display and edit node-specific parameters
            for param, value in node.params.items():
                if isinstance(value, str):
                    new_value = st.text_input(param, value)
                elif isinstance(value, (int, float)):
                    new_value = st.number_input(param, value=value)
                elif isinstance(value, bool):
                    new_value = st.checkbox(param, value)
                elif isinstance(value, Image.Image):
                    new_value = st.file_uploader(f"Upload {param}", type=["png", "jpg", "jpeg"])
                    if new_value:
                        new_value = Image.open(new_value)
                else:
                    new_value = st.text_input(param, str(value))
                new_properties[param] = new_value
            
            if st.button("Update Properties"):
                update_node_properties(st.session_state.selected_node, new_properties)
            
            if st.button("Remove Node"):
                remove_node(st.session_state.selected_node)
                st.session_state.selected_node = None

    st.subheader("Pipeline Execution")
    if st.session_state.workflow.nodes:
        start_nodes = [n for n, d in st.session_state.workflow.in_degree() if d == 0]
        if start_nodes:
            if st.button("Run Pipeline", key="run_pipeline"):
                if st.session_state.replicate_api_key:
                    with st.spinner("Processing..."):
                        st.session_state.generated_files = []
                        for node_id in nx.topological_sort(st.session_state.workflow):
                            node = st.session_state.workflow.nodes[node_id]['node']
                            
                            with st.expander(f"Processing: {node.name}", expanded=True):
                                st.info(f"Processing node: {node.name}")
                                
                                # Handle input nodes
                                if node.input_type is None:
                                    if node.output_type == "text":
                                        current_output = st.text_area(f"Enter text for {node.name}", node.params.get("text", ""))
                                    elif node.output_type == "image":
                                        uploaded_file = st.file_uploader(f"Upload image for {node.name}", type=["png", "jpg", "jpeg"])
                                        if uploaded_file:
                                            current_output = Image.open(uploaded_file)
                                        else:
                                            st.warning(f"Please upload an image for {node.name}")
                                            break
                                else:
                                    # Get input from previous nodes
                                    prev_nodes = list(st.session_state.workflow.predecessors(node_id))
                                    if prev_nodes:
                                        current_output = st.session_state.generated_files[-1]  # Get output from the last processed node
                                    else:
                                        st.warning(f"No input available for {node.name}")
                                        break
                                
                                # Process the node
                                if node.model_id:  # Only process if it's not an input node
                                    current_output = process_replicate(node, current_output)
                                
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
                elif not st.session_state.replicate_api_key:
                    st.warning("Please enter your API key in the sidebar.")
        else:
            st.warning("Your workflow has no starting point. Please add input nodes to the workflow.")
    else:
        st.warning("Please add at least one node to the workflow.")

    # Download button for generated files
    if st.session_state.generated_files:
        st.subheader("Download Generated Files")
        download_link = create_download_zip(st.session_state.generated_files)
        st.markdown(download_link, unsafe_allow_html=True)

    # Display previously generated files
    if st.session_state.generated_files:
        st.subheader("Previously Generated Files")
        for i, file in enumerate(st.session_state.generated_files):
            if isinstance(file, Image.Image):
                st.image(file, caption=f"Generated Image {i+1}", use_column_width=True)
            elif isinstance(file, str) and file.startswith("http"):
                if file.endswith((".png", ".jpg", ".jpeg")):
                    st.image(file, caption=f"Generated Image {i+1}", use_column_width=True)
                elif file.endswith(".mp4"):
                    st.video(file)
                elif file.endswith((".mp3", ".wav")):
                    st.audio(file)
                else:
                    st.write(f"Generated File {i+1}: {file}")
            else:
                st.write(f"Generated Output {i+1}: {file}")

if __name__ == "__main__":
    main()
