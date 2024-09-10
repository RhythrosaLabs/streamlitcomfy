import streamlit as st
import requests
from PIL import Image
import io
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import logging
import zipfile
import os
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AINode:
    def __init__(self, id, name, model_id, input_type, output_type):
        self.id = id
        self.name = name
        self.model_id = model_id
        self.input_type = input_type
        self.output_type = output_type

# Verify Stable Diffusion API key
def verify_sd_api_key(sd_api_key):
    try:
        response = requests.get("https://api.stability.ai/v2beta/image-to-video", headers={"Authorization": f"Bearer {sd_api_key}"})
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Stable Diffusion API key verification failed: {str(e)}")
        return False

# Process image to video using Stable Diffusion API
def process_sd_image_to_video(api_key, image, seed=0, cfg_scale=1.8, motion_bucket_id=127):
    try:
        url = "https://api.stability.ai/v2beta/image-to-video"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        files = {
            "image": image
        }
        data = {
            "seed": seed,
            "cfg_scale": cfg_scale,
            "motion_bucket_id": motion_bucket_id
        }

        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()

        # Poll for the result using the returned ID
        generation_id = response.json().get('id')
        return generation_id
    except Exception as e:
        logger.error(f"Stable Diffusion API Error: {str(e)}")
        st.error(f"Stable Diffusion API Error: {str(e)}")
        return None

# Poll for video generation result
def poll_video_result(api_key, generation_id):
    try:
        url = f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get('video_url')
        return None
    except Exception as e:
        logger.error(f"Error polling for video result: {str(e)}")
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

def main():
    st.set_page_config(page_title="Replicate AI Pipeline Builder", layout="wide")
    
    st.title("Replicate AI Pipeline Builder")
    st.sidebar.header("Settings")

    if 'sd_api_key' not in st.session_state:
        st.session_state.sd_api_key = ""

    sd_api_key = st.sidebar.text_input("Stable Diffusion API Key", type="password", value=st.session_state.sd_api_key)

    if sd_api_key:
        st.session_state.sd_api_key = sd_api_key
        if verify_sd_api_key(sd_api_key):
            st.sidebar.success("Stable Diffusion API key verified successfully!")
        else:
            st.sidebar.error("Invalid Stable Diffusion API key. Please check and try again.")

    available_nodes = [
        AINode("video", "Stable Diffusion Image-to-Video", "custom-stable-diffusion-video", "image", "video"),
    ]

    if 'workflow' not in st.session_state:
        st.session_state.workflow = nx.DiGraph()
    if 'node_positions' not in st.session_state:
        st.session_state.node_positions = {}

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Pipeline Builder")
        
        with st.expander("Add Node", expanded=True):
            selected_node = st.selectbox("Select a node to add", available_nodes, format_func=lambda x: x.name)
            if st.button("Add Node", key="add_node"):
                if selected_node.id not in st.session_state.workflow.nodes:
                    st.session_state.workflow.add_node(selected_node.id, node=selected_node)
                    st.session_state.node_positions[selected_node.id] = (len(st.session_state.workflow.nodes) * 100, 0)
                    st.success(f"Added {selected_node.name} to the workflow.")
                else:
                    st.warning(f"{selected_node.name} is already in the workflow.")

    with col2:
        st.subheader("Workflow Visualization")
        nodes = [Node(id=n, label=data['node'].name, x=st.session_state.node_positions[n][0], y=st.session_state.node_positions[n][1]) 
                 for n, data in st.session_state.workflow.nodes(data=True)]
        edges = [Edge(source=u, target=v) for u, v in st.session_state.workflow.edges()]
        config = Config(width=600, height=400, directed=True, physics=True, hierarchical=False)
        agraph(nodes=nodes, edges=edges, config=config)

    st.subheader("Pipeline Execution")
    if st.session_state.workflow.nodes:
        start_nodes = [n for n, d in st.session_state.workflow.in_degree() if d == 0]
        if start_nodes:
            start_node = start_nodes[0]
            input_type = st.session_state.workflow.nodes[start_node]['node'].input_type
            if input_type == "image":
                user_input = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
                if user_input:
                    image = user_input.read()

            if st.button("Run Pipeline", key="run_pipeline"):
                if user_input and st.session_state.sd_api_key:
                    with st.spinner("Processing..."):
                        generation_id = process_sd_image_to_video(st.session_state.sd_api_key, image)
                        if generation_id:
                            video_url = poll_video_result(st.session_state.sd_api_key, generation_id)
                            if video_url:
                                st.video(video_url)
                            else:
                                st.error("Failed to generate the video.")
                        else:
                            st.error("Failed to start video generation.")
                else:
                    st.warning("Please provide input and your Stable Diffusion API key.")
        else:
            st.warning("Your workflow has no starting point. Please connect your nodes.")

    # Download button for generated files
    if st.session_state.generated_files:
        st.subheader("Download Generated Files")
        download_link = create_download_zip(st.session_state.generated_files)
        st.markdown(download_link, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
