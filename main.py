import streamlit as st
import replicate
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

def verify_api_key(api_key):
    try:
        client = replicate.Client(api_token=api_key)
        models = client.models.list()
        next(models)  # Try to get the first model
        return True
    except replicate.exceptions.ReplicateError as e:
        logger.error(f"API key verification failed: {str(e)}")
        return False
    except StopIteration:
        # If we can't get any models but no exception was raised, the key is probably valid
        return True
    except Exception as e:
        logger.error(f"Unexpected error during API key verification: {str(e)}")
        return False

def process_replicate(node, input_data, **kwargs):
    try:
        if not st.session_state.api_key:
            logger.error("API key is missing")
            st.error("API key is missing. Please enter your Replicate API key in the sidebar.")
            return None

        logger.info(f"Processing node: {node.name}")
        logger.info(f"Input type: {type(input_data)}")
        logger.info(f"Additional kwargs: {kwargs}")

        model_input = kwargs.copy()

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

        if node.id == "dynami-crafter":
            model_input["i2v_input_image"] = model_input.pop("image", None)
        
        logger.info(f"Model input: {model_input}")
        
        client = replicate.Client(api_token=st.session_state.api_key)
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

def main():
    st.set_page_config(page_title="Replicate AI Pipeline Builder", layout="wide")
    
    st.title("Replicate AI Pipeline Builder")
    st.sidebar.header("Settings")

    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    if 'generated_files' not in st.session_state:
        st.session_state.generated_files = []
    
    api_key = st.sidebar.text_input("Replicate API Key", type="password", value=st.session_state.api_key)

    if api_key:
        st.session_state.api_key = api_key
        if verify_api_key(api_key):
            st.sidebar.success("API key verified successfully!")
        else:
            st.sidebar.error("Invalid API key. Please check and try again.")

    available_nodes = [
        AINode("dynami-crafter", "DynamiCrafter", "camenduru/dynami-crafter-576x1024:e79ff8d01e81cbd90acfa1df4f209f637da2c68307891d77a6e4227f4ec350f1", "image", "video"),
        AINode("sdxl", "Stable Diffusion XL", "stability-ai/sdxl:a00d0b7dcbb9c3fbb34ba87d2d5b46c56969c84a628bf778a7fdaec30b1b99c5", "text", "image"),
        AINode("upscale", "Image Upscaling", "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b", "image", "image"),
        AINode("remove-bg", "Remove Background", "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003", "image", "image"),
        AINode("music-gen", "MusicGen", "meta/musicgen:7a76a8258b23fae65c5a22debb8841d1d7e816b75c2f24218cd2bd8573787906", "text", "audio"),
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

        if len(st.session_state.workflow.nodes) > 1:
            with st.expander("Connect Nodes", expanded=True):
                source = st.selectbox("From", list(st.session_state.workflow.nodes), format_func=lambda x: st.session_state.workflow.nodes[x]['node'].name)
                target = st.selectbox("To", [n for n in st.session_state.workflow.nodes if n != source], format_func=lambda x: st.session_state.workflow.nodes[x]['node'].name)
                if st.button("Connect Nodes", key="connect_nodes"):
                    if not st.session_state.workflow.has_edge(source, target):
                        st.session_state.workflow.add_edge(source, target)
                        st.success(f"Connected {st.session_state.workflow.nodes[source]['node'].name} to {st.session_state.workflow.nodes[target]['node'].name}.")
                    else:
                        st.warning("These nodes are already connected.")

        with st.expander("Remove Node or Connection", expanded=True):
            remove_type = st.radio("Select what to remove:", ["Node", "Connection"])
            if remove_type == "Node":
                node_to_remove = st.selectbox("Select node to remove", list(st.session_state.workflow.nodes), format_func=lambda x: st.session_state.workflow.nodes[x]['node'].name)
                if st.button("Remove Node", key="remove_node"):
                    st.session_state.workflow.remove_node(node_to_remove)
                    del st.session_state.node_positions[node_to_remove]
                    st.success(f"Removed {node_to_remove} from the workflow.")
            else:
                edges = list(st.session_state.workflow.edges)
                if edges:
                    edge_to_remove = st.selectbox("Select connection to remove", edges, format_func=lambda x: f"{st.session_state.workflow.nodes[x[0]]['node'].name} -> {st.session_state.workflow.nodes[x[1]]['node'].name}")
                    if st.button("Remove Connection", key="remove_edge"):
                        st.session_state.workflow.remove_edge(*edge_to_remove)
                        st.success(f"Removed connection between {st.session_state.workflow.nodes[edge_to_remove[0]]['node'].name} and {st.session_state.workflow.nodes[edge_to_remove[1]]['node'].name}.")
                else:
                    st.write("No connections to remove.")

        if st.button("Remove All Nodes", key="remove_all_nodes"):
            st.session_state.workflow.clear()
            st.session_state.node_positions.clear()
            st.success("All nodes and connections have been removed.")

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
            if input_type == "text":
                user_input = st.text_area("Enter your prompt:")
            elif input_type == "image":
                user_input = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
                if user_input:
                    user_input = Image.open(user_input)

            if st.button("Run Pipeline", key="run_pipeline"):
                if user_input and st.session_state.api_key:
                    with st.spinner("Processing..."):
                        current_output = user_input
                        st.session_state.generated_files = []
                        for node_id in nx.topological_sort(st.session_state.workflow):
                            node = st.session_state.workflow.nodes[node_id]['node']
                            
                            with st.expander(f"Processing: {node.name}", expanded=True):
                                st.info(f"Processing node: {node.name}")
                                if node.id == "dynami-crafter":
                                    i2v_eta = st.slider("Eta", 0.0, 2.0, 1.0, 0.1)
                                    i2v_seed = st.number_input("Seed", 0, 1000000, 123)
                                    i2v_steps = st.slider("Steps", 1, 100, 50)
                                    i2v_motion = st.slider("Motion", 1, 10, 4)
                                    i2v_cfg_scale = st.slider("CFG Scale", 0.0, 15.0, 7.5, 0.1)
                                    i2v_input_text = st.text_input("Input Text", "rocket launches")
                                    
                                    current_output = process_replicate(
                                        node,
                                        current_output,
                                        i2v_eta=i2v_eta,
                                        i2v_seed=i2v_seed,
                                        i2v_steps=i2v_steps,
                                        i2v_motion=i2v_motion,
                                        i2v_cfg_scale=i2v_cfg_scale,
                                        i2v_input_text=i2v_input_text
                                    )
                                else:
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
                elif not st.session_state.api_key:
                    st.warning("Please enter your API key in the sidebar.")
                else:
                    st.warning("Please provide input.")
        else:
            st.warning("Your workflow has no starting point. Please connect your nodes.")
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
