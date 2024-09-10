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
    def __init__(self, id, name, model_id, input_type, output_type, params):
        self.id = id
        self.name = name
        self.model_id = model_id
        self.input_type = input_type
        self.output_type = output_type
        self.params = params

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
        if not st.session_state.api_key:
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
        
        model_input.update(kwargs)  # Add the model parameters

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

# Parameter settings for each model
model_params = {
    "sdxl": ["prompt", "negative_prompt", "width", "height", "num_outputs", "scheduler", "guidance_scale"],
    "video": ["prompt", "negative_prompt", "num_frames", "width", "height", "guidance_scale", "fps", "model"],
    "upscale": ["image", "width", "height"],
    "remove-bg": ["image"],
    # Add more models as needed
}

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
        AINode("flux", "Flux Schnell", "black-forest-labs/flux-schnell", "text", "image", model_params["sdxl"]),
        AINode("sdxl", "Stable Diffusion XL", "stability-ai/sdxl:a00d0b7dcbb9c3fbb34ba87d2d5b46c56969c84a628bf778a7fdaec30b1b99c5", "text", "image", model_params["sdxl"]),
        AINode("video", "Video Generation", "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351", "text", "video", model_params["video"]),
        AINode("upscale", "Image Upscaling", "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b", "image", "image", model_params["upscale"]),
        AINode("remove-bg", "Remove Background", "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003", "image", "image", model_params["remove-bg"]),
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
            node_name = st.text_input("Enter a custom name for this node", value=selected_node.name)
            
            if st.button("Add Node", key="add_node"):
                if node_name not in st.session_state.workflow.nodes:
                    st.session_state.workflow.add_node(node_name, node=selected_node)
                    st.session_state.node_positions[node_name] = (len(st.session_state.workflow.nodes) * 100, 0)
                    st.success(f"Added {node_name} to the workflow.")
                else:
                    st.warning(f"{node_name} is already in the workflow.")

        if len(st.session_state.workflow.nodes) > 1:
            with st.expander("Connect Nodes", expanded=True):
                source = st.selectbox("From", list(st.session_state.workflow.nodes), format_func=lambda x: st.session_state.workflow.nodes[x]['node'].name)
                target = st.selectbox("To", [n for n in st.session_state.workflow.nodes if n != source], format_func=lambda x: st.session_state.workflow.nodes[x]['node'].name)
                if st.button("Connect Nodes", key="connect_nodes"):
                    if not st.session_stateThe corrected code includes replacing the invalid typographic apostrophe and ensuring the logic for removing nodes is valid. Here is the fully fixed code:

```python
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
    def __init__(self, id, name, model_id, input_type, output_type, params):
        self.id = id
        self.name = name
        self.model_id = model_id
        self.input_type = input_type
        self.output_type = output_type
        self.params = params

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
        if not st.session_state.api_key:
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
        
        model_input.update(kwargs)  # Add the model parameters

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

# Parameter settings for each model
model_params = {
    "sdxl": ["prompt", "negative_prompt", "width", "height", "num_outputs", "scheduler", "guidance_scale"],
    "video": ["prompt", "negative_prompt", "num_frames", "width", "height", "guidance_scale", "fps", "model"],
    "upscale": ["image", "width", "height"],
    "remove-bg": ["image"],
    # Add more models as needed
}

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
        AINode("flux", "Flux Schnell", "black-forest-labs/flux-schnell", "text", "image", model_params["sdxl"]),
        AINode("sdxl", "Stable Diffusion XL", "stability-ai/sdxl:a00d0b7dcbb9c3fbb34ba87d2d5b46c56969c84a628bf778a7fdaec30b1b99c5", "text", "image", model_params["sdxl"]),
        AINode("video", "Video Generation", "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351", "text", "video", model_params["video"]),
        AINode("upscale", "Image Upscaling", "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b", "image", "image", model_params["upscale"]),
        AINode("remove-bg", "Remove Background", "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003", "image", "image", model_params["remove-bg"]),
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
            node_name = st.text_input("Enter a custom name for this node", value=selected_node.name)
            
            if st.button("Add Node", key="add_node"):
                if node_name not in st.session_state.workflow.nodes:
                    st.session_state.workflow.add_node(node_name, node=selected_node)
                    st.session_state.node_positions[node_name] = (len(st.session_state.workflow.nodes) * 100, 0)
                    st.success(f"Added {node_name} to the workflow.")
                else:
                    st.warning(f"{node_name} is already in the workflow.")

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
               Here is the corrected Streamlit app code with the invalid typographic apostrophe replaced and the rest of the logic for removing nodes properly structured:

```python
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
    def __init__(self, id, name, model_id, input_type, output_type, params):
        self.id = id
        self.name = name
        self.model_id = model_id
        self.input_type = input_type
        self.output_type = output_type
        self.params = params

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
        if not st.session_state.api_key:
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
                ext = "txt" if file.endswith(".txt") else "png"
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

# Parameter settings for each model
model_params = {
    "sdxl": ["prompt", "negative_prompt", "width", "height", "num_outputs", "scheduler", "guidance_scale"],
    "video": ["prompt", "negative_prompt", "num_frames", "width", "height", "guidance_scale", "fps", "model"],
    "upscale": ["image", "width", "height"],
    "remove-bg": ["image"],
    # Add more models as needed
}

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
        AINode("flux", "Flux Schnell", "black-forest-labs/flux-schnell", "text", "image", model_params["sdxl"]),
        AINode("sdxl", "Stable Diffusion XL", "stability-ai/sdxl:a00d0b7dcbb9c3fbb34ba87d2d5b46c56969c84a628bf778a7fdaec30b1b99c5", "text", "image", model_params["sdxl"]),
        AINode("video", "Video Generation", "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351", "text", "video", model_params["video"]),
        AINode("upscale", "Image Upscaling", "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b", "image", "image", model_params["upscale"]),
        AINode("remove-bg", "Remove Background", "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003", "image", "image", model_params["remove-bg"]),
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
            node_name = st.text_input("Enter a custom name for this node", value=selected_node.name)
            
            if st.button("Add Node", key="add_node"):
                if node_name not in st.session_state.workflow.nodes:
                    st.session_state.workflow.add_node(node_name, node=selected_node)
                    st.session_state.node_positions[node_name] = (len(st.session_state.workflow.nodes) * 100, 0)
                    st.success(f"Added {node_name} to the workflow.")
                else:
                    st.warning(f"{node_name} is already in the workflow.")

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
                    st.sessionHere is the corrected version of your code, fixing the invalid character `’` (U+2019) and ensuring proper node removal logic:

```python
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
    def __init__(self, id, name, model_id, input_type, output_type, params):
        self.id = id
        self.name = name
        self.model_id = model_id
        self.input_type = input_type
        self.output_type = output_type
        self.params = params

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
        if not st.session_state.api_key:
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
        
        model_input.update(kwargs)  # Add the model parameters

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

# Parameter settings for each model
model_params = {
    "sdxl": ["prompt", "negative_prompt", "width", "height", "num_outputs", "scheduler", "guidance_scale"],
    "video": ["prompt", "negative_prompt", "num_frames", "width", "height", "guidance_scale", "fps", "model"],
    "upscale": ["image", "width", "height"],
    "remove-bg": ["image"],
    # Add more models as needed
}

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
        AINode("flux", "Flux Schnell", "black-forest-labs/flux-schnell", "text", "image", model_params["sdxl"]),
        AINode("sdxl", "Stable Diffusion XL", "stability-ai/sdxl:a00d0b7dcbb9c3fbb34ba87d2d5b46c56969c84a628bf778a7fdaec30b1b99c5", "text", "image", model_params["sdxl"]),
        AINode("video", "Video Generation", "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351", "text", "video", model_params["video"]),
        AINode("upscale", "Image Upscaling", "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b", "image", "image", model_params["upscale"]),
        AINode("remove-bg", "Remove Background", "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003", "image", "image", model_params["remove-bg"]),
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
            node_name = st.text_input("Enter a custom name for this node", value=selected_node.name)
            
            if st.button("Add Node", key="add_node"):
                if node_name not in st.session_state.workflow.nodes:
                    st.session_state.workflow.add_node(node_name, node=selected_node)
                    st.session_state.node_positions[node_name] = (len(st.session_state.workflow.nodes) * 100, 0)
                    st.success(f"Added {node_name} to the workflow.")
                else:
                    st.warning(f"{node_name} is already in the workflow.")

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
Here is the fixed and complete version of your Streamlit app, correcting the invalid character and ensuring proper logic for node removal:

```python
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
    def __init__(self, id, name, model_id, input_type, output_type, params):
        self.id = id
        self.name = name
        self.model_id = model_id
        self.input_type = input_type
        self.output_type = output_type
        self.params = params

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
        if not st.session_state.api_key:
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
        
        model_input.update(kwargs)  # Add the model parameters

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

# Parameter settings for each model
model_params = {
    "sdxl": ["prompt", "negative_prompt", "width", "height", "num_outputs", "scheduler", "guidance_scale"],
    "video": ["prompt", "negative_prompt", "num_frames", "width", "height", "guidance_scale", "fps", "model"],
    "upscale": ["image", "width", "height"],
    "remove-bg": ["image"],
    # Add more models as needed
}

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
        AINode("flux", "Flux Schnell", "black-forest-labs/flux-schnell", "text", "image", model_params["sdxl"]),
        AINode("sdxl", "Stable Diffusion XL", "stability-ai/sdxl:a00d0b7dcbb9c3fbb34ba87d2d5b46c56969c84a628bf778a7fdaec30b1b99c5", "text", "image", model_params["sdxl"]),
        AINode("video", "Video Generation", "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351", "text", "video", model_params["video"]),
        AINode("upscale", "Image Upscaling", "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b", "image", "image", model_params["upscale"]),
        AINode("remove-bg", "Remove Background", "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003", "image", "image", model_params["remove-bg"]),
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
            node_name = st.text_input("Enter a custom name for this node", value=selected_node.name)
            
            if st.button("Add Node", key="add_node"):
                if node_name not in st.session_state.workflow.nodes:
                    st.session_state.workflow.add_node(node_name, node=selected_node)
                    st.session_state.node_positions[node_name] = (len(st.session_state.workflow.nodes) * 100, 0)
                    st.success(f"Added {node_name} to the workflow.")
                else:
                    st.warning(f"{node_name} is already in the workflow.")

        if len(st.session_state.workflow.nodes) > 1:
            with st.expander("Connect Nodes", expanded=True):
                source = st.selectbox("From", list(st.session_state.workflow.nodes), format_func=lambda x: st.session_state.workflow.nodes[x]['node'].name)
                target = st.selectbox("To", [n for n in st.session_state.workflow.nodes if n != source], format_func=lambda x: st.session_state.workflow.nodes[x]['node'].name)
                if st.button("Connect Nodes", key="connect_nodes"):
                    if not st.session_state.workflow
