import streamlit as st
import replicate
import requests
import base64
from PIL import Image
import io
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config

class AINode:
    def __init__(self, id, name, api_type, model_id, input_type, output_type):
        self.id = id
        self.name = name
        self.api_type = api_type
        self.model_id = model_id
        self.input_type = input_type
        self.output_type = output_type

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))

def process_replicate(node, input_data, api_key, **kwargs):
    try:
        client = replicate.Client(api_token=api_key)
        
        # Prepare the input based on the node's input_type
        if node.input_type == "text":
            model_input = {"prompt": input_data}
        elif node.input_type == "image":
            if isinstance(input_data, Image.Image):
                # Convert PIL Image to base64
                buffered = io.BytesIO()
                input_data.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                model_input = {"image": img_str}
            elif isinstance(input_data, str) and input_data.startswith("http"):
                # If input_data is a URL, download the image and convert to base64
                response = requests.get(input_data)
                img = Image.open(io.BytesIO(response.content))
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                model_input = {"image": img_str}
            elif isinstance(input_data, bytes):
                # If input_data is already bytes, encode it to base64
                img_str = base64.b64encode(input_data).decode()
                model_input = {"image": img_str}
            else:
                model_input = {"image": input_data}
        else:
            raise ValueError(f"Unsupported input type: {node.input_type}")
        
        # Merge any additional kwargs
        model_input.update(kwargs)
        
        # Run the model
        output = client.run(node.model_id, input=model_input)
        
        # Handle different types of output
        if isinstance(output, list) and len(output) > 0:
            if isinstance(output[0], str) and output[0].startswith("http"):
                # If the output is a list with a URL, download the image
                return download_image(output[0])
            else:
                return output[0]  # Return the first element of the list
        elif isinstance(output, str) and output.startswith("http"):
            # If the output is a URL, download the image
            return download_image(output)
        else:
            return output
    except replicate.exceptions.ReplicateError as e:
        st.error(f"Replicate API Error: {str(e)}")
        st.error("Please check your API key and model settings.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

def process_stability(node, input_data, api_key, **kwargs):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "text_prompts": [{"text": input_data}],
            **kwargs
        }
        response = requests.post(f"https://api.stability.ai/v1/generation/{node.model_id}/text-to-image", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        image_data = base64.b64decode(data['artifacts'][0]['base64'])
        return Image.open(io.BytesIO(image_data))
    except requests.exceptions.RequestException as e:
        st.error(f"Stability AI API Error: {str(e)}")
        st.error("Please check your API key and model settings.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

def main():
    st.title("Custom AI Pipeline")

    # API key inputs
    api_keys = {
        "replicate": st.sidebar.text_input("Replicate API Key", type="password"),
        "stability": st.sidebar.text_input("Stability AI API Key", type="password"),
    }

    # Available AI nodes
    available_nodes = [
        AINode("sd", "Stable Diffusion", "replicate", "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf", "text", "image"),
        AINode("dalle", "DALL-E 3", "stability", "stable-diffusion-xl-1024-v1-0", "text", "image"),
        AINode("video", "Video Generation", "replicate", "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351", "text", "video"),
        AINode("upscale", "Image Upscaling", "replicate", "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b", "image", "image"),
    ]

    # Initialize session state
    if 'workflow' not in st.session_state:
        st.session_state.workflow = nx.DiGraph()
    if 'node_positions' not in st.session_state:
        st.session_state.node_positions = {}

    # Node management
    st.subheader("Node Management")
    col1, col2 = st.columns(2)

    with col1:
        # Add node
        selected_node = st.selectbox("Select a node to add", available_nodes, format_func=lambda x: x.name)
        if st.button("Add Node"):
            if selected_node.id not in st.session_state.workflow.nodes:
                st.session_state.workflow.add_node(selected_node.id, node=selected_node)
                st.session_state.node_positions[selected_node.id] = (len(st.session_state.workflow.nodes) * 100, 0)
                st.success(f"Added {selected_node.name} to the workflow.")
            else:
                st.warning(f"{selected_node.name} is already in the workflow.")

    with col2:
        # Remove node
        if st.session_state.workflow.nodes:
            node_to_remove = st.selectbox("Select node to remove", list(st.session_state.workflow.nodes), format_func=lambda x: st.session_state.workflow.nodes[x]['node'].name)
            if st.button("Remove Node"):
                st.session_state.workflow.remove_node(node_to_remove)
                del st.session_state.node_positions[node_to_remove]
                st.success(f"Removed {node_to_remove} from the workflow.")
        else:
            st.write("No nodes to remove.")

    # Node connection
    st.subheader("Node Connection")
    if len(st.session_state.workflow.nodes) > 1:
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Select source node", list(st.session_state.workflow.nodes), format_func=lambda x: st.session_state.workflow.nodes[x]['node'].name)
        with col2:
            target = st.selectbox("Select target node", [n for n in st.session_state.workflow.nodes if n != source], format_func=lambda x: st.session_state.workflow.nodes[x]['node'].name)
        if st.button("Connect Nodes"):
            if not st.session_state.workflow.has_edge(source, target):
                st.session_state.workflow.add_edge(source, target)
                st.success(f"Connected {st.session_state.workflow.nodes[source]['node'].name} to {st.session_state.workflow.nodes[target]['node'].name}.")
            else:
                st.warning("These nodes are already connected.")
    else:
        st.write("Add at least two nodes to create connections.")

    # Edge removal
    st.subheader("Edge Removal")
    edges = list(st.session_state.workflow.edges)
    if edges:
        edge_to_remove = st.selectbox("Select edge to remove", edges, format_func=lambda x: f"{st.session_state.workflow.nodes[x[0]]['node'].name} -> {st.session_state.workflow.nodes[x[1]]['node'].name}")
        if st.button("Remove Edge"):
            st.session_state.workflow.remove_edge(*edge_to_remove)
            st.success(f"Removed connection between {st.session_state.workflow.nodes[edge_to_remove[0]]['node'].name} and {st.session_state.workflow.nodes[edge_to_remove[1]]['node'].name}.")
    else:
        st.write("No edges to remove.")

    # Visualize workflow
    st.subheader("Workflow Visualization")
    nodes = [Node(id=n, label=data['node'].name, x=st.session_state.node_positions[n][0], y=st.session_state.node_positions[n][1]) 
             for n, data in st.session_state.workflow.nodes(data=True)]
    edges = [Edge(source=u, target=v) for u, v in st.session_state.workflow.edges()]
    config = Config(width=800, height=400, directed=True, physics=True, hierarchical=False)
    agraph(nodes=nodes, edges=edges, config=config)

    # Input and pipeline execution
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

            if st.button("Run Pipeline"):
                if user_input:
                    with st.spinner("Processing..."):
                        current_output = user_input
                        for node_id in nx.topological_sort(st.session_state.workflow):
                            node = st.session_state.workflow.nodes[node_id]['node']
                            api_key = api_keys.get(node.api_type)
                            if not api_key:
                                st.error(f"API key missing for {node.api_type}")
                                break
                            
                            st.write(f"Processing node: {node.name}")
                            if node.api_type == "replicate":
                                current_output = process_replicate(node, current_output, api_key)
                            elif node.api_type == "stability":
                                current_output = process_stability(node, current_output, api_key)
                            else:
                                st.error(f"Unsupported API type: {node.api_type}")
                                break
                            
                            if current_output is None:
                                st.error(f"Processing failed at node: {node.name}")
                                break
                            
                            st.write(f"Output from {node.name}:")
                            if isinstance(current_output, Image.Image):
                                st.image(current_output, caption=f"Output from {node.name}", use_column_width=True)
                            elif isinstance(current_output, str) and current_output.startswith("http"):
                                st.image(current_output, caption=f"Output from {node.name}", use_column_width=True)
                            else:
                                st.write(current_output)
                        
                        st.success("Pipeline execution completed!")
                else:
                    st.warning("Please provide input.")
        else:
            st.warning("Your workflow has no starting point. Please connect your nodes.")
    else:
        st.warning("Please add at least one node to the workflow.")

if __name__ == "__main__":
    main()
