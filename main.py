import streamlit as st
import replicate
import requests
from PIL import Image
import io
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config

class AINode:
    def __init__(self, id, name, model_id, input_type, output_type):
        self.id = id
        self.name = name
        self.model_id = model_id
        self.input_type = input_type
        self.output_type = output_type

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))

def process_replicate(node, input_data, **kwargs):
    try:
        if not st.session_state.api_key:
            st.error("API key is missing. Please enter your Replicate API key in the sidebar.")
            return None

        if node.input_type == "text":
            model_input = {"prompt": input_data}
        elif node.input_type == "image":
            if isinstance(input_data, Image.Image):
                img_byte_arr = io.BytesIO()
                input_data.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                temp_file = st.file_uploader("Temporary file (don't change this)", type=["png"], key="temp_file")
                if temp_file is None:
                    st.session_state.temp_file = img_byte_arr
                    st.experimental_rerun()
                
                model_input = {"image": temp_file}
            elif isinstance(input_data, str) and input_data.startswith("http"):
                model_input = {"image": input_data}
            else:
                raise ValueError(f"Unsupported image input type: {type(input_data)}")
        else:
            raise ValueError(f"Unsupported input type: {node.input_type}")
        
        model_input.update(kwargs)
        
        # Use replicate.run with the model_id, input, and API key
        output = replicate.run(node.model_id, input=model_input, api_token=st.session_state.api_key)
        
        if isinstance(output, list) and len(output) > 0:
            return output[0] if isinstance(output[0], str) and output[0].startswith("http") else output[0]
        elif isinstance(output, str) and output.startswith("http"):
            return output
        else:
            return output
    except replicate.exceptions.ReplicateError as e:
        st.error(f"Replicate API Error: {str(e)}")
        st.error("Please check your API key and model settings.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Replicate AI Pipeline Builder", layout="wide")
    
    st.title("Replicate AI Pipeline Builder")
    st.sidebar.header("Settings")

    # API key input
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    api_key = st.sidebar.text_input("Replicate API Key", type="password", value=st.session_state.api_key)
    if api_key:
        st.session_state.api_key = api_key

    # Available AI nodes
    available_nodes = [
        AINode("flux", "Flux Schnell", "black-forest-labs/flux-schnell", "text", "image"),
        AINode("sdxl", "Stable Diffusion XL", "stability-ai/sdxl:a00d0b7dcbb9c3fbb34ba87d2d5b46c56969c84a628bf778a7fdaec30b1b99c5", "text", "image"),
        AINode("video", "Video Generation", "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351", "text", "video"),
        AINode("upscale", "Image Upscaling", "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b", "image", "image"),
    ]

    # Initialize session state
    if 'workflow' not in st.session_state:
        st.session_state.workflow = nx.DiGraph()
    if 'node_positions' not in st.session_state:
        st.session_state.node_positions = {}

    # Main area
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Pipeline Builder")
        
        # Node management
        with st.expander("Add Node", expanded=True):
            selected_node = st.selectbox("Select a node to add", available_nodes, format_func=lambda x: x.name)
            if st.button("Add Node", key="add_node"):
                if selected_node.id not in st.session_state.workflow.nodes:
                    st.session_state.workflow.add_node(selected_node.id, node=selected_node)
                    st.session_state.node_positions[selected_node.id] = (len(st.session_state.workflow.nodes) * 100, 0)
                    st.success(f"Added {selected_node.name} to the workflow.")
                else:
                    st.warning(f"{selected_node.name} is already in the workflow.")

        # Node connection
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

        # Node and Edge removal
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

    with col2:
        st.subheader("Workflow Visualization")
        nodes = [Node(id=n, label=data['node'].name, x=st.session_state.node_positions[n][0], y=st.session_state.node_positions[n][1]) 
                 for n, data in st.session_state.workflow.nodes(data=True)]
        edges = [Edge(source=u, target=v) for u, v in st.session_state.workflow.edges()]
        config = Config(width=600, height=400, directed=True, physics=True, hierarchical=False)
        agraph(nodes=nodes, edges=edges, config=config)

    # Pipeline execution
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
                        for node_id in nx.topological_sort(st.session_state.workflow):
                            node = st.session_state.workflow.nodes[node_id]['node']
                            
                            with st.expander(f"Processing: {node.name}", expanded=True):
                                st.info(f"Processing node: {node.name}")
                                if node.id == "flux":
                                    current_output = process_replicate(node, current_output, 
                                                                       num_outputs=1, 
                                                                       aspect_ratio="1:1", 
                                                                       output_format="webp", 
                                                                       output_quality=80)
                                else:
                                    current_output = process_replicate(node, current_output)
                                
                                if current_output is None:
                                    st.error(f"Processing failed at node: {node.name}")
                                    break
                                
                                st.success(f"Output from {node.name}:")
                                if isinstance(current_output, Image.Image):
                                    st.image(current_output, caption=f"Output from {node.name}", use_column_width=True)
                                elif isinstance(current_output, str) and current_output.startswith("http"):
                                    st.image(current_output, caption=f"Output from {node.name}", use_column_width=True)
                                else:
                                    st.write(current_output)
                        
                        st.success("Pipeline execution completed!")
                elif not st.session_state.api_key:
                    st.warning("Please enter your Replicate API key in the sidebar.")
                else:
                    st.warning("Please provide input.")
        else:
            st.warning("Your workflow has no starting point. Please connect your nodes.")
    else:
        st.warning("Please add at least one node to the workflow.")

if __name__ == "__main__":
    main()
