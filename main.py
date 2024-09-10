import streamlit as st
import replicate
import requests
import base64
from PIL import Image
import io
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_draggable import draggable

class AINode:
    def __init__(self, id, name, api_type, model_id, input_type, output_type):
        self.id = id
        self.name = name
        self.api_type = api_type
        self.model_id = model_id
        self.input_type = input_type
        self.output_type = output_type

def process_replicate(node, input_data, api_key, **kwargs):
    client = replicate.Client(api_token=api_key)
    output = client.run(node.model_id, input={"image": input_data, **kwargs})
    return output

def process_stability(node, input_data, api_key, **kwargs):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "text_prompts": [{"text": input_data}],
        **kwargs
    }
    response = requests.post(f"https://api.stability.ai/v1/generation/{node.model_id}/text-to-image", headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        image_data = base64.b64decode(data['artifacts'][0]['base64'])
        return Image.open(io.BytesIO(image_data))
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
        return None

def main():
    st.title("Enhanced Custom AI Pipeline")

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

    # Node selection and drag-and-drop
    col1, col2 = st.columns(2)
    with col1:
        selected_node = st.selectbox("Select a node to add", available_nodes, format_func=lambda x: x.name)
        if st.button("Add Node"):
            if selected_node.id not in st.session_state.workflow.nodes:
                st.session_state.workflow.add_node(selected_node.id, node=selected_node)
                st.session_state.node_positions[selected_node.id] = (len(st.session_state.workflow.nodes) * 100, 0)

    with col2:
        st.write("Drag and drop nodes to rearrange:")
        for node_id, data in st.session_state.workflow.nodes(data=True):
            draggable_key = f"drag_{node_id}"
            new_pos = draggable(data['node'].name, key=draggable_key)
            if new_pos:
                st.session_state.node_positions[node_id] = new_pos

    # Dynamic edge creation
    st.write("Connect nodes:")
    source = st.selectbox("Select source node", list(st.session_state.workflow.nodes), format_func=lambda x: st.session_state.workflow.nodes[x]['node'].name)
    target = st.selectbox("Select target node", [n for n in st.session_state.workflow.nodes if n != source], format_func=lambda x: st.session_state.workflow.nodes[x]['node'].name)
    if st.button("Connect Nodes"):
        if not st.session_state.workflow.has_edge(source, target):
            st.session_state.workflow.add_edge(source, target)

    # Remove node or edge
    st.write("Remove node or edge:")
    remove_type = st.radio("Select type to remove:", ["Node", "Edge"])
    if remove_type == "Node":
        node_to_remove = st.selectbox("Select node to remove", list(st.session_state.workflow.nodes), format_func=lambda x: st.session_state.workflow.nodes[x]['node'].name)
        if st.button("Remove Node"):
            st.session_state.workflow.remove_node(node_to_remove)
            del st.session_state.node_positions[node_to_remove]
    else:
        edges = list(st.session_state.workflow.edges)
        if edges:
            edge_to_remove = st.selectbox("Select edge to remove", edges, format_func=lambda x: f"{st.session_state.workflow.nodes[x[0]]['node'].name} -> {st.session_state.workflow.nodes[x[1]]['node'].name}")
            if st.button("Remove Edge"):
                st.session_state.workflow.remove_edge(*edge_to_remove)
        else:
            st.write("No edges to remove.")

    # Visualize workflow
    nodes = [Node(id=n, label=data['node'].name, x=st.session_state.node_positions[n][0], y=st.session_state.node_positions[n][1]) 
             for n, data in st.session_state.workflow.nodes(data=True)]
    edges = [Edge(source=u, target=v) for u, v in st.session_state.workflow.edges()]
    config = Config(width=800, height=400, directed=True, physics=True, hierarchical=False)
    agraph(nodes=nodes, edges=edges, config=config)

    # Input and pipeline execution (same as before)
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
                            
                            if node.api_type == "replicate":
                                current_output = process_replicate(node, current_output, api_key)
                            elif node.api_type == "stability":
                                current_output = process_stability(node, current_output, api_key)
                            else:
                                st.error(f"Unsupported API type: {node.api_type}")
                                break
                        
                        if isinstance(current_output, Image.Image):
                            st.image(current_output, caption="Generated Image", use_column_width=True)
                        elif isinstance(current_output, str) and current_output.startswith("http"):
                            st.video(current_output)
                        else:
                            st.write(current_output)
                else:
                    st.warning("Please provide input.")
        else:
            st.warning("Your workflow has no starting point. Please connect your nodes.")
    else:
        st.warning("Please add at least one node to the workflow.")

if __name__ == "__main__":
    main()
