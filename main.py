import streamlit as st
import requests
import replicate
import base64
import os
from PIL import Image
import io
import json
from graphviz import Digraph

class AINode:
    def __init__(self, name, api_type, model_id, input_type, output_type, parameters):
        self.name = name
        self.api_type = api_type
        self.model_id = model_id
        self.input_type = input_type
        self.output_type = output_type
        self.parameters = parameters

    def process(self, input_data, api_key, **kwargs):
        try:
            if self.api_type == "replicate":
                client = replicate.Client(api_token=api_key)
                output = client.run(self.model_id, input={"image": input_data, **kwargs})
                return output
            elif self.api_type == "stability":
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "text_prompts": [{"text": input_data}],
                    **kwargs
                }
                response = requests.post(f"https://api.stability.ai/v1/generation/{self.model_id}/text-to-image", headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    image_data = base64.b64decode(data['artifacts'][0]['base64'])
                    return Image.open(io.BytesIO(image_data))
                else:
                    raise Exception(f"API Error: {response.status_code}, {response.text}")
        except Exception as e:
            raise Exception(f"Error processing {self.name}: {str(e)}")

class AIPipeline:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def run(self, initial_input, api_keys):
        current_output = initial_input
        outputs = []
        for node in self.nodes:
            api_key = api_keys.get(node.api_type)
            if api_key:
                current_output = node.process(current_output, api_key)
                outputs.append(current_output)
            else:
                raise Exception(f"API key missing for {node.api_type}")
        return outputs

    def visualize(self):
        dot = Digraph(comment='AI Pipeline')
        for i, node in enumerate(self.nodes):
            dot.node(f'node_{i}', node.name)
            if i > 0:
                dot.edge(f'node_{i-1}', f'node_{i}')
        return dot

def render_output(output, output_type):
    if output_type == "image":
        st.image(output, caption="Generated Image", use_column_width=True)
    elif output_type == "video" and isinstance(output, str) and output.startswith("http"):
        st.video(output)
    else:
        st.write(output)

def main():
    st.title("Enhanced Multi-Modal AI Pipeline")

    # API key inputs
    api_keys = {
        "replicate": st.sidebar.text_input("Replicate API Key", type="password"),
        "stability": st.sidebar.text_input("Stability AI API Key", type="password"),
    }

    # Available AI nodes
    available_nodes = [
        AINode("Stable Diffusion", "replicate", "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf", "text", "image", 
               ["prompt", "num_inference_steps", "guidance_scale"]),
        AINode("DALL-E 3", "stability", "stable-diffusion-xl-1024-v1-0", "text", "image", 
               ["prompt", "width", "height"]),
        AINode("Video Generation", "replicate", "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351", "text", "video", 
               ["prompt", "frames", "fps"]),
        AINode("Image Upscaling", "replicate", "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b", "image", "image", 
               ["image", "scale"]),
    ]

    # Pipeline creation
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = AIPipeline()

    selected_node = st.selectbox("Select AI node to add to your pipeline", available_nodes, format_func=lambda x: x.name)
    if st.button("Add Node"):
        st.session_state.pipeline.add_node(selected_node)

    # Display and configure nodes
    for i, node in enumerate(st.session_state.pipeline.nodes):
        st.write(f"### Node {i+1}: {node.name}")
        for param in node.parameters:
            if param == "image":
                node.parameters[param] = st.file_uploader(f"{param} for {node.name}", type=["png", "jpg", "jpeg"])
            elif param in ["num_inference_steps", "width", "height", "frames", "fps", "scale"]:
                node.parameters[param] = st.number_input(f"{param} for {node.name}", min_value=1, value=50)
            elif param == "guidance_scale":
                node.parameters[param] = st.slider(f"{param} for {node.name}", min_value=1.0, max_value=20.0, value=7.5)
            else:
                node.parameters[param] = st.text_input(f"{param} for {node.name}")

        if st.button(f"Remove Node {i+1}"):
            st.session_state.pipeline.nodes.pop(i)
            st.experimental_rerun()

    # Visualize pipeline
    if st.session_state.pipeline.nodes:
        st.graphviz_chart(st.session_state.pipeline.visualize())

    # Run pipeline
    if st.button("Run Pipeline"):
        try:
            with st.spinner("Processing..."):
                outputs = st.session_state.pipeline.run(st.session_state.pipeline.nodes[0].parameters["prompt"], api_keys)
                for i, output in enumerate(outputs):
                    st.write(f"### Output from Node {i+1}")
                    render_output(output, st.session_state.pipeline.nodes[i].output_type)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Save/Load Pipeline
    if st.button("Save Pipeline"):
        pipeline_data = json.dumps([{"name": node.name, "parameters": node.parameters} for node in st.session_state.pipeline.nodes])
        st.download_button("Download Pipeline", pipeline_data, "pipeline.json", "application/json")

    uploaded_file = st.file_uploader("Load Pipeline", type="json")
    if uploaded_file is not None:
        pipeline_data = json.load(uploaded_file)
        st.session_state.pipeline = AIPipeline()
        for node_data in pipeline_data:
            node = next((n for n in available_nodes if n.name == node_data["name"]), None)
            if node:
                node.parameters = node_data["parameters"]
                st.session_state.pipeline.add_node(node)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
