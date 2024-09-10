import streamlit as st
import requests
import replicate
import base64
import os
from PIL import Image
import io

class AINode:
    def __init__(self, name, api_type, model_id, input_type, output_type):
        self.name = name
        self.api_type = api_type
        self.model_id = model_id
        self.input_type = input_type
        self.output_type = output_type

    def process(self, input_data, api_key, **kwargs):
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
                st.error(f"Error: {response.status_code}, {response.text}")
                return None
        # Add more API types as needed

class AIPipeline:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def run(self, initial_input, api_keys):
        current_output = initial_input
        for node in self.nodes:
            api_key = api_keys.get(node.api_type)
            if api_key:
                current_output = node.process(current_output, api_key)
            else:
                st.error(f"API key missing for {node.api_type}")
                break
        return current_output

def main():
    st.title("Multi-Modal AI Pipeline")

    # API key inputs
    api_keys = {
        "replicate": st.sidebar.text_input("Replicate API Key", type="password"),
        "stability": st.sidebar.text_input("Stability AI API Key", type="password"),
        # Add more API keys as needed
    }

    # Available AI nodes
    available_nodes = [
        AINode("Stable Diffusion", "replicate", "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf", "text", "image"),
        AINode("DALL-E 3", "stability", "stable-diffusion-xl-1024-v1-0", "text", "image"),
        AINode("Video Generation", "replicate", "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351", "text", "video"),
        AINode("Image Upscaling", "replicate", "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b", "image", "image"),
        # Add more nodes as needed
    ]

    # Pipeline creation
    pipeline = AIPipeline()
    selected_nodes = st.multiselect("Select AI nodes for your pipeline", available_nodes, format_func=lambda x: x.name)
    for node in selected_nodes:
        pipeline.add_node(node)

    # Input
    input_type = pipeline.nodes[0].input_type if pipeline.nodes else "text"
    if input_type == "text":
        user_input = st.text_area("Enter your prompt:")
    elif input_type == "image":
        user_input = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
        if user_input:
            user_input = Image.open(user_input)

    # Run pipeline
    if st.button("Run Pipeline"):
        if user_input:
            with st.spinner("Processing..."):
                output = pipeline.run(user_input, api_keys)
                if isinstance(output, Image.Image):
                    st.image(output, caption="Generated Image", use_column_width=True)
                elif isinstance(output, str) and output.startswith("http"):
                    st.video(output)
                else:
                    st.write(output)
        else:
            st.warning("Please provide input.")

if __name__ == "__main__":
    main()
