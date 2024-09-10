import streamlit as st
import replicate
import requests
import base64
from PIL import Image
import io

class AINode:
    def __init__(self, name, api_type, model_id, input_type, output_type):
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
    st.title("Multi-Modal AI Pipeline")

    # API key inputs
    api_keys = {
        "replicate": st.sidebar.text_input("Replicate API Key", type="password"),
        "stability": st.sidebar.text_input("Stability AI API Key", type="password"),
    }

    # Available AI nodes
    available_nodes = [
        AINode("Stable Diffusion", "replicate", "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf", "text", "image"),
        AINode("DALL-E 3", "stability", "stable-diffusion-xl-1024-v1-0", "text", "image"),
        AINode("Video Generation", "replicate", "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351", "text", "video"),
        AINode("Image Upscaling", "replicate", "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b", "image", "image"),
    ]

    # Pipeline creation
    selected_nodes = st.multiselect("Select AI nodes for your pipeline", available_nodes, format_func=lambda x: x.name)

    # Input
    input_type = selected_nodes[0].input_type if selected_nodes else "text"
    if input_type == "text":
        user_input = st.text_area("Enter your prompt:")
    elif input_type == "image":
        user_input = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
        if user_input:
            user_input = Image.open(user_input)

    # Run pipeline
    if st.button("Run Pipeline"):
        if user_input and selected_nodes:
            with st.spinner("Processing..."):
                current_output = user_input
                for node in selected_nodes:
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
            st.warning("Please provide input and select at least one node.")

if __name__ == "__main__":
    main()
