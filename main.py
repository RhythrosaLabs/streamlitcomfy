# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import replicate
import requests
import base64
from PIL import Image
import io

app = FastAPI()

class AINode(BaseModel):
    name: str
    api_type: str
    model_id: str
    input_type: str
    output_type: str

class AIRequest(BaseModel):
    nodes: List[AINode]
    input_data: Any
    api_keys: Dict[str, str]

def process_replicate(node: AINode, input_data: Any, api_key: str, **kwargs):
    client = replicate.Client(api_token=api_key)
    output = client.run(node.model_id, input={"image": input_data, **kwargs})
    return output

def process_stability(node: AINode, input_data: str, api_key: str, **kwargs):
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
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.post("/process")
async def process_pipeline(request: AIRequest):
    current_output = request.input_data
    for node in request.nodes:
        api_key = request.api_keys.get(node.api_type)
        if not api_key:
            raise HTTPException(status_code=400, detail=f"API key missing for {node.api_type}")
        
        if node.api_type == "replicate":
            current_output = process_replicate(node, current_output, api_key)
        elif node.api_type == "stability":
            current_output = process_stability(node, current_output, api_key)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported API type: {node.api_type}")
    
    return {"result": current_output}

# Streamlit frontend
import streamlit as st
import requests

st.title("Multi-Modal AI Pipeline")

# API key inputs
api_keys = {
    "replicate": st.sidebar.text_input("Replicate API Key", type="password"),
    "stability": st.sidebar.text_input("Stability AI API Key", type="password"),
}

# Available AI nodes
available_nodes = [
    AINode(name="Stable Diffusion", api_type="replicate", model_id="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf", input_type="text", output_type="image"),
    AINode(name="DALL-E 3", api_type="stability", model_id="stable-diffusion-xl-1024-v1-0", input_type="text", output_type="image"),
    AINode(name="Video Generation", api_type="replicate", model_id="anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351", input_type="text", output_type="video"),
    AINode(name="Image Upscaling", api_type="replicate", model_id="nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b", input_type="image", output_type="image"),
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
            request_data = AIRequest(nodes=selected_nodes, input_data=user_input, api_keys=api_keys)
            response = requests.post("http://localhost:8000/process", json=request_data.dict())
            if response.status_code == 200:
                output = response.json()["result"]
                if isinstance(output, dict) and "image" in output:
                    st.image(output["image"], caption="Generated Image", use_column_width=True)
                elif isinstance(output, str) and output.startswith("http"):
                    st.video(output)
                else:
                    st.write(output)
            else:
                st.error(f"Error: {response.status_code}, {response.text}")
    else:
        st.warning("Please provide input and select at least one node.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
