import streamlit as st
import replicate
from PIL import Image
import requests
from io import BytesIO

class AINode:
    def __init__(self, name, model_id, input_schema, output_schema):
        self.name = name
        self.model_id = model_id
        self.input_schema = input_schema
        self.output_schema = output_schema

    def process(self, input_data, api_key):
        client = replicate.Client(api_token=api_key)
        return client.run(self.model_id, input=input_data)

class AIPipeline:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def run(self, initial_input, api_key):
        current_output = initial_input
        for node in self.nodes:
            current_output = node.process(current_output, api_key)
        return current_output

def create_input_fields(schema):
    inputs = {}
    for key, value in schema['properties'].items():
        if key == 'prompt':
            inputs[key] = st.text_area(value['title'], help=value.get('description', ''))
        elif value['type'] == 'integer':
            inputs[key] = st.number_input(value['title'], min_value=value.get('minimum'), max_value=value.get('maximum'), value=value.get('default', 0), help=value.get('description', ''))
        elif value['type'] == 'number':
            inputs[key] = st.slider(value['title'], min_value=value.get('minimum', 0.0), max_value=value.get('maximum', 1.0), value=value.get('default', 0.5), help=value.get('description', ''))
        elif value['type'] == 'string' and 'enum' in value:
            inputs[key] = st.selectbox(value['title'], options=value['enum'], index=value['enum'].index(value.get('default', value['enum'][0])), help=value.get('description', ''))
        elif value['type'] == 'boolean':
            inputs[key] = st.checkbox(value['title'], value=value.get('default', False), help=value.get('description', ''))
        elif value['type'] == 'string':
            inputs[key] = st.text_input(value['title'], value=value.get('default', ''), help=value.get('description', ''))
    return inputs

def main():
    st.title("Replicate-based Multi-Modal AI Pipeline")

    api_key = st.sidebar.text_input("Replicate API Key", type="password")

    flux_pro_input_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "title": "Prompt", "description": "Text prompt for image generation"},
            "seed": {"type": "integer", "title": "Seed", "description": "Random seed. Set for reproducible generation"},
            "steps": {"type": "integer", "title": "Steps", "default": 25, "maximum": 50, "minimum": 1, "description": "Number of diffusion steps"},
            "guidance": {"type": "number", "title": "Guidance", "default": 3, "maximum": 5, "minimum": 2, "description": "Controls the balance between adherence to the text prompt and image quality/diversity"},
            "interval": {"type": "number", "title": "Interval", "default": 2, "maximum": 4, "minimum": 1, "description": "Increases the variance in possible outputs"},
            "aspect_ratio": {"type": "string", "title": "Aspect Ratio", "enum": ["1:1", "16:9", "2:3", "3:2", "4:5", "5:4", "9:16"], "default": "1:1"},
            "output_format": {"type": "string", "title": "Output Format", "enum": ["webp", "jpg", "png"], "default": "webp"},
            "output_quality": {"type": "integer", "title": "Output Quality", "default": 80, "maximum": 100, "minimum": 0},
            "safety_tolerance": {"type": "integer", "title": "Safety Tolerance", "default": 2, "maximum": 5, "minimum": 1}
        }
    }

    llama_input_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "title": "Prompt", "description": "Prompt to send to the model"},
            "top_p": {"type": "number", "title": "Top P", "default": 0.9, "maximum": 1, "minimum": 0},
            "min_tokens": {"type": "integer", "title": "Min Tokens", "default": 0, "minimum": 0},
            "temperature": {"type": "number", "title": "Temperature", "default": 0.6, "maximum": 5, "minimum": 0},
            "prompt_template": {"type": "string", "title": "Prompt Template", "default": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"},
            "presence_penalty": {"type": "number", "title": "Presence Penalty", "default": 1.15, "maximum": 2, "minimum": 0}
        }
    }

    available_nodes = [
        AINode("Flux Pro (Image Generation)", "black-forest-labs/flux-pro", flux_pro_input_schema, {"type": "string", "format": "uri"}),
        AINode("Meta Llama 3 (Text Generation)", "meta/meta-llama-3-70b-instruct", llama_input_schema, {"type": "array", "items": {"type": "string"}})
    ]

    pipeline = AIPipeline()
    selected_node = st.selectbox("Select AI model", available_nodes, format_func=lambda x: x.name)
    pipeline.add_node(selected_node)

    st.subheader(f"Input for {selected_node.name}")
    input_data = create_input_fields(selected_node.input_schema)

    if st.button("Run Pipeline"):
        if api_key:
            with st.spinner("Processing..."):
                output = pipeline.run(input_data, api_key)
                if isinstance(output, str) and output.startswith("http"):
                    response = requests.get(output)
                    img = Image.open(BytesIO(response.content))
                    st.image(img, caption="Generated Image", use_column_width=True)
                elif isinstance(output, list):
                    st.write("".join(output))
                else:
                    st.write(output)
        else:
            st.warning("Please provide a Replicate API key.")

if __name__ == "__main__":
    main()
