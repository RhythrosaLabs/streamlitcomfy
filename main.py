import streamlit as st
import replicate

# Initialize session state for nodes and API key
if "nodes" not in st.session_state:
    st.session_state["nodes"] = []

if "api_key" not in st.session_state:
    st.session_state["api_key"] = None

# Function to validate API key by running a basic test model
def validate_api_key(api_key):
    try:
        replicate.run(
            "stability-ai/stable-diffusion:latest",
            input={"prompt": "A beautiful sunset over the mountains"},
            api_token=api_key
        )
        return True
    except Exception as e:
        return False, str(e)

# Function to run the models based on node configuration
def run_model(node, api_key):
    model_id = node["model_id"]
    params = node["params"]
    try:
        output = replicate.run(
            f"{model_id}:latest",
            input=params,
            api_token=api_key
        )
        return output
    except Exception as e:
        return f"Error: {e}"

# Input for API Key and validation
api_key_input = st.text_input("Enter your Replicate API Key", type="password")
if st.button("Save and Validate API Key"):
    if api_key_input:
        api_key_input = api_key_input.strip()  # Remove any extra spaces
        valid, error = validate_api_key(api_key_input)
        if valid:
            st.session_state["api_key"] = api_key_input
            st.success("API Key is valid and saved!")
        else:
            st.error(f"API Key validation failed: {error}")
    else:
        st.warning("Please enter a valid API key.")

# Check if API key is set
if st.session_state["api_key"]:
    api_key = st.session_state["api_key"]

    # Add new node button
    if st.button("Add New Node"):
        st.session_state["nodes"].append({
            "model_id": "stability-ai/stable-diffusion",  # Default model
            "params": {"prompt": "default prompt"}
        })

    # Display and configure nodes
    for i, node in enumerate(st.session_state["nodes"]):
        st.write(f"### Node {i + 1}")

        # Select model
        model_id = st.selectbox(
            f"Choose Model for Node {i + 1}",
            ["stability-ai/stable-diffusion", "replicate/llama-70b", "black-forest-labs/flux-pro"],
            key=f"model_id_{i}"
        )
        st.session_state["nodes"][i]["model_id"] = model_id

        # Input parameters based on model type
        if model_id == "stability-ai/stable-diffusion":
            prompt = st.text_area(f"Prompt for Node {i + 1}", key=f"prompt_{i}")
            st.session_state["nodes"][i]["params"] = {"prompt": prompt}

        elif model_id == "replicate/llama-70b":
            prompt = st.text_area(f"Text Prompt for Node {i + 1}", key=f"llama_prompt_{i}")
            max_length = st.slider(f"Max Length for Node {i + 1}", 50, 500, 150, key=f"max_length_{i}")
            temperature = st.slider(f"Temperature for Node {i + 1}", 0.0, 1.0, 0.75, key=f"temperature_{i}")
            st.session_state["nodes"][i]["params"] = {
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature
            }

        elif model_id == "black-forest-labs/flux-pro":
            prompt = st.text_area(f"Art Prompt for Node {i + 1}", key=f"flux_prompt_{i}")
            style = st.selectbox(f"Style for Node {i + 1}", ["abstract", "cyberpunk", "fantasy", "realistic"], key=f"style_{i}")
            guidance_scale = st.slider(f"Guidance Scale for Node {i + 1}", 0.5, 20.0, 7.5, key=f"guidance_scale_{i}")
            st.session_state["nodes"][i]["params"] = {
                "prompt": prompt,
                "style": style,
                "guidance_scale": guidance_scale
            }

        # Button to remove node
        if st.button(f"Remove Node {i + 1}"):
            st.session_state["nodes"].pop(i)
            st.experimental_rerun()

        # Run the model and display output
        if st.button(f"Run Node {i + 1}"):
            result = run_model(st.session_state["nodes"][i], api_key)
            st.write(f"Output for Node {i + 1}: {result}")

    # Chain outputs between nodes
    if len(st.session_state["nodes"]) > 1:
        st.write("### Chaining Outputs Between Nodes")
        for i in range(len(st.session_state["nodes"]) - 1):
            if "output" in st.session_state["nodes"][i]:
                st.session_state["nodes"][i + 1]["params"]["prompt"] = st.session_state["nodes"][i]["output"]
                st.write(f"Passing output from Node {i + 1} to Node {i + 2}")

else:
    st.warning("Please enter your Replicate API key and click 'Save and Validate API Key' to proceed.")
